/* unicc_loader.c - backend probing, loading, identification and diagnostics.
 *
 * Mechanism modeled on UniMPI's loader (src/loader.c): a small fixed backend
 * table, an env-driven deterministic selection, dlopen with an alternate
 * soname fallback, identify-by-feature-symbol, and a per-symbol diagnose that
 * is invaluable on hosts with no NCCL/RCCL installed. */
#include "unicc_backends.h"
#include "unicc_version.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Backend definitions.
 *
 * NVIDIA NCCL ships a shared library whose dev name is libnccl.so and whose
 * runtime soname is libnccl.so.2 (older releases used .1); prefer the dev
 * name and fall back to the soname.
 * AMD RCCL ships librccl.so (soname librccl.so.1).
 * A future domestic-GPU collective library is a one-line addition here. */
const unicc_backend_info_t unicc_backends[UNICC_MAX_BACKENDS] = {
    {UNICC_BACKEND_NCCL, "nccl", "libnccl.so", "libnccl.so.2", 2},
    {UNICC_BACKEND_RCCL, "rccl", "librccl.so", "librccl.so.1", 3}
};

/* Check whether a backend type is sensible on this platform. */
static int backend_supported_on_platform(unicc_backend_type_t type) {
#ifdef _WIN32
    (void)type;
    return 0;
#elif defined(__linux__) || defined(__APPLE__)
    (void)type;
    return 1;
#else
    (void)type;
    return 0;
#endif
}

static const char* get_nonempty_env(const char *name) {
    const char *value = getenv(name);
    return (value && value[0] != '\0') ? value : NULL;
}

static const char* backend_name_from_type(unicc_backend_type_t type) {
    for (int i = 0; i < UNICC_MAX_BACKENDS; i++) {
        if (unicc_backends[i].type == type) {
            return unicc_backends[i].name;
        }
    }
    return "unknown";
}

const char* unicc_loader_get_env_backend(void) {
    return get_nonempty_env("UNICC_BACKEND");
}

const char* unicc_loader_get_env_libpath(void) {
    return get_nonempty_env("UNICC_LIBRARY");
}

int unicc_loader_detect_backend(const char **out_lib_path) {
    if (!out_lib_path) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }

    const char *env_libpath = unicc_loader_get_env_libpath();
    const char *env_backend = unicc_loader_get_env_backend();

    /* Priority 1: exact library path (used by CI and tests). */
    if (env_libpath) {
        fprintf(stderr, "[UniCCL] Using library path from UNICC_LIBRARY: %s\n", env_libpath);
        *out_lib_path = env_libpath;
        return UNICC_OK;
    }

    /* Priority 2: backend name. An unrecognized name is treated as a path. */
    if (env_backend) {
        fprintf(stderr, "[UniCCL] Using backend from UNICC_BACKEND: %s\n", env_backend);
        for (int i = 0; i < UNICC_MAX_BACKENDS; i++) {
            if (strcmp(unicc_backends[i].name, env_backend) == 0) {
                *out_lib_path = unicc_backends[i].lib_name;
                return UNICC_OK;
            }
        }
        *out_lib_path = env_backend;
        return UNICC_OK;
    }

    /* Priority 3: platform default = NVIDIA NCCL. CUDA/ROCm presence
     * detection to pick a smarter default is a future refinement (M2 keeps
     * this deliberately simple; docs/BACKENDS.md says so). */
    fprintf(stderr, "[UniCCL] Auto-detecting collective-communication backend...\n");
    *out_lib_path = unicc_backends[0].lib_name;
    return UNICC_OK;
}

int unicc_loader_load(const char *lib_path, unicc_lib_handle_t *out_handle) {
    if (!out_handle) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    *out_handle = NULL;

    if (!lib_path) {
        fprintf(stderr, "[UniCCL:ERROR] No backend library path provided\n");
        return UNICC_ERR_NO_BACKEND;
    }

    fprintf(stderr, "[UniCCL] Loading backend library: %s\n", lib_path);

    unicc_lib_handle_t handle = unicc_platform_dlopen(lib_path);
    if (!handle) {
        /* A backend may prefer a specific name (libnccl.so) but fall back to
         * its soname (libnccl.so.2) when the dev name is absent. */
        const char *alt = NULL;
        for (int i = 0; i < UNICC_MAX_BACKENDS; i++) {
            if (unicc_backends[i].lib_name &&
                strcmp(unicc_backends[i].lib_name, lib_path) == 0) {
                alt = unicc_backends[i].lib_name_alt;
                break;
            }
        }
        if (alt) {
            fprintf(stderr, "[UniCCL] %s not found, trying fallback %s\n", lib_path, alt);
            handle = unicc_platform_dlopen(alt);
        }
    }
    if (!handle) {
        fprintf(stderr, "[UniCCL:ERROR] Failed to load backend library: %s\n", lib_path);
        fprintf(stderr, "[UniCCL:ERROR] %s\n", unicc_platform_dlerror());
        return UNICC_ERR_BACKEND_LOAD;
    }

    fprintf(stderr, "[UniCCL] Successfully loaded backend library\n");
    *out_handle = handle;
    return UNICC_OK;
}

void unicc_loader_unload(unicc_lib_handle_t handle) {
    unicc_platform_dlclose(handle);
}

unicc_backend_type_t unicc_loader_identify_backend(unicc_lib_handle_t handle) {
    if (!handle) {
        fprintf(stderr, "[UniCCL:WARN] Cannot identify backend: null handle\n");
        return UNICC_BACKEND_UNKNOWN;
    }

    fprintf(stderr, "[UniCCL] Identifying backend type...\n");

    /* Defensive dual-family rule: if a library exports rccl*, a librccl that
     * also carries ncclGetVersion must never be misidentified as NVIDIA NCCL,
     * so the rccl* probe stays first. NOTE (docs/official/): modern AMD RCCL
     * exposes no public rccl* symbols, so on real RCCL/DCU this branch does
     * not fire and the library resolves via ncclGetVersion below as NCCL. */
    if (unicc_platform_dlsym(handle, "rcclGetVersion") != NULL) {
        fprintf(stderr, "[UniCCL] Detected RCCL backend\n");
        return UNICC_BACKEND_RCCL;
    }

    if (unicc_platform_dlsym(handle, "ncclGetVersion") != NULL) {
        fprintf(stderr, "[UniCCL] Detected NCCL backend\n");
        return UNICC_BACKEND_NCCL;
    }

    fprintf(stderr, "[UniCCL:WARN] Could not identify backend type\n");
    return UNICC_BACKEND_UNKNOWN;
}

int unicc_loader_check_platform_support(unicc_backend_type_t backend) {
    if (!backend_supported_on_platform(backend)) {
        fprintf(stderr, "[UniCCL:ERROR] Backend '%s' is not supported on this platform\n",
                backend_name_from_type(backend));
        fprintf(stderr, "  NCCL and RCCL are supported on Linux (NCCL also on macOS)\n");
        return UNICC_ERR_BACKEND_NOT_SUPPORTED;
    }
    return UNICC_OK;
}

void unicc_diagnose_backend(const char *lib_path) {
    fprintf(stderr, "\n[UniCCL] === Backend Diagnostics ===\n");
    fprintf(stderr, "Library path: %s\n", lib_path ? lib_path : "(null)");

    const char *env_backend = getenv("UNICC_BACKEND");
    const char *env_libpath = getenv("UNICC_LIBRARY");
    fprintf(stderr, "UNICC_BACKEND: %s\n", env_backend ? env_backend : "(not set)");
    fprintf(stderr, "UNICC_LIBRARY: %s\n", env_libpath ? env_libpath : "(not set)");

    unicc_lib_handle_t handle;
    int ret = unicc_loader_load(lib_path, &handle);
    if (ret != UNICC_OK) {
        fprintf(stderr, "Failed to load backend: %s\n", unicc_error_string(ret));
        if (ret == UNICC_ERR_BACKEND_LOAD) {
            fprintf(stderr, "%s", unicc_platform_load_advice());
        }
        return;
    }

    unicc_backend_type_t type = unicc_loader_identify_backend(handle);
    fprintf(stderr, "Identified backend type: %s\n",
            type == UNICC_BACKEND_NCCL ? "NCCL" :
            type == UNICC_BACKEND_RCCL ? "RCCL" : "Unknown");

    fprintf(stderr, "\nChecking symbols:\n");
    /* Both families are listed so the report is useful regardless of which
     * backend's library is being diagnosed. */
    const char *required_symbols[] = {
        "ncclGetVersion", "rcclGetVersion",
        "ncclCommInitRank", "rcclCommInitRank",
        "ncclAllReduce", "rcclAllReduce",
        "ncclBroadcast", "rcclBroadcast",
        "ncclGroupStart", "rcclGroupStart",
        "ncclGroupEnd", "rcclGroupEnd",
        NULL
    };
    for (int i = 0; required_symbols[i] != NULL; i++) {
        void *sym = unicc_platform_dlsym(handle, required_symbols[i]);
        fprintf(stderr, "  %-18s %s\n", required_symbols[i], sym ? "OK" : "NOT FOUND");
    }

    unicc_loader_unload(handle);
    fprintf(stderr, "\n=== End Diagnostics ===\n\n");
}

int unicc_print_backend_info(void) {
    fprintf(stderr, "\n[UniCCL] Supported Backends:\n");
    fprintf(stderr, "  %-8s %-12s %s\n", "Name", "Library", "Fallback");
    fprintf(stderr, "  %-8s %-12s %s\n", "----------", "------------", "----------");
    for (int i = 0; i < UNICC_MAX_BACKENDS; i++) {
        fprintf(stderr, "  %-8s %-12s %s\n",
                unicc_backends[i].name,
                unicc_backends[i].lib_name,
                unicc_backends[i].lib_name_alt ? unicc_backends[i].lib_name_alt : "-");
    }
    fprintf(stderr, "\n");
    return UNICC_OK;
}
