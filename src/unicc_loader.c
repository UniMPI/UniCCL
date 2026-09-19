/* unicc_loader.c - backend probing, loading, identification and diagnostics.
 *
 * Mechanism modeled on UniMPI's loader (src/loader.c): a small fixed backend
 * table, an env-driven deterministic selection, dlopen with an alternate
 * soname fallback, identify-by-feature-symbol, and a per-symbol diagnose that
 * is invaluable on hosts with no NCCL/RCCL installed. */
#include "unicc_backends.h"
#include "unicc_native_id.h"   /* UNICC_ONECCL_UNIQUE_ID_BYTES (table row below) */
#include "unicc_version.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Backend definitions.
 *
 * NVIDIA NCCL ships a shared library whose dev name is libnccl.so and whose
 * runtime soname is libnccl.so.2 (older releases used .1); prefer the dev
 * name and fall back to the soname.
 * AMD RCCL ships librccl.so (soname librccl.so.1). Modern RCCL's public API is
 * nccl*-named, so on a real RCCL the loader resolves it via the nccl binding;
 * the rccl entry stays as the defensive rccl*-family path.
 * Intel oneCCL v2 (the NCCL-aligned C API, default branch since 2022.1) ships
 * libccl.so.2; the classic C++-API line used libccl.so.1. Symbol prefix
 * oneccl*.
 * Enflame ECCL ships libeccl.so (TopsRider suite); soname inferred from
 * torch-gcu usage, not yet confirmed by nm -D on a real host.
 * Each row is the single source of every per-family fact (prefix, id size,
 * signature quirks). Identification probes rows in DESCENDING priority (see
 * identify_backend), so keep the table ordered by ascending priority and put
 * the specific-prefix families ahead of the generic nccl fallback. */
const unicc_backend_info_t unicc_backends[UNICC_MAX_BACKENDS] = {
    {UNICC_BACKEND_NCCL,   "nccl",   "libnccl.so",  "libnccl.so.2", "ncclGetVersion",   2,
     128, UNICC_UID_NCCL,   0, 0, 1},
    {UNICC_BACKEND_RCCL,   "rccl",   "librccl.so",  "librccl.so.1", "rcclGetVersion",   3,
     128, UNICC_UID_NCCL,   0, 0, 1},
    {UNICC_BACKEND_ONECCL, "oneccl", "libccl.so.2", "libccl.so",    "onecclGetVersion", 4,
     UNICC_ONECCL_UNIQUE_ID_BYTES, UNICC_UID_ONECCL, 1, 1, 1},
    {UNICC_BACKEND_ECCL,   "eccl",   "libeccl.so",  NULL,           "ecclGetVersion",   5,
     128, UNICC_UID_NCCL,   1, 1, 0}
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

    /* Priority 2: backend name. An unrecognized name is treated as a library
     * path (a power-user escape hatch), but warn first so a typo like
     * UNICC_BACKEND=ncccl is not silently swallowed as a path (f). */
    if (env_backend) {
        fprintf(stderr, "[UniCCL] Using backend from UNICC_BACKEND: %s\n", env_backend);
        for (int i = 0; i < UNICC_MAX_BACKENDS; i++) {
            if (strcmp(unicc_backends[i].name, env_backend) == 0) {
                *out_lib_path = unicc_backends[i].lib_name;
                return UNICC_OK;
            }
        }
        fprintf(stderr, "[UniCCL] '%s' is not a known backend name; treating it as a library path\n",
                env_backend);
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

int unicc_loader_load(const char *lib_path, unicc_lib_handle_t *out_handle,
                      char *resolved_path, size_t resolved_cap) {
    if (!out_handle) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    *out_handle = NULL;
    if (resolved_path && resolved_cap) {
        resolved_path[0] = '\0';
    }

    if (!lib_path) {
        fprintf(stderr, "[UniCCL:ERROR] No backend library path provided\n");
        return UNICC_ERR_NO_BACKEND;
    }

    fprintf(stderr, "[UniCCL] Loading backend library: %s\n", lib_path);

    const char *loaded = lib_path;
    unicc_lib_handle_t handle = unicc_platform_dlopen(lib_path);
    if (!handle) {
        /* dlerror() is single-buffered; capture the PRIMARY failure now, before
         * any fallback dlopen overwrites it, so a corrupt primary is not
         * misreported as an absent fallback (F15). */
        const char *primary_err = unicc_platform_dlerror();

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
            if (handle) {
                loaded = alt;
            }
        }

#if defined(__APPLE__)
        /* macOS names shared libraries *.dylib, but every table lib_name is a
         * .so. Retry the .dylib spelling of the requested name before giving
         * up; this covers locally-built dylibs (NCCL ships no macOS binary -
         * see docs/BACKENDS.md "Platform support"). */
        if (!handle) {
            size_t n = strlen(lib_path);
            if (n > 3 && strcmp(lib_path + n - 3, ".so") == 0) {
                char dylib[160];
                snprintf(dylib, sizeof dylib, "%.*s.dylib", (int)(n - 3), lib_path);
                fprintf(stderr, "[UniCCL] %s not found, trying macOS spelling %s\n",
                        lib_path, dylib);
                handle = unicc_platform_dlopen(dylib);
                if (handle) {
                    loaded = dylib;
                }
            }
        }
#endif /* __APPLE__ */

        if (!handle) {
            fprintf(stderr, "[UniCCL:ERROR] Failed to load backend library: %s\n", lib_path);
            fprintf(stderr, "[UniCCL:ERROR] %s\n", primary_err ? primary_err : "(no error)");
            return UNICC_ERR_BACKEND_LOAD;
        }
    }

    fprintf(stderr, "[UniCCL] Successfully loaded backend library\n");
    /* Report the name that actually dlopen'd (F6): lib_path, the soname fallback,
     * or the macOS spelling - never a guess. */
    if (resolved_path && resolved_cap) {
        snprintf(resolved_path, resolved_cap, "%s", loaded);
    }
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

    /* Probe order matters: each library exports its own prefix, so the
     * specific families (eccl / oneccl / rccl) must be probed BEFORE the
     * generic nccl fallback - a library that also happens to export
     * ncclGetVersion (real RCCL / Hygon DCU) must never be misidentified as
     * NVIDIA NCCL. unicc_backends[] is ordered by ASCENDING priority, so this
     * walks it from the last row downward (descending priority). probe_symbol
     * is the single source of identifying symbols (F11). Defensive note
     * (docs/official/): modern AMD RCCL exposes no public rccl symbols, so on
     * real RCCL/DCU the rccl row does not fire and the library resolves as
     * NCCL below. */
    for (int i = UNICC_MAX_BACKENDS - 1; i >= 0; i--) {
        if (unicc_platform_dlsym(handle, unicc_backends[i].probe_symbol) != NULL) {
            fprintf(stderr, "[UniCCL] Detected %s backend\n", unicc_backends[i].name);
            return unicc_backends[i].type;
        }
    }

    fprintf(stderr, "[UniCCL:WARN] Could not identify backend type\n");
    return UNICC_BACKEND_UNKNOWN;
}

int unicc_loader_check_platform_support(unicc_backend_type_t backend) {
    if (!backend_supported_on_platform(backend)) {
        fprintf(stderr, "[UniCCL:ERROR] Backend '%s' is not supported on this platform\n",
                backend_name_from_type(backend));
        fprintf(stderr, "  NCCL, RCCL, oneCCL and ECCL are supported on Linux\n");
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
    int ret = unicc_loader_load(lib_path, &handle, NULL, 0);
    if (ret != UNICC_OK) {
        fprintf(stderr, "Failed to load backend: %s\n", unicc_error_string(ret));
        if (ret == UNICC_ERR_BACKEND_LOAD) {
            fprintf(stderr, "%s", unicc_platform_load_advice());
        }
        return;
    }

    unicc_backend_type_t type = unicc_loader_identify_backend(handle);
    fprintf(stderr, "Identified backend type: %s\n", backend_name_from_type(type));

    fprintf(stderr, "\nChecking symbols:\n");
    /* Every registered family is listed so the report is useful regardless of
     * which backend's library is being diagnosed. Built from unicc_backends[]
     * (the single source of per-family names, F11) - no hand-maintained list
     * to drift. */
    static const char *suffixes[] = {
        "GetVersion", "CommInitRank", "AllReduce", "Broadcast",
        "GroupStart", "GroupEnd", NULL
    };
    char sym[96];
    for (int b = 0; b < UNICC_MAX_BACKENDS; b++) {
        for (int s = 0; suffixes[s]; s++) {
            snprintf(sym, sizeof sym, "%s%s", unicc_backends[b].name, suffixes[s]);
            void *found = unicc_platform_dlsym(handle, sym);
            fprintf(stderr, "  %-22s %s\n", sym, found ? "OK" : "NOT FOUND");
        }
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
