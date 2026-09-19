/* test_loader.c - loader-level unit tests (no unicc_init needed).
 *
 * Exercises backend selection, dlopen failure, and - critically - backend
 * identification ordering: a library that exports BOTH rccl* and nccl*
 * symbols (as real RCCL does) must be identified as RCCL, never NCCL.
 *
 * Fake fixture paths are passed on argv by tests/CMakeLists.txt. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unicc_backends.h"

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
    } \
} while (0)

static void reset_env(void) {
    unsetenv("UNICC_LIBRARY");
    unsetenv("UNICC_BACKEND");
}

int main(int argc, char **argv) {
    const char *fake_nccl = NULL;
    const char *fake_rccl = NULL;
    const char *fake_nccl_missing = NULL;
    const char *fake_oneccl = NULL;
    const char *fake_eccl = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fake-nccl") == 0 && i + 1 < argc) { fake_nccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-rccl") == 0 && i + 1 < argc) { fake_rccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-nccl-missing") == 0 && i + 1 < argc) { fake_nccl_missing = argv[++i]; }
        else if (strcmp(argv[i], "--fake-oneccl") == 0 && i + 1 < argc) { fake_oneccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-eccl") == 0 && i + 1 < argc) { fake_eccl = argv[++i]; }
    }

    /* Default detection -> NVIDIA NCCL. */
    reset_env();
    const char *path = NULL;
    CHECK(unicc_loader_detect_backend(&path) == UNICC_OK);
    CHECK(path && strstr(path, "libnccl"));

    /* UNICC_BACKEND=rccl -> librccl.so. */
    setenv("UNICC_BACKEND", "rccl", 1);
    CHECK(unicc_loader_detect_backend(&path) == UNICC_OK);
    CHECK(path && strstr(path, "librccl"));

    /* Unknown backend name is treated as a library path. */
    setenv("UNICC_BACKEND", "/some/other/libfoo.so", 1);
    CHECK(unicc_loader_detect_backend(&path) == UNICC_OK);
    CHECK(path && strcmp(path, "/some/other/libfoo.so") == 0);

    /* UNICC_LIBRARY (exact path) beats UNICC_BACKEND (name). */
    setenv("UNICC_BACKEND", "rccl", 1);
    setenv("UNICC_LIBRARY", "/tmp/precise.so", 1);
    CHECK(unicc_loader_detect_backend(&path) == UNICC_OK);
    CHECK(path && strcmp(path, "/tmp/precise.so") == 0);
    reset_env();

    /* dlopen failure on a bogus path (also exercises the soname fallback). */
    unicc_lib_handle_t handle = NULL;
    CHECK(unicc_loader_load("/no/such/libnccl.so.99", &handle, NULL, 0) == UNICC_ERR_BACKEND_LOAD);
    CHECK(handle == NULL);

    /* Identify (null handle) -> UNKNOWN. */
    CHECK(unicc_loader_identify_backend(NULL) == UNICC_BACKEND_UNKNOWN);

    /* Identify a pure NCCL-shaped fake -> NCCL. */
    if (fake_nccl) {
        CHECK(unicc_loader_load(fake_nccl, &handle, NULL, 0) == UNICC_OK);
        CHECK(unicc_loader_identify_backend(handle) == UNICC_BACKEND_NCCL);
        unicc_loader_unload(handle);
    }

    /* Identify the RCCL fake, which exports nccl* compat symbols as well ->
     * MUST be RCCL (rccl* exclusives win). This is the discrimination test. */
    if (fake_rccl) {
        CHECK(unicc_loader_load(fake_rccl, &handle, NULL, 0) == UNICC_OK);
        CHECK(unicc_platform_dlsym(handle, "ncclGetVersion") != NULL); /* compat present */
        CHECK(unicc_platform_dlsym(handle, "rcclGetVersion") != NULL); /* native present */
        CHECK(unicc_loader_identify_backend(handle) == UNICC_BACKEND_RCCL);
        unicc_loader_unload(handle);
    }

    /* Missing-symbol variant is still an NCCL identity. */
    if (fake_nccl_missing) {
        CHECK(unicc_loader_load(fake_nccl_missing, &handle, NULL, 0) == UNICC_OK);
        CHECK(unicc_loader_identify_backend(handle) == UNICC_BACKEND_NCCL);
        unicc_loader_unload(handle);
    }

    /* oneCCL v2-shaped fake -> ONECCL (its oneccl* prefixes are specific enough
     * that the generic nccl* probe must NOT claim it). */
    if (fake_oneccl) {
        CHECK(unicc_loader_load(fake_oneccl, &handle, NULL, 0) == UNICC_OK);
        CHECK(unicc_platform_dlsym(handle, "onecclGetVersion") != NULL);
        CHECK(unicc_platform_dlsym(handle, "ncclGetVersion") == NULL); /* no nccl* family */
        CHECK(unicc_loader_identify_backend(handle) == UNICC_BACKEND_ONECCL);
        unicc_loader_unload(handle);
    }

    /* ECCL-shaped fake -> ECCL. */
    if (fake_eccl) {
        CHECK(unicc_loader_load(fake_eccl, &handle, NULL, 0) == UNICC_OK);
        CHECK(unicc_platform_dlsym(handle, "ecclGetVersion") != NULL);
        CHECK(unicc_loader_identify_backend(handle) == UNICC_BACKEND_ECCL);
        unicc_loader_unload(handle);
    }

    /* Backend table sanity. */
    CHECK(unicc_backends[UNICC_BACKEND_NCCL - 1].type == UNICC_BACKEND_NCCL);
    CHECK(strcmp(unicc_backends[0].name, "nccl") == 0);
    CHECK(strcmp(unicc_backends[1].name, "rccl") == 0);
    CHECK(strcmp(unicc_backends[2].name, "oneccl") == 0);
    CHECK(strcmp(unicc_backends[3].name, "eccl") == 0);
    CHECK(strcmp(unicc_backends[2].probe_symbol, "onecclGetVersion") == 0);
    CHECK(strcmp(unicc_backends[3].probe_symbol, "ecclGetVersion") == 0);
    CHECK(unicc_print_backend_info() == UNICC_OK);

    if (g_failures == 0) {
        printf("test_loader: ALL TESTS PASSED\n");
        return 0;
    }
    printf("test_loader: %d FAILURE(S)\n", g_failures);
    return 1;
}
