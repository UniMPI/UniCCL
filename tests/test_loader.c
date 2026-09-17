/* test_loader.c - loader-level unit tests (no xcc_init needed).
 *
 * Exercises backend selection, dlopen failure, and - critically - backend
 * identification ordering: a library that exports BOTH rccl* and nccl*
 * symbols (as real RCCL does) must be identified as RCCL, never NCCL.
 *
 * Fake fixture paths are passed on argv by tests/CMakeLists.txt. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xcc_backends.h"

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
    } \
} while (0)

static void reset_env(void) {
    unsetenv("XCCL_LIBRARY");
    unsetenv("XCCL_BACKEND");
}

int main(int argc, char **argv) {
    const char *fake_nccl = NULL;
    const char *fake_rccl = NULL;
    const char *fake_nccl_missing = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fake-nccl") == 0 && i + 1 < argc) { fake_nccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-rccl") == 0 && i + 1 < argc) { fake_rccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-nccl-missing") == 0 && i + 1 < argc) { fake_nccl_missing = argv[++i]; }
    }

    /* Default detection -> NVIDIA NCCL. */
    reset_env();
    const char *path = NULL;
    CHECK(xcc_loader_detect_backend(&path) == XCC_OK);
    CHECK(path && strstr(path, "libnccl"));

    /* XCCL_BACKEND=rccl -> librccl.so. */
    setenv("XCCL_BACKEND", "rccl", 1);
    CHECK(xcc_loader_detect_backend(&path) == XCC_OK);
    CHECK(path && strstr(path, "librccl"));

    /* Unknown backend name is treated as a library path. */
    setenv("XCCL_BACKEND", "/some/other/libfoo.so", 1);
    CHECK(xcc_loader_detect_backend(&path) == XCC_OK);
    CHECK(path && strcmp(path, "/some/other/libfoo.so") == 0);

    /* XCCL_LIBRARY (exact path) beats XCCL_BACKEND (name). */
    setenv("XCCL_BACKEND", "rccl", 1);
    setenv("XCCL_LIBRARY", "/tmp/precise.so", 1);
    CHECK(xcc_loader_detect_backend(&path) == XCC_OK);
    CHECK(path && strcmp(path, "/tmp/precise.so") == 0);
    reset_env();

    /* dlopen failure on a bogus path (also exercises the soname fallback). */
    xcc_lib_handle_t handle = NULL;
    CHECK(xcc_loader_load("/no/such/libnccl.so.99", &handle) == XCC_ERR_BACKEND_LOAD);
    CHECK(handle == NULL);

    /* Identify (null handle) -> UNKNOWN. */
    CHECK(xcc_loader_identify_backend(NULL) == XCC_BACKEND_UNKNOWN);

    /* Identify a pure NCCL-shaped fake -> NCCL. */
    if (fake_nccl) {
        CHECK(xcc_loader_load(fake_nccl, &handle) == XCC_OK);
        CHECK(xcc_loader_identify_backend(handle) == XCC_BACKEND_NCCL);
        xcc_loader_unload(handle);
    }

    /* Identify the RCCL fake, which exports nccl* compat symbols as well ->
     * MUST be RCCL (rccl* exclusives win). This is the discrimination test. */
    if (fake_rccl) {
        CHECK(xcc_loader_load(fake_rccl, &handle) == XCC_OK);
        CHECK(xcc_platform_dlsym(handle, "ncclGetVersion") != NULL); /* compat present */
        CHECK(xcc_platform_dlsym(handle, "rcclGetVersion") != NULL); /* native present */
        CHECK(xcc_loader_identify_backend(handle) == XCC_BACKEND_RCCL);
        xcc_loader_unload(handle);
    }

    /* Missing-symbol variant is still an NCCL identity. */
    if (fake_nccl_missing) {
        CHECK(xcc_loader_load(fake_nccl_missing, &handle) == XCC_OK);
        CHECK(xcc_loader_identify_backend(handle) == XCC_BACKEND_NCCL);
        xcc_loader_unload(handle);
    }

    /* Backend table sanity. */
    CHECK(xcc_backends[XCC_BACKEND_NCCL - 1].type == XCC_BACKEND_NCCL);
    CHECK(strcmp(xcc_backends[0].name, "nccl") == 0);
    CHECK(strcmp(xcc_backends[1].name, "rccl") == 0);
    CHECK(xcc_print_backend_info() == XCC_OK);

    if (g_failures == 0) {
        printf("test_loader: ALL TESTS PASSED\n");
        return 0;
    }
    printf("test_loader: %d FAILURE(S)\n", g_failures);
    return 1;
}
