/* xcc_vtable.c - zero-initialized dispatch table, core validation, per-backend
 * init dispatch and cleanup.
 *
 * Mechanism modeled on UniMPI's vtable (src/vtable.c): a global
 * zero-initialized table, a small core-symbol validation, identification of
 * the loaded library, and a switch that fills the table from the matching
 * backend binding. */
#include "xcc_vtable.h"
#include "xcc_backends.h"
#include "xcc_platform.h"
#include "xcc_errors.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global vtable instance - zero-initialized. */
xcc_vtable_t xcc = {0};

static xcc_lib_handle_t g_backend_handle = NULL;
static xcc_backend_type_t g_backend_type = XCC_BACKEND_UNKNOWN;

/* Backend bindings live in separate translation units. */
int xcc_vtable_init_nccl(xcc_lib_handle_t handle);
int xcc_vtable_init_rccl(xcc_lib_handle_t handle);

xcc_backend_type_t xcc_get_backend_type(void) {
    return g_backend_type;
}

static void* load_symbol(xcc_lib_handle_t handle, const char *name) {
    return xcc_platform_dlsym(handle, name);
}

/* Validate the minimal set a backend must export for the M2 collective face to
 * work. Each core slot accepts either the nccl* or the rccl* spelling, because
 * validation runs before identification. Symbols outside this set degrade to a
 * NULL slot and are gated by *_available() (see docs/SUPPORT_MATRIX.md). */
int xcc_vtable_validate_core(xcc_lib_handle_t handle) {
    if ((!load_symbol(handle, "ncclGetVersion") && !load_symbol(handle, "rcclGetVersion")) ||
        (!load_symbol(handle, "ncclCommInitRank") && !load_symbol(handle, "rcclCommInitRank")) ||
        (!load_symbol(handle, "ncclAllReduce") && !load_symbol(handle, "rcclAllReduce")) ||
        (!load_symbol(handle, "ncclBroadcast") && !load_symbol(handle, "rcclBroadcast"))) {
        fprintf(stderr, "[xccl:ERROR] Backend library does not export the required core symbols\n");
        return XCC_ERR_SYMBOL_NOT_FOUND;
    }
    return XCC_OK;
}

/* Initialize the vtable for the given loaded backend library. */
int xcc_vtable_init(xcc_lib_handle_t handle) {
    int ret;

    /* Validate the core surface first. */
    ret = xcc_vtable_validate_core(handle);
    if (ret != XCC_OK) {
        return ret;
    }

    /* Identify the family, then bind the matching symbols. */
    g_backend_type = xcc_loader_identify_backend(handle);
    g_backend_handle = handle;

    ret = xcc_loader_check_platform_support(g_backend_type);
    if (ret != XCC_OK) {
        return ret;
    }

    switch (g_backend_type) {
        case XCC_BACKEND_NCCL:
            ret = xcc_vtable_init_nccl(handle);
            break;
        case XCC_BACKEND_RCCL:
            ret = xcc_vtable_init_rccl(handle);
            break;
        default:
            fprintf(stderr, "[xccl:ERROR] Unknown backend type\n");
            ret = XCC_ERR_BACKEND_INIT_FAILED;
            break;
    }

    return ret;
}

/* Reset the dispatch table to zeros. Does not unload the backend handle;
 * ownership of the handle belongs to the API layer (xcc_api.c). */
void xcc_vtable_cleanup(void) {
    memset(&xcc, 0, sizeof(xcc_vtable_t));
    g_backend_handle = NULL;
    g_backend_type = XCC_BACKEND_UNKNOWN;
}
