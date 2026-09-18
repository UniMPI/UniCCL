/* unicc_vtable.c - zero-initialized dispatch table, core validation, per-backend
 * init dispatch and cleanup.
 *
 * Mechanism modeled on UniMPI's vtable (src/vtable.c): a global
 * zero-initialized table, a small core-symbol validation, identification of
 * the loaded library, and a switch that fills the table from the matching
 * backend binding. */
#include "unicc_vtable.h"
#include "unicc_backends.h"
#include "unicc_platform.h"
#include "unicc_errors.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global vtable instance - zero-initialized. */
unicc_vtable_t unicc = {0};

static unicc_lib_handle_t g_backend_handle = NULL;
static unicc_backend_type_t g_backend_type = UNICC_BACKEND_UNKNOWN;

/* Backend bindings live in separate translation units. */
int unicc_vtable_init_nccl(unicc_lib_handle_t handle);
int unicc_vtable_init_rccl(unicc_lib_handle_t handle);

unicc_backend_type_t unicc_get_backend_type(void) {
    return g_backend_type;
}

static void* load_symbol(unicc_lib_handle_t handle, const char *name) {
    return unicc_platform_dlsym(handle, name);
}

/* Validate the minimal set a backend must export for the M2 collective face to
 * work. Each core slot accepts either the nccl* or the rccl* spelling, because
 * validation runs before identification. Symbols outside this set degrade to a
 * NULL slot and are gated by *_available() (see docs/SUPPORT_MATRIX.md). */
int unicc_vtable_validate_core(unicc_lib_handle_t handle) {
    if ((!load_symbol(handle, "ncclGetVersion") && !load_symbol(handle, "rcclGetVersion")) ||
        (!load_symbol(handle, "ncclCommInitRank") && !load_symbol(handle, "rcclCommInitRank")) ||
        (!load_symbol(handle, "ncclAllReduce") && !load_symbol(handle, "rcclAllReduce")) ||
        (!load_symbol(handle, "ncclBroadcast") && !load_symbol(handle, "rcclBroadcast"))) {
        fprintf(stderr, "[UniCCL:ERROR] Backend library does not export the required core symbols\n");
        return UNICC_ERR_SYMBOL_NOT_FOUND;
    }
    return UNICC_OK;
}

/* Initialize the vtable for the given loaded backend library. */
int unicc_vtable_init(unicc_lib_handle_t handle) {
    int ret;

    /* Validate the core surface first. */
    ret = unicc_vtable_validate_core(handle);
    if (ret != UNICC_OK) {
        return ret;
    }

    /* Identify the family, then bind the matching symbols. */
    g_backend_type = unicc_loader_identify_backend(handle);
    g_backend_handle = handle;

    ret = unicc_loader_check_platform_support(g_backend_type);
    if (ret != UNICC_OK) {
        return ret;
    }

    switch (g_backend_type) {
        case UNICC_BACKEND_NCCL:
            ret = unicc_vtable_init_nccl(handle);
            break;
        case UNICC_BACKEND_RCCL:
            ret = unicc_vtable_init_rccl(handle);
            break;
        default:
            fprintf(stderr, "[UniCCL:ERROR] Unknown backend type\n");
            ret = UNICC_ERR_BACKEND_INIT_FAILED;
            break;
    }

    return ret;
}

/* Reset the dispatch table to zeros. Does not unload the backend handle;
 * ownership of the handle belongs to the API layer (unicc_api.c). */
void unicc_vtable_cleanup(void) {
    memset(&unicc, 0, sizeof(unicc_vtable_t));
    g_backend_handle = NULL;
    g_backend_type = UNICC_BACKEND_UNKNOWN;
}
