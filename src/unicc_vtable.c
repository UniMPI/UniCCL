/* unicc_vtable.c - zero-initialized dispatch table, per-backend init dispatch
 * and cleanup.
 *
 * Mechanism modeled on UniMPI's vtable (src/vtable.c): a global
 * zero-initialized table, identification of the loaded library, and a switch
 * that fills the table from the matching backend binding. Core-symbol
 * validation happens inside the binding, and only against the IDENTIFIED
 * family (unicc_bind.c) - never a mixed-family OR-set, so a hybrid or
 * incomplete library cannot ride one family's validation into another
 * family's binding (review F3). */
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
int unicc_vtable_init_oneccl(unicc_lib_handle_t handle);
int unicc_vtable_init_eccl(unicc_lib_handle_t handle);

unicc_backend_type_t unicc_get_backend_type(void) {
    return g_backend_type;
}

/* Initialize the vtable for the given loaded backend library.
 *
 * Flow: identify FIRST - a library is bound by its own family, never by some
 * other family's spelling of the core symbols. The identified family's core
 * symbols are then required inside unicc_vtable_bind (missing =>
 * UNICC_ERR_SYMBOL_NOT_FOUND => unicc_init fails). Optional symbols degrade to
 * a NULL slot and are gated by *_available(). The global type/handle are
 * published only after everything succeeded, and a failed init restores the
 * table to all-zeros - no torn slots, no dangling handle (F7). */
int unicc_vtable_init(unicc_lib_handle_t handle) {
    unicc_backend_type_t type = unicc_loader_identify_backend(handle);
    if (type == UNICC_BACKEND_UNKNOWN) {
        fprintf(stderr, "[UniCCL:ERROR] Could not identify backend type\n");
        return UNICC_ERR_SYMBOL_NOT_FOUND;
    }

    int ret = unicc_loader_check_platform_support(type);
    if (ret != UNICC_OK) {
        return ret;
    }

    switch (type) {
        case UNICC_BACKEND_NCCL:
            ret = unicc_vtable_init_nccl(handle);
            break;
        case UNICC_BACKEND_RCCL:
            ret = unicc_vtable_init_rccl(handle);
            break;
        case UNICC_BACKEND_ONECCL:
            ret = unicc_vtable_init_oneccl(handle);
            break;
        case UNICC_BACKEND_ECCL:
            ret = unicc_vtable_init_eccl(handle);
            break;
        default:
            fprintf(stderr, "[UniCCL:ERROR] Unknown backend type\n");
            ret = UNICC_ERR_BACKEND_INIT_FAILED;
            break;
    }

    if (ret != UNICC_OK) {
        /* Do not leave a torn table or a stale identified type behind. */
        memset(&unicc, 0, sizeof(unicc_vtable_t));
        return ret;
    }

    g_backend_type = type;
    g_backend_handle = handle;
    return UNICC_OK;
}

/* Reset the dispatch table to zeros. Does not unload the backend handle;
 * ownership of the handle belongs to the API layer (unicc_api.c). */
void unicc_vtable_cleanup(void) {
    memset(&unicc, 0, sizeof(unicc_vtable_t));
    g_backend_handle = NULL;
    g_backend_type = UNICC_BACKEND_UNKNOWN;
}
