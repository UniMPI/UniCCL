/* unicc_vtable.c - zero-initialized dispatch table, core validation, per-backend
 * init dispatch and cleanup.
 *
 * Mechanism modeled on UniMPI's vtable (src/vtable.c): a global
 * zero-initialized table, a small core-symbol validation, identification of
 * the loaded library, and a switch that fills the table from the matching
 * backend binding. */
#include "unicc_vtable.h"
#include "unicc_backends.h"
#include "unicc_dtmap.h"
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

static void* load_symbol(unicc_lib_handle_t handle, const char *name) {
    return unicc_platform_dlsym(handle, name);
}

/* Validate the minimal set a backend must export for the collective face to
 * work. Each core slot accepts any registered vendor's spelling, because
 * validation runs before identification; a pure oneCCL or ECCL library must
 * not fail this check for lacking nccl* symbols. Symbols outside this set
 * degrade to a NULL slot and are gated by *_available()
 * (see docs/SUPPORT_MATRIX.md). P2 extends each OR-set with the CNCL/HCCL/MCCL
 * spellings when those backends land. */
static int core_symbol_ok(unicc_lib_handle_t handle, const char **spellings) {
    for (int i = 0; spellings[i] != NULL; i++) {
        if (load_symbol(handle, spellings[i]) != NULL) {
            return 1;
        }
    }
    return 0;
}

int unicc_vtable_validate_core(unicc_lib_handle_t handle) {
    const char *get_version[] = {"ncclGetVersion", "rcclGetVersion",
                                 "onecclGetVersion", "ecclGetVersion", NULL};
    const char *init_rank[]   = {"ncclCommInitRank", "rcclCommInitRank",
                                 "onecclCommInitRank", "ecclCommInitRank", NULL};
    const char *allreduce[]   = {"ncclAllReduce", "rcclAllReduce",
                                 "onecclAllReduce", "ecclAllReduce", NULL};
    const char *broadcast[]   = {"ncclBroadcast", "rcclBroadcast",
                                 "onecclBroadcast", "ecclBroadcast", NULL};
    if (!core_symbol_ok(handle, get_version) ||
        !core_symbol_ok(handle, init_rank) ||
        !core_symbol_ok(handle, allreduce) ||
        !core_symbol_ok(handle, broadcast)) {
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

    /* Datatype/op tables start at the NCCL-family numbering; the binding below
     * may overwrite entries where a vendor's numbering differs. This reset
     * keeps every init deterministic even if a previous init left a divergent
     * backend's table in place. */
    unicc_dtmap_reset_nccl();

    switch (g_backend_type) {
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

    return ret;
}

/* Reset the dispatch table to zeros. Does not unload the backend handle;
 * ownership of the handle belongs to the API layer (unicc_api.c). */
void unicc_vtable_cleanup(void) {
    memset(&unicc, 0, sizeof(unicc_vtable_t));
    g_backend_handle = NULL;
    g_backend_type = UNICC_BACKEND_UNKNOWN;
}
