/* unicc_backends/rccl.c - AMD RCCL binding (defensive rccl* family).
 *
 * Binds the rccl* symbol family. NOTE (docs/official/): modern AMD RCCL's
 * public API is nccl*-named and exposes no public rccl* symbols, so on a real
 * RCCL the loader resolves it via the nccl binding (nccl.c), and this file is
 * a defensive path reached only if a library actually exports rccl*
 * (historical / third-party shape). When reached, we bind rccl* throughout
 * rather than mixing with the nccl* family. Like nccl.c, a missing symbol
 * leaves the slot NULL and the *_available() gate reports it. The cold-path
 * id slots go through the same 128-byte adapters as nccl.c. */
#include "unicc_vtable.h"
#include "unicc_platform.h"
#include "unicc_errors.h"
#include <string.h>

typedef struct { char internal[UNICC_UNIQUE_ID_BYTES]; } native_uid_t;

static int (*s_rccl_get_unique_id)(native_uid_t *uid);
static int (*s_rccl_comm_init_rank)(unicc_comm_t *comm, int nranks,
                                    native_uid_t uid, int rank);

static int wrap_get_unique_id(unicc_comm_id_t *id) {
    native_uid_t uid;
    int rc = s_rccl_get_unique_id(&uid);
    if (rc == 0) {
        id->len = UNICC_UNIQUE_ID_BYTES;
        memcpy(id->data, uid.internal, UNICC_UNIQUE_ID_BYTES);
    }
    return rc;
}

static int wrap_comm_init_rank(unicc_comm_t *comm, int nranks,
                               const unicc_comm_id_t *id, int rank) {
    if (id->len != UNICC_UNIQUE_ID_BYTES) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    native_uid_t uid;
    memcpy(uid.internal, id->data, UNICC_UNIQUE_ID_BYTES);
    return s_rccl_comm_init_rank(comm, nranks, uid, rank);
}

int unicc_vtable_init_rccl(unicc_lib_handle_t handle) {
    /* core */
    unicc.get_version   = (int(*)(int*))unicc_platform_dlsym(handle, "rcclGetVersion");
    s_rccl_comm_init_rank = (int(*)(unicc_comm_t*, int, native_uid_t, int))
        unicc_platform_dlsym(handle, "rcclCommInitRank");
    unicc.comm_init_rank = wrap_comm_init_rank;
    unicc.allreduce     = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "rcclAllReduce");
    unicc.broadcast     = (int(*)(void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "rcclBroadcast");

    /* optional (may remain NULL -> gate via *_available) */
    s_rccl_get_unique_id = (int(*)(native_uid_t*))
        unicc_platform_dlsym(handle, "rcclGetUniqueId");
    unicc.get_unique_id = wrap_get_unique_id;
    unicc.comm_destroy  = (int(*)(unicc_comm_t))unicc_platform_dlsym(handle, "rcclCommDestroy");
    unicc.comm_count    = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "rcclCommCount");
    unicc.comm_user_rank = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "rcclCommUserRank");
    unicc.group_start   = (int(*)(void))unicc_platform_dlsym(handle, "rcclGroupStart");
    unicc.group_end     = (int(*)(void))unicc_platform_dlsym(handle, "rcclGroupEnd");

    return UNICC_OK;
}
