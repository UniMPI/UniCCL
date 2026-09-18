/* unicc_backends/nccl.c - NVIDIA NCCL binding.
 *
 * Each slot the loaded library exports is filled by dlsym; a missing symbol
 * leaves the slot NULL and the *_available() gate reports it. Native
 * signatures only - UniCCL never links against or includes nccl.h.
 *
 * The two cold-path id slots use adapters: the vtable carries
 * unicc_comm_id_t (length + 4 KiB buffer) while ncclUniqueId is a fixed
 * 128-byte struct (by-value to ncclCommInitRank, by-pointer to
 * ncclGetUniqueId). Both are a plain memcpy, run once per bootstrap, never on
 * the collective hot path. All other slots are straight casts. */
#include "unicc_vtable.h"
#include "unicc_platform.h"
#include "unicc_errors.h"
#include <string.h>

/* Opaque NCCL unique id: NCCL_UNIQUE_ID_BYTES bytes, matching ncclUniqueId. */
typedef struct { char internal[UNICC_UNIQUE_ID_BYTES]; } native_uid_t;

static int (*s_nccl_get_unique_id)(native_uid_t *uid);
static int (*s_nccl_comm_init_rank)(unicc_comm_t *comm, int nranks,
                                    native_uid_t uid, int rank);

static int wrap_get_unique_id(unicc_comm_id_t *id) {
    native_uid_t uid;
    int rc = s_nccl_get_unique_id(&uid);
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
    return s_nccl_comm_init_rank(comm, nranks, uid, rank);
}

int unicc_vtable_init_nccl(unicc_lib_handle_t handle) {
    /* core */
    unicc.get_version   = (int(*)(int*))unicc_platform_dlsym(handle, "ncclGetVersion");
    s_nccl_comm_init_rank = (int(*)(unicc_comm_t*, int, native_uid_t, int))
        unicc_platform_dlsym(handle, "ncclCommInitRank");
    unicc.comm_init_rank = wrap_comm_init_rank;
    unicc.allreduce     = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "ncclAllReduce");
    unicc.broadcast     = (int(*)(void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "ncclBroadcast");

    /* optional (may remain NULL -> gate via *_available) */
    s_nccl_get_unique_id = (int(*)(native_uid_t*))
        unicc_platform_dlsym(handle, "ncclGetUniqueId");
    unicc.get_unique_id = wrap_get_unique_id;
    unicc.comm_destroy  = (int(*)(unicc_comm_t))unicc_platform_dlsym(handle, "ncclCommDestroy");
    unicc.comm_count    = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "ncclCommCount");
    unicc.comm_user_rank = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "ncclCommUserRank");
    unicc.group_start   = (int(*)(void))unicc_platform_dlsym(handle, "ncclGroupStart");
    unicc.group_end     = (int(*)(void))unicc_platform_dlsym(handle, "ncclGroupEnd");

    return UNICC_OK;
}
