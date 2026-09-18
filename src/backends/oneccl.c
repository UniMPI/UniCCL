/* unicc_backends/oneccl.c - Intel oneCCL v2 C API binding.
 *
 * oneCCL v2 (the NCCL-aligned C API, default branch since release 2022.1;
 * libccl.so.2) exports the oneccl* symbol family, datatype/op numerically
 * identical to NCCL (so the NCCL-family tables apply unchanged), onecclCommInitRank
 * matching ncclCommInitRank, and a onecclUniqueId of 4096 bytes. The classic
 * C++-API line (libccl.so.1) exports no C symbols and is deliberately not
 * bound here. Evidence: uxlfoundation/oneCCL master-v2 sources + docs
 * (docs/official/ccL-ecosystem-survey-2026-09-17.md).
 *
 * Adapters needed: onecclUniqueId is 4096 bytes (vtable carries a length +
 * buffer id, cold path); onecclBroadcast takes separate send/recv buffers, the
 * in-place form passes the same buffer for both. Everything else is a straight
 * cast. */
#include "unicc_vtable.h"
#include "unicc_platform.h"
#include "unicc_errors.h"
#include <string.h>

#define ONECCL_UNIQUE_ID_BYTES 4096

typedef struct { char internal[ONECCL_UNIQUE_ID_BYTES]; } native_uid_t;

static int (*s_get_unique_id)(native_uid_t *uid);
static int (*s_comm_init_rank)(unicc_comm_t *comm, size_t nranks,
                               native_uid_t uid, int rank);
static int (*s_broadcast)(const void *sendbuff, void *recvbuff, size_t count,
                          int datatype, int root, unicc_comm_t comm, void *stream);

static int wrap_get_unique_id(unicc_comm_id_t *id) {
    native_uid_t uid;
    int rc = s_get_unique_id(&uid);
    if (rc == 0) {
        id->len = ONECCL_UNIQUE_ID_BYTES;
        memcpy(id->data, uid.internal, ONECCL_UNIQUE_ID_BYTES);
    }
    return rc;
}

static int wrap_comm_init_rank(unicc_comm_t *comm, int nranks,
                               const unicc_comm_id_t *id, int rank) {
    if (id->len != ONECCL_UNIQUE_ID_BYTES) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    native_uid_t uid;
    memcpy(uid.internal, id->data, ONECCL_UNIQUE_ID_BYTES);
    return s_comm_init_rank(comm, (size_t)nranks, uid, rank);
}

/* onecclBroadcast is double-buffered; the vtable slot is in-place, so pass the
 * same buffer for send and recv. No copy is involved. */
static int wrap_broadcast(void *buf, size_t count, int datatype, int root,
                          unicc_comm_t comm, void *stream) {
    return s_broadcast(buf, buf, count, datatype, root, comm, stream);
}

int unicc_vtable_init_oneccl(unicc_lib_handle_t handle) {
    /* core */
    unicc.get_version   = (int(*)(int*))unicc_platform_dlsym(handle, "onecclGetVersion");
    s_comm_init_rank = (int(*)(unicc_comm_t*, size_t, native_uid_t, int))
        unicc_platform_dlsym(handle, "onecclCommInitRank");
    unicc.comm_init_rank = wrap_comm_init_rank;
    unicc.allreduce     = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "onecclAllReduce");
    s_broadcast = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "onecclBroadcast");
    unicc.broadcast = wrap_broadcast;

    /* optional (may remain NULL -> gate via *_available) */
    s_get_unique_id = (int(*)(native_uid_t*))
        unicc_platform_dlsym(handle, "onecclGetUniqueId");
    unicc.get_unique_id = wrap_get_unique_id;
    unicc.comm_destroy  = (int(*)(unicc_comm_t))unicc_platform_dlsym(handle, "onecclCommDestroy");
    unicc.comm_count    = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "onecclCommCount");
    unicc.comm_user_rank = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "onecclCommUserRank");
    unicc.group_start   = (int(*)(void))unicc_platform_dlsym(handle, "onecclGroupStart");
    unicc.group_end     = (int(*)(void))unicc_platform_dlsym(handle, "onecclGroupEnd");

    return UNICC_OK;
}
