/* unicc_backends/eccl.c - Enflame ECCL binding.
 *
 * ECCL (Enflame Collective Communication Library, part of the TopsRider suite)
 * is an NCCL-shaped API: ecclGetVersion / ecclCommInitRank / ecclAllReduce /
 * ecclBroadcast / ecclGroupStart-End, prefix eccl*. Datatype/op constants are
 * NCCL-mirroring by name; numeric values live in the vendor eccl.h and are NOT
 * confirmed on the source (needs nm -D on a real TopsRider host), so the
 * NCCL-family tables are used with that caveat recorded in docs/BACKENDS.md.
 * ecclCommUserRank is not exported by the ecosystem (torch-gcu tracks rank in
 * its process group), so that slot stays NULL -> unicc_comm_user_rank() reports
 * NOT_SUPPORTED. The unique id is NCCL-sized (ECCL_UNIQUE_ID_BYTES unconfirmed;
 * assumed 128 like NCCL, flagged for real-host verification). */
#include "unicc_vtable.h"
#include "unicc_platform.h"
#include "unicc_errors.h"
#include <string.h>

typedef struct { char internal[UNICC_UNIQUE_ID_BYTES]; } native_uid_t;

static int (*s_get_unique_id)(native_uid_t *uid);
static int (*s_comm_init_rank)(unicc_comm_t *comm, size_t numranks,
                               native_uid_t uid, int rank);
static int (*s_broadcast)(const void *sendbuff, void *recvbuff, size_t count,
                          int datatype, int root, unicc_comm_t comm, void *stream);

static int wrap_get_unique_id(unicc_comm_id_t *id) {
    native_uid_t uid;
    int rc = s_get_unique_id(&uid);
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
    return s_comm_init_rank(comm, (size_t)nranks, uid, rank);
}

/* Double-buffered native broadcast, in-place vtable slot: same buffer both ways. */
static int wrap_broadcast(void *buf, size_t count, int datatype, int root,
                          unicc_comm_t comm, void *stream) {
    return s_broadcast(buf, buf, count, datatype, root, comm, stream);
}

int unicc_vtable_init_eccl(unicc_lib_handle_t handle) {
    /* core */
    unicc.get_version   = (int(*)(int*))unicc_platform_dlsym(handle, "ecclGetVersion");
    s_comm_init_rank = (int(*)(unicc_comm_t*, size_t, native_uid_t, int))
        unicc_platform_dlsym(handle, "ecclCommInitRank");
    unicc.comm_init_rank = wrap_comm_init_rank;
    unicc.allreduce     = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "ecclAllReduce");
    s_broadcast = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "ecclBroadcast");
    unicc.broadcast = wrap_broadcast;

    /* optional (may remain NULL -> gate via *_available).
     * ecclCommUserRank is intentionally not bound: no consumer in the
     * ecosystem exports it (see file header). */
    s_get_unique_id = (int(*)(native_uid_t*))
        unicc_platform_dlsym(handle, "ecclGetUniqueId");
    unicc.get_unique_id = wrap_get_unique_id;
    unicc.comm_destroy  = (int(*)(unicc_comm_t))unicc_platform_dlsym(handle, "ecclCommDestroy");
    unicc.comm_count    = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "ecclCommCount");
    /* comm_user_rank stays NULL for ECCL. */
    unicc.group_start   = (int(*)(void))unicc_platform_dlsym(handle, "ecclGroupStart");
    unicc.group_end     = (int(*)(void))unicc_platform_dlsym(handle, "ecclGroupEnd");

    return UNICC_OK;
}
