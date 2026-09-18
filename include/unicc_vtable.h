#ifndef UNICC_VTABLE_H
#define UNICC_VTABLE_H

#include <stddef.h>

#include "unicc_version.h"
#include "unicc_platform.h"

/* Opaque communicator handle. NCCL's ncclComm_t is a pointer to an opaque
 * internal struct and RCCL is ABI-compatible; UniCCL stores the backend's
 * handle unchanged in this slot and never inspects it. */
typedef void* unicc_comm_t;

/* Unique communicator id. Layout-identical to ncclUniqueId (128 bytes), so a
 * uid obtained via unicc_get_unique_id can be passed straight to the backend's
 * comm-init call; the by-value convention below matches the native prototype
 * ncclCommInitRank(ncclComm_t*, int, ncclUniqueId, int). */
typedef struct {
    unsigned char data[UNICC_UNIQUE_ID_BYTES];
} unicc_unique_id_t;

/* Backend function-pointer dispatch table (the "vtable"; mirrors UniMPI).
 *
 * The process-wide instance `unicc` is zero-initialized. Each backend binding
 * (backends/nccl.c, backends/rccl.c) fills the slots it can resolve with
 * dlsym and leaves the rest NULL. A NULL slot means "this backend cannot
 * perform the operation": the unicc_* wrappers check the slot and return
 * UNICC_ERR_NOT_SUPPORTED, and the *_available() predicates let callers probe
 * availability ahead of time.
 *
 * Slots keep the backends' native signatures so bindings are straight casts.
 *  - ncclGetVersion(int*), ncclGetUniqueId(ncclUniqueId*)
 *  - ncclAllReduce(..., ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t)
 *  - ncclBroadcast(..., int root, ncclComm_t, cudaStream_t)
 * cudaStream_t is an opaque host pointer, so the stream parameter is `void*`.
 * The collection Op / Datatype parameters are `int` because NCCL and RCCL use
 * numerically identical enum values (0..10 datatypes, 0..3 ops); UniCCL maps its
 * own unicc_datatype_t / unicc_reduce_op_t onto those values in unicc_api.c.
 */
typedef struct {
    /* --- core: required; checked by unicc_vtable_validate_core --- */
    int (*get_version)(int *version);
    int (*comm_init_rank)(unicc_comm_t *comm, int nranks,
                          unicc_unique_id_t uid, int rank);
    int (*allreduce)(const void *sendbuf, void *recvbuf, size_t count,
                     int datatype, int op, unicc_comm_t comm, void *stream);
    int (*broadcast)(void *buf, size_t count, int datatype, int root,
                     unicc_comm_t comm, void *stream);

    /* --- optional: may be NULL; *_available() gates callers --- */
    int (*get_unique_id)(unicc_unique_id_t *uid);
    int (*comm_destroy)(unicc_comm_t comm);
    int (*comm_count)(unicc_comm_t comm, int *count);
    int (*comm_user_rank)(unicc_comm_t comm, int *rank);
    int (*group_start)(void);
    int (*group_end)(void);
} unicc_vtable_t;

/* Process-wide dispatch table. Zero-initialized at load; populated by
 * unicc_vtable_init and cleared by unicc_vtable_cleanup. */
extern unicc_vtable_t unicc;

/* Backend vtable lifecycle (used by unicc_api.c; exposed for tests).
 * - validate_core: refuse a backend that lacks the M2 core symbols.
 * - init: validate, identify, dispatch into the matching binding.
 * - cleanup: restore the table to all-zeros (does not unload the handle). */
int unicc_vtable_validate_core(unicc_lib_handle_t handle);
int unicc_vtable_init(unicc_lib_handle_t handle);
void unicc_vtable_cleanup(void);

#endif /* UNICC_VTABLE_H */
