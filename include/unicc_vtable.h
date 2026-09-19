#ifndef UNICC_VTABLE_H
#define UNICC_VTABLE_H

#include <stddef.h>

#include "unicc_version.h"
#include "unicc_platform.h"

/* Opaque communicator handle. NCCL's ncclComm_t is a pointer to an opaque
 * internal struct and RCCL is ABI-compatible; UniCCL stores the backend's
 * handle unchanged in this slot and never inspects it. */
typedef void* unicc_comm_t;

/* Communicator bootstrap id. Vendors disagree on id size, so the id carries its
 * length and the wrapper binds native symbols through small adapter functions
 * in each backend file (never a direct dlsym cast for init/get_unique_id).
 *  - NCCL family (nccl/rccl/DCU/MCCL): 128 bytes, layout = ncclUniqueId;
 *  - Cambricon CNCL: cnclCliqueId is 136 bytes (128B data + uint64 hash);
 *  - Intel oneCCL v2: on ecclUniqueId is 4096 bytes;
 *  - Ascend HCCL: HcclRootInfo is 4108 bytes.
 * The cold path (get_unique_id / comm_init_rank / transport of the id) is the
 * only place that sees this; collectives never do. */
typedef struct {
    size_t len;
    unsigned char data[UNICC_COMM_ID_MAX];
} unicc_comm_id_t;

/* Backend function-pointer dispatch table (the "vtable"; mirrors UniMPI).
 *
 * The process-wide instance `unicc` is zero-initialized. Each backend binding
 * (backends/nccl.c, backends/rccl.c) fills the slots it can resolve with
 * dlsym and leaves the rest NULL. A NULL slot means "this backend cannot
 * perform the operation": the unicc_* wrappers check the slot and return
 * UNICC_ERR_NOT_SUPPORTED, and the *_available() predicates let callers probe
 * availability ahead of time.
 *
 * Hot-path slots (allreduce/broadcast/group_*) and the simple accessors keep the
 * backends' native signatures so binding is a straight cast. The cold-path id
 * slots (get_unique_id / comm_init_rank) take const/pointer unicc_comm_id_t and
 * are backed by small per-backend adapters, because the native id is by-value
 * and vendor-sized (128..4108 bytes) — never a direct dlsym cast.
 *  - ncclGetVersion(int*), ncclGetUniqueId(ncclUniqueId*)
 *  - ncclAllReduce(..., ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t)
 *  - ncclBroadcast(..., int root, ncclComm_t, cudaStream_t)
 * cudaStream_t is an opaque host pointer, so the stream parameter is `void*`.
 * Datatype / Op parameters are `int`; the wrapper maps its own enums onto the
 * active backend's numeric values through the identity accessors in
 * unicc_dtmap.h (all current backends share the NCCL-family numbering — see
 * docs/BACKENDS.md).
 */
typedef struct {
    /* --- core: required; checked at bind time (unicc_vtable_bind) --- */
    int (*get_version)(int *version);
    int (*comm_init_rank)(unicc_comm_t *comm, int nranks,
                          const unicc_comm_id_t *id, int rank);
    int (*allreduce)(const void *sendbuf, void *recvbuf, size_t count,
                     int datatype, int op, unicc_comm_t comm, void *stream);
    int (*broadcast)(void *buf, size_t count, int datatype, int root,
                     unicc_comm_t comm, void *stream);

    /* --- optional: may be NULL; *_available() gates callers --- */
    int (*get_unique_id)(unicc_comm_id_t *id);
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
 * - init: identify the family, then dispatch into its binding - which validates
 *   and requires that family's own core symbols (unicc_vtable_bind).
 * - cleanup: restore the table to all-zeros (does not unload the handle). */
int unicc_vtable_init(unicc_lib_handle_t handle);
void unicc_vtable_cleanup(void);

#endif /* UNICC_VTABLE_H */
