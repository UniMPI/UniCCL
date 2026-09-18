#ifndef UNICC_H
#define UNICC_H

/* UniCCL - a unified collective-communication layer over NCCL / RCCL (and
 * future GPU collective libraries).
 *
 * Upper layers (UMC, applications) depend ONLY on this header. Nothing here
 * refers to nccl.h / rccl.h or to CUDA / ROCm, so consumers build without any
 * GPU vendor headers and pick a backend at runtime through the environment:
 *   UNICC_LIBRARY=<exact path>  UNICC_BACKEND=nccl|rccl
 * (see docs/BACKENDS.md for the full selection rules). */

#ifdef __cplusplus
extern "C" {
#endif

#include "unicc_version.h"
#include "unicc_errors.h"
#include "unicc_vtable.h"

/* --- Data types ------------------------------------------------ */
/* UniCCL's own enumerations; mapped onto each backend's native datatype inside
 * the implementation so consumers never need the vendor headers. Values of
 * the NCCL/RCCL enums are numerically stable but UniCCL defines its own names
 * (see docs/BACKENDS.md for the mapping table). */
typedef enum {
    UNICC_I8   = 0,
    UNICC_U8   = 1,
    UNICC_I32  = 2,
    UNICC_U32  = 3,
    UNICC_I64  = 4,
    UNICC_U64  = 5,
    UNICC_F16  = 6,
    UNICC_F32  = 7,
    UNICC_F64  = 8,
    UNICC_BF16 = 9
} unicc_datatype_t;

/* Reduction operations. Semantics are identical across NCCL and RCCL. */
typedef enum {
    UNICC_SUM  = 0,
    UNICC_PROD = 1,
    UNICC_MAX  = 2,
    UNICC_MIN  = 3
} unicc_reduce_op_t;

/* --- Lifecycle ------------------------------------------------- */
/* Load + identify + bind the backend selected by the environment and prime
 * the dispatch table. Returns UNICC_OK, or a unified error code. */
int unicc_init(void);
int unicc_finalize(void);
int unicc_is_initialized(void);

/* --- Version / diagnostics -------------------------------------- */
/* Name of the active backend: "nccl", "rccl", "oneccl", "eccl", or "unknown". */
const char* unicc_backend_name(void);
/* Path of the library that was actually loaded ("" if none). */
const char* unicc_get_library_path(void);
/* UniCCL wrapper's own version string. */
int unicc_get_version(char *buf, size_t len);
/* Backend's reported version integer (nccl/rccl GetVersion). */
int unicc_backend_version(int *version);
/* Print the backend table, then diagnose the selected library's symbol
 * coverage on stderr. */
int unicc_print_backend_info(void);
int unicc_diagnose(void);

/* --- Communicators ---------------------------------------------- */
/* Bootstrap ids carry an explicit byte length because vendors disagree on size
 * (NCCL family 128B, Intel oneCCL 4096B, CNCL 136B, HCCL 4108B). Get the id
 * on rank 0, transport it verbatim (data + len), and hand it to every rank's
 * comm_init_rank. */
int unicc_get_unique_id(unicc_comm_id_t *id);
int unicc_comm_init_rank(unicc_comm_t *comm, int nranks, const unicc_comm_id_t *id, int rank);
int unicc_comm_destroy(unicc_comm_t comm);
int unicc_comm_count(unicc_comm_t comm, int *count);
int unicc_comm_user_rank(unicc_comm_t comm, int *rank);
/* 1 if communicator bootstrap is usable with the active backend. */
int unicc_comm_available(void);

/* --- Collectives ------------------------------------------------ */
/* stream may be NULL (default stream) or an opaque stream handle. */
int unicc_allreduce(const void *sendbuf, void *recvbuf, size_t count,
                  unicc_datatype_t datatype, unicc_reduce_op_t op,
                  unicc_comm_t comm, void *stream);
int unicc_broadcast(void *buf, size_t count, unicc_datatype_t datatype,
                  int root, unicc_comm_t comm, void *stream);
int unicc_group_start(void);
int unicc_group_end(void);

/* --- Availability gates ----------------------------------------- */
/* 1 iff the active backend exports the symbol backing this operation. */
int unicc_allreduce_available(void);
int unicc_broadcast_available(void);
int unicc_group_start_available(void);
int unicc_group_end_available(void);

/* --- Error passthrough ------------------------------------------ */
/* Raw result code of the last failed backend call (0 until a backend call
 * fails). Always returns UNICC_OK unless ptr is NULL. */
int unicc_get_last_error(int *raw_backend_result);

#ifdef __cplusplus
}
#endif

#endif /* UNICC_H */
