#ifndef XCC_H
#define XCC_H

/* XCCL - a unified collective-communication layer over NCCL / RCCL (and
 * future GPU collective libraries).
 *
 * Upper layers (UMC, applications) depend ONLY on this header. Nothing here
 * refers to nccl.h / rccl.h or to CUDA / ROCm, so consumers build without any
 * GPU vendor headers and pick a backend at runtime through the environment:
 *   XCCL_LIBRARY=<exact path>  XCCL_BACKEND=nccl|rccl
 * (see docs/BACKENDS.md for the full selection rules). */

#ifdef __cplusplus
extern "C" {
#endif

#include "xcc_version.h"
#include "xcc_errors.h"
#include "xcc_vtable.h"

/* --- Data types ------------------------------------------------ */
/* XCCL's own enumerations; mapped onto each backend's native datatype inside
 * the implementation so consumers never need the vendor headers. Values of
 * the NCCL/RCCL enums are numerically stable but XCCL defines its own names
 * (see docs/BACKENDS.md for the mapping table). */
typedef enum {
    XCC_I8   = 0,
    XCC_U8   = 1,
    XCC_I32  = 2,
    XCC_U32  = 3,
    XCC_I64  = 4,
    XCC_U64  = 5,
    XCC_F16  = 6,
    XCC_F32  = 7,
    XCC_F64  = 8,
    XCC_BF16 = 9
} xcc_datatype_t;

/* Reduction operations. Semantics are identical across NCCL and RCCL. */
typedef enum {
    XCC_SUM  = 0,
    XCC_PROD = 1,
    XCC_MAX  = 2,
    XCC_MIN  = 3
} xcc_reduce_op_t;

/* --- Lifecycle ------------------------------------------------- */
/* Load + identify + bind the backend selected by the environment and prime
 * the dispatch table. Returns XCC_OK, or a unified error code. */
int xcc_init(void);
int xcc_finalize(void);
int xcc_is_initialized(void);

/* --- Version / diagnostics -------------------------------------- */
/* Name of the active backend: "nccl", "rccl", or "unknown". */
const char* xcc_backend_name(void);
/* Path of the library that was actually loaded ("" if none). */
const char* xcc_get_library_path(void);
/* XCCL wrapper's own version string. */
int xcc_get_version(char *buf, size_t len);
/* Backend's reported version integer (nccl/rccl GetVersion). */
int xcc_backend_version(int *version);
/* Print the backend table, then diagnose the selected library's symbol
 * coverage on stderr. */
int xcc_print_backend_info(void);
int xcc_diagnose(void);

/* --- Communicators ---------------------------------------------- */
int xcc_get_unique_id(xcc_unique_id_t *uid);
int xcc_comm_init_rank(xcc_comm_t *comm, int nranks, xcc_unique_id_t uid, int rank);
int xcc_comm_destroy(xcc_comm_t comm);
int xcc_comm_count(xcc_comm_t comm, int *count);
int xcc_comm_user_rank(xcc_comm_t comm, int *rank);
/* 1 if communicator bootstrap is usable with the active backend. */
int xcc_comm_available(void);

/* --- Collectives ------------------------------------------------ */
/* stream may be NULL (default stream) or an opaque stream handle. */
int xcc_allreduce(const void *sendbuf, void *recvbuf, size_t count,
                  xcc_datatype_t datatype, xcc_reduce_op_t op,
                  xcc_comm_t comm, void *stream);
int xcc_broadcast(void *buf, size_t count, xcc_datatype_t datatype,
                  int root, xcc_comm_t comm, void *stream);
int xcc_group_start(void);
int xcc_group_end(void);

/* --- Availability gates ----------------------------------------- */
/* 1 iff the active backend exports the symbol backing this operation. */
int xcc_allreduce_available(void);
int xcc_broadcast_available(void);
int xcc_group_start_available(void);
int xcc_group_end_available(void);

/* --- Error passthrough ------------------------------------------ */
/* Raw result code of the last failed backend call (0 until a backend call
 * fails). Always returns XCC_OK unless ptr is NULL. */
int xcc_get_last_error(int *raw_backend_result);

#ifdef __cplusplus
}
#endif

#endif /* XCC_H */
