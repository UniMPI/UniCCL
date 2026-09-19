/* unicc_backends/eccl.c - Enflame ECCL binding.
 *
 * ECCL (Enflame Collective Communication Library, part of the TopsRider suite)
 * is an NCCL-shaped API: ecclGetVersion / ecclCommInitRank / ecclAllReduce /
 * ecclBroadcast / ecclGroupStart-End, prefix eccl*. Datatype/op constants are
 * NCCL-mirroring by name; numeric values live in the vendor eccl.h and are NOT
 * confirmed on the source (needs nm -D on a real TopsRider host), so the
 * NCCL-family numbering is used with that caveat recorded in docs/BACKENDS.md.
 * ecclCommUserRank is not exported by the ecosystem (torch-gcu tracks rank in
 * its process group), so the descriptor leaves that slot NULL ->
 * unicc_comm_user_rank() reports NOT_SUPPORTED. The unique id is NCCL-sized
 * (ECCL_UNIQUE_ID_BYTES unconfirmed; assumed 128 like NCCL, flagged for
 * real-host verification). Binds via the generic binder (unicc_bind.c):
 * 128 B id, size_t nranks, double-buffered broadcast. */
#include "unicc_bind.h"

int unicc_vtable_init_eccl(unicc_lib_handle_t handle) {
    return unicc_vtable_bind(handle, UNICC_BACKEND_ECCL);
}
