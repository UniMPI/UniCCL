/* unicc_backends/nccl.c - NVIDIA NCCL binding.
 *
 * The generic binder (unicc_bind.c) does all the work; the per-family facts
 * (symbol prefix, bootstrap-id size, signature quirks) live in the
 * unicc_backends[] row selected by type (see docs/BACKENDS.md). NCCL: nccl*
 * prefix, 128 B id, int nranks, in-place broadcast. */
#include "unicc_bind.h"

int unicc_vtable_init_nccl(unicc_lib_handle_t handle) {
    return unicc_vtable_bind(handle, UNICC_BACKEND_NCCL);
}
