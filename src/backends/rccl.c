/* unicc_backends/rccl.c - AMD RCCL binding (defensive rccl* family).
 *
 * Binds the rccl* symbol family via the generic binder (unicc_bind.c). NOTE
 * (docs/official/): modern AMD RCCL's public API is nccl*-named and exposes no
 * public rccl* symbols, so on a real RCCL the loader resolves it via the nccl
 * binding (nccl.c); this file is the defensive path reached only if a library
 * actually exports rccl* (historical / third-party shape). Like nccl.c: 128 B
 * id, int nranks, in-place broadcast. */
#include "unicc_bind.h"

int unicc_vtable_init_rccl(unicc_lib_handle_t handle) {
    return unicc_vtable_bind(handle, UNICC_BACKEND_RCCL);
}
