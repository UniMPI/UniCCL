/* unicc_backends/oneccl.c - Intel oneCCL v2 C API binding.
 *
 * oneCCL v2 (the NCCL-aligned C API, default branch since release 2022.1;
 * libccl.so.2) exports the oneccl* symbol family, datatype/op numerically
 * identical to NCCL, and a onecclUniqueId of 4096 B. The classic C++-API line
 * (libccl.so.1) exports no C symbols and is deliberately not bound here.
 * Binds via the generic binder (unicc_bind.c) with the oneccl* descriptor:
 * 4096 B id, size_t nranks, double-buffered broadcast. Evidence:
 * uxlfoundation/oneCCL master-v2 sources + docs
 * (docs/official/ccL-ecosystem-survey-2026-09-17.md). */
#include "unicc_bind.h"

int unicc_vtable_init_oneccl(unicc_lib_handle_t handle) {
    return unicc_vtable_bind(handle, UNICC_BACKEND_ONECCL);
}
