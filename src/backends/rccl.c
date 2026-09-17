/* xcc_backends/rccl.c - AMD RCCL binding (defensive rccl* family).
 *
 * Binds the rccl* symbol family. NOTE (docs/official/): modern AMD RCCL's
 * public API is nccl*-named and exposes no public rccl* symbols, so on a real
 * RCCL the loader resolves it via the nccl binding (nccl.c), and this file is
 * a defensive path reached only if a library actually exports rccl*
 * (historical / third-party shape). When reached, we bind rccl* throughout
 * rather than mixing with the nccl* family. Like nccl.c, a missing symbol
 * leaves the slot NULL and the *_available() gate reports it. */
#include "xcc_vtable.h"
#include "xcc_platform.h"
#include "xcc_errors.h"

int xcc_vtable_init_rccl(xcc_lib_handle_t handle) {
    /* core */
    xcc.get_version   = (int(*)(int*))xcc_platform_dlsym(handle, "rcclGetVersion");
    xcc.comm_init_rank = (int(*)(xcc_comm_t*, int, xcc_unique_id_t, int))
        xcc_platform_dlsym(handle, "rcclCommInitRank");
    xcc.allreduce     = (int(*)(const void*, void*, size_t, int, int, xcc_comm_t, void*))
        xcc_platform_dlsym(handle, "rcclAllReduce");
    xcc.broadcast     = (int(*)(void*, size_t, int, int, xcc_comm_t, void*))
        xcc_platform_dlsym(handle, "rcclBroadcast");

    /* optional (may remain NULL -> gate via *_available) */
    xcc.get_unique_id = (int(*)(xcc_unique_id_t*))xcc_platform_dlsym(handle, "rcclGetUniqueId");
    xcc.comm_destroy  = (int(*)(xcc_comm_t))xcc_platform_dlsym(handle, "rcclCommDestroy");
    xcc.comm_count    = (int(*)(xcc_comm_t, int*))xcc_platform_dlsym(handle, "rcclCommCount");
    xcc.comm_user_rank = (int(*)(xcc_comm_t, int*))xcc_platform_dlsym(handle, "rcclCommUserRank");
    xcc.group_start   = (int(*)(void))xcc_platform_dlsym(handle, "rcclGroupStart");
    xcc.group_end     = (int(*)(void))xcc_platform_dlsym(handle, "rcclGroupEnd");

    return XCC_OK;
}
