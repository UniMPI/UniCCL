/* unicc_backends/rccl.c - AMD RCCL binding (defensive rccl* family).
 *
 * Binds the rccl* symbol family. NOTE (docs/official/): modern AMD RCCL's
 * public API is nccl*-named and exposes no public rccl* symbols, so on a real
 * RCCL the loader resolves it via the nccl binding (nccl.c), and this file is
 * a defensive path reached only if a library actually exports rccl*
 * (historical / third-party shape). When reached, we bind rccl* throughout
 * rather than mixing with the nccl* family. Like nccl.c, a missing symbol
 * leaves the slot NULL and the *_available() gate reports it. */
#include "unicc_vtable.h"
#include "unicc_platform.h"
#include "unicc_errors.h"

int unicc_vtable_init_rccl(unicc_lib_handle_t handle) {
    /* core */
    unicc.get_version   = (int(*)(int*))unicc_platform_dlsym(handle, "rcclGetVersion");
    unicc.comm_init_rank = (int(*)(unicc_comm_t*, int, unicc_unique_id_t, int))
        unicc_platform_dlsym(handle, "rcclCommInitRank");
    unicc.allreduce     = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "rcclAllReduce");
    unicc.broadcast     = (int(*)(void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "rcclBroadcast");

    /* optional (may remain NULL -> gate via *_available) */
    unicc.get_unique_id = (int(*)(unicc_unique_id_t*))unicc_platform_dlsym(handle, "rcclGetUniqueId");
    unicc.comm_destroy  = (int(*)(unicc_comm_t))unicc_platform_dlsym(handle, "rcclCommDestroy");
    unicc.comm_count    = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "rcclCommCount");
    unicc.comm_user_rank = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "rcclCommUserRank");
    unicc.group_start   = (int(*)(void))unicc_platform_dlsym(handle, "rcclGroupStart");
    unicc.group_end     = (int(*)(void))unicc_platform_dlsym(handle, "rcclGroupEnd");

    return UNICC_OK;
}
