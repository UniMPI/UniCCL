/* unicc_backends/nccl.c - NVIDIA NCCL binding.
 *
 * Each slot the loaded library exports is filled directly by dlsym; a missing
 * symbol leaves the slot NULL and the *_available() gate reports it. Native
 * signatures only - UniCCL never links against or includes nccl.h. */
#include "unicc_vtable.h"
#include "unicc_platform.h"
#include "unicc_errors.h"

int unicc_vtable_init_nccl(unicc_lib_handle_t handle) {
    /* core */
    unicc.get_version   = (int(*)(int*))unicc_platform_dlsym(handle, "ncclGetVersion");
    unicc.comm_init_rank = (int(*)(unicc_comm_t*, int, unicc_unique_id_t, int))
        unicc_platform_dlsym(handle, "ncclCommInitRank");
    unicc.allreduce     = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "ncclAllReduce");
    unicc.broadcast     = (int(*)(void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, "ncclBroadcast");

    /* optional (may remain NULL -> gate via *_available) */
    unicc.get_unique_id = (int(*)(unicc_unique_id_t*))unicc_platform_dlsym(handle, "ncclGetUniqueId");
    unicc.comm_destroy  = (int(*)(unicc_comm_t))unicc_platform_dlsym(handle, "ncclCommDestroy");
    unicc.comm_count    = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "ncclCommCount");
    unicc.comm_user_rank = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, "ncclCommUserRank");
    unicc.group_start   = (int(*)(void))unicc_platform_dlsym(handle, "ncclGroupStart");
    unicc.group_end     = (int(*)(void))unicc_platform_dlsym(handle, "ncclGroupEnd");

    return UNICC_OK;
}
