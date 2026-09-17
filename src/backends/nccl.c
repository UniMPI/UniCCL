/* xcc_backends/nccl.c - NVIDIA NCCL binding.
 *
 * Each slot the loaded library exports is filled directly by dlsym; a missing
 * symbol leaves the slot NULL and the *_available() gate reports it. Native
 * signatures only - XCCL never links against or includes nccl.h. */
#include "xcc_vtable.h"
#include "xcc_platform.h"
#include "xcc_errors.h"

int xcc_vtable_init_nccl(xcc_lib_handle_t handle) {
    /* core */
    xcc.get_version   = (int(*)(int*))xcc_platform_dlsym(handle, "ncclGetVersion");
    xcc.comm_init_rank = (int(*)(xcc_comm_t*, int, xcc_unique_id_t, int))
        xcc_platform_dlsym(handle, "ncclCommInitRank");
    xcc.allreduce     = (int(*)(const void*, void*, size_t, int, int, xcc_comm_t, void*))
        xcc_platform_dlsym(handle, "ncclAllReduce");
    xcc.broadcast     = (int(*)(void*, size_t, int, int, xcc_comm_t, void*))
        xcc_platform_dlsym(handle, "ncclBroadcast");

    /* optional (may remain NULL -> gate via *_available) */
    xcc.get_unique_id = (int(*)(xcc_unique_id_t*))xcc_platform_dlsym(handle, "ncclGetUniqueId");
    xcc.comm_destroy  = (int(*)(xcc_comm_t))xcc_platform_dlsym(handle, "ncclCommDestroy");
    xcc.comm_count    = (int(*)(xcc_comm_t, int*))xcc_platform_dlsym(handle, "ncclCommCount");
    xcc.comm_user_rank = (int(*)(xcc_comm_t, int*))xcc_platform_dlsym(handle, "ncclCommUserRank");
    xcc.group_start   = (int(*)(void))xcc_platform_dlsym(handle, "ncclGroupStart");
    xcc.group_end     = (int(*)(void))xcc_platform_dlsym(handle, "ncclGroupEnd");

    return XCC_OK;
}
