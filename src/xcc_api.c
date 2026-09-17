/* xcc_api.c - unified xcc_* semantic API.
 *
 * This is the only layer UMC and applications call. It owns the lifecycle
 * state machine, maps XCCL's own datatype/op enums onto the active backend's
 * numeric values, gates every slot for NULL (-> XCC_ERR_NOT_SUPPORTED) and
 * keeps the raw result of the last failed backend call. */
#include "xcc.h"
#include "xcc_backends.h"
#include "xcc_vtable.h"
#include "xcc_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    XCC_STATE_UNINIT = 0,
    XCC_STATE_INIT
} xcc_state_t;

static xcc_state_t g_state = XCC_STATE_UNINIT;
static xcc_lib_handle_t g_handle = NULL;
static int g_last_backend_result = 0;
static char g_lib_path[256] = "";

/* Translate a backend's native result (0 == success) into a unified code. */
static xcc_result_t map_backend_result(int rc) {
    if (rc == 0) {
        return XCC_OK;
    }
    g_last_backend_result = rc;
    return XCC_ERR_UNHANDLED_BACKEND;
}

/* Map XCCL's datatype enum onto the backend's numeric datatype. NCCL and RCCL
 * use numerically identical enum values, so a single static mapping suffices;
 * if a future backend diverges, this table moves into per-backend files
 * (docs/BACKENDS.md "enum mapping"). */
static int map_datatype(xcc_datatype_t dt) {
    switch (dt) {
        case XCC_I8:   return 0;   /* ncclInt8 */
        case XCC_U8:   return 2;   /* ncclUint8 */
        case XCC_I32:  return 3;   /* ncclInt32 */
        case XCC_U32:  return 4;   /* ncclUint32 */
        case XCC_I64:  return 5;   /* ncclInt64 */
        case XCC_U64:  return 6;   /* ncclUint64 */
        case XCC_F16:  return 9;   /* ncclHalf */
        case XCC_F32:  return 7;   /* ncclFloat */
        case XCC_F64:  return 8;   /* ncclDouble */
        case XCC_BF16: return 10;  /* ncclBfloat16 */
    }
    return -1;
}

/* Map XCCL's reduce-op enum onto the backend's numeric op. */
static int map_reduce_op(xcc_reduce_op_t op) {
    switch (op) {
        case XCC_SUM:  return 0;   /* ncclSum */
        case XCC_PROD: return 1;   /* ncclProd */
        case XCC_MAX:  return 2;   /* ncclMax */
        case XCC_MIN:  return 3;   /* ncclMin */
    }
    return -1;
}

static const char* backend_name_from_type(xcc_backend_type_t type) {
    for (int i = 0; i < XCC_MAX_BACKENDS; i++) {
        if (xcc_backends[i].type == type) {
            return xcc_backends[i].name;
        }
    }
    return "unknown";
}

int xcc_init(void) {
    if (g_state == XCC_STATE_INIT) {
        return XCC_ERR_ALREADY_INITIALIZED;
    }

    const char *lib_path = NULL;
    xcc_result_t rc = xcc_loader_detect_backend(&lib_path);
    if (rc != XCC_OK) {
        return rc;
    }
    if (!lib_path) {
        return XCC_ERR_NO_BACKEND;
    }
    if (strlen(lib_path) >= sizeof(g_lib_path)) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    strcpy(g_lib_path, lib_path);

    xcc_lib_handle_t handle;
    rc = xcc_loader_load(lib_path, &handle);
    if (rc != XCC_OK) {
        g_lib_path[0] = '\0';
        return rc;
    }

    rc = xcc_vtable_init(handle);
    if (rc != XCC_OK) {
        xcc_loader_unload(handle);
        g_lib_path[0] = '\0';
        return rc;
    }

    g_handle = handle;
    g_state = XCC_STATE_INIT;
    return XCC_OK;
}

int xcc_finalize(void) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    xcc_vtable_cleanup();
    xcc_loader_unload(g_handle);
    g_handle = NULL;
    g_lib_path[0] = '\0';
    /* Finalize returns to the uninitialized state so the wrapper library can
     * be re-initiated in the same process (e.g. with a different backend). */
    g_state = XCC_STATE_UNINIT;
    return XCC_OK;
}

int xcc_is_initialized(void) {
    return g_state == XCC_STATE_INIT;
}

const char* xcc_backend_name(void) {
    if (g_state != XCC_STATE_INIT) {
        return "unknown";
    }
    return backend_name_from_type(xcc_get_backend_type());
}

const char* xcc_get_library_path(void) {
    return g_lib_path;
}

int xcc_get_version(char *buf, size_t len) {
    if (!buf || len == 0) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    snprintf(buf, len, "XCCL %s", XCC_VERSION_STRING);
    return XCC_OK;
}

int xcc_backend_version(int *version) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.get_version) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    if (!version) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(xcc.get_version(version));
}

int xcc_diagnose(void) {
    if (g_state == XCC_STATE_INIT) {
        xcc_diagnose_backend(g_lib_path);
        return XCC_OK;
    }
    /* Not initialized: report what the loader would pick. */
    const char *path = NULL;
    if (xcc_loader_detect_backend(&path) == XCC_OK) {
        xcc_diagnose_backend(path);
        return XCC_OK;
    }
    xcc_diagnose_backend(NULL);
    return XCC_OK;
}

int xcc_get_unique_id(xcc_unique_id_t *uid) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.get_unique_id) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    if (!uid) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(xcc.get_unique_id(uid));
}

int xcc_comm_init_rank(xcc_comm_t *comm, int nranks, xcc_unique_id_t uid, int rank) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.comm_init_rank) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    if (!comm || nranks < 1 || rank < 0 || rank >= nranks) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(xcc.comm_init_rank(comm, nranks, uid, rank));
}

int xcc_comm_destroy(xcc_comm_t comm) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.comm_destroy) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    return map_backend_result(xcc.comm_destroy(comm));
}

int xcc_comm_count(xcc_comm_t comm, int *count) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.comm_count) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    if (!count) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(xcc.comm_count(comm, count));
}

int xcc_comm_user_rank(xcc_comm_t comm, int *rank) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.comm_user_rank) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    if (!rank) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(xcc.comm_user_rank(comm, rank));
}

int xcc_comm_available(void) {
    return (g_state == XCC_STATE_INIT && xcc.comm_init_rank != NULL) ? 1 : 0;
}

int xcc_allreduce(const void *sendbuf, void *recvbuf, size_t count,
                  xcc_datatype_t datatype, xcc_reduce_op_t op,
                  xcc_comm_t comm, void *stream) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.allreduce) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    if (!sendbuf || !recvbuf || count == 0) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    int dt = map_datatype(datatype);
    int rp = map_reduce_op(op);
    if (dt < 0 || rp < 0) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(xcc.allreduce(sendbuf, recvbuf, count, dt, rp, comm, stream));
}

int xcc_broadcast(void *buf, size_t count, xcc_datatype_t datatype,
                  int root, xcc_comm_t comm, void *stream) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.broadcast) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    if (!buf || count == 0) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    int dt = map_datatype(datatype);
    if (dt < 0) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(xcc.broadcast(buf, count, dt, root, comm, stream));
}

int xcc_group_start(void) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.group_start) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    return map_backend_result(xcc.group_start());
}

int xcc_group_end(void) {
    if (g_state != XCC_STATE_INIT) {
        return XCC_ERR_NOT_INITIALIZED;
    }
    if (!xcc.group_end) {
        return XCC_ERR_NOT_SUPPORTED;
    }
    return map_backend_result(xcc.group_end());
}

int xcc_allreduce_available(void) {
    return (g_state == XCC_STATE_INIT && xcc.allreduce != NULL) ? 1 : 0;
}

int xcc_broadcast_available(void) {
    return (g_state == XCC_STATE_INIT && xcc.broadcast != NULL) ? 1 : 0;
}

int xcc_group_start_available(void) {
    return (g_state == XCC_STATE_INIT && xcc.group_start != NULL) ? 1 : 0;
}

int xcc_group_end_available(void) {
    return (g_state == XCC_STATE_INIT && xcc.group_end != NULL) ? 1 : 0;
}

int xcc_get_last_error(int *raw_backend_result) {
    if (!raw_backend_result) {
        return XCC_ERR_INVALID_ARGUMENT;
    }
    *raw_backend_result = g_last_backend_result;
    return XCC_OK;
}
