/* unicc_api.c - unified unicc_* semantic API.
 *
 * This is the only layer UniMemCom and applications call. It owns the lifecycle
 * state machine, maps UniCCL's own datatype/op enums onto the active backend's
 * numeric values, gates every slot for NULL (-> UNICC_ERR_NOT_SUPPORTED) and
 * keeps the raw result of the last failed backend call. */
#include "unicc.h"
#include "unicc_backends.h"
#include "unicc_vtable.h"
#include "unicc_dtmap.h"
#include "unicc_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    UNICC_STATE_UNINIT = 0,
    UNICC_STATE_INIT
} unicc_state_t;

static unicc_state_t g_state = UNICC_STATE_UNINIT;
static unicc_lib_handle_t g_handle = NULL;
static int g_last_backend_result = 0;
static char g_lib_path[256] = "";

/* Translate a backend's native result (0 == success) into a unified code.
 *
 * Vendor results are ints where 0 == success and errors are POSITIVE
 * (ncclSystemError=1, ...). A negative code from a binder adapter is a
 * UniCCL-internal error (e.g. UNICC_ERR_INVALID_ARGUMENT for a wrong-length
 * bootstrap id): pass it through unchanged - never stash a UniCCL enum in the
 * raw backend-result channel, so the "raw value of the last failed BACKEND
 * call" contract (errors.h) stays exact (review F2). */
static unicc_result_t map_backend_result(int rc) {
    if (rc == 0) {
        return UNICC_OK;
    }
    if (rc < 0) {
        return (unicc_result_t)rc;
    }
    g_last_backend_result = rc;
    return UNICC_ERR_UNHANDLED_BACKEND;
}

static const char* backend_name_from_type(unicc_backend_type_t type) {
    for (int i = 0; i < UNICC_MAX_BACKENDS; i++) {
        if (unicc_backends[i].type == type) {
            return unicc_backends[i].name;
        }
    }
    return "unknown";
}

int unicc_init(void) {
    if (g_state == UNICC_STATE_INIT) {
        return UNICC_ERR_ALREADY_INITIALIZED;
    }

    const char *lib_path = NULL;
    unicc_result_t rc = unicc_loader_detect_backend(&lib_path);
    if (rc != UNICC_OK) {
        return rc;
    }
    if (!lib_path) {
        return UNICC_ERR_NO_BACKEND;
    }
    /* The requested path must fit so the resolved path (this or a short
     * soname fallback) always does too. */
    if (strlen(lib_path) >= sizeof(g_lib_path)) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }

    unicc_lib_handle_t handle;
    /* load reports the name it ACTUALLY dlopen'd into g_lib_path (F6): the
     * requested path, its soname fallback, or the macOS spelling - never the
     * (possibly unloadable) requested name. */
    rc = unicc_loader_load(lib_path, &handle, g_lib_path, sizeof(g_lib_path));
    if (rc != UNICC_OK) {
        /* Distinguish "no backend configured" from "configured but broken":
         * with neither UNICC_LIBRARY nor UNICC_BACKEND set, detect probes a
         * default library (libnccl.so); a failed probe is NO_BACKEND (absent
         * backend is the expected state on non-GPU hosts), while a failure to
         * load an explicitly selected backend stays BACKEND_LOAD. */
        if (rc == UNICC_ERR_BACKEND_LOAD &&
            !unicc_loader_get_env_backend() &&
            !unicc_loader_get_env_libpath()) {
            return UNICC_ERR_NO_BACKEND;
        }
        return rc;
    }

    rc = unicc_vtable_init(handle);
    if (rc != UNICC_OK) {
        unicc_loader_unload(handle);
        g_lib_path[0] = '\0';
        return rc;
    }

    /* A fresh identity: a stale raw backend result from a previous lifecycle
     * must not survive into this one (F5 - "0 until a backend call fails"). */
    g_last_backend_result = 0;
    g_handle = handle;
    g_state = UNICC_STATE_INIT;
    return UNICC_OK;
}

int unicc_finalize(void) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    unicc_vtable_cleanup();
    unicc_loader_unload(g_handle);
    g_handle = NULL;
    g_lib_path[0] = '\0';
    g_last_backend_result = 0;   /* no backend is live to have failed a call (F5) */
    /* Finalize returns to the uninitialized state so the wrapper library can
     * be re-initiated in the same process (e.g. with a different backend). */
    g_state = UNICC_STATE_UNINIT;
    return UNICC_OK;
}

int unicc_is_initialized(void) {
    return g_state == UNICC_STATE_INIT;
}

const char* unicc_backend_name(void) {
    if (g_state != UNICC_STATE_INIT) {
        return "unknown";
    }
    return backend_name_from_type(unicc_get_backend_type());
}

const char* unicc_get_library_path(void) {
    return g_lib_path;
}

int unicc_get_version(char *buf, size_t len) {
    if (!buf || len == 0) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    snprintf(buf, len, "UniCCL %s", UNICC_VERSION_STRING);
    return UNICC_OK;
}

int unicc_backend_version(int *version) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.get_version) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    if (!version) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(unicc.get_version(version));
}

int unicc_diagnose(void) {
    if (g_state == UNICC_STATE_INIT) {
        unicc_diagnose_backend(g_lib_path);
        return UNICC_OK;
    }
    /* Not initialized: report what the loader would pick. */
    const char *path = NULL;
    if (unicc_loader_detect_backend(&path) == UNICC_OK) {
        unicc_diagnose_backend(path);
        return UNICC_OK;
    }
    unicc_diagnose_backend(NULL);
    return UNICC_OK;
}

int unicc_get_unique_id(unicc_comm_id_t *id) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.get_unique_id) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    if (!id) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(unicc.get_unique_id(id));
}

int unicc_comm_init_rank(unicc_comm_t *comm, int nranks, const unicc_comm_id_t *id, int rank) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.comm_init_rank) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    if (!comm || nranks < 1 || rank < 0 || rank >= nranks || !id || id->len == 0) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(unicc.comm_init_rank(comm, nranks, id, rank));
}

int unicc_comm_destroy(unicc_comm_t comm) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.comm_destroy) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    return map_backend_result(unicc.comm_destroy(comm));
}

int unicc_comm_count(unicc_comm_t comm, int *count) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.comm_count) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    if (!count) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(unicc.comm_count(comm, count));
}

int unicc_comm_user_rank(unicc_comm_t comm, int *rank) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.comm_user_rank) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    if (!rank) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(unicc.comm_user_rank(comm, rank));
}

int unicc_comm_available(void) {
    /* Bootstrap is "usable" only when BOTH halves resolve: getting a fresh id
     * AND initializing a communicator from one (F4). */
    return (g_state == UNICC_STATE_INIT &&
            unicc.comm_init_rank != NULL && unicc.get_unique_id != NULL) ? 1 : 0;
}

int unicc_get_unique_id_available(void) {
    return (g_state == UNICC_STATE_INIT && unicc.get_unique_id != NULL) ? 1 : 0;
}

int unicc_allreduce(const void *sendbuf, void *recvbuf, size_t count,
                  unicc_datatype_t datatype, unicc_reduce_op_t op,
                  unicc_comm_t comm, void *stream) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.allreduce) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    if (!sendbuf || !recvbuf || count == 0) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    /* Header-inlined identity mapping (unicc_dtmap.h): a bounds check + one
     * load per collective, no cross-TU call, no backend branch (F8). */
    int dt = unicc_dtmap_lookup_dt((int)datatype);
    int rp = unicc_dtmap_lookup_op((int)op);
    if (dt < 0 || rp < 0) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(unicc.allreduce(sendbuf, recvbuf, count, dt, rp, comm, stream));
}

int unicc_broadcast(void *buf, size_t count, unicc_datatype_t datatype,
                  int root, unicc_comm_t comm, void *stream) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.broadcast) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    if (!buf || count == 0) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    int dt = unicc_dtmap_lookup_dt((int)datatype);
    if (dt < 0) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    return map_backend_result(unicc.broadcast(buf, count, dt, root, comm, stream));
}

int unicc_group_start(void) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.group_start) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    return map_backend_result(unicc.group_start());
}

int unicc_group_end(void) {
    if (g_state != UNICC_STATE_INIT) {
        return UNICC_ERR_NOT_INITIALIZED;
    }
    if (!unicc.group_end) {
        return UNICC_ERR_NOT_SUPPORTED;
    }
    return map_backend_result(unicc.group_end());
}

int unicc_allreduce_available(void) {
    return (g_state == UNICC_STATE_INIT && unicc.allreduce != NULL) ? 1 : 0;
}

int unicc_broadcast_available(void) {
    return (g_state == UNICC_STATE_INIT && unicc.broadcast != NULL) ? 1 : 0;
}

int unicc_group_start_available(void) {
    return (g_state == UNICC_STATE_INIT && unicc.group_start != NULL) ? 1 : 0;
}

int unicc_group_end_available(void) {
    return (g_state == UNICC_STATE_INIT && unicc.group_end != NULL) ? 1 : 0;
}

int unicc_get_last_error(int *raw_backend_result) {
    if (!raw_backend_result) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    *raw_backend_result = g_last_backend_result;
    return UNICC_OK;
}
