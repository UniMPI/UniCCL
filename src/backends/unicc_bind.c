/* unicc_bind.c - the single generic backend binder shared by all families.
 *
 * Replaces four ~60-line copy-pasted binding files (nccl.c / rccl.c / oneccl.c
 * / eccl.c) with one binder plus a 3-line init per family. Every per-family
 * fact comes from the unicc_backends[] row selected by type, so a new family
 * is one table row (plus, for a brand-new bootstrap-id size, one adapter).
 *
 * UBSan-clean by construction: the canonical native function pointers below
 * are typed to the exact fixture/vendor signatures (`-fsanitize=function`
 * compares declared and actual types, and the fake fixtures are instrumented
 * in CI). The distinct layouts in the wild are:
 *   - nranks: int (NCCL family) vs size_t (oneCCL / ECCL CommInitRank);
 *   - bootstrap id: 128 B (unicc_native_uid_t) vs 4096 B (oneccl).
 * Hence three CommInitRank adapters and two GetUniqueId adapters; collectives
 * need no adapter at all. */
#include "unicc_bind.h"
#include "unicc_vtable.h"
#include "unicc_native_id.h"
#include "unicc_errors.h"
#include <string.h>
#include <stdio.h>

/* --- canonical native signatures (the real layouts in the ecosystem) ------- */
static int (*s_get_version)(int *version);
static int (*s_allreduce)(const void *sendbuf, void *recvbuf, size_t count,
                          int datatype, int op, unicc_comm_t comm, void *stream);

/* Broadcast is in-place for the NCCL family, double-buffered for oneCCL/ECCL. */
static int (*s_broadcast)(void *buf, size_t count, int datatype, int root,
                          unicc_comm_t comm, void *stream);
static int (*s_broadcast_db)(const void *sendbuff, void *recvbuff, size_t count,
                             int datatype, int root, unicc_comm_t comm, void *stream);

static int (*s_rank_init_n128)(unicc_comm_t *comm, int nranks,
                               unicc_native_uid_t uid, int rank);
static int (*s_rank_init_s128)(unicc_comm_t *comm, size_t nranks,
                               unicc_native_uid_t uid, int rank);
static int (*s_rank_init_s4096)(unicc_comm_t *comm, size_t nranks,
                                unicc_native_uid_oneccl_t uid, int rank);

static int (*s_uid_get_128)(unicc_native_uid_t *uid);
static int (*s_uid_get_4096)(unicc_native_uid_oneccl_t *uid);

static int (*s_comm_destroy)(unicc_comm_t comm);
static int (*s_comm_count)(unicc_comm_t comm, int *count);
static int (*s_comm_user_rank)(unicc_comm_t comm, int *rank);
static int (*s_group_start)(void);
static int (*s_group_end)(void);

static const unicc_backend_info_t* family_of(unicc_backend_type_t type) {
    for (int i = 0; i < UNICC_MAX_BACKENDS; i++) {
        if (unicc_backends[i].type == type) {
            return &unicc_backends[i];
        }
    }
    return NULL;
}

/* --- cold-path id adapters -------------------------------------------------- */

static int wrap_get_unique_id_128(unicc_comm_id_t *id) {
    unicc_native_uid_t uid;
    int rc = s_uid_get_128(&uid);
    if (rc == 0) {
        id->len = UNICC_UNIQUE_ID_BYTES;
        memcpy(id->data, uid.internal, UNICC_UNIQUE_ID_BYTES);
    }
    return rc;
}

static int wrap_get_unique_id_4096(unicc_comm_id_t *id) {
    unicc_native_uid_oneccl_t uid;
    int rc = s_uid_get_4096(&uid);
    if (rc == 0) {
        id->len = UNICC_ONECCL_UNIQUE_ID_BYTES;
        memcpy(id->data, uid.internal, UNICC_ONECCL_UNIQUE_ID_BYTES);
    }
    return rc;
}

/* The length guard returns a UniCCL-internal code (UNICC_ERR_INVALID_ARGUMENT);
 * map_backend_result passes negative codes through unchanged, so it never lands
 * in the raw backend-result channel (F2) and never misreports as an unhandled
 * backend error. */
static int wrap_comm_init_rank_n128(unicc_comm_t *comm, int nranks,
                                    const unicc_comm_id_t *id, int rank) {
    if (id->len != UNICC_UNIQUE_ID_BYTES) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    unicc_native_uid_t uid;
    memcpy(uid.internal, id->data, UNICC_UNIQUE_ID_BYTES);
    return s_rank_init_n128(comm, nranks, uid, rank);
}

static int wrap_comm_init_rank_s128(unicc_comm_t *comm, int nranks,
                                    const unicc_comm_id_t *id, int rank) {
    if (id->len != UNICC_UNIQUE_ID_BYTES) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    unicc_native_uid_t uid;
    memcpy(uid.internal, id->data, UNICC_UNIQUE_ID_BYTES);
    return s_rank_init_s128(comm, (size_t)nranks, uid, rank);
}

static int wrap_comm_init_rank_s4096(unicc_comm_t *comm, int nranks,
                                     const unicc_comm_id_t *id, int rank) {
    if (id->len != UNICC_ONECCL_UNIQUE_ID_BYTES) {
        return UNICC_ERR_INVALID_ARGUMENT;
    }
    unicc_native_uid_oneccl_t uid;
    memcpy(uid.internal, id->data, UNICC_ONECCL_UNIQUE_ID_BYTES);
    return s_rank_init_s4096(comm, (size_t)nranks, uid, rank);
}

/* Double-buffered native broadcast; the vtable slot is in-place, so pass the
 * same buffer for send and recv. No copy is involved. */
static int wrap_broadcast_db(void *buf, size_t count, int datatype, int root,
                             unicc_comm_t comm, void *stream) {
    return s_broadcast_db(buf, buf, count, datatype, root, comm, stream);
}

/* --- binder ---------------------------------------------------------------- */

int unicc_vtable_bind(unicc_lib_handle_t handle, unicc_backend_type_t type) {
    const unicc_backend_info_t *d = family_of(type);
    if (!d) {
        fprintf(stderr, "[UniCCL:ERROR] No binding descriptor for backend type %d\n",
                (int)type);
        return UNICC_ERR_BACKEND_INIT_FAILED;
    }

    char sym[96];

    /* Core: every required symbol must resolve for THIS family (the identified
     * one - never a mixed-family OR-set). Validate first, install after. */
    snprintf(sym, sizeof sym, "%sGetVersion", d->name);
    s_get_version = (int(*)(int*))unicc_platform_dlsym(handle, sym);

    snprintf(sym, sizeof sym, "%sAllReduce", d->name);
    s_allreduce = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
        unicc_platform_dlsym(handle, sym);

    if (d->broadcast_double_buffered) {
        snprintf(sym, sizeof sym, "%sBroadcast", d->name);
        s_broadcast_db = (int(*)(const void*, void*, size_t, int, int, unicc_comm_t, void*))
            unicc_platform_dlsym(handle, sym);
    } else {
        snprintf(sym, sizeof sym, "%sBroadcast", d->name);
        s_broadcast = (int(*)(void*, size_t, int, int, unicc_comm_t, void*))
            unicc_platform_dlsym(handle, sym);
    }

    snprintf(sym, sizeof sym, "%sCommInitRank", d->name);
    if (d->nranks_is_size_t) {
        if (d->uid_kind == UNICC_UID_ONECCL) {
            s_rank_init_s4096 = (int(*)(unicc_comm_t*, size_t, unicc_native_uid_oneccl_t, int))
                unicc_platform_dlsym(handle, sym);
        } else {
            s_rank_init_s128 = (int(*)(unicc_comm_t*, size_t, unicc_native_uid_t, int))
                unicc_platform_dlsym(handle, sym);
        }
    } else {
        s_rank_init_n128 = (int(*)(unicc_comm_t*, int, unicc_native_uid_t, int))
            unicc_platform_dlsym(handle, sym);
    }

    if (!s_get_version || !s_allreduce ||
        (d->broadcast_double_buffered ? !s_broadcast_db : !s_broadcast) ||
        (d->nranks_is_size_t
             ? (d->uid_kind == UNICC_UID_ONECCL ? !s_rank_init_s4096 : !s_rank_init_s128)
             : !s_rank_init_n128)) {
        fprintf(stderr, "[UniCCL:ERROR] %s backend library does not export the required core symbols\n",
                d->name);
        return UNICC_ERR_SYMBOL_NOT_FOUND;
    }

    unicc.get_version = s_get_version;
    unicc.allreduce   = s_allreduce;
    unicc.broadcast   = d->broadcast_double_buffered ? wrap_broadcast_db : s_broadcast;
    /* The three wrappers already have the vtable slot signature (int nranks);
     * the size_t cast happens inside against the family's native signature. */
    unicc.comm_init_rank = d->nranks_is_size_t
        ? (d->uid_kind == UNICC_UID_ONECCL ? wrap_comm_init_rank_s4096
                                           : wrap_comm_init_rank_s128)
        : wrap_comm_init_rank_n128;

    /* Optional: a wrapper is installed only over a resolved symbol (never a
     * NULL inner call - F1); a missing symbol leaves the slot NULL for the
     * *_available() gates and UNICC_ERR_NOT_SUPPORTED. */
    snprintf(sym, sizeof sym, "%sGetUniqueId", d->name);
    if (d->uid_kind == UNICC_UID_ONECCL) {
        s_uid_get_4096 = (int(*)(unicc_native_uid_oneccl_t*))unicc_platform_dlsym(handle, sym);
        unicc.get_unique_id = s_uid_get_4096 ? wrap_get_unique_id_4096 : NULL;
    } else {
        s_uid_get_128 = (int(*)(unicc_native_uid_t*))unicc_platform_dlsym(handle, sym);
        unicc.get_unique_id = s_uid_get_128 ? wrap_get_unique_id_128 : NULL;
    }

    snprintf(sym, sizeof sym, "%sCommDestroy", d->name);
    s_comm_destroy = (int(*)(unicc_comm_t))unicc_platform_dlsym(handle, sym);
    unicc.comm_destroy = s_comm_destroy;

    snprintf(sym, sizeof sym, "%sCommCount", d->name);
    s_comm_count = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, sym);
    unicc.comm_count = s_comm_count;

    if (d->has_comm_user_rank) {
        snprintf(sym, sizeof sym, "%sCommUserRank", d->name);
        s_comm_user_rank = (int(*)(unicc_comm_t, int*))unicc_platform_dlsym(handle, sym);
        unicc.comm_user_rank = s_comm_user_rank;
    } /* else: slot stays NULL (ECCL exports no such symbol). */

    snprintf(sym, sizeof sym, "%sGroupStart", d->name);
    s_group_start = (int(*)(void))unicc_platform_dlsym(handle, sym);
    unicc.group_start = s_group_start;

    snprintf(sym, sizeof sym, "%sGroupEnd", d->name);
    s_group_end = (int(*)(void))unicc_platform_dlsym(handle, sym);
    unicc.group_end = s_group_end;

    return UNICC_OK;
}
