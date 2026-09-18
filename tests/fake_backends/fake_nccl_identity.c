/* fake_nccl_identity.c - NCCL-shaped fixture library.
 *
 * A shared library that exports an NCCL-shaped symbol set with NCCL's native
 * signatures, so the loader / vtable / API tests run on hosts with no real
 * NCCL installed. Exports a distinctive version and implements a host-side
 * element-wise SUM allreduce, which lets tests prove the nccl* family bound
 * and that data actually round-trips through the UniCCL layers.
 *
 * Compile-time variant (mirrors unimpi's fake fixtures):
 *   UNICC_FAKE_OMIT_GROUP_END - leave ncclGroupEnd out; the vtable slot
 *   degrades to NULL and unicc_group_end() returns UNICC_ERR_NOT_SUPPORTED.
 */
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define FAKE_EXPORT __declspec(dllexport)
#else
#define FAKE_EXPORT __attribute__((visibility("default")))
#endif

typedef struct { char internal[128]; } ncclUniqueId;
typedef void* ncclComm_t;

/* Distinctive fake version, NCCL encoding (major<<22)|(minor<<12)|patch. */
#define FAKE_NCCL_VERSION ((2U << 22) | (19U << 12) | 7U)   /* 2.19.7 */

static size_t fake_dt_size(int dt) {
    switch (dt) {
        case 0: case 1: case 2:  return 1;  /* int8 / char / uint8 */
        case 3: case 4:          return 4;  /* int32 / uint32 */
        case 5: case 6:          return 8;  /* int64 / uint64 */
        case 7:                  return 4;  /* float32 */
        case 8:                  return 8;  /* float64 */
        case 9: case 10:         return 2;  /* half / bfloat16 */
    }
    return 0;
}

/* Host-side element-wise reduction for the tested numeric types; anything
 * else is a plain copy so the fixture stays obviously correct. */
static void fake_allreduce_impl(const void *send, void *recv, size_t count,
                                int datatype, int op) {
    if (op == 0 /* ncclSum */) {
        if (datatype == 7 /* float32 */) {
            const float *s = (const float*)send;
            float *r = (float*)recv;
            for (size_t i = 0; i < count; i++) r[i] += s[i];
            return;
        }
        if (datatype == 3 /* int32 */) {
            const int *s = (const int*)send;
            int *r = (int*)recv;
            for (size_t i = 0; i < count; i++) r[i] += s[i];
            return;
        }
    }
    size_t sz = fake_dt_size(datatype);
    if (sz && count) {
        memcpy(recv, send, count * sz);
    }
}

FAKE_EXPORT int ncclGetVersion(int *version) {
    if (version) *version = (int)FAKE_NCCL_VERSION;
    return 0;
}

FAKE_EXPORT int ncclGetUniqueId(ncclUniqueId *id) {
    if (id) memset(id, 0xAB, sizeof(*id));
    return 0;
}

FAKE_EXPORT int ncclCommInitRank(ncclComm_t *comm, int nranks,
                                 ncclUniqueId uid, int rank) {
    (void)nranks; (void)uid; (void)rank;
    if (comm) *comm = (void*)0x1;
    return 0;
}

FAKE_EXPORT int ncclCommDestroy(ncclComm_t comm) {
    (void)comm;
    return 0;
}

FAKE_EXPORT int ncclCommCount(ncclComm_t comm, int *count) {
    (void)comm;
    if (count) *count = 1;
    return 0;
}

FAKE_EXPORT int ncclCommUserRank(ncclComm_t comm, int *rank) {
    (void)comm;
    if (rank) *rank = 0;
    return 0;
}

FAKE_EXPORT int ncclAllReduce(const void *sendbuf, void *recvbuf, size_t count,
                              int datatype, int op, ncclComm_t comm, void *stream) {
    (void)comm; (void)stream;
    fake_allreduce_impl(sendbuf, recvbuf, count, datatype, op);
    return 0;
}

FAKE_EXPORT int ncclBroadcast(void *buf, size_t count, int datatype, int root,
                              ncclComm_t comm, void *stream) {
    (void)buf; (void)count; (void)datatype; (void)root; (void)comm; (void)stream;
    return 0;
}

FAKE_EXPORT int ncclGroupStart(void) {
    return 0;
}

#ifndef UNICC_FAKE_OMIT_GROUP_END
FAKE_EXPORT int ncclGroupEnd(void) {
    return 0;
}
#endif
