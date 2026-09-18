/* fake_rccl_identity.c - rccl*+nccl* dual-family fixture (defensive).
 *
 * Deliberately exports BOTH the rccl* (native) and nccl* (ABI-compatibility)
 * families. NOTE: modern AMD RCCL's public API is nccl*-named with no public
 * rccl* symbols (docs/official/), so this fixture models a historical /
 * third-party shape, not current AMD RCCL. It stays as the discriminating
 * check: UniCCL must identify such a library as RCCL (rccl* exclusives are
 * checked first) and bind the rccl* symbols - never the nccl* ones.
 * rcclGetVersion and rcclAllReduce return values distinct from their nccl*
 * twins so tests can prove the rccl* family was actually bound.
 */
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define FAKE_EXPORT __declspec(dllexport)
#else
#define FAKE_EXPORT __attribute__((visibility("default")))
#endif

typedef struct { char internal[128]; } rcclUniqueId;
typedef struct { char internal[128]; } ncclUniqueId;
typedef void* rcclComm_t;
typedef void* ncclComm_t;

/* Distinctive fake versions (must differ from the nccl*-compat values). */
#define FAKE_RCCL_VERSION ((6U << 20) | (1U << 12) | 96U)   /* 6.1.96 */
#define FAKE_RCCL_NCCL_COMPAT_VERSION ((2U << 22) | (18U << 12) | 5U)  /* 2.18.5 */

static size_t fake_dt_size(int dt) {
    switch (dt) {
        case 0: case 1: case 2:  return 1;
        case 3: case 4:          return 4;
        case 5: case 6:          return 8;
        case 7:                  return 4;
        case 8:                  return 8;
        case 9: case 10:         return 2;
    }
    return 0;
}

/* rcclAllReduce adds a distinctive +1000.0f on float32 so tests can prove the
 * rccl* (native) path was bound rather than any nccl*-compat path. */
static void rccl_allreduce_impl(const void *send, void *recv, size_t count,
                                int datatype, int op) {
    if (op == 0 /* fcclSum */) {
        if (datatype == 7 /* float32 */) {
            const float *s = (const float*)send;
            float *r = (float*)recv;
            for (size_t i = 0; i < count; i++) r[i] += s[i] + 1000.0f;
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

/* ---- rccl* (native) family ---- */

FAKE_EXPORT int rcclGetVersion(int *version) {
    if (version) *version = (int)FAKE_RCCL_VERSION;
    return 0;
}

FAKE_EXPORT int rcclGetUniqueId(rcclUniqueId *id) {
    if (id) memset(id, 0xCD, sizeof(*id));
    return 0;
}

FAKE_EXPORT int rcclCommInitRank(rcclComm_t *comm, int nranks,
                                 rcclUniqueId uid, int rank) {
    (void)nranks; (void)uid; (void)rank;
    if (comm) *comm = (void*)0x2;
    return 0;
}

FAKE_EXPORT int rcclCommDestroy(rcclComm_t comm) {
    (void)comm;
    return 0;
}

FAKE_EXPORT int rcclCommCount(rcclComm_t comm, int *count) {
    (void)comm;
    if (count) *count = 1;
    return 0;
}

FAKE_EXPORT int rcclCommUserRank(rcclComm_t comm, int *rank) {
    (void)comm;
    if (rank) *rank = 0;
    return 0;
}

FAKE_EXPORT int rcclAllReduce(const void *sendbuf, void *recvbuf, size_t count,
                              int datatype, int op, rcclComm_t comm, void *stream) {
    (void)comm; (void)stream;
    rccl_allreduce_impl(sendbuf, recvbuf, count, datatype, op);
    return 0;
}

FAKE_EXPORT int rcclBroadcast(void *buf, size_t count, int datatype, int root,
                              rcclComm_t comm, void *stream) {
    (void)buf; (void)count; (void)datatype; (void)root; (void)comm; (void)stream;
    return 0;
}

FAKE_EXPORT int rcclGroupStart(void) {
    return 0;
}

FAKE_EXPORT int rcclGroupEnd(void) {
    return 0;
}

/* ---- nccl* (ABI-compatibility) family, mirroring real RCCL ---- */

FAKE_EXPORT int ncclGetVersion(int *version) {
    if (version) *version = (int)FAKE_RCCL_NCCL_COMPAT_VERSION;
    return 0;
}

FAKE_EXPORT int ncclGetUniqueId(ncclUniqueId *id) {
    if (id) memset(id, 0xEF, sizeof(*id));
    return 0;
}

FAKE_EXPORT int ncclCommInitRank(ncclComm_t *comm, int nranks,
                                 ncclUniqueId uid, int rank) {
    (void)nranks; (void)uid; (void)rank;
    if (comm) *comm = (void*)0x2;
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
    rccl_allreduce_impl(sendbuf, recvbuf, count, datatype, op); /* compat = same math */
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

FAKE_EXPORT int ncclGroupEnd(void) {
    return 0;
}
