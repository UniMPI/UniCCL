/* fake_oneccl_identity.c - Intel oneCCL v2 C API fixture.
 *
 * Exports the oneccl* symbol family with oneCCL v2's distinguishing traits:
 * a 4096-byte onecclUniqueId (double-buffered onecclBroadcast, allreduce that
 * adds a distinctive +2000.0f on float32 so tests can prove the oneccl*
 * binding was actually bound rather than some nccl*-family path). */
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define FAKE_EXPORT __declspec(dllexport)
#else
#define FAKE_EXPORT __attribute__((visibility("default")))
#endif

#define FAKE_ONECCL_UID_BYTES 4096
#define FAKE_ONECCL_VERSION ((2022U << 22) | (1U << 12) | 0U)  /* 2022.1.0 */

typedef struct { char internal[FAKE_ONECCL_UID_BYTES]; } onecclUniqueId;
typedef void* onecclComm_t;

static size_t fake_dt_size(int dt) {
    switch (dt) {
        case 0: case 1: case 2:  return 1;
        case 3: case 4:          return 4;
        case 5: case 6:          return 8;
        case 7:                  return 4;
        case 8:                  return 8;
        case 9:                  return 2;
    }
    return 0;
}

static void oneccl_allreduce_impl(const void *send, void *recv, size_t count,
                                  int datatype, int op) {
    if (op == 0 /* onecclSum */) {
        if (datatype == 7 /* float32 */) {
            const float *s = (const float*)send;
            float *r = (float*)recv;
            for (size_t i = 0; i < count; i++) r[i] += s[i] + 2000.0f;
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

/* ---- oneccl* (v2 C API) family ---- */

FAKE_EXPORT int onecclGetVersion(int *version) {
    if (version) *version = (int)FAKE_ONECCL_VERSION;
    return 0;
}

FAKE_EXPORT int onecclGetUniqueId(onecclUniqueId *id) {
    if (id) memset(id, 0x11, sizeof(*id));
    return 0;
}

FAKE_EXPORT int onecclCommInitRank(onecclComm_t *comm, size_t nranks,
                                   onecclUniqueId uid, int rank) {
    (void)nranks; (void)uid; (void)rank;
    if (comm) *comm = (void*)0x3;
    return 0;
}

FAKE_EXPORT int onecclCommDestroy(onecclComm_t comm) {
    (void)comm;
    return 0;
}

FAKE_EXPORT int onecclCommCount(const onecclComm_t comm, int *count) {
    (void)comm;
    if (count) *count = 1;
    return 0;
}

FAKE_EXPORT int onecclCommUserRank(onecclComm_t comm, int *rank) {
    (void)comm;
    if (rank) *rank = 0;
    return 0;
}

FAKE_EXPORT int onecclAllReduce(void *sendbuff, void *recvbuff, size_t count,
                                int datatype, int op, onecclComm_t comm, void *stream) {
    (void)comm; (void)stream;
    oneccl_allreduce_impl(sendbuff, recvbuff, count, datatype, op);
    return 0;
}

/* oneCCL v2 broadcast is double-buffered (const send, void recv). */
FAKE_EXPORT int onecclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                                int datatype, int root, onecclComm_t comm, void *stream) {
    (void)datatype; (void)root; (void)comm; (void)stream;
    size_t sz = fake_dt_size(datatype);
    if (sz && count) {
        memcpy(recvbuff, sendbuff, count * sz);
    }
    return 0;
}

FAKE_EXPORT int onecclGroupStart(void) {
    return 0;
}

FAKE_EXPORT int onecclGroupEnd(void) {
    return 0;
}
