/* fake_eccl_identity.c - Enflame ECCL fixture.
 *
 * Exports the eccl* symbol family mirroring the ecosystem call sites in
 * torch-gcu: NCCL-shaped, 128-byte id, double-buffered broadcast, and - the
 * trait this fixture exists to prove - NO ecclCommUserRank (torch-gcu keeps
 * rank in its own process group), so unicc_comm_user_rank() must report
 * NOT_SUPPORTED while everything else works. allreduce adds a distinctive
 * +3000.0f on float32 so tests can prove the eccl* binding was really bound. */
#include <stdlib.h>
#include <string.h>
#include "unicc_native_id.h"

#ifdef _WIN32
#define FAKE_EXPORT __declspec(dllexport)
#else
#define FAKE_EXPORT __attribute__((visibility("default")))
#endif

#define FAKE_ECCL_VERSION ((3U << 22) | (5U << 12) | 1U)  /* 3.5.1 */

/* 128-byte unique id: the shared type from unicc_native_id.h - it must be
 * the exact same type the eccl binding's dlsym cast uses. */
typedef void* ecclComm_t;

static size_t fake_dt_size(int dt) {
    /* Modern NCCL numbering (ncclInt8=0 .. ncclBfloat16=9), matching
     * docs/BACKENDS.md "Enum mapping". */
    switch (dt) {
        case 0: return 1;  /* I8  */
        case 1: return 1;  /* U8  */
        case 2: return 4;  /* I32 */
        case 3: return 4;  /* U32 */
        case 4: return 8;  /* I64 */
        case 5: return 8;  /* U64 */
        case 6: return 2;  /* F16 */
        case 7: return 4;  /* F32 */
        case 8: return 8;  /* F64 */
        case 9: return 2;  /* BF16*/
    }
    return 0;
}

static void eccl_allreduce_impl(const void *send, void *recv, size_t count,
                                int datatype, int op) {
    if (op == 0 /* ecclSum */) {
        if (datatype == 7 /* float32 */) {
            const float *s = (const float*)send;
            float *r = (float*)recv;
            for (size_t i = 0; i < count; i++) r[i] += s[i] + 3000.0f;
            return;
        }
        if (datatype == 2 /* int32 (ncclInt32) */) {
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

/* ---- eccl* family ---- */

FAKE_EXPORT int ecclGetVersion(int *version) {
    if (version) *version = (int)FAKE_ECCL_VERSION;
    return 0;
}

FAKE_EXPORT int ecclGetUniqueId(unicc_native_uid_t *id) {
    if (id) memset(id, 0x22, sizeof(*id));
    return 0;
}

FAKE_EXPORT int ecclCommInitRank(ecclComm_t *comm, size_t numranks,
                                 unicc_native_uid_t commId, int rank) {
    (void)numranks; (void)commId; (void)rank;
    if (comm) *comm = (void*)0x4;
    return 0;
}

FAKE_EXPORT int ecclCommDestroy(ecclComm_t comm) {
    (void)comm;
    return 0;
}

FAKE_EXPORT int ecclCommCount(ecclComm_t comm, int *count) {
    (void)comm;
    if (count) *count = 1;
    return 0;
}

FAKE_EXPORT int ecclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int op, ecclComm_t comm, void *stream) {
    (void)comm; (void)stream;
    eccl_allreduce_impl(sendbuff, recvbuff, count, datatype, op);
    return 0;
}

FAKE_EXPORT int ecclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int root, ecclComm_t comm, void *stream) {
    (void)datatype; (void)root; (void)comm; (void)stream;
    size_t sz = fake_dt_size(datatype);
    if (sz && count) {
        memcpy(recvbuff, sendbuff, count * sz);
    }
    return 0;
}

FAKE_EXPORT int ecclGroupStart(void) {
    return 0;
}

FAKE_EXPORT int ecclGroupEnd(void) {
    return 0;
}
