/* test_api.c - end-to-end tests of the public xcc_* API against the fake
 * fixtures. Proves, on a host with no NCCL/RCCL installed:
 *   1. a full fake-nccl round trip (init/version/comm/allreduce/broadcast/group,
 *      finalize and post-finalize/lifecycle errors);
 *   2. a fake-rccl library that also exports nccl* compats is identified and
 *      driven through its rccl* (native) path, never the nccl* compat one;
 *   3. a backend missing an optional symbol degrades to a NULL slot:
 *      xcc_group_end() returns XCC_ERR_NOT_SUPPORTED, everything else works. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xcc.h"

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
    } \
} while (0)

/* Must match the values hardcoded in the fixtures. */
#define EXPECT_FAKE_NCCL_VERSION ((2U << 22) | (19U << 12) | 7U)
#define EXPECT_FAKE_RCCL_VERSION ((6U << 20) | (1U << 12) | 96U)

static void switch_fixture(const char *path) {
    unsetenv("XCCL_BACKEND");
    setenv("XCCL_LIBRARY", path, 1);
}

static void test_fake_nccl_full(const char *fake_nccl) {
    printf("[test_api] fake-nccl full round trip...\n");
    switch_fixture(fake_nccl);

    /* pre-init guards */
    int raw = 12345;
    float in[4] = {1, 2, 3, 4}, out[4] = {0, 0, 0, 0};
    CHECK(xcc_allreduce(in, out, 4, XCC_F32, XCC_SUM, NULL, NULL) == XCC_ERR_NOT_INITIALIZED);
    CHECK(xcc_finalize() == XCC_ERR_NOT_INITIALIZED);
    CHECK(xcc_is_initialized() == 0);

    CHECK(xcc_init() == XCC_OK);
    CHECK(xcc_is_initialized() == 1);
    CHECK(xcc_init() == XCC_ERR_ALREADY_INITIALIZED);
    CHECK(strcmp(xcc_backend_name(), "nccl") == 0);
    CHECK(xcc_get_library_path()[0] != '\0');

    int bv = 0;
    CHECK(xcc_backend_version(&bv) == XCC_OK);
    CHECK((unsigned)bv == EXPECT_FAKE_NCCL_VERSION);
    char ver[64] = {0};
    CHECK(xcc_get_version(ver, sizeof(ver)) == XCC_OK);
    CHECK(strstr(ver, "XCCL") != NULL);

    /* communicator bootstrap */
    CHECK(xcc_comm_available() == 1);
    xcc_unique_id_t uid;
    CHECK(xcc_get_unique_id(&uid) == XCC_OK);
    xcc_comm_t comm = NULL;
    CHECK(xcc_comm_init_rank(&comm, 1, uid, 0) == XCC_OK);
    CHECK(comm != NULL);
    int count = 0, crank = -1;
    CHECK(xcc_comm_count(comm, &count) == XCC_OK);
    CHECK(count == 1);
    CHECK(xcc_comm_user_rank(comm, &crank) == XCC_OK);
    CHECK(crank == 0);

    /* allreduce: fake sums element-wise into out (out pre-filled to prove "+= ") */
    for (int i = 0; i < 4; i++) out[i] = 10.0f;
    CHECK(xcc_allreduce(in, out, 4, XCC_F32, XCC_SUM, comm, NULL) == XCC_OK);
    for (int i = 0; i < 4; i++) CHECK(out[i] == 10.0f + in[i]);

    /* invalid datatype / op rejected by the mapping */
    CHECK(xcc_allreduce(in, out, 4, (xcc_datatype_t)99, XCC_SUM, comm, NULL)
          == XCC_ERR_INVALID_ARGUMENT);
    CHECK(xcc_allreduce(in, out, 4, XCC_F32, (xcc_reduce_op_t)99, comm, NULL)
          == XCC_ERR_INVALID_ARGUMENT);

    CHECK(xcc_broadcast(out, 4, XCC_F32, 0, comm, NULL) == XCC_OK);
    CHECK(xcc_allreduce_available() == 1);
    CHECK(xcc_broadcast_available() == 1);
    CHECK(xcc_group_start() == XCC_OK);
    CHECK(xcc_group_end() == XCC_OK);

    /* last error: raw is 0 so far (no backend failure yet) */
    raw = 999;
    CHECK(xcc_get_last_error(&raw) == XCC_OK);
    CHECK(raw == 0);

    CHECK(xcc_comm_destroy(comm) == XCC_OK);

    CHECK(xcc_finalize() == XCC_OK);
    CHECK(xcc_is_initialized() == 0);
    CHECK(strcmp(xcc_backend_name(), "unknown") == 0);

    /* post-finalize guard */
    CHECK(xcc_allreduce(in, out, 4, XCC_F32, XCC_SUM, comm, NULL) == XCC_ERR_NOT_INITIALIZED);
}

static void test_fake_rccl_identified_rccl(const char *fake_rccl) {
    printf("[test_api] fake-rccl (dual rccl*+nccl*) identified and bound as rccl...\n");
    switch_fixture(fake_rccl);

    CHECK(xcc_init() == XCC_OK);
    CHECK(strcmp(xcc_backend_name(), "rccl") == 0);   /* not "nccl"! */

    int bv = 0;
    CHECK(xcc_backend_version(&bv) == XCC_OK);
    CHECK((unsigned)bv == EXPECT_FAKE_RCCL_VERSION);  /* rcclGetVersion bound */

    xcc_unique_id_t uid;
    CHECK(xcc_get_unique_id(&uid) == XCC_OK);
    xcc_comm_t comm = NULL;
    CHECK(xcc_comm_init_rank(&comm, 1, uid, 0) == XCC_OK);

    /* rcclAllReduce adds a distinctive +1000.0f on float32: if the nccl*-compat
     * family had been bound instead, the +1000 marker would be absent. This is
     * the proof that the rccl* (native) path is the one wired up. */
    float in[2] = {5.0f, 7.0f}, out[2] = {0.0f, 0.0f};
    CHECK(xcc_allreduce(in, out, 2, XCC_F32, XCC_SUM, comm, NULL) == XCC_OK);
    CHECK(out[0] == 1005.0f && out[1] == 1007.0f);

    CHECK(xcc_comm_destroy(comm) == XCC_OK);
    CHECK(xcc_finalize() == XCC_OK);
}

static void test_group_end_degrade(const char *fake_nccl_missing) {
    printf("[test_api] missing-symbol degrade (group_end NULL -> NOT_SUPPORTED)...\n");
    switch_fixture(fake_nccl_missing);

    CHECK(xcc_init() == XCC_OK);              /* core intact -> init succeeds */
    CHECK(strcmp(xcc_backend_name(), "nccl") == 0);

    CHECK(xcc_group_start() == XCC_OK);
    CHECK(xcc_group_end_available() == 0);
    CHECK(xcc_group_end() == XCC_ERR_NOT_SUPPORTED);  /* slot NULL -> degrade */

    /* everything else still works */
    float in[2] = {1, 2}, out[2] = {0, 0};
    xcc_comm_t comm = NULL;
    xcc_unique_id_t uid;
    CHECK(xcc_get_unique_id(&uid) == XCC_OK);
    CHECK(xcc_comm_init_rank(&comm, 1, uid, 0) == XCC_OK);
    CHECK(xcc_allreduce(in, out, 2, XCC_F32, XCC_SUM, comm, NULL) == XCC_OK);
    CHECK(out[0] == 1.0f && out[1] == 2.0f);
    CHECK(xcc_comm_destroy(comm) == XCC_OK);

    CHECK(xcc_finalize() == XCC_OK);
}

int main(int argc, char **argv) {
    const char *fake_nccl = NULL;
    const char *fake_rccl = NULL;
    const char *fake_nccl_missing = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fake-nccl") == 0 && i + 1 < argc) { fake_nccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-rccl") == 0 && i + 1 < argc) { fake_rccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-nccl-missing") == 0 && i + 1 < argc) { fake_nccl_missing = argv[++i]; }
    }

    printf("xcc_version_string = %s\n", XCC_VERSION_STRING);
    CHECK(xcc_print_backend_info() == XCC_OK);

    if (fake_nccl) test_fake_nccl_full(fake_nccl);
    if (fake_rccl) test_fake_rccl_identified_rccl(fake_rccl);
    if (fake_nccl_missing) test_group_end_degrade(fake_nccl_missing);

    if (g_failures == 0) {
        printf("test_api: ALL TESTS PASSED\n");
        return 0;
    }
    printf("test_api: %d FAILURE(S)\n", g_failures);
    return 1;
}
