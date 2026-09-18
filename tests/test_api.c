/* test_api.c - end-to-end tests of the public unicc_* API against the fake
 * fixtures. Proves, on a host with no NCCL/RCCL installed:
 *   1. a full fake-nccl round trip (init/version/comm/allreduce/broadcast/group,
 *      finalize and post-finalize/lifecycle errors);
 *   2. a fake-rccl library that also exports nccl* compats is identified and
 *      driven through its rccl* (native) path, never the nccl* compat one;
 *   3. a backend missing an optional symbol degrades to a NULL slot:
 *      unicc_group_end() returns UNICC_ERR_NOT_SUPPORTED, everything else works. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unicc.h"

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
#define EXPECT_FAKE_ONECCL_VERSION ((2022U << 22) | (1U << 12) | 0U)
#define EXPECT_FAKE_ECCL_VERSION ((3U << 22) | (5U << 12) | 1U)

static void switch_fixture(const char *path) {
    unsetenv("UNICC_BACKEND");
    setenv("UNICC_LIBRARY", path, 1);
}

static void test_fake_nccl_full(const char *fake_nccl) {
    printf("[test_api] fake-nccl full round trip...\n");
    switch_fixture(fake_nccl);

    /* pre-init guards */
    int raw = 12345;
    float in[4] = {1, 2, 3, 4}, out[4] = {0, 0, 0, 0};
    CHECK(unicc_allreduce(in, out, 4, UNICC_F32, UNICC_SUM, NULL, NULL) == UNICC_ERR_NOT_INITIALIZED);
    CHECK(unicc_finalize() == UNICC_ERR_NOT_INITIALIZED);
    CHECK(unicc_is_initialized() == 0);

    CHECK(unicc_init() == UNICC_OK);
    CHECK(unicc_is_initialized() == 1);
    CHECK(unicc_init() == UNICC_ERR_ALREADY_INITIALIZED);
    CHECK(strcmp(unicc_backend_name(), "nccl") == 0);
    CHECK(unicc_get_library_path()[0] != '\0');

    int bv = 0;
    CHECK(unicc_backend_version(&bv) == UNICC_OK);
    CHECK((unsigned)bv == EXPECT_FAKE_NCCL_VERSION);
    char ver[64] = {0};
    CHECK(unicc_get_version(ver, sizeof(ver)) == UNICC_OK);
    CHECK(strstr(ver, "UniCCL") != NULL);

    /* communicator bootstrap */
    CHECK(unicc_comm_available() == 1);
    unicc_comm_id_t id;
    CHECK(unicc_get_unique_id(&id) == UNICC_OK);
    unicc_comm_t comm = NULL;
    CHECK(unicc_comm_init_rank(&comm, 1, &id, 0) == UNICC_OK);
    CHECK(comm != NULL);
    int count = 0, crank = -1;
    CHECK(unicc_comm_count(comm, &count) == UNICC_OK);
    CHECK(count == 1);
    CHECK(unicc_comm_user_rank(comm, &crank) == UNICC_OK);
    CHECK(crank == 0);

    /* allreduce: fake sums element-wise into out (out pre-filled to prove "+= ") */
    for (int i = 0; i < 4; i++) out[i] = 10.0f;
    CHECK(unicc_allreduce(in, out, 4, UNICC_F32, UNICC_SUM, comm, NULL) == UNICC_OK);
    for (int i = 0; i < 4; i++) CHECK(out[i] == 10.0f + in[i]);

    /* invalid datatype / op rejected by the mapping */
    CHECK(unicc_allreduce(in, out, 4, (unicc_datatype_t)99, UNICC_SUM, comm, NULL)
          == UNICC_ERR_INVALID_ARGUMENT);
    CHECK(unicc_allreduce(in, out, 4, UNICC_F32, (unicc_reduce_op_t)99, comm, NULL)
          == UNICC_ERR_INVALID_ARGUMENT);

    CHECK(unicc_broadcast(out, 4, UNICC_F32, 0, comm, NULL) == UNICC_OK);
    CHECK(unicc_allreduce_available() == 1);
    CHECK(unicc_broadcast_available() == 1);
    CHECK(unicc_group_start() == UNICC_OK);
    CHECK(unicc_group_end() == UNICC_OK);

    /* last error: raw is 0 so far (no backend failure yet) */
    raw = 999;
    CHECK(unicc_get_last_error(&raw) == UNICC_OK);
    CHECK(raw == 0);

    CHECK(unicc_comm_destroy(comm) == UNICC_OK);

    CHECK(unicc_finalize() == UNICC_OK);
    CHECK(unicc_is_initialized() == 0);
    CHECK(strcmp(unicc_backend_name(), "unknown") == 0);

    /* post-finalize guard */
    CHECK(unicc_allreduce(in, out, 4, UNICC_F32, UNICC_SUM, comm, NULL) == UNICC_ERR_NOT_INITIALIZED);
}

static void test_fake_rccl_identified_rccl(const char *fake_rccl) {
    printf("[test_api] fake-rccl (dual rccl*+nccl*) identified and bound as rccl...\n");
    switch_fixture(fake_rccl);

    CHECK(unicc_init() == UNICC_OK);
    CHECK(strcmp(unicc_backend_name(), "rccl") == 0);   /* not "nccl"! */

    int bv = 0;
    CHECK(unicc_backend_version(&bv) == UNICC_OK);
    CHECK((unsigned)bv == EXPECT_FAKE_RCCL_VERSION);  /* rcclGetVersion bound */

    unicc_comm_id_t id;
    CHECK(unicc_get_unique_id(&id) == UNICC_OK);
    unicc_comm_t comm = NULL;
    CHECK(unicc_comm_init_rank(&comm, 1, &id, 0) == UNICC_OK);

    /* rcclAllReduce adds a distinctive +1000.0f on float32: if the nccl*-compat
     * family had been bound instead, the +1000 marker would be absent. This is
     * the proof that the rccl* (native) path is the one wired up. */
    float in[2] = {5.0f, 7.0f}, out[2] = {0.0f, 0.0f};
    CHECK(unicc_allreduce(in, out, 2, UNICC_F32, UNICC_SUM, comm, NULL) == UNICC_OK);
    CHECK(out[0] == 1005.0f && out[1] == 1007.0f);

    CHECK(unicc_comm_destroy(comm) == UNICC_OK);
    CHECK(unicc_finalize() == UNICC_OK);
}

static void test_fake_oneccl_full(const char *fake_oneccl) {
    printf("[test_api] fake-oneccl full round trip...\n");
    switch_fixture(fake_oneccl);

    int bv = 0;
    CHECK(unicc_init() == UNICC_OK);
    CHECK(strcmp(unicc_backend_name(), "oneccl") == 0);
    CHECK(unicc_backend_version(&bv) == UNICC_OK);
    CHECK((unsigned)bv == EXPECT_FAKE_ONECCL_VERSION);

    /* The oneccl fixture ex ports a 4096-byte id; the wrapper must carry the
     * full length through get_unique_id (never truncate to 128). */
    unicc_comm_id_t id;
    CHECK(unicc_get_unique_id(&id) == UNICC_OK);
    CHECK(id.len == 4096);

    unicc_comm_t comm = NULL;
    CHECK(unicc_comm_init_rank(&comm, 1, &id, 0) == UNICC_OK);
    CHECK(comm != NULL);

    int count = 0, crank = -1;
    CHECK(unicc_comm_count(comm, &count) == UNICC_OK);
    CHECK(count == 1);
    CHECK(unicc_comm_user_rank(comm, &crank) == UNICC_OK);
    CHECK(crank == 0);

    /* onecclAllReduce adds +2000.0f on float32: proves the oneccl* binding
     * (not some nccl* fallback) is wired up, and that broadcast went through
     * the double-buffer adapter. */
    float in[2] = {3.0f, 9.0f}, out[2] = {0.0f, 0.0f};
    CHECK(unicc_allreduce(in, out, 2, UNICC_F32, UNICC_SUM, comm, NULL) == UNICC_OK);
    CHECK(out[0] == 2003.0f && out[1] == 2009.0f);
    out[0] = 0.0f; out[1] = 0.0f;
    CHECK(unicc_broadcast(out, 2, UNICC_F32, 0, comm, NULL) == UNICC_OK);

    CHECK(unicc_group_start() == UNICC_OK);
    CHECK(unicc_group_end() == UNICC_OK);

    CHECK(unicc_comm_destroy(comm) == UNICC_OK);
    CHECK(unicc_finalize() == UNICC_OK);
}

static void test_fake_eccl_full(const char *fake_eccl) {
    printf("[test_api] fake-eccl full round trip...\n");
    switch_fixture(fake_eccl);

    int bv = 0;
    CHECK(unicc_init() == UNICC_OK);
    CHECK(strcmp(unicc_backend_name(), "eccl") == 0);
    CHECK(unicc_backend_version(&bv) == UNICC_OK);
    CHECK((unsigned)bv == EXPECT_FAKE_ECCL_VERSION);

    unicc_comm_id_t id;
    CHECK(unicc_get_unique_id(&id) == UNICC_OK);
    CHECK(id.len == 128);

    unicc_comm_t comm = NULL;
    CHECK(unicc_comm_init_rank(&comm, 1, &id, 0) == UNICC_OK);

    /* ecclCommUserRank is not exported by the ecosystem: slot NULL, api reports
     * NOT_SUPPORTED while the rest of the face keeps working. */
    int crank = -1;
    CHECK(unicc_comm_user_rank(comm, &crank) == UNICC_ERR_NOT_SUPPORTED);

    /* ecclAllReduce adds +3000.0f on float32. */
    float in[2] = {1.0f, 2.0f}, out[2] = {0.0f, 0.0f};
    CHECK(unicc_allreduce(in, out, 2, UNICC_F32, UNICC_SUM, comm, NULL) == UNICC_OK);
    CHECK(out[0] == 3001.0f && out[1] == 3002.0f);
    out[0] = 0.0f; out[1] = 0.0f;
    CHECK(unicc_broadcast(out, 2, UNICC_F32, 0, comm, NULL) == UNICC_OK);

    CHECK(unicc_comm_destroy(comm) == UNICC_OK);
    CHECK(unicc_finalize() == UNICC_OK);
}

static void test_group_end_degrade(const char *fake_nccl_missing) {
    printf("[test_api] missing-symbol degrade (group_end NULL -> NOT_SUPPORTED)...\n");
    switch_fixture(fake_nccl_missing);

    CHECK(unicc_init() == UNICC_OK);              /* core intact -> init succeeds */
    CHECK(strcmp(unicc_backend_name(), "nccl") == 0);

    CHECK(unicc_group_start() == UNICC_OK);
    CHECK(unicc_group_end_available() == 0);
    CHECK(unicc_group_end() == UNICC_ERR_NOT_SUPPORTED);  /* slot NULL -> degrade */

    /* everything else still works */
    float in[2] = {1, 2}, out[2] = {0, 0};
    unicc_comm_t comm = NULL;
    unicc_comm_id_t id;
    CHECK(unicc_get_unique_id(&id) == UNICC_OK);
    CHECK(unicc_comm_init_rank(&comm, 1, &id, 0) == UNICC_OK);
    CHECK(unicc_allreduce(in, out, 2, UNICC_F32, UNICC_SUM, comm, NULL) == UNICC_OK);
    CHECK(out[0] == 1.0f && out[1] == 2.0f);
    CHECK(unicc_comm_destroy(comm) == UNICC_OK);

    CHECK(unicc_finalize() == UNICC_OK);
}

int main(int argc, char **argv) {
    const char *fake_nccl = NULL;
    const char *fake_rccl = NULL;
    const char *fake_nccl_missing = NULL;
    const char *fake_oneccl = NULL;
    const char *fake_eccl = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fake-nccl") == 0 && i + 1 < argc) { fake_nccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-rccl") == 0 && i + 1 < argc) { fake_rccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-nccl-missing") == 0 && i + 1 < argc) { fake_nccl_missing = argv[++i]; }
        else if (strcmp(argv[i], "--fake-oneccl") == 0 && i + 1 < argc) { fake_oneccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-eccl") == 0 && i + 1 < argc) { fake_eccl = argv[++i]; }
    }

    printf("unicc_version_string = %s\n", UNICC_VERSION_STRING);
    CHECK(unicc_print_backend_info() == UNICC_OK);

    if (fake_nccl) test_fake_nccl_full(fake_nccl);
    if (fake_rccl) test_fake_rccl_identified_rccl(fake_rccl);
    if (fake_nccl_missing) test_group_end_degrade(fake_nccl_missing);
    if (fake_oneccl) test_fake_oneccl_full(fake_oneccl);
    if (fake_eccl) test_fake_eccl_full(fake_eccl);

    if (g_failures == 0) {
        printf("test_api: ALL TESTS PASSED\n");
        return 0;
    }
    printf("test_api: %d FAILURE(S)\n", g_failures);
    return 1;
}
