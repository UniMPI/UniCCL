/* test_vtable.c - vtable layout, zero-init, core validation and binding.
 *
 * Verifies the dispatch table contract: it starts fully NULL, a full backend
 * fills every slot, a backend missing an optional symbol leaves that slot
 * NULL (the degrade pattern), and cleanup returns to full zeros. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unicc_vtable.h"
#include "unicc_backends.h"

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
    } \
} while (0)

int main(int argc, char **argv) {
    const char *fake_nccl = NULL;
    const char *fake_nccl_missing = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fake-nccl") == 0 && i + 1 < argc) { fake_nccl = argv[++i]; }
        else if (strcmp(argv[i], "--fake-nccl-missing") == 0 && i + 1 < argc) { fake_nccl_missing = argv[++i]; }
    }

    /* The global table must start fully zero-initialized. */
    CHECK(unicc.get_version == NULL);
    CHECK(unicc.comm_init_rank == NULL);
    CHECK(unicc.allreduce == NULL);
    CHECK(unicc.broadcast == NULL);
    CHECK(unicc.get_unique_id == NULL);
    CHECK(unicc.comm_destroy == NULL);
    CHECK(unicc.comm_count == NULL);
    CHECK(unicc.comm_user_rank == NULL);
    CHECK(unicc.group_start == NULL);
    CHECK(unicc.group_end == NULL);

    /* Full NCCL-shaped fake: init fills every slot. */
    if (fake_nccl) {
        unicc_lib_handle_t h = NULL;
        CHECK(unicc_loader_load(fake_nccl, &h) == UNICC_OK);
        CHECK(unicc_vtable_init(h) == UNICC_OK);
        CHECK(unicc_get_backend_type() == UNICC_BACKEND_NCCL);
        CHECK(unicc.get_version != NULL);
        CHECK(unicc.comm_init_rank != NULL);
        CHECK(unicc.allreduce != NULL);
        CHECK(unicc.broadcast != NULL);
        CHECK(unicc.get_unique_id != NULL);
        CHECK(unicc.comm_destroy != NULL);
        CHECK(unicc.comm_count != NULL);
        CHECK(unicc.comm_user_rank != NULL);
        CHECK(unicc.group_start != NULL);
        CHECK(unicc.group_end != NULL);

        unicc_vtable_cleanup();
        CHECK(unicc.get_version == NULL);
        CHECK(unicc.allreduce == NULL);
        CHECK(unicc.group_end == NULL);
        unicc_loader_unload(h);
    }

    /* Missing-symbol variant: optional slot stays NULL (degrade pattern);
     * core symbols are still present so vtable init succeeds. */
    if (fake_nccl_missing) {
        unicc_lib_handle_t h = NULL;
        CHECK(unicc_loader_load(fake_nccl_missing, &h) == UNICC_OK);
        CHECK(unicc_vtable_init(h) == UNICC_OK);
        CHECK(unicc.allreduce != NULL);
        CHECK(unicc.group_start != NULL);
        CHECK(unicc.group_end == NULL);   /* ncclGroupEnd omitted by the fixture */
        unicc_vtable_cleanup();
        unicc_loader_unload(h);
    }

    if (g_failures == 0) {
        printf("test_vtable: ALL TESTS PASSED\n");
        return 0;
    }
    printf("test_vtable: %d FAILURE(S)\n", g_failures);
    return 1;
}
