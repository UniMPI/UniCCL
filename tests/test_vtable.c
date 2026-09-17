/* test_vtable.c - vtable layout, zero-init, core validation and binding.
 *
 * Verifies the dispatch table contract: it starts fully NULL, a full backend
 * fills every slot, a backend missing an optional symbol leaves that slot
 * NULL (the degrade pattern), and cleanup returns to full zeros. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xcc_vtable.h"
#include "xcc_backends.h"

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
    CHECK(xcc.get_version == NULL);
    CHECK(xcc.comm_init_rank == NULL);
    CHECK(xcc.allreduce == NULL);
    CHECK(xcc.broadcast == NULL);
    CHECK(xcc.get_unique_id == NULL);
    CHECK(xcc.comm_destroy == NULL);
    CHECK(xcc.comm_count == NULL);
    CHECK(xcc.comm_user_rank == NULL);
    CHECK(xcc.group_start == NULL);
    CHECK(xcc.group_end == NULL);

    /* Full NCCL-shaped fake: init fills every slot. */
    if (fake_nccl) {
        xcc_lib_handle_t h = NULL;
        CHECK(xcc_loader_load(fake_nccl, &h) == XCC_OK);
        CHECK(xcc_vtable_init(h) == XCC_OK);
        CHECK(xcc_get_backend_type() == XCC_BACKEND_NCCL);
        CHECK(xcc.get_version != NULL);
        CHECK(xcc.comm_init_rank != NULL);
        CHECK(xcc.allreduce != NULL);
        CHECK(xcc.broadcast != NULL);
        CHECK(xcc.get_unique_id != NULL);
        CHECK(xcc.comm_destroy != NULL);
        CHECK(xcc.comm_count != NULL);
        CHECK(xcc.comm_user_rank != NULL);
        CHECK(xcc.group_start != NULL);
        CHECK(xcc.group_end != NULL);

        xcc_vtable_cleanup();
        CHECK(xcc.get_version == NULL);
        CHECK(xcc.allreduce == NULL);
        CHECK(xcc.group_end == NULL);
        xcc_loader_unload(h);
    }

    /* Missing-symbol variant: optional slot stays NULL (degrade pattern);
     * core symbols are still present so vtable init succeeds. */
    if (fake_nccl_missing) {
        xcc_lib_handle_t h = NULL;
        CHECK(xcc_loader_load(fake_nccl_missing, &h) == XCC_OK);
        CHECK(xcc_vtable_init(h) == XCC_OK);
        CHECK(xcc.allreduce != NULL);
        CHECK(xcc.group_start != NULL);
        CHECK(xcc.group_end == NULL);   /* ncclGroupEnd omitted by the fixture */
        xcc_vtable_cleanup();
        xcc_loader_unload(h);
    }

    if (g_failures == 0) {
        printf("test_vtable: ALL TESTS PASSED\n");
        return 0;
    }
    printf("test_vtable: %d FAILURE(S)\n", g_failures);
    return 1;
}
