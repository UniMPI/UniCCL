/* test_integration.c - real-backend (NCCL / RCCL) end-to-end test.
 *
 * Built only with XCCL_BUILD_TESTS_INTEGRATION=ON. Requires a GPU with NCCL
 * (NVIDIA) or RCCL (AMD) and the corresponding runtime, e.g. the tf-builder
 * ubuntu-24.04-gcc13-cuda / -rocm images or a GPU host. The backend itself is
 * chosen exactly like production, via XCCL_BACKEND / XCCL_LIBRARY.
 *
 * World layout comes from the environment, so no MPI is needed:
 *   XCCL_TEST_WORLD_SIZE  number of ranks (default 1)
 *   XCCL_TEST_WORLD_RANK  this process's rank   (default 0)
 *   XCCL_TEST_UID_FILE    file rank 0 uses to publish the unique id
 *                         (required when size > 1)
 *
 * Each rank sends its own distinct vector; the SUM allreduce must return the
 * analytic sum over all ranks. Device buffers come from xcc_devmem (devmem.c),
 * preserving the no-vendor-header build guarantee. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "xcc.h"
#include "devmem.h"

static const char* env_or(const char *name, const char *dflt) {
    const char *v = getenv(name);
    return (v && v[0]) ? v : dflt;
}

int main(void) {
    int world = atoi(env_or("XCCL_TEST_WORLD_SIZE", "1"));
    int rank  = atoi(env_or("XCCL_TEST_WORLD_RANK", "0"));
    const char *uid_file = getenv("XCCL_TEST_UID_FILE");

    if (world < 1 || rank < 0 || rank >= world) {
        fprintf(stderr, "bad world config: size=%d rank=%d\n", world, rank);
        return 2;
    }

    xcc_result_t rc = xcc_init();
    if (rc != XCC_OK) {
        fprintf(stderr, "xcc_init failed: %s\n", xcc_error_string(rc));
        return 2;
    }
    printf("[rank %d] backend=%s library=%s\n", rank, xcc_backend_name(),
           xcc_get_library_path());

    int bv = 0;
    xcc_backend_version(&bv);

    /* Bootstrap the unique id: rank 0 publishes, others poll for it. */
    xcc_unique_id_t uid;
    if (rank == 0) {
        rc = xcc_get_unique_id(&uid);
        if (rc != XCC_OK) {
            fprintf(stderr, "xcc_get_unique_id failed: %s\n", xcc_error_string(rc));
            return 2;
        }
        if (world > 1) {
            if (!uid_file) {
                fprintf(stderr, "XCCL_TEST_UID_FILE required when size > 1\n");
                return 2;
            }
            FILE *f = fopen(uid_file, "w");
            if (!f || fwrite(uid.data, 1, XCC_UNIQUE_ID_BYTES, f) != XCC_UNIQUE_ID_BYTES) {
                fprintf(stderr, "failed to publish uid\n");
                return 2;
            }
            fclose(f);
        }
    } else {
        FILE *f = NULL;
        for (int tries = 0; tries < 3000 && !f; tries++) {
            f = fopen(uid_file, "r");
            if (!f) usleep(10000);
        }
        if (!f || fread(uid.data, 1, XCC_UNIQUE_ID_BYTES, f) != XCC_UNIQUE_ID_BYTES) {
            fprintf(stderr, "failed to read published uid\n");
            return 2;
        }
        fclose(f);
    }

    xcc_comm_t comm = NULL;
    rc = xcc_comm_init_rank(&comm, world, uid, rank);
    if (rc != XCC_OK) {
        fprintf(stderr, "xcc_comm_init_rank failed: %s\n", xcc_error_string(rc));
        return 2;
    }

    xcc_devmem_t dm;
    if (xcc_devmem_load(&dm) != 0) {
        fprintf(stderr, "no CUDA/HIP runtime available\n");
        return 2;
    }

    const size_t n = 4;
    float *d_send = NULL, *d_recv = NULL;
    if (dm.dev_malloc((void**)&d_send, n * sizeof(float)) != 0 ||
        dm.dev_malloc((void**)&d_recv, n * sizeof(float)) != 0) {
        fprintf(stderr, "device malloc failed\n");
        return 2;
    }

    float h_send[4], h_recv[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < n; i++) h_send[i] = (float)(rank + 1);

    dm.dev_memcpy((void*)d_send, h_send, n * sizeof(float), XCC_DEVMEM_H2D);
    dm.dev_memset((void*)d_recv, 0, n * sizeof(float));

    rc = xcc_allreduce(d_send, d_recv, n, XCC_F32, XCC_SUM, comm, NULL);
    if (rc != XCC_OK) {
        fprintf(stderr, "xcc_allreduce failed: %s\n", xcc_error_string(rc));
        return 2;
    }

    dm.dev_memcpy(h_recv, (const void*)d_recv, n * sizeof(float), XCC_DEVMEM_D2H);

    /* Expected SUM over ranks: 1 + 2 + ... + world = world(world+1)/2. */
    float expected = (float)(world) * (float)(world + 1) / 2.0f;
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        if (h_recv[i] != expected) {
            fprintf(stderr, "[rank %d] mismatch at %zu: got %f want %f\n",
                    rank, i, h_recv[i], expected);
            ok = 0;
        }
    }
    if (!ok) return 2;

    int count = 0;
    if (xcc_comm_count(comm, &count) != XCC_OK || count != world) {
        fprintf(stderr, "comm count mismatch\n");
        return 2;
    }

    dm.dev_free(d_send);
    dm.dev_free(d_recv);
    if (xcc_comm_destroy(comm) != XCC_OK) return 2;

    printf("[rank %d] PASS (backend=%s version=%d sum=%.0f)\n",
           rank, xcc_backend_name(), bv, expected);
    rc = xcc_finalize();
    return rc == XCC_OK ? 0 : 2;
}
