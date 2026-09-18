/* test_integration.c - real-backend (NCCL / RCCL) end-to-end test.
 *
 * Built only with UNICC_BUILD_TESTS_INTEGRATION=ON. Requires a GPU with NCCL
 * (NVIDIA) or RCCL (AMD) and the corresponding runtime, e.g. the tf-builder
 * ubuntu-24.04-gcc13-cuda / -rocm images or a GPU host. The backend itself is
 * chosen exactly like production, via UNICC_BACKEND / UNICC_LIBRARY.
 *
 * World layout comes from the environment, so no MPI is needed:
 *   UNICC_TEST_WORLD_SIZE  number of ranks (default 1)
 *   UNICC_TEST_WORLD_RANK  this process's rank   (default 0)
 *   UNICC_TEST_UID_FILE    file rank 0 uses to publish the unique id
 *                         (required when size > 1)
 *
 * Each rank sends its own distinct vector; the SUM allreduce must return the
 * analytic sum over all ranks. Device buffers come from unicc_devmem (devmem.c),
 * preserving the no-vendor-header build guarantee. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "unicc.h"
#include "devmem.h"

static const char* env_or(const char *name, const char *dflt) {
    const char *v = getenv(name);
    return (v && v[0]) ? v : dflt;
}

int main(void) {
    int world = atoi(env_or("UNICC_TEST_WORLD_SIZE", "1"));
    int rank  = atoi(env_or("UNICC_TEST_WORLD_RANK", "0"));
    const char *uid_file = getenv("UNICC_TEST_UID_FILE");

    if (world < 1 || rank < 0 || rank >= world) {
        fprintf(stderr, "bad world config: size=%d rank=%d\n", world, rank);
        return 2;
    }

    unicc_result_t rc = unicc_init();
    if (rc != UNICC_OK) {
        fprintf(stderr, "unicc_init failed: %s\n", unicc_error_string(rc));
        return 2;
    }
    printf("[rank %d] backend=%s library=%s\n", rank, unicc_backend_name(),
           unicc_get_library_path());

    int bv = 0;
    unicc_backend_version(&bv);

    /* Bootstrap the unique id: rank 0 publishes (id.data + id.len), others
     * poll and consume it. The id carries its length so any of the supported
     * id sizes (128B NCCL .. 4KiB oneCCL/HCCL) round-trips unchanged. */
    unicc_comm_id_t id;
    if (rank == 0) {
        rc = unicc_get_unique_id(&id);
        if (rc != UNICC_OK) {
            fprintf(stderr, "unicc_get_unique_id failed: %s\n", unicc_error_string(rc));
            return 2;
        }
        if (world > 1) {
            if (!uid_file) {
                fprintf(stderr, "UNICC_TEST_UID_FILE required when size > 1\n");
                return 2;
            }
            FILE *f = fopen(uid_file, "w");
            if (!f || fwrite(id.data, 1, id.len, f) != id.len) {
                fprintf(stderr, "failed to publish id\n");
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
        if (!f) {
            fprintf(stderr, "failed to open published id file\n");
            return 2;
        }
        id.len = fread(id.data, 1, UNICC_COMM_ID_MAX, f);
        if (id.len == 0 || id.len > UNICC_COMM_ID_MAX) {
            fprintf(stderr, "failed to read published id\n");
            return 2;
        }
        fclose(f);
    }

    unicc_devmem_t dm;
    if (unicc_devmem_load(&dm) != 0) {
        fprintf(stderr, "no CUDA/HIP runtime available\n");
        return 2;
    }

    /* Bind each rank to its own device BEFORE communicator bootstrap: NCCL
     * refuses two ranks of one communicator on the same GPU, and without an
     * explicit cudaSetDevice every process defaults to device 0. The launcher
     * must expose at least `world` GPUs. */
    int gpu_count = 0;
    if (dm.dev_get_device_count(&gpu_count) != 0 || gpu_count < 1) {
        fprintf(stderr, "device count query failed\n");
        return 2;
    }
    if (world > gpu_count) {
        fprintf(stderr, "world=%d exceeds visible GPUs=%d (bind one rank per GPU)\n",
                world, gpu_count);
        return 2;
    }
    if (dm.dev_set_device(rank) != 0) {
        fprintf(stderr, "cudaSetDevice(%d) failed\n", rank);
        return 2;
    }

    unicc_comm_t comm = NULL;
    rc = unicc_comm_init_rank(&comm, world, &id, rank);
    if (rc != UNICC_OK) {
        fprintf(stderr, "unicc_comm_init_rank failed: %s\n", unicc_error_string(rc));
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

    dm.dev_memcpy((void*)d_send, h_send, n * sizeof(float), UNICC_DEVMEM_H2D);
    dm.dev_memset((void*)d_recv, 0, n * sizeof(float));

    rc = unicc_allreduce(d_send, d_recv, n, UNICC_F32, UNICC_SUM, comm, NULL);
    if (rc != UNICC_OK) {
        fprintf(stderr, "unicc_allreduce failed: %s\n", unicc_error_string(rc));
        return 2;
    }

    dm.dev_memcpy(h_recv, (const void*)d_recv, n * sizeof(float), UNICC_DEVMEM_D2H);

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
    if (unicc_comm_count(comm, &count) != UNICC_OK || count != world) {
        fprintf(stderr, "comm count mismatch\n");
        return 2;
    }

    dm.dev_free(d_send);
    dm.dev_free(d_recv);
    if (unicc_comm_destroy(comm) != UNICC_OK) return 2;

    printf("[rank %d] PASS (backend=%s version=%d sum=%.0f)\n",
           rank, unicc_backend_name(), bv, expected);
    rc = unicc_finalize();
    return rc == UNICC_OK ? 0 : 2;
}
