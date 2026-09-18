/* minimal.c - smallest UniCCL usage example.
 *
 * Compiles with no NCCL/RCCL/CUDA/ROCm headers (include/unicc.h is enough).
 * On a host without a real backend, point UNICC_LIBRARY at one of the fake
 * fixtures and this example drives the stand-in end to end:
 *
 *   mkdir -p build && cmake -S . -B build && cmake --build build
 *   UNICC_LIBRARY=build/tests/fake/fake_nccl_identity.so ./build/minimal
 *   UNICC_LIBRARY=build/tests/fake/fake_rccl_identity.so ./build/minimal
 *
 * On a GPU host: UNICC_BACKEND=nccl ./build/minimal
 */
#include <stdio.h>
#include "unicc.h"

int main(void) {
    unicc_result_t rc = unicc_init();
    if (rc != UNICC_OK) {
        fprintf(stderr, "unicc_init failed: %s\n", unicc_error_string(rc));
        return 1;
    }

    char ver[64];
    int bv = 0;
    unicc_get_version(ver, sizeof(ver));
    unicc_backend_version(&bv);
    printf("%s | backend=%s version=0x%06x library=%s\n",
           ver, unicc_backend_name(), bv, unicc_get_library_path());

    if (unicc_comm_available()) {
        unicc_unique_id_t uid;
        if (unicc_get_unique_id(&uid) == UNICC_OK) {
            unicc_comm_t comm = NULL;
            if (unicc_comm_init_rank(&comm, 1, uid, 0) == UNICC_OK) {
                if (unicc_allreduce_available()) {
                    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    if (unicc_allreduce(in, out, 4, UNICC_F32, UNICC_SUM, comm, NULL) == UNICC_OK) {
                        printf("allreduce(f32 sum, n=4): [%g %g %g %g]\n",
                               out[0], out[1], out[2], out[3]);
                    }
                }
                unicc_comm_destroy(comm);
            }
        }
    }

    unicc_finalize();
    return 0;
}
