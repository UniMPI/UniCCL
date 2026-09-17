/* minimal.c - smallest XCCL usage example.
 *
 * Compiles with no NCCL/RCCL/CUDA/ROCm headers (include/xcc.h is enough).
 * On a host without a real backend, point XCCL_LIBRARY at one of the fake
 * fixtures and this example drives the stand-in end to end:
 *
 *   mkdir -p build && cmake -S . -B build && cmake --build build
 *   XCCL_LIBRARY=build/tests/fake/fake_nccl_identity.so ./build/minimal
 *   XCCL_LIBRARY=build/tests/fake/fake_rccl_identity.so ./build/minimal
 *
 * On a GPU host: XCCL_BACKEND=nccl ./build/minimal
 */
#include <stdio.h>
#include "xcc.h"

int main(void) {
    xcc_result_t rc = xcc_init();
    if (rc != XCC_OK) {
        fprintf(stderr, "xcc_init failed: %s\n", xcc_error_string(rc));
        return 1;
    }

    char ver[64];
    int bv = 0;
    xcc_get_version(ver, sizeof(ver));
    xcc_backend_version(&bv);
    printf("%s | backend=%s version=0x%06x library=%s\n",
           ver, xcc_backend_name(), bv, xcc_get_library_path());

    if (xcc_comm_available()) {
        xcc_unique_id_t uid;
        if (xcc_get_unique_id(&uid) == XCC_OK) {
            xcc_comm_t comm = NULL;
            if (xcc_comm_init_rank(&comm, 1, uid, 0) == XCC_OK) {
                if (xcc_allreduce_available()) {
                    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    if (xcc_allreduce(in, out, 4, XCC_F32, XCC_SUM, comm, NULL) == XCC_OK) {
                        printf("allreduce(f32 sum, n=4): [%g %g %g %g]\n",
                               out[0], out[1], out[2], out[3]);
                    }
                }
                xcc_comm_destroy(comm);
            }
        }
    }

    xcc_finalize();
    return 0;
}
