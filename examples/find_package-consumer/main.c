/* main.c - find_package consumer smoke test.
 *
 * The point is not to be rich: proving find_package(UniCCL CONFIG) +
 * link unicc::unicc compiles AND links AND initializes the wrapper is the
 * whole smoke.
 *
 * On a host without a real backend, point UNICC_LIBRARY at a fake fixture
 * (UNICC_LIBRARY=<path>/fake_nccl_identity.so) so unicc_init has a real
 * library to bring up; see the top-level README "Quick local run".
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

    printf("find_package consumer: version=%s backend=%s backend_version=0x%06x "
           "available(allreduce)=%d\n",
           ver, unicc_backend_name(), bv, unicc_allreduce_available());

    unicc_finalize();
    return 0;
}
