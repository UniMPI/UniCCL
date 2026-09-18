/* src/unicc_native_id.h - native unique-id types shared by the binders and
 * the fake fixtures.
 *
 * UniCCL never includes vendor headers, so both the binders (src/backends/)
 * and the fake fixtures (tests/fake_backends/) need a concrete layout for a
 * vendor's fixed-size unique id. Both sides must use the *same* type: under
 * -fsanitize=function (UBSan) a call through a function pointer whose
 * declared parameter type differs from the exported function's real type is
 * undefined behavior even when the layouts are identical. Including this one
 * header gives both sides an identical typedef; the layouts match the vendor
 * structs exactly (128 B NCCL ncclUniqueId, 4096 B Intel onecclUniqueId), so
 * the ABI against real libraries is unchanged. Internal only - not installed.
 */
#ifndef UNICC_NATIVE_ID_H
#define UNICC_NATIVE_ID_H

#include "unicc_version.h"   /* UNICC_UNIQUE_ID_BYTES (=128) */

/* Intel oneCCL v2 unique id is 4096 bytes (the NCCL family uses 128). */
#define UNICC_ONECCL_UNIQUE_ID_BYTES 4096

/* NCCL family: NVIDIA NCCL ncclUniqueId, RCCL and Hygon DCU (both nccl*-
 * compat), Enflame ECCL. */
typedef struct { char internal[UNICC_UNIQUE_ID_BYTES]; } unicc_native_uid_t;

/* Intel oneCCL v2 C API. */
typedef struct { char internal[UNICC_ONECCL_UNIQUE_ID_BYTES]; } unicc_native_uid_oneccl_t;

#endif /* UNICC_NATIVE_ID_H */
