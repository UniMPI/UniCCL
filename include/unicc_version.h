#ifndef UNICC_VERSION_H
#define UNICC_VERSION_H

/* UniCCL library version. This is the unified wrapper's own release version and
 * is kept separate from any backend's version (see unicc_backend_version in
 * unicc.h). */
#define UNICC_VERSION_MAJOR 0
#define UNICC_VERSION_MINOR 2
#define UNICC_VERSION_PATCH 0
#define UNICC_VERSION_SUFFIX "-alpha"

#define UNICC_VERSION_STRING "0.2.0-alpha"

/* Stable backend ABI facts shared by the wrapper, the fixtures and the docs.
 *
 * UNICC_UNIQUE_ID_BYTES is the NCCL-family unique-id size (NCCL defines
 * NCCL_UNIQUE_ID_BYTES = 128; RCCL / Hygon DCU and MCCL share it).
 * UNICC_COMM_ID_MAX is the largest bootstrap id UniCCL must be able to hold:
 * NCCL-family 128, Cambricon CNCL cliqueId 136, Intel oneCCL uniqueId 4096,
 * Ascend HCCL rootInfo 4108. 4112 leaves a little headroom and is 8-byte
 * aligned. The public comm-id type (unicc_comm_id_t) carries its length
 * explicitly because vendors disagree on id size; the NCCL-family binding
 * still uses UNICC_UNIQUE_ID_BYTES. */
#define UNICC_UNIQUE_ID_BYTES 128
#define UNICC_COMM_ID_MAX 4112

#endif /* UNICC_VERSION_H */
