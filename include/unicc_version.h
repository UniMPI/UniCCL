#ifndef UNICC_VERSION_H
#define UNICC_VERSION_H

/* UniCCL library version. This is the unified wrapper's own release version and
 * is kept separate from any backend's version (see unicc_backend_version in
 * unicc.h). */
#define UNICC_VERSION_MAJOR 0
#define UNICC_VERSION_MINOR 1
#define UNICC_VERSION_PATCH 0
#define UNICC_VERSION_SUFFIX "-alpha"

#define UNICC_VERSION_STRING "0.1.0-alpha"

/* Stable backend ABI facts shared by the wrapper, the fixtures and the docs.
 * NVIDIA NCCL defines NCCL_UNIQUE_ID_BYTES = 128 and every ncclUniqueId is
 * exactly that many bytes; AMD RCCL is ABI-compatible and uses the same
 * value, so unicc_unique_id_t round-trips unchanged through both. */
#define UNICC_UNIQUE_ID_BYTES 128

#endif /* UNICC_VERSION_H */
