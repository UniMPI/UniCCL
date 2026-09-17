#ifndef XCC_VERSION_H
#define XCC_VERSION_H

/* XCCL library version. This is the unified wrapper's own release version and
 * is kept separate from any backend's version (see xcc_backend_version in
 * xcc.h). */
#define XCC_VERSION_MAJOR 0
#define XCC_VERSION_MINOR 1
#define XCC_VERSION_PATCH 0
#define XCC_VERSION_SUFFIX "-alpha"

#define XCC_VERSION_STRING "0.1.0-alpha"

/* Stable backend ABI facts shared by the wrapper, the fixtures and the docs.
 * NVIDIA NCCL defines NCCL_UNIQUE_ID_BYTES = 128 and every ncclUniqueId is
 * exactly that many bytes; AMD RCCL is ABI-compatible and uses the same
 * value, so xcc_unique_id_t round-trips unchanged through both. */
#define XCC_UNIQUE_ID_BYTES 128

#endif /* XCC_VERSION_H */
