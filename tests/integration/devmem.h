#ifndef XCC_DEVMEM_H
#define XCC_DEVMEM_H

#include <stddef.h>

/* Minimal device-memory adapter for the real-backend integration test.
 *
 * The whole project compiles with zero vendor headers; this tiny vtable
 * dlopens the CUDA or ROCm runtime at run time to allocate device buffers so
 * test_integration can hand XCCL real device pointers without ever including
 * cuda_runtime.h / hip_runtime.h. */

#ifdef __cplusplus
extern "C" {
#endif

/* Direction kinds match cudaMemcpyKind / hipMemcpyKind (both H2D=1, D2H=2). */
#define XCC_DEVMEM_H2D 1
#define XCC_DEVMEM_D2H 2

typedef struct {
    int  (*dev_malloc)(void **ptr, size_t bytes);
    int  (*dev_memcpy)(void *dst, const void *src, size_t bytes, int kind);
    int  (*dev_memset)(void *ptr, int value, size_t bytes);
    int  (*dev_free)(void *ptr);
    const char *runtime; /* "cuda", "hip", or NULL */
} xcc_devmem_t;

/* Load CUDA (preferred) or HIP runtime. Returns 0 on success. */
int xcc_devmem_load(xcc_devmem_t *api);

#ifdef __cplusplus
}
#endif

#endif /* XCC_DEVMEM_H */
