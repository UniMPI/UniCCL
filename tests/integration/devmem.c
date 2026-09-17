/* devmem.c - runtime-loaded CUDA/HIP device-memory adapter. */
#include "devmem.h"
#include <dlfcn.h>
#include <stdio.h>

typedef int (*malloc_fn)(void **, size_t);
typedef int (*memcpy_fn)(void *, const void *, size_t, int);
typedef int (*memset_fn)(void *, int, size_t);
typedef int (*free_fn)(void *);

static int load_runtime(const char *const *libs, const char *prefix,
                        xcc_devmem_t *api) {
    void *h = NULL;
    for (int i = 0; libs[i] && !h; i++) {
        h = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
    }
    if (!h) {
        fprintf(stderr, "[devmem] no device runtime found (%s): %s\n",
                prefix, dlerror());
        return -1;
    }

    char sym[64];
    snprintf(sym, sizeof(sym), "%sMalloc", prefix);
    api->dev_malloc  = (malloc_fn)dlsym(h, sym);
    snprintf(sym, sizeof(sym), "%sMemcpy", prefix);
    api->dev_memcpy  = (memcpy_fn)dlsym(h, sym);
    snprintf(sym, sizeof(sym), "%sMemset", prefix);
    api->dev_memset  = (memset_fn)dlsym(h, sym);
    snprintf(sym, sizeof(sym), "%sFree", prefix);
    api->dev_free    = (free_fn)dlsym(h, sym);
    api->runtime = prefix;

    if (!api->dev_malloc || !api->dev_memcpy || !api->dev_memset || !api->dev_free) {
        fprintf(stderr, "[devmem] %s runtime missing required symbols\n", prefix);
        return -1;
    }
    return 0;
}

int xcc_devmem_load(xcc_devmem_t *api) {
    const char *cuda_libs[] = {"libcudart.so", "libcudart.so.13",
                               "libcudart.so.12", NULL};
    const char *hip_libs[] = {"libamdhip64.so", "libamdhip64.so.6",
                              "libamdhip64.so.5", NULL};
    if (load_runtime(cuda_libs, "cuda", api) == 0) {
        return 0;
    }
    if (load_runtime(hip_libs, "hip", api) == 0) {
        return 0;
    }
    return -1;
}
