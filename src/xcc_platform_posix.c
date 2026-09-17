/* xcc_platform_posix.c - dl* abstraction for POSIX (Linux/macOS). */
#include "xcc_platform.h"
#include <dlfcn.h>

xcc_lib_handle_t xcc_platform_dlopen(const char *path) {
    return dlopen(path, RTLD_NOW | RTLD_GLOBAL);
}

void xcc_platform_dlclose(xcc_lib_handle_t handle) {
    if (handle) {
        dlclose(handle);
    }
}

void* xcc_platform_dlsym(xcc_lib_handle_t handle, const char *symbol) {
    return dlsym(handle, symbol);
}

const char* xcc_platform_dlerror(void) {
    return dlerror();
}

const char* xcc_platform_load_advice(void) {
#ifdef __APPLE__
    return "Troubleshooting:\n"
           "1. Check the library exists: ls -la <library_path>\n"
           "2. Check library dependencies: otool -L <library_path>\n"
           "3. Ensure DYLD_LIBRARY_PATH includes the GPU library directory\n"
           "4. Verify NCCL/RCCL is installed for the target GPU platform\n";
#else
    return "Troubleshooting:\n"
           "1. Check the library exists: ls -la <library_path>\n"
           "2. Check library dependencies: ldd <library_path>\n"
           "3. List cached libraries: ldconfig -p | grep -E 'nccl|rccl'\n"
           "4. Ensure LD_LIBRARY_PATH includes the GPU library directory\n"
           "5. Verify NCCL/RCCL is installed for the target GPU platform\n";
#endif
}
