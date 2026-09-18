/* unicc_platform_windows.c - dl* abstraction for Windows (LoadLibrary family).
 * Compiled and exercised only on Windows; kept for parity with the POSIX
 * implementation so the plan for future Windows NCCL support stays intact. */
#include "unicc_platform.h"
#include <stdio.h>

unicc_lib_handle_t unicc_platform_dlopen(const char *path) {
    return LoadLibraryA(path);
}

void unicc_platform_dlclose(unicc_lib_handle_t handle) {
    if (handle) {
        FreeLibrary(handle);
    }
}

void* unicc_platform_dlsym(unicc_lib_handle_t handle, const char *symbol) {
    return (void*)GetProcAddress(handle, symbol);
}

const char* unicc_platform_dlerror(void) {
    static char buf[256];
    snprintf(buf, sizeof(buf), "GetLastError() = %lu", (unsigned long)GetLastError());
    return buf;
}

const char* unicc_platform_load_advice(void) {
    return "Troubleshooting:\n"
           "1. Check the DLL exists and its path is on PATH\n"
           "2. Check dependencies with dumpbin /dependents\n"
           "3. Verify NCCL (or an NCCL DLL) supports Windows\n";
}
