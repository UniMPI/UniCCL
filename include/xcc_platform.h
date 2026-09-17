#ifndef XCC_PLATFORM_H
#define XCC_PLATFORM_H

/* Platform detection. */
#ifdef _WIN32
    #define XCC_WINDOWS
#else
    #define XCC_POSIX
#endif

/* Dynamic library handle. */
#ifdef XCC_WINDOWS
    #include <windows.h>
    typedef HMODULE xcc_lib_handle_t;
#else
    typedef void* xcc_lib_handle_t;
#endif

/* Platform dl* abstraction. The loader and every backend binding goes through
 * these so that only this file (and its per-platform implementation) knows
 * anything about dlopen/LoadLibrary. Modeled on unimpi_platform.h. */
xcc_lib_handle_t xcc_platform_dlopen(const char *path);
void xcc_platform_dlclose(xcc_lib_handle_t handle);
void* xcc_platform_dlsym(xcc_lib_handle_t handle, const char *symbol);
const char* xcc_platform_dlerror(void);
const char* xcc_platform_load_advice(void);

#endif /* XCC_PLATFORM_H */
