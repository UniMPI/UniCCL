#ifndef UNICC_PLATFORM_H
#define UNICC_PLATFORM_H

/* Platform detection. */
#ifdef _WIN32
    #define UNICC_WINDOWS
#else
    #define UNICC_POSIX
#endif

/* Dynamic library handle. */
#ifdef UNICC_WINDOWS
    #include <windows.h>
    typedef HMODULE unicc_lib_handle_t;
#else
    typedef void* unicc_lib_handle_t;
#endif

/* Platform dl* abstraction. The loader and every backend binding goes through
 * these so that only this file (and its per-platform implementation) knows
 * anything about dlopen/LoadLibrary. Modeled on unimpi_platform.h. */
unicc_lib_handle_t unicc_platform_dlopen(const char *path);
void unicc_platform_dlclose(unicc_lib_handle_t handle);
void* unicc_platform_dlsym(unicc_lib_handle_t handle, const char *symbol);
const char* unicc_platform_dlerror(void);
const char* unicc_platform_load_advice(void);

#endif /* UNICC_PLATFORM_H */
