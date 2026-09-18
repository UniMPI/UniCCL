#ifndef UNICC_BACKENDS_H
#define UNICC_BACKENDS_H

#include "unicc_platform.h"
#include "unicc_errors.h"

/* Backend types. */
typedef enum {
    UNICC_BACKEND_UNKNOWN = 0,
    UNICC_BACKEND_NCCL,
    UNICC_BACKEND_RCCL
} unicc_backend_type_t;

/* Backend description table entry. */
typedef struct {
    unicc_backend_type_t type;
    const char *name;          /* value accepted by UNICC_BACKEND */
    const char *lib_name;      /* preferred library name to dlopen */
    const char *lib_name_alt;  /* fallback when lib_name fails (NULL = none) */
    int priority;              /* future use: platform-default ordering */
} unicc_backend_info_t;

/* Backend detection and loading API (mirrors unimpi_loader.h). Selection is
 * deterministic: UNICC_LIBRARY (exact path) -> UNICC_BACKEND (name) -> default. */
/* Type of the backend identified after the last successful unicc_vtable_init. */
unicc_backend_type_t unicc_get_backend_type(void);

const char* unicc_loader_get_env_backend(void);
const char* unicc_loader_get_env_libpath(void);
int unicc_loader_detect_backend(const char **out_lib_path);
int unicc_loader_load(const char *lib_path, unicc_lib_handle_t *out_handle);
void unicc_loader_unload(unicc_lib_handle_t handle);
unicc_backend_type_t unicc_loader_identify_backend(unicc_lib_handle_t handle);

/* Platform support check for a backend type. */
int unicc_loader_check_platform_support(unicc_backend_type_t backend);

/* Diagnostics: symbol coverage against a concrete library, plus the backend
 * table. Writing a precise path to stderr is very useful on hosts that do not
 * have NCCL/RCCL installed (see docs/BACKENDS.md). */
void unicc_diagnose_backend(const char *lib_path);
int unicc_print_backend_info(void);

/* Known backends. */
#define UNICC_MAX_BACKENDS 2
extern const unicc_backend_info_t unicc_backends[UNICC_MAX_BACKENDS];

#endif /* UNICC_BACKENDS_H */
