#ifndef XCC_BACKENDS_H
#define XCC_BACKENDS_H

#include "xcc_platform.h"
#include "xcc_errors.h"

/* Backend types. */
typedef enum {
    XCC_BACKEND_UNKNOWN = 0,
    XCC_BACKEND_NCCL,
    XCC_BACKEND_RCCL
} xcc_backend_type_t;

/* Backend description table entry. */
typedef struct {
    xcc_backend_type_t type;
    const char *name;          /* value accepted by XCCL_BACKEND */
    const char *lib_name;      /* preferred library name to dlopen */
    const char *lib_name_alt;  /* fallback when lib_name fails (NULL = none) */
    int priority;              /* future use: platform-default ordering */
} xcc_backend_info_t;

/* Backend detection and loading API (mirrors unimpi_loader.h). Selection is
 * deterministic: XCCL_LIBRARY (exact path) -> XCCL_BACKEND (name) -> default. */
/* Type of the backend identified after the last successful xcc_vtable_init. */
xcc_backend_type_t xcc_get_backend_type(void);

const char* xcc_loader_get_env_backend(void);
const char* xcc_loader_get_env_libpath(void);
int xcc_loader_detect_backend(const char **out_lib_path);
int xcc_loader_load(const char *lib_path, xcc_lib_handle_t *out_handle);
void xcc_loader_unload(xcc_lib_handle_t handle);
xcc_backend_type_t xcc_loader_identify_backend(xcc_lib_handle_t handle);

/* Platform support check for a backend type. */
int xcc_loader_check_platform_support(xcc_backend_type_t backend);

/* Diagnostics: symbol coverage against a concrete library, plus the backend
 * table. Writing a precise path to stderr is very useful on hosts that do not
 * have NCCL/RCCL installed (see docs/BACKENDS.md). */
void xcc_diagnose_backend(const char *lib_path);
int xcc_print_backend_info(void);

/* Known backends. */
#define XCC_MAX_BACKENDS 2
extern const xcc_backend_info_t xcc_backends[XCC_MAX_BACKENDS];

#endif /* XCC_BACKENDS_H */
