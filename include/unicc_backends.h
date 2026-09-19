#ifndef UNICC_BACKENDS_H
#define UNICC_BACKENDS_H

#include <stddef.h>

#include "unicc_platform.h"
#include "unicc_errors.h"

/* Backend types. The CCL vendor prefixes below are the symbol families this
 * wrapper loads and binds (see docs/BACKENDS.md). */
typedef enum {
    UNICC_BACKEND_UNKNOWN = 0,
    UNICC_BACKEND_NCCL,    /* NVIDIA NCCL          (nccl*)   */
    UNICC_BACKEND_RCCL,    /* AMD RCCL, defensive  (rccl + nccl family) */
    UNICC_BACKEND_ONECCL,  /* Intel oneCCL v2      (oneccl*) */
    UNICC_BACKEND_ECCL     /* Enflame ECCL         (eccl*)   */
    /* P2 adds: UNICC_BACKEND_CNCL, UNICC_BACKEND_HCCL,
     *          UNICC_BACKEND_MCCL_MUSA, UNICC_BACKEND_MCCL_METAX */
} unicc_backend_type_t;

/* Which native bootstrap-id typedef a family's GetUniqueId / CommInitRank take.
 * Only two exist (NCCL-family 128 B, Intel oneCCL 4096 B); they are kept
 * distinct because the binder's dlsym casts must match the target signature
 * exactly (-fsanitize=function; the fixtures share the same typedefs). */
typedef enum {
    UNICC_UID_NCCL = 0,   /* unicc_native_uid_t (128 B) */
    UNICC_UID_ONECCL      /* unicc_native_uid_oneccl_t (4096 B) */
} unicc_uid_kind_t;

/* Per-family descriptor: the single source of truth for identification,
 * binding, diagnostics and (future) default selection. Every field is a static
 * fact about the vendor's symbol family (see docs/BACKENDS.md). */
typedef struct {
    unicc_backend_type_t type;
    const char *name;                /* symbol prefix + UNICC_BACKEND value ("nccl") */
    const char *lib_name;            /* preferred library name to dlopen */
    const char *lib_name_alt;        /* fallback when lib_name fails (NULL = none) */
    const char *probe_symbol;        /* identifying symbol (unicc_loader_identify_backend) */
    int priority;                    /* identify/select order: higher is probed first */
    size_t uid_bytes;                /* bootstrap id size: 128 / 4096 */
    unicc_uid_kind_t uid_kind;       /* native id typedef for the dlsym casts */
    int nranks_is_size_t;            /* CommInitRank nranks is size_t (oneCCL/ECCL) */
    int broadcast_double_buffered;   /* Broadcast is send+recv (oneCCL/ECCL), else in-place */
    int has_comm_user_rank;          /* 0 == ECCL leaves the slot NULL (no symbol) */
} unicc_backend_info_t;

/* Backend detection and loading API (mirrors unimpi_loader.h). Selection is
 * deterministic: UNICC_LIBRARY (exact path) -> UNICC_BACKEND (name) -> default. */
/* Type of the backend identified after the last successful unicc_vtable_init. */
unicc_backend_type_t unicc_get_backend_type(void);

const char* unicc_loader_get_env_backend(void);
const char* unicc_loader_get_env_libpath(void);
int unicc_loader_detect_backend(const char **out_lib_path);
/* dlopen lib_path with the family's soname fallback. On success (and when
 * resolved_path is non-NULL and resolved_cap > 0), the name that actually
 * dlopen'd (lib_path or the fallback) is written to resolved_path. */
int unicc_loader_load(const char *lib_path, unicc_lib_handle_t *out_handle,
                      char *resolved_path, size_t resolved_cap);
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
#define UNICC_MAX_BACKENDS 4
extern const unicc_backend_info_t unicc_backends[UNICC_MAX_BACKENDS];

#endif /* UNICC_BACKENDS_H */
