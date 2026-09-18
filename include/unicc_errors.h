#ifndef UNICC_ERRORS_H
#define UNICC_ERRORS_H

/* Unified result codes returned by the unicc_* semantic API. Backend results
 * (NCCL/RCCL use compatible int codes where 0 == success) are mapped onto
 * these inside the implementation; UniCCL never leaks a backend's raw codes at
 * the public boundary. The raw value of the last failed backend call is kept
 * and surfaced on demand through unicc_get_last_error. */
typedef enum {
    UNICC_OK = 0,
    UNICC_ERR_NO_BACKEND = -1,            /* no backend library could be located */
    UNICC_ERR_BACKEND_LOAD = -2,          /* dlopen failed (see unicc_platform_dlerror) */
    UNICC_ERR_BACKEND_INIT_FAILED = -3,   /* vtable init failed for a known backend */
    UNICC_ERR_SYMBOL_NOT_FOUND = -4,      /* a required core symbol is missing */
    UNICC_ERR_NOT_INITIALIZED = -5,       /* unicc_* called before unicc_init */
    UNICC_ERR_ALREADY_INITIALIZED = -6,   /* unicc_init called twice */
    UNICC_ERR_FINALIZED = -7,             /* unicc_* called after unicc_finalize */
    UNICC_ERR_INVALID_ARGUMENT = -8,
    UNICC_ERR_NOT_SUPPORTED = -9,         /* backend does not export the symbol (slot NULL) */
    UNICC_ERR_UNHANDLED_BACKEND = -10,    /* backend returned a non-zero native result */
    UNICC_ERR_BACKEND_NOT_SUPPORTED = -11,/* backend type unsupported on this platform */
    UNICC_ERR_INVALID_STATE = -12,
    UNICC_ERR_OUT_OF_MEMORY = -13
} unicc_result_t;

/* Human-readable description of a unified result code. */
const char* unicc_error_string(unicc_result_t rc);

#endif /* UNICC_ERRORS_H */
