#ifndef XCC_ERRORS_H
#define XCC_ERRORS_H

/* Unified result codes returned by the xcc_* semantic API. Backend results
 * (NCCL/RCCL use compatible int codes where 0 == success) are mapped onto
 * these inside the implementation; XCCL never leaks a backend's raw codes at
 * the public boundary. The raw value of the last failed backend call is kept
 * and surfaced on demand through xcc_get_last_error. */
typedef enum {
    XCC_OK = 0,
    XCC_ERR_NO_BACKEND = -1,            /* no backend library could be located */
    XCC_ERR_BACKEND_LOAD = -2,          /* dlopen failed (see xcc_platform_dlerror) */
    XCC_ERR_BACKEND_INIT_FAILED = -3,   /* vtable init failed for a known backend */
    XCC_ERR_SYMBOL_NOT_FOUND = -4,      /* a required core symbol is missing */
    XCC_ERR_NOT_INITIALIZED = -5,       /* xcc_* called before xcc_init */
    XCC_ERR_ALREADY_INITIALIZED = -6,   /* xcc_init called twice */
    XCC_ERR_FINALIZED = -7,             /* xcc_* called after xcc_finalize */
    XCC_ERR_INVALID_ARGUMENT = -8,
    XCC_ERR_NOT_SUPPORTED = -9,         /* backend does not export the symbol (slot NULL) */
    XCC_ERR_UNHANDLED_BACKEND = -10,    /* backend returned a non-zero native result */
    XCC_ERR_BACKEND_NOT_SUPPORTED = -11,/* backend type unsupported on this platform */
    XCC_ERR_INVALID_STATE = -12,
    XCC_ERR_OUT_OF_MEMORY = -13
} xcc_result_t;

/* Human-readable description of a unified result code. */
const char* xcc_error_string(xcc_result_t rc);

#endif /* XCC_ERRORS_H */
