/* xcc_error.c - unified result code descriptions. */
#include "xcc_errors.h"

const char* xcc_error_string(xcc_result_t rc) {
    switch (rc) {
        case XCC_OK: return "success";
        case XCC_ERR_NO_BACKEND: return "no collective-communication backend library could be located";
        case XCC_ERR_BACKEND_LOAD: return "failed to load backend library";
        case XCC_ERR_BACKEND_INIT_FAILED: return "backend vtable initialization failed";
        case XCC_ERR_SYMBOL_NOT_FOUND: return "a required core symbol is missing from the backend library";
        case XCC_ERR_NOT_INITIALIZED: return "xcc not initialized (call xcc_init first)";
        case XCC_ERR_ALREADY_INITIALIZED: return "xcc already initialized";
        case XCC_ERR_FINALIZED: return "xcc has been finalized";
        case XCC_ERR_INVALID_ARGUMENT: return "invalid argument";
        case XCC_ERR_NOT_SUPPORTED: return "operation not supported by the active backend (symbol missing)";
        case XCC_ERR_UNHANDLED_BACKEND: return "backend returned a non-zero native result (see xcc_get_last_error)";
        case XCC_ERR_BACKEND_NOT_SUPPORTED: return "backend type is not supported on this platform";
        case XCC_ERR_INVALID_STATE: return "invalid internal state";
        case XCC_ERR_OUT_OF_MEMORY: return "out of memory";
    }
    return "unknown error code";
}
