/* unicc_error.c - unified result code descriptions. */
#include "unicc_errors.h"

const char* unicc_error_string(unicc_result_t rc) {
    switch (rc) {
        case UNICC_OK: return "success";
        case UNICC_ERR_NO_BACKEND: return "no collective-communication backend library could be located";
        case UNICC_ERR_BACKEND_LOAD: return "failed to load backend library";
        case UNICC_ERR_BACKEND_INIT_FAILED: return "backend vtable initialization failed";
        case UNICC_ERR_SYMBOL_NOT_FOUND: return "a required core symbol is missing from the backend library";
        case UNICC_ERR_NOT_INITIALIZED: return "unicc not initialized (call unicc_init first)";
        case UNICC_ERR_ALREADY_INITIALIZED: return "unicc already initialized";
        case UNICC_ERR_FINALIZED: return "unicc has been finalized";
        case UNICC_ERR_INVALID_ARGUMENT: return "invalid argument";
        case UNICC_ERR_NOT_SUPPORTED: return "operation not supported by the active backend (symbol missing)";
        case UNICC_ERR_UNHANDLED_BACKEND: return "backend returned a non-zero native result (see unicc_get_last_error)";
        case UNICC_ERR_BACKEND_NOT_SUPPORTED: return "backend type is not supported on this platform";
        case UNICC_ERR_INVALID_STATE: return "invalid internal state";
        case UNICC_ERR_OUT_OF_MEMORY: return "out of memory";
    }
    return "unknown error code";
}
