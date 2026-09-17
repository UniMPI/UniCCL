# XCCL API reference

XCCL exposes a small unified C API, `include/xcc.h`. All functions return the
unified `xcc_result_t` (see `include/xcc_errors.h`); `0`/`XCC_OK` means
success. Backend-native results are never leaked to callers — the raw value of
the last failed backend call is available through `xcc_get_last_error`.

Every collective degrades gracefully: if the active backend does not export the
symbol behind an operation, the call returns `XCC_ERR_NOT_SUPPORTED` and the
corresponding `*_available()` predicate returns `0`. Callers that want to be
portable across backends gate on `*_available()`.

## Data types and reduce ops

```c
typedef enum { XCC_I8, XCC_U8, XCC_I32, XCC_U32, XCC_I64,
               XCC_U64, XCC_F16, XCC_F32, XCC_F64, XCC_BF16 } xcc_datatype_t;
typedef enum { XCC_SUM, XCC_PROD, XCC_MAX, XCC_MIN } xcc_reduce_op_t;
```

These are XCCL's own enums; the mapping onto each backend's numeric values
lives in `src/xcc_api.c` (documented in `BACKENDS.md`).

## Lifecycle

| Function | Description |
|---|---|
| `xcc_init()` | Select the backend from the environment, load it, validate core symbols, fill the vtable. |
| `xcc_finalize()` | Clear the vtable, unload the backend. |
| `xcc_is_initialized()` | 1 once `xcc_init` succeeded and before `xcc_finalize`. |

`xcc_init` follows the loader priority `XCCL_LIBRARY` → `XCCL_BACKEND` →
platform default.

## Version / diagnostics

| Function | Description |
|---|---|
| `const char* xcc_backend_name()` | `"nccl"`, `"rccl"`, or `"unknown"`. |
| `const char* xcc_get_library_path()` | Path of the loaded library (`""` if none). |
| `int xcc_get_version(char* buf, size_t len)` | XCCL wrapper version string (e.g. `"XCCL 0.1.0-alpha"`). |
| `int xcc_backend_version(int* version)` | Backend-reported version integer (nccl/rccl GetVersion). |
| `int xcc_print_backend_info()` | Print the known-backend table to stderr. |
| `int xcc_diagnose()` | Load + identity + per-symbol coverage of the selected library. |

`xcc_diagnose()` is the first thing to run on a host with no NCCL/RCCL — it
prints which symbols are missing so the failure is obvious.

## Communicators

| Function | Description |
|---|---|
| `xcc_get_unique_id(xcc_unique_id_t* uid)` | Obtain a fresh 128-byte communicator id (layout-identical to `ncclUniqueId`). |
| `xcc_comm_init_rank(xcc_comm_t* comm, int nranks, xcc_unique_id_t uid, int rank)` | Create a communicator; the uid is passed **by value** (matches the native NCCL signature). `comm` is an opaque handle. |
| `xcc_comm_destroy(xcc_comm_t comm)` | Destroy a communicator. |
| `xcc_comm_count(xcc_comm_t comm, int* count)` | Number of ranks in the communicator. |
| `xcc_comm_user_rank(xcc_comm_t comm, int* rank)` | This rank's id in the communicator. |
| `xcc_comm_available()` | 1 if communicator bootstrap is usable with this backend. |

Note: XCCL does not create a "world"-style communicator for you — like NCCL,
you call `xcc_get_unique_id` (once, e.g. on rank 0), transfer the uid to every
participant, then each calls `xcc_comm_init_rank` with `nranks`/`rank`.

## Collectives

| Function | Description |
|---|---|
| `xcc_allreduce(sendbuf, recvbuf, count, datatype, op, comm, stream)` | All-reduce into `recvbuf`. |
| `xcc_broadcast(buf, count, datatype, root, comm, stream)` | Broadcast from `root`. |
| `xcc_group_start()` / `xcc_group_end()` | Begin / end a group of collective calls (accelerate batched operations). |
| `xcc_allreduce_available()`, `xcc_broadcast_available()`, `xcc_group_start/end_available()` | Gate: 1 iff the backend exports the backing symbol. |

`stream` may be `NULL` (default stream) or an opaque stream handle; XCCL does
not interpret it.

## Error handling

- `const char* xcc_error_string(xcc_result_t rc)` — description of a unified code.
- `int xcc_get_last_error(int* raw_backend_result)` — value returned by the last
  failed backend call (`0` if none yet).

Common codes: `XCC_ERR_NOT_INITIALIZED` (use before init), `XCC_ERR_FINALIZED`
(after finalize), `XCC_ERR_ALREADY_INITIALIZED` (double init),
`XCC_ERR_NOT_SUPPORTED` (backend lacks the operation — slot NULL),
`XCC_ERR_UNHANDLED_BACKEND` (backend returned nonzero; check
`xcc_get_last_error`).

## Minimal example

```c
#include <xcc.h>

int main(void) {
    if (xcc_init() != XCC_OK) { xcc_diagnose(); return 1; }
    printf("backend=%s\n", xcc_backend_name());

    xcc_unique_id_t uid;  xcc_get_unique_id(&uid);
    xcc_comm_t comm;      xcc_comm_init_rank(&comm, 1, uid, 0);

    float in[4] = {1,2,3,4}, out[4];
    if (xcc_allreduce_available())
        xcc_allreduce(in, out, 4, XCC_F32, XCC_SUM, comm, NULL);

    xcc_comm_destroy(comm);
    xcc_finalize();
    return 0;
}
```

See `examples/minimal.c` for a compilable version.
