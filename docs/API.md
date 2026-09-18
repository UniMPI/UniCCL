# UniCCL API reference

UniCCL exposes a small unified C API, `include/unicc.h`. All functions return the
unified `unicc_result_t` (see `include/unicc_errors.h`); `0`/`UNICC_OK` means
success. Backend-native results are never leaked to callers — the raw value of
the last failed backend call is available through `unicc_get_last_error`.

Every collective degrades gracefully: if the active backend does not export the
symbol behind an operation, the call returns `UNICC_ERR_NOT_SUPPORTED` and the
corresponding `*_available()` predicate returns `0`. Callers that want to be
portable across backends gate on `*_available()`.

## Data types and reduce ops

```c
typedef enum { UNICC_I8, UNICC_U8, UNICC_I32, UNICC_U32, UNICC_I64,
               UNICC_U64, UNICC_F16, UNICC_F32, UNICC_F64, UNICC_BF16 } unicc_datatype_t;
typedef enum { UNICC_SUM, UNICC_PROD, UNICC_MAX, UNICC_MIN } unicc_reduce_op_t;
```

These are UniCCL's own enums; the mapping onto each backend's numeric values
lives in `src/unicc_api.c` (documented in `BACKENDS.md`).

## Lifecycle

| Function | Description |
|---|---|
| `unicc_init()` | Select the backend from the environment, load it, validate core symbols, fill the vtable. |
| `unicc_finalize()` | Clear the vtable, unload the backend. |
| `unicc_is_initialized()` | 1 once `unicc_init` succeeded and before `unicc_finalize`. |

`unicc_init` follows the loader priority `UNICC_LIBRARY` → `UNICC_BACKEND` →
platform default.

## Version / diagnostics

| Function | Description |
|---|---|
| `const char* unicc_backend_name()` | `"nccl"`, `"rccl"`, or `"unknown"`. |
| `const char* unicc_get_library_path()` | Path of the loaded library (`""` if none). |
| `int unicc_get_version(char* buf, size_t len)` | UniCCL wrapper version string (e.g. `"UniCCL 0.1.0-alpha"`). |
| `int unicc_backend_version(int* version)` | Backend-reported version integer (nccl/rccl GetVersion). |
| `int unicc_print_backend_info()` | Print the known-backend table to stderr. |
| `int unicc_diagnose()` | Load + identity + per-symbol coverage of the selected library. |

`unicc_diagnose()` is the first thing to run on a host with no NCCL/RCCL — it
prints which symbols are missing so the failure is obvious.

## Communicators

| Function | Description |
|---|---|
| `unicc_get_unique_id(unicc_unique_id_t* uid)` | Obtain a fresh 128-byte communicator id (layout-identical to `ncclUniqueId`). |
| `unicc_comm_init_rank(unicc_comm_t* comm, int nranks, unicc_unique_id_t uid, int rank)` | Create a communicator; the uid is passed **by value** (matches the native NCCL signature). `comm` is an opaque handle. |
| `unicc_comm_destroy(unicc_comm_t comm)` | Destroy a communicator. |
| `unicc_comm_count(unicc_comm_t comm, int* count)` | Number of ranks in the communicator. |
| `unicc_comm_user_rank(unicc_comm_t comm, int* rank)` | This rank's id in the communicator. |
| `unicc_comm_available()` | 1 if communicator bootstrap is usable with this backend. |

Note: UniCCL does not create a "world"-style communicator for you — like NCCL,
you call `unicc_get_unique_id` (once, e.g. on rank 0), transfer the uid to every
participant, then each calls `unicc_comm_init_rank` with `nranks`/`rank`.

## Collectives

| Function | Description |
|---|---|
| `unicc_allreduce(sendbuf, recvbuf, count, datatype, op, comm, stream)` | All-reduce into `recvbuf`. |
| `unicc_broadcast(buf, count, datatype, root, comm, stream)` | Broadcast from `root`. |
| `unicc_group_start()` / `unicc_group_end()` | Begin / end a group of collective calls (accelerate batched operations). |
| `unicc_allreduce_available()`, `unicc_broadcast_available()`, `unicc_group_start/end_available()` | Gate: 1 iff the backend exports the backing symbol. |

`stream` may be `NULL` (default stream) or an opaque stream handle; UniCCL does
not interpret it.

## Error handling

- `const char* unicc_error_string(unicc_result_t rc)` — description of a unified code.
- `int unicc_get_last_error(int* raw_backend_result)` — value returned by the last
  failed backend call (`0` if none yet).

Common codes: `UNICC_ERR_NOT_INITIALIZED` (use before init), `UNICC_ERR_FINALIZED`
(after finalize), `UNICC_ERR_ALREADY_INITIALIZED` (double init),
`UNICC_ERR_NOT_SUPPORTED` (backend lacks the operation — slot NULL),
`UNICC_ERR_UNHANDLED_BACKEND` (backend returned nonzero; check
`unicc_get_last_error`).

## Minimal example

```c
#include <unicc.h>

int main(void) {
    if (unicc_init() != UNICC_OK) { unicc_diagnose(); return 1; }
    printf("backend=%s\n", unicc_backend_name());

    unicc_unique_id_t uid;  unicc_get_unique_id(&uid);
    unicc_comm_t comm;      unicc_comm_init_rank(&comm, 1, uid, 0);

    float in[4] = {1,2,3,4}, out[4];
    if (unicc_allreduce_available())
        unicc_allreduce(in, out, 4, UNICC_F32, UNICC_SUM, comm, NULL);

    unicc_comm_destroy(comm);
    unicc_finalize();
    return 0;
}
```

See `examples/minimal.c` for a compilable version.
