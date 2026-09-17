# XCCL support and verification matrix

This document separates three different claims (the same discipline UniMPI
uses):

1. **a vtable slot exists** — a function-pointer field in `xcc_vtable_t`;
2. **a backend exports a symbol** that can populate that slot;
3. **a test has exercised the operation** with real code paths.

These are not equivalent. A zero-initialized vtable slot is not proof of
backend availability, and a fake-backend test is not proof against a real
NCCL/RCCL library. Each table below states exactly which of the three claims
it is making.

## M2 API inventory

`include/xcc.h` is the source of truth for what compiles. XCCL claims only
the operations below; everything else is future work (allgather,
reduce-scatter, send/recv, alltoall, …).

Core (required; `xcc_vtable_validate_core` refuses a backend without them):

| Operation | vtable slot | Optional? |
|---|---|---|
| version | `get_version` | core |
| communicator bootstrap | `comm_init_rank` | core |
| allreduce | `allreduce` | core |
| broadcast | `broadcast` | core |

Optional (missing symbol → `NULL` slot → `xcc_*()` returns
`XCC_ERR_NOT_SUPPORTED`, `*_available()` returns 0):

| Operation | vtable slot |
|---|---|
| unique id | `get_unique_id` |
| comm destroy | `comm_destroy` |
| comm count | `comm_count` |
| comm user rank | `comm_user_rank` |
| group start / end | `group_start` / `group_end` |

## Verification levels

| Level | Meaning |
|---|---|
| Passed (fake) | Exercised against a host-side fake fixture on this build host. |
| Compiled only | Built but not executed (real-backend integration test). |
| Not verified | Real NCCL / RCCL execution pending (needs a GPU image / host). |

## Backend symbols and test coverage

| Slot | nccl exports? | rccl exports? | Symbol tested? | Where |
|---|---|---|---|---|
| `get_version` | `ncclGetVersion` | `rcclGetVersion` | fake: both; real: not verified | test_api, test_loader, integration |
| `comm_init_rank` | `ncclCommInitRank` | `rcclCommInitRank` | fake: both; real: not verified | test_api, integration |
| `allreduce` | `ncclAllReduce` | `rcclAllReduce` | fake: both; real: not verified | test_api, integration |
| `broadcast` | `ncclBroadcast` | `rcclBroadcast` | fake: both; real: not verified | test_api |
| `get_unique_id` | `ncclGetUniqueId` | `rcclGetUniqueId` | fake: both; real: not verified | test_api, integration |
| `comm_destroy` | `ncclCommDestroy` | `rcclCommDestroy` | fake: both; real: not verified | test_api, integration |
| `comm_count` | `ncclCommCount` | `rcclCommCount` | fake: both; real: not verified | test_api, integration |
| `comm_user_rank` | `ncclCommUserRank` | `rcclCommUserRank` | fake: both; real: not verified | test_api |
| `group_start` | `ncclGroupStart` | `rcclGroupStart` | fake: both; real: not verified | test_api |
| `group_end` | `ncclGroupEnd` | `rcclGroupEnd` | fake: both; real: not verified | test_api, degrade fixture |

## Identification (the RCCL cross-check)

| Fixture / library shape | Expected identification | Test | Status |
|---|---|---|---|
| `libnccl.so` with only `nccl*` symbols | NCCL | test_loader / test_api | Passed (fake) |
| `librccl.so` with **both** `rccl*` + `nccl*` (real RCCL shape) | **RCCL** | test_loader / test_api | Passed (fake); real RCCL not verified |
| backend missing an optional symbol (`ncclGroupEnd` absent) | NCCL, `group_end` slot NULL | test_vtable / test_api | Passed (fake) |

The dual-symbol case is the one the design spec calls out: `rcclGetVersion`
must be probed before `ncclGetVersion` or any RCCL library (which ships an
`nccl*` compat layer) would be misidentified as NCCL. The fake proves the
wrapper binds the `rccl*` native family (distinct version + `+1000.0f`
allreduce marker).

## What is not verified yet

- **Real NCCL / RCCL execution** (any operation above on a GPU host/image).
  The harness `tests/run_integration.sh` and executable
  `tests/integration/test_integration` are in place (built with
  `-DXCCL_BUILD_TESTS_INTEGRATION=ON`, run via the script or CI) but the
  matrix row remains **Not verified** until a run lands on
  `ubuntu-24.04-gcc13-cuda` / `-rocm` (or a GPU host). Until then the
  "exports?" column above is derived from the public API documentation, not
  from a loaded real library.
- Version gating beyond the `*GetVersion` value (runtime-only in M2).
- `xcc_get_last_error` string passthrough (raw backend code only).

## How to record a real-backend result

1. Build with `-DXCCL_BUILD_TESTS_INTEGRATION=ON` on the GPU image/host.
2. `tests/run_integration.sh nccl 2` and `tests/run_integration.sh rccl 2`.
3. Flip the corresponding rows above from "Not verified" to "Passed (real)"
   and note the image, NCCL/RCCL versions and world size.

Each claim stays separate: a passing fake test and a passing real test are two
different entries, and neither implies the other.
