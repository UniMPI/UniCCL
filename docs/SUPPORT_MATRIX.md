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
| Passed (fake) | Exercised against a host-side fake fixture on the build host. |
| Passed (real) | Exercised against a real NCCL/RCCL library on a GPU host (see the record below). |
| Compiled only | Built but not executed. |
| Not verified | Neither fake nor real execution recorded. |

## Backend symbols and test coverage

The symbol each slot binds is enumerated in `docs/BACKENDS.md` ("Backend
symbol manifest"). What follows is the *test* column of the three-way split —
fake vs live-library execution per slot. "Not exercised (real)" means the slot
is bound and fake-tested but the real-backend integration run did not call it.

| Slot | fake test | nccl (real) | rccl (real) |
|---|---|---|---|
| `get_version` | Passed (fake) | Passed — iota, 2026-09-17 | Not verified |
| `comm_init_rank` | Passed (fake) | Passed — iota, 2026-09-17 | Not verified |
| `get_unique_id` | Passed (fake) | Passed — iota, 2026-09-17 | Not verified |
| `allreduce` | Passed (fake) | Passed — iota, 2026-09-17 (sum=3, world=2) | Not verified |
| `comm_count` | Passed (fake) | Passed — iota, 2026-09-17 (world check) | Not verified |
| `comm_destroy` | Passed (fake) | Passed — iota, 2026-09-17 | Not verified |
| `broadcast` | Passed (fake) | Not exercised (real) | Not verified |
| `comm_user_rank` | Passed (fake) | Not exercised (real) | Not verified |
| `group_start` | Passed (fake) | Not exercised (real) | Not verified |
| `group_end` | Passed (fake) | Not exercised (real); degrade case fake-tested | Not verified |

## NCCL real-backend verification record

- **Date / machine**: 2026-09-17, `iota` (172.18.7.45, x86_64).
- **GPUs**: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (97.8 GiB each);
  driver 580.105.08 (CUDA 13.0); CUDA toolkit 12.9.
- **Backend library**: HPC SDK 25.11 bundled NCCL
  `/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/13.0/nccl/lib/libnccl.so.2`
  → `*GetVersion` reported **2.28.7** (int 22807).
- **Run**: `tests/run_integration.sh` equivalent, `world=2` (one rank per GPU);
  each rank sent its own vector, device memory via the zero-header
  `xcc_devmem` adapter, uid published by rank 0 / consumed by rank 1.
- **Result**: both ranks `PASS (backend=nccl version=22807 sum=3)` — the SUM
  allreduce returned the analytic `1+2=3` on both GPUs. Rows above flipped to
  "Passed (real)" accordingly.
- **Notes**: this exercised the *core* path only (no broadcast/group on real
  NCCL yet). A one-rank-per-GPU binding is required: NCCL refuses two ranks of
  one communicator on the same GPU; `test_integration` binds the device in
  process (`set_device(rank)` with a `world <= GPUs` guard) and also gates on
  the visible device count.
- **Workstation choice**: the coordinator's original NVIDIA target (theta) has
  a single GPU and no NCCL, so the user directed this verification to `iota`
  (dual-GPU) instead.

## RCCL / DCU status

Not verified. Target machine `centos-8-dcu` (Hygon DCU) is shared with the dtk
special; the **collective-communication library form on that host
(`librccl.so` vs a DCU-native library) is pending confirmation** — that
determines whether `src/backends/rccl.c` covers it directly or a new backend is
needed. The fixture `fake_rccl_identity.so` (dual `rccl*`+`nccl*`) already
locks in the RCCL-first identification and native `rccl*` binding on the fake
side.

## Identification (the RCCL cross-check)

| Fixture / library shape | Expected identification | Test | Status |
|---|---|---|---|
| `libnccl.so` with only `nccl*` symbols | NCCL | test_loader / test_api | Passed (fake); **Passed (real) on iota** |
| `librccl.so` with **both** `rccl*` + `nccl*` (real RCCL shape) | **RCCL** | test_loader / test_api | Passed (fake); real RCCL pending |
| backend missing an optional symbol (`ncclGroupEnd` absent) | NCCL, `group_end` slot NULL | test_vtable / test_api | Passed (fake) |

The dual-symbol case is the one the design spec calls out: `rcclGetVersion`
must be probed before `ncclGetVersion` or any RCCL library (which ships an
`nccl*` compat layer) would be misidentified as NCCL. The fake proves the
wrapper binds the `rccl*` native family (distinct version + `+1000.0f`
allreduce marker).

## What is not verified yet

- **Real RCCL / DCU** (any operation). Pending confirmation of the DCU
  collective-library form on `centos-8-dcu` and a slot on that shared host.
- **Real broadcast / group / comm_user_rank / comm_count** against real NCCL:
  the iota run exercised the allreduce path; these slots are fake-tested only.
- Version gating beyond the `*GetVersion` value (runtime-only in M2).
- `xcc_get_last_error` string passthrough (raw backend code only).

## How to record a real-backend result

1. Build with `-DXCCL_BUILD_TESTS_INTEGRATION=ON` on the GPU host.
2. `tests/run_integration.sh nccl <world>` / `rccl <world>` (env: `XCCL_BACKEND`
   or `XCCL_LIBRARY=<exact libnccl.so.2>`; `LD_LIBRARY_PATH` only if the device
   runtime (`libcudart.so`) is off the default search path).
3. `test_integration` binds one rank per GPU and reports `PASS (backend=… )`
   with the expected analytic sum.
4. Flip the matching rows above to "Passed (real)" and append to the
   verification record with machine/version/world.

Each claim stays separate: a passing fake test and a passing real test are two
different entries, and neither implies the other.
