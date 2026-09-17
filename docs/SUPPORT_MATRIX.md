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

| Slot | fake test | nccl (real) | dcu (real, nccl\*-compat) | rccl (native) |
|---|---|---|---|---|
| `get_version` | Passed (fake) | Passed — iota, 2026-09-17 | Passed — dcu, 2026-09-17 | N/A on Hygon host (no rccl\*) |
| `comm_init_rank` | Passed (fake) | Passed — iota, 2026-09-17 | Passed — dcu, 2026-09-17 | N/A on Hygon host |
| `get_unique_id` | Passed (fake) | Passed — iota, 2026-09-17 | Passed — dcu, 2026-09-17 | N/A on Hygon host |
| `allreduce` | Passed (fake) | Passed — iota, 2026-09-17 (sum=3, world=2) | Passed — dcu, 2026-09-17 (sum=3, world=2) | N/A on Hygon host |
| `comm_count` | Passed (fake) | Passed — iota, 2026-09-17 (world check) | Passed — dcu, 2026-09-17 (world check) | N/A on Hygon host |
| `comm_destroy` | Passed (fake) | Passed — iota, 2026-09-17 | Passed — dcu, 2026-09-17 | N/A on Hygon host |
| `broadcast` | Passed (fake) | Not exercised (real) | Not exercised (real) | N/A on Hygon host |
| `comm_user_rank` | Passed (fake) | Not exercised (real) | Not exercised (real) | N/A on Hygon host |
| `group_start` | Passed (fake) | Not exercised (real) | Not exercised (real) | N/A on Hygon host |
| `group_end` | Passed (fake) | Not exercised (real) | Not exercised (real); degrade case fake-tested | N/A on Hygon host |

> The `rccl (native)` column means an AMD RCCL library exporting `rccl*`
> symbols. DCU here means **Hygon's DTK `librccl.so`**, which is a pure
> `nccl*`-compatibility layer (see the record below); that host has no `rccl*`
> symbols at all, so the native-RCCL row stays unevaluated there.

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

## DCU real-backend verification record

- **Date / machine**: 2026-09-17, `centos-8-dcu` (DCUSERVER, x86_64, CentOS 8).
- **Hardware**: 3× Hygon DCU (`/dev/dri` card0–card2), shared with the dtk
  special (two ranks were used, one GPU left idle).
- **Software**: Hygon DTK **23.10.1** (the current `/opt/dtk` target).
- **Collective library**: `librccl.so.1.0` → **`ncclGetVersion` reported
  21304**; both DTK 23.10.1 and 25.04 export **only the `nccl*` symbol family
  (GetVersion/CommInitRank/CommInitAll/AllReduce/Broadcast/CommCount/
  CommUserRank/CommDestroy/GetUniqueId/GroupStart/GroupEnd, …)** — no `rccl*`
  exclusives and no `hccl*` symbols. In other words Hygon's `librccl.so` is a
  **pure `nccl*`-compatibility layer**, the reverse of AMD RCCL (which exports
  both families). It depends on `libgalaxyhip.so` (Hygon HIP runtime),
  `librocm_smi64.so.2`, `libhsa-runtime64.so.1` (all under
  `/opt/dtk-23.10.1/lib`).
- **Result**: `world=2`, one rank per DCU, device memory via the zero-header
  adapter (HIP path). Both ranks `PASS (backend=nccl version=21304 sum=3)`.
  XCCL identified the library as **NCCL** (`ncclGetVersion` present, no
  `rccl*`) and bound the `nccl*` compat symbols — which is exactly what the
  design anticipated for a library that only exports the `nccl*` family. **No
  interface change was needed.**
- **Design implication**: if the plan wants a distinct "Hygon DCU" identity
  (better `xcc_backend_name()` and a future Hygon-native backend), the
  identifer needs an additional Hygon-specific probe. Today it deliberately
  looks like NCCL because the library's only surface is the `nccl*` compat
  layer.

## RCCL (native AMD) status

No AMD RCCL host assigned yet. The `rccl*`-native path (`src/backends/rccl.c`)
is verified only against the dual-symbol `fake_rccl_identity.so` fixture, which
locks in the RCCL-first identification and native `rccl*` binding. On the Hygon
DCU host there are no `rccl*` symbols, so AMD-RCCL-native verification remains
pending.

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

- **Native AMD RCCL** (`rccl*` symbols): no AMD host assigned; fake-only so
  far. On the Hygon DCU host there is no `rccl*` to verify against.
- **Real broadcast / group / comm_user_rank** against a live library: the iota
  (NCCL) and dcu (Hygon `nccl*`-compat) runs exercised the allreduce path;
  these slots are fake-tested only.
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
