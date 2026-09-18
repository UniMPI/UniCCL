# UniCCL support and verification matrix

This document separates three different claims (the same discipline UniMPI
uses):

1. **a vtable slot exists** — a function-pointer field in `unicc_vtable_t`;
2. **a backend exports a symbol** that can populate that slot;
3. **a test has exercised the operation** with real code paths.

These are not equivalent. A zero-initialized vtable slot is not proof of
backend availability, and a fake-backend test is not proof against a real
NCCL/RCCL library. Each table below states exactly which of the three claims
it is making.

## M2 API inventory

`include/unicc.h` is the source of truth for what compiles. UniCCL claims only
the operations below; everything else is future work (allgather,
reduce-scatter, send/recv, alltoall, …).

Core (required; `unicc_vtable_validate_core` refuses a backend without them):

| Operation | vtable slot | Optional? |
|---|---|---|
| version | `get_version` | core |
| communicator bootstrap | `comm_init_rank` | core |
| allreduce | `allreduce` | core |
| broadcast | `broadcast` | core |

Optional (missing symbol → `NULL` slot → `unicc_*()` returns
`UNICC_ERR_NOT_SUPPORTED`, `*_available()` returns 0):

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

> The `rccl (native)` column tracks an AMD RCCL library exporting `rccl*`
> symbols. Per the official record (`docs/official/`) current RCCL exposes no
> public `rccl*` symbols, so this column is a defensive/historical row, not a
> claim about modern AMD RCCL. DCU here means **Hygon's DTK `librccl.so`**,
> which is a pure `nccl*`-compatibility layer (see the record below); that
> host has no `rccl*` symbols at all, so the native-RCCL row stays
> unevaluated there.

## P1 backend-matrix coverage (v0.2.0)

The `oneccl` and `eccl` backends are **fake-verified only** at this milestone:
each has a host-side fixture (`fake_oneccl_identity.so`, `fake_eccl_identity.so`)
exercising identification, full vtable binding, the id adapters (oneCCL
4096-byte id, double-buffered broadcast) and the ECCL `comm_user_rank` degrade
path. No real oneCCL / ECCL host is assigned; real-symbol rows above
deliberately stay "Not verified" until `nm -D` on live libraries (see the
verification list in
`docs/official/ccL-ecosystem-survey-2026-09-17.md`).

## NCCL real-backend verification record

- **Date / machine**: 2026-09-17, `iota` (172.18.7.45, x86_64).
- **GPUs**: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (97.8 GiB each);
  driver 580.105.08 (CUDA 13.0); CUDA toolkit 12.9.
- **Backend library**: HPC SDK 25.11 bundled NCCL
  `/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/comm_libs/13.0/nccl/lib/libnccl.so.2`
  → `*GetVersion` reported **2.28.7** (int 22807).
- **Run**: `tests/run_integration.sh` equivalent, `world=2` (one rank per GPU);
  each rank sent its own vector, device memory via the zero-header
  `unicc_devmem` adapter, uid published by rank 0 / consumed by rank 1.
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
  UniCCL identified the library as **NCCL** (`ncclGetVersion` present, no
  `rccl*`) and bound the `nccl*` compat symbols — which is exactly what the
  design anticipated for a library that only exports the `nccl*` family. **No
  interface change was needed.**
- **Design implication**: if the plan wants a distinct "Hygon DCU" identity
  (better `unicc_backend_name()` and a future Hygon-native backend), the
  identifer needs an additional Hygon-specific probe. Today it deliberately
  looks like NCCL because the library's only surface is the `nccl*` compat
  layer.

## RCCL (native AMD) status

No AMD RCCL host is assigned. More importantly, the official record
(`docs/official/amdc-rccl-official.md` / `amdc-rccl-source-evidence.md`)
shows the assumed shape was wrong: **AMD RCCL 2.30.4's public API is
`nccl*`-named** (documented functions/types/macros, header `nccl.h`), and the
RCCL source exposes no public `rccl*` symbols (`rccl*` survives only as
internal implementation names). So on current RCCL the `nccl` binding — not
`src/backends/rccl.c` — is the *native* path:

| Question | Current status |
|---|---|
| Recognize a real RCCL AS RCCL (not as NCCL) | Not possible via symbol prefix today; needs a vendor probe (dependency fingerprint) — recorded as future work in BACKENDS.md |
| `rccl*`-native binding (`src/backends/rccl.c`, identification + `rcclGetVersion` probe) | Verified only against `fake_rccl_identity.so` (which ships both families); re-framed as a defensive/historical path, not a model of current AMD RCCL |
| Definitively settle whether any shipped `librccl.so` exports `rccl*` | Pending: `nm -D librccl.so` on a real AMD host; the source-level answer today is "no public `rccl*`" |

## Identification (the RCCL cross-check)

| Fixture / library shape | Expected identification | Test | Status |
|---|---|---|---|
| `libnccl.so` with only `nccl*` symbols | NCCL | test_loader / test_api | Passed (fake); **Passed (real) on iota** |
| `librccl.so` with **both** `rccl*` + `nccl*` (historical / third-party shape) | **RCCL** | test_loader / test_api | Passed (fake); defensive only |
| current AMD RCCL shape (official API is `nccl*` only) | NCCL* | — (no AMD host) | **pending real `nm -D librccl.so`**; *see note* |
| backend missing an optional symbol (`ncclGroupEnd` absent) | NCCL, `group_end` slot NULL | test_vtable / test_api | Passed (fake) |

The dual-symbol row is retained as a defensive check: *if* a library exports
`rccl*`, `rcclGetVersion` probing must win so the wrapper binds the `rccl*`
family (the fake proves this via a distinct version + `+1000.0f` allreduce
marker). But note the third row: the official RCCL record says current AMD
RCCL exposes **no** public `rccl*` symbols, so on a real RCCL today the
identifier resolves via `ncclGetVersion` → NCCL, exactly as it does on Hygon
DCU. The `*` flags that identification as provisional: vendor identity should
come from a dedicated probe (dependency fingerprint), not from the symbol
prefix — future work per BACKENDS.md.

## What is not verified yet

- **Native AMD RCCL**: no AMD host assigned. Official records say RCCL's
  public API is `nccl*`-named and there are no public `rccl*` symbols, so the
  `rccl*`-native path remains fake-only (defensive) and the vendor-identity
  probe is unimplemented. A real AMD host (`nm -D librccl.so`) settles the
  remaining per-binary question.
- **Real broadcast / group / comm_user_rank** against a live library: the iota
  (NCCL) and dcu (Hygon `nccl*`-compat) runs exercised the allreduce path;
  these slots are fake-tested only.
- Version gating beyond the `*GetVersion` value (runtime-only in M2).
- `unicc_get_last_error` string passthrough (raw backend code only).

## How to record a real-backend result

1. Build with `-DUNICC_BUILD_TESTS_INTEGRATION=ON` on the GPU host.
2. `tests/run_integration.sh nccl <world>` / `rccl <world>` (env: `UNICC_BACKEND`
   or `UNICC_LIBRARY=<exact libnccl.so.2>`; `LD_LIBRARY_PATH` only if the device
   runtime (`libcudart.so`) is off the default search path).
3. `test_integration` binds one rank per GPU and reports `PASS (backend=… )`
   with the expected analytic sum.
4. Flip the matching rows above to "Passed (real)" and append to the
   verification record with machine/version/world.

Each claim stays separate: a passing fake test and a passing real test are two
different entries, and neither implies the other.
