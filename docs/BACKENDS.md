# XCCL backends

How the wrapper finds, loads, identifies and binds NCCL / RCCL — and the one
identification trap that matters (RCCL's `nccl*` compatibility symbols).

## Backend table (`src/xcc_loader.c`)

| name | type | `lib_name` (preferred) | `lib_name_alt` (fallback) |
|---|---|---|---|
| `nccl` | NCCL | `libnccl.so` | `libnccl.so.2` (older releases shipped `.1`) |
| `rccl` | RCCL | `librccl.so` | `librccl.so.1` |

Adding a future domestic-GPU collective library is a one-entry addition to
`xcc_backends[]` plus an identify rule and a binding file.

## Selection priority

Deterministic, modeled on UniMPI:

1. `XCCL_LIBRARY` — an exact library path or loader-resolvable name.
   *(Use this in CI, in tests and whenever several GPU stacks are installed.)*
2. `XCCL_BACKEND` — `nccl` or `rccl`. An unrecognized value is treated as a
   library path.
3. platform default — `libnccl.so` (NVIDIA).

M2 keeps the default simple (NCCL). Detecting CUDA vs ROCm presence to pick a
smarter default is a documented future refinement.

## Loading and fallback

`dlopen` is attempted with `lib_name`; on failure the backend's `lib_name_alt`
is tried (e.g. `libnccl.so` → `libnccl.so.2`). Unlike UniMPI there is no MPI
"standard ABI" library to reject — XCCL has no such class of ambiguity.

## Identification — the `rccl*` premise is obsolete (vendor records)

Earlier drafts of this document assumed **"real RCCL exports a `rccl*` native
family plus an `nccl*` ABI-compatibility layer"** and built the identifier on
checking `rcclGetVersion` first. The official records in `docs/official/`
(`amdc-rccl-official.md`, `amdc-rccl-source-evidence.md`) show that is wrong
for modern AMD RCCL:

- The RCCL 2.30.4 public API reference lists **only `nccl*`** functions,
  types and macros (header `nccl.h`); `ncclGetVersion` is documented as
  returning "the RCCL_VERSION_CODE of RCCL".
- GitHub code search finds **no public `rccl*` symbols** in the RCCL source
  (`rcclGetVersion`, `rcclCommInitRank`, `rcclBroadcast`, … all 0 hits);
  `rccl*` survives only as *internal* names (`rccl_wrap.cc`,
  `rcclAllReduceShouldTakeDdaPath`, `RcclTunableColls`).

So on the current ecosystem the symbol reality is one shared family:

| Library | `nccl*`-named public API | `rccl*`-named public API |
|---|---|---|
| NVIDIA NCCL | yes (native) | no |
| AMD RCCL | **yes (native — it *is* the official API)** | no (internal names only) |
| Hygon DCU DTK `librccl.so` | yes (only family) | no |

Consequences for XCCL, aligned with the design stance "load each backend's own
native symbols":

- The `rcclGetVersion`-first rule (still in `src/xcc_loader.c` for now) has
  **no real target**: on current RCCL, NCCL *and* DCU the `nccl*`-family check
  wins. The `rccl` binding (`src/backends/rccl.c`) was written for a symbol
  shape that modern AMD RCCL does not export.
- **Vendor identity is a separate question from the symbol family.** Telling
  NVIDIA NCCL from AMD RCCL from Hygon DCU needs a probe other than the
  symbol prefix: e.g. a dependency fingerprint (`libcudart` vs
  `libamdhip64`/`libhsa-runtime64` vs `libgalaxyhip`) or a vendor-specific
  symbol. That probe is a documented future refinement (`xcc_backend_name`
  stays generic until it lands).
- The fixture `fake_rccl_identity.c` ships both `rccl*` and `nccl*` families
  and its tests prove that *if* a library ever exports `rccl*`, XCCL binds the
  `rccl*` family. That remains a valid defensive path — but it no longer
  claims to model current AMD RCCL; it models a historical/third-party
  `rccl*`-exporting shape.

*Status note:* the code keeps the `rccl*` path for backward compatibility
until a real AMD host runs `nm -D librccl.so` and settles the per-binary
question (see `SUPPORT_MATRIX.md`). The identifier logic itself is unchanged
in M2; what changed is the documented model of what "RCCL" exports.

## Binding: dlsym per symbol, NULL on missing

Each backend binding (`src/backends/nccl.c`, `rccl.c`) resolves its family's
symbols into the process-wide vtable (`xcc`, declared in `xcc_vtable.h`). A
symbol the loaded library does not export leaves the slot `NULL` — there is no
stub and no abort. Callers gate with `xcc_*_available()`; the wrapper returns
`XCC_ERR_NOT_SUPPORTED` for a `NULL` slot.

The following are **core** (validated, missing ⇒ `xcc_init` fails):

- `*GetVersion`
- `*CommInitRank`
- `*AllReduce`
- `*Broadcast`

Everything else is **optional** (missing ⇒ `NULL` slot ⇒ graceful degrade):

- `*GetUniqueId`, `*CommDestroy`, `*CommCount`, `*CommUserRank`,
  `*GroupStart`, `*GroupEnd`

## Backend symbol manifest (M2)

The three-way separation discipline (docs/SUPPORT_MATRIX.md) distinguishes a
vtable slot from the symbol a backend actually exports. This manifest lists,
for every M2 slot, the exact symbol each backend family is expected to export.
"Core" slots are required for `xcc_init`; "optional" slots degrade to a NULL
slot and `xcc_*_available() == 0` when absent.

| vtable slot | role | nccl symbol (NCCL) | rccl symbol (RCCL) |
|---|---|---|---|
| `get_version` | core | `ncclGetVersion` | `rcclGetVersion` |
| `comm_init_rank` | core | `ncclCommInitRank` | `rcclCommInitRank` |
| `allreduce` | core | `ncclAllReduce` | `rcclAllReduce` |
| `broadcast` | core | `ncclBroadcast` | `rcclBroadcast` |
| `get_unique_id` | optional | `ncclGetUniqueId` | `rcclGetUniqueId` |
| `comm_destroy` | optional | `ncclCommDestroy` | `rcclCommDestroy` |
| `comm_count` | optional | `ncclCommCount` | `rcclCommCount` |
| `comm_user_rank` | optional | `ncclCommUserRank` | `rcclCommUserRank` |
| `group_start` | optional | `ncclGroupStart` | `rcclGroupStart` |
| `group_end` | optional | `ncclGroupEnd` | `rcclGroupEnd` |

The RCCL column is a *defensive* row, not a description of modern AMD RCCL:
real RCCL's public API is `nccl*`-named (see the "Identification" section and
`docs/official/`), so on a current RCCL the `nccl` column is the one actually
bound. The `rccl*` column stays as the family XCCL would bind *if* a loaded
library exports it (historical / third-party shape), reached only after a
`rcclGetVersion`-style probe. Coverage against live NCCL/RCCL binaries is
tracked in SUPPORT_MATRIX.md; until executed on a real AMD host the `rccl*`
rows above are the documented fallback, not a loaded-library measurement.

## Enum mapping

NCCL and RCCL use identical numeric values for datatypes and ops, so XCCL maps
its own enums onto them with a single table in `src/xcc_api.c`:

| XCC | nccl/rccl value | meaning |
|---|---|---|
| `XCC_I8` | 0 | int8 |
| `XCC_U8` | 2 | uint8 |
| `XCC_I32` | 3 | int32 |
| `XCC_U32` | 4 | uint32 |
| `XCC_I64` | 5 | int64 |
| `XCC_U64` | 6 | uint64 |
| `XCC_F16` | 9 | half |
| `XCC_F32` | 7 | float32 |
| `XCC_F64` | 8 | float64 |
| `XCC_BF16` | 10 | bfloat16 |
| `XCC_SUM` / `XCC_PROD` / `XCC_MAX` / `XCC_MIN` | 0 / 1 / 2 / 3 | reduce ops |

If a future backend diverges from these values, the table moves into
per-backend files and the wrapper keeps swapping at runtime.

## Error mapping

Backend results are ints where `0 == success`. XCCL maps `0 → XCC_OK` and any
non-zero → `XCC_ERR_UNHANDLED_BACKEND`, keeping the raw value for
`xcc_get_last_error`. The wrapper's own errors (`NOT_INITIALIZED`,
`NOT_SUPPORTED`, …) are returned unchanged.

## Platform support

| Platform | nccl | rccl |
|---|:---:|:---:|
| Linux | yes | yes |
| macOS | yes | no (ROCm) |
| Windows | not yet (planned) | no |

## M2 simplifications (deliberate)

- **Version gating is runtime-only**: `xcc_backend_version()` +
  document-level checks. No compile-time gate machinery (NCCL's API surface is
  small; UniMPI's MPI-version gating was judged unnecessary here).
- **Default backend is a fixed name**, not a CUDA/ROCm scan.
- **Vendor identity probe pending**: distinguishing NVIDIA NCCL from AMD RCCL
  from Hygon DCU (all `nccl*`-named today) needs a dependency/vendor probe,
  not the symbol prefix — recorded as future work; until then
  `xcc_backend_name()` reports whatever the loader resolved (`nccl`).
- **Error-string passthrough** (`ncclGetErrorString`) is pending; the raw
  backend result is surfaced today.

## Diagnosing on hosts with no NCCL/RCCL

```bash
XCCL_BACKEND=nccl ./build/minimal        # observe the loader messages
./build/minimal && ...                   # or call xcc_diagnose() from code
```

`XCCL_LIBRARY` pointing at a missing file, `libnccl.so` absent from the loader
path, or a wrong soname are the common causes — `xcc_platform_load_advice()`
describes the standard checks (`ldconfig -p | grep -E 'nccl|rccl'`, `ldd`, …).
