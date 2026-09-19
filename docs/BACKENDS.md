# UniCCL backends

How the wrapper finds, loads, identifies and binds NCCL / RCCL — and the one
identification trap that matters (RCCL's `nccl*` compatibility symbols).

## Backend table (`src/unicc_loader.c`)

| name | type | prefix | `lib_name` (preferred) | `lib_name_alt` (fallback) | probe symbol |
|---|---|---|---|---|---|
| `nccl` | NVIDIA NCCL | `nccl*` | `libnccl.so` | `libnccl.so.2` (older releases shipped `.1`) | `ncclGetVersion` |
| `rccl` | AMD RCCL (defensive) | `rccl*` | `librccl.so` | `librccl.so.1` | `rcclGetVersion` |
| `oneccl` | Intel oneCCL v2 | `oneccl*` | `libccl.so.2` | `libccl.so` | `onecclGetVersion` |
| `eccl` | Enflame ECCL | `eccl*` | `libeccl.so` | — | `ecclGetVersion` |

Adding a future domestic-GPU collective library is a one-entry addition to
`unicc_backends[]` (plus its probe symbol) and a binding file. Vendor prefixes
and library names come from the evidence records in `docs/official/`
(`ccL-ecosystem-survey-2026-09-17.md`): every vendor keeps its own prefix.
Intel oneCCL v2's C API (`libccl.so.2`, not the classic C++ `libccl.so.1`) and
Enflame ECCL are NCCL-shaped; the domestic-card libraries land in a later
milestone.

## Selection priority

Deterministic, modeled on UniMPI:

1. `UNICC_LIBRARY` — an exact library path or loader-resolvable name.
   *(Use this in CI, in tests and whenever several GPU stacks are installed.)*
2. `UNICC_BACKEND` — `nccl`, `rccl`, `oneccl` or `eccl`. An unrecognized value
   is treated as a library path.
3. platform default — `libnccl.so` (NVIDIA).

M2 keeps the default simple (NCCL). Detecting CUDA vs ROCm presence to pick a
smarter default is a documented future refinement.

## Loading and fallback

`dlopen` is attempted with `lib_name`; on failure the backend's `lib_name_alt`
is tried (e.g. `libnccl.so` → `libnccl.so.2`). Unlike UniMPI there is no MPI
"standard ABI" library to reject — UniCCL has no such class of ambiguity.

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

Consequences for UniCCL, aligned with the design stance "load each backend's own
native symbols":

- The `rcclGetVersion`-first rule (still in `src/unicc_loader.c` for now) has
  **no real target**: on current RCCL, NCCL *and* DCU the `nccl*`-family check
  wins. The `rccl` binding (`src/backends/rccl.c`) was written for a symbol
  shape that modern AMD RCCL does not export.
- **Vendor identity is a separate question from the symbol family.** Telling
  NVIDIA NCCL from AMD RCCL from Hygon DCU needs a probe other than the
  symbol prefix: e.g. a dependency fingerprint (`libcudart` vs
  `libamdhip64`/`libhsa-runtime64` vs `libgalaxyhip`) or a vendor-specific
  symbol. That probe is a documented future refinement (`unicc_backend_name`
  stays generic until it lands).
- The fixture `fake_rccl_identity.c` ships both `rccl*` and `nccl*` families
  and its tests prove that *if* a library ever exports `rccl*`, UniCCL binds the
  `rccl*` family. That remains a valid defensive path — but it no longer
  claims to model current AMD RCCL; it models a historical/third-party
  `rccl*`-exporting shape.

*Status note:* the code keeps the `rccl*` path for backward compatibility
until a real AMD host runs `nm -D librccl.so` and settles the per-binary
question (see `SUPPORT_MATRIX.md`). What changed in the documented model of
what "RCCL" exports is covered above.

## Identification order (multi-backend)

`unicc_loader_identify_backend` probes each family's specific symbol in a
fixed order — the vendor-specific families first, the generic `nccl` probe
last, so a library that *also* happens to export `ncclGetVersion` is never
misidentified:

1. `rcclGetVersion` → RCCL (defensive)
2. `onecclGetVersion` → oneCCL
3. `ecclGetVersion` → ECCL
4. `ncclGetVersion` → NCCL (also real RCCL / Hygon DCU, whose public API is
   `nccl*`-named)

P2 adds the domestic-card probes (Cambricon `cnclGetLibVersion`, Ascend
`Hccl*`, dual MCCL `mccl*` — note the two MCCL libraries share the `mccl*`
prefix and are told apart by a secondary probe / exact `UNICC_LIBRARY` path).

## Binding: one generic binder, dlsym per symbol, NULL on missing

A single generic binder (`src/backends/unicc_bind.c`) serves every family;
each family (`nccl.c`, `rccl.c`, …) is a one-line init that points at its row
in the `unicc_backends[]` descriptor (the single source of symbol names, id
size, and signature quirks). Core validation runs **at bind time and only
against the IDENTIFIED family's symbols** — never a mixed-family OR-set, so an
incomplete or hybrid library cannot ride one family's validation into another
family's binding and crash. A symbol the loaded library does not export leaves
the slot `NULL` — there is no stub and no abort. Callers gate with
`unicc_*_available()`; the wrapper returns `UNICC_ERR_NOT_SUPPORTED` for a
`NULL` slot. Optional wrappers are installed only over a resolved symbol —
never a wrapper whose inner call is NULL (a library without `GetUniqueId`
keeps `unicc_get_unique_id()` degrading to `NOT_SUPPORTED` instead of crashing).

The following are **core** (validated, missing ⇒ `unicc_init` fails):

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
"Core" slots are required for `unicc_init`; "optional" slots degrade to a NULL
slot and `unicc_*_available() == 0` when absent.

| vtable slot | role | nccl (NCCL) | rccl (RCCL, defensive) | oneccl (Intel) | eccl (Enflame) |
|---|---|---|---|---|---|
| `get_version` | core | `ncclGetVersion` | `rcclGetVersion` | `onecclGetVersion` | `ecclGetVersion` |
| `comm_init_rank` | core | `ncclCommInitRank` | `rcclCommInitRank` | `onecclCommInitRank` | `ecclCommInitRank` |
| `allreduce` | core | `ncclAllReduce` | `rcclAllReduce` | `onecclAllReduce` | `ecclAllReduce` |
| `broadcast` | core | `ncclBroadcast` | `rcclBroadcast` | `onecclBroadcast`¹ | `ecclBroadcast`¹ |
| `get_unique_id` | optional | `ncclGetUniqueId` | `rcclGetUniqueId` | `onecclGetUniqueId` | `ecclGetUniqueId` |
| `comm_destroy` | optional | `ncclCommDestroy` | `rcclCommDestroy` | `onecclCommDestroy` | `ecclCommDestroy` |
| `comm_count` | optional | `ncclCommCount` | `rcclCommCount` | `onecclCommCount` | `ecclCommCount` |
| `comm_user_rank` | optional | `ncclCommUserRank` | `rcclCommUserRank` | `onecclCommUserRank` | — (not exported) |
| `group_start` | optional | `ncclGroupStart` | `rcclGroupStart` | `onecclGroupStart` | `ecclGroupStart` |
| `group_end` | optional | `ncclGroupEnd` | `rcclGroupEnd` | `onecclGroupEnd` | `ecclGroupEnd` |

¹ oneCCL v2 / ECCL broadcast are double-buffered (separate send/recv); the
in-place vtable slot passes the same buffer for both — no extra copy.

Notes on the table:
- The `rccl` column is a *defensive* row, not a description of modern AMD
  RCCL: real RCCL's public API is `nccl*`-named (see "Identification" and
  `docs/official/`), so on a current RCCL the `nccl` column is the one bound.
- oneCCL / ECCL entries come from the vendor sources in `docs/official/`
  (oneCCL v2 `include/oneapi/ccl.h`; torch-gcu `eccl*` call sites). ECCL has
  no `ecclCommUserRank` in the ecosystem, so that slot is intentionally NULL.
- Coverage against live binaries is tracked in SUPPORT_MATRIX.md; the
  oneccl/eccl rows are fake-fixture-backed today, real-host verification is
  pending (`nm -D libccl.so.2`, `nm -D libeccl.so`).

## Enum mapping (identity for all current backends)

Every backend bound today (NCCL, RCCL, Hygon DCU, oneCCL v2, ECCL) uses the
NCCL-family numbering, so the mapping is the **identity** and lives as
header-inlined accessors in `include/unicc_dtmap.h` — the collective hot path
is a single bounds-checked load: no table cell to fill, no init-time work, no
per-call backend branch. This mirrors UniMPI's steady-state philosophy: the
mapping costs a load, not a function call.

The identity (NCCL-family) mapping — also used by RCCL, Hygon DCU and oneCCL v2
/ ECCL (numerically identical per vendor source):

| UniCCL | value | meaning |
|---|---|---|
| `UNICC_I8` | 0 | int8 |
| `UNICC_U8` | 1 | uint8 |
| `UNICC_I32` | 2 | int32 |
| `UNICC_U32` | 3 | uint32 |
| `UNICC_I64` | 4 | int64 |
| `UNICC_U64` | 5 | uint64 |
| `UNICC_F16` | 6 | half |
| `UNICC_F32` | 7 | float32 |
| `UNICC_F64` | 8 | float64 |
| `UNICC_BF16` | 9 | bfloat16 |
| `UNICC_SUM` / `UNICC_PROD` / `UNICC_MAX` / `UNICC_MIN` | 0 / 1 / 2 / 3 | reduce ops |

(NCCL and oneCCL v2 list exactly these values; earlier versions of this
document carried the older NCCL numbering F16=9/BF16=10, which does not match
modern NCCL — corrected here.)

Diverging backends grow a real table behind the same accessors in the P2
milestone: Moore/MetaX MCCL use the RCCL-style numbering with `Uint32` inserted
(Int8=0, Uint8=1, Int32=2, Uint32=3, Int64=4, Uint64=5, Float16=6, Float32=7,
Float64=8, Bfloat16=9), Cambricon CNCL uses a hex coding (Int8=0x20, Float32=0x12,
…), Ascend HCCL its own layout — all recorded from vendor source in
`docs/official/`. The hot path accessors stay put; only a divergent table is
added.

## Bootstrap id: length-carrying (`unicc_comm_id_t`)

Backends disagree on how large the bootstrap id is: NCCL family 128 bytes,
Intel oneCCL v2 4096 bytes, CNCL cliqueId 136, HCCL rootInfo 4108. UniCCL
therefore carries ids as `unicc_comm_id_t { size_t len; unsigned char
data[UNICC_COMM_ID_MAX] }` (max 4112); `unicc_get_unique_id` produces it,
applications transport `data` + `len` verbatim, `unicc_comm_init_rank` consumes
it. The id slots are the cold path: only `get_unique_id` / `comm_init_rank`
touch it, through small per-backend adapters (memcpy into the native-sized id),
never the collectives.

## Error mapping

Backend results are ints where `0 == success`. UniCCL maps `0 → UNICC_OK` and any
non-zero → `UNICC_ERR_UNHANDLED_BACKEND`, keeping the raw value for
`unicc_get_last_error`. The wrapper's own errors (`NOT_INITIALIZED`,
`NOT_SUPPORTED`, …) are returned unchanged.

## Platform support

| Platform | nccl | rccl | oneccl | eccl |
|---|:---:|:---:|:---:|:---:|
| Linux | yes | yes | yes | yes |
| macOS | no (NCCL ships Linux-only; the loader tries the `.dylib` spelling for locally-built libs) | no (ROCm) | unverified | no |
| Windows | not yet (planned) | no | no | no |

## Deliberate simplifications

- **Default backend is a fixed name**, not a CUDA/ROCm/oneAPI scan.
- **Vendor identity probe pending**: distinguishing NVIDIA NCCL from AMD RCCL
  from Hygon DCU (all `nccl*`-named today) needs a dependency/vendor probe,
  not the symbol prefix — recorded as future work; until then
  `unicc_backend_name()` reports the resolved family name (e.g. `nccl` for any
  of the three).
- **Version gating is runtime-only**: `unicc_backend_version()` +
  document-level checks. No compile-time gate machinery (the vendor API
  surfaces are small; UniMPI's MPI-version gating was judged unnecessary).
- **Error-string passthrough** is pending (per vendor); the raw backend result
  is surfaced today.
- **ECCL versioning**: oneCCL numbers its releases by year (2022.1); ECCL
  numbers like NCCL — both read directly from the backend.

## Diagnosing on hosts with no NCCL/RCCL

```bash
UNICC_BACKEND=nccl ./build/minimal        # observe the loader messages
./build/minimal && ...                   # or call unicc_diagnose() from code
```

`UNICC_LIBRARY` pointing at a missing file, `libnccl.so` absent from the loader
path, or a wrong soname are the common causes — `unicc_platform_load_advice()`
describes the standard checks (`ldconfig -p | grep -E 'nccl|rccl'`, `ldd`, …).
