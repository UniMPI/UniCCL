# AMD RCCL official documentation — snapshot

> Retrieval date: **2026-09-17**. Data fetched via HTTP, converted to Markdown
> (navigation chrome removed). Full pages live at the URLs listed; this file is
> the curated, verifiable excerpt that XCCL's design facts cite.

## Recorded pages

| Page | URL | Notes |
|---|---|---|
| RCCL doc index | https://rocm.docs.amd.com/projects/rccl/en/latest/ | version banner: **RCCL 2.30.4 Documentation** |
| What is RCCL? | https://rocm.docs.amd.com/projects/rccl/en/latest/what-is-rccl.html | scope/feature description |
| Library specification | https://rocm.docs.amd.com/projects/rccl/en/latest/api-reference/library-specification.html | function list, all `nccl*`-named |
| API library | https://rocm.docs.amd.com/projects/rccl/en/latest/api-reference/api-library.html | types/enums/macros/functions, all `nccl*`-named; header `nccl.h.in` |
| RCCL GitHub README | https://github.com/ROCm/rccl (redirected to rocm-systems notice) | repo retired; canonical source now `ROCm/rocm-systems` |

## The fact this record pins down

> **RCCL's public, documented API surface is `nccl*`-named.** On RCCL 2.30.4,
> every documented function, type, macro and the public header use the `nccl`
> prefix — `ncclAllReduce`, `ncclBroadcast`, `ncclCommInitRank`,
> `ncclGetUniqueId`, `ncclGroupStart`, `ncclCommDestroy`, `ncclCommCount`,
> `ncclCommUserRank`, `ncclResult_t`, `ncclRedOp_t`, `ncclDataType_t`,
> `ncclConfig_t`, `ncclUniqueId`, … There is no `rccl*` entry in the public API
> reference.

Supporting quotes from the recorded pages:

- `ncclGetVersion` (API library): *"Return the RCCL_VERSION_CODE of RCCL in
  the supplied integer. This integer is coded with the MAJOR, MINOR and PATCH
  level of RCCL."* — i.e. the NCCL-named symbol is the version entrypoint of
  *RCCL itself*.
- API library header block: `_file_ nccl.h.in` with `NCCL_MAJOR /
  NCCL_MINOR / NCCL_PATCH / NCCL_VERSION_CODE / NCCL_VERSION(X,Y,Z)` — RCCL
  ships a header literally named/derived from `nccl.h`.
- Datatype enum is `ncclDataType_t` with `ncclInt8 … ncclBfloat16 …
  ncclFloat8e4m3 … ncclFloat8e5m2`; reduce-op enum is `ncclRedOp_t` with
  `ncclSum / ncclProd / ncclMax / ncclMin / ncclAvg`; result enum is
  `ncclResult_t` with `ncclSuccess / ncclUnhandledCudaError / …` — the same
  nominal values NCCL uses.
- Collective signatures embed the HIP stream type (`hipStream_t`), e.g.
  `ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t
  stream)`.

## Version alignment

RCCL's documentation banner reads **"RCCL 2.30.4"** while shared-vendor
cluster stacks are typically NCCL 2.x as well (e.g. NCCL 2.28.7 on our `iota`
host). RCCL tracks the NCCL-compatible numbering — itself evidence that RCCL
positions itself as the NCCL-compatible collective library on ROCm.

## Repository status

The ROCm RCCL GitHub README opens with a literal warning:

> "The rccl repository is retired, please use the ROCm/rocm-systems
> repository"

and the follow-up note:

> "The RCCL public repository is located within the rocm-systems repo at
> ROCm/rocm-systems (projects/rccl)."

RCCL documentation says it lives in
`github.com/ROCm/rocm-systems/tree/develop/projects/rccl`; plugin examples are
under `projects/rccl/plugins/{tuner,net}/example`.

## What this record does NOT claim

- It does not claim whether a specific shipped binary still exports any
  `rccl*` symbol (that needs `nm -D librccl.so` on a real AMD host). See
  `amdc-rccl-source-evidence.md` for the source-level answer today.
- It does not reconstruct the historical `rccl*`-era API (pre-NCCL-compat
  RCCL). Whether any real deployment still depends on `rccl*` symbols is an
  open, host-level question.

## How to record a newer snapshot

Re-fetch the URLs above on the next survey, note the version banner, and bump
the rows in `BACKENDS.md` / `SUPPORT_MATRIX.md` that cite this record.
