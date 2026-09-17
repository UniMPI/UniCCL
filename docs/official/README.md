# Official source records (XCCL)

> Networked evidence pinned to this repo so the design facts in `BACKENDS.md`
> and `SUPPORT_MATRIX.md` stay traceable to their primary sources. Each record
> states its URL, retrieval date, and what claim it backs. This is the
> authoritative-facts discipline the project follows: claims about a backend's
> API shape are quoted from vendor documentation or source, never from
> folklore or from our own fixtures.

| File | Source / URL | Facts it backs |
|---|---|---|
| `amdc-rccl-official.md` | ROCm RCCL 2.30.4 documentation (rocm.docs.amd.com) | RCCL's public API surface is `nccl*`-named (functions, types, macros, `nccl.h`); RCCL version tracks the NCCL numbering; repo retired into `ROCm/rocm-systems/projects/rccl` |
| `amdc-rccl-source-evidence.md` | GitHub code search over `ROCm/rocm-systems` | `rccl*`-prefixed public symbols are absent; `rccl*` survives only as internal implementation names (`rcclAllReduceShouldTakeDdaPath`, `RcclTunableColls`, `rccl_wrap.cc`) |

## Why these records exist

The M2 skeleton originally assumed *"real RCCL exports a `rccl*` native family
plus an `nccl*` ABI-compatibility layer"*, and built recognition on
`rcclGetVersion` (checked before `ncclGetVersion`). The official documentation
and source contradict that model: **modern AMD RCCL's public API is `nccl*`** —
same symbol prefix as NCCL itself. The recognition rule and the `rccl*` binding
were written for a shape that does not exist on current AMD RCCL.

These records pin down what the vendors actually say, so the wrapper can be
corrected against evidence rather than assumption. The practical consequences:

- On a real NCCL, real AMD RCCL, and Hygon DCU, the same `nccl*`-named API
  family is the genuine native surface. A wrapper that loads "each backend's own
  native symbols" therefore uses the `nccl` binding for all three — identity
  (which vendor shipped the library) is a separate question from the symbol
  family it exports.
- Distinguishing NVIDIA NCCL from AMD RCCL / Hygon DCU at runtime needs a
  vendor probe other than `rccl*` (dependency fingerprint, runtime `dlerror`
  chains, or a vendor-specific symbol) — see `BACKENDS.md`.

## Refresh procedure

Records are snapshots, not mirrors. To refresh or extend:

1. Re-fetch the documented URLs below; keep the page/version you cite.
2. Record the date and the version shown on the page.
3. Update the affected claim rows in `BACKENDS.md` / `SUPPORT_MATRIX.md` if the
   facts moved.
