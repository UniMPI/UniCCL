# UniCCL — a unified entry point for any collective-communication library

UniCCL's point is **not** to make everything "be NCCL". It is to expose one
semantic `unicc_*` API over *any* collective-communication library — NVIDIA
**NCCL**, AMD **RCCL**, and in time domestic-GPU collective libraries — while
each backend **loads and binds its own native symbol family**. A backend is
chosen at run time via the environment; the wrapper loads it with `dlopen`,
binds its symbols through a vtable, and degrades gracefully when a symbol is
missing. Upper layers (notably UMC, the Unified Memory & Communication
middleware) depend only on `include/unicc.h` and never touch `nccl.h` /
`rccl.h` / CUDA / ROCm.

One fact worth stating up front (evidence in `docs/official/`): NVIDIA NCCL,
AMD RCCL *and* Hygon's DCU collective library all expose the **`nccl*`-named**
public API — for RCCL this *is* the genuine native API, not a compatibility
shim. So "load each backend's native symbols" resolves to the `nccl` binding
for all three today; the *vendor identity* of a loaded library is a separate
question, answered by its own probe, not by the symbol prefix.

This is the **M2 skeleton**: platform abstraction, runtime backend loading,
the minimal `unicc_*` collective face, `nccl` / `rccl` bindings and a fake-backend
unit suite that runs on any Linux host with no GPU at all.

```
┌─────────────────────────┐
│ upper layer / UMC       │   depends only on unicc.h
└────────────┬────────────┘
             │  unicc_* API (unicc_api.c)
┌────────────▼────────────┐
│ UniCCL wrapper            │   loader + vtable + availability gates
└──────┬────────────┬─────┘
       │            │
  backends/        backends/
  nccl.c           rccl.c
       │            │
   libnccl.so   librccl.so     (dlopen'd at run time)
```

## Feature map (milestone M2)

| # | Mechanism | Where |
|---|-----------|-------|
| 1 | dlopen/dlsym platform abstraction | `src/unicc_platform_posix.c` / `_windows.c` |
| 2 | backend table (nccl, rccl) + soname fallback | `src/unicc_loader.c` |
| 3 | selection priority `UNICC_LIBRARY` → `UNICC_BACKEND` → default | `src/unicc_loader.c` |
| 4 | identify-by-feature-symbol (**rccl\* checked before nccl\*, see BACKENDS.md**) | `src/unicc_loader.c` |
| 5 | zero-initialized vtable + core check + per-backend dispatch | `src/unicc_vtable.c` |
| 6 | dlsym bindings, missing symbol → NULL | `src/backends/nccl.c`, `src/backends/rccl.c` |
| 7 | unified `unicc_*` semantic API + `*_available()` gates | `src/unicc_api.c`, `include/unicc.h` |
| 8 | fake-backend unit tests (host, no GPU) | `tests/` |
| 9 | real-backend integration test (opt-in) | `tests/integration/` |

## Build

No vendor dependencies at all (C99 + CMake):

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure    # unit suite, all green
```

Optional integration build (requires a real NCCL/RCCL host to *run*; it still
compiles anywhere): `-DUNICC_BUILD_TESTS_INTEGRATION=ON`.

## Quick local run (no GPU needed)

```bash
UNICC_LIBRARY=build/tests/fake/fake_nccl_identity.so ./build/minimal
UNICC_LIBRARY=build/tests/fake/fake_rccl_identity.so   ./build/minimal
```

Real backend:

```bash
UNICC_BACKEND=nccl ./build/minimal          # NVIDIA host
UNICC_BACKEND=rccl ./build/minimal          # AMD / ROCm host
```

## Runtime backend selection

1. `UNICC_LIBRARY` — exact library path or loader-resolvable name
   (the CI / test path);
2. `UNICC_BACKEND` — `nccl` or `rccl`;
3. platform default — NVIDIA NCCL (`libnccl.so`, falling back to
   `libnccl.so.2`). CUDA/ROCm-presence defaulting is left for a later
   milestone.

See `docs/BACKENDS.md` for the full rules, including the RCCL
dual-symbol (`rccl*` + `nccl*` compat) identification subtlety.

## Repo layout

```
include/     public contract: unicc.h, unicc_vtable.h, unicc_backends.h,
             unicc_platform.h, unicc_errors.h, unicc_version.h
src/         loader, vtable, api, platform, backends/{nccl,rccl}.c
tests/       fake fixtures + unit tests + integration test + runner
examples/    minimal.c
docs/        API.md, BACKENDS.md, SUPPORT_MATRIX.md, official/ (vendor
             doc/source records backing the backend facts)
```

## Status

- Verified on this host: loader / vtable / fake-backend unit tests (all
  green), `minimal` against both fake fixtures.
- Verified on real hardware:
  - NCCL 2.28.7 world=2 allreduce PASSED on `iota` (2× RTX PRO 6000 Blackwell,
    via HPC SDK bundled NCCL);
  - Hygon DCU world=2 allreduce PASSED on `centos-8-dcu` (3× DCU, DTK 23.10.1
    `librccl.so` — a pure `nccl*`-compat layer, driven as "nccl" with no
    interface change; both DTK 23.10.1 and 25.04 export no `rccl*`/`hccl*`).
  Full records in `docs/SUPPORT_MATRIX.md`.
- Not yet verified: native AMD RCCL (`rccl*` symbols) — no AMD host assigned;
  Hygon DCU has no `rccl*` to verify against.

## Milestone map (from the D2 unified-communication-stack decision)

- **M2.1** platform + loader + vtable ✅
- **M2.2** minimal `unicc_*` API + nccl/rccl bindings + fake tests ✅
- **M2.3** real-backend integration + SUPPORT_MATRIX (runner + harness in place;
  execution on GPU images pending)
- **M2.4** UMC pilot integration (next)
