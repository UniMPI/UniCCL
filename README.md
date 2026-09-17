# XCCL — unified collective-communication layer (NCCL / RCCL)

XCCL wraps NVIDIA **NCCL** and AMD **RCCL** (and, in time, domestic-GPU
collective libraries) behind a single semantic `xcc_*` API. A backend is
chosen at run time via the environment; the wrapper loads it with `dlopen`,
binds its symbols through a vtable, and degrades gracefully when a symbol is
missing. Upper layers (notably UMC, the Unified Memory & Communication
middleware) depend only on `include/xcc.h` and never touch `nccl.h` /
`rccl.h` / CUDA / ROCm.

This is the **M2 skeleton**: platform abstraction, runtime backend loading,
the minimal `xcc_*` collective face, `nccl` / `rccl` bindings and a fake-backend
unit suite that runs on any Linux host with no GPU at all.

```
┌─────────────────────────┐
│ upper layer / UMC       │   depends only on xcc.h
└────────────┬────────────┘
             │  xcc_* API (xcc_api.c)
┌────────────▼────────────┐
│ XCCL wrapper            │   loader + vtable + availability gates
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
| 1 | dlopen/dlsym platform abstraction | `src/xcc_platform_posix.c` / `_windows.c` |
| 2 | backend table (nccl, rccl) + soname fallback | `src/xcc_loader.c` |
| 3 | selection priority `XCCL_LIBRARY` → `XCCL_BACKEND` → default | `src/xcc_loader.c` |
| 4 | identify-by-feature-symbol (**rccl\* checked before nccl\*, see BACKENDS.md**) | `src/xcc_loader.c` |
| 5 | zero-initialized vtable + core check + per-backend dispatch | `src/xcc_vtable.c` |
| 6 | dlsym bindings, missing symbol → NULL | `src/backends/nccl.c`, `src/backends/rccl.c` |
| 7 | unified `xcc_*` semantic API + `*_available()` gates | `src/xcc_api.c`, `include/xcc.h` |
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
compiles anywhere): `-DXCCL_BUILD_TESTS_INTEGRATION=ON`.

## Quick local run (no GPU needed)

```bash
XCCL_LIBRARY=build/tests/fake/fake_nccl_identity.so ./build/minimal
XCCL_LIBRARY=build/tests/fake/fake_rccl_identity.so   ./build/minimal
```

Real backend:

```bash
XCCL_BACKEND=nccl ./build/minimal          # NVIDIA host
XCCL_BACKEND=rccl ./build/minimal          # AMD / ROCm host
```

## Runtime backend selection

1. `XCCL_LIBRARY` — exact library path or loader-resolvable name
   (the CI / test path);
2. `XCCL_BACKEND` — `nccl` or `rccl`;
3. platform default — NVIDIA NCCL (`libnccl.so`, falling back to
   `libnccl.so.2`). CUDA/ROCm-presence defaulting is left for a later
   milestone.

See `docs/BACKENDS.md` for the full rules, including the RCCL
dual-symbol (`rccl*` + `nccl*` compat) identification subtlety.

## Repo layout

```
include/     public contract: xcc.h, xcc_vtable.h, xcc_backends.h,
             xcc_platform.h, xcc_errors.h, xcc_version.h
src/         loader, vtable, api, platform, backends/{nccl,rccl}.c
tests/       fake fixtures + unit tests + integration test + runner
examples/    minimal.c
docs/        API.md, BACKENDS.md, SUPPORT_MATRIX.md
```

## Status

- Verified on this host: loader / vtable / fake-backend unit tests (all
  green), `minimal` against both fake fixtures.
- Verified on real hardware: NCCL 2.28.7 world=2 allreduce PASSED on `iota`
  (2× RTX PRO 6000 Blackwell, via HPC SDK bundled NCCL) — see
  `docs/SUPPORT_MATRIX.md` for the full record.
- Not yet verified: real RCCL / DCU collective library (pending the DCU host
  `centos-8-dcu` and its collective-library form).

## Milestone map (from the D2 unified-communication-stack decision)

- **M2.1** platform + loader + vtable ✅
- **M2.2** minimal `xcc_*` API + nccl/rccl bindings + fake tests ✅
- **M2.3** real-backend integration + SUPPORT_MATRIX (runner + harness in place;
  execution on GPU images pending)
- **M2.4** UMC pilot integration (next)
