# Collective-communication ecosystem survey — does every vendor use NCCL?

> Survey date: **2026-09-17**. Two parallel investigations (domestic-GPU
> panorama + Intel / industry-standard status). Sources: vendor documentation,
> vendor apt repositories (fetched directly), vendor GitHub source, PyTorch
> source, and secondary community material. Every claim is tagged with an
> evidence grade; "no authoritative evidence" is written out explicitly.
> This record answers the strategy question *"if everyone uses NCCL, do we
> still need the unified collective layer?"* — the short answer is **yes,
> but with a recalibrated thesis**, because the premise is false at the
> symbol level. (The layer, written as *XCCL* throughout the original
> survey, was renamed **UniCCL** on 2026-09-17 — see "Naming hazard".)

## Headline

- **Symbol/ABI level — NOT everyone uses `nccl*`.** Only NVIDIA NCCL, AMD
  RCCL (a fork of NCCL; its official API reference lists 69 `nccl*`
  functions) and Hygon DCU's DTK `librccl.so` share the `nccl*` symbol
  family. Everyone else ships its own prefix: Intel `oneccl*`, Cambricon
  `cncl*`, Ascend `hccl*`, Moore Threads `mccl*`, MetaX `mccl*`, Enflame
  `eccl*`, Iluvatar `ixcc*` (low confidence). **No domestic vendor
  officially exports `nccl*` symbols or claims binary compatibility.**
- **Contract/semantic level — everyone mirrors NCCL.** All training-card
  vendors align primitive sets, environment variables and PyTorch backend
  names to NCCL: Intel oneCCL's new C API "closely follows the NVIDIA NCCL
  API standard" (official), Moore Threads' MCCL and MetaX's MCCL are renamed
  ports of NCCL, and PyTorch registers `cncl`/`hccl`/`mccl`/`eccl` backends
  plus a built-in Intel `"xccl"` backend.
- **Convergence verdict: "NCCL is the de-facto reference" — strong; "all
  ecosystems are symbol-level `nccl*`" — false.**

## Landscape table

| Vendor | Collective lib | Own prefix | `nccl*` symbols? | NCCL alignment | Evidence grade | Notes / source |
|---|---|---|---|---|---|---|
| NVIDIA | NCCL | `nccl*` | native | — | verified | `PASS` on iota (2.28.7) |
| AMD | RCCL (NCCL fork) | `nccl*` | yes (fork; 69 `nccl*` fns) | full | high (official) | 2.30.4 docs; ROCm/rccl, rocm-systems |
| Hygon DCU | DTK `librccl.so` | `nccl*` | yes (only family) | full | verified (real host) | `version=21304`, `sum=3` |
| Intel | oneCCL (v2 C API) / PyTorch built-in **XCCL** ("xccl") | `oneccl*` | no | strong (official: "closely follows NCCL API standard") | high (official) | uxlfoundation/oneCCL `include/oneapi/ccl.h`; pytorch `distributed_c10d.py` `Backend.XCCL`; intel/torch-xpu-ops `src/xccl` |
| Cambricon 寒武纪 | **CNCL** (Neuware/CNToolkit) | `cncl*` | no | strong (same-shape API: `cnclCliqueId_t`, `cnclInitComms`…) | high (official source) | Cambricon/torch_mlu: `cncl_utils.cpp`, `ProcessGroupCNCL.cpp`, `FindCNCL.cmake` |
| Ascend 昇腾 | **HCCL** (CANN) | `hccl*` | no | strong/medium (community says NCCL-compatible; officially own API) | medium-high | Ascend/pytorch (torch_npu, `TORCH_HCCL_*`), community |
| Moore Threads 摩尔线程 | **MCCL** (MUSA) | `mccl*` | no | renamed port of NCCL (residual `nccl*` calls in source) | high (official source) | MooreThreads/torch_musa `ProcessGroupMCCL.cpp`; **official apt: `mccl-s4000` / `mccl-s5000`** |
| MetaX 沐曦 | **MCCL** (MetaX) + `mxccl_plugin` | `mccl*` | plugin claims "compatible with NCCL 21605" (to verify) | strong (env vars mirror `NCCL_*`), official doc "对应 NVIDIA 的 NCCL" | high (official apt + docs) | **official apt: `mccl_*`, `mxccl_plugin_*` — "Maca Collective Communication Lib plugin which is compatible with NCCL 21605"**; ms-swift Metax docs |
| Enflame 燧原 | **ECCL** (TopsRider) | `eccl*` | no | strong (official maps backend "nccl"→"eccl") | high (official README) | EnflameTechnology/torch-gcu (`dist.init_process_group("eccl")`) |
| Iluvatar 天数智芯 | IXCCL (name unconfirmed) | `ixcc*`? | unknown | unknown | low (community) | no official CCL repo found; low-confidence articles |
| Biren 壁仞 | unconfirmed (SCCL/BLink in one low-confidence article) | unknown | unknown | unknown | low | no official CCL evidence |
| JingJiaWei 景嘉微 | none | — | no | n/a | not applicable | graphics/inference positioning only |
| Apple MPS | none (PyTorch falls back to Gloo) | — | no | no | medium | pytorch `default_device_backend_map["mps"]=GLOO` |
| Google TPU | XLA/GSPMD (own path) | — | no | no | medium | XLA collective semantics |
| AWS Trainium / Tesla Dojo | own stacks | — | no | no | no authoritative public evidence found | n/a |

## Local first-hand evidence (from this repo's ecosystem)

The vendor apt repositories surfaced via this repo's docker definitions
(`~/Projects/tf-qflux/docker/base/*-maca`, `*-musa`):

- **MetaX official apt** (`repos.metax-tech.com/r/maca-sdk-deb`): packages
  `mccl_*` (collective lib) and `mxccl_plugin_*` whose description reads
  verbatim: *"Maca Collective Communication Lib plugin which is compatible
  with NCCL 21605."* (NCCL 2.16.5.) Also ships `mxucx` (UCX port),
  `mxompi` (OpenMPI port), `mxshmem`, `metax-fabricmanager`.
- **Moore Threads official apt** (`dl.mthreads.com/repo/.../ubuntu2204`):
  packages `mccl-s4000` / `mccl-s5000` (+ `-dev`, `-bench`) — MCCL is
  versioned per GPU family (S4000/S5000), aligned with the MUSA toolkit.

These are the strongest primary evidence that domestic vendors ship **their
own** collective libraries (own prefixes / own packages) rather than
repackaging `libnccl.so`, and that "compatible with NCCL" is a *claim about
the NCCL contract*, not a statement of `nccl*` ABI export (to be confirmed
on real hardware by `nm -D`).

## Do we still need it (UniCCL)? — reasoned answer

- **Premise check**: the worry "if everyone is NCCL, no unified layer is
  needed" is **disproven at the symbol level** — Intel and every domestic
  training-card vendor deliberately keep their own prefixes; only AMD/Hygon
  reuse `nccl*` through a fork / compat layer.
- **Case split**:
  - If UniCCL's consumers are only NVIDIA/AMD/Hygon (all `nccl*`): value is
    thin — firmware-ish: fixed sonames, one identity/capability gate. Limited.
  - If UniCCL must let upper layers (UMC) run on *any* CCL (Intel + domestic
    cards): **UniCCL is necessary**. The upper layer faces five-plus prefixes
    (`oneccl`/`cncl`/`hccl`/`mccl`/`eccl`) and distinct `.so` names — exactly
    the "unified entry + per-backend native symbols" model the project
    adopted.
- **Where the high value is** (recalibrated): (1) **backend matrix** — one
  binding file per family (`oneccl`/`cncl`/`hccl`/`mccl`/`eccl`); (2)
  **semantic-contract mapping** — since all vendors mirror the NCCL contract,
  the `xcc_*` enum/op/error mapping tables become the load-bearing piece;
  (3) **load/identity/capability layering** — the already-registered vendor
  identity fingerprint (dependency-based) becomes necessary, not nice-to-have.
- **Explicit ceiling (honesty)**: no effort should go into `nccl*` ABI
  compatibility (no vendor endorses it; only the AMD fork benefits). If the
  ecosystem ever truly ABI-unifies (unlikely — AMD/Intel won't drop their
  brands), UniCCL degrades to a loading layer and should be re-examined on
  cost grounds then.

## Open items (verification list)

1. **AMD RCCL per-binary symbol face**: `nm -D librccl.so` on a real AMD host
   (settles whether any shipped RCCL still exports `rccl*`; fake-only today).
2. **MetaX `mxccl_plugin`**: does it actually export `nccl*` symbols (as
   "compatible with NCCL 21605" hints), or `mccl*` only? Needs a real MetaX
   host `nm -D`.
3. **Moore Threads `mccl-s*`**: project has no `nccl*` export confirmed;
   needs a real host `nm -D`.
4. **Iluvatar / Biren**: no authoritative CCL evidence found; treat as
   unknown until a real software stack is examined.

## Naming hazard (recorded) — resolved by the rename

PyTorch registers a built-in Intel backend named **XCCL** (`"xccl"`), which
is oneCCL's entry into PyTorch for Intel XPU. The unified layer was *also*
originally called XCCL, and the two are unrelated. On **2026-09-17** the
project was renamed to **UniCCL** — API prefix `unicc_`, headers
`include/unicc*.h`, env `UNICC_*`, CMake target `unicc::unicc` — precisely
to avoid that collision, and as a sibling of the UniMPI naming family this
mechanism is modeled on. Any "XCCL" above that does not refer to PyTorch's
backend is historical text for this layer written before the rename.

## Source URLs

- AMD RCCL: https://rocm.docs.amd.com/projects/rccl/en/latest/ (+ `api-reference/api-library.html`) · https://github.com/ROCm/rccl · ROCm/rocm-systems
- Intel oneCCL: https://uxlfoundation.github.io/oneCCL/index.html · https://github.com/uxlfoundation/oneCCL (`include/oneapi/ccl.h`)
- PyTorch: https://github.com/pytorch/pytorch (`torch/distributed/distributed_c10d.py`, `Backend.XCCL`, `default_device_backend_map`) · https://docs.pytorch.org/docs/2.14/distributed.html
- Intel XCCL impl: https://github.com/intel/torch-xpu-ops (`src/xccl/`, `src/xccl/xccl.h`)
- Cambricon: https://github.com/Cambricon/torch_mlu (`cmake/modules/FindCNCL.cmake`, `.../distributed/cncl_utils.cpp`, `process_group_cncl.cpp`)
- Ascend: https://github.com/Ascend/pytorch (torch_npu)
- Moore Threads: https://github.com/MooreThreads/torch_musa (`ProcessGroupMCCL.cpp`) · official apt `dl.mthreads.com/repo/repository/ubuntu2204/pool/jammy/amd64/` (`mccl-s4000`, `mccl-s5000`)
- MetaX: official apt `https://repos.metax-tech.com/r/maca-sdk-deb/dists/stable/main/binary-amd64/Packages` (`mccl_*`, `mxccl_plugin_*`, description "compatible with NCCL 21605") · developer.metax-tech.com · ms-swift MetaX support docs
- Enflame: https://github.com/EnflameTechnology/torch-gcu (README: backend "eccl", maps "nccl"→"eccl")
- Community (secondary): juejin.cn/post/7632249669431214134 (NCCL as de-facto standard, domestic stacks), CSDN articles for CNCL/HCCL/IXCCL usage.
