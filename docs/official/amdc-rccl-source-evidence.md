# AMD RCCL source evidence — GitHub code search

> Retrieval date: **2026-09-17**, via the GitHub code-search API
> (`gh api search/code`). Queries run against both the retired `ROCm/rccl`
> repository and the canonical `ROCm/rocm-systems` repository (develop).

## The question

Does the RCCL source tree export public `rccl*` symbols (e.g.
`rcclGetVersion`, `rcclCommInitRank`, `rcclBroadcast`), or is the public
surface `nccl*`-only?

## Query results

| Query | Hits | Read |
|---|---|---|
| `repo:ROCm/rocm-systems rcclGetVersion` | 0 | RCCL has no `rcclGetVersion` anywhere |
| `repo:ROCm/rocm-systems rcclCommInitRank` | 0 | no `rcclCommInitRank` |
| `repo:ROCm/rocm-systems rcclGetUniqueId` | 0 | no `rcclGetUniqueId` |
| `repo:ROCm/rocm-systems rcclBroadcast` | 0 | no `rcclBroadcast` |
| `repo:ROCm/rocm-systems rcclGroupStart` | 0 | no `rcclGroupStart` |
| `repo:ROCm/rocm-systems rcclCommDestroy` | 0 | no `rcclCommDestroy` |
| `repo:ROCm/rocm-systems ncclGetVersion` | 33 | the `nccl*` implementation family is the real one |
| `repo:ROCm/rocm-systems rcclAllReduce` | 7 | all internal/wrap/decision code, not a public API |
| `repo:ROCm/rccl rcclGetVersion` | 0 | consistent in the retired repo too |
| `repo:ROCm/rccl rcclAllReduce` | 1 | single reference, not an API export |

## Where the `rccl*` names actually live

`rcclAllReduce` hits (all internal to the implementation):

```
projects/rccl/src/device/all_reduce.h
projects/rccl/src/rccl_wrap.cc
projects/rccl/src/collectives.cc
projects/rccl/src/include/rccl_common.h
projects/rccl/test/host/fakes/wrap_fakes.cc
projects/rccl/test/RcclWrapTests.cpp
projects/rccl/test/host/wrap-test.cc
```

- `src/include/rccl_common.h` starts with `#include "nccl.h"` and defines
  *internal* plumbing: `RcclTunableColls` (an internal tunable-collective
  index enumeration) — an internal name, not an exported API.
- `src/rccl_wrap.cc` is the wrap layer that glues the implementation to the
  `nccl.h` entry points; its `rccl` occurrences are internal helpers such as
  `rcclAllReduceShouldTakeDdaPath(comm, count, datatype, ...)` (a path-choice
  predicate), not callable public symbols.
- The test tree (fakes, RcclWrapTests, wrap-test) exercises the *wrap* layer —
  i.e. even AMD's own tests drive the `nccl*` surface.

## Conclusion

As of the retrieval date, RCCL implements and exposes **only the `nccl*`
public symbol family**; `rccl*` exists exclusively inside the implementation
(internal helpers, the wrap layer, tuner enums). No public `rccl*` API named
in the old assumption (`rcclGetVersion`, `rcclCommInitRank`, `rcclBroadcast`,
…) is present in the source.

Implication for UniCCL: the "identify by `rccl*` presence" rule has no real
target on current AMD RCCL. Real RCCL is `nccl*`-only, exactly like Hygon
DCU's `nccl*`-compat layer — so on a first pass UniCCL would identify both
as NCCL, and vendor identity must come from a separate probe (see
`BACKENDS.md`).

## Caveats

- Code search is a snapshot; RCCL may reintroduce or have shipped `rccl*`
  symbols in some release. The definitive per-binary check is
  `nm -D librccl.so | grep -E '^rccl|^nccl'` on a real AMD host (see
  `SUPPORT_MATRIX.md` "RCCL (native AMD) status").
- `ROCm/rccl` is retired and may not be indexed in full; cross-checking both
  repos agreed on zero for every `rccl*` public symbol we queried.
