# Sisyphus Compiler Design

## Overview

Sisyphus is a SysY compiler for both ARM64 (AArch64) and RISC-V (rv64gc). It uses a unified frontend and IR pipeline, then lowers to architecture-specific backends.

Build target binary name is `compiler`.

## CLI Contract

Mandatory competition-compatible invocations:

```bash
compiler testcase.sy -S -o testcase.s
compiler testcase.sy -S -o testcase.s -O1
compiler testcase.sy -S -o testcase.s -O2
```

Additional controls:

- `--target=riscv|arm` (default from CMake `DEFAULT_TARGET`)
- `-O0`, `-O1`, and `-O2`
- `--emit-ir`
- `--dump-pass-timing`
- `--verify-ir`
- `--inline-threshold`, `--late-inline-threshold`
- `--disable-loop-rotate`, `--enable-loop-rotate`, `--disable-const-unroll`
- `--enable-experimental` (kept opt-in)
- `--enable-hir-pipeline`, `--disable-hir-pipeline`, `--use-legacy-codegen`
- `--dump-hir`, `--dump-cfg`, `--verify-hir`, `--verify-cfg`

## Pipeline

### Frontend Staging (default)

Default frontend path is dialect-based:

- `AST -> HIR Build -> HIR Verify -> HIR Canonicalize -> HIR Verify -> HIR->CFG -> CFG Verify -> CFG->Legacy ModuleOp`
- Stage boundaries emit timing with `--dump-pass-timing` as `[stage-timing] ...`
- Legacy frontend remains available with `--use-legacy-codegen` for A/B rollback during migration

The key architectural split is intentional:

- HIR keeps structured loops and array accesses visible long enough for affine
  analysis and loop rewrites.
- CFG provides explicit control-flow for SSA, MemorySSA-style analysis,
  scalar optimization, and verifier checks.
- The legacy `ModuleOp` adapter lets the mature mid-end and backends continue
  to operate while the dialect pipeline is incrementally expanded.

### O0 (stable baseline)

- MoveAlloca
- EarlyConstFold / Pureness
- RaiseToFor / DCE / Lower
- FlattenCFG
- Mem2Reg
- RegularFold / DCE / SimplifyCFG / Select / InstSchedule
- Target lowering (ARM or RISC-V)

### O1 (performance pipeline)

- Structured CFG optimization: const fold, inlining, loop cleanup, memory tidy, scalar transforms
- FlattenCFG + SSA conversion (Mem2Reg)
- Alias + DSE + DLE + DAE + GVN + LoopSimplify-style canonicalization,
  hardened LoopRotate, LICM, and loop transforms
- Late inline and cleanup rounds
- Final schedule + target backend passes

### O2 (aggressive profile on top of O1)

- Structured stage: enable `Fusion` and `Unswitch`
- Mem2Reg stage: add `Reassociate`
- Select stage: enable `Range + RangeAwareFold + Splice` by default
- Tail stage: add one more `CanonicalizeLoop -> LICM -> SCEV -> GVN -> RegularFold`
- `Cached` remains experimental-only (`--enable-experimental`)
- Defaults: inline/late-inline thresholds are `256/256` unless explicitly overridden.
  RISC-V O1 enables hardened LoopRotate by default; `--disable-loop-rotate`
  or `SISY_ENABLE_LOOP_ROTATE=0` disables the default path for bisection.
  Store-containing loop rotation is kept behind
  `SISY_ENABLE_LOOP_ROTATE_STORES=1` until MemorySSA and guaranteed-execute
  checks can discharge conditional store cases.

### Compliance Boundary In The Pipeline

Default O1/O2 should only enable general compiler transformations. Semantic
whole-function recognizers and algorithm helper replacements are strict-mode
experiments, not architectural requirements. The intended optimization stack is:

- HIR: affine loop legality, reduction privatization, fusion/interchange,
  guarded-loop simplification, and optional tiling after dependence proof.
  `AffineNestAnalysis` is the shared MLIR-like side table for loop domain,
  access-rank, contiguity/stride, side-effect, reduction, and pressure summaries;
  individual Linalg/Stenciling transforms should consume it instead of
  reimplementing case-specific shape tests.
- CFG/mid-end: SSA, MemorySSA-style load/store reasoning, alias analysis, LICM,
  DSE/DLE, SCCP/range folds, SCEV, inlining, and select conversion.
- Loop mid-end: canonical loop preheaders, guarded do-while rotation,
  LCSSA-compatible exit-phi repair, and LICM load motion only from blocks
  proven to execute after the guard.  Default RISC-V O1 rotation currently
  skips store-containing loops; store motion still requires a stronger
  guaranteed-execution proof than the loop guard alone provides.
- Backend: target lowering, instruction scheduling, hotness-aware register
  allocation, rematerialization-friendly decisions, and peepholes.

See `docs/Compliance.md` for pass families that must remain disabled by
default.

## Backends

### RISC-V

- Target ISA: rv64gc
- Dedicated lowering, combine, DCE, register allocation, assembly dump
- Assembly generation designed for large-address execution environments used by contest toolchains
- Extra low-risk peephole cleanup in regalloc phase:
  - redundant `mv/li/addi` to zero register elimination
  - adjacent address-add plus load fold for compact addressing

### ARM64

- Target ISA: ARMv8-A AArch64
- Dedicated lowering, combine, ARM-specific cleanup, post-increment legalization, register allocation, assembly dump
- Extra low-risk peephole cleanup in regalloc phase:
  - redundant `mov` and writes to `xzr` elimination
  - adjacent `add` + load/store address-offset folding with conservative guards

## Validation

- `scripts/run_smoke.sh` compiles smoke cases for both targets
- `scripts/regression.sh` and `scripts/compare.sh` validate `O0/O1/O2` on custom suites
- `--compare` semantic guard is executed on stabilized pre-backend checkpoint (`inst-schedule`)
- `scripts/compare.sh` defaults to functional-style gate (skips `perf/*` unless `COMPARE_INCLUDE_PERF=1`)
- `scripts/eval-o1-matrix.sh` keeps O1-only tuning workflow
- `scripts/eval-profile-matrix.sh` evaluates O1/O2 matrices with fixed ranking metrics
- `scripts/eval-official-adapter.sh` connects official functional/perf dirs when provided
- `scripts/suite-sync.sh` and `scripts/suite-index.sh` manage external public suite mirrors and index metadata
- `scripts/gen-reference-out.sh compiler-dev` generates hard-gate references for compiler-dev cases
- `scripts/eval-runtime.sh` executes runtime checks in Docker-first dual-target toolchains
- `scripts/eval-vs-biframe.sh` reports `pass_rate`, ratio metrics and staged thresholds against local biframe
- `--verify-ir` runs IR verifier after SSA phase
- `--dump-pass-timing` provides per-pass timing for optimization tuning
## LLVM/MLIR-Style Infrastructure

The compiler now carries a shadow MLIR-style infrastructure layer documented in
`docs/MLIRRoadmap.md`: restricted YAML op descriptors, generated descriptor
tables, opt-in greedy canonicalization patterns, typed analysis-cache keys,
MemRef alias side tables, and a target cost model. These pieces are deliberately
introduced without changing the default RISC-V O1 legality boundary.
