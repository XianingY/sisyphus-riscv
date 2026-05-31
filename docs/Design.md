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
- `--dump-hir`, `--dump-cfg`, `--verify-hir`, `--verify-cfg`

## Pipeline

### Frontend Staging (default)

Default frontend path is dialect-based:

- `AST -> self-MLIR(sysy/scf/memref/arith/affine) -> dialect conversion -> rv_machine/arm_machine -> asm`
- Stage boundaries emit timing with `--dump-pass-timing` as `[stage-timing] ...`

The key architectural split is intentional:

- self-MLIR keeps structured loops, regions, block arguments, memrefs, and
  array accesses visible for affine and MemorySSA-style reasoning.
- Dialect conversion makes the target legality boundary explicit before the
  module reaches `rv_machine` or `arm_machine`.
- The native machine-dialect emitter owns the default RISC-V/ARM assembly path;
  legacy HIR/CFG notes are historical compatibility material, not the intended
  production optimization substrate.

### O0 (stable baseline)

- MoveAlloca
- EarlyConstFold / Pureness
- RaiseToFor / DCE / Lower
- FlattenCFG
- Mem2Reg
- RegularFold / DCE / SimplifyCFG / Select / InstSchedule
- Target lowering (ARM or RISC-V)

### O1 (performance pipeline)

- AST lowering to self-MLIR, global/memref cleanup, affine loop recovery, loop
  fusion/interchange when legality is proven, IPO summaries, and DRR
  canonicalization.
- Dialect conversion to machine operations with verifier checks before and
  after canonicalization.
- Native machine emission with default scalar RISC-V `rv64gc` or ARM64 code,
  plus conservative address-mode folding and assembly-level accounting.

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

- self-MLIR affine/scf: loop legality, fusion/interchange, guarded-loop
  simplification, and optional tiling after dependence proof.
- self-MLIR memref: SSA use-def cleanup, MemorySSA-style load/store reasoning,
  alias queries, LICM, DSE/DLE, SCCP/range folds, SCEV-style induction facts,
  inlining, and select conversion.
- machine dialect: target legality, scheduling, hotness-aware register
  allocation, rematerialization-friendly decisions, address-mode folding, and
  peepholes.

See `docs/Compliance.md` for pass families that must remain disabled by
default.

## Backends

### RISC-V

- Target ISA: rv64gc
- self-MLIR dialect conversion to `rv_machine`, native machine emission, and
  conservative scalar peepholes
- Assembly generation designed for large-address execution environments used by contest toolchains
- Extra low-risk peephole cleanup in regalloc phase:
  - redundant `mv/li/addi` to zero register elimination
  - adjacent address-add plus load fold for compact addressing

### ARM64

- Target ISA: ARMv8-A AArch64
- self-MLIR dialect conversion to `arm_machine`, native machine emission, and
  conservative scalar peepholes
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

The compiler now carries a production self-MLIR infrastructure layer documented
in `docs/MLIRRoadmap.md`: hash-consed types/attrs/locations, Operation/Region/
Block/BlockArgument IR, restricted YAML op descriptors, generated descriptor
tables, DRR canonicalization patterns, typed analysis-cache keys, MemRef alias
side tables, and a target cost model. These pieces preserve the default RISC-V
O1 legality boundary: no RVV, no fixed-output helpers, and no benchmark-name
triggers by default.
