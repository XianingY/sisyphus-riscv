# LLVM/MLIR-Style Refactor Roadmap

This project now has a conservative first layer of MLIR-like infrastructure.
It is intentionally shadow/opt-in for the default RISC-V O1 path.

## Implemented Skeleton

- `src/ir/op_schema.yml` defines a restricted YAML op schema for the first
  `arith`, `scf`, and `memref` descriptors.
- `scripts/gen-op-descriptors.py` generates the committed
  `src/ir/GeneratedOpDescriptors.inc` file without PyYAML or other external
  dependencies.
- `sys::ir::OpDescriptorTable` verifies and dumps generated op metadata through
  `--dump-op-descriptors`.
- `PatternRewriter`, `PatternBenefit`, and `RewriteDriver` provide a greedy
  local rewrite framework. `PatternCanonicalize` is opt-in via
  `--pass-pipeline=pattern-canonicalize` or `SISY_ENABLE_PATTERN_CANONICALIZE=1`.
- `AnalysisManager` now tracks typed cache keys for DataLayout, AffineNest
  summary, and MemRefAlias in addition to dominators, loops, MemorySSA, alias,
  and block frequency.
- `MemRefType` and `MemRefAliasAnalysis` provide a base+offset+layout alias
  side table. It is currently a queryable analysis layer, not a default
  behavior change.
- `TargetCostModel` centralizes target latency/pressure facts and feeds the
  existing local scheduler while preserving current RISC-V scalar defaults.
- `--pass-pipeline=<passes>` adds a development-only textual pipeline prefix.
  Unknown pass names fail fast.

## Default Policy

The official path remains:

```text
compiler testcase.sy -S -o testcase.s -O1 --target=riscv
```

The new infrastructure does not enable RVV, semantic helper replacement,
fixed-output behavior, input/output precomputation, checksum triggers, or
source-name triggers.

## Next Migration Steps

1. Move more `RegularFold` rules into declarative canonicalization patterns.
2. Replace per-pass affine shape matching with cached `AffineNestAnalysis`
   summaries.
3. Feed `MemRefAliasAnalysis` into LICM, DSE, DLE, and scalar replacement behind
   dedicated compare gates.
4. Add a vector dialect layer for explicit RVV/NEON opt-in lowering.
5. Gradually table-drive backend peepholes before attempting full instruction
   selection generation.
