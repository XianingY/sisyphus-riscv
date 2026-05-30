# LLVM/MLIR-Style Refactor Roadmap

This project now has a conservative first layer of MLIR-like infrastructure.
It is intentionally shadow/opt-in for the default RISC-V O1 path.

## Implemented Skeleton

- `src/ir/op_schema.yml` defines a restricted YAML op schema for the first
  `arith`, `scf`, and `memref` descriptors.
- `scripts/gen-op-descriptors.py` generates the committed
  `src/ir/GeneratedOpDescriptors.inc` and `src/ir/GeneratedOpClasses.inc`
  files without PyYAML or other external dependencies. The class file is an
  ODS-style adapter layer with typed operand getters and verify stubs.
- `sys::ir::OpDescriptorTable` verifies and dumps generated op metadata through
  `--dump-op-descriptors`.
- `sys::ir::IRContext` interns lightweight Type and Attribute handles, including
  parameterized memref/vector types and location attributes. This is a shadow
  bridge toward MLIR-style hash-consed type/attribute storage while legacy
  `Value::Type` remains the default execution path.
- Legacy `LocationAttr` is available so production ops can gradually adopt
  mandatory location tracking without forcing an all-at-once IR migration.
- `BasicBlock` now carries a block-argument side table. Phi ops remain the
  production SSA representation for now, but the new API lets CFG utilities,
  tests, and future conversions model MLIR-style block arguments incrementally.
- `PatternRewriter`, `PatternBenefit`, and `RewriteDriver` provide a greedy
  local rewrite framework. `PatternCanonicalize` is opt-in via
  `--pass-pipeline=pattern-canonicalize` or `SISY_ENABLE_PATTERN_CANONICALIZE=1`.
- `AnalysisManager` now tracks typed cache keys for DataLayout, AffineNest
  summary, and MemRefAlias in addition to dominators, loops, MemorySSA, alias,
  and block frequency.
- `ScopedPassRegistry` records MLIR-like pass scope, required analyses,
  preserved analyses, and parallelizability for the first module/function/loop
  and block passes. It is introspectable through `--dump-pass-scopes`.
- `DialectConversionDriver` provides a small ConversionTarget/TypeConverter
  scaffold. The current standard scalar target legalizes `arith`, `scf`, and
  `memref` descriptors and can be inspected with `--dump-dialect-conversion`.
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
2. Start replacing Phi users with block arguments at region boundaries where
   the predecessor set is structurally simple.
3. Replace per-pass affine shape matching with cached `AffineNestAnalysis`
   summaries.
4. Feed `MemRefAliasAnalysis` into LICM, DSE, DLE, and scalar replacement behind
   dedicated compare gates.
5. Add a vector dialect layer for explicit RVV/NEON opt-in lowering.
6. Gradually table-drive backend peepholes before attempting full instruction
   selection generation.
