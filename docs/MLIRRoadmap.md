# LLVM/MLIR-Style Refactor Roadmap

This project now has a self-hosted MLIR core in addition to the historical
legacy compiler path.  The new core is not an upstream LLVM dependency; it is a
project-local implementation of the MLIR object model and conversion contracts.

## Implemented Skeleton

- `src/ir/op_schema.yml` defines a restricted YAML op schema for the first
  `arith`, `scf`, and `memref` descriptors.
- `scripts/gen-op-descriptors.py` generates the committed
  `src/ir/GeneratedOpDescriptors.inc` and `src/ir/GeneratedOpClasses.inc`
  files without PyYAML or other external dependencies. The class file is an
  ODS-style adapter layer with typed operand getters and verify stubs.
- `sys::ir::OpDescriptorTable` verifies and dumps generated op metadata through
  `--dump-op-descriptors`.
- `sys::ir::Operation` is now the bridge core between legacy `Op` and the
  MLIR-style side tables. Legacy ops can expose an `Operation*`, ODS wrappers
  wrap `Operation*`, and `--dump-operation-ir` prints a minimal MLIR-like view.
- `sys::ir::IRContext` interns lightweight Type and Attribute handles, including
  parameterized memref/vector types and location attributes. This is a shadow
  bridge toward MLIR-style hash-consed type/attribute storage while legacy
  `Value::Type` remains the default execution path.
- Legacy `LocationAttr` is available so production ops can gradually adopt
  mandatory location tracking without forcing an all-at-once IR migration.
- `BasicBlock` now carries a block-argument side table. Phi ops remain the
  production SSA representation for now, but the new API lets CFG utilities,
  tests, and future conversions model MLIR-style block arguments incrementally.
- `PhiToBlockArgumentBridge` can materialize simple Phi nodes into block
  argument side-table entries and lower them back, giving the project a safe
  round-trip test path before replacing production Phi users.
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
  `--run-dialect-conversion=legacy` performs a non-mutating legacy dry run, and
  `--run-dialect-conversion=rollback-test` exercises rollback accounting.
- `ScopedPassRegistry` is verified by the real `PassManager`; with pass timing
  enabled, pass scope declarations are surfaced alongside normal timing output.
- `MemRefType` and `MemRefAliasAnalysis` provide a base+offset+layout alias
  side table. It is currently a queryable analysis layer, not a default
  behavior change.
- `TargetCostModel` centralizes target latency/pressure facts and feeds the
  existing local scheduler while preserving current RISC-V scalar defaults.
- `--pass-pipeline=<passes>` adds a development-only textual pipeline prefix.
  Unknown pass names fail fast.

## Self-MLIR Core

- `src/mlir/SelfMLIR.*` defines the production-candidate MLIR object model:
  `Context`, hash-consed `Type`/`Attribute`/`Location`, `Operation`, `Region`,
  `Block`, `BlockArgument`, `Value`, `Builder`, verifier, and printer.
- The self-MLIR verifier rejects `legacy.*` and Phi-shaped operations.  The
  sample module uses only region ops and block arguments.
- `src/mlir/canonicalize.drr` is the first declarative rewrite rule file.
  `--run-self-mlir-core-tests` parses the DRR seed and runs a greedy rewrite
  driver over a pure self-MLIR module.
- `--run-self-mlir-conversion-tests` exercises target legalization and rollback
  for both `rv_machine` and `arm_machine` without touching legacy IR.
- `scripts/run_mlir_core_tests.sh` and
  `scripts/run_dialect_conversion_tests.sh` are the new core gates that must
  pass before migrating default compilation stages onto the self-MLIR path.

## Default Policy

The official path remains:

```text
compiler testcase.sy -S -o testcase.s -O1 --target=riscv
```

The new infrastructure does not enable RVV, semantic helper replacement,
fixed-output behavior, input/output precomputation, checksum triggers, or
source-name triggers.

## Next Migration Steps

1. Replace AST/HIR construction with direct `sysy + scf + memref + arith`
   self-MLIR emission.
2. Port Mem2Reg, DCE, SCCP, LICM, SCEV, MemorySSA, affine transforms, and
   vectorization to `OperationPass<OpT>` over self-MLIR operations.
3. Replace RV/ARM lowering with `DialectConversion` into
   `rv_machine`/`arm_machine`.
4. Move register allocation, live-range splitting, scheduling, and asm printing
   onto machine dialect operations.
5. Delete legacy bridge files once default RISC-V and ARM gates are green on
   the self-MLIR pipeline.
