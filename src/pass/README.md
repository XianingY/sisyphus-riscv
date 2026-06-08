# Pass Mapping

## Current State
- Canonical pass entry: `src/pass/PassRegistry.h`
- Current implementation lives in:
  - `src/mlir/ASTToMLIR.cpp`: production pipeline wiring
  - `src/mlir/SelfMLIROpt.cpp`: self-MLIR scalar, affine, polyhedral,
    memory, bitwise, and vector rewrites
  - `src/mlir/SelfMLIR.cpp`: optimization profile defaults and dialect
    conversion patterns
  - `src/mlir/SelfMLIRNative.cpp`: RISC-V/ARM native lowering, liveness, and
    register allocation

## Boundary Rule
- New optimization wiring should flow through `OptimizationConfig` and
  `runProductionGateFromAST`.
- Default-profile legality must be decided from IR facts, not source names,
  paths, fixed outputs, or benchmark fingerprints.

## Next Steps
- Keep pass ordering centralized in the self-MLIR production pipeline.
- Extend stats when adding new optimization families so compile/runtime
  attribution remains reproducible.
