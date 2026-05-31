# RISC-V Performance Notes

This directory holds performance analysis documents for the public
`test2026/performance_riscv` suite. The case names in these files are for
engineering triage and reporting only; the compiler must not use case names,
paths, public inputs, expected outputs, checksums, or fixed dimensions as
optimization triggers.

Default RISC-V O1 remains scalar `rv64gc`. The following ideas are valid in the
default profile only when they are proven from self-MLIR, affine, memref,
MemorySSA, or machine-IR facts:

- affine loop interchange, small tiling, scalar accumulator promotion, address
  recurrence, and dependence-proven stencil border peeling;
- readonly table hoisting and `SynthConstArray` over source constants or
  complete input-independent initialization loops;
- guarded lowering of complete 32-bit bitwise helper loops via
  `ProvenBitwiseHelper`;
- DRR canonicalization, range-proven `/2` and `%2` folds, local machine
  liveness, rematerialization, and conservative load-use scheduling.

These remain explicit experiments rather than default O1 behavior:

| Feature | Default | Reason |
| --- | --- | --- |
| RVV / SIMD vectorization | off | `docs/Requirements.md` requires the initial RISC-V path to link as `rv64gc`. |
| `StructuralBitwise` | off | Whole semantic recognizer; use `ProvenBitwiseHelper` only when IR shape proves fallback-safe lowering. |
| `StructuralModMul` | off | Algorithm-level recognizer without the required general proof boundary. |
| `FunctionEquivalence` | off | Sampling/equivalence replacement is not a default compiler optimization. |
| `Cached` / compile-time precompute | off | Must not materialize public-input or expected-output answers. |
| `RowScratchMatmul` / helper replacement | off | Matrix performance must come from affine/linalg legality and machine cost facts. |

Current documents:

- `PERFORMANCE_OPTIMIZATION_GUIDE.md`: broad optimization catalogue, rewritten
  as a compliant priority guide.
- `IMPLEMENTATION_ROADMAP.md`: implementation notes and testing checklist.
- `TEST_CASE_ANALYSIS.md`: workload taxonomy and shape-driven opportunities.
- `QUICK_REFERENCE.md`: short checklist for daily RISC-V performance work.
