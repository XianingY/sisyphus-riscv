# Optimization Workflow

This document records how Sisyphus optimizations are designed, enabled, and
validated. It complements `docs/Compliance.md`; if there is any conflict, the
compliance policy wins.

Last reviewed: 2026-05-27.

## 1. Default Profile Philosophy

The current default profile is a general compiler optimization profile, not a
benchmark-recognition profile. It enables broad IR and backend improvements and
keeps high-risk semantic recognizers off by default.

Default-preferred work:

- improve canonical IR shape so existing passes can reason about it;
- prove memory safety with alias analysis, readonly/pure facts, and
  MemorySSA-style reaching definitions;
- prove loop legality with affine access analysis and dependence directions;
- reduce arithmetic with normal algebraic and range-based identities;
- improve backend register allocation, scheduling, and peepholes.

Do not optimize by recognizing public case names, expected checksums, hidden
input behavior, or algorithm-specific helper substitutions.

## 2. Current Pass Inventory

### Stable General Optimizations

These are appropriate for default use when verifier and regression checks pass:

- `Mem2Reg`, `RegularFold`, `SCCP`, `Range`, `RangeAwareFold`, `EqClass`,
  `GVN`, `DCE`, `AggressiveDCE`, `SimplifyCFG`, and `Select`.
- Alias-aware `DSE`/`DLE`, load forwarding, readonly call handling, and LICM.
- `SCEV` and ordinary loop strength reduction.
- HIR affine transforms: fusion, interchange, reduction privatization,
  invariant guard hoisting, partial unroll, and unroll-and-jam when dependence
  analysis proves legality.
- Runtime memoization for proven pure recursive functions. It adds runtime
  cache tables and does not precompute answers at compile time.
- RISC-V and ARM backend lowering, scheduling, hotness-aware register
  allocation, rematerialization-friendly heuristics, and target peepholes.

### Default-Off Or Strict-Mode Experiments

These files may be useful for experiments, but they are not the default
compliance path:

- `FunctionEquivalence.cpp`
- `StructuralBitwise.cpp`
- `StructuralModMul.cpp`
- `RowScratchMatmul.cpp`
- `Cached.cpp`
- `SynthConstArray.cpp`
- `AdvancedConv2DTransform.cpp`
- HIR stencil/interior dispatcher when used as a conv-specific shortcut

Use their environment switches only for explicit comparison runs. Do not submit
a profile that depends on them unless the implementation has been redesigned as
a general, local, legality-proven transform.

## 3. Standard Optimization Loop

1. Establish a baseline.
   - Build natively with `scripts/build.sh`.
   - Build `build-linux/compiler` for Docker/QEMU runtime checks on macOS.
   - Run affected local cases and record commit id, command line, and stats.

2. Identify the missed compiler fact.
   - Was a value range not propagated?
   - Was a load not forwarded because alias analysis was too weak?
   - Did affine extraction reject a legal loop nest?
   - Did register allocation spill a hot loop value?
   - Did scheduling lengthen live ranges or fail to hide load latency?

3. Write the legality argument.
   - State the IR shape transformed.
   - State required dominance, CFG, alias, side-effect, range, overflow, and
     type conditions.
   - State the conservative bailout path.

4. Implement the smallest general change.
   - Prefer strengthening existing analyses over adding one-off passes.
   - If a numeric threshold is needed, tie it to cache/register/code-size cost,
     not to a benchmark dimension.
   - Keep risky changes behind an environment switch until stable.

5. Verify.
   - `scripts/build.sh`
   - targeted compile/runtime probe
   - one unrelated regression subset
   - `git diff --check`
   - inspect pass stats for both transformed and rejected cases

6. Submit cleanly.
   - Commit only related files.
   - Push the active branch to GitLab and GitHub when appropriate.
   - Keep online result notes separate from source changes unless updating
     documentation.

## 4. High-Value Safe Directions

### Matrix, Many-Matrix, And Transpose

Best legal path:

- stronger HIR affine nest extraction;
- 2D/3D dependence direction solving;
- cache-aware tiling and register-pressure-aware unroll-and-jam;
- vectorization handoff for contiguous inner loops;
- backend address-mode folding and hot spill avoidance.

Avoid replacing matrix loops with hand-written helpers. The transform must keep
the original program computation and only reorder, tile, scalarize, vectorize,
or schedule it under proven dependence constraints.

### Conv2D

Best legal path:

- guarded/imperfect affine extraction;
- border peeling so interior regions can be optimized without per-pixel guards;
- small constant kernel-loop unrolling based on loop trip count and body cost;
- generic tiling and vectorization of stride-1 inner loops.

Winograd/im2col-style lowering is high risk unless implemented as a general
affine lowering with complete padding, boundary, alias, and shape proofs. It
must not be a conv2d-case dispatcher.

### CRC And Huffman

Best legal path:

- range-proved `/ 2 -> >> 1` and `% 2 -> & 1`;
- parity if-conversion and select generation;
- spill-aware partial unroll, not indiscriminate full unroll;
- readonly/load-forwarding/LICM for tables and repeated global loads.

Do not use semantic function equivalence or structural bitwise recognizers in
the default profile.

### FFT

Best legal path:

- tail-call elimination and generic single-recursion-to-loop conversion;
- range and algebraic folds that expose `/2`, `%2`, and conditional modular
  add/sub reduction;
- ordinary inlining and LICM where pure/noalias facts prove safety.

Do not replace recursive modular multiplication with a direct hardware modmul
recognizer in the default profile.

### Scheduling And Backend Hot Loops

Best legal path:

- loop-depth-weighted spill cost;
- hot block live-range splitting or local copy isolation;
- rematerialization for immediates and addresses;
- pre-RA scheduling that respects memory dependencies and does not inflate live
  ranges unnecessarily;
- post-RA peepholes for redundant moves and address calculation chains.

## 5. Environment Switches

Common bisection and experiment flags are listed in `docs/Commands.md`. The
important defaults as of this review:

- `SISY_ENABLE_FUNCTION_EQUIVALENCE=false`
- `SISY_ENABLE_STRUCTURAL_BITWISE=false`
- `SISY_ENABLE_STRUCTURAL_MODMUL=false`
- `SISY_ENABLE_ROW_SCRATCH_MATMUL=false`
- `SISY_ENABLE_CACHED_PRECOMPUTE=false`
- `SISY_ENABLE_SYNTH_CONST_ARRAY=false`
- `SISY_ENABLE_ADVANCED_CONV2D=false`
- `SISY_HIR_ENABLE_TILING=false`
- `SISY_HIR_ENABLE_STENCIL_INTERIOR=false`
- `SISY_HIR_ENABLE_FUSION=true`
- `SISY_HIR_ENABLE_INTERCHANGE=true`
- `SISY_HIR_ENABLE_UNROLL_JAM=true`
- `SISY_HIR_ENABLE_REDUCTION_PRIVATIZE=true`
- `SISY_HIR_ENABLE_INVARIANT_GUARD_HOIST=true`

Always verify current defaults in `src/main/PipelineProfiles.cpp`,
`src/hir/HIRPolyhedral.cpp`, and the pass implementation before relying on a
documented switch.

## 6. Retired Or Rejected Ideas

The following ideas have either caused correctness risk or sit outside the
default compliance boundary:

- compile-time recursive precomputation;
- structural bitwise/modmul recognition;
- row-scratch matrix helper replacement;
- SMT-synthesized constant arrays;
- semantic matrix summaries, transpose summaries, or checksum summaries;
- conv-specific Winograd/im2col dispatch without a complete affine legality
  model;
- timed-loop or output-shape replacement.

If an idea from this list is revisited, it must be reframed as a general
compiler transformation with a written proof and a conservative bailout path.
