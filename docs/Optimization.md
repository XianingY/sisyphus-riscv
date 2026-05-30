# Optimization Workflow

This document records how Sisyphus optimizations are designed, enabled, and
validated. It complements `docs/Compliance.md`; if there is any conflict, the
compliance policy wins.

Last reviewed: 2026-05-30.

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
- Hardened `LoopRotate` for default scalar RISC-V O1.  It rewrites canonical
  positive-step while loops into a guarded do-while form only after
  LoopSimplify/LCSSA-style canonicalization, keeps a single-edge loop preheader,
  repairs exit phis for zero-trip semantics, and leaves calls in place.  The
  current default legality gate intentionally excludes loops containing stores
  until the downstream rotated-loop consumers have a per-store
  guaranteed-execution proof.
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
- proven source-constant table synthesis for small readonly lookup arrays when
  each element matches a fixed DSL expression and every replaced load has an
  affine IR index.

Do not use semantic function equivalence or structural bitwise recognizers in
the default profile.

### Proven Source-Constant Synthesis

The default RISC-V O1 profile may run `SynthConstArray` on source-program
integer lookup tables. This follows the aggressive-but-compliant boundary used
by strong reference compilers: constants already present in the source may be
represented in a cheaper form, provided the compiler proves the rewrite from
IR facts instead of from benchmark identity.

The pass must prove readonly/non-escaping global-address use, keep array size
small, derive `base + 4 * affine_index` from existing IR/SCEV address facts,
learn only constants for a fixed arithmetic/bitwise DSL, and validate the
selected expression against every table element. It must not inspect source
paths, case names, expected output, public input files, or checksums, and it
must not rewrite loop exit conditions.

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

### Affine Reduction Microtiling

The default RISC-V HIR path can panelize proven integer additive reductions
over affine two-dimensional outputs. It preserves a row scratch buffer when
in-place reads require write isolation, holds a small contiguous column tile
in scalar registers while processing a `k` panel, and writes partial sums only
at panel boundaries. The profitability gate rejects conditional updates that
increase spill pressure on the current scalar backend.

Conditional integer reductions use a narrower path by default. Instead of
forcing a long-lived register panel, the row-scratch loop may be unrolled by a
small lane factor when the update is out-of-place, has no array stores, and all
array reads are affine globals. This keeps the original `k` order and contiguous
row-buffer accesses while reducing loop-control overhead without adding spills.
The heavier conditional panel path remains opt-in for A/B testing via
`SISY_HIR_ENABLE_CONDITIONAL_REDUCTION_MICROTILE=1`.

This transform is driven only by canonical loop shape, affine accesses,
dependence legality, cache budget, and register-pressure estimates. It does
not identify matrix test cases, dimensions, or algorithm names. Use
`SISY_HIR_ENABLE_REDUCTION_MICROTILE=0` for bisection and
`SISY_HIR_ENABLE_CONDITIONAL_ROW_JAM=0` to disable the conditional row-scratch
unroll. `SISY_HIR_MICRO_NR` / `SISY_HIR_MICRO_KC` and
`SISY_HIR_COND_MICRO_NR` / `SISY_HIR_COND_MICRO_KC` are reserved for cost-model
experiments.

## 5. Environment Switches

Common bisection and experiment flags are listed in `docs/Commands.md`. The
important defaults as of this review:

- `SISY_ENABLE_FUNCTION_EQUIVALENCE=false`
- `SISY_ENABLE_STRUCTURAL_BITWISE=false`
- `SISY_ENABLE_STRUCTURAL_MODMUL=false`
- `SISY_ENABLE_ROW_SCRATCH_MATMUL=false`
- `SISY_ENABLE_CACHED_PRECOMPUTE=false`
- `SISY_ENABLE_SYNTH_CONST_ARRAY=true` for default RISC-V O1, with
  `SISY_ENABLE_SYNTH_CONST_ARRAY=0` as the bisection kill switch
- `SISY_ENABLE_LOOP_ROTATE=true` for default scalar RISC-V O1, with
  `SISY_ENABLE_LOOP_ROTATE=0` as the bisection kill switch.  The public CLI
  controls `--disable-loop-rotate` and `--enable-loop-rotate` remain available.
- `SISY_ENABLE_UNROLL_INTERNAL_ROTATE=false`; the old unroll-local rotation
  helpers remain available for experiments but are not part of the default
  correctness envelope.
- `SISY_ENABLE_LOOP_ROTATE_STORES=false`; store-containing loop rotation is
  available only for explicit experiments such as RVV vectorization tests.  The
  default scalar RISC-V O1 route keeps those loops unrotated until downstream
  passes have stronger store-path proofs.
- `SISY_ENABLE_RVV_STRIDED=false`; RVV strided `vlse32.v`/`vsse32.v`
  lowering is experimental and requires both RVV enablement and this explicit
  strided-vector switch.  The default RISC-V O1 profile must remain `rv64gc`
  scalar code.
- `SISY_SYNTH_CONST_ARRAY_MAX=4096` caps the default source-constant table
  synthesis budget, including input-independent initialization loops
- `SISY_ENABLE_FINAL_ITER_COLLAPSE=true` enables the narrow IR-proven
  countdown-loop final-iteration collapse used for overwrite-style repeats;
  loops whose stores depend on loop-local loads or call results are rejected
- `SISY_ENABLE_ADVANCED_CONV2D=false`
- `SISY_HIR_ENABLE_TILING=true`
- `SISY_HIR_ENABLE_STENCIL_INTERIOR=false`
- `SISY_HIR_ENABLE_FUSION=true`
- `SISY_HIR_ENABLE_INTERCHANGE=true`
- `SISY_HIR_ENABLE_UNROLL_JAM=true`
- `SISY_HIR_ENABLE_REDUCTION_PRIVATIZE=true`
- `SISY_HIR_ENABLE_REDUCTION_MICROTILE=true`
- `SISY_HIR_ENABLE_INVARIANT_GUARD_HOIST=true`
- `SISY_RV_ENABLE_SUPERBLOCK=true`
- `SISY_RV_SUPERBLOCK_PRESSURE_BUDGET=20`
- `SISY_RV_ENABLE_SCHEDULE=true`
- `SISY_RV_SCHEDULE_HEIGHT_WEIGHT=5`
- `SISY_RV_ENABLE_REMATERIALIZATION=true`
- `SISY_RV_ENABLE_LIVE_RANGE_SPLIT=true`

Always verify current defaults in `src/main/PipelineProfiles.cpp`,
`src/hir/HIRPolyhedral.cpp`, and the pass implementation before relying on a
documented switch.

## 6. Retired Or Rejected Ideas

The following ideas have either caused correctness risk or sit outside the
default compliance boundary:

- compile-time recursive precomputation;
- structural bitwise/modmul recognition;
- row-scratch matrix helper replacement;
- synthesized constant arrays that lack full-table validation, readonly proof,
  or affine-index proof;
- semantic matrix summaries, transpose summaries, or checksum summaries;
- conv-specific Winograd/im2col dispatch without a complete affine legality
  model;
- timed-loop or output-shape replacement.

If an idea from this list is revisited, it must be reframed as a general
compiler transformation with a written proof and a conservative bailout path.

## 7. Recent Additions (Off-by-Default Infrastructure)

All entries below are passive infrastructure: default invocation produces
byte-identical output. Each has an env kill switch so it can be enabled
incrementally.

- **HIR affine nest summaries** (`src/hir/HIRAffine.cpp`):
  `collectAffineNest` returns canonical loops, per-loop domain (iv, upper,
  step), innermost accesses with surrounding IVs, guards, and a coarse
  side-effect summary plus `imperfect`/`hasSymbolicAccesses` flags.
  Current HIR pipeline passes (`HIRPolyInterchange`, `HIRPolyhedral`)
  still use per-pass `matchCanonicalWhile` + `collectArrayAccesses` for
  granular stats; migrating them to share one `AffineNest` per nest is a
  follow-up refactor (correctness equivalence needs per-pass stat
  reconciliation).
- **Function summary** (`src/opt/FunctionSummary.cpp`): attaches
  `FunctionSummaryAttr { pure, readonly, norecurse, argReadMask,
  argWriteMask }`. Bits set only when proven. Kill switch:
  `SISY_ENABLE_FN_SUMMARY=0`.
- **Thin summary dump** (`src/opt/ThinSummary.cpp`): emits per-function
  summary to `$SISY_THIN_SUMMARY_DUMP`. No-op when unset. Seed for future
  thin-link IPCP/IPDCE.
- **Block frequency** (`src/opt/BlockFrequency.cpp`): static loop-depth
  frequency in a singleton side table; `BlockFrequency::freqOf(bb)` /
  `isCold(bb)`. Profile import via `$SISY_PROFILE`. Kill switch:
  `SISY_ENABLE_BFI=0`.
- **RV superblock schedule v1** (`src/rv/SuperblockSchedule.cpp`): hoists
  pure RV loads across single-pred / single-succ hot edges, gated by
  shape, dependency, `AliasAttr::neverAlias`, and pressure budget. Off by
  default; enable via `SISY_RV_ENABLE_SUPERBLOCK=1`.  Recommended tuning
  sweep before flipping the default: hold `SISY_RV_SUPERBLOCK_PRESSURE_BUDGET`
  at one of {12, 16, 20, 24, 28} and compare cycle counts against the
  closed-default baseline on the public regression suite; pick the
  smallest budget at which cycles do not regress on any case.
- **SMT proof helper** (`src/opt/SmtProver.cpp`):
  `smt_prover::tryProveEqualI32(a, b)` returns `Equal` / `NotEqual` /
  `Unknown`. Consumers must treat `Unknown` as "do not fold".

## 7a. Imperfect Reduction Nest Status

The scalar-promotion + I-K-J interchange transform that the project plan
once flagged as "Plan A" is already present:

- `PolyhedralOptimizer::tryReductionInterchange`
  (`src/hir/HIRPolyReduction.cpp`) detects the canonical
  `init; for k { acc += ... }; store(acc)` shape and performs the
  semantic interchange behind `SISY_HIR_ENABLE_REDUCTION_PRIVATIZE`.
- `tryReductionRowPrivatize` introduces a per-row scratch buffer when the
  cleaner in-place interchange is not legal.

If a particular matmul-like nest still appears unmodified, the gap is in
legality checks (`privatizedReductionLegal`,
`strictReductionInterchangeLegal`), not in pattern detection.  Adding a
parallel pre-pass would duplicate analysis; the responsible follow-up is
to relax the existing legality predicates with proof obligations, not to
introduce a new transformer.

## 8. Compliance Boundary For New Infrastructure

Every component in section 7 is gated such that default invocation
produces byte-identical output to the prior commit. Defaults may flip
only in a separate commit demonstrating correctness on the regression
suite plus a measurable performance reason. Off-by-default switches must
remain queryable as a kill switch.
