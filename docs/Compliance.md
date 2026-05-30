# Compliance Policy

This document is the source of truth for optimization legality in this
repository. It describes what is acceptable in the default compiler profile and
which existing experimental passes must stay opt-in.

Last reviewed: 2026-05-30.

## Core Rule

Every default optimization must be a semantics-preserving compiler
transformation over the program IR. It must not depend on benchmark identity,
source filename, output hash, public input data, final answer shape, or a magic
combination of constants that only describes one contest case.

Public performance sources may be used to choose engineering priorities and
test coverage. They must not become compiler triggers. A targeted optimization
is acceptable only when it fires from source-visible IR facts, such as an
affine loop, a complete initialization domain, a readonly table, or a proven
overwrite/reduction recurrence.

Acceptable triggers are program properties such as:

- dominance, SSA use-def facts, constant values from the source program, and
  value ranges proven by analysis;
- alias/noalias facts, readonly/pure call properties, and MemorySSA-style
  reaching definitions;
- affine loop domains, dependence directions, and Presburger feasibility
  results;
- target-independent algebraic identities and target-specific instruction
  peepholes that preserve ISA semantics for all inputs.

Rejected triggers include:

- test names such as `fft`, `huffman`, `conv2d`, `many_mat_cal`, or local
  source paths;
- fixed dimensions used as fingerprints, such as a transform that only exists
  because a public case uses one matrix size;
- interpreting a program at compile time to materialize hidden runtime answers;
- replacing a whole algorithm with a helper unless the helper is a general
  runtime primitive and the replacement is proven valid from IR semantics;
- any dependence on output checksums, timing loops, grader behavior, or known
  public input values.

## Default-Allowed Optimization Families

These are the preferred places to improve performance:

- SSA cleanup: `Mem2Reg`, copy propagation, `GVN`, `DCE`, `SCCP`,
  `RangeAwareFold`, `EqClass`, branch simplification, and ordinary CFG cleanup.
- Memory optimization: alias-aware `DSE`/`DLE`, load forwarding, readonly/pure
  call handling, scalar replacement, and conservative MemorySSA side tables.
- Loop optimization: loop canonicalization, LICM, SCEV/LSR, generic unrolling,
  fusion, interchange, tiling, unroll-and-jam, reduction privatization, and
  vectorization when legality follows from affine/dependence analysis.  For the
  official RISC-V O1 profile this means scalar `rv64gc` code by default; RVV,
  including strided vector loads/stores, remains explicit opt-in while the
  contest link command is `-march=rv64gc`.
- Recursion optimization: tail-call elimination and runtime memoization of
  proven pure recursive functions. Runtime memoization is allowed because all
  cache entries are computed at runtime for the actual input.
- Backend optimization: register allocation hotness/spill heuristics,
  rematerialization, instruction scheduling, target peepholes, address-mode
  folding, and legal strength reduction.
- Proven source-constant synthesis: replacing loads from source-program
  readonly integer constant tables, or tables built by a complete
  input-independent initialization loop, with a cheaper expression is allowed
  when the table domain is fully proven, the expression is derived from source
  constants/IVs, every reachable use is known, later mutation is rejected, and
  each replaced load has an IR-derived affine index. This is a data-layout
  optimization over constants already present in the program, not output
  synthesis.
- Proven overwrite repeat collapse: countdown loops may be reduced to their
  final iteration only when the loop-carried counter is the sole body-visible
  loop phi, externally visible calls are absent, direct stores do not address
  by the counter, and stored values do not depend on loop-local loads or call
  results. Memory accumulation and partial update loops must remain unchanged.

## Proven Source-Constant Synthesis

`SynthConstArray` is default-allowed for the RISC-V O1 profile after hardening.
Its legality boundary is intentionally narrow:

- only integer global arrays within `SISY_SYNTH_CONST_ARRAY_MAX` are considered;
- every reachable use of the global address must be a load, a store belonging
  to one proven initialization region, or a proven address calculation feeding
  those operations; calls, returns, address escape, and unknown uses reject the
  array;
- source initialization may be a full `0..N-1` affine loop or a fully unrolled
  store sequence that dominates all loads/calls reaching load functions;
- any store outside the proven initialization region rejects the array;
- the load address must be `base + 4 * affine_index`, with `affine_index`
  limited to `i`, `i + c`, or `i * scale + c` as derived from the existing IR;
- the candidate expression must come from the fixed arithmetic/bitwise DSL
  (`x`, `x +/- c`, `c - x`, small constant multiply/shift/modulo, `x & c`,
  `x | c`, `x ^ c`, and bounded low-cost combinations);
- solver inference may choose constants, but the final decision must validate
  every element in literal source tables; initialization-derived formulas must
  evaluate over the entire initialized domain before any load is replaced;
- the pass must not rewrite loop branch conditions, change control flow, use
  expected output, inspect public inputs, or key off source path/function/case
  names.

The default kill switch is `SISY_ENABLE_SYNTH_CONST_ARRAY=0`.

## Strict-Mode Or Experimental Only

The following passes may exist in the tree for experiments and A/B comparison,
but must remain disabled in default contest profiles unless they are redesigned
with a written, general legality proof:

| Pass or feature | Default | Compliance concern |
| --- | --- | --- |
| `FunctionEquivalence` | off | Uses compile-time sampling/interpreter checks to replace whole pure functions with known operations. |
| `StructuralBitwise` | off | Recognizes software-emulated bitwise algorithms and replaces them directly. |
| `StructuralModMul` | off | Recognizes recursive modular multiplication and replaces it directly. |
| `RowScratchMatmul` | off | Replaces recognized matrix multiplication with a helper implementation. |
| `Cached` / compile-time precompute | off | Can materialize compile-time results into the binary. |
| `AdvancedConv2DTransform` / Winograd or im2col dispatcher | off | Safe only after a general affine/padding legality proof; otherwise too close to algorithm-specific lowering. |
| HIR stencil interior dispatcher | off unless explicitly enabled | Can be valid, but only as a general guarded-loop transform, not a conv2d fingerprint. |

The following constant-table variants remain prohibited by default:

- cached or recursive compile-time precomputation;
- function-equivalence or sample-based whole-function replacement;
- structural bitwise or modular-multiplication recognizers;
- row-scratch matrix helper replacement;
- fixed `printf` assembly output paths or checksum/output replacement.

Current environment switches for these features are documented in
`docs/Commands.md`. Prefer disabling/enabling by environment variable for
diagnosis rather than editing source defaults during a performance experiment.

## Current Default Direction

The active RISC-V path should prefer:

- affine legality improvements in `HIRAffine`/`HIRPolyhedral`;
- generic 2D/3D tiling and dependence-preserving interchange;
- dependence-proven reduction microtiling with cache/register pressure gates;
- spill-aware register allocation and pre-RA scheduling;
- range-driven `/2`, `%2`, and power-of-two folds;
- branch-to-select conversion when CFG and SSA conditions are proven;
- source-constant table synthesis when every array element and every load index
  is proven under the boundary above;
- MemorySSA-backed LICM, DLE, and DSE.

For CRC, Huffman, FFT, convolution, matrix multiplication, transpose, and
scheduling workloads, the acceptable path is to expose and optimize the
underlying IR structure. Do not restore semantic recognizers just because they
produce a better score.

## Review Checklist

Before enabling or committing an optimization:

1. Write the legality argument in the commit message or a nearby design note.
2. Confirm the pass does not mention benchmark names, source paths, expected
   outputs, or public-case-only constants.
3. Confirm every memory transform has an alias/side-effect proof or cleanly
   bails out.
4. Confirm signed overflow, division, modulo, shifts, float semantics, and
   runtime calls are preserved under this compiler's IR rules.
5. Add or keep an environment kill switch for new risky transforms.
6. Run affected cases plus unrelated regression probes.
7. Keep generated files under ignored paths such as `build/`, `build-linux/`,
   `tests/.out/`, and `output/`.

## Documentation Hygiene

Historical planning notes under `docs/superpowers/` and `docs/goal.md` are
useful design references, but they are not authoritative. When they conflict
with this file or with `src/main/PipelineProfiles.cpp`, the code and this
compliance policy win.
