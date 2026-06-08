# Compliance Policy

This document is the source of truth for optimization legality in this
repository. It describes what is acceptable in the default compiler profile and
which existing experimental passes must stay opt-in.

Last reviewed: 2026-06-06.

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

Branches exposing this pass must keep `SISY_ENABLE_SYNTH_CONST_ARRAY=0` as the
default-profile kill switch.

## Proven Prefix And Helper Reductions

The default RISC-V O1 profile may also use narrow proof-driven reductions:

- `PrefixCallReduction` may collapse a countdown loop to its final positive
  iteration only when the loop's observable value is overwritten each iteration,
  the zero-trip guard remains intact, and no external side effect or
  counter-dependent store is present.  This covers repeated prefix-call shapes
  without inspecting public inputs or expected outputs.
- `ProvenBitwiseHelper` may lower a fully classified self-MLIR helper that
  computes bitwise `and`/`xor` through a complete 32-iteration arithmetic loop.
  The pass fires from IR shape only, never from helper name.  If both operands
  are not statically non-negative, it emits a runtime guard and keeps the
  original helper as fallback; the opt-in `StructuralBitwise` whole recognizer
  remains disabled.
- `RotateHelperFold` may fold source-defined shift-scale helper functions only
  when every case is an explicit `n == k` branch returning `x * 2^k` or
  `x / 2^k`, all cases agree on one direction, the default return is the
  original `x`, and dynamic shifts lower to a bounded helper with the original
  in-range semantics. It must not infer rotation algorithms from function names,
  source paths, public cases, or output behavior.

The kill switches are `SISY_ENABLE_PREFIX_CALL_REDUCTION=0` and
`SISY_ENABLE_SELF_PROVEN_BITWISE=0`; use `SISY_ENABLE_SELF_ROT_HELPER=0` to
disable rotate-helper folding for bisection.

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
| self-MLIR semantic/native kernel emitters | off | `SISY_ENABLE_SELF_SEMANTIC_KERNELS` and `SISY_ENABLE_SELF_STRUCTURAL_KERNELS` emit whole helper/native kernels such as digest, matmul summary, conv2d interior, LUDCMP, Nussinov, TRSM, hash aggregate, modular multiply/power, and memcopy. |

The following constant-table variants remain prohibited by default:

- cached or recursive compile-time precomputation;
- function-equivalence or sample-based whole-function replacement;
- unguarded structural bitwise or modular-multiplication recognizers;
- row-scratch matrix helper replacement;
- self-MLIR whole-kernel replacements under `SISY_ENABLE_SELF_*KERNEL` switches;
- fixed `printf` assembly output paths or checksum/output replacement.

Current environment switches for these features are documented in
`docs/Commands.md`. Prefer disabling/enabling by environment variable for
diagnosis rather than editing source defaults during a performance experiment.

## Current Default Direction

The active RISC-V path is now self-MLIR first:

```text
AST -> self-MLIR(sysy/scf/memref/arith/affine)
    -> dialect conversion
    -> rv_machine/arm_machine
    -> native asm
```

The legacy HIR/CFG vocabulary may still appear in old notes and removed
experiments, but new default performance work should prefer:

- affine legality improvements in self-MLIR `affine`/`scf` regions;
- generic 2D/3D tiling, dependence-preserving interchange, border peeling, and
  reduction scalar promotion driven by affine summaries;
- self-MLIR `memref` and MemorySSA-backed LICM, DLE, DSE, and load forwarding;
- DRR/canonicalization rules for local arithmetic identities, including
  range-safe `/1`, `%1`, and branch-to-select cleanup;
- source-constant table synthesis when every array element and every load index
  is proven under the boundary above;
- prefix/final-value reduction and guarded bitwise-helper lowering under the
  proof boundaries above;
- machine-dialect address-mode folding, rematerialization, scheduling, and
  spill-aware register allocation.

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
