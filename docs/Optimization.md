# Optimization Workflow

This document defines the optimization workflow for this compiler. The current
`pure-rv` branch defaults to an all-optimization profile, while keeping
environment kill switches for risky passes so regressions can be bisected
quickly. Even in all-optimization mode, transformations must not depend on test
identity, expected answers, source filenames, or benchmark-specific output
behavior.

## 1. Compliance Rules

All optimizations must be justified as ordinary compiler transformations over
the program IR. Do not add transformations that depend on public or hidden test
identity, benchmark dimensions, magic constant combinations, expected output
shape, or runtime-library timing patterns.

Allowed examples:

- Constant propagation, copy propagation, GVN, DCE, DSE, DLE.
- LICM when alias and side-effect rules prove safety.
- Loop rotation, canonicalization, interchange, unroll, and strength reduction
  when legality follows from loop structure and dependence analysis.
- Backend peepholes that preserve target ISA semantics for all inputs.
- Generic function inlining and constant-argument specialization under code-size
  limits, without recognizing benchmark names or algorithm-specific constants.

Forbidden examples:

- Matching a benchmark by exact array dimensions such as `1000`, `1024`, or a
  fixed cache table size chosen from a public case.
- Matching a unique group of constants from one case and replacing the algorithm
  with a handwritten helper.
- Encoding benchmark source line numbers, output sampling rules, final checksum
  formulas, or known runtime state-machine behavior.
- Replacing a whole timed region with a summary computation that is not derived
  by a general, local proof valid for arbitrary programs of the same IR form.
- Naming helper functions after a benchmark family or using public case names in
  compiler source.

## 2. Optimization Profiles

Current default:

- RISC-V O1/O2 keeps the general optimization stack enabled by default.
- High-risk semantic or structural recognizers are strict-mode opt-in.
- Risky passes remain individually controllable through `SISY_ENABLE_*`
  environment variables for explicit comparison runs.

Useful switches:

```bash
SISY_ENABLE_FUNCTION_EQUIVALENCE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_VECTORIZE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O2
SISY_HIR_ENABLE_INTERCHANGE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_ENABLE_UNROLL_JAM=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_JAM_FACTOR=4 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_STRUCTURAL_BITWISE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_STRUCTURAL_MODMUL=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_ROW_SCRATCH_MATMUL=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_CACHED_PRECOMPUTE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O2
SISY_ENABLE_SYNTH_CONST_ARRAY=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O2
```

Strict-mode runs should leave the recognizers above disabled unless a specific
comparison needs them. The default path should prefer runtime memoization,
affine loop transforms, scalar replacement, DSE/DLE, SCEV, and ordinary
algebraic folds.

## 3. Standard Optimization Loop

1. Establish a baseline.
   - Build natively with `scripts/build.sh`.
   - Build a Linux compiler for Docker runtime evaluation when working on macOS.
   - Run the relevant local subset with `scripts/eval-runtime.sh`.
   - Save the CSV path and the commit id.

2. Identify a compiler-level bottleneck.
   - Inspect generated IR or assembly.
   - Use pass stats to determine which generic pass failed to simplify the code.
   - Prefer missed canonicalization, missed alias proof, missed CSE, missed LICM,
     missed strength reduction, or register allocation pressure.

3. Write a legality argument.
   - State exactly which IR pattern is transformed.
   - State required dominance, alias, side-effect, type, overflow, and CFG
     conditions.
   - State what happens when any condition is not proven.

4. Implement the smallest general change.
   - Prefer improving an existing pass over adding a new pass.
   - Avoid benchmark constants unless they come from the program's own IR as a
     variable or constant and the transform remains valid for any value.
   - Add an environment kill switch for risky optimizations.

5. Verify.
   - Build successfully.
   - Run affected single-case Docker runtime probes.
   - Run the affected performance subset and at least one unrelated hotspot.
   - Run a wide compile-only sweep when changing pipeline defaults.
   - Confirm `git diff --check` is clean.

6. Submit.
   - Commit the change with a concise message.
   - Push the active branch to GitLab.
   - Record the online result before starting the next optimization.

See `docs/Commands.md` for the exact build and runtime commands currently used.

## 4. Pass Design Checklist

Before adding or modifying a pass, answer these questions:

- Does the pass mention a public test name, benchmark family, source line, or
  expected output format? If yes, do not add it.
- Would the pass still be useful on a different SysY program with different
  dimensions and constants? If no, do not add it.
- Are all memory reads and writes proven safe under alias analysis? If no, leave
  the IR unchanged.
- Does the transform preserve signed overflow, division, modulo, and shift
  semantics used by this compiler? If no, leave the IR unchanged.
- Can the transform be disabled with an environment variable while debugging? If
  no, add one for new risky transforms.
- Is the improvement measurable without hiding a correctness regression? If no,
  do not keep it.

## 5. Recommended Safe Areas

These areas are preferred because they improve broad code quality without
benchmark fingerprints:

- More complete `Mem2Reg` and phi simplification.
- Better `GVN` and copy propagation across simple CFG joins.
- Stronger alias-aware `DSE` and `DLE`.
- LICM for loop-invariant pure expressions and proven no-alias loads.
- General affine loop dependence checks for interchange and scalar replacement.
- RISC-V peepholes for redundant moves, add/sub zero, immediate materialization,
  and safe multiply/divide strength reduction.
- Register allocation heuristics based on loop depth and use count.
- Generic constant-argument specialization with conservative code-size limits.

## 6. Retired Or Strict-Mode-Only Ideas

The following ideas are not acceptable unless redesigned as general,
legality-proven compiler transformations. Some related pass names may still
exist in the tree for all-optimization experiments, but they must stay
test-identity agnostic and should retain kill switches.

- Semantic matrix summary and matrix recurrence fast paths.
- Matrix summary and row-scratch replacements tied to benchmark dimensions.
- Scheduling precompute with embedded transition matrices.
- Huffman bit-buffer predecode tied to source-specific layout.
- Random-step and row-reduce checksum semantic replacement.
- Runtime recursive memoization selected by public-case parameter sizes.
- Repeat-overwrite collapse of timed repetition loops.
- Semantic transpose replacement.

Do not reintroduce benchmark constants, output behavior assumptions, or case
names. When in doubt, keep the transform disabled by default or guard it with a
clear environment switch until the legality argument is written down.

## 7. Next-Gen Optimization Specifications

### 7.1 Register Allocator Live-Range Splitting
To dramatically reduce stack spilling in compute-intensive loops (e.g., matrix multiplication, transpose), Sisyphus supports backend **Live-Range Splitting (LRS)** inside `RegAlloc::runImpl`. 
LRS isolates a long active live range spanning multiple loop boundaries by inserting copy operations at the entrances of hot basic blocks (hotness weight >= 64):
- **Candidate Variable Selection**: Targets scalar registers (`i32`, `i64`, `f32`) that are live-in to a hot basic block, are non-constants (`!isa<LiOp>`, `!isa<LaOp>`), and are actively used inside the block.
- **Copy Insertion & Operand Re-linking**: Inserts `MvOp` (integer) or `FmvOp` (float) copies immediately after the block's Phi nodes. All local instruction uses within the block are rewritten to reference the new copy.
- **Liveness Recalculation**: Immediately triggers `region->updateLiveness()` post-splitting to update liveness intervals, letting downstream regalloc allocate local registers for loop copies and spill original variables strictly outside the loops.

### 7.2 Presburger-based 3D Loop Dependence Direction Solver
To verify the correctness of multidimensional loop interchanges and unroll-and-jam factor selections, Sisyphus utilizes a generalized, multi-dimensional Presburger Set analysis inside `HIRAffine.cpp`:
- **Domain Modeling**: Builds a generalized 2D-dimensional (e.g., 6-dimensional for 3D loops) Presburger tableau mapping loop bounds and multidimensional array index equality constraints:
  $$f_{A, m}(\vec{I}) - f_{B, m}(\vec{J}) = 0$$
- **Direction Vector Extraction**: Iteratively intersects the base iteration set with direction hypothesis vectors $\vec{d} = (d_1, d_2, d_3)$ (where $d_k \in \{<, =, >\}$) and tests feasibility using `!set.empty()`.
- **Legality Proving**: Proves loop interchange safety by ensuring that all swaps result in lexicographically positive dependence vectors (i.e. first non-equal element is less-than ($<$)).
