# Optimization Workflow

This document defines the allowed optimization workflow for this compiler.
It intentionally excludes case-specific semantic replacement passes, hard-coded
benchmark constants, output-shape shortcuts, and source-line or test-shape
fingerprints.

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

## 2. Standard Optimization Loop

1. Establish a baseline.
   - Build in Docker with `DEFAULT_TARGET=riscv scripts/build.sh`.
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
   - Run `official-functional riscv O1`.
   - Run the affected performance subset and at least one unrelated hotspot
     subset.
   - Confirm `git diff --check` is clean.

6. Submit.
   - Commit the change with a concise message.
   - Push `master` to GitLab.
   - Record the online result before starting the next optimization.

## 3. Pass Design Checklist

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

## 4. Recommended Safe Areas

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

## 5. Removed Non-Compliant Optimizations

The following previously existing pass families were removed because they were
too specific to public benchmark semantics or used fragile fingerprint-like
conditions:

- Semantic matrix summary and matrix recurrence fast paths.
- Row-scratch matrix multiplication replacement.
- Scheduling precompute with embedded transition matrices.
- Huffman bit-buffer predecode and bitwise helper semantic replacement.
- Random-step and row-reduce checksum semantic replacement.
- Runtime recursive memoization with fixed cache dimensions.
- Repeat-overwrite collapse of timed repetition loops.
- Semantic transpose replacement.

Do not reintroduce these ideas unless they are redesigned as fully general,
legality-proven compiler optimizations without benchmark constants or output
behavior assumptions.
