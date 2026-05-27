# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sisyphus is a SysY compiler for the 2026 compiler contest. It has a unified frontend/IR pipeline and two architecture-specific backends: RISC-V `rv64gc` and ARM64/AArch64.

## Build

```bash
# Default native build; DEFAULT_TARGET defaults to riscv
scripts/build.sh

# ARM default-target build
DEFAULT_TARGET=arm scripts/build.sh

# Manual CMake build
cmake -S . -B build -DDEFAULT_TARGET=riscv
cmake --build build -j

# Linux compiler for Docker/QEMU runtime evaluation on macOS
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'cmake -S . -B build-linux -G "Unix Makefiles" && cmake --build build-linux -j 8'
```

Build output is `build/compiler`. `scripts/build.sh` accepts `JOBS=<N>` and `BUILD_TYPE=<type>` through the environment.

There is no dedicated lint/format target in CMake or `package.json`; use `git diff --check` for whitespace checks before committing.

## Test and Verification Commands

```bash
# Smoke test: compile tests/smoke/*.sy for both targets at -O0 with IR verification
scripts/run_smoke.sh

# CTest wrappers for registered smoke suites
ctest --test-dir build --output-on-failure
ctest --test-dir build -R smoke_compile_dual_target --output-on-failure

# Compile-only regression over any case directory
scripts/regression.sh tests/smoke riscv O0
scripts/regression.sh tests/smoke arm O0
scripts/regression.sh tests/polyhedral riscv O1 --verify-ir

# Single-case compile via helper
scripts/compile_case.sh tests/smoke/basic.sy tests/.out/basic.rv.s riscv O1
scripts/compile_case.sh tests/smoke/basic.sy tests/.out/basic.arm.s arm O1

# Direct single-case compiler invocation with diagnostics
./build/compiler tests/smoke/basic.sy -S -o tests/.out/basic.s --target=riscv -O1 --verify-ir --dump-pass-timing
```

Runtime evaluation uses Docker/QEMU and the SysY runtime in `runtime/`:

```bash
scripts/eval-runtime.sh official-functional riscv O1

# On macOS, prefer the Docker-built Linux compiler for runtime execution
SISY_COMPILER_PATH="$PWD/build-linux/compiler" \
RUNTIME_SYLIB_C="$PWD/runtime/sylib.c" \
scripts/eval-runtime.sh official-functional riscv O1

# Single-case runtime probe within a suite
RUNTIME_CASE_FILTER=fft0 RUNTIME_CASE_LIMIT=1 \
RUNTIME_SOFT_PERF=1 RUNTIME_PERF_TIMEOUT_SEC=30 \
scripts/eval-runtime.sh official-riscv-perf riscv O1
```

Generated artifacts should stay under ignored directories such as `build/`, `build-linux/`, `tests/.out/`, and `output/`.

## Compiler Usage

```bash
./build/compiler testcase.sy -S -o testcase.s -O1
./build/compiler testcase.sy -S -o testcase.s -O2 --target=arm
./build/compiler testcase.sy -S -o testcase.s -O1 --emit-ir --verify-ir
./build/compiler testcase.sy -S -o testcase.s -O1 --dump-hir --dump-cfg --verify-hir --verify-cfg
```

Useful flags:

- `--target=riscv|arm` selects the backend target.
- `-O0`, `-O1`, `-O2` select optimization profiles.
- `--emit-ir`, `--verify-ir`, `--dump-pass-timing`, `--stats` help inspect the legacy IR pipeline.
- `--dump-hir`, `--verify-hir`, `--dump-cfg`, `--verify-cfg` help inspect the dialect frontend pipeline.
- `--use-legacy-codegen` forces the legacy AST-to-IR frontend path.
- `--force-dialect-codegen` disables frontend fallback and fails fast on unsupported dialect lowering.
- `SISY_COMPILER_EXTRA_ARGS="..."` forwards extra compiler flags through regression/runtime scripts.

## Architecture

The default frontend path is the dialect pipeline:

`AST -> HIR build -> HIR verify/canonicalize -> CFG -> CFG verify -> legacy ModuleOp adapter -> existing optimization/backend pipeline`

The legacy frontend path (`AST -> CodeGen -> ModuleOp`) is still available with `--use-legacy-codegen`. Backend and O1/O2 pipelines operate on the existing legacy `ModuleOp` representation after the CFG adapter.

High-level source layout:

- `src/main/`: CLI parsing, default target selection, and optimization pipeline profile construction.
- `src/parse/`: SysY lexer/parser/type-system pieces that produce the AST.
- `src/frontend/`: frontend facade; new top-level integration should include this facade instead of raw parse/codegen internals.
- `src/hir/`: structured high-level IR, builder, verifier, canonicalization, affine/polyhedral utilities, and migration lowering.
- `src/cfg/`: explicit control-flow dialect between HIR and legacy IR, including CFG verification and the CFG-to-legacy adapter.
- `src/codegen/`: legacy IR data model (`ModuleOp`/ops/attrs) and legacy AST-to-IR lowering.
- `src/pre-opt/`: early structured/CFG cleanup passes such as alloca movement, const folding, loop raising, CFG flattening, and mem2reg.
- `src/opt/`: mid-level optimization passes such as alias analysis, DSE/DLE, GVN, LICM, inlining, loop transforms, range folding, vectorization, and pass management.
- `src/pass/`: thin forwarding header (`PassRegistry.h`) that re-exports `PassManager` and `PipelineProfiles`; actual pass ordering is centralized in `src/main/PipelineProfiles.cpp`.
- `src/rv/` and `src/arm/`: current RISC-V and ARM backend lowering, scheduling, register allocation, and assembly dumping.
- `src/backend/`: thin architecture-boundary facade headers (`BackendPasses.h` per target, `RegAllocHotness.h` shared utility, `RiscvParams.h` hardware constants); primary implementation still lives in `src/rv/` and `src/arm/`.
- `src/utils/`: common utilities, including Presburger/SMT helpers used by loop and affine analyses.
- `runtime/`: SysY runtime library used by runtime evaluation scripts.

## Optimization Profiles and Bisection

- **O0**: early cleanup and canonicalization sufficient for straightforward code generation.
- **O1**: O0 plus structured CFG optimization, alias analysis, DSE/DLE, GVN, LICM, and inlining.
- **O2**: O1 plus more aggressive loop, range, splice, vectorization, and tail optimization rounds.

Before changing pipeline defaults, read `docs/Compliance.md`, `docs/Commands.md`, and `docs/Optimization.md`. Optimizations must be general compiler transformations over IR and must not depend on benchmark names, source filenames, public test identity, expected output formats, hidden inputs, or magic constants unique to a case.

The default profile should prefer SSA, MemorySSA-style load/store reasoning,
alias/noalias proofs, affine dependence analysis, range/SCEV folds,
vectorization legality, register-allocation hotness, scheduling, and target
peepholes. Do not enable semantic whole-function recognizers or algorithm
helper replacements as a default scoring path.

Common bisection switches:

```bash
SISY_ENABLE_VECTORIZE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O2
SISY_HIR_ENABLE_INTERCHANGE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_ENABLE_UNROLL_JAM=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_FUNCTION_EQUIVALENCE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_STRUCTURAL_BITWISE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_STRUCTURAL_MODMUL=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_ROW_SCRATCH_MATMUL=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_CACHED_PRECOMPUTE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O2
SISY_ENABLE_SYNTH_CONST_ARRAY=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O2
SISY_ENABLE_ADVANCED_CONV2D=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
```

The switches set to `1` above are strict-mode/experiment toggles, not default
submission guidance.

## Dual-Target Submission

- **RISC-V track**: submit the `master` branch.
- **ARM track**: create/use an `arm` branch and set `SISYPHUS_DEFAULT_TARGET_ARM=1` in `src/main/DefaultTarget.h`.

Contest evaluators may compile `src/*.cpp` directly without CMake and may not pass `--target`, so the source-level default target in `src/main/DefaultTarget.h` controls submitted branch behavior.
