# Command Reference

This file records the commands currently used for local development,
verification, and submit-before-checks. Generated artifacts should stay under
ignored directories such as `build/`, `build-linux/`, `tests/.out/`, and
`output/`.

## Build

Native build:

```bash
scripts/build.sh
```

Manual native build:

```bash
cmake -S . -B build -DDEFAULT_TARGET=riscv
cmake --build build -j
```

Linux compiler build for Docker/QEMU runtime evaluation:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'cmake -S . -B build-linux -G "Unix Makefiles" && cmake --build build-linux -j 8'
```

ARM default-target build:

```bash
DEFAULT_TARGET=arm scripts/build.sh
```

## Compile

```bash
./build/compiler test2026/performance_riscv/fft0.sy -S -o tests/.out/fft0.rv.s --target=riscv -O1
./build/compiler test2026/performance_arm/fft0.sy -S -o tests/.out/fft0.arm.s --target=arm -O1
./build/compiler test2026/performance_riscv/fft0.sy -S -o tests/.out/fft0.o2.s --target=riscv -O2
```

Useful diagnostics:

```bash
./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1 --stats
./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1 --dump-pass-timing
./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1 --verify-ir
SISY_DUMP_SELF_MLIR=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1 --stats
```

## Smoke And Regression

Small built-in smoke set:

```bash
scripts/run_smoke.sh
```

Compile-only regression over any local case directory:

```bash
scripts/regression.sh tests/smoke riscv O0
scripts/regression.sh tests/smoke arm O0
```

## Runtime Evaluation

Use the Linux compiler for runtime evaluation on macOS:

```bash
export SISY_COMPILER_PATH="$PWD/build-linux/compiler"
export RUNTIME_SYLIB_C="$PWD/runtime/sylib.c"
```

Functional suite:

```bash
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-suite-root" \
  scripts/eval-runtime.sh official-functional riscv O1
```

RISC-V performance suite:

```bash
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-perf-suite-root" \
RUNTIME_SOFT_PERF=1 RUNTIME_PERF_TIMEOUT_SEC=20 \
  scripts/eval-runtime.sh official-riscv-perf riscv O1
```

Native smoke and single-case runtime probes:

```bash
scripts/run_native_riscv_runtime_smoke.sh
SISY_REQUIRE_RISCV_RUNTIME=1 scripts/run_native_riscv_runtime_smoke.sh
```

The native smoke script reports `SKIP` instead of success when the local
RISC-V GCC/QEMU toolchain is unavailable.

```bash
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-perf-suite-root" \
RUNTIME_CASE_FILTER=fft0 RUNTIME_CASE_LIMIT=1 \
RUNTIME_SOFT_PERF=1 RUNTIME_PERF_TIMEOUT_SEC=30 \
  scripts/eval-runtime.sh official-riscv-perf riscv O1
```

self-MLIR compile/stat baseline over the public RISC-V performance sources:

```bash
scripts/eval-self-mlir-perf-baseline.sh
```

The report is written to `tests/.out/self-mlir-perf-baseline/baseline.csv` and
contains compile status, assembly size, self-MLIR op counts, native machine op
counts, and local optimization stats. Case names are used only for reporting
categories in this script, never as compiler triggers.

## Wide Compile Checks

Compile all known 2026 RISC-V functional and performance sources:

```bash
mkdir -p tests/.out/wide-compile
for f in $(find test2026/riscv_func test2026/performance_riscv -name '*.sy' | sort); do
  out="tests/.out/wide-compile/$(printf '%s' "$f" | sed 's#[/ ]#__#g; s#\.sy$#.s#')"
  ./build/compiler "$f" -S -o "$out" --target=riscv -O1
done
```

## Optimization Switches

The default RISC-V profile keeps general optimizations enabled. High-risk
semantic or structural recognizers are strict-mode opt-in, while ordinary
generic passes keep bisection kill switches. Examples:

```bash
SISY_ENABLE_SELF_AFFINE_OPT=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_SELF_MEMOPT=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_SELF_PROVEN_BITWISE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_SELF_MACHINE_LIVENESS=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_SELF_MACHINE_SCHED=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_VECTORIZE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O2
SISY_HIR_ENABLE_INTERCHANGE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_ENABLE_UNROLL_JAM=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_ENABLE_REDUCTION_PRIVATIZE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_ENABLE_REDUCTION_MICROTILE=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_ENABLE_CONDITIONAL_ROW_JAM=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_ENABLE_CONDITIONAL_REDUCTION_MICROTILE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_SYNTH_CONST_ARRAY=0 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
# Diagnostic cost-model overrides; never key off case identity.
SISY_HIR_MICRO_NR=2 SISY_HIR_MICRO_KC=32 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_HIR_ENABLE_STENCIL_INTERIOR=1 SISY_HIR_STENCIL_SPLIT_COLUMNS=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
```

Performance reporting helpers:

```bash
scripts/eval-self-mlir-perf-baseline.sh
```

The generated CSV and Markdown summary classify public `performance_riscv`
cases for reporting only. Case names and paths are not compiler triggers.

Strict-mode comparison switches:

```bash
SISY_ENABLE_FUNCTION_EQUIVALENCE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_STRUCTURAL_BITWISE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_STRUCTURAL_MODMUL=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_ROW_SCRATCH_MATMUL=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_CACHED_PRECOMPUTE=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O2
SISY_ENABLE_ADVANCED_CONV2D=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
SISY_ENABLE_CFG_BOUNDS_CHECK=1 ./build/compiler testcase.sy -S -o tests/.out/case.s --target=riscv -O1
```

Do not use the strict-mode switches as submission defaults without updating
`docs/Compliance.md` with a general legality argument and running a broad
correctness sweep.
