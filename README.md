# Sisyphus

Sisyphus is a SysY compiler project for the 2026 compiler contest track.

## Features

- Unified frontend + IR pipeline
- Dual backends:
  - RISC-V (`rv64gc`)
  - ARM64 (`AArch64`, ARMv8-A)
- Competition-compatible CLI:
  - `compiler testcase.sy -S -o testcase.s`
  - `compiler testcase.sy -S -o testcase.s -O1`
  - `compiler testcase.sy -S -o testcase.s -O2`
- Debug and tuning flags:
  - `--target=riscv|arm`
  - `--emit-ir`
  - `--verify-ir`
  - `--dump-pass-timing`
  - `--enable-experimental` (opt-in for additional experimental O1/O2 behavior)
  - `--use-legacy-codegen` (escape hatch during dialect migration)
  - `--force-dialect-codegen` (forbid automatic fallback; fail fast on unsupported dialect lowering)
  - `--dialect-fallback-report=stderr|<path>` (emit frontend path and fallback reason codes)
  - `--enable-hir-pipeline` / `--disable-hir-pipeline`
  - `--dump-hir` / `--dump-cfg`
  - `--verify-hir` / `--verify-cfg`
  - `--inline-threshold=<N>`
  - `--late-inline-threshold=<N>`
  - `--disable-loop-rotate`
  - `--enable-loop-rotate`
  - `--disable-const-unroll`
  - Defaults: `-O1 => inline/late=200`, `-O2 => inline/late=256` unless explicitly overridden

## Build

```bash
scripts/build.sh
```

`scripts/build.sh` is portable across Linux and macOS. Use `JOBS=<N>` to
override parallelism.

Contest evaluators may compile all `src/*.cpp` files directly instead of using
CMake. In that mode the default target comes from
`src/main/DefaultTarget.h`.

Default target can be changed at configure time:

```bash
DEFAULT_TARGET=arm scripts/build.sh
```

or with raw CMake:

```bash
cmake -S . -B build -DDEFAULT_TARGET=arm
cmake --build build -j
```

For Docker/QEMU runtime evaluation on macOS, build a Linux compiler as well:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'cmake -S . -B build-linux -G "Unix Makefiles" && cmake --build build-linux -j 8'
```

## Usage

```bash
# RISC-V (default)
./build/compiler tests/smoke/basic.sy -S -o basic.rv.s -O1

# ARM
./build/compiler tests/smoke/basic.sy -S -o basic.arm.s -O1 --target=arm

# Aggressive profile
./build/compiler tests/smoke/basic.sy -S -o basic.rv.o2.s -O2

# Default frontend path is now dialect pipeline (HIR->CFG->legacy ModuleOp adapter)
./build/compiler tests/smoke/basic.sy -S -o basic.rv.dialect.s -O2 --dump-cfg

# Legacy fallback path for A/B and rollback
./build/compiler tests/smoke/basic.sy -S -o basic.rv.legacy.s -O2 --use-legacy-codegen

# Forced pure dialect path (no fallback)
./build/compiler tests/smoke/basic.sy -S -o basic.rv.forced.s -O1 --force-dialect-codegen
```

## 2026 Dual-Track Submission

Use the same GitLab repository with two branches:

- RISC-V track: submit `https://gitlab.eduxiji.net/T2026104862010407/compiler2026-sisyphus.git master`.
- ARM track: create an `arm` branch, set `SISYPHUS_DEFAULT_TARGET_ARM` to `1`
  in `src/main/DefaultTarget.h`, then submit
  `https://gitlab.eduxiji.net/T2026104862010407/compiler2026-sisyphus.git arm`.

The evaluator command does not pass `--target`, so each submitted branch must
have the correct default target. The compiler still accepts `--target=riscv`
and `--target=arm` for local testing.

## Smoke Test

```bash
scripts/run_smoke.sh
```

## Regression

```bash
# Compile-only smoke regression
scripts/regression.sh tests/smoke riscv O0
scripts/regression.sh tests/smoke arm O0

# Sync official ZIP suites into ignored local samples and generate suite index
scripts/suite-sync.sh --update --src-root /Users/byzantium/github/compiler2025
scripts/suite-index.sh

# Runtime eval (Docker-first, QEMU user-mode)
scripts/eval-runtime.sh official-functional riscv O1
RUNTIME_SOFT_PERF=1 RUNTIME_PERF_TIMEOUT_SEC=20 scripts/eval-runtime.sh official-arm-perf arm O2
RUNTIME_SOFT_PERF=1 RUNTIME_PERF_TIMEOUT_SEC=20 scripts/eval-runtime.sh official-riscv-perf riscv O1

# Dialect coverage report (default vs forced-dialect on same suite)
scripts/eval-dialect-coverage.sh official-functional riscv O1

# Pass extra compiler args into eval-runtime/regression/compare
SISY_COMPILER_EXTRA_ARGS="--force-dialect-codegen" scripts/eval-runtime.sh official-functional riscv O1

# Compare against local biframe compiler
scripts/eval-vs-biframe.sh official-functional riscv O1

# Official dataset adapter (safe-skip if dirs are absent)
scripts/eval-official-adapter.sh /path/to/official/functional /path/to/official/perf /path/to/runtime
```

When running runtime checks with the Docker-built Linux compiler:

```bash
export SISY_COMPILER_PATH="$PWD/build-linux/compiler"
export RUNTIME_SYLIB_C="$PWD/runtime/sylib.c"
```

## Runtime

A local runtime library is provided in `runtime/sylib.c` and `runtime/sylib.h`.

Runtime evaluation environment variables:

- `SISY_DOCKER_IMAGE` (default: `sisyphus/compiler-dev-dual:latest`)
- `BIFRAME_COMPILER` (default: `/home/wslootie/github/cpe/biframe/build/sysc`)
- `RUNTIME_CASE_LIMIT` / `RUNTIME_CASE_FILTER` (optional smoke/debug subset controls)
- Runtime CSV includes `suspect_input_underflow` to label likely input-underflow/UB-sensitive cases.
- Runtime CSV also includes `frontend_path` and `fallback_reason_codes` (for dialect migration coverage tracking).
- `SISY_COMPILER_EXTRA_ARGS` can be used to forward extra compiler flags in runtime/regression/compare scripts.

Optimization bisection switches:

- `SISY_ENABLE_FUNCTION_EQUIVALENCE=0`
- `SISY_ENABLE_VECTORIZE=0`
- `SISY_HIR_ENABLE_INTERCHANGE=0`
- `SISY_HIR_ENABLE_UNROLL_JAM=0`
- `SISY_HIR_ENABLE_REDUCTION_MICROTILE=0`

Strict-mode opt-in switches for high-risk recognizers:

- `SISY_ENABLE_STRUCTURAL_BITWISE=1`
- `SISY_ENABLE_STRUCTURAL_MODMUL=1`
- `SISY_ENABLE_ROW_SCRATCH_MATMUL=1`
- `SISY_ENABLE_CACHED_PRECOMPUTE=1`
- `SISY_ENABLE_SYNTH_CONST_ARRAY=1`

Compare/validator environment variables:

- `COMPARE_TIMEOUT_SEC` (default: `30`, per-case compiler compare timeout)
- `COMPARE_INCLUDE_PERF` (default: `0`, skip `perf/*` in `scripts/compare.sh`)
- `SISY_EXEC_STEP_LIMIT` (default: `20000000`, interpreter step budget used by `--compare`)

## QEMU

- AArch64 launcher: `scripts/qemu-aarch64.sh`
- RISC-V launcher: `scripts/qemu-riscv64.sh`
- Workflow details: `docs/QEMU.md`

## Public Suites

Official suites are extracted under `test/official/` for local evaluation. This
directory is ignored by Git and should not be uploaded to the contest GitLab.

- `official-functional`: hard gate (`functional.zip`, SysY standard functional set)
- `official-arm-perf`: soft perf gate (`ARM-性能.zip`)
- `official-riscv-perf`: soft perf gate (`RISCV-性能.zip`)
- `official-arm-final-perf`: soft perf gate (`ARM决赛性能用例.zip`)
- `official-riscv-final-perf`: soft perf gate (`RISCV决赛性能用例.zip`)

Legacy public-suite repos (`open-test-cases`, `compiler-dev-test-cases`, `lvx`) are no longer part of this project's default baseline.

## Design Docs

- `docs/Design.md`
- `docs/Compliance.md`
- `docs/Commands.md`
- `docs/Optimization.md`

## Optimization And Compliance

Default optimization profiles are intended to be general compiler profiles, not
benchmark recognizers. High-risk semantic recognizers such as function
equivalence sampling, structural bitwise/modmul replacement, row-scratch matrix
helper replacement, compile-time recursive precompute, synthesized constant
arrays, and advanced conv2d dispatch are disabled by default and should only be
used for explicit comparison runs.

Preferred optimization work should go through ordinary IR facts: SSA,
MemorySSA-style reaching definitions, alias/noalias proofs, affine loop
dependence, range analysis, SCEV, vector legality, register allocation hotness,
and target peepholes. See `docs/Compliance.md` for the required legality
boundary before changing defaults.

## Notes

This repository is built as a standalone implementation. Other repositories are used only as design references.
