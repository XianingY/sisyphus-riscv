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
  - `--enable-experimental` (opt-in for experimental O1/O2 passes; off by default)
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
  - Defaults: `-O1 => inline/late=200`, `-O2 => inline/late=256` (unless explicitly overridden), and loop-rotate off by default for O1/O2

## Build

```bash
scripts/build.sh
```

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
# Compile-only regression
scripts/regression.sh test/custom riscv O0
scripts/regression.sh test/custom riscv O1
scripts/regression.sh test/custom riscv O2
scripts/regression.sh test/custom arm O0
scripts/regression.sh test/custom arm O1
scripts/regression.sh test/custom arm O2

# Semantic compare (interpreter vs expected output)
scripts/compare.sh test/custom riscv O1
scripts/compare.sh test/custom arm O1
scripts/compare.sh test/custom riscv O2
scripts/compare.sh test/custom arm O2

# Fast O0/O1 assembly-size proxy
scripts/asm-delta.sh test/custom riscv

# Matrix evaluation for O1 tuning candidates
scripts/eval-o1-matrix.sh test/custom

# Unified O1/O2 matrix evaluation (RISC-V proxy + ARM consistency checks)
scripts/eval-profile-matrix.sh test/custom

# Sync official ZIP suites and generate suite index
scripts/suite-sync.sh --update --src-root /home/wslootie/github/cpe/compiler2025
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

## Runtime

A local runtime library is provided in `runtime/sylib.c` and `runtime/sylib.h`.

Runtime evaluation environment variables:

- `SISY_DOCKER_IMAGE` (default: `sisyphus/compiler-dev-dual:latest`)
- `BIFRAME_COMPILER` (default: `/home/wslootie/github/cpe/biframe/build/sysc`)
- `RUNTIME_CASE_LIMIT` / `RUNTIME_CASE_FILTER` (optional smoke/debug subset controls)
- Runtime CSV includes `suspect_input_underflow` to label likely input-underflow/UB-sensitive cases.
- Runtime CSV also includes `frontend_path` and `fallback_reason_codes` (for dialect migration coverage tracking).
- `SISY_COMPILER_EXTRA_ARGS` can be used to forward extra compiler flags in runtime/regression/compare scripts.

Compare/validator environment variables:

- `COMPARE_TIMEOUT_SEC` (default: `30`, per-case compiler compare timeout)
- `COMPARE_INCLUDE_PERF` (default: `0`, skip `perf/*` in `scripts/compare.sh`)
- `SISY_EXEC_STEP_LIMIT` (default: `20000000`, interpreter step budget used by `--compare`)

## QEMU

- AArch64 launcher: `scripts/qemu-aarch64.sh`
- RISC-V launcher: `scripts/qemu-riscv64.sh`
- Workflow details: `docs/QEMU.md`

## Public Suites

- `official-functional`: hard gate (`functional.zip`, SysY standard functional set)
- `official-arm-perf`: soft perf gate (`ARM-性能.zip`)
- `official-riscv-perf`: soft perf gate (`RISCV-性能.zip`)
- `official-arm-final-perf`: soft perf gate (`ARM决赛性能用例.zip`)
- `official-riscv-final-perf`: soft perf gate (`RISCV决赛性能用例.zip`)

Legacy public-suite repos (`open-test-cases`, `compiler-dev-test-cases`, `lvx`) are no longer part of this project's default baseline.

## Design Docs

- `docs/Design.md`
- `docs/Compliance.md`

## Notes

This repository is built as a standalone implementation. Other repositories are used only as design references.
