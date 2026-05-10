# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sisyphus is a SysY compiler for ARM64 (AArch64) and RISC-V (rv64gc) used in the 2026 compiler contest. It uses a unified frontend/IR pipeline with architecture-specific backends.

## Build

```bash
# Default (RISC-V)
scripts/build.sh

# ARM target
DEFAULT_TARGET=arm scripts/build.sh

# Manual CMake
cmake -S . -B build -DDEFAULT_TARGET=riscv
cmake --build build -j
```

Output: `build/compiler`

## Common Commands

```bash
# Smoke test (compiles all tests/smoke/*.sy for both targets with -O0)
scripts/run_smoke.sh

# Regression on a test suite
scripts/regression.sh tests/smoke riscv O0
scripts/regression.sh tests/smoke arm O0

# Runtime evaluation
scripts/eval-runtime.sh official-functional riscv O1
```

## Compiler Usage

```bash
./build/compiler testcase.sy -S -o testcase.s -O1
./build/compiler testcase.sy -S -o testcase.s -O2 --target=arm
```

Key flags:
- `--target=riscv|arm` (default from CMake `DEFAULT_TARGET`)
- `-O0`, `-O1`, `-O2`
- `--emit-ir`, `--verify-ir`, `--dump-pass-timing`
- `--dump-cfg`, `--verify-cfg`
- `--use-legacy-codegen` (fallback frontend path)
- `--force-dialect-codegen` (no fallback, fail fast)

## Architecture

```
src/
├── main/          # Entry point, CLI options, pipeline profiles
├── parse/         # Lexer + Parser (SysY -> AST)
├── frontend/      # AST -> HIR lowering
├── hir/          # High-level IR (HIRBuilder, HIRCanonicalize, HIRVerifier)
├── cfg/          # Control Flow Graph representation
├── codegen/      # CFG -> legacy ModuleOp lowering
├── pre-opt/      # Early optimization (MoveAlloca, EarlyConstFold, RaiseToFor, FlattenCFG, Mem2Reg)
├── opt/          # Optimization passes (O1/O2): Alias, DSE, DLE, GVN, LICM, Inline, LateInline, etc.
├── ir/           # Core IR types
├── pass/         # Pass registry
├── backend/
│   ├── riscv/    # RISC-V backend (lowering, combine, regalloc, asm dump)
│   ├── arm/      # ARM64 backend
│   └── shared/   # Shared backend utilities
├── utils/        # Common utilities (Arena, Bitstream, etc.)
└── rt/           # Runtime support
```

### Dialect Pipeline (default)

`AST -> HIR Build -> HIR Verify -> HIR Canonicalize -> HIR->CFG -> CFG Verify -> CFG->Legacy ModuleOp`

### Optimization Levels

- **O0**: MoveAlloca, EarlyConstFold, RaiseToFor, FlattenCFG, Mem2Reg, RegularFold
- **O1**: O0 + structured CFG optimization, alias analysis, DSE, DLE, GVN, LICM, inlining
- **O2**: O1 + Fusion, Unswitch, Reassociate, Range+Splice, additional tail optimization rounds

### Key Optimization Passes

`src/opt/`: Alias.cpp, DSE.cpp, DLE.cpp, GVN.cpp, LICM.cpp, Inline.cpp, LateInline.cpp, CanonicalizeLoop.cpp, FlattenCFG.cpp, GCM.cpp (Globalize.cpp, AggressiveDCE.cpp)

## Dual-Target Submission

- **RISC-V track**: submit `master` branch
- **ARM track**: create `arm` branch, set `SISYPHUS_DEFAULT_TARGET_ARM=1` in `src/main/DefaultTarget.h`

Contest evaluators compile `src/*.cpp` directly without CMake, so the `DefaultTarget.h` source-level define controls the default target.
