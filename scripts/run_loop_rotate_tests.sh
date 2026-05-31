#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/loop_rotate"
OUT_DIR="${ROOT_DIR}/tests/.out/loop_rotate"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

compile_case() {
  local name="$1"
  local source_name="$2"
  local asm="${OUT_DIR}/${name}.rv.s"
  local stats="${OUT_DIR}/${name}.stats"
  shift 2
  echo "[loop-cfg] compile ${name}"
  "$@" "${COMPILER}" "${CASE_DIR}/${source_name}.sy" -S -o "${asm}" \
    -O1 --target=riscv --verify-ir --stats >"${stats}" 2>&1
  if ! grep -Eq '\[self-mlir\].*frontend_path=self-mlir.*failed=0' "${stats}" ||
     ! grep -Eq '\[native-asm\].*emitted=1.*legacy-ops=0.*phi-like-ops=0' "${stats}"; then
    cat "${stats}" >&2
    echo "expected self-MLIR native RISC-V compile for ${name}" >&2
    exit 1
  fi
  if grep -Eq '\bvsetvli\b|\bvle[0-9]+\.v\b|\bvse[0-9]+\.v\b' "${asm}"; then
    echo "loop/control-flow tests unexpectedly emitted RVV instructions for ${name}" >&2
    exit 1
  fi
}

compile_case positive_loops positive_loops env
compile_case positive_loops.no-rotate positive_loops env SISY_ENABLE_LOOP_ROTATE=0
compile_case zero_trip_call zero_trip_call env
compile_case call_loop call_loop env
compile_case store_loop store_loop env
compile_case store_loop.store-rotate store_loop env SISY_ENABLE_LOOP_ROTATE_STORES=1
compile_case exit_phi exit_phi env
compile_case conditional_memory conditional_memory env

if ! grep -Eq 'affine-loops=[1-9]|scf-loops=[1-9]' "${OUT_DIR}/positive_loops.stats"; then
  cat "${OUT_DIR}/positive_loops.stats" >&2
  echo "expected positive_loops to exercise self-MLIR loop lowering" >&2
  exit 1
fi

echo "RISC-V self-MLIR loop/control-flow compile tests passed."
