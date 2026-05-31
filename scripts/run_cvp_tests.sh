#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/cvp"
OUT_DIR="${ROOT_DIR}/tests/.out/cvp"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

cases=(
  le_refines_lt
  eq_refines_ne
  reversed_operand_refines
  nonzero_refines_not
  eq_value_refines_sub
  jump_thread_lt_chain
)

for name in "${cases[@]}"; do
  asm="${OUT_DIR}/${name}.rv.s"
  stats="${OUT_DIR}/${name}.stats"
  "${COMPILER}" "${CASE_DIR}/${name}.sy" -S -o "${asm}" \
    -O1 --target=riscv --verify-ir --stats >"${stats}" 2>&1
  cat "${stats}"
  if ! grep -Eq '\[self-mlir\].*frontend_path=self-mlir.*conversion-failed=0' "${stats}" ||
     ! grep -Eq '\[native-asm\].*emitted=1.*unsupported=0' "${stats}"; then
    echo "expected self-MLIR correlated-branch case to lower cleanly: ${name}" >&2
    exit 1
  fi
done

echo "self-MLIR correlated branch lowering tests passed."
