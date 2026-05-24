#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/scev"
OUT_DIR="${ROOT_DIR}/tests/.out/scev"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run cmake --build build -j first"
  exit 1
fi

reverse_ir_file="${OUT_DIR}/reverse_address_lsr.after-scev.ir"
"${COMPILER}" "${CASE_DIR}/reverse_address_lsr.sy" -S \
  -o "${OUT_DIR}/reverse_address_lsr.rv.s" \
  -O1 --target=riscv --verify-ir --print-after scev >"${reverse_ir_file}" 2>&1

if ! grep -Eq 'int <-4>' "${reverse_ir_file}"; then
  echo "expected SCEV LSR to use a -4 byte stride for reverse int-array traversal" >&2
  tail -120 "${reverse_ir_file}" >&2
  exit 1
fi

echo "SCEV loop strength reduction tests passed."
