#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/scev"
OUT_DIR="${ROOT_DIR}/tests/.out/scev"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

stats="${OUT_DIR}/reverse_address_lsr.stats"
asm="${OUT_DIR}/reverse_address_lsr.rv.s"
"${COMPILER}" "${CASE_DIR}/reverse_address_lsr.sy" -S -o "${asm}" \
  -O1 --target=riscv --verify-ir --stats >"${stats}" 2>&1
cat "${stats}"

if ! grep -Eq '\[self-mlir\].*affine-loops=[1-9].*conversion-failed=0' "${stats}"; then
  echo "expected self-MLIR affine/SCEV-style loop recovery for reverse traversal" >&2
  exit 1
fi

if ! grep -Eq 'li[[:space:]]+t6,[[:space:]]+1' "${asm}" ||
   ! grep -Eq 'addw[[:space:]]+s[0-9]+,[[:space:]]+s[0-9]+,[[:space:]]+t6' "${asm}"; then
  echo "expected native emitter to preserve the canonical +1 induction update" >&2
  exit 1
fi

echo "self-MLIR SCEV/induction lowering tests passed."
