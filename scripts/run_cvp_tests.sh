#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/cvp"
OUT_DIR="${ROOT_DIR}/tests/.out/cvp"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run cmake --build build -j first"
  exit 1
fi

stats="$("${COMPILER}" "${CASE_DIR}/le_refines_lt.sy" -S \
  -o "${OUT_DIR}/le_refines_lt.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${stats}"

if ! grep -A2 '^range-aware-fold:$' <<<"${stats}" | grep -Eq 'folded-ops : [1-9]'; then
  echo "expected RangeAwareFold to fold a branch condition proven by correlated <= range" >&2
  exit 1
fi

eq_stats="$("${COMPILER}" "${CASE_DIR}/eq_refines_ne.sy" -S \
  -o "${OUT_DIR}/eq_refines_ne.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${eq_stats}"

if ! grep -A2 '^range-aware-fold:$' <<<"${eq_stats}" | grep -Eq 'folded-ops : [1-9]'; then
  echo "expected RangeAwareFold to fold a condition proven by correlated == range" >&2
  exit 1
fi

reversed_stats="$("${COMPILER}" "${CASE_DIR}/reversed_operand_refines.sy" -S \
  -o "${OUT_DIR}/reversed_operand_refines.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${reversed_stats}"

if ! grep -A2 '^range-aware-fold:$' <<<"${reversed_stats}" | grep -Eq 'folded-ops : [1-9]'; then
  echo "expected RangeAwareFold to fold a condition with the refined value on RHS" >&2
  exit 1
fi

echo "Correlated value propagation range folding tests passed."
