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

nonzero_stats="$("${COMPILER}" "${CASE_DIR}/nonzero_refines_not.sy" -S \
  -o "${OUT_DIR}/nonzero_refines_not.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${nonzero_stats}"

if ! grep -A2 '^range-aware-fold:$' <<<"${nonzero_stats}" | grep -Eq 'folded-ops : [1-9]'; then
  echo "expected RangeAwareFold to fold !x when correlated range proves x is nonzero" >&2
  exit 1
fi

value_eq_stats="$("${COMPILER}" "${CASE_DIR}/eq_value_refines_sub.sy" -S \
  -o "${OUT_DIR}/eq_value_refines_sub.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${value_eq_stats}"

if ! grep -A4 '^range-aware-fold:$' <<<"${value_eq_stats}" | grep -Eq 'path-replacements : [1-9]'; then
  echo "expected RangeAwareFold to substitute a value using path-sensitive x == y equality" >&2
  exit 1
fi

jump_thread_stats="$("${COMPILER}" "${CASE_DIR}/jump_thread_lt_chain.sy" -S \
  -o "${OUT_DIR}/jump_thread_lt_chain.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${jump_thread_stats}"

if ! grep -A5 '^range-aware-fold:$' <<<"${jump_thread_stats}" | grep -Eq 'threaded-edges : [1-9]'; then
  echo "expected RangeAwareFold to thread an edge whose successor branch is decided by predecessor range" >&2
  exit 1
fi

echo "Correlated value propagation range folding tests passed."
