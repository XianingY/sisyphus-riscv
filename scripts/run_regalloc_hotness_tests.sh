#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/regalloc"
OUT_DIR="${ROOT_DIR}/tests/.out/regalloc"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run cmake --build build -j first"
  exit 1
fi

stats="$("${COMPILER}" "${CASE_DIR}/nested_loop_hotness.sy" -S \
  -o "${OUT_DIR}/nested_loop_hotness.rv.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"

echo "${stats}"

if ! grep -A4 '^rv-regalloc:$' <<<"${stats}" | grep -Eq 'max-block-hotness : ([6-9][4-9]|[1-9][0-9]{2,})'; then
  echo "expected RV regalloc to expose nested-loop hotness >= 64" >&2
  exit 1
fi

split_stats="$("${COMPILER}" "${CASE_DIR}/hot_loop_no_call_split.sy" -S \
  -o "${OUT_DIR}/hot_loop_no_call_split.rv.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"

echo "${split_stats}"

if ! grep -A5 '^rv-regalloc:$' <<<"${split_stats}" | grep -Eq 'live-range-splits : [1-9]'; then
  echo "expected RV regalloc to split live-in values for hot loops without calls" >&2
  exit 1
fi

echo "RV register-allocation loop hotness tests passed."
