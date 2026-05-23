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

extract_pass_stat() {
  local stats="$1"
  local pass="$2"
  local key="$3"
  awk -v pass="${pass}" -v key="${key}" '
    $0 == pass ":" { in_pass = 1; next }
    in_pass && /^[^[:space:]].*:$/ { in_pass = 0 }
    in_pass && $1 == key { print $3; exit }
  ' <<<"${stats}"
}

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

if ! grep -A5 '^rv-regalloc:$' <<<"${split_stats}" | grep -Eq 'live-range-splits : ([3-9]|[1-9][0-9]+)'; then
  echo "expected RV regalloc to split all hot live-in scalar values, not only one per block" >&2
  exit 1
fi

arm_split_stats="$("${COMPILER}" "${CASE_DIR}/hot_loop_no_call_split.sy" -S \
  -o "${OUT_DIR}/hot_loop_no_call_split.arm.s" \
  -O0 --target=arm --verify-ir --stats 2>&1 >/dev/null)"

echo "${arm_split_stats}"

if ! grep -A5 '^arm-regalloc:$' <<<"${arm_split_stats}" | grep -Eq 'live-range-splits : ([3-9]|[1-9][0-9]+)'; then
  echo "expected ARM regalloc to split hot live-in scalar values" >&2
  exit 1
fi

call_split_stats="$("${COMPILER}" "${CASE_DIR}/live_range_split_hot_loop.sy" -S \
  -o "${OUT_DIR}/live_range_split_hot_loop.rv.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"
call_nosplit_stats="$(SISY_RV_ENABLE_LIVE_RANGE_SPLIT=0 "${COMPILER}" "${CASE_DIR}/live_range_split_hot_loop.sy" -S \
  -o "${OUT_DIR}/live_range_split_hot_loop.rv.nosplit.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"

echo "${call_split_stats}"
echo "${call_nosplit_stats}"

call_split_spills="$(extract_pass_stat "${call_split_stats}" "rv-regalloc" "spilled")"
call_nosplit_spills="$(extract_pass_stat "${call_nosplit_stats}" "rv-regalloc" "spilled")"
if [[ -z "${call_split_spills}" || -z "${call_nosplit_spills}" ]]; then
  echo "failed to read RV regalloc spill stats for call-bearing split test" >&2
  exit 1
fi
if (( call_split_spills > call_nosplit_spills )); then
  echo "expected RV live-range splitting not to increase spill count in call-bearing hot loops" >&2
  exit 1
fi

echo "Register-allocation loop hotness tests passed."
