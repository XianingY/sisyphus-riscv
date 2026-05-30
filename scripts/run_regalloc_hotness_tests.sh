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

rv_hotness="$(extract_pass_stat "${stats}" "rv-regalloc" "max-block-hotness")"
if [[ -z "${rv_hotness}" || "${rv_hotness}" -lt 64 ]]; then
  echo "expected RV regalloc to expose nested-loop hotness >= 64" >&2
  exit 1
fi

split_stats="$("${COMPILER}" "${CASE_DIR}/hot_loop_no_call_split.sy" -S \
  -o "${OUT_DIR}/hot_loop_no_call_split.rv.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"

echo "${split_stats}"

rv_splits="$(extract_pass_stat "${split_stats}" "rv-regalloc" "live-range-splits")"
if [[ -z "${rv_splits}" || "${rv_splits}" -lt 1 ]]; then
  echo "expected RV regalloc to split live-in values for hot loops without calls" >&2
  exit 1
fi

if [[ "${rv_splits}" -lt 3 ]]; then
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

split_spill_stats="$("${COMPILER}" "${CASE_DIR}/live_range_split_hot_loop.sy" -S \
  -o "${OUT_DIR}/live_range_split_hot_loop.rv.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"
nosplit_spill_stats="$(SISY_RV_ENABLE_LIVE_RANGE_SPLIT=0 "${COMPILER}" "${CASE_DIR}/live_range_split_hot_loop.sy" -S \
  -o "${OUT_DIR}/live_range_split_hot_loop.rv.nosplit.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"

echo "${split_spill_stats}"
echo "${nosplit_spill_stats}"

rv_spill_splits="$(extract_pass_stat "${split_spill_stats}" "rv-regalloc" "live-range-splits")"
rv_nosplit_splits="$(extract_pass_stat "${nosplit_spill_stats}" "rv-regalloc" "live-range-splits")"
if [[ -z "${rv_spill_splits}" || "${rv_spill_splits}" -lt 1 ]]; then
  echo "expected RV live-range splitting to run on hot live-in values" >&2
  exit 1
fi
if [[ -z "${rv_nosplit_splits}" || "${rv_nosplit_splits}" -ne 0 ]]; then
  echo "expected RV live-range splitting kill switch to disable splits" >&2
  exit 1
fi

echo "Register-allocation loop hotness tests passed."
