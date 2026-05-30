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

stats="$("${COMPILER}" "${CASE_DIR}/positive_loops.sy" -S \
  -o "${OUT_DIR}/positive_loops.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${CASE_DIR}/positive_loops.out" \
  2>&1 >/dev/null)"
printf '%s\n' "${stats}" >"${OUT_DIR}/positive_loops.stats"
grep -A6 '^loop-rotate:$' "${OUT_DIR}/positive_loops.stats" || true

if ! grep -A8 '^loop-rotate:$' <<<"${stats}" | grep -Eq 'rotated-loops : [1-9]'; then
  echo "expected default RISC-V O1 to rotate canonical while loops" >&2
  exit 1
fi

off_stats="$(SISY_ENABLE_LOOP_ROTATE=0 "${COMPILER}" "${CASE_DIR}/positive_loops.sy" -S \
  -o "${OUT_DIR}/positive_loops.no-rotate.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${CASE_DIR}/positive_loops.out" \
  2>&1 >/dev/null)"
printf '%s\n' "${off_stats}" >"${OUT_DIR}/positive_loops.no-rotate.stats"

if grep -q '^loop-rotate:$' <<<"${off_stats}"; then
  echo "SISY_ENABLE_LOOP_ROTATE=0 should disable the default loop-rotate pass" >&2
  exit 1
fi

"${COMPILER}" "${CASE_DIR}/zero_trip_call.sy" -S \
  -o "${OUT_DIR}/zero_trip_call.rv.s" \
  -O1 --target=riscv --verify-ir \
  --compare "${CASE_DIR}/zero_trip_call.out" \
  >"${OUT_DIR}/zero_trip_call.log" 2>&1 || {
    cat "${OUT_DIR}/zero_trip_call.log" >&2
    exit 1
  }

call_stats="$("${COMPILER}" "${CASE_DIR}/call_loop.sy" -S \
  -o "${OUT_DIR}/call_loop.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${CASE_DIR}/call_loop.out" \
  2>&1 >/dev/null)"
printf '%s\n' "${call_stats}" >"${OUT_DIR}/call_loop.stats"
grep -A6 '^loop-rotate:$' "${OUT_DIR}/call_loop.stats" || true

if ! grep -A8 '^loop-rotate:$' <<<"${call_stats}" | grep -Eq 'rotated-loops : [1-9]'; then
  echo "expected loop-rotate to allow loops with calls when no stores are present" >&2
  exit 1
fi

"${COMPILER}" "${CASE_DIR}/exit_phi.sy" -S \
  -o "${OUT_DIR}/exit_phi.rv.s" \
  -O1 --target=riscv --verify-ir \
  --compare "${CASE_DIR}/exit_phi.out" \
  -i "${CASE_DIR}/exit_phi.in" \
  >"${OUT_DIR}/exit_phi.log" 2>&1 || {
    cat "${OUT_DIR}/exit_phi.log" >&2
    exit 1
  }

"${COMPILER}" "${CASE_DIR}/conditional_memory.sy" -S \
  -o "${OUT_DIR}/conditional_memory.rv.s" \
  -O1 --target=riscv --verify-ir \
  --compare "${CASE_DIR}/conditional_memory.out" \
  >"${OUT_DIR}/conditional_memory.log" 2>&1 || {
    cat "${OUT_DIR}/conditional_memory.log" >&2
    exit 1
  }

if grep -Eq '\bvsetvli\b|\bvle[0-9]+\.v\b|\bvse[0-9]+\.v\b' \
    "${OUT_DIR}/positive_loops.rv.s"; then
  echo "loop-rotate tests unexpectedly emitted RVV instructions" >&2
  exit 1
fi

echo "RISC-V loop rotation tests passed."
