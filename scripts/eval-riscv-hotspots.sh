#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_RUNTIME="${ROOT_DIR}/scripts/eval-runtime.sh"
TARGET="riscv"
OPT="${1:-O1}"
PERF_TIMEOUT_SEC="${RUNTIME_PERF_TIMEOUT_SEC:-60}"
LABEL="${RUNTIME_LABEL:-sisyphus}"
OUT_DIR="${ROOT_DIR}/tests/.out/riscv-hotspots"
TS="$(date +%Y%m%d-%H%M%S)"
SUMMARY="${OUT_DIR}/summary-${OPT}-${TS}.txt"

mkdir -p "${OUT_DIR}"
: >"${SUMMARY}"

if [[ "${OPT}" != "O1" && "${OPT}" != "O2" ]]; then
  echo "error: opt must be O1|O2"
  exit 1
fi

if [[ -z "${SISY_OFFICIAL_SUITE_ROOT:-}" && ! -d "${ROOT_DIR}/test/official" && -d "${ROOT_DIR}/test2026/performance" ]]; then
  export SISY_OFFICIAL_SUITE_ROOT="${ROOT_DIR}/tests/.out/test2026-perf-suite-root"
  mkdir -p \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-functional" \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-arm-perf" \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf" \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-arm-final-perf" \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-final-perf"
  if [[ ! -e "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf/.sisy-source" ]]; then
    rm -rf "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf"
    cp -R "${ROOT_DIR}/test2026/performance" "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf"
    : >"${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf/.sisy-source"
  fi
fi

summarize_csv() {
  local csv="$1"
  awk -F, '
    NR > 1 {
      total++;
      if ($8 == "1") pass++;
      else fail++;
      if ($6 == "timeout") timeout++;
      if ($6 == "compile_fail") compile_fail++;
      if ($6 == "compile_crash") compile_crash++;
      if ($6 == "mismatch") mismatch++;
      median += ($9 == "" ? 0 : $9);
    }
    END {
      printf "total=%d pass=%d fail=%d timeout=%d mismatch=%d compile_fail=%d compile_crash=%d median_sum_ms=%.3f",
        total + 0, pass + 0, fail + 0, timeout + 0, mismatch + 0,
        compile_fail + 0, compile_crash + 0, median;
    }
  ' "${csv}"
}

asm_bytes() {
  local csv="$1"
  awk -F, 'NR > 1 { print $14 }' "${csv}" |
    while IFS= read -r asm; do
      if [[ -f "${asm}" ]]; then
        stat -c '%s' "${asm}" 2>/dev/null || stat -f '%z' "${asm}" 2>/dev/null || echo 0
      else
        echo 0
      fi
    done |
    awk '{ s += $1 } END { print s + 0 }'
}

run_group() {
  local suite="$1"
  local filter="$2"
  local name="$3"
  local csv="${OUT_DIR}/${suite}-${OPT}-${name}-${TS}.csv"

  echo "[hotspot] ${suite} ${TARGET} ${OPT} filter=${filter}"
  RUNTIME_LABEL="${LABEL}" \
  RUNTIME_SOFT_PERF=1 \
  RUNTIME_PERF_TIMEOUT_SEC="${PERF_TIMEOUT_SEC}" \
  RUNTIME_CASE_FILTER="${filter}" \
  RUNTIME_CSV="${csv}" \
    "${EVAL_RUNTIME}" "${suite}" "${TARGET}" "${OPT}" >/dev/null

  {
    printf "%s/%s: " "${suite}" "${name}"
    summarize_csv "${csv}"
    printf " asm_bytes=%s csv=%s\n" "$(asm_bytes "${csv}")" "${csv}"
    awk -F, 'NR > 1 { printf "  %s status=%s pass=%s median=%s asm=%s\n", $2, $6, $8, $9, $14 }' "${csv}"
  } | tee -a "${SUMMARY}"
}

filters=(
  "crypto"
  "03_sort"
  "crc"
  "fft"
  "huffman"
  "conv2d"
  "many_mat_cal"
  "matmul"
  "transpose"
  "sl"
)

for filter in "${filters[@]}"; do
  run_group "official-riscv-perf" "${filter}" "${filter}"
done

if [[ "${HOTSPOT_INCLUDE_FINAL:-0}" == "1" ]]; then
  for filter in "${filters[@]}"; do
    run_group "official-riscv-final-perf" "${filter}" "final-${filter}"
  done
fi

echo "summary: ${SUMMARY}"
