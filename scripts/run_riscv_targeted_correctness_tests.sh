#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
OUT_DIR="${ROOT_DIR}/tests/.out/riscv-targeted-correctness"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

require_stat() {
  local haystack="$1"
  local pass="$2"
  local pattern="$3"
  local message="$4"
  if ! grep -A16 "^${pass}:$" <<<"${haystack}" | grep -Eq "${pattern}"; then
    echo "${message}" >&2
    exit 1
  fi
}

select_positive="$("${COMPILER}" "${ROOT_DIR}/tests/select/multi_phi_pure_branch.sy" \
  -S -o "${OUT_DIR}/multi_phi_pure_branch.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/select/multi_phi_pure_branch.out" \
  -i "${ROOT_DIR}/tests/select/multi_phi_pure_branch.in" \
  2>&1 >/dev/null)"
echo "${select_positive}" >"${OUT_DIR}/multi_phi_pure_branch.stats"
grep -A4 '^select:$' "${OUT_DIR}/multi_phi_pure_branch.stats" || true
require_stat "${select_positive}" "select" "raised-selects : [2-9]" \
  "expected pure short branch with multiple i32 phis to become selects"

select_negative="$("${COMPILER}" "${ROOT_DIR}/tests/select/load_branch_negative.sy" \
  -S -o "${OUT_DIR}/load_branch_negative.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/select/load_branch_negative.out" \
  -i "${ROOT_DIR}/tests/select/load_branch_negative.in" \
  2>&1 >/dev/null)"
echo "${select_negative}" >"${OUT_DIR}/load_branch_negative.stats"
require_stat "${select_negative}" "select" "raised-selects : 0" \
  "load-containing branch must not be speculated into selects"

perf_cases=(
  matmul1 matmul2 matmul3
  many_mat_cal-1 many_mat_cal-2 many_mat_cal-3
  optimization_scheduling1 optimization_scheduling2 optimization_scheduling3
  01_mm1 01_mm2 01_mm3
  conv2d-1 conv2d-2 conv2d-3
  crc1 crc2 crc3
  huffman-01 huffman-02 huffman-03
  transpose0 transpose1 transpose2
  shuffle0 shuffle1 shuffle2
  sl1 sl2 sl3
)

for case in "${perf_cases[@]}"; do
  src="${ROOT_DIR}/test2026/performance_riscv/${case}.sy"
  [[ -f "${src}" ]] || continue
  asm="${OUT_DIR}/${case}.rv.s"
  stats="${OUT_DIR}/${case}.stats"
  echo "[targeted] compile ${case}"
  "${COMPILER}" "${src}" -S -o "${asm}" -O1 --target=riscv --verify-ir --stats \
    >"${stats}" 2>&1
  if grep -Eq '\bvsetvli\b|\bv[ls]e[0-9]+\.v\b|\bv[ls]se[0-9]+\.v\b' "${asm}"; then
    echo "default RISC-V O1 emitted RVV for ${case}" >&2
    exit 1
  fi
  if grep -Eq '\.L(functional|matrix|scheduling)_fixup_output|call[[:space:]]+printf' "${asm}"; then
    echo "default RISC-V O1 emitted fixed-output helper for ${case}" >&2
    exit 1
  fi
done

if [[ "${SISY_TARGETED_COMPARE_PERF:-0}" == "1" ]]; then
  for case in \
    crc1 conv2d-1 transpose0 sl1 \
    many_mat_cal-1 many_mat_cal-2 many_mat_cal-3 \
    optimization_scheduling1 optimization_scheduling2 optimization_scheduling3; do
    src="${ROOT_DIR}/test2026/performance_riscv/${case}.sy"
    out="${ROOT_DIR}/test2026/performance_riscv/${case}.out"
    in="${ROOT_DIR}/test2026/performance_riscv/${case}.in"
    [[ -f "${src}" && -f "${out}" ]] || continue
    echo "[targeted] compare ${case}"
    cmd=("${COMPILER}" "${src}" -S -o "${OUT_DIR}/${case}.compare.rv.s" \
      -O1 --target=riscv --verify-ir --compare "${out}")
    [[ -f "${in}" ]] && cmd+=(-i "${in}")
    "${cmd[@]}" >"${OUT_DIR}/${case}.compare.log" 2>&1 || {
      tail -120 "${OUT_DIR}/${case}.compare.log" >&2
      exit 1
    }
  done
fi

echo "RISC-V targeted correctness checks passed."
