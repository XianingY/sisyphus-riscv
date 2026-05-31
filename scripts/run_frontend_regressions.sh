#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/regression/frontend"
OUT_DIR="${ROOT_DIR}/tests/.out/frontend-regression"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first"
  exit 1
fi

run_compare_cases() {
  local target="$1"
  shopt -s nullglob
  local cases=("${CASE_DIR}"/*.sy)
  shopt -u nullglob
  if [[ "${#cases[@]}" -eq 0 ]]; then
    echo "[frontend-compare] ${target} no cases under ${CASE_DIR}; skipping"
    return
  fi
  for case in "${cases[@]}"; do
    local base out stem asm
    base="${case%.sy}"
    out="${base}.out"
    stem="$(basename "${base}")"
    asm="${OUT_DIR}/${stem}.${target}.s"
    echo "[frontend-compare] ${target} ${stem}"
    "${COMPILER}" "${case}" -S -o "${asm}" "--target=${target}" -O1 --compare "${out}"
  done
}

expect_missing_value_fail() {
  local optname="$1"
  local sample="${ROOT_DIR}/tests/smoke/basic.sy"
  local asm="${OUT_DIR}/missing-value.s"
  local log="${OUT_DIR}/missing-value-${optname//[^a-zA-Z0-9]/_}.log"

  set +e
  "${COMPILER}" "${sample}" -S -o "${asm}" --target=riscv -O1 "${optname}" >"${log}" 2>&1
  local rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    echo "[frontend-cli] expected failure for ${optname}, but succeeded"
    cat "${log}"
    exit 1
  fi
  if ! rg -q "requires value" "${log}"; then
    echo "[frontend-cli] expected 'requires value' in error output for ${optname}"
    cat "${log}"
    exit 1
  fi
  echo "[frontend-cli] ${optname} missing value -> guarded"
}

run_compare_cases riscv
run_compare_cases arm
expect_missing_value_fail "--print-after"
expect_missing_value_fail "--print-before"
expect_missing_value_fail "--compare"
expect_missing_value_fail "-i"

echo "frontend regressions passed."
