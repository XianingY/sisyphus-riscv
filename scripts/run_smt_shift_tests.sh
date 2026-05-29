#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${SISY_RISCV_PERF_CASE_DIR:-${ROOT_DIR}/test2026/performance_riscv}"
OUT_DIR="${ROOT_DIR}/tests/.out/smt_shift"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

for case in fft0 fft1 fft2; do
  src="${CASE_DIR}/${case}.sy"
  if [[ ! -f "${src}" ]]; then
    echo "missing ${src}" >&2
    exit 1
  fi

  stats="$("${COMPILER}" "${src}" -S \
    -o "${OUT_DIR}/${case}.rv.s" \
    -O1 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"
  echo "${stats}"

  if grep -q 'unknown op: (rsh' <<<"${stats}"; then
    echo "SMT bitvector solver rejected arithmetic right shift in ${case}" >&2
    exit 1
  fi
  if ! grep -A5 '^smt-synth:$' <<<"${stats}" | grep -q 'unknown :'; then
    echo "expected default SMT synthesis to run on ${case}" >&2
    exit 1
  fi
done

echo "SMT shift synthesis regression tests passed."
