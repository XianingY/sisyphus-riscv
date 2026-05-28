#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
OUT_DIR="${ROOT_DIR}/tests/.out/mir_legalizer"
CASE="${ROOT_DIR}/tests/smoke/basic.sy"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

rv_stats="$(SISY_MIR_LEGALIZER_STRICT=1 "${COMPILER}" "${CASE}" -S \
  -o "${OUT_DIR}/basic.rv.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"
arm_stats="$(SISY_MIR_LEGALIZER_STRICT=1 "${COMPILER}" "${CASE}" -S \
  -o "${OUT_DIR}/basic.arm.s" \
  -O0 --target=arm --verify-ir --stats 2>&1 >/dev/null)"

echo "${rv_stats}"
echo "${arm_stats}"

if ! grep -A8 '^rv-legalize:$' <<<"${rv_stats}" | grep -q 'verifier-errors : 0'; then
  echo "expected strict RV MIR legalizer verifier to accept the smoke case" >&2
  exit 1
fi

if ! grep -A8 '^arm-legalize:$' <<<"${arm_stats}" | grep -q 'verifier-errors : 0'; then
  echo "expected strict ARM MIR legalizer verifier to accept the smoke case" >&2
  exit 1
fi

disabled_stats="$(SISY_ENABLE_MIR_LEGALIZER=0 "${COMPILER}" "${CASE}" -S \
  -o "${OUT_DIR}/basic.rv.no-legalizer.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"

if grep -q '^rv-legalize:$' <<<"${disabled_stats}"; then
  echo "expected SISY_ENABLE_MIR_LEGALIZER=0 to remove rv-legalize from the pipeline" >&2
  exit 1
fi

echo "MIR legalizer strict and kill-switch tests passed."
