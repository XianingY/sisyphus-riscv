#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE="${ROOT_DIR}/tests/smoke/basic.sy"
OUT_DIR="${ROOT_DIR}/tests/.out/compliance"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

require_present() {
  local haystack="$1"
  local pattern="$2"
  local message="$3"
  if ! grep -Eq "${pattern}" <<<"${haystack}"; then
    echo "${message}" >&2
    exit 1
  fi
}

require_absent() {
  local haystack="$1"
  local pattern="$2"
  local message="$3"
  if grep -Eq "${pattern}" <<<"${haystack}"; then
    echo "${message}" >&2
    exit 1
  fi
}

o1_stats="$("${COMPILER}" "${CASE}" -S -o "${OUT_DIR}/basic.o1.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${o1_stats}"

require_present "${o1_stats}" "\\[hir-poly\\]" \
  "expected HIR polyhedral optimizer stats in strict default O1"
require_present "${o1_stats}" "^runtime-memoize:" \
  "expected runtime memoization pass to remain in strict default O1"
require_absent "${o1_stats}" "^structural-bitwise:" \
  "structural bitwise recognizer must be opt-in under strict defaults"
require_absent "${o1_stats}" "^structural-modmul:" \
  "structural modular multiplication recognizer must be opt-in under strict defaults"
require_absent "${o1_stats}" "^row-scratch-matmul:" \
  "row-scratch matrix helper replacement must be opt-in under strict defaults"

o2_stats="$("${COMPILER}" "${CASE}" -S -o "${OUT_DIR}/basic.o2.rv.s" \
  -O2 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${o2_stats}"

require_absent "${o2_stats}" "^cached:" \
  "compile-time recursive precompute cache must be opt-in under strict defaults"
require_absent "${o2_stats}" "^synth-const-array:" \
  "SMT synthesized constant arrays must be opt-in under strict defaults"

echo "Strict compliance default-pass tests passed."
