#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/memoryssa"
OUT_DIR="${ROOT_DIR}/tests/.out/memoryssa"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

field() {
  local line="$1"
  local key="$2"
  sed -n "s/.* ${key}=\\([^ ]*\\).*/\\1/p" <<<"${line}"
}

check_case() {
  local name="$1"
  local min_forwarded="$2"
  local asm="${OUT_DIR}/${name}.rv.s"
  local stats="${OUT_DIR}/${name}.stats"

  "${COMPILER}" "${CASE_DIR}/${name}.sy" -S -o "${asm}" \
    -O1 --target=riscv --verify-ir --stats >"${stats}" 2>&1
  cat "${stats}"

  local self_line native_line forwarded failed
  self_line="$(grep '^\[self-mlir\]' "${stats}" | tail -1 || true)"
  native_line="$(grep '^\[native-asm\]' "${stats}" | tail -1 || true)"
  forwarded="$(field "${self_line}" mem-forwarded-loads)"
  failed="$(field "${self_line}" conversion-failed)"

  if [[ -z "${self_line}" || -z "${native_line}" ]]; then
    echo "expected self-MLIR/native-asm stats for ${name}" >&2
    exit 1
  fi
  if [[ "${failed:-1}" != "0" ]]; then
    echo "self-MLIR dialect conversion failed for ${name}" >&2
    exit 1
  fi
  if [[ "${forwarded:-0}" -lt "${min_forwarded}" ]]; then
    echo "expected self-MLIR memref reaching-store forwarding for ${name}; saw ${forwarded:-0}" >&2
    exit 1
  fi
}

check_case join_same_store_load 1
check_case join_same_runtime_store_load 1
check_case join_different_store_phi_load 1
check_case branch_common_store_sink 1
check_case branch_different_store_phi_sink 1
check_case readonly_call_preserves_store 1

echo "self-MLIR memref reaching-store tests passed."
