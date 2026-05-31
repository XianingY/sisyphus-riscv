#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${COMPILER:-${ROOT_DIR}/build/compiler}"
CASE_DIR="${ROOT_DIR}/tests/polyhedral"
OUT_DIR="${ROOT_DIR}/tests/.out/polyhedral"
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
  local min_affine="$2"
  local asm="${OUT_DIR}/${name}.rv.s"
  local stats="${OUT_DIR}/${name}.stats"

  "${COMPILER}" "${CASE_DIR}/${name}.sy" -S -o "${asm}" \
    -O1 --target=riscv --verify-ir --stats >"${stats}" 2>&1
  cat "${stats}"

  local self_line native_line affine failed
  self_line="$(grep '^\[self-mlir\]' "${stats}" | tail -1 || true)"
  native_line="$(grep '^\[native-asm\]' "${stats}" | tail -1 || true)"
  affine="$(field "${self_line}" affine-loops)"
  failed="$(field "${self_line}" conversion-failed)"

  if [[ -z "${self_line}" || -z "${native_line}" ]]; then
    echo "expected self-MLIR/native-asm stats for ${name}" >&2
    exit 1
  fi
  if [[ "${failed:-1}" != "0" ]]; then
    echo "self-MLIR dialect conversion failed for ${name}" >&2
    exit 1
  fi
  if [[ "${affine:-0}" -lt "${min_affine}" ]]; then
    echo "expected self-MLIR affine recovery for ${name}; saw ${affine:-0}" >&2
    exit 1
  fi
}

check_case fusion_disjoint_domains 2
check_case fusion_commuted_step 2
check_case reduction_unroll_jam 2
check_case interchange_3d_jk_safe 3
check_case interchange_3d_jk_cross_dim_disjoint 3
check_case interchange_3d_jk_cross_dim_overlap_safe 3
check_case interchange_3d_jk_unsafe 3
check_case triangular_partial_unroll 2
check_case invariant_guard_hoist 2

echo "self-MLIR affine/polyhedral recovery tests passed."
