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

LAST_SELF_LINE=""
LAST_NATIVE_LINE=""

compile_case() {
  local name="$1"
  shift
  local asm="${OUT_DIR}/${name}.rv.s"
  local stats="${OUT_DIR}/${name}.stats"

  env "$@" "${COMPILER}" "${CASE_DIR}/${name}.sy" -S -o "${asm}" \
    -O1 --target=riscv --verify-ir --stats >"${stats}" 2>&1
  cat "${stats}"

  LAST_SELF_LINE="$(grep '^\[self-mlir\]' "${stats}" | tail -1 || true)"
  LAST_NATIVE_LINE="$(grep '^\[native-asm\]' "${stats}" | tail -1 || true)"

  if [[ -z "${LAST_SELF_LINE}" || -z "${LAST_NATIVE_LINE}" ]]; then
    echo "expected self-MLIR/native-asm stats for ${name}" >&2
    exit 1
  fi
  local failed
  failed="$(field "${LAST_SELF_LINE}" conversion-failed)"
  if [[ "${failed:-1}" != "0" ]]; then
    echo "self-MLIR dialect conversion failed for ${name}" >&2
    exit 1
  fi
}

require_stat_at_least() {
  local name="$1"
  local key="$2"
  local min_value="$3"
  local value
  value="$(field "${LAST_SELF_LINE}" "${key}")"
  if [[ "${value:-0}" -lt "${min_value}" ]]; then
    echo "expected ${key} >= ${min_value} for ${name}; saw ${value:-0}" >&2
    exit 1
  fi
}

check_case() {
  local name="$1"
  local min_affine="$2"
  shift 2
  compile_case "${name}" "$@"

  local affine recovered observed
  affine="$(field "${LAST_SELF_LINE}" affine-loops)"
  recovered="$(field "${LAST_SELF_LINE}" affine-worklist-items)"
  observed="${affine:-0}"
  if [[ "${recovered:-0}" -gt "${observed}" ]]; then
    observed="${recovered}"
  fi
  if [[ "${observed:-0}" -lt "${min_affine}" ]]; then
    echo "expected self-MLIR affine recovery for ${name}; saw affine-loops=${affine:-0} affine-worklist-items=${recovered:-0}" >&2
    exit 1
  fi
}

check_poly_tile_probe() {
  local name="$1"
  shift
  compile_case "${name}" "$@"
  require_stat_at_least "${name}" poly-nests 1
  require_stat_at_least "${name}" poly-deps-proved 1
  require_stat_at_least "${name}" poly-tiles 1
}

check_poly_permutation_probe() {
  local name="$1"
  shift
  compile_case "${name}" "$@"
  require_stat_at_least "${name}" poly-nests 1
  require_stat_at_least "${name}" poly-deps-proved 1
  require_stat_at_least "${name}" poly-permutations 1
  require_stat_at_least "${name}" poly-tiles 1
}

check_pure_call_tile_probe() {
  local name="$1"
  shift
  compile_case "${name}" "$@"
  require_stat_at_least "${name}" calls 1
  require_stat_at_least "${name}" poly-tiles 1
}

check_imperfect_reduction_probe() {
  local name="$1"
  shift
  compile_case "${name}" "$@"
  require_stat_at_least "${name}" poly-tiles 1
  require_stat_at_least "${name}" poly-matrix-nests 1
  require_stat_at_least "${name}" poly-dep-directions 1
  require_stat_at_least "${name}" reduction-blocks 1
  require_stat_at_least "${name}" matrix-register-tiles 1
  require_stat_at_least "${name}" imperfect-nests 1
}

check_conditional_reduction_probe() {
  local name="$1"
  shift
  compile_case "${name}" "$@"
  require_stat_at_least "${name}" poly-matrix-nests 1
  require_stat_at_least "${name}" poly-dep-directions 1
  require_stat_at_least "${name}" poly-tiles 1
  require_stat_at_least "${name}" matrix-register-tiles 1
  require_stat_at_least "${name}" reduction-blocks 1
}

check_terminator_probe() {
  local name="$1"
  shift
  compile_case "${name}" "$@"
  require_stat_at_least "${name}" poly-tiles 1
  require_stat_at_least "${name}" poly-dep-directions 1
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
check_poly_tile_probe tile_3d_probe SISY_SELF_TILE_SIZE=4
check_poly_tile_probe tile_step2_probe
check_pure_call_tile_probe tile_pure_call_probe SISY_ENABLE_SELF_INLINE=0
check_poly_permutation_probe permute_3d_stride_probe SISY_SELF_TILE_SIZE=4
check_imperfect_reduction_probe tile_imperfect_reduction_probe SISY_SELF_TILE_SIZE=4
check_imperfect_reduction_probe tile_imperfect_direct_zero_probe SISY_SELF_TILE_SIZE=4
check_imperfect_reduction_probe matrix_symbolic_bound_probe SISY_SELF_TILE_SIZE=4
check_conditional_reduction_probe matrix_conditional_reduction_probe SISY_SELF_TILE_SIZE=4
check_terminator_probe matmul_verify_terminator_probe SISY_SELF_TILE_SIZE=4

echo "self-MLIR affine/polyhedral recovery tests passed."
