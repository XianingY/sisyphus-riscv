#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
OUT_DIR="${ROOT_DIR}/tests/.out/deep-opt"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run cmake --build build -j first"
  exit 1
fi

require_stat() {
  local stats="$1"
  local pass="$2"
  local pattern="$3"
  local message="$4"
  if ! grep -A12 "^${pass}:$" <<<"${stats}" | grep -Eq "${pattern}"; then
    echo "${message}" >&2
    exit 1
  fi
}

regalloc_stats="$("${COMPILER}" "${ROOT_DIR}/tests/regalloc/live_range_split_hot_loop.sy" -S \
  -o "${OUT_DIR}/live_range_split_hot_loop.rv.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"
echo "${regalloc_stats}"
require_stat "${regalloc_stats}" "rv-regalloc" "live-range-splits : [1-9]" \
  "expected RV RegAlloc to split live ranges at hot loop entries"

schedule_stats="$("${COMPILER}" "${ROOT_DIR}/tests/schedule/critical_path_priority.sy" -S \
  -o "${OUT_DIR}/critical_path_priority.rv.s" \
  -O0 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"
echo "${schedule_stats}"
require_stat "${schedule_stats}" "rv-schedule" "critical-path-nodes : [1-9]" \
  "expected RV scheduler to compute critical-path heights"
require_stat "${schedule_stats}" "rv-schedule" "critical-path-max-height : ([4-9]|[1-9][0-9]+)" \
  "expected RV scheduler to expose a non-trivial critical path"

scalar_stats="$("${COMPILER}" "${ROOT_DIR}/tests/scalar_replace/local_array_scalarize.sy" -S \
  -o "${OUT_DIR}/local_array_scalarize.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"
echo "${scalar_stats}"
require_stat "${scalar_stats}" "scalar-replace" "arrays-scalarized : [1-9]" \
  "expected ScalarReplace to scalarize a non-escaping local array"
require_stat "${scalar_stats}" "scalar-replace" "array-accesses : [1-9]" \
  "expected ScalarReplace to rewrite local array element accesses"

sccp_stats="$("${COMPILER}" "${ROOT_DIR}/tests/sccp/phi_branch_constant.sy" -S \
  -o "${OUT_DIR}/phi_branch_constant.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"
echo "${sccp_stats}"
require_stat "${sccp_stats}" "sccp" "replaced-values : [1-9]" \
  "expected SCCP to replace constants discovered through SSA phis"
require_stat "${sccp_stats}" "sccp" "folded-branches : [1-9]" \
  "expected SCCP to fold a branch controlled by a discovered constant"

SISY_COMPARE_STEP_LIMIT=2000000000 "${COMPILER}" \
  "${ROOT_DIR}/tests/regression/nussinov_o1_signed_mod.sy" -S \
  -o "${OUT_DIR}/nussinov_o1_signed_mod.rv.s" \
  -O1 --target=riscv \
  --compare "${ROOT_DIR}/tests/regression/nussinov_o1_signed_mod.out" \
  -i "${ROOT_DIR}/tests/regression/nussinov_o1_signed_mod.in"

"${ROOT_DIR}/scripts/run_polyhedral_tests.sh" >/tmp/sisyphus-polyhedral-deep.log 2>&1
cat /tmp/sisyphus-polyhedral-deep.log

echo "Deep optimization tests passed."
