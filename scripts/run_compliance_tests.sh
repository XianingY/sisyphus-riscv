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

if rg -n 'inputFile\.find|emitKnown.*Fixup|fixup_output|SISY_ENABLE_.*FIXUPS' \
    "${ROOT_DIR}/src" >/tmp/sisy-compliance-source.txt; then
  cat /tmp/sisy-compliance-source.txt >&2
  echo "source-name or fixed-output optimization trigger found under src/" >&2
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

o1_stats="$("${COMPILER}" "${CASE}" -S -o "${OUT_DIR}/basic.o1.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${o1_stats}"

require_present "${o1_stats}" "\\[hir-poly\\]" \
  "expected HIR polyhedral optimizer stats in reference-compliant default O1"
require_present "${o1_stats}" "^runtime-memoize:" \
  "expected runtime memoization pass to remain in reference-compliant default O1"
require_absent "${o1_stats}" "^function-equivalence:" \
  "semantic function equivalence replacement must be opt-in under reference-compliant defaults"
require_absent "${o1_stats}" "^structural-bitwise:" \
  "structural bitwise recognizer must be opt-in under reference-compliant defaults"
require_absent "${o1_stats}" "^structural-modmul:" \
  "structural modular multiplication recognizer must be opt-in under reference-compliant defaults"
require_absent "${o1_stats}" "^row-scratch-matmul:" \
  "row-scratch matrix helper replacement must be opt-in under reference-compliant defaults"
require_absent "${o1_stats}" "^cached:" \
  "compile-time recursive precompute cache must be opt-in under reference-compliant defaults"

synth_positive="$("${COMPILER}" "${ROOT_DIR}/tests/compliance/synth_const_array_positive.sy" \
  -S -o "${OUT_DIR}/synth_const_array_positive.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/compliance/synth_const_array_positive.out" \
  2>&1 >/dev/null)"
echo "${synth_positive}"
require_stat "${synth_positive}" "synth-const-array" "arrays-synthesized : [1-9]" \
  "expected proven source-constant arrays to synthesize under reference-compliant O1"
require_stat "${synth_positive}" "synth-const-array" "loads-replaced : [1-9]" \
  "expected proven source-constant array loads to be replaced under reference-compliant O1"

synth_dynamic="$("${COMPILER}" "${ROOT_DIR}/tests/compliance/synth_const_array_dynamic_init.sy" \
  -S -o "${OUT_DIR}/synth_const_array_dynamic_init.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/compliance/synth_const_array_dynamic_init.out" \
  2>&1 >/dev/null)"
echo "${synth_dynamic}"
require_stat "${synth_dynamic}" "synth-const-array" "source-init-tables : [1-9]" \
  "expected deterministic source-initialized tables to be accepted"
require_stat "${synth_dynamic}" "synth-const-array" "loads-replaced : [1-9]" \
  "expected source-initialized table loads to be replaced"

synth_dynamic_mod="$("${COMPILER}" "${ROOT_DIR}/tests/compliance/synth_const_array_dynamic_mod.sy" \
  -S -o "${OUT_DIR}/synth_const_array_dynamic_mod.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/compliance/synth_const_array_dynamic_mod.out" \
  2>&1 >/dev/null)"
echo "${synth_dynamic_mod}"
require_stat "${synth_dynamic_mod}" "synth-const-array" "formula-mod : [1-9]" \
  "expected small modulo source-initialized table formulas to be recognized"
require_stat "${synth_dynamic_mod}" "synth-const-array" "loads-replaced : [1-9]" \
  "expected modulo source-initialized table loads to be replaced"

synth_mutable="$("${COMPILER}" "${ROOT_DIR}/tests/compliance/synth_const_array_mutable_negative.sy" \
  -S -o "${OUT_DIR}/synth_const_array_mutable_negative.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/compliance/synth_const_array_mutable_negative.out" \
  2>&1 >/dev/null)"
echo "${synth_mutable}"
require_stat "${synth_mutable}" "synth-const-array" "loads-replaced : 0" \
  "mutable or escaping arrays must not be synthesized"
require_stat "${synth_mutable}" "synth-const-array" "reject-mutable : [1-9]" \
  "expected mutable global-array rejection to be reported"

synth_dynamic_mutable="$("${COMPILER}" "${ROOT_DIR}/tests/compliance/synth_const_array_dynamic_mutable_negative.sy" \
  -S -o "${OUT_DIR}/synth_const_array_dynamic_mutable_negative.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/compliance/synth_const_array_dynamic_mutable_negative.out" \
  2>&1 >/dev/null)"
echo "${synth_dynamic_mutable}"
require_stat "${synth_dynamic_mutable}" "synth-const-array" "loads-replaced : 0" \
  "source-initialized arrays mutated after initialization must not be synthesized"
require_stat "${synth_dynamic_mutable}" "synth-const-array" "reject-mutable : [1-9]" \
  "expected post-init mutation rejection to be reported"

synth_nonformula="$("${COMPILER}" "${ROOT_DIR}/tests/compliance/synth_const_array_nonformula_negative.sy" \
  -S -o "${OUT_DIR}/synth_const_array_nonformula_negative.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/compliance/synth_const_array_nonformula_negative.out" \
  2>&1 >/dev/null)"
echo "${synth_nonformula}"
require_stat "${synth_nonformula}" "synth-const-array" "loads-replaced : 0" \
  "arrays outside the formula DSL must not be synthesized"
require_stat "${synth_nonformula}" "synth-const-array" "reject-no-formula : [1-9]" \
  "expected no-formula global-array rejection to be reported"

final_iter_accum="$("${COMPILER}" "${ROOT_DIR}/tests/compliance/final_iteration_memory_accum_negative.sy" \
  -S -o "${OUT_DIR}/final_iteration_memory_accum_negative.rv.s" \
  -O1 --target=riscv --verify-ir --stats \
  --compare "${ROOT_DIR}/tests/compliance/final_iteration_memory_accum_negative.out" \
  -i "${ROOT_DIR}/tests/compliance/final_iteration_memory_accum_negative.in" \
  2>&1 >/dev/null)"
echo "${final_iter_accum}"
require_stat "${final_iter_accum}" "repeat-invariant-reduction" "final-iteration-collapsed : 0" \
  "memory accumulation loops must not be collapsed to the final iteration"

o2_stats="$("${COMPILER}" "${CASE}" -S -o "${OUT_DIR}/basic.o2.rv.s" \
  -O2 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${o2_stats}"

require_absent "${o2_stats}" "^function-equivalence:" \
  "semantic function equivalence replacement must be opt-in under reference-compliant default O2"
require_absent "${o2_stats}" "^cached:" \
  "compile-time recursive precompute cache must be opt-in under reference-compliant defaults"
require_absent "${o2_stats}" "^synth-const-array:" \
  "O2 heavy synthesized constant-array path remains opt-in outside the O1 target profile"

check_no_fixed_output() {
  local src="$1"
  local asm="$2"
  local label="$3"
  if [[ ! -f "${src}" ]]; then
    echo "missing compliance probe source: ${src}" >&2
    exit 1
  fi
  "${COMPILER}" "${src}" -S -o "${asm}" -O1 --target=riscv --verify-ir \
    >/dev/null 2>"${asm}.log"
  if grep -Eq '\.L(functional|matrix|scheduling)_fixup_output|call[[:space:]]+printf' "${asm}"; then
    echo "synthetic fixed-output helper emitted for ${label}" >&2
    exit 1
  fi
}

check_no_fixed_output \
  "${ROOT_DIR}/test2026/riscv_func/h_functional/29_long_line.sy" \
  "${OUT_DIR}/29_long_line.rv.s" \
  "29_long_line"
check_no_fixed_output \
  "${ROOT_DIR}/test2026/performance_riscv/matmul1.sy" \
  "${OUT_DIR}/matmul1.rv.s" \
  "matmul1"
check_no_fixed_output \
  "${ROOT_DIR}/test2026/performance_riscv/optimization_scheduling1.sy" \
  "${OUT_DIR}/optimization_scheduling1.rv.s" \
  "optimization_scheduling1"

echo "Reference-compliant default-pass tests passed."
