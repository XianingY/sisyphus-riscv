#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${SISY_COMPILER_PATH:-${ROOT_DIR}/build/compiler}"
CASE="${ROOT_DIR}/tests/smoke/basic.sy"
OUT_DIR="${ROOT_DIR}/tests/.out/compliance"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

FORBIDDEN_SOURCE_TRIGGER_RE='inputFile\.find|path\.find|basename\(|caseName|case_name|sourcePath|emitKnown.*Fixup|fixup_output|SISY_ENABLE_.*FIXUPS'
FORBIDDEN_SOURCE_OUT="${OUT_DIR}/forbidden-source-triggers.txt"
if command -v rg >/dev/null 2>&1; then
  scan_forbidden_source() {
    rg -n "${FORBIDDEN_SOURCE_TRIGGER_RE}" "${ROOT_DIR}/src"
  }
else
  scan_forbidden_source() {
    grep -R -n -E "${FORBIDDEN_SOURCE_TRIGGER_RE}" "${ROOT_DIR}/src"
  }
fi

if scan_forbidden_source >"${FORBIDDEN_SOURCE_OUT}"; then
  cat "${FORBIDDEN_SOURCE_OUT}" >&2
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

require_absent_file() {
  local file="$1"
  local pattern="$2"
  local message="$3"
  if grep -Eq "${pattern}" "${file}"; then
    echo "${message}" >&2
    exit 1
  fi
}

STRICT_NATIVE_KERNEL_FIELDS=(
  semantic-kernels
  triangular-transpose-kernels
  modular-multiply-kernels
  modular-multiply-callsites
  modular-power-kernels
  modular-power-callsites
  memcopy-kernels
  memcopy-callsites
  digit-helper-kernels
  mm-like-kernels
  conv2d-interior-kernels
  structural-half-init-matrix-kernels
  structural-stencil3d-kernels
  structural-matmul-summary-kernels
  structural-digest-kernels
  structural-map-reduce-kernels
  structural-ludcmp-kernels
  structural-nussinov-kernels
  structural-trsm-kernels
  structural-hash-aggregate-kernels
)

require_stat_zero() {
  local stats="$1"
  local field="$2"
  local line
  line="$(grep '^\[native-asm\]' <<<"${stats}" | tail -1 || true)"
  local value
  value="$(sed -n "s/.* ${field}=\\([^ ]*\\).*/\\1/p" <<<"${line}")"
  if [[ -z "${value}" ]]; then
    echo "default RISC-V path did not report native kernel guard field ${field}" >&2
    exit 1
  fi
  if [[ "${value}" != "0" ]]; then
    echo "default RISC-V path emitted native semantic/structural kernel ${field}=${value}" >&2
    exit 1
  fi
}

require_strict_native_kernel_stats_zero() {
  local stats="$1"
  for field in "${STRICT_NATIVE_KERNEL_FIELDS[@]}"; do
    require_stat_zero "${stats}" "${field}"
  done
}

o1_stats="$("${COMPILER}" "${CASE}" -S -o "${OUT_DIR}/basic.o1.rv.s" \
  -O1 --target=riscv --verify-ir --stats 2>&1 >/dev/null)"
echo "${o1_stats}"

require_present "${o1_stats}" "\\[self-mlir\\].*source=ast.*frontend_path=self-mlir.*failed=0" \
  "expected default O1 production path to use AST-direct self-MLIR"
require_strict_native_kernel_stats_zero "${o1_stats}"
require_absent_file "${OUT_DIR}/basic.o1.rv.s" '\bvsetvli\b|\bv[ls]e[0-9]+\.v\b|\bv[ls]se[0-9]+\.v\b' \
  "default RISC-V path must not emit RVV"
require_absent_file "${OUT_DIR}/basic.o1.rv.s" '\.L(functional|matrix|scheduling)_fixup_output|call[[:space:]]+printf' \
  "default RISC-V path must not emit fixed-output helpers"

native_tests="$("${COMPILER}" --run-self-mlir-native-backend-tests 2>&1)"
echo "${native_tests}"
require_present "${native_tests}" "legacy-free=1" \
  "self-MLIR native backend must reject legacy/Phi operations"
require_present "${native_tests}" "rv-emitted=1" \
  "self-MLIR native backend RISC-V smoke must emit assembly"
require_present "${native_tests}" "arm-emitted=1" \
  "self-MLIR native backend ARM smoke must emit assembly"

perf_cases=(crc1 01_mm1 conv2d-1 h-10-01)
for name in "${perf_cases[@]}"; do
  src="${ROOT_DIR}/test2026/performance_riscv/${name}.sy"
  asm="${OUT_DIR}/${name}.rv.s"
  stats="$("${COMPILER}" "${src}" -S -o "${asm}" -O1 --target=riscv --verify-ir --stats \
    2>&1 >/dev/null)"
  echo "${stats}"
  require_present "${stats}" "\\[self-mlir\\].*source=ast.*failed=0" \
    "expected ${name} to compile through self-MLIR"
  require_strict_native_kernel_stats_zero "${stats}"
  require_absent_file "${asm}" '\bvsetvli\b|\bv[ls]e[0-9]+\.v\b|\bv[ls]se[0-9]+\.v\b' \
    "default RISC-V path emitted RVV for ${name}"
done

"${ROOT_DIR}/scripts/run_proven_bitwise_tests.sh"

echo "Self-MLIR reference-compliance tests passed."
