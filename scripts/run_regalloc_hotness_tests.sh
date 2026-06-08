#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/regalloc"
OUT_DIR="${ROOT_DIR}/tests/.out/regalloc"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run cmake --build build -j first"
  exit 1
fi

field() {
  local line="$1"
  local key="$2"
  sed -n "s/.* ${key}=\\([^ ]*\\).*/\\1/p" <<<"${line}"
}

compile_stats() {
  local target="$1"
  local source="$2"
  local output="$3"
  shift 3
  "$@" "${COMPILER}" "${source}" -S -o "${output}" \
    -O0 --target="${target}" --verify-ir --stats 2>&1 >/dev/null
}

stats="$(compile_stats riscv "${CASE_DIR}/nested_loop_hotness.sy" \
  "${OUT_DIR}/nested_loop_hotness.rv.s" \
  env SISY_ENABLE_SELF_MACHINE_LIVENESS=1)"

self_line="$(grep '^\[self-mlir\]' <<<"${stats}" | tail -1 || true)"
native_line="$(grep '^\[native-asm\]' <<<"${stats}" | tail -1 || true)"
echo "${self_line}"
echo "${native_line}"
if ! grep -Eq '\[self-mlir\].*frontend_path=self-mlir.*failed=0' <<<"${self_line}" ||
   ! grep -Eq '\[native-asm\].*emitted=1.*legacy-ops=0.*phi-like-ops=0' <<<"${native_line}"; then
  echo "expected self-MLIR native RISC-V backend stats" >&2
  exit 1
fi

dead_avoided="$(field "${native_line}" dead-spills-avoided)"
live_spills="$(field "${native_line}" live-spills)"
lsra_avoided="$(field "${native_line}" lsra-spills-avoided)"
if [[ -z "${lsra_avoided}" || "${lsra_avoided}" -lt 1 ]]; then
  echo "expected self-MLIR machine liveness/LSRA to avoid at least one home spill" >&2
  exit 1
fi
if [[ -z "${live_spills}" || "${live_spills}" -ne 0 ]]; then
  echo "expected self-MLIR native backend liveness to eliminate live home spills in hot loop" >&2
  exit 1
fi

disabled_stats="$(compile_stats riscv "${CASE_DIR}/nested_loop_hotness.sy" \
  "${OUT_DIR}/nested_loop_hotness.rv.no-liveness.s" \
  env SISY_ENABLE_SELF_MACHINE_LIVENESS=0)"
disabled_native="$(grep '^\[native-asm\]' <<<"${disabled_stats}" | tail -1 || true)"
echo "${disabled_native}"
disabled_dead="$(field "${disabled_native}" dead-spills-avoided)"
disabled_live="$(field "${disabled_native}" live-spills)"
disabled_lsra_avoided="$(field "${disabled_native}" lsra-spills-avoided)"
if [[ -z "${disabled_dead}" || "${disabled_dead}" -ne 0 ]]; then
  echo "expected SISY_ENABLE_SELF_MACHINE_LIVENESS=0 to disable dead-spill avoidance stats" >&2
  exit 1
fi
if [[ -z "${disabled_lsra_avoided}" || "${disabled_lsra_avoided}" -ne 0 ||
      -z "${disabled_live}" || "${disabled_live}" -le "${live_spills}" ]]; then
  echo "expected SISY_ENABLE_SELF_MACHINE_LIVENESS=0 to lose LSRA spill avoidance" >&2
  exit 1
fi

arm_stats="$(compile_stats arm "${CASE_DIR}/hot_loop_no_call_split.sy" \
  "${OUT_DIR}/hot_loop_no_call_split.arm.s" env)"
arm_native="$(grep '^\[native-asm\]' <<<"${arm_stats}" | tail -1 || true)"
echo "${arm_native}"
if ! grep -Eq '\[native-asm\].*emitted=1.*legacy-ops=0.*phi-like-ops=0' <<<"${arm_native}"; then
  echo "expected self-MLIR native ARM backend stats" >&2
  exit 1
fi

echo "Self-MLIR machine liveness/regalloc hotness tests passed."
