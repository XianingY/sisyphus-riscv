#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${SISY_RISCV_PERF_CASE_DIR:-${ROOT_DIR}/test2026/performance_riscv}"
OUT_DIR="${ROOT_DIR}/tests/.out/riscv-perf-compile"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi
if [[ ! -d "${CASE_DIR}" ]]; then
  echo "RISC-V performance case dir not found: ${CASE_DIR}" >&2
  exit 1
fi

if rg -n 'inputFile\.find|emitKnown.*Fixup|fixup_output|SISY_ENABLE_.*FIXUPS' \
    "${ROOT_DIR}/src" >/tmp/sisy-riscv-guard-source.txt; then
  cat /tmp/sisy-riscv-guard-source.txt >&2
  echo "source-name or fixed-output optimization trigger found under src/" >&2
  exit 1
fi

shopt -s nullglob
cases=("${CASE_DIR}"/*.sy)
if [[ "${#cases[@]}" -eq 0 ]]; then
  echo "no performance cases found under ${CASE_DIR}" >&2
  exit 1
fi

for src in "${cases[@]}"; do
  name="$(basename "${src}" .sy)"
  asm="${OUT_DIR}/${name}.rv.s"
  stats="${OUT_DIR}/${name}.stats"
  echo "[riscv-guard] compile ${name}"
  "${COMPILER}" "${src}" -S -o "${asm}" -O1 --target=riscv --verify-ir --stats \
    >"${stats}" 2>&1

  if grep -Eq '\bvsetvli\b|\bv[ls]e[0-9]+\.v\b|\bv[ls]se[0-9]+\.v\b' "${asm}"; then
    echo "default RISC-V path emitted RVV instructions for ${name}; use --enable-rvv for RVV" >&2
    exit 1
  fi
  if grep -Eq '\.L(functional|matrix|scheduling)_fixup_output|call[[:space:]]+printf' "${asm}"; then
    echo "default RISC-V path emitted a synthetic fixed-output helper for ${name}" >&2
    exit 1
  fi

  for forbidden in \
    'cached' \
    'function-equivalence' \
    'structural-bitwise' \
    'structural-modmul'; do
    if grep -Eq "^${forbidden}:$" "${stats}" &&
       ! grep -A8 "^${forbidden}:$" "${stats}" | grep -Eq ': 0$|<no stats>'; then
      echo "forbidden default semantic replacement pass appears active: ${forbidden} on ${name}" >&2
      exit 1
    fi
  done
done

echo "RISC-V performance compile guardrails passed (${#cases[@]} cases)."
