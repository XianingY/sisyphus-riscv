#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/smoke"
OUT_DIR="${ROOT_DIR}/tests/.out"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first"
  exit 1
fi

cases=()
while IFS= read -r rel_case; do
  cases+=("${rel_case}")
done < <(git -C "${ROOT_DIR}" ls-files 'tests/smoke/*.sy')
if [[ ${#cases[@]} -eq 0 ]]; then
  echo "no smoke tests found under ${CASE_DIR}"
  exit 1
fi

for rel_case in "${cases[@]}"; do
  case="${ROOT_DIR}/${rel_case}"
  base="$(basename "${case}" .sy)"
  echo "[smoke] ${base}"
  "${COMPILER}" "${case}" -S -o "${OUT_DIR}/${base}.rv.s" -O0 --target=riscv --verify-ir
  "${COMPILER}" "${case}" -S -o "${OUT_DIR}/${base}.arm.s" -O0 --target=arm --verify-ir
done

echo "Smoke compile succeeded for ARM + RISC-V."
