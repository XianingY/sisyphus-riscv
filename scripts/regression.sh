#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <case_dir> [riscv|arm] [O0|O1|O2] [extra compiler args...]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="$(python3 - "$1" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).resolve())
PY
)"
TARGET="${2:-riscv}"
OPT="${3:-O1}"
EXTRA_ARGS=()
if [[ $# -gt 3 ]]; then
  EXTRA_ARGS=("${@:4}")
fi
if [[ -n "${SISY_COMPILER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(${SISY_COMPILER_EXTRA_ARGS})
fi
TAG="${OUT_TAG:-}"
OUT_DIR="${ROOT_DIR}/tests/.out/${TARGET}-${OPT}"
if [[ -n "${TAG}" ]]; then
  OUT_DIR="${OUT_DIR}-${TAG}"
fi

mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first"
  exit 1
fi

if [[ ! -d "${CASE_DIR}" ]]; then
  echo "case directory not found: ${CASE_DIR}"
  exit 1
fi

safe_stem() {
  local rel="$1"
  rel="${rel//\//__}"
  rel="${rel// /_}"
  rel="${rel//$'\t'/_}"
  printf "%s" "${rel%.*}"
}

count=0
while IFS= read -r -d '' f; do
  rel="${f#${CASE_DIR}/}"
  stem="$(safe_stem "${rel}")"
  cmd=("${COMPILER}" "${f}" -S -o "${OUT_DIR}/${stem}.s" "--target=${TARGET}" "-${OPT}")
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi
  if ! "${cmd[@]}"; then
    echo "[fail] ${rel}"
    exit 1
  fi
  count=$((count + 1))
done < <(find "${CASE_DIR}" -type f \( -name "*.sy" -o -name "*.c" \) -print0 | sort -z)

echo "Compiled ${count} cases to ${OUT_DIR}"
