#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_DIR="${SISY_OFFICIAL_SUITE_ROOT:-${ROOT_DIR}/test/official}"
OUT_DIR="${ROOT_DIR}/tests/.out/suites"
OUT_CSV="${OUT_DIR}/index.csv"

if [[ $# -ne 0 ]]; then
  echo "usage: $0"
  exit 1
fi

if [[ ! -d "${OFFICIAL_DIR}" ]]; then
  echo "missing ${OFFICIAL_DIR}; run scripts/suite-sync.sh first"
  exit 1
fi

mkdir -p "${OUT_DIR}"
printf 'suite,tier,kind,case_id,src,in,out,enabled\n' >"${OUT_CSV}"

TMP_FILES=()
cleanup() {
  if [[ "${#TMP_FILES[@]}" -gt 0 ]]; then
    rm -f "${TMP_FILES[@]}"
  fi
}
trap cleanup EXIT

emit_suite_cases() {
  local suite="$1"
  local tier="$2"
  local kind="$3"
  local root="${OFFICIAL_DIR}/${suite}"
  local enabled="1"
  local list_file root_abs

  if [[ ! -d "${root}" ]]; then
    echo "error: missing suite dir '${root}'. run scripts/suite-sync.sh first."
    exit 1
  fi

  root_abs="$(cd "${root}" && pwd -P)"
  list_file="$(mktemp "${TMPDIR:-/tmp}/sisy-suite-index.XXXXXX")"
  TMP_FILES+=("${list_file}")
  find "${root_abs}" -type f -name '*.sy' -print | LC_ALL=C sort >"${list_file}"

  while IFS= read -r src; do
    local rel case_id in_file out_file
    [[ -n "${src}" ]] || continue
    rel="${src#${root_abs}/}"
    case_id="${rel%.*}"
    in_file="${src%.*}.in"
    out_file="${src%.*}.out"

    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${suite}" "${tier}" "${kind}" "${case_id}" \
      "${src}" "${in_file}" "${out_file}" "${enabled}" \
      >>"${OUT_CSV}"
  done <"${list_file}"
}

emit_suite_cases "official-functional" "hard" "functional"
emit_suite_cases "official-arm-perf" "soft" "perf"
emit_suite_cases "official-riscv-perf" "soft" "perf"
emit_suite_cases "official-arm-final-perf" "soft" "perf"
emit_suite_cases "official-riscv-final-perf" "soft" "perf"

count="$(awk 'END{print NR-1}' "${OUT_CSV}")"
echo "wrote ${OUT_CSV} (${count} rows)"
