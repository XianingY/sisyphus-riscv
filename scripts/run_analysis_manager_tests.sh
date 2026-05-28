#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
OUT_DIR="${ROOT_DIR}/tests/.out/analysis_manager"
CASE="${ROOT_DIR}/tests/memoryssa/join_same_store_load.sy"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

stats="$(SISY_DUMP_ANALYSIS_CACHE=1 "${COMPILER}" "${CASE}" -S \
  -o "${OUT_DIR}/basic.default.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${stats}"

for fact in loop alias block-frequency domtree memoryssa; do
  if ! grep -q "\\[analysis-cache\\] ${fact}" <<<"${stats}"; then
    echo "expected analysis cache stats for ${fact}" >&2
    exit 1
  fi
done

if ! grep -Eq '\[analysis-cache\] domtree\.[0-9]+ builds=[1-9]' <<<"${stats}"; then
  echo "expected cached dominator tree build count" >&2
  exit 1
fi

if ! grep -Eq '\[analysis-cache\] memoryssa\.[0-9]+ builds=[1-9]' <<<"${stats}"; then
  echo "expected cached MemorySSA build count" >&2
  exit 1
fi

SISY_ENABLE_ANALYSIS_MANAGER=0 "${COMPILER}" "${CASE}" -S \
  -o "${OUT_DIR}/basic.legacy.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir

if ! cmp -s "${OUT_DIR}/basic.default.rv.s" "${OUT_DIR}/basic.legacy.rv.s"; then
  echo "AnalysisManager default path changed emitted assembly versus legacy path" >&2
  exit 1
fi

echo "Analysis manager cache and legacy-equivalence tests passed."
