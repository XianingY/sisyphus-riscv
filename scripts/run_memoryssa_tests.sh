#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/memoryssa"
OUT_DIR="${ROOT_DIR}/tests/.out/memoryssa"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run cmake --build build -j first"
  exit 1
fi

stats="$("${COMPILER}" "${CASE_DIR}/join_same_store_load.sy" -S \
  -o "${OUT_DIR}/join_same_store_load.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${stats}"

if ! grep -A2 '^dle:$' <<<"${stats}" | grep -Eq 'removed-loads : [1-9]'; then
  echo "expected DLE/MemorySSA to replace a merge load with the common reaching store value" >&2
  exit 1
fi

if grep -q 'non-dominating:' <<<"${stats}"; then
  echo "DLE/MemorySSA introduced a non-dominating replacement value" >&2
  exit 1
fi

runtime_stats="$("${COMPILER}" "${CASE_DIR}/join_same_runtime_store_load.sy" -S \
  -o "${OUT_DIR}/join_same_runtime_store_load.rv.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null)"

echo "${runtime_stats}"

if ! grep -A2 '^dle:$' <<<"${runtime_stats}" | grep -Eq 'removed-loads : [1-9]'; then
  echo "expected DLE/MemorySSA to forward a common reaching runtime store value" >&2
  exit 1
fi

if grep -q 'non-dominating:' <<<"${runtime_stats}"; then
  echo "runtime reaching-store forwarding introduced a non-dominating replacement value" >&2
  exit 1
fi

echo "MemorySSA reaching-store load elimination tests passed."
