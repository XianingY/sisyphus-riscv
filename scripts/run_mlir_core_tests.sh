#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

out="$("${COMPILER}" --run-self-mlir-core-tests)"
echo "${out}"

grep -q '\[self-mlir-core\]' <<<"${out}" || {
  echo "missing self-MLIR core stats" >&2
  exit 1
}
grep -q 'uniqued=1' <<<"${out}" || {
  echo "expected context hash-consing to unique types and locations" >&2
  exit 1
}
grep -q 'block-args=1' <<<"${out}" || {
  echo "expected block argument SSA in self-MLIR sample" >&2
  exit 1
}
grep -q 'rewrites=1' <<<"${out}" || {
  echo "expected DRR/greedy rewrite to fold addi-zero" >&2
  exit 1
}
grep -q '"builtin.module"' <<<"${out}" || {
  echo "expected MLIR-like module printer output" >&2
  exit 1
}
if grep -Eq 'legacy\.|PhiOp|phi' <<<"${out}"; then
  echo "self-MLIR core output must not contain legacy or Phi operations" >&2
  exit 1
fi

echo "Self-MLIR core tests passed."
