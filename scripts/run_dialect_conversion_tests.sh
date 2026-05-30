#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

out="$("${COMPILER}" --run-self-mlir-conversion-tests)"
echo "${out}"

grep -q '\[self-mlir-conversion\]' <<<"${out}" || {
  echo "missing self-MLIR dialect conversion stats" >&2
  exit 1
}
grep -q 'rv-converted=2' <<<"${out}" || {
  echo "expected arith -> rv_machine conversion" >&2
  exit 1
}
grep -q 'arm-converted=2' <<<"${out}" || {
  echo "expected arith -> arm_machine conversion" >&2
  exit 1
}
grep -q 'rollback-count=1' <<<"${out}" || {
  echo "expected conversion rollback on missing pattern" >&2
  exit 1
}
grep -q '"rv_machine.addw"' <<<"${out}" || {
  echo "expected rv_machine op in converted module" >&2
  exit 1
}
grep -q '"arm_machine.add"' <<<"${out}" || {
  echo "expected arm_machine op in converted module" >&2
  exit 1
}
if grep -Eq 'legacy\.|PhiOp|phi' <<<"${out}"; then
  echo "dialect conversion output must not contain legacy or Phi operations" >&2
  exit 1
fi

echo "Self-MLIR dialect conversion tests passed."
