#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/bitwise_arith"
OUT_DIR="${ROOT_DIR}/tests/.out/bitwise_arith"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run cmake --build build -j first"
  exit 1
fi

compile_case() {
  local name="$1"
  "${COMPILER}" "${CASE_DIR}/${name}.sy" -S \
    -o "${OUT_DIR}/${name}.rv.s" \
    -O1 --target=riscv --use-legacy-codegen --verify-ir --stats 2>&1 >/dev/null
}

stats="$(compile_case nonnegative_pow2_ops)"
echo "${stats}"

if grep -q '^structural-bitwise:$' <<<"${stats}"; then
  echo "StructuralBitwise should stay disabled by default; this test must rely on generic arithmetic folding" >&2
  exit 1
fi

if ! grep -A4 '^range-aware-fold:$' <<<"${stats}" | grep -Eq 'folded-ops : [1-9]'; then
  echo "expected RangeAwareFold to participate in nonnegative power-of-two arithmetic folding" >&2
  exit 1
fi

if ! grep -A2 '^regular-fold:$' <<<"${stats}" | grep -Eq 'folded-ops : [1-9]'; then
  echo "expected RegularFold to participate in generic power-of-two arithmetic folding" >&2
  exit 1
fi

if grep -Eq '(^|[[:space:],])(divw|remw)($|[[:space:],])' "${OUT_DIR}/nonnegative_pow2_ops.rv.s"; then
  echo "expected no RISC-V divw/remw after nonnegative power-of-two arithmetic folding" >&2
  exit 1
fi

if ! grep -Eq '(^|[[:space:],])(srliw|sraiw|srlw|sraw|andi|and|slliw|sllw)($|[[:space:],])' "${OUT_DIR}/nonnegative_pow2_ops.rv.s"; then
  echo "expected generated assembly to contain shift/mask instructions" >&2
  exit 1
fi

runtime_stats="$(compile_case runtime_nonnegative_pow2_ops)"
echo "${runtime_stats}"

if grep -q '^structural-bitwise:$' <<<"${runtime_stats}"; then
  echo "StructuralBitwise should stay disabled by default for runtime-valued arithmetic too" >&2
  exit 1
fi

if grep -Eq '(^|[[:space:],])(divw|remw)($|[[:space:],])' "${OUT_DIR}/runtime_nonnegative_pow2_ops.rv.s"; then
  echo "expected backend strength reduction to remove divw/remw for power-of-two arithmetic" >&2
  exit 1
fi

echo "Generic bitwise arithmetic folding tests passed."
