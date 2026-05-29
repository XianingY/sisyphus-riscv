#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/compliant_replacements"
POLY_DIR="${ROOT_DIR}/tests/polyhedral"
OUT_DIR="${ROOT_DIR}/tests/.out/compliant_replacements"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

wide_stats="$("${COMPILER}" "${CASE_DIR}/wide_modmul_direct.sy" -S \
  -o "${OUT_DIR}/wide_modmul_direct.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${wide_stats}"

if grep -q '^structural-modmul:$' <<<"${wide_stats}"; then
  echo "StructuralModMul must stay opt-in; direct modular arithmetic should use wide arithmetic promotion" >&2
  exit 1
fi

if ! grep -A5 '^wide-arith-promotion:$' <<<"${wide_stats}" | grep -Eq 'promoted : [1-9]'; then
  echo "expected WideArithmeticPromotion to promote nonnegative (a*b)%mod into i64 arithmetic" >&2
  exit 1
fi

if grep -Eq '(^|[[:space:],])(mulw|remw)($|[[:space:],])' "${OUT_DIR}/wide_modmul_direct.rv.s"; then
  echo "expected promoted modular multiplication to avoid 32-bit mulw/remw" >&2
  exit 1
fi

poly_stats="$("${COMPILER}" "${POLY_DIR}/interchange_3d_jk_safe.sy" -S \
  -o "${OUT_DIR}/interchange_3d_jk_safe.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${poly_stats}"

if grep -q '^row-scratch-matmul:$' <<<"${poly_stats}"; then
  echo "RowScratchMatmul helper replacement must stay opt-in; matrix-like loops should use HIR affine transforms" >&2
  exit 1
fi

if ! grep -q 'interchange-3d-applied=1' <<<"${poly_stats}"; then
  echo "expected HIR polyhedral 3D interchange to remain the matrix-loop replacement path" >&2
  exit 1
fi

const_stats="$("${COMPILER}" "${CASE_DIR}/const_array_scalar_replace.sy" -S \
  -o "${OUT_DIR}/const_array_scalar_replace.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${const_stats}"

if ! grep -A3 '^inline-store:$' <<<"${const_stats}" | grep -Eq 'const-loads : [1-9]'; then
  echo "expected InlineStore to replace readonly constant-array loads with scalar constants" >&2
  exit 1
fi

if grep -Eq '(^|[[:space:],])(lw|flw)($|[[:space:],])' "${OUT_DIR}/const_array_scalar_replace.rv.s"; then
  echo "expected constant-array scalar replacement to remove data loads" >&2
  exit 1
fi

echo "Compliant replacement optimization tests passed."
