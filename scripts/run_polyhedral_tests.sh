#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${COMPILER:-${ROOT_DIR}/build/compiler}"
CASE_DIR="${ROOT_DIR}/tests/polyhedral"
OUT_DIR="${ROOT_DIR}/tests/.out/polyhedral"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run cmake --build build -j first"
  exit 1
fi

fusion_stats="$("${COMPILER}" "${CASE_DIR}/fusion_disjoint_domains.sy" -S \
  -o "${OUT_DIR}/fusion_disjoint_domains.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${fusion_stats}"

if ! grep -q "fusion-applied=1" <<<"${fusion_stats}"; then
  echo "expected HIR Presburger fusion to prove disjoint shifted domains" >&2
  exit 1
fi

if grep -q "fusion-reject-memory=[1-9]" <<<"${fusion_stats}"; then
  echo "unexpected HIR fusion memory rejection for disjoint shifted domains" >&2
  exit 1
fi

commuted_step_stats="$("${COMPILER}" "${CASE_DIR}/fusion_commuted_step.sy" -S \
  -o "${OUT_DIR}/fusion_commuted_step.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${commuted_step_stats}"

if ! grep -q "fusion-applied=1" <<<"${commuted_step_stats}"; then
  echo "expected HIR fusion to accept commuted induction steps" >&2
  exit 1
fi

jam_stats="$("${COMPILER}" "${CASE_DIR}/reduction_unroll_jam.sy" -S \
  -o "${OUT_DIR}/reduction_unroll_jam.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${jam_stats}"

if ! grep -q "reduction-jammed=1" <<<"${jam_stats}"; then
  echo "expected HIR reduction unroll-and-jam to fire" >&2
  exit 1
fi

if grep -q "reduction-interchanged=[1-9]" <<<"${jam_stats}"; then
  echo "expected in-place reduction to use jam, not interchange" >&2
  exit 1
fi

interchange_3d_stats="$("${COMPILER}" "${CASE_DIR}/interchange_3d_jk_safe.sy" -S \
  -o "${OUT_DIR}/interchange_3d_jk_safe.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${interchange_3d_stats}"

if ! grep -q "interchange-3d-applied=1" <<<"${interchange_3d_stats}"; then
  echo "expected HIR 3D Presburger direction analysis to interchange j/k" >&2
  exit 1
fi

cross_dim_3d_stats="$("${COMPILER}" "${CASE_DIR}/interchange_3d_jk_cross_dim_disjoint.sy" -S \
  -o "${OUT_DIR}/interchange_3d_jk_cross_dim_disjoint.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${cross_dim_3d_stats}"

if ! grep -q "interchange-3d-applied=1" <<<"${cross_dim_3d_stats}"; then
  echo "expected HIR 3D Presburger analysis to prove cross-dimensional disjointness" >&2
  exit 1
fi

if grep -q "interchange-3d-reject-memory=[1-9]" <<<"${cross_dim_3d_stats}"; then
  echo "unexpected HIR 3D memory rejection for cross-dimensional disjoint access" >&2
  exit 1
fi

cross_dim_overlap_3d_stats="$("${COMPILER}" "${CASE_DIR}/interchange_3d_jk_cross_dim_overlap_safe.sy" -S \
  -o "${OUT_DIR}/interchange_3d_jk_cross_dim_overlap_safe.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${cross_dim_overlap_3d_stats}"

if ! grep -q "interchange-3d-applied=1" <<<"${cross_dim_overlap_3d_stats}"; then
  echo "expected HIR 3D direction vectors to prove cross-dimensional overlap safe" >&2
  exit 1
fi

if grep -q "interchange-3d-reject-memory=[1-9]" <<<"${cross_dim_overlap_3d_stats}"; then
  echo "unexpected HIR 3D memory rejection for safe cross-dimensional overlap" >&2
  exit 1
fi

unsafe_3d_stats="$("${COMPILER}" "${CASE_DIR}/interchange_3d_jk_unsafe.sy" -S \
  -o "${OUT_DIR}/interchange_3d_jk_unsafe.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${unsafe_3d_stats}"

if grep -q "interchange-3d-applied=[1-9]" <<<"${unsafe_3d_stats}"; then
  echo "unexpected HIR 3D interchange across loop-carried dependence" >&2
  exit 1
fi

if ! grep -q "interchange-3d-reject-memory=[1-9]" <<<"${unsafe_3d_stats}"; then
  echo "expected HIR 3D interchange to reject the unsafe case by memory dependence" >&2
  exit 1
fi

triangular_stats="$("${COMPILER}" "${CASE_DIR}/triangular_partial_unroll.sy" -S \
  -o "${OUT_DIR}/triangular_partial_unroll.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${triangular_stats}"

if ! grep -q "monotone-guard-tightened=1" <<<"${triangular_stats}"; then
  echo "expected HIR monotone guard tightening on triangular loop" >&2
  exit 1
fi

if ! grep -q "partial-unrolled=1" <<<"${triangular_stats}"; then
  echo "expected HIR order-preserving partial unroll after triangular tightening" >&2
  exit 1
fi

guard_stats="$("${COMPILER}" "${CASE_DIR}/invariant_guard_hoist.sy" -S \
  -o "${OUT_DIR}/invariant_guard_hoist.rv.s" \
  -O1 --target=riscv --verify-hir --verify-ir --stats 2>&1 >/dev/null)"

echo "${guard_stats}"

if ! grep -q "invariant-guard-hoisted=1" <<<"${guard_stats}"; then
  echo "expected HIR loop-invariant guard hoisting on split conjunction" >&2
  exit 1
fi

echo "HIR polyhedral Presburger fusion, interchange, jam, triangular unroll, and guard-hoist tests passed."
