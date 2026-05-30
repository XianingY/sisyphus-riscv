#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
OUT_DIR="${ROOT_DIR}/tests/.out/mlir_infra"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/gen-op-descriptors.py" --check \
  "${ROOT_DIR}/src/ir/op_schema.yml" \
  "${ROOT_DIR}/src/ir/GeneratedOpDescriptors.inc"

if "${ROOT_DIR}/scripts/gen-op-descriptors.py" \
    "${ROOT_DIR}/tests/opgen/bad_missing_field.yml" >/dev/null 2>"${OUT_DIR}/bad-missing.log"; then
  echo "expected opgen to reject missing field" >&2
  exit 1
fi
if ! grep -q "missing fold" "${OUT_DIR}/bad-missing.log"; then
  echo "missing-field diagnostic did not mention fold" >&2
  cat "${OUT_DIR}/bad-missing.log" >&2
  exit 1
fi

if "${ROOT_DIR}/scripts/gen-op-descriptors.py" \
    "${ROOT_DIR}/tests/opgen/bad_trait.yml" >/dev/null 2>"${OUT_DIR}/bad-trait.log"; then
  echo "expected opgen to reject unknown trait" >&2
  exit 1
fi
if ! grep -q "unknown trait" "${OUT_DIR}/bad-trait.log"; then
  echo "bad-trait diagnostic did not mention unknown trait" >&2
  cat "${OUT_DIR}/bad-trait.log" >&2
  exit 1
fi

descriptors="$("${COMPILER}" --dump-op-descriptors)"
for op in arith.addi arith.select scf.for memref.load; do
  if ! grep -q "${op}" <<<"${descriptors}"; then
    echo "expected op descriptor dump to contain ${op}" >&2
    exit 1
  fi
done

stats="$(SISY_ENABLE_PATTERN_CANONICALIZE=1 "${COMPILER}" \
  "${ROOT_DIR}/tests/mlir_infra/pattern_fold.sy" -S \
  -o "${OUT_DIR}/pattern_fold.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --stats \
  --pass-pipeline=pattern-canonicalize \
  --compare "${ROOT_DIR}/tests/mlir_infra/pattern_fold.out" 2>&1 >/dev/null)"

echo "${stats}"
if ! grep -q "pattern-canonicalize:" <<<"${stats}"; then
  echo "expected pattern-canonicalize stats" >&2
  exit 1
fi
if ! grep -Eq "rewrites : [1-9]" <<<"${stats}"; then
  echo "expected pattern-canonicalize to rewrite at least one op" >&2
  exit 1
fi

cache="$("${COMPILER}" "${ROOT_DIR}/tests/memoryssa/join_same_store_load.sy" -S \
  -o "${OUT_DIR}/analysis-cache.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir --dump-analysis-cache 2>&1 >/dev/null)"
echo "${cache}"
for fact in data-layout affine-nest memref-alias; do
  if ! grep -q "\\[analysis-cache\\] ${fact}" <<<"${cache}"; then
    echo "expected analysis cache stats for ${fact}" >&2
    exit 1
  fi
done

if "${COMPILER}" "${ROOT_DIR}/tests/mlir_infra/pattern_fold.sy" -S \
    -o "${OUT_DIR}/bad-pipeline.s" \
    -O1 --target=riscv --pass-pipeline=does-not-exist >/dev/null 2>"${OUT_DIR}/bad-pipeline.log"; then
  echo "expected unknown text pipeline pass to fail" >&2
  exit 1
fi
if ! grep -q "unknown pass" "${OUT_DIR}/bad-pipeline.log"; then
  echo "unknown pipeline diagnostic missing" >&2
  cat "${OUT_DIR}/bad-pipeline.log" >&2
  exit 1
fi

echo "MLIR-style infra tests passed."
