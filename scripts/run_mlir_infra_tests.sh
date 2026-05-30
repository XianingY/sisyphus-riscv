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

"${ROOT_DIR}/scripts/gen-op-descriptors.py" --check --emit=descriptors \
  "${ROOT_DIR}/src/ir/op_schema.yml" \
  "${ROOT_DIR}/src/ir/GeneratedOpDescriptors.inc"
"${ROOT_DIR}/scripts/gen-op-descriptors.py" --check --emit=classes \
  "${ROOT_DIR}/src/ir/op_schema.yml" \
  "${ROOT_DIR}/src/ir/GeneratedOpClasses.inc"

if ! grep -q "class ArithAddiODSOp" "${ROOT_DIR}/src/ir/GeneratedOpClasses.inc"; then
  echo "expected generated ODS wrapper for arith.addi" >&2
  exit 1
fi
if ! grep -q "getLhs" "${ROOT_DIR}/src/ir/GeneratedOpClasses.inc"; then
  echo "expected generated typed getter for arith.addi.lhs" >&2
  exit 1
fi
if ! grep -q "sys::ir::Operation \\*op" "${ROOT_DIR}/src/ir/GeneratedOpClasses.inc"; then
  echo "expected generated ODS wrappers to use Operation*" >&2
  exit 1
fi

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
if ! grep -q "operands=\\[lhs,rhs\\]" <<<"${descriptors}"; then
  echo "expected descriptor dump to include operand names" >&2
  exit 1
fi
if ! grep -q "interfaces=\\[PureOpInterface\\]" <<<"${descriptors}"; then
  echo "expected descriptor dump to include generated interfaces" >&2
  exit 1
fi
if ! grep -q "cppClass=ArithAddiODSOp" <<<"${descriptors}"; then
  echo "expected descriptor dump to include ODS class metadata" >&2
  exit 1
fi

context="$("${COMPILER}" --dump-ir-context)"
echo "${context}"
if ! grep -q "\\[ir-context\\] type i32 uniqued=1" <<<"${context}"; then
  echo "expected uniqued i32 in IRContext dump" >&2
  exit 1
fi
if ! grep -q "\\[ir-context\\] attr loc(\"unknown\":0:0) uniqued=1" <<<"${context}"; then
  echo "expected uniqued location attr in IRContext dump" >&2
  exit 1
fi

scopes="$("${COMPILER}" --dump-pass-scopes)"
echo "${scopes}"
if ! grep -q "\\[pass-scope\\] function mem2reg" <<<"${scopes}"; then
  echo "expected scoped pass dump to contain function mem2reg" >&2
  exit 1
fi
if ! grep -q "\\[pass-scope\\] loop licm" <<<"${scopes}"; then
  echo "expected scoped pass dump to contain loop licm" >&2
  exit 1
fi

conversion="$("${COMPILER}" --dump-dialect-conversion)"
echo "${conversion}"
if ! grep -q "\\[dialect-conversion\\] legal-dialect arith" <<<"${conversion}"; then
  echo "expected dialect conversion dump to contain arith" >&2
  exit 1
fi
if ! grep -q "illegal-count=0" <<<"${conversion}"; then
  echo "expected standard scalar target to legalize all generated ops" >&2
  exit 1
fi

blockargs="$("${COMPILER}" --dump-block-arguments)"
echo "${blockargs}"
if ! grep -q "\\[block-arg\\] count=1" <<<"${blockargs}"; then
  echo "expected block argument dump" >&2
  exit 1
fi
if ! grep -q "first=iv" <<<"${blockargs}"; then
  echo "expected named block argument in dump" >&2
  exit 1
fi

operation_ir="$("${COMPILER}" --dump-operation-ir)"
echo "${operation_ir}"
if ! grep -q "\\[operation-ir\\].*arith.addi" <<<"${operation_ir}"; then
  echo "expected operation IR dump to bridge legacy AddIOp to arith.addi" >&2
  exit 1
fi
if ! grep -q "loc(loc(\"unknown\":0:0))" <<<"${operation_ir}"; then
  echo "expected operation IR dump to include default location" >&2
  exit 1
fi

bridge="$("${COMPILER}" --verify-operation-bridge)"
echo "${bridge}"
if ! grep -q "\\[operation-bridge\\] checked=" <<<"${bridge}"; then
  echo "expected operation bridge verification stats" >&2
  exit 1
fi
if ! grep -q "failures=0" <<<"${bridge}"; then
  echo "expected operation bridge verification to succeed" >&2
  exit 1
fi
if ! grep -q "\\[block-arg-bridge\\]" <<<"${bridge}"; then
  echo "expected block argument bridge round-trip stats" >&2
  exit 1
fi

rollback="$("${COMPILER}" --run-dialect-conversion=rollback-test)"
echo "${rollback}"
if ! grep -q "rollbacks=1" <<<"${rollback}"; then
  echo "expected dialect conversion rollback self-test" >&2
  exit 1
fi

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

bridge_stats="$("${COMPILER}" "${ROOT_DIR}/tests/mlir_infra/pattern_fold.sy" -S \
  -o "${OUT_DIR}/operation_bridge.s" \
  -O1 --target=riscv --use-legacy-codegen --verify-ir \
  --verify-operation-bridge --run-dialect-conversion=legacy \
  --compare "${ROOT_DIR}/tests/mlir_infra/pattern_fold.out" 2>&1)"
echo "${bridge_stats}"
if ! grep -q "\\[operation-bridge\\] checked=" <<<"${bridge_stats}"; then
  echo "expected input operation bridge verification stats" >&2
  exit 1
fi
if ! grep -q "\\[dialect-conversion\\] target=legacy" <<<"${bridge_stats}"; then
  echo "expected legacy dialect conversion dry-run stats" >&2
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
