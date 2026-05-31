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
grep -q 'main-symbol=1' <<<"${out}" || {
  echo "expected self-MLIR symbol table to resolve main" >&2
  exit 1
}
grep -q 'add-uses-before=1' <<<"${out}" || {
  echo "expected self-MLIR use-def query to see the return use" >&2
  exit 1
}
grep -q 'erased-dead-const=1' <<<"${out}" || {
  echo "expected self-MLIR erase API to remove a dead op" >&2
  exit 1
}
grep -q 'moved-op=1' <<<"${out}" || {
  echo "expected self-MLIR move API to move an operation" >&2
  exit 1
}
grep -q 'replaced-op=1' <<<"${out}" || {
  echo "expected self-MLIR replace API to rewrite an operation" >&2
  exit 1
}
grep -q 'roundtrip-verify=1' <<<"${out}" || {
  echo "expected self-MLIR parser/printer round-trip to verify" >&2
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

native="$("${COMPILER}" --run-self-mlir-native-backend-tests)"
echo "${native}"
grep -q '\[self-mlir-native-backend\]' <<<"${native}" || {
  echo "missing self-MLIR native backend stats" >&2
  exit 1
}
grep -q 'rv-emitted=1' <<<"${native}" || {
  echo "expected native RISC-V machine dialect asm emission" >&2
  exit 1
}
grep -q 'arm-emitted=1' <<<"${native}" || {
  echo "expected native ARM machine dialect asm emission" >&2
  exit 1
}
grep -q 'legacy-free=1' <<<"${native}" || {
  echo "native self-MLIR backend smoke must be legacy/Phi free" >&2
  exit 1
}
grep -q 'addw' <<<"${native}" || {
  echo "expected RISC-V native asm to contain addw" >&2
  exit 1
}
grep -q 'add w' <<<"${native}" || {
  echo "expected ARM native asm to contain add" >&2
  exit 1
}
if grep -Eq 'legacy\.|PhiOp|\"phi\"' <<<"${native}"; then
  echo "native self-MLIR backend output must not contain legacy or Phi operations" >&2
  exit 1
fi

OUT_DIR="${ROOT_DIR}/tests/.out/self_mlir_core"
mkdir -p "${OUT_DIR}"

prod="$("${COMPILER}" "${ROOT_DIR}/tests/smoke/basic.sy" -S -o "${OUT_DIR}/basic.rv.s" -O0 --target=riscv --stats --dialect-fallback-report=stderr 2>&1 >/dev/null)"
echo "${prod}"
grep -q '\[self-mlir\]' <<<"${prod}" || {
  echo "default compile did not run the self-MLIR production gate" >&2
  exit 1
}
grep -q 'source=ast' <<<"${prod}" || {
  echo "default self-MLIR production gate must lower directly from AST" >&2
  exit 1
}
grep -q 'frontend_path=self-mlir' <<<"${prod}" || {
  echo "default dialect report must identify self-MLIR as the frontend path" >&2
  exit 1
}
grep -q 'failed=0' <<<"${prod}" || {
  echo "self-MLIR target legalization failed during default compile" >&2
  exit 1
}

dump="$(SISY_DUMP_SELF_MLIR=1 "${COMPILER}" "${ROOT_DIR}/tests/smoke/basic.sy" -S -o "${OUT_DIR}/basic.dump.rv.s" -O0 --target=riscv --dialect-fallback-report=stderr 2>&1 >/dev/null)"
echo "${dump}"
grep -q '===== self-MLIR production =====' <<<"${dump}" || {
  echo "SISY_DUMP_SELF_MLIR did not dump the production module" >&2
  exit 1
}
if grep -Eq 'legacy\.|PhiOp|\"phi\"' <<<"${dump}"; then
  echo "production self-MLIR dump must not contain legacy or Phi operations" >&2
  exit 1
fi

cat >"${OUT_DIR}/ast_direct.sy" <<'SYSY'
int main() {
  int x = 1;
  x = x + 2;
  if (x < 4) {
    x = x + 1;
  }
  return x;
}
SYSY
ast_dump="$(SISY_DUMP_SELF_MLIR=1 "${COMPILER}" "${OUT_DIR}/ast_direct.sy" -S -o "${OUT_DIR}/ast_direct.rv.s" -O0 --target=riscv --stats --dialect-fallback-report=stderr 2>&1 >/dev/null)"
echo "${ast_dump}"
grep -q 'source=ast' <<<"${ast_dump}" || {
  echo "AST direct smoke did not use AST self-MLIR lowering" >&2
  exit 1
}
grep -q '"sysy.alloca"' <<<"${ast_dump}" || {
  echo "AST direct smoke should emit sysy.alloca for local variables" >&2
  exit 1
}
grep -q '"sysy.store"' <<<"${ast_dump}" || {
  echo "AST direct smoke should emit sysy.store for assignments" >&2
  exit 1
}
grep -q '"scf.if"' <<<"${ast_dump}" || {
  echo "AST direct smoke should preserve structured scf.if" >&2
  exit 1
}

arm_prod="$("${COMPILER}" "${ROOT_DIR}/tests/smoke/basic.sy" -S -o "${OUT_DIR}/basic.arm.s" -O0 --target=arm --stats --dialect-fallback-report=stderr 2>&1 >/dev/null)"
echo "${arm_prod}"
grep -q '\[self-mlir\] target=arm' <<<"${arm_prod}" || {
  echo "ARM compile did not run the self-MLIR production gate" >&2
  exit 1
}
grep -q 'source=ast' <<<"${arm_prod}" || {
  echo "ARM self-MLIR production gate must lower directly from AST" >&2
  exit 1
}
grep -q 'frontend_path=self-mlir' <<<"${arm_prod}" || {
  echo "ARM dialect report must identify self-MLIR as the frontend path" >&2
  exit 1
}
grep -q 'failed=0' <<<"${arm_prod}" || {
  echo "self-MLIR ARM legalization failed during default compile" >&2
  exit 1
}

echo "Self-MLIR core tests passed."
