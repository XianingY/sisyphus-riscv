#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${SISY_COMPILER_PATH:-${ROOT_DIR}/build/compiler}"
OUT_DIR="${ROOT_DIR}/tests/.out/proven-bitwise"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

field() {
  local line="$1"
  local key="$2"
  sed -n "s/.* ${key}=\\([^ ]*\\).*/\\1/p" <<<"${line}"
}

write_and_helper() {
  local file="$1"
  cat >"${file}" <<'SYSY'
int bit_helper(int a, int b) {
  int bit_a, bit_b;
  int len = 32, result = 0, power = 1;
  while (len) {
    bit_a = a % 2;
    bit_b = b % 2;
    a = a / 2;
    b = b / 2;
    if (bit_a == 1 && bit_b == 1) {
      result = result + power;
    }
    power = power * 2;
    len = len - 1;
  }
  return result;
}
SYSY
}

write_xor_helper() {
  local file="$1"
  cat >"${file}" <<'SYSY'
int bit_helper(int a, int b) {
  int bit_a, bit_b;
  int len = 32, result = 0, power = 1;
  while (len) {
    bit_a = a % 2;
    bit_b = b % 2;
    a = a / 2;
    b = b / 2;
    if (bit_a != bit_b) {
      result = result + power;
    }
    power = power * 2;
    len = len - 1;
  }
  return result;
}
SYSY
}

write_or_helper() {
  local file="$1"
  cat >"${file}" <<'SYSY'
int bit_helper(int a, int b) {
  int bit_a, bit_b;
  int len = 32, result = 0, power = 1;
  while (len) {
    bit_a = a % 2;
    bit_b = b % 2;
    a = a / 2;
    b = b / 2;
    if (bit_a == 1 || bit_b == 1) {
      result = result + power;
    }
    power = power * 2;
    len = len - 1;
  }
  return result;
}
SYSY
}

write_short_circuit_or_helper() {
  local file="$1"
  cat >"${file}" <<'SYSY'
int bit_helper(int a, int b) {
  int bit_a, bit_b;
  int len = 32, result = 0, power = 1;
  while (len) {
    bit_a = a % 2;
    bit_b = b % 2;
    a = a / 2;
    b = b / 2;
    if (bit_a == 1) {
      result = result + power;
    } else {
      if (bit_b == 1) {
        result = result + power;
      }
    }
    power = power * 2;
    len = len - 1;
  }
  return result;
}
SYSY
}

run_case() {
  local name="$1"
  local expect_key="$2"
  local min_value="$3"
  local sy="${OUT_DIR}/${name}.sy"
  local asm="${OUT_DIR}/${name}.s"
  local stats="${OUT_DIR}/${name}.stats"
  local in="${OUT_DIR}/${name}.in"
  local out="${OUT_DIR}/${name}.out"

  echo "[proven-bitwise] ${name}"
  "${COMPILER}" "${sy}" -S -o "${asm}" -O1 --target=riscv --stats \
    --compare "${out}" -i "${in}" >"${stats}" 2>&1

  local self_line
  self_line="$(grep '^\[self-mlir\]' "${stats}" | tail -1 || true)"
  local actual
  actual="$(field "${self_line}" "${expect_key}")"
  if [[ -z "${actual}" || "${actual}" -lt "${min_value}" ]]; then
    cat "${stats}" >&2
    echo "expected ${expect_key} >= ${min_value}, got ${actual:-<missing>}" >&2
    exit 1
  fi
}

direct_sy="${OUT_DIR}/direct_and.sy"
write_and_helper "${direct_sy}"
cat >>"${direct_sy}" <<'SYSY'
int main() {
  putint(bit_helper(13, 7));
  putch(10);
  return 0;
}
SYSY
: >"${OUT_DIR}/direct_and.in"
printf '5\n' >"${OUT_DIR}/direct_and.out"
run_case direct_and bitwise-rewritten-calls 1
if ! grep -Eq '^[[:space:]]+and[[:space:]]' "${OUT_DIR}/direct_and.s" ||
   ! grep -Eq '^[[:space:]]+addiw[[:space:]]' "${OUT_DIR}/direct_and.s"; then
  echo "expected RISC-V bitwise lowering to sign-extend i32 and result" >&2
  exit 1
fi

direct_or_sy="${OUT_DIR}/direct_or.sy"
write_or_helper "${direct_or_sy}"
cat >>"${direct_or_sy}" <<'SYSY'
int main() {
  putint(bit_helper(5, 2));
  putch(10);
  return 0;
}
SYSY
: >"${OUT_DIR}/direct_or.in"
printf '7\n' >"${OUT_DIR}/direct_or.out"
run_case direct_or bitwise-rewritten-calls 1
if ! grep -Eq '^[[:space:]]+or[[:space:]]' "${OUT_DIR}/direct_or.s"; then
  echo "expected direct OR helper lowering" >&2
  exit 1
fi

nested_or_sy="${OUT_DIR}/nested_or.sy"
write_short_circuit_or_helper "${nested_or_sy}"
cat >>"${nested_or_sy}" <<'SYSY'
int main() {
  putint(bit_helper(8, 4));
  putch(10);
  return 0;
}
SYSY
: >"${OUT_DIR}/nested_or.in"
printf '12\n' >"${OUT_DIR}/nested_or.out"
run_case nested_or bitwise-rewritten-calls 1
if ! grep -Eq '^[[:space:]]+or[[:space:]]' "${OUT_DIR}/nested_or.s"; then
  echo "expected nested short-circuit OR helper lowering" >&2
  exit 1
fi

guarded_sy="${OUT_DIR}/guarded_xor.sy"
write_xor_helper "${guarded_sy}"
cat >>"${guarded_sy}" <<'SYSY'
int main() {
  int x = getint();
  int y = getint();
  putint(bit_helper(x, y));
  putch(10);
  return 0;
}
SYSY
printf '5 3\n' >"${OUT_DIR}/guarded_xor.in"
printf '6\n' >"${OUT_DIR}/guarded_xor.out"
run_case guarded_xor bitwise-guarded-calls 1
if ! grep -Eq '^[[:space:]]+xor[[:space:]]' "${OUT_DIR}/guarded_xor.s" ||
   ! grep -Eq '^[[:space:]]+addiw[[:space:]]' "${OUT_DIR}/guarded_xor.s"; then
  echo "expected RISC-V bitwise lowering to sign-extend i32 xor result" >&2
  exit 1
fi

negative_sy="${OUT_DIR}/negative_shape.sy"
cat >"${negative_sy}" <<'SYSY'
int bit_helper(int a, int b) {
  return a + b;
}
int main() {
  putint(bit_helper(13, 7));
  putch(10);
  return 0;
}
SYSY
: >"${OUT_DIR}/negative_shape.in"
printf '20\n' >"${OUT_DIR}/negative_shape.out"
"${COMPILER}" "${negative_sy}" -S -o "${OUT_DIR}/negative_shape.s" -O1 \
  --target=riscv --stats --compare "${OUT_DIR}/negative_shape.out" \
  -i "${OUT_DIR}/negative_shape.in" >"${OUT_DIR}/negative_shape.stats" 2>&1
self_line="$(grep '^\[self-mlir\]' "${OUT_DIR}/negative_shape.stats" | tail -1 || true)"
rewritten="$(field "${self_line}" bitwise-rewritten-calls)"
guarded="$(field "${self_line}" bitwise-guarded-calls)"
if [[ "${rewritten:-0}" -ne 0 || "${guarded:-0}" -ne 0 ]]; then
  cat "${OUT_DIR}/negative_shape.stats" >&2
  echo "non-bitwise helper shape was rewritten" >&2
  exit 1
fi

echo "Proven bitwise helper tests passed."
