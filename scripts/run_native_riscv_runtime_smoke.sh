#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${SISY_COMPILER_PATH:-${ROOT_DIR}/build/compiler}"
OUT_DIR="${ROOT_DIR}/tests/.out/native-riscv-smoke"
SYLIB_C="${ROOT_DIR}/runtime/sylib.c"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi

GCC_BIN="${RISCV_GCC:-}"
if [[ -z "${GCC_BIN}" ]]; then
  for candidate in riscv64-linux-gnu-gcc riscv64-unknown-linux-gnu-gcc; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      GCC_BIN="${candidate}"
      break
    fi
  done
fi

QEMU_BIN="${RISCV_QEMU:-}"
if [[ -z "${QEMU_BIN}" ]]; then
  for candidate in qemu-riscv64-static qemu-riscv64; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      QEMU_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${GCC_BIN}" || -z "${QEMU_BIN}" ]]; then
  echo "SKIP: RISC-V runtime smoke requires riscv64 gcc and qemu-riscv64." >&2
  echo "      Set RISCV_GCC/RISCV_QEMU or install the toolchain to run it." >&2
  if [[ "${SISY_REQUIRE_RISCV_RUNTIME:-0}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

case_names=()
case_expect=()
case_input=()

write_case() {
  local name="$1"
  local expect="$2"
  local input="${3:-}"
  local file="${OUT_DIR}/${name}.sy"
  case_names+=("${name}")
  case_expect+=("${expect}")
  case_input+=("${input}")
  cat >"${file}"
}

write_case "if_return_label" "7" <<'SYSY'
int main() {
  if (1) {
    putint(7);
    return 0;
  }
  putint(3);
  return 0;
}
SYSY

write_case "while_continue" "12" <<'SYSY'
int main() {
  int i = 0;
  int s = 0;
  while (i < 5) {
    i = i + 1;
    if (i == 3) {
      continue;
    }
    s = s + i;
  }
  putint(s);
  return 0;
}
SYSY

write_case "large_global_bss" "3" <<'SYSY'
int a[2097152];
int main() {
  a[0] = 1;
  a[2097151] = 2;
  putint(a[0] + a[2097151]);
  return 0;
}
SYSY

write_case "local_array_zero_init" "0 7" <<'SYSY'
int main() {
  int cnt[16] = {};
  cnt[3] = cnt[3] + 7;
  putint(cnt[0]);
  putch(32);
  putint(cnt[3]);
  return 0;
}
SYSY

write_case "nested_affine_iv_unique" "63" <<'SYSY'
int main() {
  int i = 0;
  int sum = 0;
  while (i < 3) {
    int j = 0;
    while (j < 2) {
      sum = sum + i * 10 + j;
      j = j + 1;
    }
    i = i + 1;
  }
  putint(sum);
  return 0;
}
SYSY

write_case "multidim_param_row_major" "11" <<'SYSY'
void write_row(int a[][4]) {
  int i = 0;
  while (i < 3) {
    int j = 0;
    while (j < 4) {
      a[i][j] = i * 10 + j;
      j = j + 1;
    }
    i = i + 1;
  }
}

int main() {
  int a[3][4];
  write_row(a);
  putint(a[2][3] - a[1][2]);
  return 0;
}
SYSY

write_case "global_scalar_init" "5" <<'SYSY'
const int K = 5;
int main() {
  putint(K);
  return 0;
}
SYSY

write_case "void_return_fallthrough" "123" <<'SYSY'
void emit(int x) {
  if (x == 0) {
    putint(1);
    return;
  }
  putint(2);
}

int main() {
  emit(0);
  emit(1);
  putint(3);
  return 0;
}
SYSY

write_case "rotate_helper_fold" "96 8" <<'SYSY'
int rotrN(int x, int n) {
  if (n == 1) return x / 2;
  if (n == 2) return x / 4;
  if (n == 3) return x / 8;
  if (n == 4) return x / 16;
  if (n == 5) return x / 32;
  if (n == 6) return x / 64;
  if (n == 7) return x / 128;
  if (n == 8) return x / 256;
  return x;
}
int rotlN(int x, int n) {
  if (n == 1) return x * 2;
  if (n == 2) return x * 4;
  if (n == 3) return x * 8;
  if (n == 4) return x * 16;
  if (n == 5) return x * 32;
  if (n == 6) return x * 64;
  if (n == 7) return x * 128;
  if (n == 8) return x * 256;
  return x;
}
int main() {
  putint(rotlN(3, 5));
  putch(32);
  putint(rotrN(64, 3));
  return 0;
}
SYSY

write_case "rotate_helper_dynamic" "96 8" "5 3" <<'SYSY'
int rotrN(int x, int n) {
  if (n == 1) return x / 2;
  if (n == 2) return x / 4;
  if (n == 3) return x / 8;
  if (n == 4) return x / 16;
  if (n == 5) return x / 32;
  if (n == 6) return x / 64;
  if (n == 7) return x / 128;
  if (n == 8) return x / 256;
  return x;
}
int rotlN(int x, int n) {
  if (n == 1) return x * 2;
  if (n == 2) return x * 4;
  if (n == 3) return x * 8;
  if (n == 4) return x * 16;
  if (n == 5) return x * 32;
  if (n == 6) return x * 64;
  if (n == 7) return x * 128;
  if (n == 8) return x * 256;
  return x;
}
int main() {
  int a = getint();
  int b = getint();
  putint(rotlN(3, a));
  putch(32);
  putint(rotrN(64, b));
  return 0;
}
SYSY

write_case "signed_pow2_strength" "-2 -1" <<'SYSY'
int main() {
  int x = -9;
  putint(x / 4);
  putch(32);
  putint(x % 4);
  return 0;
}
SYSY

if [[ "${SISY_NATIVE_SMOKE_INCLUDE_PERF:-0}" == "1" ]]; then
  for perf in 01_mm1 crc1 conv2d-1 matmul1; do
    src="${ROOT_DIR}/test2026/performance_riscv/${perf}.sy"
    input="${ROOT_DIR}/test2026/performance_riscv/${perf}.in"
    expect="${ROOT_DIR}/test2026/performance_riscv/${perf}.out"
    [[ -f "${src}" && -f "${input}" && -f "${expect}" ]] || continue
    cp "${src}" "${OUT_DIR}/${perf}.sy"
    cp "${input}" "${OUT_DIR}/${perf}.in"
    cp "${expect}" "${OUT_DIR}/${perf}.expect"
    case_names+=("${perf}")
    case_expect+=("@file:${OUT_DIR}/${perf}.expect")
    case_input+=("@file:${OUT_DIR}/${perf}.in")
  done
fi

for i in "${!case_names[@]}"; do
  name="${case_names[$i]}"
  expect="${case_expect[$i]}"
  input_data="${case_input[$i]}"
  src="${OUT_DIR}/${name}.sy"
  asm="${OUT_DIR}/${name}.s"
  exe="${OUT_DIR}/${name}"
  in_file="${OUT_DIR}/${name}.in"
  actual="${OUT_DIR}/${name}.actual"
  log="${OUT_DIR}/${name}.log"
  if [[ "${input_data}" == @file:* ]]; then
    input_path="${input_data#@file:}"
    if [[ "${input_path}" != "${in_file}" ]]; then
      cp "${input_path}" "${in_file}"
    fi
  elif [[ -n "${input_data}" ]]; then
    printf '%s\n' "${input_data}" >"${in_file}"
  else
    : >"${in_file}"
  fi

  echo "[native-riscv-smoke] ${name}"
  "${COMPILER}" "${src}" -S -o "${asm}" -O1 --target=riscv --verify-ir --stats >"${log}" 2>&1
  if [[ "${name}" == rotate_helper_* ]] &&
     grep -Eq 'call[[:space:]]+(rotlN|rotrN)' "${asm}"; then
    cat "${log}" >&2
    echo "rotate helper fold left rotlN/rotrN calls in ${asm}" >&2
    exit 1
  fi
  if [[ "${name}" == "signed_pow2_strength" ]] &&
     grep -Eq '(^|[[:space:],])(divw|remw)($|[[:space:],])' "${asm}"; then
    cat "${log}" >&2
    echo "power-of-two strength reduction left divw/remw in ${asm}" >&2
    exit 1
  fi
  "${GCC_BIN}" -static "${asm}" "${SYLIB_C}" -lm -o "${exe}" >>"${log}" 2>&1
  set +e
  timeout "${SISY_NATIVE_SMOKE_TIMEOUT:-20}" "${QEMU_BIN}" "${exe}" <"${in_file}" >"${actual}" 2>>"${log}"
  run_rc=$?
  set -e
  if [[ "${run_rc}" -eq 124 ]]; then
    cat "${log}" >&2
    echo "native RISC-V smoke timeout for ${name}" >&2
    exit 1
  fi

  if [[ "${expect}" == @file:* ]]; then
    if [[ -s "${actual}" ]]; then
      last_byte="$(tail -c 1 "${actual}" | od -An -t x1 | tr -d '[:space:]')"
      if [[ "${last_byte}" != "0a" ]]; then
        echo >>"${actual}"
      fi
    fi
    printf '%d\n' "${run_rc}" >>"${actual}"
    diff -u "${expect#@file:}" "${actual}" >>"${log}" 2>&1 || {
      cat "${log}" >&2
      echo "native RISC-V smoke output mismatch for ${name}" >&2
      exit 1
    }
  elif [[ "$(tr -d '\r\n' <"${actual}")" != "${expect}" ]]; then
    cat "${log}" >&2
    echo "native RISC-V smoke output mismatch for ${name}: expected '${expect}', got '$(cat "${actual}")'" >&2
    exit 1
  fi
done

echo "Native RISC-V runtime smoke passed (${#case_names[@]} cases)."
