#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
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

write_case() {
  local name="$1"
  local expect="$2"
  local file="${OUT_DIR}/${name}.sy"
  case_names+=("${name}")
  case_expect+=("${expect}")
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
  done
fi

for i in "${!case_names[@]}"; do
  name="${case_names[$i]}"
  expect="${case_expect[$i]}"
  src="${OUT_DIR}/${name}.sy"
  asm="${OUT_DIR}/${name}.s"
  exe="${OUT_DIR}/${name}"
  in_file="${OUT_DIR}/${name}.in"
  actual="${OUT_DIR}/${name}.actual"
  log="${OUT_DIR}/${name}.log"
  : >"${in_file}"

  echo "[native-riscv-smoke] ${name}"
  "${COMPILER}" "${src}" -S -o "${asm}" -O1 --target=riscv --verify-ir --stats >"${log}" 2>&1
  "${GCC_BIN}" -static "${asm}" "${SYLIB_C}" -lm -o "${exe}" >>"${log}" 2>&1
  timeout "${SISY_NATIVE_SMOKE_TIMEOUT:-20}" "${QEMU_BIN}" "${exe}" <"${in_file}" >"${actual}" 2>>"${log}"

  if [[ "${expect}" == @file:* ]]; then
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
