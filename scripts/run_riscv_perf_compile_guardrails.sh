#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${SISY_RISCV_PERF_CASE_DIR:-${ROOT_DIR}/test2026/performance_riscv}"
OUT_DIR="${ROOT_DIR}/tests/.out/riscv-perf-compile"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi
if [[ ! -d "${CASE_DIR}" ]]; then
  echo "RISC-V performance case dir not found: ${CASE_DIR}" >&2
  exit 1
fi

if rg -n 'inputFile\.find|path\.find|basename\(|caseName|case_name|sourcePath|emitKnown.*Fixup|fixup_output|SISY_ENABLE_.*FIXUPS' \
    "${ROOT_DIR}/src" >/tmp/sisy-riscv-guard-source.txt; then
  cat /tmp/sisy-riscv-guard-source.txt >&2
  echo "source-name or fixed-output optimization trigger found under src/" >&2
  exit 1
fi

shopt -s nullglob
cases=("${CASE_DIR}"/*.sy)
if [[ "${#cases[@]}" -eq 0 ]]; then
  echo "no performance cases found under ${CASE_DIR}" >&2
  exit 1
fi

for src in "${cases[@]}"; do
  name="$(basename "${src}" .sy)"
  asm="${OUT_DIR}/${name}.rv.s"
  stats="${OUT_DIR}/${name}.stats"
  echo "[riscv-guard] compile ${name}"
  "${COMPILER}" "${src}" -S -o "${asm}" -O1 --target=riscv --verify-ir --stats \
    >"${stats}" 2>&1

  if ! grep -Eq '\[self-mlir\].*frontend_path=self-mlir.*failed=0' "${stats}" ||
     ! grep -Eq '\[native-asm\].*emitted=1' "${stats}"; then
    cat "${stats}" >&2
    echo "default RISC-V path did not report a complete self-MLIR/native-asm compile for ${name}" >&2
    exit 1
  fi
  self_line="$(grep '^\[self-mlir\]' "${stats}" | tail -1 || true)"
  globals_promoted="$(sed -n 's/.* globals-promoted=\([^ ]*\).*/\1/p' <<<"${self_line}")"
  if [[ -n "${globals_promoted}" && "${globals_promoted}" != "0" ]]; then
    cat "${stats}" >&2
    echo "default RISC-V path promoted globals to stack for ${name}; keep globals in .bss unless explicitly opted in" >&2
    exit 1
  fi
  native_line="$(grep '^\[native-asm\]' "${stats}" | tail -1 || true)"
  # mm-like-kernels is a structural lowering guarded by concrete call-site
  # noalias proof; keep it visible in stats but do not classify it with the
  # source/benchmark semantic replacements below.
  for field in \
    semantic-kernels \
    modular-multiply-kernels \
    digit-helper-kernels \
    experimental-many-mat-cal-kernels \
    experimental-sl-stencil-kernels \
    experimental-matmul-summary-kernels \
    experimental-conv2d-interior-kernels; do
    value="$(sed -n "s/.* ${field}=\\([^ ]*\\).*/\\1/p" <<<"${native_line}")"
    if [[ -n "${value}" && "${value}" != "0" ]]; then
      cat "${stats}" >&2
      echo "default RISC-V path emitted semantic native kernel ${field}=${value} for ${name}" >&2
      exit 1
    fi
  done

  if awk '
    $1 == "addi" {
      gsub(",", "", $2); gsub(",", "", $3);
      imm = $4 + 0;
      if (imm < -2048 || imm > 2047) bad = 1;
    }
    $1 ~ /^(lw|sw|ld|sd|flw|fsw)$/ {
      operand = $3;
      gsub(",", "", operand);
      if (operand ~ /^-?[0-9]+\(sp\)$/) {
        imm = operand;
        sub(/\(sp\)$/, "", imm);
        if (imm + 0 < -2048 || imm + 0 > 2047) bad = 1;
      }
    }
    END { exit bad ? 0 : 1 }
  ' "${asm}"; then
    echo "default RISC-V path emitted an out-of-range RISC-V stack immediate for ${name}" >&2
    exit 1
  fi

  if grep -Eq '\bvsetvli\b|\bv[ls]e[0-9]+\.v\b|\bv[ls]se[0-9]+\.v\b' "${asm}"; then
    echo "default RISC-V path emitted RVV instructions for ${name}; use --enable-rvv for RVV" >&2
    exit 1
  fi
  if grep -Eq '\.L(functional|matrix|scheduling)_fixup_output|call[[:space:]]+printf' "${asm}"; then
    echo "default RISC-V path emitted a synthetic fixed-output helper for ${name}" >&2
    exit 1
  fi

  for forbidden in \
    'cached' \
    'function-equivalence' \
    'structural-bitwise' \
    'structural-modmul'; do
    if grep -Eq "^${forbidden}:$" "${stats}" &&
       ! grep -A8 "^${forbidden}:$" "${stats}" | grep -Eq ': 0$|<no stats>'; then
      echo "forbidden default semantic replacement pass appears active: ${forbidden} on ${name}" >&2
      exit 1
    fi
  done
done

echo "RISC-V performance compile guardrails passed (${#cases[@]} cases)."
