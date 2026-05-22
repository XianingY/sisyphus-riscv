#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${ROOT_DIR}/build/compiler"
CASE_DIR="${ROOT_DIR}/tests/vectorize"
OUT_DIR="${ROOT_DIR}/tests/.out/vectorize"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first"
  exit 1
fi

asm="${OUT_DIR}/rvv_add_loop.s"
"${COMPILER}" "${CASE_DIR}/rvv_add_loop.sy" -S -o "${asm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vadd.vv" "vse32.v"; do
  if ! grep -q "${needle}" "${asm}"; then
    echo "missing expected RVV instruction '${needle}' in ${asm}"
    exit 1
  fi
done

inplace_fasm="${OUT_DIR}/rvv_loop_inplace_add.s"
"${COMPILER}" "${CASE_DIR}/loop_inplace_add.sy" -S -o "${inplace_fasm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vadd.vv" "vse32.v"; do
  if ! grep -q "${needle}" "${inplace_fasm}"; then
    echo "missing expected RVV in-place loop instruction '${needle}' in ${inplace_fasm}"
    exit 1
  fi
done

same_base_fasm="${OUT_DIR}/rvv_loop_same_base_no_vector.s"
"${COMPILER}" "${CASE_DIR}/loop_same_base_no_vector.sy" -S -o "${same_base_fasm}" -O1 --target=riscv --enable-experimental --verify-ir

if grep -q "vadd.vv" "${same_base_fasm}"; then
  echo "unexpected RVV loop vectorization for loop-carried same-base dependence in ${same_base_fasm}"
  exit 1
fi

fasm="${OUT_DIR}/rvv_fadd_loop.s"
"${COMPILER}" "${CASE_DIR}/rvv_fadd_loop.sy" -S -o "${fasm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vfadd.vv" "vse32.v"; do
  if ! grep -q "${needle}" "${fasm}"; then
    echo "missing expected RVV float instruction '${needle}' in ${fasm}"
    exit 1
  fi
done

arm_fasm="${OUT_DIR}/arm_fadd_loop.s"
"${COMPILER}" "${CASE_DIR}/rvv_fadd_loop.sy" -S -o "${arm_fasm}" -O2 --target=arm --enable-experimental --verify-ir

for needle in "ld1" "fadd v" "st1"; do
  if ! grep -q "${needle}" "${arm_fasm}"; then
    echo "missing expected ARM NEON float instruction '${needle}' in ${arm_fasm}"
    exit 1
  fi
done

slp_fasm="${OUT_DIR}/rvv_slp_fadd_straightline.s"
"${COMPILER}" "${CASE_DIR}/slp_fadd_straightline.sy" -S -o "${slp_fasm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vfadd.vv" "vse32.v"; do
  if ! grep -q "${needle}" "${slp_fasm}"; then
    echo "missing expected RVV SLP float instruction '${needle}' in ${slp_fasm}"
    exit 1
  fi
done

slp_iadd_asm="${OUT_DIR}/rvv_slp_iadd_straightline.s"
"${COMPILER}" "${CASE_DIR}/slp_iadd_straightline.sy" -S -o "${slp_iadd_asm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vadd.vv" "vse32.v"; do
  if ! grep -q "${needle}" "${slp_iadd_asm}"; then
    echo "missing expected RVV SLP int-add instruction '${needle}' in ${slp_iadd_asm}"
    exit 1
  fi
done

slp_copy_asm="${OUT_DIR}/rvv_slp_copy_straightline.s"
"${COMPILER}" "${CASE_DIR}/slp_copy_straightline.sy" -S -o "${slp_copy_asm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vse32.v"; do
  if ! grep -q "${needle}" "${slp_copy_asm}"; then
    echo "missing expected RVV SLP copy instruction '${needle}' in ${slp_copy_asm}"
    exit 1
  fi
done

copy_overlap_asm="${OUT_DIR}/rvv_slp_copy_overlap_vector.s"
"${COMPILER}" "${CASE_DIR}/slp_copy_overlap_vector.sy" -S -o "${copy_overlap_asm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vse32.v"; do
  if ! grep -q "${needle}" "${copy_overlap_asm}"; then
    echo "missing expected RVV SLP overlap-copy instruction '${needle}' in ${copy_overlap_asm}"
    exit 1
  fi
done

arm_slp_fasm="${OUT_DIR}/arm_slp_fadd_straightline.s"
"${COMPILER}" "${CASE_DIR}/slp_fadd_straightline.sy" -S -o "${arm_slp_fasm}" -O2 --target=arm --enable-experimental --verify-ir

for needle in "ld1" "fadd v" "st1"; do
  if ! grep -q "${needle}" "${arm_slp_fasm}"; then
    echo "missing expected ARM NEON SLP float instruction '${needle}' in ${arm_slp_fasm}"
    exit 1
  fi
done

slp_sub_fasm="${OUT_DIR}/rvv_slp_fsub_straightline.s"
"${COMPILER}" "${CASE_DIR}/slp_fsub_straightline.sy" -S -o "${slp_sub_fasm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vfsub.vv" "vse32.v"; do
  if ! grep -q "${needle}" "${slp_sub_fasm}"; then
    echo "missing expected RVV SLP float-sub instruction '${needle}' in ${slp_sub_fasm}"
    exit 1
  fi
done

arm_slp_sub_fasm="${OUT_DIR}/arm_slp_fsub_straightline.s"
"${COMPILER}" "${CASE_DIR}/slp_fsub_straightline.sy" -S -o "${arm_slp_sub_fasm}" -O2 --target=arm --enable-experimental --verify-ir

for needle in "ld1" "fsub v" "st1"; do
  if ! grep -q "${needle}" "${arm_slp_sub_fasm}"; then
    echo "missing expected ARM NEON SLP float-sub instruction '${needle}' in ${arm_slp_sub_fasm}"
    exit 1
  fi
done

slp_splat_fasm="${OUT_DIR}/rvv_slp_fadd_splat.s"
"${COMPILER}" "${CASE_DIR}/slp_fadd_splat.sy" -S -o "${slp_splat_fasm}" -O1 --target=riscv --enable-experimental --verify-ir

for needle in "vsetvli" "vle32.v" "vfmv.v.f" "vfadd.vv" "vse32.v"; do
  if ! grep -q "${needle}" "${slp_splat_fasm}"; then
    echo "missing expected RVV SLP splat instruction '${needle}' in ${slp_splat_fasm}"
    exit 1
  fi
done

arm_slp_splat_fasm="${OUT_DIR}/arm_slp_fadd_splat.s"
"${COMPILER}" "${CASE_DIR}/slp_fadd_splat.sy" -S -o "${arm_slp_splat_fasm}" -O2 --target=arm --enable-experimental --verify-ir

for needle in "ld1" "dup v" "fadd v" "st1"; do
  if ! grep -q "${needle}" "${arm_slp_splat_fasm}"; then
    echo "missing expected ARM NEON SLP splat instruction '${needle}' in ${arm_slp_splat_fasm}"
    exit 1
  fi
done

overlap_fasm="${OUT_DIR}/rvv_slp_overlap_no_vector.s"
"${COMPILER}" "${CASE_DIR}/slp_overlap_no_vector.sy" -S -o "${overlap_fasm}" -O1 --target=riscv --enable-experimental --verify-ir

if grep -q "vfadd.vv" "${overlap_fasm}"; then
  echo "unexpected RVV SLP vectorization for overlapping dependence in ${overlap_fasm}"
  exit 1
fi

echo "RISC-V RVV and ARM NEON loop/SLP vectorization smoke tests passed."
