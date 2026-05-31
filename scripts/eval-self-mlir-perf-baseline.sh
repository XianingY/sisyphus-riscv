#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${SISY_COMPILER_PATH:-${ROOT_DIR}/build/compiler}"
CASE_DIR="${SISY_RISCV_PERF_CASE_DIR:-${ROOT_DIR}/test2026/performance_riscv}"
OUT_DIR="${SISY_SELF_MLIR_BASELINE_OUT:-${ROOT_DIR}/tests/.out/self-mlir-perf-baseline}"
REPORT="${OUT_DIR}/baseline.csv"
SUMMARY="${OUT_DIR}/summary.md"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${COMPILER}" ]]; then
  echo "compiler not found at ${COMPILER}; run scripts/build.sh first" >&2
  exit 1
fi
if [[ ! -d "${CASE_DIR}" ]]; then
  echo "RISC-V performance case dir not found: ${CASE_DIR}" >&2
  exit 1
fi

category_for() {
  local name="$1"
  case "${name}" in
    01_mm*|matmul*|many_mat_cal*) echo "matrix" ;;
    transpose*|shuffle*) echo "transpose-shuffle" ;;
    conv2d-*|sl*) echo "stencil" ;;
    crc*|huffman-*|crypto-*|h-*) echo "lookup-bitwise" ;;
    03_sort*|knapsack_naive*) echo "branch-dp" ;;
    fft*|optimization_scheduling*|prime_search*) echo "math-scheduling" ;;
    *) echo "other" ;;
  esac
}

field() {
  local line="$1"
  local key="$2"
  sed -n "s/.* ${key}=\\([^ ]*\\).*/\\1/p" <<<"${line}"
}

printf 'case,category,status,asm_bytes,ast_nodes,ops_before,ops_after,rewrites,affine_loops,scf_loops,memref_ops,loads,stores,calls,machine_ops,globals_promoted,globals_erased,mem_forwarded_loads,mem_removed_stores,bitwise_candidates,bitwise_rewritten_calls,bitwise_guarded_calls,affine_summary_loops,affine_summary_memory_ops,affine_summary_side_effects,conversion_converted,conversion_failed,native_machine_ops,native_unsupported,native_live_spills,native_dead_spills_avoided,native_call_boundary_spills,stats_file,asm_file\n' >"${REPORT}"

shopt -s nullglob
cases=("${CASE_DIR}"/*.sy)
if [[ "${#cases[@]}" -eq 0 ]]; then
  echo "no performance cases found under ${CASE_DIR}" >&2
  exit 1
fi

for src in "${cases[@]}"; do
  name="$(basename "${src}" .sy)"
  asm="${OUT_DIR}/${name}.s"
  stats="${OUT_DIR}/${name}.stats"
  category="$(category_for "${name}")"
  echo "[self-mlir-baseline] compile ${name}"
  status="ok"
  if ! "${COMPILER}" "${src}" -S -o "${asm}" -O1 --target=riscv --stats >"${stats}" 2>&1; then
    status="compile-fail"
  fi

  self_line="$(grep '^\[self-mlir\]' "${stats}" | tail -1 || true)"
  asm_line="$(grep '^\[native-asm\]' "${stats}" | tail -1 || true)"
  asm_bytes=0
  [[ -f "${asm}" ]] && asm_bytes="$(wc -c <"${asm}" | tr -d ' ')"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${name}" "${category}" "${status}" "${asm_bytes}" \
    "$(field "${self_line}" ast-nodes)" \
    "$(field "${self_line}" ops-before)" \
    "$(field "${self_line}" ops-after)" \
    "$(field "${self_line}" rewrites)" \
    "$(field "${self_line}" affine-loops)" \
    "$(field "${self_line}" scf-loops)" \
    "$(field "${self_line}" memref-ops)" \
    "$(field "${self_line}" loads)" \
    "$(field "${self_line}" stores)" \
    "$(field "${self_line}" calls)" \
    "$(field "${self_line}" machine-ops)" \
    "$(field "${self_line}" globals-promoted)" \
    "$(field "${self_line}" globals-erased)" \
    "$(field "${self_line}" mem-forwarded-loads)" \
    "$(field "${self_line}" mem-removed-stores)" \
    "$(field "${self_line}" bitwise-candidates)" \
    "$(field "${self_line}" bitwise-rewritten-calls)" \
    "$(field "${self_line}" bitwise-guarded-calls)" \
    "$(field "${self_line}" affine-summary-loops)" \
    "$(field "${self_line}" affine-summary-memory-ops)" \
    "$(field "${self_line}" affine-summary-side-effects)" \
    "$(field "${self_line}" conversion-converted)" \
    "$(field "${self_line}" conversion-failed)" \
    "$(field "${asm_line}" machine-ops)" \
    "$(field "${asm_line}" unsupported)" \
    "$(field "${asm_line}" live-spills)" \
    "$(field "${asm_line}" dead-spills-avoided)" \
    "$(field "${asm_line}" call-boundary-spills)" \
    "${stats}" "${asm}" >>"${REPORT}"
done

awk -F, '
NR == 1 { next }
{
  total++;
  cat = $2;
  cats[cat] = 1;
  count[cat]++;
  if ($3 == "ok")
    ok[cat]++;
  asm[cat] += $4 + 0;
  bitRewrite[cat] += $21 + 0;
  bitGuard[cat] += $22 + 0;
  affineLoops[cat] += $23 + 0;
  machineOps[cat] += $28 + 0;
  liveSpills[cat] += $30 + 0;
  deadAvoided[cat] += $31 + 0;
}
END {
  print "# self-MLIR RISC-V Performance Baseline";
  print "";
  print "Case categories in this report are for analysis only. The compiler must not use case names or paths as optimization triggers.";
  print "";
  print "| Category | Cases | OK | ASM bytes | Bitwise direct | Bitwise guarded | Affine loops | Machine ops | Live spills | Dead spills avoided |";
  print "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |";
  for (cat in cats) {
    printf("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d |\n",
           cat, count[cat], ok[cat], asm[cat], bitRewrite[cat],
           bitGuard[cat], affineLoops[cat], machineOps[cat],
           liveSpills[cat], deadAvoided[cat]);
  }
  print "";
  printf("Total cases: %d\n", total);
}
' "${REPORT}" >"${SUMMARY}"

echo "self-MLIR RISC-V perf baseline written to ${REPORT}"
echo "self-MLIR RISC-V perf summary written to ${SUMMARY}"
