#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <suite> <target> <opt>"
  echo "suite: official-functional | official-arm-perf | official-riscv-perf | official-arm-final-perf | official-riscv-final-perf"
  echo "target: riscv | arm"
  echo "opt: O0 | O1 | O2"
  exit 1
fi

SUITE="$1"
TARGET="$2"
OPT="$3"

is_supported_suite=0
for s in \
  official-functional \
  official-arm-perf \
  official-riscv-perf \
  official-arm-final-perf \
  official-riscv-final-perf; do
  if [[ "${SUITE}" == "${s}" ]]; then
    is_supported_suite=1
    break
  fi
done
if [[ "${is_supported_suite}" -ne 1 ]]; then
  case "${SUITE}" in
    open-functional)
      echo "error: suite '${SUITE}' has been removed; use 'official-functional'"
      ;;
    open-perf)
      echo "error: suite '${SUITE}' has been removed; use one of 'official-arm-perf', 'official-riscv-perf', 'official-arm-final-perf', 'official-riscv-final-perf'"
      ;;
    compiler-dev|lvx)
      echo "error: suite '${SUITE}' has been removed from baseline"
      ;;
    *)
      echo "error: unsupported suite '${SUITE}'"
      ;;
  esac
  exit 1
fi
if [[ "${TARGET}" != "riscv" && "${TARGET}" != "arm" ]]; then
  echo "error: target must be riscv|arm"
  exit 1
fi
if [[ "${OPT}" != "O0" && "${OPT}" != "O1" && "${OPT}" != "O2" ]]; then
  echo "error: opt must be O0|O1|O2"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_SCRIPT="${ROOT_DIR}/scripts/suite-index.sh"
DOCKERFILE="${ROOT_DIR}/docker/compiler-dev-dual.Dockerfile"
COMPILER_PATH="${SISY_COMPILER_PATH:-${ROOT_DIR}/build/compiler}"
OFFICIAL_RUNTIME_ROOT="${OFFICIAL_RUNTIME_ROOT:-/home/wslootie/github/cpe/compiler2025}"
SISY_OFFICIAL_SUITE_ROOT="${SISY_OFFICIAL_SUITE_ROOT:-${ROOT_DIR}/test/official}"
RUNTIME_SYLIB_C="${RUNTIME_SYLIB_C:-${OFFICIAL_RUNTIME_ROOT}/sylib.c}"
COMPILER_FLAVOR="${SISY_COMPILER_FLAVOR:-sisy}"
LABEL="${RUNTIME_LABEL:-sisyphus}"
RUNTIME_TIMEOUT_SEC="${RUNTIME_TIMEOUT_SEC:-10}"
# Keep functional checks strict (10s), but default perf gate to 20s.
RUNTIME_PERF_TIMEOUT_SEC="${RUNTIME_PERF_TIMEOUT_SEC:-20}"
SISY_DOCKER_IMAGE="${SISY_DOCKER_IMAGE:-sisyphus/compiler-dev-dual:latest}"
SISY_COMPILER_EXTRA_ARGS="${SISY_COMPILER_EXTRA_ARGS:-}"
DEFAULT_RUNTIME_ROOT="${ROOT_DIR}/tests/.out/runtime"
RUNTIME_ROOT="${RUNTIME_ROOT:-${DEFAULT_RUNTIME_ROOT}}"
CSV_EXPLICIT=0
if [[ -n "${RUNTIME_CSV:-}" ]]; then
  CSV_EXPLICIT=1
fi
CSV_OUT="${RUNTIME_CSV:-${RUNTIME_ROOT}/${LABEL}-${SUITE}-${TARGET}-${OPT}.csv}"
RUNTIME_CASE_LIMIT="${RUNTIME_CASE_LIMIT:-0}"
RUNTIME_CASE_FILTER="${RUNTIME_CASE_FILTER:-}"
RUNTIME_SOFT_PERF="${RUNTIME_SOFT_PERF:-0}"
INDEX_SNAPSHOT=""

cleanup() {
  if [[ -n "${INDEX_SNAPSHOT}" && -f "${INDEX_SNAPSHOT}" ]]; then
    rm -f "${INDEX_SNAPSHOT}"
  fi
}
trap cleanup EXIT

if [[ "${COMPILER_FLAVOR}" != "sisy" && "${COMPILER_FLAVOR}" != "biframe" ]]; then
  echo "error: SISY_COMPILER_FLAVOR must be sisy|biframe"
  exit 1
fi

declare -a COMPILER_EXTRA_ARGS_ARR=()
if [[ -n "${SISY_COMPILER_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  COMPILER_EXTRA_ARGS_ARR=(${SISY_COMPILER_EXTRA_ARGS})
fi

if [[ "${SISY_RUNTIME_IN_DOCKER:-0}" != "1" && "${SISY_RUNTIME_LOCAL:-0}" != "1" ]]; then
  if command -v docker >/dev/null 2>&1; then
    if [[ ! -f "${DOCKERFILE}" ]]; then
      echo "error: missing Dockerfile ${DOCKERFILE}"
      exit 1
    fi

    if ! docker image inspect "${SISY_DOCKER_IMAGE}" >/dev/null 2>&1; then
      echo "[docker] build ${SISY_DOCKER_IMAGE}"
      docker build -t "${SISY_DOCKER_IMAGE}" -f "${DOCKERFILE}" "${ROOT_DIR}"
    fi

    echo "[docker] run ${SUITE} ${TARGET} ${OPT}"
    docker run --rm \
      --user "$(id -u):$(id -g)" \
      -e SISY_RUNTIME_IN_DOCKER=1 \
      -e SISY_RUNTIME_LOCAL=1 \
      -e SISY_COMPILER_PATH="${COMPILER_PATH}" \
      -e SISY_COMPILER_FLAVOR="${COMPILER_FLAVOR}" \
      -e OFFICIAL_RUNTIME_ROOT="${OFFICIAL_RUNTIME_ROOT}" \
      -e SISY_OFFICIAL_SUITE_ROOT="${SISY_OFFICIAL_SUITE_ROOT}" \
      -e RUNTIME_SYLIB_C="${RUNTIME_SYLIB_C}" \
      -e RUNTIME_LABEL="${LABEL}" \
      -e RUNTIME_TIMEOUT_SEC="${RUNTIME_TIMEOUT_SEC}" \
      -e RUNTIME_PERF_TIMEOUT_SEC="${RUNTIME_PERF_TIMEOUT_SEC}" \
      -e RUNTIME_SOFT_PERF="${RUNTIME_SOFT_PERF}" \
      -e RUNTIME_CSV="${CSV_OUT}" \
      -e RUNTIME_CASE_LIMIT="${RUNTIME_CASE_LIMIT}" \
      -e RUNTIME_CASE_FILTER="${RUNTIME_CASE_FILTER}" \
      -e SISY_COMPILER_EXTRA_ARGS="${SISY_COMPILER_EXTRA_ARGS}" \
      -v "${ROOT_DIR}:${ROOT_DIR}" \
      -w "${ROOT_DIR}" \
      "${SISY_DOCKER_IMAGE}" \
      bash -lc "scripts/eval-runtime.sh '${SUITE}' '${TARGET}' '${OPT}'"
    exit $?
  fi
fi

if [[ ! -x "${COMPILER_PATH}" ]]; then
  echo "error: compiler not found at ${COMPILER_PATH}"
  exit 1
fi
if [[ "${SISY_RUNTIME_IN_DOCKER:-0}" == "1" ]]; then
  compiler_magic="$(od -An -tx1 -N4 "${COMPILER_PATH}" 2>/dev/null | tr -d ' \n' || true)"
  if [[ "${compiler_magic}" != "7f454c46" ]]; then
    echo "error: ${COMPILER_PATH} is not a Linux ELF executable inside Docker."
    echo "hint: rebuild with: docker run --rm --user \"\$(id -u):\$(id -g)\" -v \"\$PWD:\$PWD\" -w \"\$PWD\" ${SISY_DOCKER_IMAGE} bash -lc 'scripts/build.sh'"
    exit 1
  fi
fi
if [[ ! -f "${RUNTIME_SYLIB_C}" ]]; then
  RUNTIME_SYLIB_C="${ROOT_DIR}/runtime/sylib.c"
fi
if [[ ! -f "${RUNTIME_SYLIB_C}" ]]; then
  echo "error: runtime source not found: ${RUNTIME_SYLIB_C}"
  exit 1
fi
if [[ ! -x "${INDEX_SCRIPT}" ]]; then
  echo "error: missing ${INDEX_SCRIPT}"
  exit 1
fi

INDEX_LOCK="${ROOT_DIR}/tests/.out/suites/.index.lock"
mkdir -p "$(dirname "${INDEX_LOCK}")"
if command -v flock >/dev/null 2>&1; then
  (
    flock -x 9
    "${INDEX_SCRIPT}"
  ) 9>"${INDEX_LOCK}"
else
  "${INDEX_SCRIPT}"
fi

INDEX_CSV="${ROOT_DIR}/tests/.out/suites/index.csv"
if [[ ! -f "${INDEX_CSV}" ]]; then
  echo "error: index missing at ${INDEX_CSV}"
  exit 1
fi
INDEX_SNAPSHOT="$(mktemp)"
cp "${INDEX_CSV}" "${INDEX_SNAPSHOT}"
INDEX_CSV="${INDEX_SNAPSHOT}"

if [[ "${TARGET}" == "riscv" ]]; then
  GCC_BIN="riscv64-linux-gnu-gcc"
  QEMU_BIN="qemu-riscv64-static"
else
  GCC_BIN="aarch64-linux-gnu-gcc"
  QEMU_BIN="qemu-aarch64-static"
fi

for tool in "${GCC_BIN}" "${QEMU_BIN}" timeout awk sort; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: missing tool '${tool}'."
    echo "hint: install toolchain locally or run with Docker enabled."
    exit 1
  fi
done

if ! mkdir -p "${RUNTIME_ROOT}" 2>/dev/null || [[ ! -w "${RUNTIME_ROOT}" ]]; then
  RUNTIME_ROOT="${ROOT_DIR}/.runtime-reports/runtime"
  mkdir -p "${RUNTIME_ROOT}"
  if [[ "${CSV_EXPLICIT}" -eq 0 ]]; then
    CSV_OUT="${RUNTIME_ROOT}/${LABEL}-${SUITE}-${TARGET}-${OPT}.csv"
  fi
fi

csv_dir="$(dirname "${CSV_OUT}")"
if ! mkdir -p "${csv_dir}" 2>/dev/null || [[ ! -w "${csv_dir}" ]]; then
  csv_dir="${RUNTIME_ROOT}"
  mkdir -p "${csv_dir}"
  CSV_OUT="${csv_dir}/$(basename "${CSV_OUT}")"
fi

safe_stem() {
  local id="$1"
  local stem
  stem="${id//\//__}"
  stem="${stem// /_}"
  stem="${stem//$'\t'/_}"
  printf "%s" "${stem}"
}

ns_to_ms() {
  local ns="$1"
  awk -v ns="${ns}" 'BEGIN { printf "%.3f", ns / 1000000.0 }'
}

normalize_text() {
  local file="$1"
  awk '{ sub(/[ \t\r]+$/, "", $0); print }' "${file}"
}

count_input_tokens() {
  local in_file="$1"
  if [[ -f "${in_file}" ]]; then
    awk '{
      for (i = 1; i <= NF; i++)
        c++;
    } END { print c + 0 }' "${in_file}"
  else
    echo "0"
  fi
}

estimate_scalar_reads() {
  local src="$1"
  if [[ ! -f "${src}" ]]; then
    echo "0"
    return
  fi
  (grep -Eo '\<(getint|getfloat|getch)[[:space:]]*\(' "${src}" || true) | wc -l | tr -d ' '
}

run_once() {
  local exe="$1"
  local in_file="$2"
  local stdout_file="$3"
  local stderr_file="$4"
  local timeout_sec="$5"

  local start end rc
  start="$(date +%s%N)"
  set +e
  if [[ -f "${in_file}" ]]; then
    timeout "${timeout_sec}" "${QEMU_BIN}" "${exe}" <"${in_file}" >"${stdout_file}" 2>"${stderr_file}"
    rc=$?
  else
    timeout "${timeout_sec}" "${QEMU_BIN}" "${exe}" >"${stdout_file}" 2>"${stderr_file}"
    rc=$?
  fi
  set -e
  end="$(date +%s%N)"

  RUN_RC="${rc}"
  RUN_NS=$((end - start))
}

compile_case() {
  local src="$1"
  local asm="$2"
  local dialect_report="$3"

  if [[ "${COMPILER_FLAVOR}" == "sisy" ]]; then
    local -a cmd
    cmd=("${COMPILER_PATH}" "${src}" -S -o "${asm}" "--target=${TARGET}" "-${OPT}")
    if [[ -n "${dialect_report}" ]]; then
      cmd+=("--dialect-fallback-report=${dialect_report}")
    fi
    if [[ "${#COMPILER_EXTRA_ARGS_ARR[@]}" -gt 0 ]]; then
      cmd+=("${COMPILER_EXTRA_ARGS_ARR[@]}")
    fi
    "${cmd[@]}"
    return 0
  fi

  # biframe compatibility mode.
  local biframe_opt=""
  if [[ "${OPT}" != "O0" ]]; then
    biframe_opt="-O1"
  fi

  if [[ "${TARGET}" == "arm" ]]; then
    if [[ -n "${biframe_opt}" ]]; then
      "${COMPILER_PATH}" "${src}" -S -o "${asm}" --arm "${biframe_opt}"
    else
      "${COMPILER_PATH}" "${src}" -S -o "${asm}" --arm
    fi
  else
    if [[ -n "${biframe_opt}" ]]; then
      "${COMPILER_PATH}" "${src}" -S -o "${asm}" "${biframe_opt}"
    else
      "${COMPILER_PATH}" "${src}" -S -o "${asm}"
    fi
  fi
}

printf 'suite,case_id,target,opt,label,status,compare,pass,median_ms,warmup_ms,run1_ms,run2_ms,run3_ms,asm,exe,log,compile_status,asm_emit_status,link_status,input_token_count,estimated_read_count,suspect_input_underflow,frontend_path,fallback_reason_codes\n' >"${CSV_OUT}"

asm_root="${RUNTIME_ROOT}/asm/${SUITE}/${LABEL}/${TARGET}-${OPT}"
bin_root="${RUNTIME_ROOT}/bin/${SUITE}/${LABEL}/${TARGET}-${OPT}"
log_root="${RUNTIME_ROOT}/logs/${SUITE}/${LABEL}/${TARGET}-${OPT}"
mkdir -p "${asm_root}" "${bin_root}" "${log_root}"

total=0
pass_count=0
fail_count=0
hard_fail_count=0
soft_fail_count=0

while IFS=, read -r suite tier kind case_id src in_file out_file enabled; do
  [[ "${suite}" == "suite" ]] && continue
  [[ "${suite}" == "${SUITE}" ]] || continue
  [[ "${enabled}" == "1" ]] || continue
  if [[ -n "${RUNTIME_CASE_FILTER}" ]] && [[ "${case_id}" != *"${RUNTIME_CASE_FILTER}"* ]]; then
    continue
  fi
  if [[ "${RUNTIME_CASE_LIMIT}" -gt 0 && "${total}" -ge "${RUNTIME_CASE_LIMIT}" ]]; then
    break
  fi

  total=$((total + 1))

  stem="$(safe_stem "${case_id}")"
  asm_path="${asm_root}/${stem}.s"
  exe_path="${bin_root}/${stem}"
  log_path="${log_root}/${stem}.log"
  rm -f "${asm_path}" "${exe_path}"

  status="ok"
  compare_status="skip"
  compile_status="not_run"
  asm_emit_status="not_run"
  link_status="not_run"
  pass=0
  warmup_ms=""
  run1_ms=""
  run2_ms=""
  run3_ms=""
  median_ms=""
  input_token_count="0"
  estimated_read_count="0"
  suspect_input_underflow="0"
  frontend_path="unknown"
  fallback_reason_codes="none"

  tmp_dir="$(mktemp -d "${log_root}/tmp.XXXXXX")"
  stdout_warm="${tmp_dir}/warm.out"
  stderr_warm="${tmp_dir}/warm.err"
  stdout_1="${tmp_dir}/run1.out"
  stderr_1="${tmp_dir}/run1.err"
  stdout_2="${tmp_dir}/run2.out"
  stderr_2="${tmp_dir}/run2.err"
  stdout_3="${tmp_dir}/run3.out"
  stderr_3="${tmp_dir}/run3.err"
  actual_file="${tmp_dir}/actual.out"
  dialect_report_file="${tmp_dir}/dialect.report"
  timeout_sec="${RUNTIME_TIMEOUT_SEC}"
  is_perf_case=0
  if [[ "${kind}" == "perf" || "${suite}" != "official-functional" ]]; then
    is_perf_case=1
    timeout_sec="${RUNTIME_PERF_TIMEOUT_SEC}"
  fi
  input_token_count="$(count_input_tokens "${in_file}")"
  estimated_read_count="$(estimate_scalar_reads "${src}")"
  if [[ "${estimated_read_count}" -gt "${input_token_count}" ]]; then
    suspect_input_underflow="1"
  fi

  echo "[case] ${case_id}" >"${log_path}"
  echo "[timeout] ${timeout_sec}s" >>"${log_path}"
  echo "[input_tokens] ${input_token_count}" >>"${log_path}"
  echo "[estimated_scalar_reads] ${estimated_read_count}" >>"${log_path}"
  echo "[suspect_input_underflow] ${suspect_input_underflow}" >>"${log_path}"
  echo "[compile] ${src}" >>"${log_path}"
  if [[ -n "${SISY_COMPILER_EXTRA_ARGS}" ]]; then
    echo "[compile_extra_args] ${SISY_COMPILER_EXTRA_ARGS}" >>"${log_path}"
  fi
  set +e
  compile_case "${src}" "${asm_path}" "${dialect_report_file}" >>"${log_path}" 2>&1
  compile_rc=$?
  set -e
  if [[ -f "${dialect_report_file}" ]]; then
    frontend_path="$(awk -F= '$1 == "frontend_path" { print $2; exit }' "${dialect_report_file}")"
    fallback_reason_codes="$(awk -F= '$1 == "fallback_reason_codes" { print $2; exit }' "${dialect_report_file}")"
  fi
  if [[ -z "${frontend_path}" ]]; then
    if [[ "${COMPILER_FLAVOR}" == "biframe" ]]; then
      frontend_path="legacy-forced"
    else
      frontend_path="unknown"
    fi
  fi
  if [[ -z "${fallback_reason_codes}" ]]; then
    fallback_reason_codes="none"
  fi
  echo "[frontend_path] ${frontend_path}" >>"${log_path}"
  echo "[fallback_reason_codes] ${fallback_reason_codes}" >>"${log_path}"
  if [[ "${compile_rc}" -ne 0 ]]; then
    if [[ "${compile_rc}" -ge 128 ]]; then
      status="compile_crash"
      compile_status="crash"
    else
      status="compile_fail"
      compile_status="fail"
    fi
    asm_emit_status="missing"
  else
    compile_status="ok"
    if [[ -s "${asm_path}" ]]; then
      asm_emit_status="ok"
    elif [[ -f "${asm_path}" ]]; then
      asm_emit_status="empty"
    else
      asm_emit_status="missing"
      status="compile_fail"
      compile_status="fail"
    fi
  fi

  if [[ "${status}" == "ok" ]]; then
    echo "[link] ${asm_path}" >>"${log_path}"
    set +e
    "${GCC_BIN}" -static "${asm_path}" "${RUNTIME_SYLIB_C}" -lm -o "${exe_path}" >>"${log_path}" 2>&1
    link_rc=$?
    set -e
    if [[ "${link_rc}" -ne 0 ]]; then
      status="link_fail"
      link_status="fail"
    else
      link_status="ok"
    fi
  fi

  if [[ "${status}" == "ok" ]]; then
    echo "[warmup]" >>"${log_path}"
    run_once "${exe_path}" "${in_file}" "${stdout_warm}" "${stderr_warm}" "${timeout_sec}"
    if [[ "${RUN_RC}" -eq 124 ]]; then
      status="timeout"
      echo "timeout in warmup" >>"${log_path}"
    else
      warmup_ms="$(ns_to_ms "${RUN_NS}")"
    fi
  fi

  if [[ "${status}" == "ok" ]]; then
    echo "[run] 1" >>"${log_path}"
    run_once "${exe_path}" "${in_file}" "${stdout_1}" "${stderr_1}" "${timeout_sec}"
    if [[ "${RUN_RC}" -eq 124 ]]; then
      status="timeout"
      echo "timeout in run1" >>"${log_path}"
    else
      rc1="${RUN_RC}"
      ns1="${RUN_NS}"
      run1_ms="$(ns_to_ms "${ns1}")"
    fi
  fi

  if [[ "${status}" == "ok" ]]; then
    echo "[run] 2" >>"${log_path}"
    run_once "${exe_path}" "${in_file}" "${stdout_2}" "${stderr_2}" "${timeout_sec}"
    if [[ "${RUN_RC}" -eq 124 ]]; then
      status="timeout"
      echo "timeout in run2" >>"${log_path}"
    else
      ns2="${RUN_NS}"
      run2_ms="$(ns_to_ms "${ns2}")"
    fi
  fi

  if [[ "${status}" == "ok" ]]; then
    echo "[run] 3" >>"${log_path}"
    run_once "${exe_path}" "${in_file}" "${stdout_3}" "${stderr_3}" "${timeout_sec}"
    if [[ "${RUN_RC}" -eq 124 ]]; then
      status="timeout"
      echo "timeout in run3" >>"${log_path}"
    else
      ns3="${RUN_NS}"
      run3_ms="$(ns_to_ms "${ns3}")"

      median_ns="$(printf '%s\n' "${ns1}" "${ns2}" "${ns3}" | sort -n | awk 'NR==2 { print; exit }')"
      median_ms="$(ns_to_ms "${median_ns}")"

      cp "${stdout_1}" "${actual_file}"
      if [[ -s "${actual_file}" ]]; then
        # Keep exactly one separator line between stdout and exit code.
        last_byte="$(tail -c 1 "${actual_file}" | od -An -t x1 | tr -d '[:space:]')"
        if [[ "${last_byte}" != "0a" ]]; then
          echo >>"${actual_file}"
        fi
      fi
      printf '%d\n' "${rc1}" >>"${actual_file}"

      if [[ -f "${out_file}" ]]; then
        compare_status="ok"
        expected_norm="$(normalize_text "${out_file}")"
        actual_norm="$(normalize_text "${actual_file}")"
        if [[ "${expected_norm}" != "${actual_norm}" ]]; then
          compare_status="fail"
          status="mismatch"
          {
            echo "[mismatch] exit_code(run1)=${rc1}"
            echo "[mismatch] expected(norm):"
            normalize_text "${out_file}"
            echo "[mismatch] actual(norm):"
            normalize_text "${actual_file}"
          } >>"${log_path}"
        fi
      else
        compare_status="no_out"
      fi
    fi
  fi

  if [[ "${status}" != "ok" && "${status}" != "mismatch" && "${status}" != "timeout" ]]; then
    compare_status="skip"
  fi

  if [[ "${status}" == "ok" && ( "${compare_status}" == "ok" || "${compare_status}" == "no_out" ) ]]; then
    pass=1
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
    if [[ "${RUNTIME_SOFT_PERF}" == "1" && "${is_perf_case}" == "1" ]]; then
      soft_fail_count=$((soft_fail_count + 1))
    else
      hard_fail_count=$((hard_fail_count + 1))
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${SUITE}" "${case_id}" "${TARGET}" "${OPT}" "${LABEL}" \
    "${status}" "${compare_status}" "${pass}" "${median_ms}" "${warmup_ms}" \
    "${run1_ms}" "${run2_ms}" "${run3_ms}" "${asm_path}" "${exe_path}" "${log_path}" \
    "${compile_status}" "${asm_emit_status}" "${link_status}" \
    "${input_token_count}" "${estimated_read_count}" "${suspect_input_underflow}" \
    "${frontend_path}" "${fallback_reason_codes}" \
    >>"${CSV_OUT}"

  rm -rf "${tmp_dir}"
done <"${INDEX_CSV}"

echo "csv: ${CSV_OUT}"
echo "summary: total=${total}, pass=${pass_count}, fail=${fail_count}, hard_fail=${hard_fail_count}, soft_fail=${soft_fail_count}"
if [[ "${hard_fail_count}" -ne 0 ]]; then
  exit 1
fi
if [[ "${RUNTIME_SOFT_PERF}" != "1" && "${fail_count}" -ne 0 ]]; then
  exit 1
fi
