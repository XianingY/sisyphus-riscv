#!/usr/bin/env bash
set -euo pipefail

TARGET_ARG="${1:-all}"
OPT_ARG="${2:-all}"

usage() {
  cat <<USAGE
usage: $0 [all|riscv|arm] [all|O0|O1|O2]

env:
  CUSTOM_MANIFEST              default: test/custom-sysy2022/manifest.csv
  CUSTOM_SUITE_ROOT            default: test/custom-sysy2022
  CUSTOM_RUNTIME_ROOT          default: tests/.out/custom-runtime
  CUSTOM_LABEL                 default: custom-sysy2022
  CUSTOM_CASE_FILTER           substring filter on case_id
  CUSTOM_CASE_LIMIT            max number of cases per target/opt (0=all)
  CUSTOM_TIMEOUT_SEC_DEFAULT   default runtime timeout for run_ok when manifest timeout missing (default: 10)
  CUSTOM_SOFT_FAIL             if 1, always exit 0 (report only)
  SISY_COMPILER_PATH           default: build/compiler
  SISY_COMPILER_EXTRA_ARGS     extra compiler args, shell-split
USAGE
}

if [[ "${TARGET_ARG}" == "-h" || "${TARGET_ARG}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${CUSTOM_MANIFEST:-${ROOT_DIR}/test/custom-sysy2022/manifest.csv}"
SUITE_ROOT="${CUSTOM_SUITE_ROOT:-${ROOT_DIR}/test/custom-sysy2022}"
OUT_ROOT="${CUSTOM_RUNTIME_ROOT:-${ROOT_DIR}/tests/.out/custom-runtime}"
LABEL="${CUSTOM_LABEL:-custom-sysy2022}"
COMPILER="${SISY_COMPILER_PATH:-${ROOT_DIR}/build/compiler}"
OFFICIAL_RUNTIME_ROOT="${OFFICIAL_RUNTIME_ROOT:-/home/wslootie/github/cpe/compiler2025}"
RUNTIME_SYLIB_C="${RUNTIME_SYLIB_C:-${OFFICIAL_RUNTIME_ROOT}/sylib.c}"
CUSTOM_CASE_FILTER="${CUSTOM_CASE_FILTER:-}"
CUSTOM_CASE_LIMIT="${CUSTOM_CASE_LIMIT:-0}"
CUSTOM_TIMEOUT_SEC_DEFAULT="${CUSTOM_TIMEOUT_SEC_DEFAULT:-10}"
CUSTOM_SOFT_FAIL="${CUSTOM_SOFT_FAIL:-0}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: missing manifest: ${MANIFEST}"
  exit 1
fi
if [[ ! -d "${SUITE_ROOT}" ]]; then
  echo "error: missing suite root: ${SUITE_ROOT}"
  exit 1
fi
if [[ ! -x "${COMPILER}" ]]; then
  echo "error: compiler not found: ${COMPILER}"
  exit 1
fi
if [[ ! -f "${RUNTIME_SYLIB_C}" ]]; then
  RUNTIME_SYLIB_C="${ROOT_DIR}/runtime/sylib.c"
fi
if [[ ! -f "${RUNTIME_SYLIB_C}" ]]; then
  echo "error: runtime source not found: ${RUNTIME_SYLIB_C}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

if [[ "${TARGET_ARG}" != "all" && "${TARGET_ARG}" != "riscv" && "${TARGET_ARG}" != "arm" ]]; then
  echo "error: target must be all|riscv|arm"
  exit 1
fi
if [[ "${OPT_ARG}" != "all" && "${OPT_ARG}" != "O0" && "${OPT_ARG}" != "O1" && "${OPT_ARG}" != "O2" ]]; then
  echo "error: opt must be all|O0|O1|O2"
  exit 1
fi

TARGETS=()
if [[ "${TARGET_ARG}" == "all" ]]; then
  TARGETS=(riscv arm)
else
  TARGETS=("${TARGET_ARG}")
fi

OPTS=()
if [[ "${OPT_ARG}" == "all" ]]; then
  OPTS=(O0 O1 O2)
else
  OPTS=("${OPT_ARG}")
fi

declare -a EXTRA_ARGS=()
if [[ -n "${SISY_COMPILER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${SISY_COMPILER_EXTRA_ARGS})
fi

normalize_text() {
  local f="$1"
  awk '{ sub(/[ \t\r]+$/, "", $0); print }' "${f}"
}

run_once() {
  local qemu_bin="$1"
  local exe="$2"
  local in_file="$3"
  local stdout_file="$4"
  local stderr_file="$5"
  local timeout_sec="$6"

  local start end rc
  start="$(date +%s%N)"
  set +e
  if [[ -n "${in_file}" && -f "${in_file}" ]]; then
    timeout "${timeout_sec}" "${qemu_bin}" "${exe}" <"${in_file}" >"${stdout_file}" 2>"${stderr_file}"
    rc=$?
  else
    timeout "${timeout_sec}" "${qemu_bin}" "${exe}" >"${stdout_file}" 2>"${stderr_file}"
    rc=$?
  fi
  set -e
  end="$(date +%s%N)"

  RUN_RC="${rc}"
  RUN_NS=$((end - start))
}

ns_to_ms() {
  local ns="$1"
  awk -v ns="${ns}" 'BEGIN { printf "%.3f", ns / 1000000.0 }'
}

safe_stem() {
  local id="$1"
  id="${id//\//__}"
  id="${id// /_}"
  id="${id//$'\t'/_}"
  printf "%s" "${id}"
}

overall_fail=0

for TARGET in "${TARGETS[@]}"; do
  if [[ "${TARGET}" == "riscv" ]]; then
    GCC_BIN="riscv64-linux-gnu-gcc"
    QEMU_BIN="qemu-riscv64-static"
  else
    GCC_BIN="aarch64-linux-gnu-gcc"
    QEMU_BIN="qemu-aarch64-static"
  fi

  for tool in "${GCC_BIN}" "${QEMU_BIN}" timeout awk sort; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "error: missing tool '${tool}'"
      exit 1
    fi
  done

  for OPT in "${OPTS[@]}"; do
    CSV_OUT="${OUT_ROOT}/${LABEL}-${TARGET}-${OPT}.csv"
    ASM_ROOT="${OUT_ROOT}/asm/${LABEL}/${TARGET}-${OPT}"
    BIN_ROOT="${OUT_ROOT}/bin/${LABEL}/${TARGET}-${OPT}"
    LOG_ROOT="${OUT_ROOT}/logs/${LABEL}/${TARGET}-${OPT}"
    mkdir -p "${ASM_ROOT}" "${BIN_ROOT}" "${LOG_ROOT}"

    printf 'case_id,tier,expect,target,opt,status,pass,compare,compile_status,link_status,run_rc,run_ms,timeout_sec,tags,input,output,src,asm,exe,log,frontend_path,fallback_reason_codes\n' >"${CSV_OUT}"

    total=0
    pass_cnt=0
    fail_cnt=0

    while IFS=, read -r case_id tier expect input_rel output_rel tags timeout_sec; do
      [[ "${case_id}" == "case_id" ]] && continue

      if [[ -n "${CUSTOM_CASE_FILTER}" && "${case_id}" != *"${CUSTOM_CASE_FILTER}"* ]]; then
        continue
      fi
      if [[ "${CUSTOM_CASE_LIMIT}" -gt 0 && "${total}" -ge "${CUSTOM_CASE_LIMIT}" ]]; then
        break
      fi
      total=$((total + 1))

      src="${SUITE_ROOT}/${tier}/${case_id}.sy"
      in_file=""
      out_file=""
      if [[ -n "${input_rel}" ]]; then
        in_file="${SUITE_ROOT}/${input_rel}"
      fi
      if [[ -n "${output_rel}" ]]; then
        out_file="${SUITE_ROOT}/${output_rel}"
      fi

      timeout_use="${timeout_sec}"
      if [[ -z "${timeout_use}" ]]; then
        timeout_use="${CUSTOM_TIMEOUT_SEC_DEFAULT}"
      fi

      stem="$(safe_stem "${case_id}")"
      asm_path="${ASM_ROOT}/${stem}.s"
      exe_path="${BIN_ROOT}/${stem}"
      log_path="${LOG_ROOT}/${stem}.log"
      tmp_dir="$(mktemp -d "${LOG_ROOT}/tmp.${stem}.XXXXXX")"
      dialect_report="${tmp_dir}/dialect.report"
      stdout_file="${tmp_dir}/run.out"
      stderr_file="${tmp_dir}/run.err"
      actual_file="${tmp_dir}/actual.out"

      rm -f "${asm_path}" "${exe_path}"

      status="ok"
      pass=0
      compare_status="skip"
      compile_status="not_run"
      link_status="not_run"
      run_rc=""
      run_ms=""
      frontend_path="unknown"
      fallback_reason_codes="none"

      {
        echo "[case] ${case_id}"
        echo "[target] ${TARGET}"
        echo "[opt] ${OPT}"
        echo "[expect] ${expect}"
      } >"${log_path}"

      cmd=("${COMPILER}" "${src}" -S -o "${asm_path}" "--target=${TARGET}" "-${OPT}" "--dialect-fallback-report=${dialect_report}")
      if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
      fi

      set +e
      "${cmd[@]}" >>"${log_path}" 2>&1
      c_rc=$?
      set -e

      if [[ -f "${dialect_report}" ]]; then
        frontend_path="$(awk -F= '$1 == "frontend_path" { print $2; exit }' "${dialect_report}")"
        fallback_reason_codes="$(awk -F= '$1 == "fallback_reason_codes" { print $2; exit }' "${dialect_report}")"
      fi
      [[ -z "${frontend_path}" ]] && frontend_path="unknown"
      [[ -z "${fallback_reason_codes}" ]] && fallback_reason_codes="none"

      if [[ "${expect}" == "compile_fail" ]]; then
        if [[ "${c_rc}" -ne 0 ]]; then
          status="compile_fail_expected"
          compile_status="fail"
          pass=1
        else
          status="unexpected_compile_success"
          compile_status="ok"
          pass=0
        fi
      else
        if [[ "${c_rc}" -ne 0 ]]; then
          if [[ "${c_rc}" -ge 128 ]]; then
            status="compile_crash"
            compile_status="crash"
          else
            status="compile_fail"
            compile_status="fail"
          fi
          pass=0
        else
          compile_status="ok"
          set +e
          "${GCC_BIN}" -static "${asm_path}" "${RUNTIME_SYLIB_C}" -lm -o "${exe_path}" >>"${log_path}" 2>&1
          l_rc=$?
          set -e
          if [[ "${l_rc}" -ne 0 ]]; then
            status="link_fail"
            link_status="fail"
            pass=0
          else
            link_status="ok"
            run_once "${QEMU_BIN}" "${exe_path}" "${in_file}" "${stdout_file}" "${stderr_file}" "${timeout_use}"
            run_rc="${RUN_RC}"
            run_ms="$(ns_to_ms "${RUN_NS}")"

            if [[ "${RUN_RC}" -eq 124 ]]; then
              status="timeout"
              pass=0
            elif [[ "${RUN_RC}" -ge 128 ]]; then
              status="runtime_crash"
              pass=0
            else
              if [[ -n "${out_file}" && -f "${out_file}" ]]; then
                cp "${stdout_file}" "${actual_file}"
                last_byte="$(tail -c 1 "${actual_file}" | od -An -t x1 | tr -d '[:space:]' || true)"
                if [[ "${last_byte}" != "0a" ]]; then
                  echo >>"${actual_file}"
                fi
                printf '%d\n' "${RUN_RC}" >>"${actual_file}"

                if [[ "$(normalize_text "${out_file}")" == "$(normalize_text "${actual_file}")" ]]; then
                  compare_status="ok"
                  status="ok"
                  pass=1
                else
                  compare_status="fail"
                  status="mismatch"
                  pass=0
                  {
                    echo "[mismatch] expected: ${out_file}"
                    normalize_text "${out_file}"
                    echo "[mismatch] actual:"
                    normalize_text "${actual_file}"
                  } >>"${log_path}"
                fi
              else
                compare_status="no_out"
                status="ok"
                pass=1
              fi
            fi
          fi
        fi
      fi

      if [[ "${pass}" -eq 1 ]]; then
        pass_cnt=$((pass_cnt + 1))
      else
        fail_cnt=$((fail_cnt + 1))
      fi

      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${case_id}" "${tier}" "${expect}" "${TARGET}" "${OPT}" "${status}" "${pass}" "${compare_status}" \
        "${compile_status}" "${link_status}" "${run_rc}" "${run_ms}" "${timeout_use}" "${tags}" \
        "${input_rel}" "${output_rel}" "${src}" "${asm_path}" "${exe_path}" "${log_path}" \
        "${frontend_path}" "${fallback_reason_codes}" >>"${CSV_OUT}"

      rm -rf "${tmp_dir}"
    done <"${MANIFEST}"

    echo "[custom] ${TARGET} ${OPT} csv=${CSV_OUT} total=${total} pass=${pass_cnt} fail=${fail_cnt}"
    if [[ "${fail_cnt}" -ne 0 ]]; then
      overall_fail=$((overall_fail + 1))
    fi
  done
done

if [[ "${CUSTOM_SOFT_FAIL}" == "1" ]]; then
  exit 0
fi

if [[ "${overall_fail}" -ne 0 ]]; then
  exit 1
fi
