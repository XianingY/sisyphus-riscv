#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_IMAGE="${SISY_DOCKER_IMAGE:-sisyphus/compiler-dev-dual:latest}"
COMPILER_PATH="${SISY_COMPILER_PATH:-${ROOT_DIR}/build-linux/compiler}"
OUT_DIR="${VERIFY_OUT_DIR:-${ROOT_DIR}/tests/.out/official-path-hashes}"
CASE_FILTER="${VERIFY_CASE_FILTER:-}"
RUN_QEMU="${VERIFY_RUN_QEMU:-1}"
RUNTIME_TIMEOUT_SEC="${RUNTIME_TIMEOUT_SEC:-10}"
RUNTIME_PERF_TIMEOUT_SEC="${RUNTIME_PERF_TIMEOUT_SEC:-60}"
BASELINE_CSV="${VERIFY_BASELINE_CSV:-}"
RUNTIME_SYLIB_C="${RUNTIME_SYLIB_C:-${ROOT_DIR}/runtime/sylib.c}"

need_runtime_tools() {
  [[ "${RUN_QEMU}" == "0" ]] && return 1
  command -v riscv64-linux-gnu-gcc >/dev/null 2>&1 &&
    command -v qemu-riscv64-static >/dev/null 2>&1 &&
    command -v timeout >/dev/null 2>&1
}

if [[ "${SISY_VERIFY_IN_DOCKER:-0}" != "1" ]] && ! need_runtime_tools; then
  if command -v docker >/dev/null 2>&1; then
    docker run --rm \
      --user "$(id -u):$(id -g)" \
      -e SISY_VERIFY_IN_DOCKER=1 \
      -e SISY_COMPILER_PATH="${COMPILER_PATH}" \
      -e VERIFY_OUT_DIR="${OUT_DIR}" \
      -e VERIFY_CASE_FILTER="${CASE_FILTER}" \
      -e VERIFY_RUN_QEMU="${RUN_QEMU}" \
      -e RUNTIME_TIMEOUT_SEC="${RUNTIME_TIMEOUT_SEC}" \
      -e RUNTIME_PERF_TIMEOUT_SEC="${RUNTIME_PERF_TIMEOUT_SEC}" \
      -e VERIFY_BASELINE_CSV="${BASELINE_CSV}" \
      -e RUNTIME_SYLIB_C="${RUNTIME_SYLIB_C}" \
      -v "${ROOT_DIR}:${ROOT_DIR}" \
      -w "${ROOT_DIR}" \
      "${DOCKER_IMAGE}" \
      bash -lc "scripts/verify_official_paths_and_hashes.sh"
    exit $?
  fi
  RUN_QEMU=0
fi

if [[ ! -x "${COMPILER_PATH}" ]]; then
  echo "error: compiler not found at ${COMPILER_PATH}" >&2
  exit 1
fi
if [[ "${SISY_VERIFY_IN_DOCKER:-0}" == "1" ]]; then
  magic="$(od -An -tx1 -N4 "${COMPILER_PATH}" 2>/dev/null | tr -d ' \n' || true)"
  if [[ "${magic}" != "7f454c46" ]]; then
    echo "error: ${COMPILER_PATH} is not a Linux ELF executable inside Docker" >&2
    exit 1
  fi
fi

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

normalize_text() {
  awk '{ sub(/[ \t\r]+$/, "", $0); print }' "$1"
}

ns_to_ms() {
  awk -v ns="$1" 'BEGIN { printf "%.3f", ns / 1000000.0 }'
}

bootstrap_suite_root() {
  local suite_root="$1"
  mkdir -p \
    "${suite_root}/official-functional" \
    "${suite_root}/official-arm-perf" \
    "${suite_root}/official-riscv-perf" \
    "${suite_root}/official-arm-final-perf" \
    "${suite_root}/official-riscv-final-perf"
  if [[ ! -e "${suite_root}/official-functional/.sisy-source" ]]; then
    rm -rf "${suite_root}/official-functional"
    cp -R "${ROOT_DIR}/test2026/riscv_func" "${suite_root}/official-functional"
    : >"${suite_root}/official-functional/.sisy-source"
  fi
  if [[ ! -e "${suite_root}/official-riscv-perf/.sisy-source" ]]; then
    rm -rf "${suite_root}/official-riscv-perf"
    cp -R "${ROOT_DIR}/test2026/performance_riscv" "${suite_root}/official-riscv-perf"
    : >"${suite_root}/official-riscv-perf/.sisy-source"
  fi
}

SUITE_ROOT="${SISY_OFFICIAL_SUITE_ROOT:-${ROOT_DIR}/test/official}"
if [[ ! -d "${SUITE_ROOT}/official-functional" ||
      ! -d "${SUITE_ROOT}/official-riscv-perf" ]]; then
  SUITE_ROOT="${ROOT_DIR}/tests/.out/test2026-official-path-suite-root"
  bootstrap_suite_root "${SUITE_ROOT}"
fi

mkdir -p "${OUT_DIR}/asm/functional" "${OUT_DIR}/asm/perf" \
         "${OUT_DIR}/bin/functional" "${OUT_DIR}/bin/perf" \
         "${OUT_DIR}/logs/functional" "${OUT_DIR}/logs/perf" \
         "${OUT_DIR}/stats/perf"

REPORT="${OUT_DIR}/report.csv"
SUMMARY="${OUT_DIR}/summary.txt"
COMPILER_SHA="$(hash_file "${COMPILER_PATH}")"
COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ "${COMMIT}" != "unknown" ]] &&
   ! git -C "${ROOT_DIR}" diff --quiet --ignore-submodules --; then
  COMMIT="${COMMIT}-dirty"
fi

declare -A BASELINE_HASHES=()
if [[ -n "${BASELINE_CSV}" && -f "${BASELINE_CSV}" ]]; then
  while IFS=, read -r suite case_id _rest; do
    [[ "${suite}" == "suite" ]] && continue
    asm_hash="$(awk -F, -v s="${suite}" -v c="${case_id}" \
      'NR > 1 && $1 == s && $2 == c { print $6; exit }' "${BASELINE_CSV}")"
    [[ -n "${asm_hash}" ]] && BASELINE_HASHES["${suite}/${case_id}"]="${asm_hash}"
  done <"${BASELINE_CSV}"
fi

printf 'suite,case_id,official_cmd,status,compare,asm_hash,baseline_asm_hash,asm_changed,median_ms,compiler_sha,commit,asm_path,stats_path,log_path\n' >"${REPORT}"

run_once() {
  local exe="$1" in_file="$2" out_file="$3" timeout_sec="$4"
  local start end rc
  start="$(date +%s%N)"
  set +e
  if [[ -f "${in_file}" ]]; then
    timeout "${timeout_sec}" qemu-riscv64-static "${exe}" <"${in_file}" >"${out_file}" 2>/dev/null
    rc=$?
  else
    timeout "${timeout_sec}" qemu-riscv64-static "${exe}" >"${out_file}" 2>/dev/null
    rc=$?
  fi
  set -e
  end="$(date +%s%N)"
  RUN_RC="${rc}"
  RUN_NS=$((end - start))
}

emit_case() {
  local suite="$1" root="$2" kind="$3" src="$4"
  local rel case_id asm_path exe_path log_path stats_path official_cmd status compare median_ms asm_hash baseline changed
  rel="${src#${root}/}"
  case_id="${rel%.sy}"
  if [[ -n "${CASE_FILTER}" && "${case_id}" != *"${CASE_FILTER}"* ]]; then
    return
  fi
  asm_path="${OUT_DIR}/asm/${kind}/${case_id//\//__}.s"
  exe_path="${OUT_DIR}/bin/${kind}/${case_id//\//__}"
  log_path="${OUT_DIR}/logs/${kind}/${case_id//\//__}.log"
  stats_path=""
  mkdir -p "$(dirname "${asm_path}")" "$(dirname "${exe_path}")" "$(dirname "${log_path}")"
  status="ok"
  compare="skip"
  median_ms=""
  if [[ "${kind}" == "perf" ]]; then
    official_cmd='compiler -S -o testcase.s testcase.sy -O1'
    "${COMPILER_PATH}" -S -o "${asm_path}" "${src}" -O1 >"${log_path}" 2>&1 || status="compile_fail"
    stats_path="${OUT_DIR}/stats/perf/${case_id//\//__}.stats"
    "${COMPILER_PATH}" -S -o /dev/null "${src}" -O1 --stats >"${stats_path}.out" 2>"${stats_path}" || true
  else
    official_cmd='compiler -S -o testcase.s testcase.sy'
    "${COMPILER_PATH}" -S -o "${asm_path}" "${src}" >"${log_path}" 2>&1 || status="compile_fail"
  fi

  if [[ "${status}" == "ok" ]]; then
    asm_hash="$(hash_file "${asm_path}")"
    baseline="${BASELINE_HASHES["${suite}/${case_id}"]:-}"
    if [[ -z "${baseline}" ]]; then
      changed="unknown"
    elif [[ "${baseline}" == "${asm_hash}" ]]; then
      changed="0"
    else
      changed="1"
    fi
  else
    asm_hash=""
    baseline="${BASELINE_HASHES["${suite}/${case_id}"]:-}"
    changed="unknown"
  fi

  if [[ "${status}" == "ok" && "${RUN_QEMU}" == "1" ]]; then
    if riscv64-linux-gnu-gcc -static "${asm_path}" "${RUNTIME_SYLIB_C}" -lm -o "${exe_path}" >>"${log_path}" 2>&1; then
      local in_file="${src%.sy}.in"
      local expected="${src%.sy}.out"
      local out1="${log_path}.run1.out" out2="${log_path}.run2.out" out3="${log_path}.run3.out"
      local timeout_sec="${RUNTIME_TIMEOUT_SEC}"
      [[ "${kind}" == "perf" ]] && timeout_sec="${RUNTIME_PERF_TIMEOUT_SEC}"
      run_once "${exe_path}" "${in_file}" "${out1}" "${timeout_sec}"
      if [[ "${RUN_RC}" == "124" ]]; then
        status="timeout"
      else
        local ns1="${RUN_NS}"
        run_once "${exe_path}" "${in_file}" "${out2}" "${timeout_sec}"
        local ns2="${RUN_NS}"
        run_once "${exe_path}" "${in_file}" "${out3}" "${timeout_sec}"
        local ns3="${RUN_NS}"
        local median_ns
        median_ns="$(printf '%s\n' "${ns1}" "${ns2}" "${ns3}" | sort -n | awk 'NR == 2 { print }')"
        median_ms="$(ns_to_ms "${median_ns}")"
        if [[ -f "${expected}" ]]; then
          local actual="${log_path}.actual"
          cp "${out1}" "${actual}"
          if [[ -s "${actual}" ]]; then
            last_byte="$(tail -c 1 "${actual}" | od -An -t x1 | tr -d '[:space:]')"
            [[ "${last_byte}" == "0a" ]] || echo >>"${actual}"
          fi
          printf '%d\n' "${RUN_RC}" >>"${actual}"
          if [[ "$(normalize_text "${expected}")" == "$(normalize_text "${actual}")" ]]; then
            compare="ok"
          else
            compare="fail"
            status="mismatch"
          fi
        fi
      fi
    else
      status="link_fail"
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${suite}" "${case_id}" "${official_cmd}" "${status}" "${compare}" \
    "${asm_hash}" "${baseline}" "${changed}" "${median_ms}" \
    "${COMPILER_SHA}" "${COMMIT}" "${asm_path}" "${stats_path}" "${log_path}" >>"${REPORT}"
}

functional_root="${SUITE_ROOT}/official-functional"
perf_root="${SUITE_ROOT}/official-riscv-perf"

while IFS= read -r src; do
  emit_case "official-functional" "${functional_root}" "functional" "${src}"
done < <(find "${functional_root}" -type f -name '*.sy' | LC_ALL=C sort)

while IFS= read -r src; do
  emit_case "official-riscv-perf" "${perf_root}" "perf" "${src}"
done < <(find "${perf_root}" -type f -name '*.sy' | LC_ALL=C sort)

{
  echo "commit=${COMMIT}"
  echo "compiler=${COMPILER_PATH}"
  echo "compiler_sha=${COMPILER_SHA}"
  echo "report=${REPORT}"
  awk -F, '
    NR > 1 {
      total[$1]++;
      if ($4 == "ok") ok[$1]++;
      if ($4 != "ok") fail[$1]++;
      if ($8 == "0") unchanged[$1]++;
      if ($8 == "1") changed[$1]++;
      if ($9 != "") median[$1] += $9;
    }
    END {
      for (s in total)
        printf "%s total=%d ok=%d fail=%d asm_changed=%d asm_unchanged=%d median_sum_ms=%.3f\n",
          s, total[s], ok[s] + 0, fail[s] + 0, changed[s] + 0,
          unchanged[s] + 0, median[s] + 0;
    }
  ' "${REPORT}"
} | tee "${SUMMARY}"
