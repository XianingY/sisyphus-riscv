#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${RUNTIME_ROOT:-${ROOT_DIR}/tests/.out/runtime}"
REPORT_ROOT="${RISCV_PERF_REPORT_ROOT:-${ROOT_DIR}/tests/.out/riscv-perf-regression}"
STAMP="${RISCV_PERF_STAMP:-$(date +%Y%m%d-%H%M%S)}"
LABEL="${RUNTIME_LABEL:-riscv-perf-${STAMP}}"
OPT="${RISCV_PERF_OPT:-O1}"
PERF_TIMEOUT_SEC="${RUNTIME_PERF_TIMEOUT_SEC:-60}"
FULL_CSV="${REPORT_ROOT}/${LABEL}-official-riscv-perf-${OPT}.csv"
HOTSPOT_SUMMARY=""

mkdir -p "${REPORT_ROOT}" "${OUT_ROOT}"

bootstrap_test2026_perf_suite() {
  if [[ -n "${SISY_OFFICIAL_SUITE_ROOT:-}" ]]; then
    return
  fi
  if [[ -d "${ROOT_DIR}/test/official" || ! -d "${ROOT_DIR}/test2026/performance" ]]; then
    return
  fi

  export SISY_OFFICIAL_SUITE_ROOT="${ROOT_DIR}/tests/.out/test2026-perf-suite-root"
  mkdir -p \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-functional" \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-arm-perf" \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf" \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-arm-final-perf" \
    "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-final-perf"
  if [[ ! -e "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf/.sisy-source" ]]; then
    rm -rf "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf"
    cp -R "${ROOT_DIR}/test2026/performance" "${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf"
    : >"${SISY_OFFICIAL_SUITE_ROOT}/official-riscv-perf/.sisy-source"
  fi
}

usage() {
  cat <<'USAGE'
usage: scripts/run_riscv_perf_regression.sh [--skip-docker-build] [--hotspots-only] [--full-only] [--baseline <csv>]

Runs the RISC-V performance regression gate in a comparable way:
  1. optionally rebuilds build/compiler inside the project Docker image
  2. runs scripts/eval-riscv-hotspots.sh
  3. runs the full official-riscv-perf riscv O1 suite
  4. prints the slowest 15 cases and an optional CSV-vs-CSV delta

Environment:
  RISCV_PERF_OPT=O1|O2
  RUNTIME_LABEL=<label>
  RUNTIME_PERF_TIMEOUT_SEC=<seconds>
  RISCV_PERF_BASELINE=<csv>
USAGE
}

SKIP_DOCKER_BUILD="${SISY_SKIP_DOCKER_BUILD:-0}"
RUN_HOTSPOTS=1
RUN_FULL=1
BASELINE_CSV="${RISCV_PERF_BASELINE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-docker-build)
      SKIP_DOCKER_BUILD=1
      shift
      ;;
    --hotspots-only)
      RUN_FULL=0
      shift
      ;;
    --full-only)
      RUN_HOTSPOTS=0
      shift
      ;;
    --baseline)
      if [[ $# -lt 2 ]]; then
        echo "error: --baseline requires a csv path" >&2
        exit 1
      fi
      BASELINE_CSV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

bootstrap_test2026_perf_suite

if [[ "${SKIP_DOCKER_BUILD}" != "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "[riscv-perf] docker not found; skipping Docker rebuild" >&2
  else
    IMAGE="${SISY_DOCKER_IMAGE:-sisyphus/compiler-dev-dual:latest}"
    DOCKERFILE="${ROOT_DIR}/docker/compiler-dev-dual.Dockerfile"
    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
      docker build -t "${IMAGE}" -f "${DOCKERFILE}" "${ROOT_DIR}"
    fi
    echo "[riscv-perf] rebuild Linux compiler in ${IMAGE}"
    docker run --rm \
      --user "$(id -u):$(id -g)" \
      -v "${ROOT_DIR}:${ROOT_DIR}" \
      -w "${ROOT_DIR}" \
      "${IMAGE}" \
      bash -lc 'scripts/build.sh'
  fi
fi

if [[ "${RUN_HOTSPOTS}" == "1" ]]; then
  echo "[riscv-perf] hotspots ${OPT}"
  RUNTIME_LABEL="${LABEL}-hotspot" scripts/eval-riscv-hotspots.sh "${OPT}"
  HOTSPOT_SUMMARY="$(ls -t "${ROOT_DIR}"/tests/.out/riscv-hotspots/summary-"${OPT}"-*.txt 2>/dev/null | head -1 || true)"
fi

if [[ "${RUN_FULL}" == "1" ]]; then
  echo "[riscv-perf] full official-riscv-perf riscv ${OPT}"
  RUNTIME_LABEL="${LABEL}" \
  RUNTIME_CSV="${FULL_CSV}" \
  RUNTIME_PERF_TIMEOUT_SEC="${PERF_TIMEOUT_SEC}" \
  RUNTIME_SOFT_PERF="${RUNTIME_SOFT_PERF:-1}" \
    scripts/eval-runtime.sh official-riscv-perf riscv "${OPT}"
fi

if [[ -f "${FULL_CSV}" ]]; then
  echo
  echo "[riscv-perf] slowest 15 cases in ${FULL_CSV}"
  awk -F, 'NR > 1 {
      printf "%10.3f ms  %-28s status=%s compare=%s asm=%s\n",
             $9 + 0, $2, $6, $7, $14
    }' "${FULL_CSV}" | sort -nr | head -15
fi

if [[ -n "${HOTSPOT_SUMMARY}" && -f "${HOTSPOT_SUMMARY}" ]]; then
  echo
  echo "[riscv-perf] hotspot summary: ${HOTSPOT_SUMMARY}"
  sed -n '1,120p' "${HOTSPOT_SUMMARY}"
fi

if [[ -n "${BASELINE_CSV}" ]]; then
  if [[ ! -f "${BASELINE_CSV}" ]]; then
    echo "error: baseline csv not found: ${BASELINE_CSV}" >&2
    exit 1
  fi
  if [[ ! -f "${FULL_CSV}" ]]; then
    echo "error: current csv not found: ${FULL_CSV}" >&2
    exit 1
  fi
  echo
  echo "[riscv-perf] delta vs ${BASELINE_CSV}"
  awk -F, '
    NR == FNR {
      if (FNR > 1) base[$2] = $9 + 0;
      next;
    }
    FNR > 1 && ($2 in base) {
      cur = $9 + 0;
      old = base[$2];
      if (old > 0) {
        delta = cur - old;
        pct = delta * 100.0 / old;
        printf "%+10.3f ms  %+8.2f%%  %-28s old=%8.3f current=%8.3f status=%s compare=%s\n",
               delta, pct, $2, old, cur, $6, $7;
      }
    }
  ' "${BASELINE_CSV}" "${FULL_CSV}" | sort -nr | head -30
fi

echo
echo "[riscv-perf] report root: ${REPORT_ROOT}"
