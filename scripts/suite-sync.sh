#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/tests/external"
OFFICIAL_DIR="${SISY_OFFICIAL_SUITE_ROOT:-${ROOT_DIR}/test/official}"
LOCK_FILE="${OFFICIAL_DIR}/lock.json"
TMP_META="$(mktemp)"
TMP_ORPHAN="$(mktemp)"
SRC_ROOT_DEFAULT="/home/wslootie/github/cpe/compiler2025"
SRC_ROOT="${SISY_SUITE_SRC_ROOT:-${SRC_ROOT_DEFAULT}}"
UPDATE=0

cleanup() {
  rm -f "${TMP_META}" "${TMP_ORPHAN}"
}
trap cleanup EXIT

usage() {
  cat <<EOF
usage: $0 [--update] [--src-root <path>]

options:
  --update            force re-extract all suites
  --src-root <path>   ZIP source root (default: ${SRC_ROOT_DEFAULT})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --update)
      UPDATE=1
      shift
      ;;
    --src-root)
      if [[ $# -lt 2 ]]; then
        echo "error: --src-root requires a value"
        usage
        exit 1
      fi
      SRC_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "error: source root not found: ${SRC_ROOT}"
  exit 1
fi

mkdir -p "${OFFICIAL_DIR}"

# Clean up old non-official suites to prevent accidental mixing.
for legacy in \
  "${EXTERNAL_DIR}/open-test-cases" \
  "${EXTERNAL_DIR}/compiler-dev-test-cases" \
  "${EXTERNAL_DIR}/sysy-testsuit-collection"; do
  if [[ -e "${legacy}" ]]; then
    echo "[cleanup] remove legacy suite dir: ${legacy}"
    rm -rf "${legacy}"
  fi
done

for required_tool in python3; do
  if ! command -v "${required_tool}" >/dev/null 2>&1; then
    echo "error: missing required tool '${required_tool}'"
    exit 1
  fi
done

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${path}" | awk '{print $1}'
  else
    echo "error: missing sha256sum or shasum" >&2
    exit 1
  fi
}

file_size() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.getsize(sys.argv[1]))
PY
}

file_mtime_utc() {
  python3 - "$1" <<'PY'
import datetime
import os
import sys
mtime = os.path.getmtime(sys.argv[1])
print(datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"))
PY
}

extract_zip() {
  local zip_path="$1"
  local dst_dir="$2"
  python3 - "${zip_path}" "${dst_dir}" <<'PY'
import pathlib
import shutil
import sys
import zipfile

zip_path = pathlib.Path(sys.argv[1])
dst_dir = pathlib.Path(sys.argv[2])
if dst_dir.exists():
    shutil.rmtree(dst_dir)
dst_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path) as zf:
    infos = zf.infolist()
    parts = []
    for info in infos:
        name = info.filename.replace("\\", "/")
        clean = [p for p in pathlib.PurePosixPath(name).parts if p not in ("", ".")]
        if clean:
            parts.append(clean)

    strip_prefix = None
    first_parts = {p[0] for p in parts}
    if len(first_parts) == 1 and any(len(p) > 1 for p in parts):
        strip_prefix = next(iter(first_parts))

    for info in infos:
        name = info.filename.replace("\\", "/")
        clean = [p for p in pathlib.PurePosixPath(name).parts if p not in ("", ".")]
        if not clean:
            continue
        if strip_prefix and clean[0] == strip_prefix:
            clean = clean[1:]
        if not clean:
            continue
        target = dst_dir.joinpath(*clean)
        if not target.resolve().is_relative_to(dst_dir.resolve()):
            raise RuntimeError(f"unsafe zip path: {info.filename}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as out:
            shutil.copyfileobj(src, out)
PY
}

append_meta() {
  local suite="$1"
  local zip_name="$2"
  local expected_sy="$3"
  local expected_in="$4"
  local expected_out="$5"

  local zip_path="${SRC_ROOT}/${zip_name}"
  local dst_dir="${OFFICIAL_DIR}/${suite}"
  local stamp_file="${dst_dir}/.zip.sha256"

  if [[ ! -f "${zip_path}" ]]; then
    echo "error: missing ZIP: ${zip_path}"
    exit 1
  fi

  local sha size mtime
  sha="$(sha256_file "${zip_path}")"
  size="$(file_size "${zip_path}")"
  mtime="$(file_mtime_utc "${zip_path}")"

  local needs_extract=0
  if [[ "${UPDATE}" -eq 1 || ! -d "${dst_dir}" ]]; then
    needs_extract=1
  elif [[ ! -f "${stamp_file}" ]]; then
    needs_extract=1
  else
    local old_sha
    old_sha="$(cat "${stamp_file}")"
    if [[ "${old_sha}" != "${sha}" ]]; then
      needs_extract=1
    fi
  fi

  if [[ "${needs_extract}" -eq 1 ]]; then
    echo "[extract] ${suite} <= ${zip_name}"
    extract_zip "${zip_path}" "${dst_dir}"
    echo "${sha}" >"${stamp_file}"
  else
    echo "[keep] ${suite} (unchanged zip)"
  fi

  local sy_count in_count out_count
  sy_count="$(find "${dst_dir}" -type f -name '*.sy' | wc -l | tr -d ' ')"
  in_count="$(find "${dst_dir}" -type f -name '*.in' | wc -l | tr -d ' ')"
  out_count="$(find "${dst_dir}" -type f -name '*.out' | wc -l | tr -d ' ')"

  if [[ "${sy_count}" != "${expected_sy}" || "${in_count}" != "${expected_in}" || "${out_count}" != "${expected_out}" ]]; then
    if [[ "${needs_extract}" -eq 0 ]]; then
      echo "[repair] re-extract ${suite} due to count mismatch"
      extract_zip "${zip_path}" "${dst_dir}"
      echo "${sha}" >"${stamp_file}"
      sy_count="$(find "${dst_dir}" -type f -name '*.sy' | wc -l | tr -d ' ')"
      in_count="$(find "${dst_dir}" -type f -name '*.in' | wc -l | tr -d ' ')"
      out_count="$(find "${dst_dir}" -type f -name '*.out' | wc -l | tr -d ' ')"
    fi
  fi

  if [[ "${sy_count}" != "${expected_sy}" || "${in_count}" != "${expected_in}" || "${out_count}" != "${expected_out}" ]]; then
    echo "error: count mismatch for ${suite}: got sy=${sy_count},in=${in_count},out=${out_count}; expected sy=${expected_sy},in=${expected_in},out=${expected_out}"
    exit 1
  fi

  local orphan_file="${OFFICIAL_DIR}/orphan_io_${suite}.txt"
  python3 - "${dst_dir}" "${orphan_file}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
out_file = pathlib.Path(sys.argv[2])
sy = set()
io = set()
for p in root.rglob("*"):
    if not p.is_file():
        continue
    suf = p.suffix.lower()
    if suf == ".sy":
        sy.add(p.with_suffix("").relative_to(root).as_posix())
    elif suf in (".in", ".out"):
        io.add(p.with_suffix("").relative_to(root).as_posix())
orphans = sorted(x for x in io if x not in sy)
out_file.write_text("\n".join(orphans) + ("\n" if orphans else ""), encoding="utf-8")
PY
  local orphan_count
  orphan_count="$(awk 'NF{c++} END{print c+0}' "${orphan_file}")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${suite}" "${zip_name}" "${zip_path}" "${sha}" "${size}" "${mtime}" \
    "${sy_count}" "${in_count}" "${out_count}" "${orphan_count}" >>"${TMP_META}"
  printf '%s\t%s\n' "${suite}" "${orphan_file}" >>"${TMP_ORPHAN}"
}

append_meta "official-functional" "functional.zip" "140" "44" "140"
append_meta "official-arm-perf" "ARM-性能.zip" "59" "55" "59"
append_meta "official-riscv-perf" "RISCV-性能.zip" "59" "55" "59"
append_meta "official-arm-final-perf" "ARM决赛性能用例.zip" "60" "60" "60"
append_meta "official-riscv-final-perf" "RISCV决赛性能用例.zip" "60" "63" "63"

python3 - "${TMP_META}" "${TMP_ORPHAN}" "${LOCK_FILE}" "${SRC_ROOT}" <<'PY'
import datetime
import json
import pathlib
import sys

meta_path = pathlib.Path(sys.argv[1])
orphan_map_path = pathlib.Path(sys.argv[2])
lock_path = pathlib.Path(sys.argv[3])
src_root = pathlib.Path(sys.argv[4]).resolve().as_posix()

orphans_by_suite = {}
for line in orphan_map_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    suite, orphan_file = line.split("\t", 1)
    p = pathlib.Path(orphan_file)
    items = []
    if p.exists():
      items = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    orphans_by_suite[suite] = items

entries = []
for line in meta_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    suite, zip_name, zip_path, sha, size, mtime, sy, in_count, out_count, orphan_count = line.split("\t")
    entries.append({
        "suite": suite,
        "zip_name": zip_name,
        "zip_path": pathlib.Path(zip_path).resolve().as_posix(),
        "sha256": sha,
        "size": int(size),
        "mtime_utc": mtime,
        "counts": {
            "sy": int(sy),
            "in": int(in_count),
            "out": int(out_count),
        },
        "orphan_io_count": int(orphan_count),
        "orphan_io": orphans_by_suite.get(suite, []),
    })

payload = {
    "generated_at": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "source_root": src_root,
    "suites": entries,
}

lock_path.parent.mkdir(parents=True, exist_ok=True)
lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "wrote ${LOCK_FILE}"
