#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/test/custom-sysy2022"
MANIFEST="${OUT_ROOT}/manifest.csv"

rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/positive" "${OUT_ROOT}/stress" "${OUT_ROOT}/negative"

printf 'case_id,tier,expect,input,output,tags,timeout_sec\n' >"${MANIFEST}"

idx=1
pos_count=0
str_count=0
neg_count=0

next_id() {
  local theme="$1"
  printf 'c%03d_%s' "${idx}" "${theme}"
}

add_manifest() {
  local case_id="$1"
  local tier="$2"
  local expect="$3"
  local input_rel="$4"
  local output_rel="$5"
  local tags="$6"
  local timeout_sec="$7"
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "${case_id}" "${tier}" "${expect}" "${input_rel}" "${output_rel}" "${tags}" "${timeout_sec}" >>"${MANIFEST}"
}

emit_run_case() {
  local tier="$1"
  local theme="$2"
  local tags="$3"
  local timeout_sec="$4"
  local input_data="$5"
  local expected_stdout="$6"

  local case_id
  case_id="$(next_id "${theme}")"
  local dir="${OUT_ROOT}/${tier}"
  local sy="${dir}/${case_id}.sy"
  local in_rel=""
  local out_rel="${tier}/${case_id}.out"

  cat >"${sy}"

  if [[ -n "${input_data}" ]]; then
    printf '%b' "${input_data}" >"${dir}/${case_id}.in"
    in_rel="${tier}/${case_id}.in"
  fi

  if [[ -n "${expected_stdout}" ]]; then
    printf '%b\n0\n' "${expected_stdout}" >"${dir}/${case_id}.out"
  else
    printf '0\n' >"${dir}/${case_id}.out"
  fi

  add_manifest "${case_id}" "${tier}" "run_ok" "${in_rel}" "${out_rel}" "${tags}" "${timeout_sec}"

  if [[ "${tier}" == "positive" ]]; then
    pos_count=$((pos_count + 1))
  elif [[ "${tier}" == "stress" ]]; then
    str_count=$((str_count + 1))
  fi

  idx=$((idx + 1))
}

emit_fail_case() {
  local theme="$1"
  local tags="$2"
  local timeout_sec="$3"

  local case_id
  case_id="$(next_id "${theme}")"
  local dir="${OUT_ROOT}/negative"
  local sy="${dir}/${case_id}.sy"

  cat >"${sy}"

  add_manifest "${case_id}" "negative" "compile_fail" "" "" "${tags}" "${timeout_sec}"

  neg_count=$((neg_count + 1))
  idx=$((idx + 1))
}

# ------------------------------
# positive: expr / precedence / short-circuit (14)
# ------------------------------
for i in $(seq 1 14); do
  a=$((i + 2))
  b=$((i % 5 + 3))
  c=$((a + b * 2 - (a % b)))
  g=0
  x=0
  if ((a < b)); then
    g=$((g + 1))
    x=$((x + 10))
  fi
  if ((a > b)); then
    x=$((x + 20))
  else
    g=$((g + 1))
    x=$((x + 20))
  fi
  ans=$((c + x + g))

  emit_run_case positive "expr_sc_${i}" "expr|precedence|short_circuit" 8 "" "${ans}" <<SRC
int g = 0;
int tick() {
  g = g + 1;
  return g;
}
int main() {
  int a = ${a};
  int b = ${b};
  int c = a + b * 2 - a % b;
  int x = 0;
  if ((a < b) && tick()) x = x + 10;
  if ((a > b) || tick()) x = x + 20;
  putint(c + x + g);
  putch(10);
  return 0;
}
SRC
done

# ------------------------------
# positive: int/float conversion & compare (14)
# ------------------------------
for i in $(seq 1 14); do
  n=$((i + 1))
  m=$((n % 3))
  x4=$((n * 2))
  y4=$((m * 4 + 1))
  ans=$((x4 * 2))
  if ((x4 > y4)); then
    ans=$((ans + 1))
  fi
  if ((x4 <= y4)); then
    ans=$((ans + 2))
  fi

  emit_run_case positive "float_cmp_${i}" "int_float|cast|compare" 8 "" "${ans}" <<SRC
int main() {
  float x = ${n} * 0.5;
  float y = ${m} + 0.25;
  int r = x * 8.0;
  if (x > y) r = r + 1;
  if (x <= y) r = r + 2;
  putint(r);
  putch(10);
  return 0;
}
SRC
done

# ------------------------------
# positive: scope / shadow / constexpr (10)
# ------------------------------
for i in $(seq 1 10); do
  base=$((i + 3))
  ans=$((4 * base + 4))

  emit_run_case positive "scope_shadow_${i}" "scope|shadow|constexpr" 8 "" "${ans}" <<SRC
const int A = ${base};
int main() {
  int x = A;
  {
    const int A = ${base} + 1;
    x = x + A;
    {
      int A = ${base} + 2;
      x = x + A;
    }
    x = x + A;
  }
  putint(x);
  putch(10);
  return 0;
}
SRC
done

# ------------------------------
# positive: array dims/init/zero-fill/param-array (18)
# ------------------------------
for i in $(seq 1 18); do
  mode=$(( (i - 1) % 3 ))
  if ((mode == 0)); then
    v=$((i + 1))
    ans=$((2 * v + 1))
    emit_run_case positive "array_init1_${i}" "array|init|zero_fill" 8 "" "${ans}" <<SRC
int main() {
  int a[5] = {${v}, ${v} + 1};
  int s = 0;
  int i = 0;
  while (i < 5) {
    s = s + a[i];
    i = i + 1;
  }
  putint(s);
  putch(10);
  return 0;
}
SRC
  elif ((mode == 1)); then
    v=$((i + 2))
    ans=$((3 * v + 3))
    emit_run_case positive "array_init2_${i}" "array|multidim|zero_fill" 8 "" "${ans}" <<SRC
int main() {
  int a[2][3] = {{${v}, ${v} + 1}, {${v} + 2}};
  int s = 0;
  int i = 0;
  while (i < 2) {
    int j = 0;
    while (j < 3) {
      s = s + a[i][j];
      j = j + 1;
    }
    i = i + 1;
  }
  putint(s);
  putch(10);
  return 0;
}
SRC
  else
    v=$((i + 1))
    ans=$((3 * v + 4))
    emit_run_case positive "array_param_${i}" "array|param_array|stride" 8 "" "${ans}" <<SRC
int sum2d(int a[][4], int n) {
  int s = 0;
  int i = 0;
  while (i < n) {
    int j = 0;
    while (j < 4) {
      s = s + a[i][j];
      j = j + 1;
    }
    i = i + 1;
  }
  return s;
}
int main() {
  int b[2][4] = {{1, ${v}, 0, ${v} + 1}, {${v} + 2}};
  putint(sum2d(b, 2));
  putch(10);
  return 0;
}
SRC
  fi
done

# ------------------------------
# positive: function call/return/recursion/many-params (10)
# ------------------------------
for i in $(seq 1 10); do
  mode=$(( (i - 1) % 3 ))
  if ((mode == 0)); then
    n=$((i % 5 + 3))
    fact=1
    for k in $(seq 1 "${n}"); do
      fact=$((fact * k))
    done
    emit_run_case positive "func_rec_${i}" "function|recursion|return" 8 "" "${fact}" <<SRC
int fact(int n) {
  if (n <= 1) return 1;
  return n * fact(n - 1);
}
int main() {
  putint(fact(${n}));
  putch(10);
  return 0;
}
SRC
  elif ((mode == 1)); then
    v=$((i + 3))
    ans=$((12 * v + 66))
    emit_run_case positive "func_many_${i}" "function|many_params|call" 8 "" "${ans}" <<SRC
int sum12(int a0,int a1,int a2,int a3,int a4,int a5,int a6,int a7,int a8,int a9,int a10,int a11) {
  return a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11;
}
int main() {
  int v = ${v};
  putint(sum12(v,v+1,v+2,v+3,v+4,v+5,v+6,v+7,v+8,v+9,v+10,v+11));
  putch(10);
  return 0;
}
SRC
  else
    v=$((i + 2))
    ans=$((3 * v + 1))
    emit_run_case positive "func_chain_${i}" "function|call_chain" 8 "" "${ans}" <<SRC
int f3(int x) { return x + 1; }
int f2(int x) { return f3(x) + x; }
int f1(int x) { return f2(x) + x; }
int main() {
  putint(f1(${v}));
  putch(10);
  return 0;
}
SRC
  fi
done

# ------------------------------
# positive: library behaviors (6)
# ------------------------------
emit_run_case positive "lib_getint" "libfunc|getint" 8 $'3 4\n' "10" <<'SRC'
int main() {
  int a = getint();
  int b = getint();
  putint(a * 2 + b);
  putch(10);
  return 0;
}
SRC

emit_run_case positive "lib_getch" "libfunc|getch" 8 $'Az\n' "187" <<'SRC'
int main() {
  int a = getch();
  int b = getch();
  putint(a + b);
  putch(10);
  return 0;
}
SRC

emit_run_case positive "lib_getarray" "libfunc|getarray|array" 8 $'5 1 2 3 4 5\n' "15" <<'SRC'
int main() {
  int a[8];
  int n = getarray(a);
  int s = 0;
  int i = 0;
  while (i < n) {
    s = s + a[i];
    i = i + 1;
  }
  putint(s);
  putch(10);
  return 0;
}
SRC

emit_run_case positive "lib_getfloat" "libfunc|getfloat|float" 8 $'1.5\n' "6" <<'SRC'
int main() {
  float x = getfloat();
  int y = x * 4.0;
  putint(y);
  putch(10);
  return 0;
}
SRC

emit_run_case positive "lib_getfarray" "libfunc|getfarray|float_array" 8 $'3 0.5 1.0 1.5\n' "15" <<'SRC'
int main() {
  float a[8];
  int n = getfarray(a);
  int s = n;
  int i = 0;
  while (i < n) {
    s = s + a[i] * 4.0;
    i = i + 1;
  }
  putint(s);
  putch(10);
  return 0;
}
SRC

emit_run_case positive "lib_timer" "libfunc|starttime|stoptime" 8 "" "7" <<'SRC'
int main() {
  int x = 1;
  starttime();
  x = x + 6;
  stoptime();
  putint(x);
  putch(10);
  return 0;
}
SRC

# ------------------------------
# stress: deep loop + CFG pressure (6)
# ------------------------------
for i in $(seq 1 6); do
  n=$((20 + i * 4))
  s=0
  for ((x = 0; x < n; x++)); do
    for ((y = 0; y < n; y++)); do
      if ((( (x + y) % 7 == 0 ))); then
        continue
      fi
      s=$((s + ((x * 3 + y * 2) % 11)))
      if ((( s % 97 == 0 ))); then
        break
      fi
    done
  done

  emit_run_case stress "stress_cfg_${i}" "stress|loop|cfg" 12 "" "${s}" <<SRC
int main() {
  int n = ${n};
  int s = 0;
  int x = 0;
  while (x < n) {
    int y = 0;
    while (y < n) {
      if ((x + y) % 7 == 0) {
        y = y + 1;
        continue;
      }
      s = s + (x * 3 + y * 2) % 11;
      if (s % 97 == 0) break;
      y = y + 1;
    }
    x = x + 1;
  }
  putint(s);
  putch(10);
  return 0;
}
SRC
done

# ------------------------------
# stress: many-params + call chain pressure (6)
# ------------------------------
for i in $(seq 1 6); do
  base=$((i + 1))
  rounds=$((600 + i * 80))
  s=0
  k=${base}
  for ((t = 0; t < rounds; t++)); do
    row=$((16 * k + 120))
    s=$((s + row))
    k=$(((k + 1) % 17))
  done
  ans=$((s % 1000003))

  emit_run_case stress "stress_manyparams_${i}" "stress|many_params|call_chain" 12 "" "${ans}" <<SRC
int row16(int a0,int a1,int a2,int a3,int a4,int a5,int a6,int a7,int a8,int a9,int a10,int a11,int a12,int a13,int a14,int a15) {
  return a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
}
int main() {
  int rounds = ${rounds};
  int k = ${base};
  int s = 0;
  int t = 0;
  while (t < rounds) {
    s = s + row16(k,k+1,k+2,k+3,k+4,k+5,k+6,k+7,k+8,k+9,k+10,k+11,k+12,k+13,k+14,k+15);
    k = (k + 1) % 17;
    t = t + 1;
  }
  putint(s % 1000003);
  putch(10);
  return 0;
}
SRC
done

# ------------------------------
# stress: multidim array stride pressure (6)
# ------------------------------
for i in $(seq 1 6); do
  base=$((i + 2))
  sum=0
  for ((x = 0; x < 4; x++)); do
    for ((y = 0; y < 5; y++)); do
      for ((z = 0; z < 6; z++)); do
        v=$((base + x * 2 + y * 3 + z))
        sum=$((sum + v))
      done
    done
  done

  emit_run_case stress "stress_stride_${i}" "stress|array|stride|multidim" 12 "" "${sum}" <<SRC
int main() {
  int a[4][5][6];
  int x = 0;
  while (x < 4) {
    int y = 0;
    while (y < 5) {
      int z = 0;
      while (z < 6) {
        a[x][y][z] = ${base} + x * 2 + y * 3 + z;
        z = z + 1;
      }
      y = y + 1;
    }
    x = x + 1;
  }

  int s = 0;
  x = 0;
  while (x < 4) {
    int y = 0;
    while (y < 5) {
      int z = 0;
      while (z < 6) {
        s = s + a[x][y][z];
        z = z + 1;
      }
      y = y + 1;
    }
    x = x + 1;
  }

  putint(s);
  putch(10);
  return 0;
}
SRC
done

# ------------------------------
# stress: float + array + control-flow pressure (6)
# ------------------------------
for i in $(seq 1 6); do
  n=$((18 + i))
  # x4_j = (j%5)+2  (represents x*4)
  sum=0
  for ((j = 0; j < n; j++)); do
    x4=$(( (j % 5) + 2 ))
    if ((j % 3 == 0)); then
      sum=$((sum + x4))
    else
      sum=$((sum + x4 / 2))
    fi
  done

  emit_run_case stress "stress_float_ctrl_${i}" "stress|float|array|control" 12 "" "${sum}" <<SRC
int main() {
  int n = ${n};
  float a[64];
  int i = 0;
  while (i < n) {
    a[i] = ((i % 5) + 2.0) / 4.0;
    i = i + 1;
  }

  int s = 0;
  i = 0;
  while (i < n) {
    if (i % 3 == 0) s = s + a[i] * 4.0;
    else s = s + a[i] * 2.0;
    i = i + 1;
  }

  putint(s);
  putch(10);
  return 0;
}
SRC
done

# ------------------------------
# negative: lexical errors (4)
# ------------------------------
emit_fail_case "neg_lex_01" "negative|lex" 6 <<'SRC'
int main() {
  int a = 1;
  @
  return a;
}
SRC

emit_fail_case "neg_lex_02" "negative|lex" 6 <<'SRC'
int main() {
  int a = 0xG1;
  return a;
}
SRC

emit_fail_case "neg_lex_03" "negative|lex" 6 <<'SRC'
int main() {
  float x = 1.2.3;
  return 0;
}
SRC

emit_fail_case "neg_lex_04" "negative|lex" 6 <<'SRC'
int main() {
  int a = 1;
  /* unterminated comment
  return a;
}
SRC

# ------------------------------
# negative: syntax errors (6)
# ------------------------------
emit_fail_case "neg_syn_01" "negative|syntax" 6 <<'SRC'
int main() {
  int a = 1
  return a;
}
SRC

emit_fail_case "neg_syn_02" "negative|syntax" 6 <<'SRC'
int main( {
  return 0;
}
SRC

emit_fail_case "neg_syn_03" "negative|syntax" 6 <<'SRC'
int main() {
  if (1) {
    return 0;
}
SRC

emit_fail_case "neg_syn_04" "negative|syntax" 6 <<'SRC'
int main() {
  while (1)
    return 0
}
SRC

emit_fail_case "neg_syn_05" "negative|syntax" 6 <<'SRC'
int main() {
  int a[2] = {1,2;
  return 0;
}
SRC

emit_fail_case "neg_syn_06" "negative|syntax" 6 <<'SRC'
int main() {
  return (1 + 2;
}
SRC

# ------------------------------
# negative: semantic errors (14)
# ------------------------------
emit_fail_case "neg_sem_01" "negative|semantic|undef" 6 <<'SRC'
int main() {
  return x;
}
SRC

emit_fail_case "neg_sem_02" "negative|semantic|redef" 6 <<'SRC'
int main() {
  int a = 1;
  int a = 2;
  return a;
}
SRC

emit_fail_case "neg_sem_03" "negative|semantic|const_assign" 6 <<'SRC'
int main() {
  const int a = 1;
  a = 2;
  return a;
}
SRC

emit_fail_case "neg_sem_04" "negative|semantic|break" 6 <<'SRC'
int main() {
  break;
  return 0;
}
SRC

emit_fail_case "neg_sem_05" "negative|semantic|continue" 6 <<'SRC'
int main() {
  continue;
  return 0;
}
SRC

emit_fail_case "neg_sem_06" "negative|semantic|call_arity" 6 <<'SRC'
int f(int x, int y) { return x + y; }
int main() {
  return f(1);
}
SRC

emit_fail_case "neg_sem_07" "negative|semantic|call_arity" 6 <<'SRC'
int f(int x) { return x; }
int main() {
  return f(1, 2);
}
SRC

emit_fail_case "neg_sem_08" "negative|semantic|void_value" 6 <<'SRC'
void f() {}
int main() {
  int x = f();
  return x;
}
SRC

emit_fail_case "neg_sem_09" "negative|semantic|return_type" 6 <<'SRC'
void f() {
  return 1;
}
int main() {
  f();
  return 0;
}
SRC

emit_fail_case "neg_sem_10" "negative|semantic|array_dim" 6 <<'SRC'
int n = 3;
int a[n];
int main() {
  return 0;
}
SRC

emit_fail_case "neg_sem_11" "negative|semantic|dup_func" 6 <<'SRC'
int f() { return 1; }
int f() { return 2; }
int main() { return f(); }
SRC

emit_fail_case "neg_sem_12" "negative|semantic|assign_rvalue" 6 <<'SRC'
int main() {
  (1 + 2) = 3;
  return 0;
}
SRC

emit_fail_case "neg_sem_13" "negative|semantic|undef_func" 6 <<'SRC'
int main() {
  return g(1);
}
SRC

emit_fail_case "neg_sem_14" "negative|semantic|param_redef" 6 <<'SRC'
int f(int x, int x) {
  return x;
}
int main() {
  return f(1, 2);
}
SRC

# ------------------------------
# Duplicate guard vs official suites and within custom set
# ------------------------------
hash_file() {
  sha256sum "$1" | awk '{ print $1 }'
}

canon_hash_file() {
  awk '{ for (i = 1; i <= NF; i++) printf "%s ", $i }' "$1" | sha256sum | awk '{ print $1 }'
}

declare -A seen_exact
declare -A seen_canon
dup_custom=0
while IFS= read -r -d '' f; do
  h="$(hash_file "$f")"
  hc="$(canon_hash_file "$f")"
  if [[ -n "${seen_exact[$h]:-}" ]]; then
    echo "error: custom duplicate (exact): $f == ${seen_exact[$h]}" >&2
    dup_custom=$((dup_custom + 1))
  else
    seen_exact[$h]="$f"
  fi
  if [[ -n "${seen_canon[$hc]:-}" ]]; then
    echo "error: custom duplicate (canonical): $f == ${seen_canon[$hc]}" >&2
    dup_custom=$((dup_custom + 1))
  else
    seen_canon[$hc]="$f"
  fi
done < <(find "${OUT_ROOT}" -type f -name '*.sy' -print0 | sort -z)

if [[ "${dup_custom}" -ne 0 ]]; then
  echo "error: found ${dup_custom} duplicate custom cases" >&2
  exit 1
fi

OFFICIAL_ROOT="${SISY_OFFICIAL_SUITE_ROOT:-${ROOT_DIR}/test/official}"
if [[ -d "${OFFICIAL_ROOT}" ]]; then
  declare -A official_exact
  declare -A official_canon
  while IFS= read -r -d '' f; do
    official_exact["$(hash_file "$f")"]="$f"
    official_canon["$(canon_hash_file "$f")"]="$f"
  done < <(find "${OFFICIAL_ROOT}" -type f -name '*.sy' -print0 | sort -z)

  dup_official=0
  while IFS= read -r -d '' f; do
    h="$(hash_file "$f")"
    hc="$(canon_hash_file "$f")"
    if [[ -n "${official_exact[$h]:-}" ]]; then
      echo "error: custom case duplicates official (exact): $f == ${official_exact[$h]}" >&2
      dup_official=$((dup_official + 1))
    fi
    if [[ -n "${official_canon[$hc]:-}" ]]; then
      echo "error: custom case duplicates official (canonical): $f == ${official_canon[$hc]}" >&2
      dup_official=$((dup_official + 1))
    fi
  done < <(find "${OUT_ROOT}" -type f -name '*.sy' -print0 | sort -z)

  if [[ "${dup_official}" -ne 0 ]]; then
    echo "error: found ${dup_official} custom/offical duplicate collisions" >&2
    exit 1
  fi
else
  echo "warning: official suite dir not found, skip duplicate guard vs official: ${OFFICIAL_ROOT}" >&2
fi

# ------------------------------
# Final assertions
# ------------------------------
if [[ "${pos_count}" -ne 72 ]]; then
  echo "error: positive count=${pos_count}, expected 72" >&2
  exit 1
fi
if [[ "${str_count}" -ne 24 ]]; then
  echo "error: stress count=${str_count}, expected 24" >&2
  exit 1
fi
if [[ "${neg_count}" -ne 24 ]]; then
  echo "error: negative count=${neg_count}, expected 24" >&2
  exit 1
fi

rows=$((idx - 1))
if [[ "${rows}" -ne 120 ]]; then
  echo "error: total cases=${rows}, expected 120" >&2
  exit 1
fi

echo "generated: ${OUT_ROOT}"
echo "manifest : ${MANIFEST}"
echo "counts   : positive=${pos_count}, stress=${str_count}, negative=${neg_count}, total=${rows}"
