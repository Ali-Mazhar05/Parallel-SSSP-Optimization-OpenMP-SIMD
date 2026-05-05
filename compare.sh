#!/usr/bin/env bash
# =============================================================================
# compare.sh  —  Side-by-side benchmark: baseline vs OpenMP-optimized SSSP
#
# Usage:
#   ./compare.sh -i <graph_file> [-a <algorithm>] [-p <param>] [-w] [-s]
#
# Options:
#   -i   Input graph file (required)
#   -a   Algorithm: rho-stepping | delta-stepping | bellman-ford  (default: rho-stepping)
#   -p   Algorithm parameter, e.g. rho or delta value
#   -w   Weighted graph
#   -s   Symmetrized graph
#   -h   Show this help
#
# What this script does:
#   1. Runs bin/sssp_baseline once
#   2. Runs bin/sssp_omp with  2 threads
#   3. Runs bin/sssp_omp with  4 threads
#   4. Runs bin/sssp_omp with  8 threads
#   5. Prints a combined report with one speedup column per thread count
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
ALGO="rho-stepping"
PARAM=""
WEIGHTED=""
SYMMETRIZED=""
INPUT=""
OMP_THREAD_COUNTS=(2 4 8)   # thread sweep — no -t flag needed

# ── Colour codes ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[1;33m'
CYN='\033[0;36m'
BLD='\033[1m'
RST='\033[0m'

usage() {
  grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
  exit 0
}

# ── Parse arguments ───────────────────────────────────────────────────────────
while getopts "i:a:p:wsh" opt; do
  case $opt in
    i) INPUT="$OPTARG"  ;;
    a) ALGO="$OPTARG"   ;;
    p) PARAM="$OPTARG"  ;;
    w) WEIGHTED="-w"    ;;
    s) SYMMETRIZED="-s" ;;
    h) usage            ;;
    *) echo "Unknown option -$OPTARG"; exit 1 ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo -e "${RED}Error: -i <graph_file> is required.${RST}"
  exit 1
fi

if [[ ! -f "bin/sssp_baseline" || ! -f "bin/sssp_omp" ]]; then
  echo -e "${RED}Error: Binaries not found. Run 'make all' first.${RST}"
  exit 1
fi

# ── Setup results directory ───────────────────────────────────────────────────
mkdir -p results
REPORT="results/comparison_report.txt"

# ── Build common argument string ──────────────────────────────────────────────
COMMON_ARGS="-i $INPUT -a $ALGO $WEIGHTED $SYMMETRIZED -v"
[[ -n "$PARAM" ]] && COMMON_ARGS="$COMMON_ARGS -p $PARAM"

BASELINE_LOG="results/baseline_run.log"

# ── Helper: parse "Average time: X" lines from a log ─────────────────────────
parse_times() { grep "^Average time:" "$1" | awk '{print $3}'; }

# ── Helper: compute speedup string safely ────────────────────────────────────
speedup_str() {
  local bt="$1" ot="$2"
  local bt_zero ot_zero
  bt_zero=$(awk "BEGIN {print ($bt+0==0)?1:0}")
  ot_zero=$(awk "BEGIN {print ($ot+0==0)?1:0}")
  if [[ "$bt_zero" == "1" || "$ot_zero" == "1" ]]; then
    echo "N/A"
  else
    awk "BEGIN {printf \"%.2fx\", $bt/$ot}"
  fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLD}${CYN}╔══════════════════════════════════════════════════════════════╗${RST}"
echo -e "${BLD}${CYN}║         SSSP Baseline vs OpenMP Comparison Suite             ║${RST}"
echo -e "${BLD}${CYN}╚══════════════════════════════════════════════════════════════╝${RST}"
echo ""
echo -e "  Graph      : ${BLD}$INPUT${RST}"
echo -e "  Algorithm  : ${BLD}$ALGO${RST}"
echo -e "  Parameter  : ${BLD}${PARAM:-default}${RST}"
echo -e "  OMP Sweeps : ${BLD}${OMP_THREAD_COUNTS[*]} threads${RST}"
echo -e "  Weighted   : ${BLD}${WEIGHTED:--}${RST}"
echo -e "  Symmetrized: ${BLD}${SYMMETRIZED:--}${RST}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Run baseline once
# ══════════════════════════════════════════════════════════════════════════════
echo -e "${YEL}[1/4] Running BASELINE...${RST}"
ts=$(date +%s%N)
./bin/sssp_baseline $COMMON_ARGS 2>&1 | tee "$BASELINE_LOG"
te=$(date +%s%N)
BASELINE_WALL=$(echo "scale=3; ($te-$ts)/1000000000" | bc)
echo -e "${GRN}Baseline wall-clock: ${BASELINE_WALL}s${RST}"
echo ""

BASE_TIMES=($(parse_times "$BASELINE_LOG"))
N=${#BASE_TIMES[@]}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Run OMP at each thread count
# ══════════════════════════════════════════════════════════════════════════════
declare -A OMP_WALL_MAP     # OMP_WALL_MAP[T]  = wall-clock string
declare -A OMP_TIMES_MAP    # OMP_TIMES_MAP[T] = space-separated avg times
declare -A OMP_CORRECT_MAP  # OMP_CORRECT_MAP[T] = "true" | "false"

step=1
for T in "${OMP_THREAD_COUNTS[@]}"; do
  ((step++))
  OMP_LOG="results/omp_${T}t_run.log"
  echo -e "${YEL}[$((step))/4] Running OPENMP with ${T} threads...${RST}"
  ts=$(date +%s%N)
  OMP_NUM_THREADS=$T ./bin/sssp_omp $COMMON_ARGS 2>&1 | tee "$OMP_LOG"
  te=$(date +%s%N)
  wall=$(echo "scale=3; ($te-$ts)/1000000000" | bc)
  OMP_WALL_MAP[$T]="$wall"
  echo -e "${GRN}OpenMP (${T}t) wall-clock: ${wall}s${RST}"
  echo ""

  OMP_TIMES_MAP[$T]="$(parse_times "$OMP_LOG" | tr '\n' ' ')"

  mismatches=$(grep -c "exp_dist\[" "$OMP_LOG" 2>/dev/null || true)
  mismatches=${mismatches:-0}
  OMP_CORRECT_MAP[$T]=$( [[ "$mismatches" -gt 0 ]] && echo "false" || echo "true" )
done

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Compute all aggregates (must happen OUTSIDE the tee subshell)
# ══════════════════════════════════════════════════════════════════════════════

# Expand per-T time strings into indexed arrays
declare -a T2_TIMES T4_TIMES T8_TIMES
read -ra T2_TIMES <<< "${OMP_TIMES_MAP[2]}"
read -ra T4_TIMES <<< "${OMP_TIMES_MAP[4]}"
read -ra T8_TIMES <<< "${OMP_TIMES_MAP[8]}"

TOTAL_BASE=0
declare -A TOTAL_OMP_T
for T in "${OMP_THREAD_COUNTS[@]}"; do TOTAL_OMP_T[$T]=0; done

ALL_CORRECT_GLOBAL=true
declare -A ALL_CORRECT_T
for T in "${OMP_THREAD_COUNTS[@]}"; do ALL_CORRECT_T[$T]=true; done

# Build per-source rows into a string
DIVIDER=$(printf '─%.0s' {1..88})
ROW_LINES=""
for ((i=0; i<N; i++)); do
  bt="${BASE_TIMES[$i]:-0}"
  row=$(printf "%-6s  %-13s" "$((i+1))" "$bt")
  for T in "${OMP_THREAD_COUNTS[@]}"; do
    case $T in
      2) ot="${T2_TIMES[$i]:-0}" ;;
      4) ot="${T4_TIMES[$i]:-0}" ;;
      8) ot="${T8_TIMES[$i]:-0}" ;;
    esac
    sp=$(speedup_str "$bt" "$ot")
    row+=$(printf "  %-13s %-8s" "$ot" "$sp")
    TOTAL_OMP_T[$T]=$(awk "BEGIN {print ${TOTAL_OMP_T[$T]} + $ot}")
  done
  TOTAL_BASE=$(awk "BEGIN {print $TOTAL_BASE + $bt}")
  ROW_LINES+="$row"$'\n'
done

# Averages
if [[ "$N" -gt 0 ]]; then
  AVG_BASE=$(awk "BEGIN {printf \"%.6f\", $TOTAL_BASE/$N}")
else
  AVG_BASE="N/A"
fi

declare -A AVG_OMP_T AVG_SPEEDUP_T WALL_SPEEDUP_T
for T in "${OMP_THREAD_COUNTS[@]}"; do
  tot="${TOTAL_OMP_T[$T]}"
  if [[ "$(awk "BEGIN {print ($tot+0>0)?1:0}")" == "1" && "$N" -gt 0 ]]; then
    AVG_OMP_T[$T]=$(awk     "BEGIN {printf \"%.6f\", $tot/$N}")
    AVG_SPEEDUP_T[$T]=$(awk "BEGIN {printf \"%.2fx\", $TOTAL_BASE/$tot}")
  else
    AVG_OMP_T[$T]="N/A"
    AVG_SPEEDUP_T[$T]="N/A"
  fi
  wall="${OMP_WALL_MAP[$T]}"
  if [[ "$(awk "BEGIN {print ($wall+0>0)?1:0}")" == "1" && \
        "$(awk "BEGIN {print ($BASELINE_WALL+0>0)?1:0}")" == "1" ]]; then
    WALL_SPEEDUP_T[$T]=$(awk "BEGIN {printf \"%.2fx\", $BASELINE_WALL/$wall}")
  else
    WALL_SPEEDUP_T[$T]="N/A"
  fi
  if [[ "${OMP_CORRECT_MAP[$T]}" == "false" ]]; then
    ALL_CORRECT_T[$T]=false
    ALL_CORRECT_GLOBAL=false
  fi
done

# ── Average row string
AVG_ROW=$(printf "%-6s  %-13s" "AVG" "$AVG_BASE")
for T in "${OMP_THREAD_COUNTS[@]}"; do
  AVG_ROW+=$(printf "  %-13s %-8s" "${AVG_OMP_T[$T]}" "${AVG_SPEEDUP_T[$T]}")
done

# ── Header row string
HDR=$(printf "%-6s  %-13s" "Src#" "Base(s)")
for T in "${OMP_THREAD_COUNTS[@]}"; do
  HDR+=$(printf "  %-13s %-8s" "OMP-${T}t(s)" "Spdup")
done

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Write report (tee subshell is fine here; all vars already set)
# ══════════════════════════════════════════════════════════════════════════════
{
printf "SSSP Performance & Correctness Comparison Report\n"
printf "Generated : %s\n"        "$(date)"
printf "Graph     : %s\n"        "$INPUT"
printf "Algorithm : %s  |  Param: %s\n\n" "$ALGO" "${PARAM:-default}"

printf "%s\n%s\n" "$HDR" "$DIVIDER"
printf "%s"       "$ROW_LINES"
printf "%s\n"     "$DIVIDER"
printf "%s\n\n"   "$AVG_ROW"

printf "Wall-clock times:\n"
printf "  %-14s %ss\n" "Baseline:" "$BASELINE_WALL"
for T in "${OMP_THREAD_COUNTS[@]}"; do
  printf "  %-14s %ss  (wall speedup: %s)\n" \
         "OMP-${T}t:" "${OMP_WALL_MAP[$T]}" "${WALL_SPEEDUP_T[$T]}"
done
printf "\n"

printf "Correctness:\n"
for T in "${OMP_THREAD_COUNTS[@]}"; do
  if ${ALL_CORRECT_T[$T]}; then
    printf "  OMP-%dt : [PASS] All distances match Dijkstra.\n" "$T"
  else
    printf "  OMP-%dt : [FAIL] Mismatches — see results/omp_%dt_run.log\n" "$T" "$T"
    grep "exp_dist\|act_dist" "results/omp_${T}t_run.log" 2>/dev/null | head -10 || true
  fi
done

} | tee "$REPORT"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Coloured terminal summary
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BLD}${CYN}═══════════════════════ SUMMARY ═══════════════════════${RST}"
echo -e "  ${BLD}Avg Baseline  : ${AVG_BASE}s${RST}"
for T in "${OMP_THREAD_COUNTS[@]}"; do
  echo -e "  ${BLD}Avg OMP-${T}t    : ${AVG_OMP_T[$T]}s  →  ${AVG_SPEEDUP_T[$T]} vs baseline${RST}"
done
echo ""
for T in "${OMP_THREAD_COUNTS[@]}"; do
  if ${ALL_CORRECT_T[$T]}; then
    echo -e "  OMP-${T}t correctness : ${GRN}${BLD}ALL CORRECT ✓${RST}"
  else
    echo -e "  OMP-${T}t correctness : ${RED}${BLD}MISMATCHES DETECTED ✗${RST}"
  fi
done
echo -e "  Full report   : ${BLD}$REPORT${RST}"
echo -e "${BLD}${CYN}═══════════════════════════════════════════════════════${RST}"
echo ""
