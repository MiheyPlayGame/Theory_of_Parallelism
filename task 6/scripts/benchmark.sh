#!/usr/bin/env bash
# Benchmark heat solver for grid sizes 128, 256, 512, 1024.
# Usage: ./scripts/benchmark.sh [host|gpu|multicore]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACC="${1:-host}"
BUILD="$ROOT/build"
OUT="$ROOT/results"
SIZES=(128 256 512 1024)
MAX_ITER=1000000
EPS=1e-6

mkdir -p "$OUT"
make -C "$ROOT" "ACC=$ACC" all

BASE="$BUILD/heat_base_${ACC}"
OPT="$BUILD/heat_opt_${ACC}"
CSV="$OUT/timing_${ACC}.csv"

echo "mode,variant,N,iterations,error,time_sec" > "$CSV"

for N in "${SIZES[@]}"; do
  for variant in base opt; do
    bin="$BASE"
    [[ "$variant" == "opt" ]] && bin="$OPT"
    line=$("$bin" --size "$N" --eps "$EPS" --max-iter "$MAX_ITER" --quiet)
    it=$(echo "$line" | sed -n 's/.*iterations=\([0-9]*\).*/\1/p')
    er=$(echo "$line" | sed -n 's/.*error=\([^ ]*\).*/\1/p')
    tm=$(echo "$line" | sed -n 's/.*time_sec=\([^ ]*\).*/\1/p')
    echo "${ACC},${variant},${N},${it},${er},${tm}" >> "$CSV"
    echo "$ACC $variant N=$N: $line"
  done
done

echo "Wrote $CSV"

if command -v gnuplot >/dev/null 2>&1; then
  gnuplot -e "
    set terminal pngcairo size 900,500;
    set output '$OUT/speedup_${ACC}.png';
    set title 'Heat solver (${ACC}): baseline vs optimized';
    set xlabel 'Grid size N';
    set ylabel 'Time (s), log scale';
    set logscale y;
    set grid;
    plot '$CSV' using 3:6 with linespoints title 'baseline' if (column(2) eq 'base'),
         '' using 3:6 with linespoints title 'optimized' if (column(2) eq 'opt');
  " 2>/dev/null || python3 "$ROOT/scripts/plot_results.py" "$CSV" "$OUT/speedup_${ACC}.png" "$ACC"
else
  python3 "$ROOT/scripts/plot_results.py" "$CSV" "$OUT/speedup_${ACC}.png" "$ACC" || true
fi
