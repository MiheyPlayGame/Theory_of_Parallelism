#!/usr/bin/env bash
# Nsight Systems profiling helper for heat solver.
# Usage: ./scripts/profile_nsight.sh [host|gpu|multicore] [N] [iters]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACC="${1:-gpu}"
N="${2:-256}"
ITERS="${3:-50}"
OUT="$ROOT/results"

mkdir -p "$OUT"
make -C "$ROOT" "ACC=$ACC" all

BIN="$ROOT/build/heat_opt_${ACC}"
REPORT="$OUT/nsys_N${N}_iter${ITERS}_${ACC}"

NSYS_BIN="$(command -v nsys || true)"
if [[ -z "${NSYS_BIN}" ]]; then
  for candidate in \
    "/usr/lib/nsight-systems/bin/nsys" \
    "$HOME/.local/nsys/usr/lib/nsight-systems/bin/nsys"
  do
    if [[ -x "$candidate" ]]; then
      NSYS_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "${NSYS_BIN}" ]]; then
  echo "nsys not found. Install NVIDIA Nsight Systems or add /usr/lib/nsight-systems/bin to PATH."
  exit 1
fi

"$NSYS_BIN" profile --stats=true -o "$REPORT" \
  "$BIN" --size "$N" --max-iter "$ITERS" --eps 1e-6 --quiet

echo "Profile output prefix: $REPORT"
echo "Open .nsys-rep in Nsight Systems GUI if import succeeded."
