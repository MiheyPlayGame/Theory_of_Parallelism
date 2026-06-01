#!/usr/bin/env bash
# Nsight Systems profiling helper for heat solver.
# Usage: ./scripts/profile_nsight.sh [host|gpu|multicore] [N] [iters]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACC="${1:-gpu}"
N="${2:-256}"
ITERS="${3:-1000000}"
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

NSYS_BIN="$(readlink -f "$NSYS_BIN")"
NSYS_HOME="$(cd "$(dirname "$NSYS_BIN")/.." && pwd)"

NSYS_HOST_DIR=""
for host_dir in \
  "$NSYS_HOME/host-linux-x64" \
  "$NSYS_HOME/../host-linux-x64" \
  "$HOME/.local/nsys/usr/lib/nsight-systems/host-linux-x64"
do
  if [[ -x "$host_dir/QdstrmImporter" ]]; then
    NSYS_HOST_DIR="$host_dir"
    break
  fi
done

if [[ -z "$NSYS_HOST_DIR" ]]; then
  echo "QdstrmImporter not found. Reinstall Nsight Systems."
  exit 1
fi

# Broken user-local layout: target-linux-x64 exists, host-linux-x64 is elsewhere.
if [[ "$NSYS_HOST_DIR" != "$NSYS_HOME/host-linux-x64" && ! -e "$NSYS_HOME/host-linux-x64" ]]; then
  ln -sfn "$NSYS_HOST_DIR" "$NSYS_HOME/host-linux-x64"
fi

# QdstrmImporter links against libssh.so (unversioned); Ubuntu ships libssh.so.4 only.
NSYS_LIB_DIR="$ROOT/.deps/nsys-lib"
mkdir -p "$NSYS_LIB_DIR"
if [[ ! -e "$NSYS_LIB_DIR/libssh.so" ]]; then
  for libssh in /usr/lib/x86_64-linux-gnu/libssh.so.4 /usr/lib64/libssh.so.4; do
    if [[ -e "$libssh" ]]; then
      ln -sf "$libssh" "$NSYS_LIB_DIR/libssh.so"
      break
    fi
  done
fi

export PATH="$NSYS_HOST_DIR:$PATH"
export LD_LIBRARY_PATH="$NSYS_LIB_DIR:$NSYS_HOST_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Old user-local nsys (2021.x) injects libToolsInjectionProxy64.so, which breaks on
# Ubuntu 22.04/WSL glibc 2.35. Disable OS/CPU sampling injection; keep CUDA trace for GPU.
NSYS_ARGS=(profile --stats=true --sample=none --force-overwrite=true -o "$REPORT")
case "$ACC" in
  gpu)       NSYS_ARGS+=(--trace=cuda) ;;
  host)      NSYS_ARGS+=(--trace=none) ;;
  multicore) NSYS_ARGS+=(--trace=none) ;;
esac

echo "Using nsys: $NSYS_BIN"
echo "Profile args: ${NSYS_ARGS[*]}"

PROFILE_LOG="$(mktemp)"
"$NSYS_BIN" "${NSYS_ARGS[@]}" \
  "$BIN" --size "$N" --max-iter "$ITERS" --eps 1e-6 --quiet 2>&1 | tee "$PROFILE_LOG"
RUN_LINE="$(grep -E 'iterations=[0-9]+' "$PROFILE_LOG" | tail -1 || true)"
rm -f "$PROFILE_LOG"
if [[ -n "$RUN_LINE" ]]; then
  echo "$RUN_LINE" > "$OUT/last_profile_timing.txt"
  echo "Timing: $RUN_LINE"
fi

echo "Profile output prefix: $REPORT"
for ext in nsys-rep qdrep sqlite; do
  if [[ -f "${REPORT}.${ext}" ]]; then
    echo "Report ready: ${REPORT}.${ext}"
  fi
done

if [[ ! -f "${REPORT}.nsys-rep" && ! -f "${REPORT}.qdrep" && -f "${REPORT}.qdstrm" ]]; then
  echo "Auto-import failed; running QdstrmImporter manually..."
  TMP_QDSTRM="$(mktemp /tmp/nsys_import_XXXXXX.qdstrm)"
  cp "${REPORT}.qdstrm" "$TMP_QDSTRM"
  "$NSYS_HOST_DIR/QdstrmImporter" \
    --input-file "$TMP_QDSTRM" \
    --output-file "${REPORT}.qdrep" \
    --force-overwrite
  rm -f "$TMP_QDSTRM"
  echo "Report ready: ${REPORT}.qdrep"
fi
