#!/usr/bin/env bash
# Full rebuild, verify, benchmark (host/gpu/multicore), Nsight profile, charts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Ensure nvc++ on PATH if installed under /opt or user-local
NVHPC_BIN=""
for root in /opt/nvidia/hpc_sdk "$HOME/.local/nvidia/hpc_sdk" "$ROOT/.deps/nvhpc"; do
  candidate="$(ls -d "$root"/Linux_x86_64/*/compilers/bin 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "$candidate" && -x "$candidate/nvc++" ]]; then
    NVHPC_BIN="$candidate"
    export PATH="$NVHPC_BIN:$PATH"
    export CXX="$NVHPC_BIN/nvc++"
    break
  fi
done
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if ! command -v nvc++ >/dev/null 2>&1; then
  echo "nvc++ not found; running install_nvhpc.sh ..."
  "$ROOT/scripts/install_nvhpc.sh"
fi

echo "=== Compiler check ==="
make check-nvhpc || true

echo "=== Clean build (host, gpu, multicore) ==="
make clean
make host
make gpu
make multicore

echo "=== Verify 10x10 / 13x13 ==="
make verify

echo "=== Benchmark host ==="
./scripts/benchmark.sh host

echo "=== Benchmark multicore ==="
./scripts/benchmark.sh multicore

echo "=== Benchmark gpu ==="
./scripts/benchmark.sh gpu

echo "=== Nsight profile (gpu, N=512) ==="
./scripts/profile_nsight.sh gpu 512 1000000

echo "=== Update optimization_stages.csv ==="
python3 - <<'PY'
import csv
from pathlib import Path

root = Path(".")
results = root / "results"

def row_from_csv(path, variant, n=512):
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["variant"] == variant and int(r["N"]) == n:
                return r
    raise SystemExit(f"missing {path} {variant} N={n}")

host_base = row_from_csv(results / "timing_host.csv", "base")
host_opt = row_from_csv(results / "timing_host.csv", "opt")
gpu_opt = row_from_csv(results / "timing_gpu.csv", "opt")

profile_path = results / "last_profile_timing.txt"
if not profile_path.exists():
    raise SystemExit("missing last_profile_timing.txt from profile_nsight.sh")
nsys_line = profile_path.read_text().strip()
parts = dict(p.split("=", 1) for p in nsys_line.split())
nsys_time = parts["time_sec"]
nsys_iter = parts["iterations"]
nsys_err = parts["error"]

rows = [
    ("1", "baseline host", host_base["time_sec"], host_base["iterations"], host_base["error"],
     "timing_host.csv variant=base N=512"),
    ("2", "opt host", host_opt["time_sec"], host_opt["iterations"], host_opt["error"],
     "timing_host.csv variant=opt N=512"),
    ("3", "opt gpu", gpu_opt["time_sec"], gpu_opt["iterations"], gpu_opt["error"],
     "timing_gpu.csv variant=opt N=512"),
    ("4", "nsight profile gpu", nsys_time, nsys_iter, nsys_err,
     "nsys profile gpu 512 max-iter 1000000"),
]

out = results / "optimization_stages.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stage", "label", "time_sec", "iterations", "error", "source"])
    w.writerows(rows)
print(f"Wrote {out}")
PY

echo "=== Plot charts ==="
python3 ./scripts/plot_report_charts.py

echo "=== Done ==="
