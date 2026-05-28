#!/usr/bin/env python3
"""Plot benchmark CSV (mode, variant, N, iterations, error, time_sec)."""

import csv
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: plot_results.py timing.csv output.png title")
        sys.exit(1)

    path, out_png, title = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = list(csv.DictReader(open(path)))
    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append((int(r["N"]), float(r["time_sec"])))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plot")
        return

    plt.figure(figsize=(9, 5))
    for variant, pts in sorted(by_var.items()):
        pts.sort()
        xs, ys = zip(*pts)
        plt.plot(xs, ys, "o-", label=variant)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Grid size N")
    plt.ylabel("Time (s)")
    plt.title(f"Heat solver ({title})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")

if __name__ == "__main__":
    main()
