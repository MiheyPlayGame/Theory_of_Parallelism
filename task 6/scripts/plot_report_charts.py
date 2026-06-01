#!/usr/bin/env python3
"""Charts for REPORT.md: optimization stages and CPU comparison."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_host_csv():
    rows = list(csv.DictReader(open(RESULTS / "timing_host.csv")))
    base = {int(r["N"]): float(r["time_sec"]) for r in rows if r["variant"] == "base"}
    opt = {int(r["N"]): float(r["time_sec"]) for r in rows if r["variant"] == "opt"}
    return base, opt


def load_gpu_csv():
    gpu_csv = RESULTS / "timing_gpu.csv"
    if not gpu_csv.exists():
        return {}, {}
    rows = list(csv.DictReader(open(gpu_csv)))
    base = {int(r["N"]): float(r["time_sec"]) for r in rows if r["variant"] == "base"}
    opt = {int(r["N"]): float(r["time_sec"]) for r in rows if r["variant"] == "opt"}
    return base, opt


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    base, opt = load_host_csv()
    gpu_base, gpu_opt = load_gpu_csv()
    sizes = sorted(base.keys())

    # Optimization stages (512x512) from benchmark CSVs:
    # 1) host baseline, 2) host optimized, 3) gpu optimized
    fig, ax = plt.subplots(figsize=(7, 4))
    stages = ["1\nbaseline", "2\nopt host", "3\nopt gpu"]
    times = [base.get(512), opt.get(512), gpu_opt.get(512)]
    stages_times = [(s, t) for s, t in zip(stages, times) if t is not None]
    stages = [s for s, _ in stages_times]
    times = [t for _, t in stages_times]
    ax.bar(stages, times, color=["#4c72b0", "#55a868", "#8172b2"][: len(stages)])
    ax.set_ylabel("Время, с")
    ax.set_xlabel("Этап оптимизации")
    ax.set_title("Этапы оптимизации (512×512)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "chart_optimization_stages.png", dpi=120)
    plt.close(fig)

    # Onecore: base vs opt as proxy until multicore filled
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(sizes))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [base[s] for s in sizes], w, label="Onecore (baseline)")
    ax.bar([i + w / 2 for i in x], [opt[s] for s in sizes], w, label="Onecore (optimized)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.set_ylabel("Время, с")
    ax.set_title("CPU-onecore: до и после оптимизации (заменить Multicore после замера)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "chart_cpu_onecore.png", dpi=120)
    plt.close(fig)
    print("Saved charts in results/")


if __name__ == "__main__":
    main()
