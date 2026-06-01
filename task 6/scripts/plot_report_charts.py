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


def load_multicore_csv():
    path = RESULTS / "timing_multicore.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(open(path)))
    return {int(r["N"]): float(r["time_sec"]) for r in rows if r["variant"] == "opt"}


def load_gpu_csv():
    gpu_csv = RESULTS / "timing_gpu.csv"
    if not gpu_csv.exists():
        return {}, {}
    rows = list(csv.DictReader(open(gpu_csv)))
    base = {int(r["N"]): float(r["time_sec"]) for r in rows if r["variant"] == "base"}
    opt = {int(r["N"]): float(r["time_sec"]) for r in rows if r["variant"] == "opt"}
    return base, opt


def load_optimization_stages():
    path = RESULTS / "optimization_stages.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(open(path)))


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    base, opt = load_host_csv()
    multi = load_multicore_csv()
    _, gpu_opt = load_gpu_csv()
    sizes = sorted(base.keys())

    stages = load_optimization_stages()
    if stages:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = [f"{row['stage']}\n{row['label']}" for row in stages]
        times = [float(row["time_sec"]) for row in stages]
        colors = ["#4c72b0", "#55a868", "#8172b2", "#c44e52"]
        bars = ax.bar(labels, times, color=colors[: len(stages)])
        ax.set_ylabel("Время, с")
        ax.set_xlabel("Этап оптимизации")
        ax.set_title("Этапы оптимизации (512×512, до сходимости)")
        ax.grid(axis="y", alpha=0.3)
        for bar, row in zip(bars, stages):
            h = bar.get_height()
            note = f"{float(row['time_sec']):.3g} s\n({row['iterations']} iter)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.08,
                note,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(RESULTS / "chart_optimization_stages.png", dpi=140)
        plt.close(fig)

    if sizes:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(sizes))
        w = 0.26
        ax.bar([i - w for i in x], [opt[s] for s in sizes], w, label="CPU-onecore")
        if multi:
            ax.bar([i for i in x], [multi.get(s, 0) for s in sizes], w, label="CPU-multicore")
        if gpu_opt:
            ax.bar([i + w for i in x], [gpu_opt.get(s, 0) for s in sizes], w, label="GPU (opt)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{s}×{s}" for s in sizes])
        ax.set_ylabel("Время, с")
        ax.set_title("CPU-onecore, CPU-multicore, GPU (opt)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "chart_cpu_multi_gpu.png", dpi=140)
        plt.close(fig)

        if multi:
            fig, ax = plt.subplots(figsize=(8, 4))
            w = 0.35
            ax.bar([i - w / 2 for i in x], [opt[s] for s in sizes], w, label="Onecore")
            ax.bar([i + w / 2 for i in x], [multi[s] for s in sizes], w, label="Multicore")
            ax.set_xticks(list(x))
            ax.set_xticklabels([f"{s}×{s}" for s in sizes])
            ax.set_ylabel("Время, с")
            ax.set_title("CPU-onecore vs CPU-multicore")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(RESULTS / "chart_cpu_one_vs_multi.png", dpi=140)
            plt.close(fig)

    print("Saved charts in results/")


if __name__ == "__main__":
    main()
