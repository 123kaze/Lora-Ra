"""Render real pilot CSVs only. English labels; PNG 450dpi + editable SVG."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0077BB", "#009988", "#EE7733", "#CC3311"]


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, destination):
    fig.savefig(destination.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_run(run):
    run = Path(run)
    output = run / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "svg.fonttype": "none"})
    summary = read(run / "summary.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    values = [float(r["val_loss_mean"]) for r in summary]
    errors = [float(r["val_loss_std"]) for r in summary]
    ax.bar(np.arange(len(summary)), values, yerr=errors, capsize=3,
           color=[COLORS[2] if r["method"] == "Gradient" else COLORS[0] for r in summary])
    ax.set_xticks(np.arange(len(summary)), [r["method"] for r in summary], rotation=20)
    ax.set_ylabel("Validation loss (nat/byte), lower is better")
    low = min(v - e for v, e in zip(values, errors))
    high = max(v + e for v, e in zip(values, errors))
    ax.set_ylim(max(0, low - max(0.02, (high - low) * 0.15)), high + max(0.02, (high - low) * 0.15))
    ax.set_title(f"{run.name}: matched adapter parameters; mean +/- sample SD")
    save(fig, output / "comparison")
    spectra = [r for r in read(run / "spectra.csv") if r["scheme"] == "true"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    layers = sorted({int(r["layer"]) for r in spectra})
    for label, color in zip(("C", "R"), COLORS):
        sensitivity = []
        for layer in layers:
            points = [r for r in spectra if r["label"] == label and int(r["layer"]) == layer and int(r["rank"]) == 1]
            sensitivity.append(np.mean([float(r["utility_per_parameter"]) for r in points]))
        axes[0].plot([v + 1 for v in layers], sensitivity, marker="o", label=label, color=color)
        points = [r for r in spectra if r["label"] == label and int(r["layer"]) == max(layers) and ".v_proj/" in r["group"]]
        axes[1].plot([int(r["rank"]) for r in points], [float(r["sigma"]) for r in points], marker=".", label=label, color=color)
    axes[0].set(xlabel="Expansion block (1-based)", ylabel="Top-rank utility / parameter\n(mean of Q and V)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_xticks([v + 1 for v in layers])
    axes[1].set(xlabel="Singular direction", ylabel="Mean-gradient singular value", title="Final block, V projection")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    save(fig, output / "sensitivity")
    sweep = read(run / "sweep.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for label, color in zip(("C", "R"), COLORS):
        rows = [r for r in sweep if r["label"] == label]
        ax.scatter([float(r["predicted_utility"]) for r in rows], [float(r["actual_val_improvement"]) for r in rows], label=label, color=color)
        for row in rows:
            ax.annotate(f"r={row['rank']}", (float(row["predicted_utility"]), float(row["actual_val_improvement"])), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set(xlabel="Predicted local utility (unit curvature)", ylabel="Actual validation loss improvement (nat/byte)", title="One projection; rank 0 excluded")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend()
    fig.tight_layout()
    save(fig, output / "prediction")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    plot_run(parser.parse_args().run)
