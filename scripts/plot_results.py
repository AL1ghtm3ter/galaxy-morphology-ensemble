"""Render only the experiment's recorded measurements, without GPU inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"ViT": "#2563eb", "Swin": "#0d9488", "CNN": "#ea580c", "Ensemble": "#7c3aed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads((ROOT / "results" / "training_history.json").read_text())["models"]
    metrics = json.loads((ROOT / "results" / "validation_metrics.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#334155",
                         "text.color": "#0f172a", "axes.titleweight": "bold", "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    axis = axes[0]
    for name, rows in history.items():
        axis.plot([row["epoch"] for row in rows], [row["validation_rmse"] for row in rows],
                  color=COLORS[name], linewidth=2, marker="o", markersize=3, label=name)
    axis.scatter([15], [metrics["global_rmse"]], marker="*", s=180, color=COLORS["Ensemble"],
                 label="Final ensemble + TTA", zorder=5)
    axis.annotate("0.07418", (15, metrics["global_rmse"]), xytext=(-58, 13),
                  textcoords="offset points", color=COLORS["Ensemble"], fontweight="bold", fontsize=9.5)
    axis.set(title="Recorded validation learning curves", xlabel="Epoch", ylabel="RMSE (lower is better)", xlim=(0.5, 16))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    axis.legend(frameon=False, fontsize=8.5, loc="upper right")
    names = ["Ensemble + TTA", "ViT", "Swin", "CNN"]
    values = [metrics["global_rmse"], history["ViT"][-1]["validation_rmse"],
              history["Swin"][-1]["validation_rmse"], history["CNN"][-1]["validation_rmse"]]
    axes[1].barh(names, values, color=[COLORS["Ensemble"], COLORS["ViT"], COLORS["Swin"], COLORS["CNN"]], height=0.55)
    axes[1].invert_yaxis()
    axes[1].set(title="Final local validation results", xlabel="RMSE", xlim=(0, 0.095))
    for index, value in enumerate(values):
        axes[1].text(value + 0.001, index, f"{value:.5f}", va="center", fontsize=9)
    axes[1].grid(axis="x", color="#e2e8f0", linewidth=0.8)
    axes[1].set_axisbelow(True)
    figure.text(0.02, 0.015, "Source: saved April 2026 experiment. Single models: per-epoch validation. Ensemble: one final TTA evaluation.",
                fontsize=8, color="#64748b")
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.savefig(args.output_dir / "validation_results.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    names, values = list(metrics["task_rmse"]), list(metrics["task_rmse"].values())
    bars = axis.barh(names, values, color="#0d9488", height=0.65)
    axis.invert_yaxis()
    axis.bar_label(bars, fmt="%.5f", padding=5, fontsize=8.5)
    axis.set(title="Ensemble error across 11 morphology tasks", xlabel="Validation RMSE (lower is better)", xlim=(0, 0.14))
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.text(0.02, 0.015, "Each bar uses its task's regression targets. Tasks have different label distributions; this is a descriptive breakdown.",
                fontsize=8, color="#64748b")
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(args.output_dir / "task_errors.png", dpi=180)
    plt.close(figure)
    print(args.output_dir)


if __name__ == "__main__":
    main()

