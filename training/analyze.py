"""
EcoAudit — Model Performance Analysis
Reads training artifacts and runs test-set evaluation to produce
a comprehensive analysis report with charts.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
RUNS_DIR    = BASE / "runs/detect/runs"
TRAIN_RUN   = RUNS_DIR / "ecoaudit_v17"
CSV_PATH    = TRAIN_RUN / "results.csv"
MODEL_PATH  = TRAIN_RUN / "weights/best.pt"
DATA_YAML   = BASE / "dataset/data.yaml"
OUT_DIR     = RUNS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Battery", "Biological", "Cable", "Can", "Cardboard",
    "Clothes", "E-waste", "Glass", "Medical Waste", "Metal",
    "Paper", "Plastic", "Shoes", "Trash",
]

CLASS_COLORS = [
    "#FF6347","#3CB371","#4682B4","#FFA500","#8B5A2B",
    "#DA70D6","#FFD700","#6495ED","#DC143C","#A9A9A9",
    "#90EE90","#1E90FF","#FFB6C1","#800080",
]

STYLE = {
    "bg": "#0F1117", "panel": "#1E2130", "accent": "#4CAF50",
    "text": "#E0E0E0", "muted": "#888888", "grid": "#2A2D3A",
}

plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["panel"],
    "axes.edgecolor":    STYLE["grid"],
    "axes.labelcolor":   STYLE["text"],
    "xtick.color":       STYLE["muted"],
    "ytick.color":       STYLE["muted"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "sans-serif",
    "font.size":         10,
})


# ── 1. Load training CSV ───────────────────────────────────────────────────────
def load_training_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Replace inf in box_loss with NaN for cleaner plots
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ── 2. Run test-set evaluation ─────────────────────────────────────────────────
def run_test_eval(model_path: Path, data_yaml: Path):
    model = YOLO(str(model_path))
    print("\n[1/4] Running validation on TEST split...")
    metrics = model.val(data=str(data_yaml), split="test", verbose=False)
    return metrics


def run_val_eval(model_path: Path, data_yaml: Path):
    model = YOLO(str(model_path))
    print("[2/4] Running validation on VAL split...")
    metrics = model.val(data=str(data_yaml), split="val", verbose=False)
    return metrics


# ── 3. Chart helpers ──────────────────────────────────────────────────────────
def _ax_style(ax, title, xlabel="Epoch", ylabel=None):
    ax.set_title(title, color=STYLE["text"], fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=STYLE["muted"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=STYLE["muted"], fontsize=9)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── 4. Training curves chart ──────────────────────────────────────────────────
def plot_training_curves(df: pd.DataFrame, out_path: Path):
    fig = plt.figure(figsize=(18, 10), facecolor=STYLE["bg"])
    fig.suptitle(
        "EcoAudit v17 — Training Curves (50 Epochs)",
        color=STYLE["text"], fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    epochs = df["epoch"]

    # 1 — Losses (train)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, df["train/box_loss"], color="#FF6B6B", lw=2, label="Box")
    ax.plot(epochs, df["train/cls_loss"], color="#4ECDC4", lw=2, label="Cls")
    ax.plot(epochs, df["train/dfl_loss"], color="#FFE66D", lw=2, label="DFL")
    ax.legend(fontsize=8, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    _ax_style(ax, "Train Loss", ylabel="Loss")

    # 2 — Losses (val)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, df["val/box_loss"], color="#FF6B6B", lw=2, label="Box")
    ax.plot(epochs, df["val/cls_loss"], color="#4ECDC4", lw=2, label="Cls")
    ax.plot(epochs, df["val/dfl_loss"], color="#FFE66D", lw=2, label="DFL")
    ax.legend(fontsize=8, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    _ax_style(ax, "Validation Loss", ylabel="Loss")

    # 3 — Precision & Recall (val)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, df["metrics/precision(B)"], color="#4CAF50", lw=2, label="Precision")
    ax.plot(epochs, df["metrics/recall(B)"],    color="#2196F3", lw=2, label="Recall")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    _ax_style(ax, "Precision & Recall (Val)", ylabel="Score")

    # 4 — mAP50
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, df["metrics/mAP50(B)"], color="#FF9800", lw=2.5)
    best_idx = df["metrics/mAP50(B)"].idxmax()
    ax.scatter(
        epochs[best_idx], df["metrics/mAP50(B)"][best_idx],
        color="white", zorder=5, s=60,
        label=f'Best: {df["metrics/mAP50(B)"][best_idx]:.4f} @ ep {epochs[best_idx]}'
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    _ax_style(ax, "mAP@0.5 (Val)", ylabel="mAP")

    # 5 — mAP50-95
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, df["metrics/mAP50-95(B)"], color="#9C27B0", lw=2.5)
    best_idx2 = df["metrics/mAP50-95(B)"].idxmax()
    ax.scatter(
        epochs[best_idx2], df["metrics/mAP50-95(B)"][best_idx2],
        color="white", zorder=5, s=60,
        label=f'Best: {df["metrics/mAP50-95(B)"][best_idx2]:.4f} @ ep {epochs[best_idx2]}'
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    _ax_style(ax, "mAP@0.5:0.95 (Val)", ylabel="mAP")

    # 6 — Learning rate
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(epochs, df["lr/pg0"], color="#00BCD4", lw=2)
    _ax_style(ax, "Learning Rate Schedule", ylabel="LR")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"    Saved → {out_path.name}")


# ── 5. Per-class bar chart ────────────────────────────────────────────────────
def plot_per_class(metrics_test, metrics_val, out_path: Path):
    # Extract per-class metrics from YOLO results object
    # metrics.box.maps  → mAP50:95 per class
    # metrics.box.ap50  → AP@50 per class
    test_ap50    = np.array(metrics_test.box.ap50)
    test_maps    = np.array(metrics_test.box.maps)
    test_p       = np.array(metrics_test.box.p)
    test_r       = np.array(metrics_test.box.r)
    val_ap50     = np.array(metrics_val.box.ap50)
    val_maps     = np.array(metrics_val.box.maps)

    n = len(CLASSES)
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), facecolor=STYLE["bg"])
    fig.suptitle(
        "EcoAudit v17 — Per-Class Performance (Test Set vs Val Set)",
        color=STYLE["text"], fontsize=14, fontweight="bold", y=0.99,
    )

    # AP@50
    ax = axes[0]
    bars_test = ax.bar(x - w/2, test_ap50, w, label="Test AP@50",  color=[CLASS_COLORS[i] for i in range(n)], alpha=0.9)
    bars_val  = ax.bar(x + w/2, val_ap50,  w, label="Val AP@50",   color=[CLASS_COLORS[i] for i in range(n)], alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    ax.axhline(np.mean(test_ap50), color="white", lw=1.2, linestyle="--",
               label=f"Mean Test AP@50: {np.mean(test_ap50):.3f}")
    _ax_style(ax, "AP@0.50 per Class", xlabel="", ylabel="AP@50")
    for bar in bars_test:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=7, color=STYLE["muted"])

    # AP@50:95
    ax = axes[1]
    ax.bar(x - w/2, test_maps, w, label="Test mAP@50:95",
           color=[CLASS_COLORS[i] for i in range(n)], alpha=0.9)
    ax.bar(x + w/2, val_maps,  w, label="Val mAP@50:95",
           color=[CLASS_COLORS[i] for i in range(n)], alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    ax.axhline(np.mean(test_maps), color="white", lw=1.2, linestyle="--",
               label=f"Mean: {np.mean(test_maps):.3f}")
    _ax_style(ax, "AP@0.50:0.95 per Class", xlabel="", ylabel="mAP@50:95")

    # Precision & Recall (test)
    ax = axes[2]
    ax.bar(x - w/2, test_p, w, label="Precision", color="#4CAF50", alpha=0.85)
    ax.bar(x + w/2, test_r, w, label="Recall",    color="#2196F3", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    _ax_style(ax, "Precision & Recall per Class (Test Set)", ylabel="Score")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"    Saved → {out_path.name}")


# ── 6. Summary scorecard ──────────────────────────────────────────────────────
def plot_scorecard(df: pd.DataFrame, metrics_test, metrics_val, out_path: Path):
    fig = plt.figure(figsize=(16, 5), facecolor=STYLE["bg"])
    fig.suptitle(
        "EcoAudit v17 — Performance Scorecard",
        color=STYLE["text"], fontsize=14, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    metrics_list = [
        ("Val mAP@50",     df["metrics/mAP50(B)"].max(),      "#FF9800"),
        ("Val mAP@50:95",  df["metrics/mAP50-95(B)"].max(),   "#9C27B0"),
        ("Test mAP@50",    float(metrics_test.box.map50),      "#4CAF50"),
        ("Test mAP@50:95", float(metrics_test.box.map),        "#2196F3"),
    ]

    for i, (label, value, color) in enumerate(metrics_list):
        ax = fig.add_subplot(gs[i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        # Outer ring (background)
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(0.5 + 0.38*np.cos(theta), 0.5 + 0.38*np.sin(theta),
                color=STYLE["grid"], lw=8, solid_capstyle="round")

        # Filled arc
        theta_fill = np.linspace(np.pi/2, np.pi/2 - value*2*np.pi, 300)
        ax.plot(0.5 + 0.38*np.cos(theta_fill), 0.5 + 0.38*np.sin(theta_fill),
                color=color, lw=8, solid_capstyle="round")

        ax.text(0.5, 0.54, f"{value:.1%}", ha="center", va="center",
                fontsize=20, fontweight="bold", color=color)
        ax.text(0.5, 0.3, label, ha="center", va="center",
                fontsize=10, color=STYLE["muted"])

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"    Saved → {out_path.name}")


# ── 7. Val vs Test comparison table ───────────────────────────────────────────
def plot_comparison_table(metrics_test, metrics_val, out_path: Path):
    test_p    = np.array(metrics_test.box.p)
    test_r    = np.array(metrics_test.box.r)
    test_ap50 = np.array(metrics_test.box.ap50)
    test_map  = np.array(metrics_test.box.maps)
    val_p     = np.array(metrics_val.box.p)
    val_r     = np.array(metrics_val.box.r)
    val_ap50  = np.array(metrics_val.box.ap50)
    val_map   = np.array(metrics_val.box.maps)

    col_labels = ["Class", "Test P", "Test R", "Test AP50", "Test mAP", "Val AP50", "Val mAP"]
    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append([
            cls,
            f"{test_p[i]:.3f}",   f"{test_r[i]:.3f}",
            f"{test_ap50[i]:.3f}", f"{test_map[i]:.3f}",
            f"{val_ap50[i]:.3f}",  f"{val_map[i]:.3f}",
        ])
    rows.append([
        "MEAN",
        f"{np.mean(test_p):.3f}",   f"{np.mean(test_r):.3f}",
        f"{np.mean(test_ap50):.3f}", f"{np.mean(test_map):.3f}",
        f"{np.mean(val_ap50):.3f}",  f"{np.mean(val_map):.3f}",
    ])

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=STYLE["bg"])
    ax.axis("off")
    fig.suptitle(
        "EcoAudit v17 — Per-Class Metrics Table (Val vs Test)",
        color=STYLE["text"], fontsize=13, fontweight="bold",
    )

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(STYLE["panel"])
        cell.set_edgecolor(STYLE["grid"])
        cell.set_text_props(color=STYLE["text"])
        if row == 0:
            cell.set_facecolor(STYLE["bg"])
            cell.set_text_props(color=STYLE["accent"], fontweight="bold")
        elif row == len(CLASSES) + 1:   # MEAN row
            cell.set_facecolor("#1A2A1A")
            cell.set_text_props(color=STYLE["accent"], fontweight="bold")
        if col == 0 and row > 0:
            cell.set_text_props(color=CLASS_COLORS[(row - 1) % len(CLASS_COLORS)], fontweight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"    Saved → {out_path.name}")


# ── 8. Print text summary ──────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, metrics_test, metrics_val):
    best_map50_val = df["metrics/mAP50(B)"].max()
    best_map_val   = df["metrics/mAP50-95(B)"].max()
    best_ep_50     = df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"]
    best_ep_5095   = df.loc[df["metrics/mAP50-95(B)"].idxmax(), "epoch"]
    final_p        = df["metrics/precision(B)"].iloc[-1]
    final_r        = df["metrics/recall(B)"].iloc[-1]

    test_p    = np.array(metrics_test.box.p)
    test_r    = np.array(metrics_test.box.r)
    test_ap50 = np.array(metrics_test.box.ap50)
    test_map  = np.array(metrics_test.box.maps)

    sep = "=" * 62
    print(f"\n{sep}")
    print("  EcoAudit v17 — Model Performance Analysis Report")
    print(sep)
    print(f"\n  Model   : YOLOv11n (nano)  |  Epochs: 50  |  Img: 640")
    print(f"  Classes : 14  |  Optimizer: Auto  |  Batch: 16")
    print(f"\n{'─'*62}")
    print("  TRAINING SUMMARY (validation split)")
    print(f"{'─'*62}")
    print(f"  Best mAP@50      : {best_map50_val:.4f}  (epoch {int(best_ep_50)})")
    print(f"  Best mAP@50:95   : {best_map_val:.4f}  (epoch {int(best_ep_5095)})")
    print(f"  Final Precision  : {final_p:.4f}")
    print(f"  Final Recall     : {final_r:.4f}")
    print(f"\n{'─'*62}")
    print("  TEST SET EVALUATION")
    print(f"{'─'*62}")
    print(f"  mAP@50           : {metrics_test.box.map50:.4f}")
    print(f"  mAP@50:95        : {metrics_test.box.map:.4f}")
    print(f"  Mean Precision   : {np.mean(test_p):.4f}")
    print(f"  Mean Recall      : {np.mean(test_r):.4f}")
    print(f"\n{'─'*62}")
    print("  PER-CLASS TEST AP@50")
    print(f"{'─'*62}")
    sorted_idx = np.argsort(test_ap50)[::-1]
    for i in sorted_idx:
        bar_len = int(test_ap50[i] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {CLASSES[i]:<15} {bar}  {test_ap50[i]:.3f}")
    print(f"\n{'─'*62}")
    print("  WEAKEST CLASSES (Test AP@50 < 0.85)")
    print(f"{'─'*62}")
    weak = [(CLASSES[i], test_ap50[i]) for i in range(len(CLASSES)) if test_ap50[i] < 0.85]
    if weak:
        for cls, score in sorted(weak, key=lambda x: x[1]):
            print(f"  ⚠  {cls:<15} AP@50={score:.3f}")
    else:
        print("  ✓  All classes AP@50 ≥ 0.85")
    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n══════════════════════════════════════════════")
    print("  EcoAudit — Model Performance Analysis")
    print("══════════════════════════════════════════════")

    df = load_training_csv(CSV_PATH)

    metrics_test = run_test_eval(MODEL_PATH, DATA_YAML)
    metrics_val  = run_val_eval(MODEL_PATH, DATA_YAML)

    print("\n[3/4] Generating charts...")
    plot_training_curves(df,                        OUT_DIR / "1_training_curves.png")
    plot_scorecard(df, metrics_test, metrics_val,   OUT_DIR / "2_scorecard.png")
    plot_per_class(metrics_test, metrics_val,       OUT_DIR / "3_per_class_metrics.png")
    plot_comparison_table(metrics_test, metrics_val, OUT_DIR / "4_metrics_table.png")

    print(f"\n[4/4] All charts saved to:\n      {OUT_DIR}\n")
    print_summary(df, metrics_test, metrics_val)


if __name__ == "__main__":
    os.chdir(BASE)
    main()
