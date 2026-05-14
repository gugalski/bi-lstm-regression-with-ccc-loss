from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import ccc


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    t = torch.from_numpy(y_true.astype(np.float32))
    p = torch.from_numpy(y_pred.astype(np.float32))
    ccc_val = ccc(t, p).item()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    return {
        "ccc": ccc_val,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
    }


def plot_cv_boxplot(fold_metrics: list[dict[str, float]], out_dir: Path) -> None:
    keys = ["ccc", "mae", "rmse", "r2", "pearson_r", "spearman_rho"]
    data = {k: [fm[k] for fm in fold_metrics] for k in keys}
    fig, axes = plt.subplots(1, len(keys), figsize=(18, 4))
    for ax, k in zip(axes, keys):
        ax.boxplot(data[k])
        ax.set_title(k.upper())
        ax.set_xticks([])
    plt.tight_layout()
    fig.savefig(out_dir / "cv_metrics_boxplot.png", dpi=150)
    plt.close(fig)


def plot_scatter(
    all_true: np.ndarray,
    all_pred: np.ndarray,
    out_dir: Path,
) -> None:
    severity = pd.cut(
        all_true,
        bins=[-1, 4, 9, 14, 27],
        labels=["minimal (0–4)", "mild (5–9)", "moderate (10–14)", "severe (15–27)"],
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    palette = sns.color_palette("tab10", 4)
    for i, band in enumerate(severity.cat.categories):
        mask = severity == band
        ax.scatter(all_true[mask], all_pred[mask], alpha=0.5, s=20, label=band, color=palette[i])
    lim = [0, 27]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True PHQ-9")
    ax.set_ylabel("Predicted PHQ-9")
    ax.set_title("Predicted vs True PHQ-9 (all folds)")
    ax.legend(title="Severity", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter_pred_vs_true.png", dpi=150)
    plt.close(fig)


def plot_error_by_severity(
    all_true: np.ndarray, all_pred: np.ndarray, out_dir: Path
) -> None:
    df = pd.DataFrame({"true": all_true, "pred": all_pred})
    df["severity"] = pd.cut(
        df["true"],
        bins=[-1, 4, 9, 14, 27],
        labels=["0–4", "5–9", "10–14", "15–27"],
    )
    df["abs_err"] = (df["true"] - df["pred"]).abs()
    mae_by_band = df.groupby("severity", observed=True)["abs_err"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    mae_by_band.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("PHQ-9 Severity Band")
    ax.set_ylabel("MAE (PHQ-9 points)")
    ax.set_title("Mean Absolute Error by Severity Band")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "error_by_severity.png", dpi=150)
    plt.close(fig)


def plot_training_curves(
    histories: list[dict[str, list[float]]], out_dir: Path
) -> None:
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for k, history in enumerate(histories):
        axes[0, k].plot(history["train_loss"], label="train")
        axes[0, k].plot(history["val_loss"], label="val")
        axes[0, k].set_title(f"Fold {k + 1} — Loss")
        axes[0, k].legend(fontsize=8)
        axes[0, k].set_xlabel("Epoch")
        axes[1, k].plot(history["train_ccc"], label="train")
        axes[1, k].plot(history["val_ccc"], label="val")
        axes[1, k].set_title(f"Fold {k + 1} — CCC")
        axes[1, k].legend(fontsize=8)
        axes[1, k].set_xlabel("Epoch")
    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def save_cv_report(fold_metrics: list[dict[str, float]], out_dir: Path) -> None:
    df = pd.DataFrame(fold_metrics)
    df.index = [f"fold_{i + 1}" for i in range(len(fold_metrics))]
    summary = df.agg(["mean", "std"])
    report = df.to_string() + "\n\n" + summary.to_string() + "\n"
    (out_dir / "cv_report.txt").write_text(report)
