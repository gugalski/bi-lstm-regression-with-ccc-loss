"""5-fold cross-validation entry-point."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from src.dataset import AudioDataset, make_loader
from src.evaluate import (
    compute_metrics,
    plot_cv_boxplot,
    plot_error_by_severity,
    plot_scatter,
    plot_training_curves,
    save_cv_report,
)
from src.model import LSTMWithAttention
from src.train import train
from src.utils import Config, get_logger, set_seed


def severity_band(score: float) -> str:
    if score <= 4:
        return "minimal"
    if score <= 9:
        return "mild"
    if score <= 14:
        return "moderate"
    return "severe"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("cv", "outputs/logs/cv.log")
    logger.info(f"Device: {device}")

    labels = pd.read_csv("labels.csv")
    labels["band"] = labels["phq9_score"].apply(severity_band)

    groups = labels[cfg.cv.group_col].values
    gkf = GroupKFold(n_splits=cfg.cv.n_splits)

    fold_metrics: list[dict[str, float]] = []
    histories: list[dict[str, list[float]]] = []
    all_true: list[float] = []
    all_pred: list[float] = []

    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(labels, labels["band"], groups)):
        logger.info(f"=== Fold {fold + 1} / {cfg.cv.n_splits} ===")
        ck_path = Path(f"outputs/checkpoints/fold_{fold + 1}/best_model.pt")

        train_df = labels.iloc[train_idx]
        val_df = labels.iloc[val_idx]

        train_ds = AudioDataset(train_df, cfg.features.cache_dir)
        val_ds = AudioDataset(val_df, cfg.features.cache_dir)

        train_loader = make_loader(train_ds, cfg.training.batch_size, shuffle=True)
        val_loader = make_loader(val_ds, cfg.training.batch_size, shuffle=False)

        model = LSTMWithAttention(cfg.model).to(device)

        history = train(
            model, train_loader, val_loader, cfg.training, ck_path, device, logger
        )
        histories.append(history)

        model.load_state_dict(torch.load(ck_path, map_location=device))
        model.eval()

        preds_fold: list[float] = []
        labels_fold: list[float] = []
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                pred, _ = model(x, lengths)
                preds_fold.extend(pred.cpu().tolist())
                labels_fold.extend(y.tolist())

        metrics = compute_metrics(np.array(labels_fold), np.array(preds_fold))
        fold_metrics.append(metrics)
        all_true.extend(labels_fold)
        all_pred.extend(preds_fold)
        logger.info(f"Fold {fold + 1} metrics: {metrics}")

    plot_cv_boxplot(fold_metrics, figures_dir)
    plot_scatter(np.array(all_true), np.array(all_pred), figures_dir)
    plot_error_by_severity(np.array(all_true), np.array(all_pred), figures_dir)
    plot_training_curves(histories, figures_dir)
    save_cv_report(fold_metrics, Path("outputs"))

    logger.info("Cross-validation complete. Results in outputs/")


if __name__ == "__main__":
    main()
