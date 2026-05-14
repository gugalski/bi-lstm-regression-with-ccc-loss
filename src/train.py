from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import TrainingConfig, build_loss, ccc


class EarlyStopping:
    def __init__(self, patience: int, checkpoint_path: str | Path) -> None:
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.best_ccc = float("-inf")
        self.counter = 0
        self.stopped = False

    def step(self, val_ccc: float, model: nn.Module) -> bool:
        if val_ccc > self.best_ccc:
            self.best_ccc = val_ccc
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds, _ = model(x, lengths)
                loss = criterion(preds, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds, _ = model(x, lengths)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(y)
        all_preds.append(preds.detach())
        all_labels.append(y.detach())

    n = len(loader.dataset)  # type: ignore[arg-type]
    epoch_loss = total_loss / n
    epoch_ccc = ccc(torch.cat(all_labels), torch.cat(all_preds)).item()
    return epoch_loss, epoch_ccc


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        preds, _ = model(x, lengths)
        loss = criterion(preds, y)
        total_loss += loss.item() * len(y)
        all_preds.append(preds)
        all_labels.append(y)

    n = len(loader.dataset)  # type: ignore[arg-type]
    epoch_loss = total_loss / n
    epoch_ccc = ccc(torch.cat(all_labels), torch.cat(all_preds)).item()
    return epoch_loss, epoch_ccc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    checkpoint_path: str | Path,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, list[float]]:
    criterion = build_loss(cfg.loss)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(cfg.early_stopping_patience, checkpoint_path)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_ccc": [],
        "val_loss": [],
        "val_ccc": [],
    }

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_ccc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_ccc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_ccc)

        history["train_loss"].append(tr_loss)
        history["train_ccc"].append(tr_ccc)
        history["val_loss"].append(vl_loss)
        history["val_ccc"].append(vl_ccc)

        logger.info(
            f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} train_ccc={tr_ccc:.4f} "
            f"val_loss={vl_loss:.4f} val_ccc={vl_ccc:.4f}"
        )

        if early_stopping.step(vl_ccc, model):
            logger.info(f"Early stopping at epoch {epoch} (best val CCC={early_stopping.best_ccc:.4f})")
            break

    return history
