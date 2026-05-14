from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 16000


@dataclass
class FeaturesConfig:
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "LowLevelDescriptors"
    cache_dir: str = "outputs/features/"


@dataclass
class ModelConfig:
    input_size: int = 88
    projection_size: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.4
    bidirectional: bool = True


@dataclass
class TrainingConfig:
    loss: str = "ccc"
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    seed: int = 42


@dataclass
class CVConfig:
    n_splits: int = 5
    group_col: str = "subject_id"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls(
            audio=AudioConfig(**raw.get("audio", {})),
            features=FeaturesConfig(**raw.get("features", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            cv=CVConfig(**raw.get("cv", {})),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ccc(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Concordance Correlation Coefficient."""
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var(unbiased=False)
    var_pred = y_pred.var(unbiased=False)
    cov = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    return (2.0 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


class CCCLoss(torch.nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return 1.0 - ccc(y_true.float(), y_pred.float())


def build_loss(loss_name: str) -> torch.nn.Module:
    if loss_name == "ccc":
        return CCCLoss()
    if loss_name == "huber":
        return torch.nn.HuberLoss()
    if loss_name == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"Unknown loss: {loss_name}")
