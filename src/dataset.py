from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import cache_path_for, extract_egemaps


class AudioDataset(Dataset):
    def __init__(
        self,
        labels_df: pd.DataFrame,
        cache_dir: str | Path,
        feature_set: str = "eGeMAPSv02",
        feature_level: str = "LowLevelDescriptors",
        recordings_dir: str | Path | None = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.records = labels_df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.feature_set = feature_set
        self.feature_level = feature_level
        self.recordings_dir = Path(recordings_dir) if recordings_dir else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[idx]
        filename = row["filename"]
        label = float(row["phq9_score"])

        cp = cache_path_for(filename, self.cache_dir)
        if cp.exists():
            features = np.load(cp).astype(np.float32)
        elif self.recordings_dir is not None:
            wav = self.recordings_dir / filename
            features = extract_egemaps(wav, self.feature_set, self.feature_level)
        else:
            raise FileNotFoundError(
                f"No cached features for {filename} and no recordings_dir set."
            )

        x = torch.from_numpy(features)  # (T, 88)
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


def collate_pad(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences to the same length within a batch."""
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max())
    feat_dim = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, feat_dim)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x
    return padded, torch.stack(ys), lengths


def make_loader(
    dataset: AudioDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pad,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
