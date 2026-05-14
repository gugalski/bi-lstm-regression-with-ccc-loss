import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import AudioDataset, collate_pad, make_loader


@pytest.fixture
def tmp_cache(tmp_path):
    for i in range(3):
        arr = np.random.randn(100, 88).astype(np.float32)
        np.save(tmp_path / f"00{i + 1}.npy", arr)
    return tmp_path


@pytest.fixture
def labels_df():
    return pd.DataFrame(
        {
            "filename": ["001.wav", "002.wav", "003.wav"],
            "subject_id": [1, 2, 3],
            "phq9_score": [5, 12, 20],
        }
    )


def test_dataset_len(tmp_cache, labels_df):
    ds = AudioDataset(labels_df, tmp_cache)
    assert len(ds) == 3


def test_dataset_item_shapes(tmp_cache, labels_df):
    ds = AudioDataset(labels_df, tmp_cache)
    x, y = ds[0]
    assert x.shape == (100, 88)
    assert y.shape == ()


def test_collate_pad():
    xs = [torch.randn(50, 88), torch.randn(30, 88), torch.randn(70, 88)]
    ys = [torch.tensor(5.0), torch.tensor(12.0), torch.tensor(20.0)]
    padded, labels, lengths = collate_pad(list(zip(xs, ys)))
    assert padded.shape == (3, 70, 88)
    assert lengths.tolist() == [50, 30, 70]


def test_make_loader(tmp_cache, labels_df):
    ds = AudioDataset(labels_df, tmp_cache)
    loader = make_loader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    x, y, lengths = batch
    assert x.ndim == 3
    assert y.ndim == 1
