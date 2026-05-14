import torch
import pytest

from src.model import LSTMWithAttention
from src.utils import ModelConfig


@pytest.fixture
def cfg():
    return ModelConfig(
        input_size=88, projection_size=64, hidden_size=64, num_layers=1, dropout=0.0, bidirectional=True
    )


def test_forward_shape(cfg):
    model = LSTMWithAttention(cfg)
    B, T = 4, 50
    x = torch.randn(B, T, cfg.input_size)
    lengths = torch.full((B,), T, dtype=torch.long)
    preds, alpha = model(x, lengths)
    assert preds.shape == (B,)
    assert alpha.shape == (B, T)


def test_output_clamped(cfg):
    model = LSTMWithAttention(cfg)
    B, T = 8, 30
    x = torch.randn(B, T, cfg.input_size) * 100
    lengths = torch.full((B,), T, dtype=torch.long)
    preds, _ = model(x, lengths)
    assert preds.min().item() >= 0.0
    assert preds.max().item() <= 27.0


def test_variable_lengths(cfg):
    model = LSTMWithAttention(cfg)
    lengths = torch.tensor([50, 30, 10])
    B = len(lengths)
    T = int(lengths.max())
    x = torch.zeros(B, T, cfg.input_size)
    for i, l in enumerate(lengths):
        x[i, :l] = torch.randn(l, cfg.input_size)
    preds, alpha = model(x, lengths)
    assert preds.shape == (B,)
