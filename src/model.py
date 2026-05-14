from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import ModelConfig


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: (B, T, H)
        scores = self.v(torch.tanh(self.W(hidden))).squeeze(-1)  # (B, T)

        mask = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
        mask = mask >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        alpha = F.softmax(scores, dim=-1)  # (B, T)
        context = (alpha.unsqueeze(-1) * hidden).sum(dim=1)  # (B, H)
        return context, alpha


class LSTMWithAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(cfg.input_size, cfg.projection_size)
        self.input_norm = nn.LayerNorm(cfg.projection_size)

        self.lstm = nn.LSTM(
            input_size=cfg.projection_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )

        lstm_out_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.attention = BahdanauAttention(lstm_out_size)
        self.norm = nn.LayerNorm(lstm_out_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(lstm_out_size, 1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 88), lengths: (B,)
        x = self.input_norm(self.projection(x))

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        context, alpha = self.attention(out, lengths)
        context = self.dropout(self.norm(context))
        pred = self.head(context).squeeze(-1)  # (B,)
        pred = pred.clamp(0.0, 27.0)
        return pred, alpha
