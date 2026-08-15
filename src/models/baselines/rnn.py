from __future__ import annotations

import torch
import torch.nn as nn


class _SeqHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(h)
        return {"point": out[:, 0], "q10": out[:, 1], "q90": out[:, 2], "dir_logit": out[:, 3]}


class GRUForecaster(nn.Module):
    def __init__(self, n_features: int = 12, hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        self.rnn = nn.GRU(
            n_features, hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = _SeqHead(hidden_size)

    def forward(self, x, scales=None, vol=None):
        h, _ = self.rnn(x)
        return self.head(h[:, -1])


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int = 12, hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        self.rnn = nn.LSTM(
            n_features, hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = _SeqHead(hidden_size)

    def forward(self, x, scales=None, vol=None):
        h, _ = self.rnn(x)
        return self.head(h[:, -1])


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 12,
        conv_channels: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=pad),
            nn.GELU(),
        )
        self.rnn = nn.LSTM(
            conv_channels, hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = _SeqHead(hidden_size)

    def forward(self, x, scales=None, vol=None):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.head(h[:, -1])
