import math
import torch
import torch.nn as nn


class _BaseTransformer(nn.Module):
    def __init__(self, c_in, lookback, d_model, nhead, num_layers, ffn_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ffn_dim, 0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, 1)

    def encode(self, x):
        h = self.input_proj(x) + self.pos[:, : x.size(1)]
        return self.encoder(h)

    def predict(self, h):
        return self.head(h[:, -1]).squeeze(-1)

    def forward(self, x, **kwargs):
        y = self.predict(self.encode(x))
        z = torch.zeros_like(y)
        return y, y, y, z


class InformerModel(_BaseTransformer):
    def __init__(self, c_in, lookback, d_model, nhead, num_layers):
        super().__init__(c_in, lookback, d_model, nhead, num_layers)
        self.distil = nn.AvgPool1d(2)


class AutoformerModel(_BaseTransformer):
    def __init__(self, c_in, lookback, d_model, nhead, num_layers):
        super().__init__(c_in, lookback, d_model, nhead, num_layers)
        self.decomp = nn.AvgPool1d(3, stride=1, padding=1)

    def encode(self, x):
        trend = self.decomp(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        h = self.input_proj(seasonal) + self.pos[:, : x.size(1)]
        return self.encoder(h) + self.input_proj(trend)


class TimesNetModel(nn.Module):
    def __init__(self, c_in, lookback, d_model, k=3):
        super().__init__()
        self.k = k
        self.convs = nn.ModuleList([
            nn.Conv1d(c_in, d_model, 3, padding=1) for _ in range(k)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, **kwargs):
        xc = x.transpose(1, 2)
        feats = [conv(xc).transpose(1, 2) for conv in self.convs]
        h = torch.stack(feats, dim=0).mean(0)
        y = self.head(h[:, -1]).squeeze(-1)
        z = torch.zeros_like(y)
        return y, y, y, z


class NHiTSModel(nn.Module):
    def __init__(self, c_in, lookback, hidden=128, stacks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c_in * lookback, hidden),
                nn.ReLU(),
                nn.Linear(hidden, lookback),
            )
            for _ in range(stacks)
        ])
        self.head = nn.Linear(lookback, 1)

    def forward(self, x, **kwargs):
        flat = x.reshape(x.size(0), -1)
        residual = x[:, :, 3]
        for block in self.blocks:
            residual = residual + block(flat).view_as(residual)
        y = self.head(residual).squeeze(-1)
        z = torch.zeros_like(y)
        return y, y, y, z


class PatchTSTModel(_BaseTransformer):
    def __init__(self, c_in, lookback, d_model, nhead, num_layers, patch=6):
        super().__init__(c_in, lookback, d_model, nhead, num_layers)
        self.patch = patch
        self.patch_proj = nn.Linear(c_in * patch, d_model)

    def encode(self, x):
        b, l, c = x.shape
        p = self.patch
        n = l // p
        x = x[:, : n * p].reshape(b, n, p * c)
        h = self.patch_proj(x) + self.pos[:, :n]
        return self.encoder(h)


class iTransformerModel(nn.Module):
    def __init__(self, c_in, lookback, d_model, nhead, num_layers, ffn_dim=256):
        super().__init__()
        self.var_proj = nn.Linear(lookback, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ffn_dim, 0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(c_in * d_model, 1)

    def forward(self, x, **kwargs):
        v = self.var_proj(x.transpose(1, 2))
        h = self.encoder(v)
        y = self.head(h.reshape(h.size(0), -1)).squeeze(-1)
        z = torch.zeros_like(y)
        return y, y, y, z
