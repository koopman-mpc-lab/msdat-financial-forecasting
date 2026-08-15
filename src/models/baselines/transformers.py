from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Head(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(h)
        return {"point": out[:, 0], "q10": out[:, 1], "q90": out[:, 2], "dir_logit": out[:, 3]}


def _encoder(d_model: int, n_layers: int, n_heads: int, ffn_dim: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
        dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class iTransformer(nn.Module):
    """Variate-token Transformer: attend over the C feature channels."""

    def __init__(self, n_features=12, lookback=60, d_model=256, n_layers=4, n_heads=8, ffn_dim=1024, dropout=0.1, **_):
        super().__init__()
        self.lookback = lookback
        self.embed = nn.Linear(lookback, d_model)
        self.var_pos = nn.Parameter(torch.zeros(1, n_features, d_model))
        self.encoder = _encoder(d_model, n_layers, n_heads, ffn_dim, dropout)
        self.head = _Head(d_model)

    def forward(self, x, scales=None, vol=None):
        # x: (B, L, C) -> (B, C, L)
        tokens = self.embed(x.transpose(1, 2)[:, :, : self.lookback])
        if tokens.size(-1) != self.var_pos.size(-1):
            tokens = tokens
        h = self.encoder(tokens + self.var_pos[:, : tokens.size(1)])
        return self.head(h.mean(dim=1))


class PatchTST(nn.Module):
    def __init__(
        self, n_features=12, d_model=256, n_layers=4, n_heads=8, ffn_dim=1024,
        patch_len=12, stride=8, dropout=0.1, max_patches=16, **_,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.embed = nn.Linear(patch_len * n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.encoder = _encoder(d_model, n_layers, n_heads, ffn_dim, dropout)
        self.head = _Head(d_model)

    def forward(self, x, scales=None, vol=None):
        b, length, c = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)
        if patches.size(1) == 0:
            pad = F.pad(x, (0, 0, 0, self.patch_len - length))
            patches = pad.unfold(1, self.patch_len, self.stride)
        tokens = self.embed(patches.reshape(b, patches.size(1), -1))
        h = self.encoder(tokens + self.pos[:, : tokens.size(1)])
        return self.head(h.mean(dim=1))


class Informer(nn.Module):
    def __init__(self, n_features=12, d_model=256, n_layers=4, n_heads=8, ffn_dim=1024, dropout=0.1, max_len=128, **_):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.encoder = _encoder(d_model, n_layers, n_heads, ffn_dim, dropout)
        self.distill = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.head = _Head(d_model)

    def forward(self, x, scales=None, vol=None):
        h = self.input_proj(x) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        h = self.distill(h.transpose(1, 2)).transpose(1, 2)
        return self.head(h[:, -1])


class Autoformer(nn.Module):
    def __init__(self, n_features=12, d_model=256, n_layers=4, n_heads=8, ffn_dim=1024, dropout=0.1, max_len=128, **_):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.season_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.encoder = _encoder(d_model, n_layers, n_heads, ffn_dim, dropout)
        self.decomp = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.head = _Head(d_model)

    def forward(self, x, scales=None, vol=None):
        trend = self.decomp(x.transpose(1, 2)).transpose(1, 2)[:, : x.size(1)]
        if trend.size(1) != x.size(1):
            trend = F.pad(trend, (0, 0, 0, x.size(1) - trend.size(1)))[:, : x.size(1)]
        season = x - trend
        h = self.input_proj(season) + self.season_proj(trend) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        return self.head(h[:, -1])


class TimesNet(nn.Module):
    def __init__(self, n_features=12, d_model=256, n_layers=4, n_kernels=3, mixer_hidden=1536, dropout=0.1, **_):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList()
        self.mixers = nn.ModuleList()
        for _ in range(n_layers):
            convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=k, padding=k // 2),
                        nn.GELU(),
                        nn.Conv2d(16, 1, kernel_size=1),
                    )
                    for k in range(3, 3 + 2 * n_kernels, 2)
                ]
            )
            self.blocks.append(convs)
            self.mixers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, mixer_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mixer_hidden, d_model),
            ))
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.head = _Head(d_model)

    def forward(self, x, scales=None, vol=None):
        h = self.input_proj(x)
        for convs, mixer in zip(self.blocks, self.mixers):
            z = h.unsqueeze(1)
            acc = 0
            for conv in convs:
                acc = acc + conv(z)
            h = self.norm(h + acc.squeeze(1))
            h = h + mixer(h)
        h = self.out_proj(h)
        return self.head(h[:, -1])


class NHiTS(nn.Module):
    def __init__(self, n_features=12, lookback=60, d_model=128, n_blocks=3, n_layers_per_block=2, hidden=256, dropout=0.1, **_):
        super().__init__()
        self.lookback = lookback
        self.blocks = nn.ModuleList()
        in_dim = lookback * n_features
        for _ in range(n_blocks):
            layers = []
            last = in_dim
            for _ in range(n_layers_per_block):
                layers.extend([nn.Linear(last, hidden), nn.ReLU(), nn.Dropout(dropout)])
                last = hidden
            layers.append(nn.Linear(hidden, d_model))
            self.blocks.append(nn.Sequential(*layers))
        self.head = _Head(d_model)

    def forward(self, x, scales=None, vol=None):
        flat = x[:, : self.lookback].reshape(x.size(0), -1)
        h = 0
        for block in self.blocks:
            h = h + block(flat)
        return self.head(h)
