from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, n_features: int = 12):
        super().__init__()
        hidden = max(n_features // 2, 1)
        self.fc1 = nn.Linear(2 * n_features, hidden)
        self.fc2 = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C)
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        attn = torch.sigmoid(self.fc2(F.relu(self.fc1(torch.cat([mean, std], dim=-1)))))
        return x * attn.unsqueeze(1), attn


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        max_len: int,
        temporal: str = "attention",
    ):
        super().__init__()
        self.temporal = temporal
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        if temporal == "gru":
            self.encoder = nn.GRU(
                d_model, d_model, num_layers=n_layers,
                batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(d_model)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.size(1)]
        if self.temporal == "gru":
            h, _ = self.encoder(h)
            return self.norm(h)
        return self.encoder(h)


class VolatilityGate(nn.Module):
    def __init__(self, d_model: int, n_scales: int):
        super().__init__()
        self.proj = nn.Linear(n_scales * d_model + 1, n_scales)

    def forward(self, branch_h: torch.Tensor, vol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # branch_h: (B, S, L, D)
        pooled = branch_h.mean(dim=2)
        feat = torch.cat([pooled.flatten(1), vol.view(-1, 1)], dim=-1)
        alpha = torch.softmax(self.proj(feat), dim=-1)
        fused = (branch_h * alpha.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return fused, alpha


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(h)
        return {
            "point": out[:, 0],
            "q10": out[:, 1],
            "q90": out[:, 2],
            "dir_logit": out[:, 3],
        }


class MSDAT(nn.Module):
    def __init__(
        self,
        n_features: int = 12,
        n_scales: int = 3,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        fusion_hidden: int = 1024,
        use_channel_attention: bool = True,
        temporal: str = "attention",
        fixed_fusion: bool = False,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.n_features = n_features
        self.use_channel_attention = use_channel_attention
        self.fixed_fusion = fixed_fusion
        self.channel = nn.ModuleList(
            [ChannelAttention(n_features) for _ in range(n_scales)]
        )
        self.encoders = nn.ModuleList(
            [
                TemporalEncoder(
                    n_features, d_model, n_layers, n_heads, ffn_dim,
                    dropout, max_len, temporal=temporal,
                )
                for _ in range(n_scales)
            ]
        )
        self.gate = VolatilityGate(d_model, n_scales)
        self.fusion_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, d_model),
        )
        self.head = PredictionHead(d_model)
        self.close_index = 3

    def _assemble_branches(self, x: torch.Tensor, scales: torch.Tensor | None) -> torch.Tensor:
        # x: (B, L, C); scales: (B, S, L) reconstructions of the close channel
        if scales is None:
            close = x[:, :, self.close_index]
            scales = close.unsqueeze(1).expand(-1, self.n_scales, -1)
        if scales.size(1) != self.n_scales:
            if scales.size(1) == 1:
                scales = scales.expand(-1, self.n_scales, -1)
            elif scales.size(1) > self.n_scales:
                scales = scales[:, : self.n_scales]
            else:
                pad = scales[:, -1:].expand(-1, self.n_scales - scales.size(1), -1)
                scales = torch.cat([scales, pad], dim=1)
        branches = []
        for s in range(self.n_scales):
            branch = x.clone()
            branch[:, :, self.close_index] = scales[:, s]
            branches.append(branch)
        return torch.stack(branches, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        scales: torch.Tensor | None = None,
        vol: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        branches = self._assemble_branches(x, scales)
        encoded = []
        attns = []
        for s in range(self.n_scales):
            xb = branches[:, s]
            if self.use_channel_attention:
                xb, attn = self.channel[s](xb)
            else:
                attn = torch.ones(x.size(0), x.size(-1), device=x.device, dtype=x.dtype) / x.size(-1)
            encoded.append(self.encoders[s](xb))
            attns.append(attn)
        branch_h = torch.stack(encoded, dim=1)
        if vol is None:
            vol = x.new_zeros(x.size(0))
        if self.fixed_fusion or self.n_scales == 1:
            alpha = torch.full(
                (x.size(0), self.n_scales), 1.0 / self.n_scales,
                device=x.device, dtype=x.dtype,
            )
            fused = branch_h.mean(dim=1)
        else:
            fused, alpha = self.gate(branch_h, vol)
        fused = fused + self.fusion_ffn(fused)
        out = self.head(fused[:, -1])
        out["alpha"] = alpha
        out["channel_attn"] = torch.stack(attns, dim=1)
        return out


def count_params(module: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return {
        "trainable": int(trainable),
        "frozen_est": int(frozen),
        "total_est": int(trainable + frozen),
    }


def format_param_line(stats: dict[str, int]) -> str:
    return (
        f"Trainable {stats['trainable']:,} | "
        f"frozen_est {stats['frozen_est']:,} | "
        f"total_est {stats['total_est']:,}"
    )
