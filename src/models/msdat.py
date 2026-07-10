import math
import numpy as np
import torch
import torch.nn as nn

from src.decomposition.ceemdan import decompose_window


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ChannelAttention(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features * 2, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        w = self.net(torch.cat([mean, std], dim=-1))
        return x * w.unsqueeze(1), w


class TemporalEncoder(nn.Module):
    def __init__(self, c_in, d_model, nhead, num_layers, ffn_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(c_in, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ffn_dim, dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        h = self.pos(self.input_proj(x))
        return self.encoder(h)


class MSDAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg["num_features"]
        d = cfg["embed_dim"]
        self.num_scales = cfg["num_scales"]
        self.tau1 = cfg["tau1"]
        self.tau2 = cfg["tau2"]
        self.ensemble = cfg["ceemdan_ensemble"]
        self.use_decomp = cfg.get("use_decomposition", True)
        self.use_channel_attn = cfg.get("use_channel_attention", True)
        self.use_temporal_attn = cfg.get("use_temporal_attention", True)
        self.use_gating = cfg.get("use_gating", True)
        self.temporal_backend = cfg.get("temporal_backend", "transformer")
        self.channel_attns = nn.ModuleList([ChannelAttention(c) for _ in range(self.num_scales)])
        if self.use_temporal_attn:
            if self.temporal_backend == "gru":
                self.temporal_encoders = nn.ModuleList([
                    nn.GRU(c, d, batch_first=True) for _ in range(self.num_scales)
                ])
            else:
                self.temporal_encoders = nn.ModuleList([
                    TemporalEncoder(c, d, cfg["num_heads"], cfg["num_layers"], cfg["ffn_dim"], cfg["dropout"])
                    for _ in range(self.num_scales)
                ])
        else:
            self.temporal_encoders = nn.ModuleList([
                nn.GRU(c, d, batch_first=True) for _ in range(self.num_scales)
            ])
        self.gate = nn.Linear(d * self.num_scales + 1, self.num_scales)
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self._attn_weights = []
        self._gate_weights = []

    def _build_branches(self, x, close_window):
        b, l, c = x.shape
        branches = []
        for i in range(b):
            xi = x[i]
            close = close_window[i].detach().cpu().numpy()
            if self.use_decomp:
                recon = decompose_window(
                    close, self.ensemble, self.tau1, self.tau2, self.num_scales
                )
            else:
                recon = [close] * self.num_scales
            scale_inputs = []
            for s in range(self.num_scales):
                branch = xi.clone()
                branch[:, 3] = torch.from_numpy(recon[s].astype(np.float32)).to(x.device)
                scale_inputs.append(branch)
            branches.append(torch.stack(scale_inputs))
        return torch.stack(branches)

    def forward(self, x, vol=None, close_window=None, return_aux=False):
        if close_window is None:
            close_window = x[..., 3]
        branch_x = self._build_branches(x, close_window)
        b, s, l, c = branch_x.shape
        hs = []
        attn_list = []
        for si in range(s):
            bx = branch_x[:, si]
            if self.use_channel_attn:
                bx, aw = self.channel_attns[si](bx)
                attn_list.append(aw)
            enc = self.temporal_encoders[si]
            if self.use_temporal_attn and self.temporal_backend != "gru":
                h = enc(bx)
            else:
                h, _ = enc(bx)
            hs.append(h)
        pooled = [h.mean(dim=1) for h in hs]
        if vol is None:
            vol = torch.zeros(b, 1, device=x.device)
        else:
            vol = vol.view(b, 1)
        if self.use_gating:
            gate_in = torch.cat(pooled + [vol], dim=-1)
            alpha = torch.softmax(self.gate(gate_in), dim=-1)
            h_stack = torch.stack(hs, dim=1)
            alpha_exp = alpha.unsqueeze(-1).unsqueeze(-1)
            fused = (h_stack * alpha_exp).sum(dim=1)
        else:
            alpha = torch.full((b, s), 1.0 / s, device=x.device)
            fused = torch.stack(hs, dim=1).mean(dim=1)
        out = self.head(fused[:, -1])
        point, q_low, q_high, direction = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        if return_aux:
            return point, q_low, q_high, direction, attn_list, alpha
        return point, q_low, q_high, direction
