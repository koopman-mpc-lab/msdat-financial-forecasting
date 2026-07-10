import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    def __init__(self, l_q=0.3, l_d=0.5, l_v=0.1):
        super().__init__()
        self.l_q = l_q
        self.l_d = l_d
        self.l_v = l_v
        self.mse = nn.MSELoss()

    def pinball(self, pred, target, q):
        err = target - pred
        return torch.max(q * err, (q - 1) * err).mean()

    def forward(self, outputs, batch, recent_preds=None):
        point, q_low, q_high, direction = outputs[:4]
        y = batch["y"]
        p_t = batch["p_t"]
        y_raw = batch["y_raw"]
        l_point = self.mse(point, y)
        l_q = self.pinball(q_low, y, 0.1) + self.pinball(q_high, y, 0.9)
        sign = torch.sign(y_raw - p_t)
        l_d = torch.relu(-direction * sign).mean()
        l_v = torch.tensor(0.0, device=y.device)
        if recent_preds is not None and len(recent_preds) >= 5:
            pred_vol = torch.std(recent_preds[-5:])
            real_vol = batch["vol"].mean()
            l_v = (pred_vol - real_vol).pow(2)
        return l_point + self.l_q * l_q + self.l_d * l_d + self.l_v * l_v
