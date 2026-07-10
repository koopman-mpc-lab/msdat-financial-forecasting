import torch
import torch.nn as nn


class _SeqHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, h):
        return self.fc(h[:, -1]).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, c_in, hidden=128, layers=2):
        super().__init__()
        self.rnn = nn.GRU(c_in, hidden, layers, batch_first=True)
        self.head = _SeqHead(hidden)

    def forward(self, x, **kwargs):
        h, _ = self.rnn(x)
        y = self.head(h)
        z = torch.zeros_like(y)
        return y, y, y, z


class LSTMModel(nn.Module):
    def __init__(self, c_in, hidden=128, layers=2):
        super().__init__()
        self.rnn = nn.LSTM(c_in, hidden, layers, batch_first=True)
        self.head = _SeqHead(hidden)

    def forward(self, x, **kwargs):
        h, _ = self.rnn(x)
        y = self.head(h)
        z = torch.zeros_like(y)
        return y, y, y, z


class CNNLSTMModel(nn.Module):
    def __init__(self, c_in, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, 1, batch_first=True)
        self.head = _SeqHead(hidden)

    def forward(self, x, **kwargs):
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(c)
        y = self.head(h)
        z = torch.zeros_like(y)
        return y, y, y, z
