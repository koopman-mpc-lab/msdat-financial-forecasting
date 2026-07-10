import torch
import torch.nn as nn

from src.models.msdat import MSDAT
from src.models.baselines.rnn import GRUModel, LSTMModel, CNNLSTMModel
from src.models.baselines.transformers import (
    InformerModel, AutoformerModel, TimesNetModel,
    NHiTSModel, PatchTSTModel, iTransformerModel,
)


def build_model(name, cfg):
    name = name.lower()
    c_in = cfg["num_features"]
    lookback = cfg["lookback"]
    if name == "msdat":
        return MSDAT(cfg)
    if name == "gru":
        return GRUModel(c_in, hidden=128, layers=2)
    if name == "lstm":
        return LSTMModel(c_in, hidden=128, layers=2)
    if name == "cnn_lstm":
        return CNNLSTMModel(c_in, hidden=128)
    if name == "informer":
        return InformerModel(c_in, lookback, cfg["embed_dim"], cfg["num_heads"], cfg["num_layers"])
    if name == "autoformer":
        return AutoformerModel(c_in, lookback, cfg["embed_dim"], cfg["num_heads"], cfg["num_layers"])
    if name == "timesnet":
        return TimesNetModel(c_in, lookback, cfg["embed_dim"])
    if name == "nhits":
        return NHiTSModel(c_in, lookback)
    if name == "patchtst":
        return PatchTSTModel(c_in, lookback, cfg["embed_dim"], cfg["num_heads"], cfg["num_layers"])
    if name == "itransformer":
        return iTransformerModel(c_in, lookback, cfg["embed_dim"], cfg["num_heads"], cfg["num_layers"])
    raise ValueError(f"Unknown model: {name}")
