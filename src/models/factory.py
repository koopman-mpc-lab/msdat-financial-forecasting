from __future__ import annotations

from typing import Any

from .baselines import (
    Autoformer,
    CNNLSTM,
    GRUForecaster,
    Informer,
    LSTMForecaster,
    NHiTS,
    PatchTST,
    TimesNet,
    iTransformer,
)
from .msdat import MSDAT

MODEL_ALIASES = {
    "MSDAT": "msdat",
    "MSDAT (ours)": "msdat",
    "GRU": "gru",
    "LSTM": "lstm",
    "CNN-LSTM": "cnn_lstm",
    "N-HiTS": "nhits",
    "iTransformer": "itransformer",
    "PatchTST": "patchtst",
    "TimesNet": "timesnet",
    "Informer": "informer",
    "Autoformer": "autoformer",
    "ARIMA(1,1,1)": "arima",
    "ARIMA": "arima",
}


def build_model(name: str, cfg: dict[str, Any], extra: dict[str, Any] | None = None):
    key = MODEL_ALIASES.get(name, name.lower().replace("-", "_"))
    extra = extra or {}
    n_features = cfg.get("n_features", 12)
    lookback = cfg.get("lookback", 60)
    msdat_cfg = dict(cfg.get("msdat", {}))
    msdat_cfg.update({k: extra[k] for k in extra if k in {
        "n_scales", "use_channel_attention", "temporal", "fixed_fusion",
    }})
    if key == "msdat":
        return MSDAT(
            n_features=n_features,
            n_scales=extra.get("n_scales", cfg.get("n_scales", 3)),
            **{k: v for k, v in msdat_cfg.items() if k in {
                "d_model", "n_layers", "n_heads", "ffn_dim", "dropout",
                "max_len", "fusion_hidden", "use_channel_attention",
                "temporal", "fixed_fusion",
            }},
        )
    bcfg = cfg.get("baselines", {})
    if key == "gru":
        return GRUForecaster(n_features=n_features, **bcfg.get("gru", {}))
    if key == "lstm":
        return LSTMForecaster(n_features=n_features, **bcfg.get("lstm", {}))
    if key == "cnn_lstm":
        return CNNLSTM(n_features=n_features, **bcfg.get("cnn_lstm", {}))
    if key == "nhits":
        return NHiTS(n_features=n_features, lookback=lookback, **bcfg.get("nhits", {}))
    if key == "itransformer":
        return iTransformer(n_features=n_features, lookback=lookback, **bcfg.get("itransformer", {}))
    if key == "patchtst":
        return PatchTST(n_features=n_features, **bcfg.get("patchtst", {}))
    if key == "timesnet":
        return TimesNet(n_features=n_features, **bcfg.get("timesnet", {}))
    if key == "informer":
        return Informer(n_features=n_features, **bcfg.get("informer", {}))
    if key == "autoformer":
        return Autoformer(n_features=n_features, **bcfg.get("autoformer", {}))
    if key == "arima":
        from .baselines.arima import ARIMABaseline
        return ARIMABaseline()
    raise ValueError(f"Unknown model: {name}")
