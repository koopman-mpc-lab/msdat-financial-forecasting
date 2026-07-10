import numpy as np
import lightgbm as lgb
import shap

from src.config import FEATURE_COLS


def attention_shap_correlation(attn_weights, X, y, feature_names=None):
    feature_names = feature_names or FEATURE_COLS
    attn_mean = np.mean(np.stack(attn_weights), axis=0)
    attn_rank = np.mean(attn_mean, axis=0)
    model = lgb.LGBMRegressor(n_estimators=500, max_depth=6, verbose=-1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_vals = np.abs(explainer.shap_values(X)).mean(axis=0)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(-attn_rank, -shap_vals)
    return float(rho), float(pval), attn_rank, shap_vals
