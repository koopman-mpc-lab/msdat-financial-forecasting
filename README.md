# MSDAT

Minimal implementation of a Multi-Scale Decomposition and Dual-Attention Transformer for financial time-series forecasting.

The code includes the model, baseline runners, training utilities, evaluation metrics, ablation scripts, multi-horizon experiments, Diebold-Mariano tests, and explainability analysis. Data acquisition and preprocessing scripts are intentionally not included.

## Installation

```bash
pip install -r requirements.txt
```

## Data

Place preprocessed CSV files in `data/processed/`:

- `spx.csv`
- `csi300.csv`
- `nasdaq20.csv`
- `btc_usd.csv`

Required columns:

```text
open, high, low, close, volume, ma5, ma10, ma20, rsi14, macd_hist, bb_width, liquidity
```

`close_raw` is optional and is used for original-scale metrics when available.

## Usage

Train MSDAT on one dataset:

```bash
python scripts/train.py --dataset spx --model msdat --seed 42
```

Run selected experiments:

```bash
python scripts/run_experiments.py --datasets spx --models msdat itransformer
python scripts/run_ablation.py --dataset spx
python scripts/run_multi_horizon.py --dataset spx
```

Configuration is stored in `configs/default.yaml`. Outputs are written to `outputs/`.
