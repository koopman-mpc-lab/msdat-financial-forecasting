# MSDAT financial forecasting

Code for the Multi-Scale Decomposition and Dual-Attention Transformer
experiments on SPX, CSI 300, NASDAQ-20, and BTC-USD.

The repository contains the model, baseline implementations, data
preparation, training, evaluation, and the analysis utilities used for
ablation, multi-horizon, sensitivity, Diebold–Mariano, cost, and
attention–SHAP comparisons. Raw market data, processed windows,
checkpoints, logs, and generated result tables are not distributed.

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Data preparation

Place locally obtained OHLCV files at `data/raw/<dataset>/<symbol>.csv`.
Dataset names are `SPX`, `CSI300`, `NASDAQ-20`, and `BTC-USD`. Each CSV
must contain:

```text
date,open,high,low,close,volume
```

An optional `turnover` column is used as the liquidity feature when
available. Build cached, chronologically split lookback windows with:

```bash
python scripts/prepare_data.py --dataset SPX
```

The generated `.npz` files are written under `data/features/`. Min–max
statistics are fit on the training partition only.

## Usage

Forward-pass check:

```bash
python scripts/smoke_forward.py
```

Train a model:

```bash
python scripts/train.py --model MSDAT --dataset SPX --seed 42
```

Evaluate a locally produced checkpoint:

```bash
python scripts/eval_checkpoint.py \
  --ckpt checkpoints/msdat_spx_s42/best.pt \
  --dataset SPX --split test --fast
```

Recompute metrics from a prediction CSV:

```bash
python scripts/evaluate.py --pred path/to/predictions.csv
```

Experiment entry points:

```bash
python scripts/run_experiments.py --datasets SPX --models MSDAT iTransformer --seeds 42
python scripts/run_ablation.py --dataset SPX --seed 42
python scripts/run_multi_horizon.py --dataset SPX --seed 42
python scripts/run_sensitivity.py --dataset SPX
python scripts/run_dm_test.py --pred-a path/to/msdat.csv --pred-b path/to/baseline.csv
python scripts/run_cost_benchmark.py --dataset SPX
python scripts/run_explainability.py --ckpt checkpoints/msdat_spx_s42/best.pt --dataset SPX
```

The default protocol is configured in `configs/default.yaml`. Dataset
metadata and experiment grids are in `configs/`.

## Layout

- `src/models/msdat.py` — MSDAT architecture
- `src/models/baselines/` — statistical, recurrent, and Transformer baselines
- `src/decomposition/` — window-local CEEMDAN and scale grouping
- `src/data/` — feature construction and cached-window datasets
- `src/training/` — composite loss, trainer, and metrics
- `src/analysis/` — statistical, cost, and attribution utilities
- `scripts/` — preparation, training, evaluation, and experiment runners
