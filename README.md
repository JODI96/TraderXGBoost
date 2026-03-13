# BTC 1-Minute Breakout Detection System

An end-to-end ML trading research system for BTCUSDT perpetual futures on the 1-minute timeframe. The project covers the full pipeline: data engineering, feature construction, supervised labeling, model training, rigorous backtesting, and a live paper-trading infrastructure with a browser dashboard.

---

## Problem Statement

Short-term price breakouts from consolidation ranges are difficult to predict reliably because:

- The raw **class imbalance** is severe: breakout events occur on fewer than 8% of bars.
- Apparent breakouts frequently **reverse immediately** (fakeouts), making naive breakout chasing unprofitable.
- Standard ML approaches train on all bars equally, but the signal-to-noise ratio is much higher **near range boundaries** — the only location where a breakout can actually occur.

This project addresses all three problems through label design, training-set filtering, and a dedicated fakeout class.

---

## Architecture

```
Raw 1m OHLCV  ──►  features.py  ──►  XGBoost (5-class softprob)
               (98 features)         p_up_clean / p_down_clean /
                                     p_up_fakeout / p_down_fakeout /
                                     p_no_break
                                              │
                         ┌────────────────────▼────────────────────┐
                         │  p_up_clean  > T_up   →  LONG           │
                         │  p_dn_clean  > T_dn   →  SHORT          │
                         │  p_fakeout   > T_fo   →  suppress entry │
                         │  else                 →  FLAT (wait)    │
                         └────────────────────┬────────────────────┘
                                              │
                       SL = −0.15% · price    │   TP = +0.45% · price  (1:3 RR)
                       Time stop after 20 bars if neither hit
```

---

## Feature Engineering

98 features computed causally from 1-minute OHLCV data (no lookahead). Multi-timeframe features (5m, 15m) are derived by resampling the 1m buffer inside `compute_features()` — no separate feeds, so train and live inference stay in exact sync.

| Group | Count | Description |
|---|---|---|
| ATR / Volatility | 10 | Fast/slow ATR, Bollinger Band width/percentile, Keltner squeeze, vol regime |
| Volume / Order Flow | 6 | Relative volume, z-score, delta ratio, CVD fast/slow/candle-strength |
| VWAP | 3 | Rolling VWAP distance, slope, absolute slope |
| Range / Levels | 27 | Range width, percentile, distance to high/low, boundary touch counts, rejection wicks, boundary volume surge — across 3 lookback windows (8, 20, 60 bars) |
| EMA / Momentum | 22 | EMA 9/21/50/200 slopes and distances, fast EMAs (3, 5), crossovers, price momentum, acceleration, CVD momentum |
| MTF — 5m | 10 | Resampled EMA slopes/trend/ATR/range distances/CVD/delta ratio |
| MTF — 15m | 10 | Same as 5m at the 15-minute timeframe |
| Regime / Alignment | 10 | 30-bar trend regime (slope + range), multi-timeframe trend agreement, CVD-trend alignment and strength |

**Key design choices:**
- All features are causal (computed from past data only — no bar-close lookahead).
- XGBoost uses tree splits, so no feature normalisation is needed.
- A single `compute_features()` function is shared between training, backtesting, and live inference, eliminating train/live skew.
- `FeatureEngine` maintains a rolling 80-bar buffer for incremental live updates.

---

## Label Design

Labels are computed using a **dual-barrier method** over a 20-candle lookahead window:

```
Entry price + 0.45%  ──  Take-Profit barrier
Entry price − 0.15%  ──  Stop-Loss barrier
```

Each bar receives one of **5 classes**:

| Class | ID | Condition |
|---|---|---|
| `DOWN_CLEAN`   | 0 | SL hit before TP; no reversal after SL |
| `NO_BREAK`     | 1 | Neither barrier reached within 20 bars |
| `UP_CLEAN`     | 2 | TP hit before SL; no reversal after TP |
| `UP_FAKEOUT`   | 3 | TP hit before SL, but price later reverses through SL |
| `DOWN_FAKEOUT` | 4 | SL hit before TP, but price later reverses through TP |

The 5-class design lets the model distinguish a clean breakout from a fakeout rather than treating them identically. At inference time, a high `p_fakeout` score suppresses entries even when `p_clean` exceeds the threshold.

**Near-range training filter:** Only bars within 3× ATR of the range boundary are included in training. This removes the large majority of trivially-easy `NO_BREAK` bars deep inside a range, forcing the model to focus on the scenario where entry decisions actually matter. This is the most impactful single change in the pipeline — it is what produces consistent positive profit factor across all three test years.

---

## Training Pipeline

```
train.py
  ├── load_all()             — load 3 years of BTCUSDT 1m data (~1.49M bars)
  ├── compute_features()     — batch feature computation
  ├── compute_labels()       — 5-class dual-barrier labeling
  ├── near-range filter      — keep only bars within 3×ATR of boundary
  ├── 5-fold walk-forward CV — temporally ordered splits, 30-bar gap between train/val
  │     └── XGBClassifier    — multi:softprob, early stopping, GPU (CUDA)
  ├── final model            — trained on full dataset
  └── artifacts →  models/
        ├── xgb_model.json
        ├── feature_columns.json   (order matters for inference)
        ├── thresholds.json
        └── label_meta.json
```

**XGBoost configuration (regularised to prevent overfit on imbalanced data):**

| Parameter | Value | Rationale |
|---|---|---|
| `max_depth` | 2 | Shallow trees → high bias, low variance |
| `min_child_weight` | 100 | Requires strong signal before each split |
| `subsample` | 0.30 | Row subsampling for diversity |
| `colsample_bytree` | 0.30 | Feature subsampling for diversity |
| `gamma` | 5 | Minimum gain required to create a split |
| `reg_alpha` | 5.0 | L1 sparsity |
| `break_weight` | 8.0 | Sample weight ratio for break vs no-break classes |
| `n_estimators` | up to 10,000 | Early-stopped per fold (avg ~3,700) |

Walk-forward CV ensures no future data contaminates any validation fold. The 30-bar gap between train and validation prevents the 20-bar label horizon from creating lookahead at fold boundaries.

---

## Backtesting

`backtest.py` runs an event-driven simulation on the held-out test split (20% of data, chronologically last):

- Uses the same `compute_features()` path as training — no separate backtest feature logic.
- Simulates SL/TP/time-stop execution bar-by-bar.
- Outputs: equity curve, per-trade log, summary metrics (profit factor, win rate, Sharpe, max drawdown).
- A Monte Carlo module (`deep_analysis.py`) samples trade sequences to estimate drawdown probability distributions.

---

## Simulation Infrastructure

The project includes a full paper-trading stack built on top of the model:

| Module | Description |
|---|---|
| `sim/replay_feed.py` | CSV row iterator with configurable speed multiplier |
| `sim/binance_ws_feed.py` | Async Binance WebSocket feed + synchronous wrapper |
| `sim/portfolio.py` | Portfolio / Position / Trade dataclasses |
| `sim/binance_portfolio.py` | Real Binance Futures order execution |
| `sim/execution.py` | ExecutionEngine: signal → order → trade log |
| `sim/sim_replay.py` | Historical replay in terminal |
| `sim/sim_replay_ui.py` | Historical replay with browser dashboard |
| `sim/sim_live_binance.py` | Live paper trading in terminal |
| `sim/sim_live_ui.py` | Live paper trading with browser dashboard |

**Browser dashboard** (Lightweight Charts, dark theme):
- Candlestick chart with VWAP overlay
- Volume histogram + CVD sub-chart
- Fixed Range Volume Profile (POC / VAH / VAL) toggle
- Trade entry/exit markers with direction and model probability
- In-trade candle highlighting
- Live trade log with running PnL and win rate
- Replay progress bar and session summary

The live feed pre-warms the `FeatureEngine` from recent REST candles on startup so inference begins on bar 1 without waiting for a 70-bar warmup period.

---

## Requirements

- Python 3.10+
- NVIDIA GPU recommended (CUDA) for training; CPU fallback works
- Internet access for live trading (Binance WebSocket)
- Windows / Linux / macOS

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/JODI96/TraderXGBoost.git
cd TraderXGBoost

# 2. Create virtual environment
python -m venv trader2

# Windows
trader2\Scripts\activate
# Linux / macOS
source trader2/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Paper trade with browser dashboard — open http://localhost:8080
python sim/sim_live_ui.py
```

> The trained model is included in `models/` — paper trading works immediately after cloning without retraining.

---

## Data

Data is not included (too large). Place Binance OHLCV CSVs here:

```
Data/
└── BTCUSDT/
    └── full_year/
        ├── 2023_1m.csv
        ├── 2023_5m.csv
        ├── 2023_15m.csv
        ├── 2024_1m.csv
        ├── 2024_5m.csv
        ├── 2024_15m.csv
        ├── 2025_1m.csv
        └── 2025_5m.csv  ...
```

Standard Binance 12-column kline format. Column 10 (`taker_buy_vol`) is used as the real delta source for CVD features — this avoids the common proxy approximation of using `(volume × (close − low) / (high − low))`.

Download from [Binance Vision](https://data.binance.vision/) or via the Binance API.

---

## Project Structure

```
C:/Trader2/
├── config.yaml               All hyperparameters — single source of truth
├── requirements.txt
├── .env.example              API key template (copy to .env for live trading)
│
├── data.py                   load_csv(), load_all(), candle_from_dict()
├── features.py               compute_features() + FeatureEngine (live incremental)
├── labels.py                 compute_labels() → 5-class labels + barrier metadata
├── train.py                  Walk-forward CV + final model + artifact export
├── eval.py                   Classification report, signal quality, plots
├── backtest.py               Event-driven vectorised backtest
├── deep_analysis.py          Monte Carlo drawdown analysis + threshold sweep
├── cache.py                  Feature/label caching for faster iteration
│
├── models/                   Generated by train.py
│   ├── xgb_model.json
│   ├── feature_columns.json
│   ├── thresholds.json
│   ├── label_meta.json
│   ├── backtest_report.json
│   ├── backtest_trades.csv
│   └── backtest_equity.png
│
├── sim/
│   ├── portfolio.py          Portfolio, Position, Trade dataclasses (paper)
│   ├── binance_portfolio.py  Real Binance Futures order execution
│   ├── execution.py          ExecutionEngine: signal → order → trade log
│   ├── replay_feed.py        ReplayFeed: CSV row iterator
│   ├── binance_ws_feed.py    BinanceWSFeed (async) + SyncBinanceWSFeed
│   ├── sim_replay.py         Historical paper-trade simulation (terminal)
│   ├── sim_replay_ui.py      Historical paper-trade simulation + browser UI
│   ├── sim_live_binance.py   Live paper trading (terminal only)
│   ├── sim_live_ui.py        Live paper trading + browser dashboard
│   ├── sim_live_trade.py     Live real-money trading + browser dashboard
│   ├── static/
│   │   └── index.html        Browser UI (Lightweight Charts, dark theme)
│   └── logs/                 Trade logs written at runtime
│
└── Data/                     OHLCV CSVs — not tracked in git
```

---

## Commands

All commands assume project root with the venv active.

### Train

```bash
./trader2/Scripts/python.exe train.py
```

Saves to `models/`: `xgb_model.json`, `feature_columns.json`, `thresholds.json`, `label_meta.json`.

### Evaluate

```bash
./trader2/Scripts/python.exe eval.py
```

Outputs classification report, confusion matrix, feature importance, threshold curve, calibration plots, and `signal_quality.csv`.

### Backtest

```bash
./trader2/Scripts/python.exe backtest.py
```

Outputs `backtest_report.json`, `backtest_trades.csv`, `backtest_equity.png`.

### Replay Simulation

```bash
# Terminal
./trader2/Scripts/python.exe sim/sim_replay.py --data Data/BTCUSDT/full_year/2025_1m.csv

# Browser dashboard
./trader2/Scripts/python.exe sim/sim_replay_ui.py --data Data/BTCUSDT/full_year/2025_1m.csv

# Max speed (no delay)
./trader2/Scripts/python.exe sim/sim_replay_ui.py --speed 0

# Date range
./trader2/Scripts/python.exe sim/sim_replay_ui.py --start 2025-01-01 --end 2025-06-01
```

### Live Paper Trading

```bash
# Terminal
./trader2/Scripts/python.exe sim/sim_live_binance.py

# Browser dashboard — open http://localhost:8080
./trader2/Scripts/python.exe sim/sim_live_ui.py
```

### Live Real Trading

```bash
cp .env.example .env
# Fill in BINANCE_API_KEY and BINANCE_API_SECRET
./trader2/Scripts/python.exe sim/sim_live_trade.py
```

> Start with a small `position_size_pct` in `config.yaml` and verify behaviour in paper mode first.

---

## Configuration Reference

All hyperparameters live in `config.yaml` with inline documentation and tuning ranges.

| Section | Key parameters |
|---|---|
| `data` | `symbol`, `years` |
| `labels` | `L_range`, `H_horizon`, `k_atr`, `sl_pct`, `tp_pct` |
| `features` | `atr_short/long`, `ema_periods`, `cvd_window`, `vwap_window`, `range_lookbacks` |
| `training` | `n_wf_splits`, `near_range_atr`, `break_weight`, `fakeout_weight`, `xgb_params` |
| `trading` | `T_up`, `T_down`, `T_fakeout_suppress`, `sl_pct`, `tp_pct`, `time_stop`, `cooldown` |
| `sweep` | `T_values`, `leverages`, `n_sims` (Monte Carlo paths) |

> `labels.sl_pct` / `labels.tp_pct` must match `trading.sl_pct` / `trading.tp_pct` — the label barriers define what the model learns, and the trade execution must match.

---

## Technical Notes

- **No lookahead:** Every feature is computed from strictly past data. The label horizon is gapped from validation folds during CV to prevent indirect future leakage.
- **Train/live parity:** `compute_features()` is called identically in training, backtesting, and live inference. `FeatureEngine` wraps the same function for incremental per-bar updates.
- **MTF without extra feeds:** 5m and 15m features are derived by causal resampling of the 1m buffer — no separate WebSocket streams needed in production.
- **Real delta source:** CVD is computed from `taker_buy_vol` (column 10 of Binance klines), not the common proxy formula. This gives the actual signed order flow for each bar.
- **NaN handling:** XGBoost handles residual NaN values in features natively. No imputation is applied.
