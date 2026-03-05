# BTC 1-Minute Scalping System

XGBoost-based breakout anticipation system for BTCUSDT perpetual futures.
Predicts UP / DOWN / NO_BREAK on 1-minute candles using 107 features across
volatility, volume, CVD/orderflow, VWAP, volume profile (POC/VAH/VAL), price levels, EMAs, and multi-timeframe context.

---

## What It Does

1. **Trains** a walk-forward XGBoost classifier on 1m Binance candle data (2023-2025)
2. **Evaluates** signal quality and precision on a held-out test split
3. **Backtests** event-driven paper trades with ATR-based stop/take-profit
4. **Simulates** live trading via Binance WebSocket — terminal mode or browser dashboard

---

## Requirements

- Python 3.10+
- Windows/Linux/macOS

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd Trader2

# 2. Create and activate virtual environment
python -m venv trader2
# Windows
trader2\Scripts\activate
# Linux/macOS
source trader2/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies (from requirements.txt)

| Package | Purpose |
|---|---|
| xgboost | Model training and inference |
| pandas | Data loading and feature engineering |
| numpy | Numerical operations |
| scikit-learn | Walk-forward CV, metrics |
| matplotlib / seaborn | Evaluation plots |
| optuna | Hyperparameter search (optional) |
| websockets / aiohttp | Live Binance WebSocket feed |
| pyyaml | Config file parsing |
| joblib | Artifact serialization |
| tqdm | Progress bars |
| pyarrow | Parquet support |

---

## Data

Data is NOT included in the repo (too large). Place Binance 1m/5m/15m OHLCV CSVs here:

```
Data/BTCUSDT/full_year/
    2023_1m.csv
    2023_5m.csv
    2023_15m.csv
    2024_1m.csv   (etc.)
    2025_1m.csv   (etc.)
```

**CSV format** — standard Binance 12-column export:

```
open_time, open, high, low, close, volume, close_time,
quote_vol, num_trades, taker_buy_vol, taker_buy_quote_vol, ignore
```

Download from [Binance Vision](https://data.binance.vision/) or via the Binance API.

---

## Configuration

All hyperparameters live in `config.yaml`. Key sections:

| Section | Controls |
|---|---|
| `data` | Symbol, years, timeframes, multi-coin list |
| `labels` | Lookback, horizon, ATR breakout threshold, RR ratio, fees |
| `features` | ATR spans, BB/Keltner periods, EMA periods, CVD windows |
| `training` | Test size, walk-forward folds, XGBoost params, class weights |
| `trading` | Entry probability thresholds, SL/TP in ATR, time stop, cooldown |
| `simulation` | Replay file, speed multiplier, WebSocket URL, log paths |

---

## Usage

All commands assume you are in the project root with the venv active.

### 1. Train

```bash
python train.py
```

- Loads 3 years of 1m data (~1.5M bars)
- Runs 5-fold walk-forward cross-validation
- Trains final model on full training split
- Saves artifacts to `models/`:
  - `xgb_model.json` — trained model
  - `feature_columns.json` — feature list (order matters)
  - `thresholds.json` — per-class probability thresholds
  - `label_meta.json` — label distribution stats

Runtime: ~3-5 minutes on a modern CPU.

### 2. Evaluate

```bash
python eval.py
```

Outputs classification report, signal quality metrics, and saves plots to `models/`:
- `confusion_matrix.png`
- `feature_importance.png`
- `threshold_curve.png`
- `calibration.png`

### 3. Backtest

```bash
python backtest.py
```

Event-driven backtest on the held-out test split using the same FeatureEngine as live.
Results saved to `models/`:
- `backtest_report.json` — win rate, Sharpe, max drawdown, PnL
- `backtest_equity.csv` / `backtest_equity.png`
- `backtest_trades.csv`

### 4. Replay Simulation — Terminal

```bash
python sim/sim_replay.py
```

Replays a CSV file through the full pipeline at configurable speed.
Logs trades to `sim/logs/replay_log.jsonl`.

### 5. Replay Simulation — Browser Dashboard

```bash
python sim/sim_replay_ui.py
python sim/sim_replay_ui.py --data Data/BTCUSDT/monthly/2026-02_1m.csv
python sim/sim_replay_ui.py --speed 0                          # max speed
python sim/sim_replay_ui.py --start 2025-06-01 --end 2025-07-01
```

Same as above, with a live browser UI at **http://localhost:8080**:
- Candlestick chart with VWAP, Volume, CVD sub-panels
- Fixed Range Volume Profile overlay (POC / VAH / VAL)
- Volume spike bubbles, trade entry/exit arrows, in-trade candle highlighting
- Running trade log with PnL and probability
- Replay progress bar; waits for browser connection before starting

### 6. Live Paper Trade — Terminal

```bash
python sim/sim_live_binance.py
```

Connects to Binance WebSocket, pre-warms the feature engine from recent REST candles,
then trades live candles in paper mode. Logs to `sim/logs/live_log.jsonl`.

### 7. Live Paper Trade — Browser Dashboard

```bash
python sim/sim_live_ui.py
python sim/sim_live_ui.py --symbol ETHUSDT
```

Same as above, plus the full browser UI at **http://localhost:8080**:
- Live candlestick chart with VWAP, Volume, CVD, Volume Profile overlay
- Trade entry/exit arrows, in-trade blue candles, running trade log
- Internal WebSocket on port 8765

---

## Project Structure

```
config.yaml             All hyperparameters
requirements.txt        Python dependencies
data.py                 load_csv(), load_all(), candle_from_dict()
features.py             compute_features() + FeatureEngine (live incremental)
labels.py               compute_labels() -> 3-class labels + fakeout columns
train.py                Walk-forward CV + final model + artifact export
eval.py                 Classification report, signal quality, plots
backtest.py             Event-driven vectorised backtest
models/                 Saved artifacts (generated by train.py)
sim/
    __init__.py
    portfolio.py        Portfolio, Position, Trade dataclasses
    execution.py        ExecutionEngine (signal -> order -> trade log)
    replay_feed.py      ReplayFeed (CSV row iterator)
    binance_ws_feed.py  BinanceWSFeed (async WS) + SyncBinanceWSFeed
    sim_replay.py       Historical paper-trade simulation (terminal)
    sim_replay_ui.py    Historical paper-trade simulation + browser dashboard
    sim_live_binance.py Live paper-trade (terminal)
    sim_live_ui.py      Live paper-trade + browser dashboard
    static/
        index.html      Browser UI (Lightweight Charts, dark theme, VP overlay)
    logs/               Trade logs (generated at runtime, not tracked)
Data/                   OHLCV CSVs (not tracked in git)
```

---

## Label Design

Labels are computed over a 30-candle lookahead horizon:

- **UP_BREAK_SOON (2)** — price breaks above the 30-candle range high by `k * ATR` and hits TP before SL within 30 bars
- **DOWN_BREAK_SOON (0)** — price breaks below the 30-candle range low by `k * ATR` and hits SL before TP within 30 bars
- **NO_BREAK (1)** — neither condition met within the horizon

Distribution (3-year, sl_atr=30, rr=3.0): DOWN ~0.7%, NO_BREAK ~98.5%, UP ~0.8%.
Break classes are upweighted (12x) during training to compensate for class imbalance.

---

## Notes

- The model uses **limit orders only** (maker fees, no slippage assumed)
- SL and TP distances are in ATR multiples — set in `config.yaml` under both `labels` and `trading` (must match)
- FeatureEngine requires ~70 candles of warmup before producing valid features
- XGBoost handles residual NaN values natively; no imputation needed
