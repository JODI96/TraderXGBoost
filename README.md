# BTC 1-Minute Scalping System

XGBoost-based breakout anticipation system for BTCUSDT (and multi-coin) perpetual futures.
Predicts **UP / DOWN / NO_BREAK** on 1-minute candles using **91 features** across ATR/volatility,
volume, CVD/orderflow, VWAP, price levels/range, EMAs, multi-timeframe context (5m + 15m), and time-of-day.

**Best result — T=0.67, 30x leverage, over 218 days:**

| Metric | Value |
|---|---|
| Total Return (218 days) | **+630.6%** |
| Win Rate | 51.3% |
| Profit Factor | 1.97 |
| Max Drawdown | −35.3% |
| Trades | 78 over 54 active days |
| P(−30% drawdown in 1 yr) | 45.3% |
| P(−50% drawdown in 1 yr) | 21.6% |
| P(−80% drawdown in 1 yr) | 3.5% |

<details>
<summary><strong>Full threshold + leverage sweep (20,000-path Monte Carlo risk)</strong></summary>

```
====================================================================================================
  SWEEP  |  test=218d  |  MC risk: 20,000 paths/scenario  |  P(-X%) = prob capital drops >=X% at any point in 1 yr
====================================================================================================
      T    Ret%    Ann%   Trd  Days  Trd/d   Win%     PF   MaxDD%     P(-30%)   P(-50%)   P(-80%)

  ──────────────────────────────────────────────────────────────────────────────────────────────────
  Leverage 10x  (TP/trade=4.50%  SL/trade=1.500%)
  ──────────────────────────────────────────────────────────────────────────────────────────────────
  0.58   -75.8%  -126.9%   863   209   3.96   32.3%   0.84    -78.8%      99.9%    97.3%    30.8%
  0.60   -48.5%   -81.3%   565   181   2.59   33.3%   0.90    -57.6%      96.9%    86.0%    20.0%
  0.62   +13.9%   +23.3%   346   133   1.59   35.8%   1.04    -36.6%      53.6%    22.4%     0.4%
  0.64   +60.3%  +100.9%   205    97   0.94   40.0%   1.23    -22.6%       9.3%     0.9%     0.0%  ***
  0.66   +99.8%  +167.1%   109    69   0.50   47.7%   1.70    -18.4%       0.5%     0.0%     0.0%  ***
  0.67  +107.1%  +179.4%    78    54   0.36   51.3%   2.07    -13.4%       0.1%     0.0%     0.0%  ***
  0.68   +52.4%   +87.8%    62    43   0.28   46.8%   1.70    -11.1%       0.4%     0.0%     0.0%  ***
  0.69   +30.3%   +50.7%    43    33   0.20   48.8%   1.64    -11.6%       0.3%     0.0%     0.0%  *
  0.70   +26.4%   +44.3%    26    23   0.12   53.8%   2.00     -8.2%       0.0%     0.0%     0.0%

  ──────────────────────────────────────────────────────────────────────────────────────────────────
  Leverage 20x  (TP/trade=9.00%  SL/trade=3.000%)
  ──────────────────────────────────────────────────────────────────────────────────────────────────
  0.58   -96.6%  -161.7%   863   209   3.96   32.3%   0.77    -97.3%     100.0%   100.0%    72.6%
  0.60   -81.6%  -136.7%   565   181   2.59   33.3%   0.86    -86.3%      96.5%    86.1%    27.5%
  0.62    +3.0%    +5.0%   346   133   1.59   35.8%   1.00    -60.8%      80.5%    60.0%    15.6%
  0.64  +123.2%  +206.3%   205    97   0.94   40.0%   1.21    -41.3%      48.6%    25.0%     2.9%  ***
  0.66  +267.4%  +447.6%   109    69   0.50   47.7%   1.68    -34.2%      16.8%     3.9%     0.1%  ***
  0.67  +301.5%  +504.8%    78    54   0.36   51.3%   2.02    -25.1%      11.4%     1.8%     0.0%  ***
  0.68  +121.0%  +202.7%    62    43   0.28   46.8%   1.66    -21.1%      11.1%     1.5%     0.0%  ***
  0.69   +64.2%  +107.5%    43    33   0.20   48.8%   1.59    -22.0%       8.4%     0.8%     0.0%  *
  0.70   +56.5%   +94.6%    26    23   0.12   53.8%   1.90    -15.8%       3.3%     0.1%     0.0%

  ──────────────────────────────────────────────────────────────────────────────────────────────────
  Leverage 30x  (TP/trade=13.50%  SL/trade=4.500%)
  ──────────────────────────────────────────────────────────────────────────────────────────────────
  0.58   -99.7%  -167.0%   863   209   3.96   30.4%   0.69    -99.8%     100.0%   100.0%   100.0%
  0.60   -95.4%  -159.7%   565   181   2.59   33.3%   0.81    -97.0%      99.3%    95.1%    39.5%
  0.62   -25.4%   -42.6%   346   133   1.59   35.8%   0.96    -76.3%      91.5%    79.1%    37.3%
  0.64  +171.5%  +287.2%   205    97   0.94   40.0%   1.19    -56.4%      69.7%    50.2%    17.9%  *
  0.66  +524.0%  +877.4%   109    69   0.50   47.7%   1.66    -47.5%      49.2%    26.0%     5.8%  ***
  0.67  +630.6% +1055.9%    78    54   0.36   51.3%   1.97    -35.3%      45.3%    21.6%     3.5%  ***
  0.68  +205.7%  +344.5%    62    43   0.28   46.8%   1.62    -30.2%      36.1%    14.9%     1.2%  ***
  0.69  +100.6%  +168.4%    43    33   0.20   48.8%   1.56    -31.4%      24.9%     7.7%     0.2%  *
  0.70   +89.9%  +150.5%    26    23   0.12   53.8%   1.82    -22.9%      16.8%     3.4%     0.0%

====================================================================================================
  Ann%   = annualised return extrapolated from 218-day test
  Trd/d  = trades per calendar day
  P(-X%) = Monte Carlo prob of experiencing a -30%/-50%/-80% drawdown at any point in 1 yr
  ***    = PF >= 1.20 and >= 50 trades  (statistically meaningful)
  *      = PF >= 1.05 and >= 30 trades  (moderate sample)
```

</details>

---

## Architecture

```
Raw 1m OHLCV  ──►  features.py  ──►  XGBoost (3-class softprob)
                  (91 features)       p_up / p_down / p_no_break
                                               │
                            ┌──────────────────▼──────────────────┐
                            │  p_up  > T_up  (0.67)  →  LONG      │
                            │  p_dn  > T_dn  (0.67)  →  SHORT     │
                            │  else           →  FLAT (wait)       │
                            └──────────────────┬──────────────────┘
                                               │
                         SL = −0.15% · price   │   TP = +0.45% · price  (1:3 RR)
                         Time stop after 20 bars if neither hit
```

**Feature groups (91 total):**

| Group | Count | Description |
|---|---|---|
| ATR / Volatility | 13 | Fast/slow ATR, Bollinger Bands, Keltner squeeze, vol contraction |
| Volume / Energy | 6 | Relative volume, volume spike, vol·range composite |
| CVD / Orderflow | 20 | Cumulative Volume Delta (real taker data), A/D, divergence, delta bubble |
| VWAP | 4 | Rolling VWAP distance, VWAP-AD bias |
| Levels / Range | 31 | POC/VAH/VAL proximity (rolling VP), range width, boundary touch counts |
| EMA / Trend | 17 | EMA 9/21/50/200, fast EMAs (3/5), trend30 slope/range, acceleration |
| MTF (5m + 15m) | 22 | Resampled ATR, trend, volume, CVD — causal resampling from 1m buffer |
| Time | 4 | Hour-of-day and day-of-week (sin/cos cyclic encoding) |

**Training:**  5-fold walk-forward cross-validation · 4 years of data (2022–2025) ·
BTCUSDT + ETHUSDT + SOLUSDT · ~2.1M bars total · ~3–5 min on modern CPU.

---

## Requirements

- Python **3.10+**
- Windows / Linux / macOS
- Internet access for live trading (Binance WebSocket)

---

## Quick Start

> **The trained model is included in the repo.**
> You can paper trade or go live immediately after cloning — no training required.

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

# 4. Paper trade immediately — open http://localhost:8080
python sim/sim_live_ui.py

# 5. (Real trading only) Set Binance Futures API keys
cp .env.example .env
# Edit .env — fill in BINANCE_API_KEY and BINANCE_API_SECRET
python trade_live.py
```

> **Note — Windows:** `asyncio.WindowsSelectorEventLoopPolicy` is set automatically. No extra config needed.

---

## Data

Data is **not included** in the repo (too large). Place Binance OHLCV CSVs here:

```
Data/
└── BTCUSDT/
│   └── full_year/
│       ├── 2022_1m.csv
│       ├── 2022_5m.csv
│       ├── 2022_15m.csv
│       ├── 2023_1m.csv   ...
│       ├── 2024_1m.csv   ...
│       └── 2025_1m.csv   ...
└── ETHUSDT/
│   └── full_year/  (same structure)
└── SOLUSDT/
    └── full_year/  (same structure)
```

**CSV format** — standard Binance 12-column kline export:

```
open_time, open, high, low, close, volume, close_time,
quote_vol, num_trades, taker_buy_vol, taker_buy_quote_vol, ignore
```

Download from [Binance Vision](https://data.binance.vision/) or via the Binance API.
Column 10 (`taker_buy_vol`) is used as the real delta source for CVD features.

---

## Project Structure

```
C:/Trader2/
├── config.yaml               All hyperparameters — single source of truth
├── requirements.txt
├── .env.example              API key template (copy to .env)
│
├── data.py                   load_csv(), load_all(), candle_from_dict()
├── features.py               compute_features() + FeatureEngine (live incremental)
├── labels.py                 compute_labels() → 3-class labels + fakeout columns
├── train.py                  Walk-forward CV + final model + artifact export
├── eval.py                   Classification report, signal quality, plots
├── backtest.py               Event-driven vectorised backtest
├── trade_live.py             *** REAL MONEY *** live trading + browser dashboard
│
├── models/                   Generated by train.py — not in git (large)
│   ├── xgb_model.json        Trained XGBoost model
│   ├── feature_columns.json  Feature list (order matters for inference)
│   ├── thresholds.json       Per-class probability thresholds
│   ├── label_meta.json       Label distribution stats
│   ├── backtest_report.json  Backtest summary metrics
│   ├── backtest_trades.csv   Per-trade log
│   └── backtest_equity.png   Equity curve chart
│
├── sim/
│   ├── portfolio.py          Portfolio, Position, Trade dataclasses (paper)
│   ├── binance_portfolio.py  BinancePortfolio — real Futures order execution
│   ├── execution.py          ExecutionEngine: signal → order → trade log
│   ├── replay_feed.py        ReplayFeed: CSV row iterator
│   ├── binance_ws_feed.py    BinanceWSFeed (async) + SyncBinanceWSFeed
│   ├── sim_replay.py         Historical paper-trade simulation (terminal)
│   ├── sim_replay_ui.py      Historical paper-trade simulation + browser UI
│   ├── sim_live_binance.py   Live paper-trade (terminal only)
│   ├── sim_live_ui.py        Live paper-trade + browser dashboard
│   ├── static/
│   │   ├── index.html        Browser UI (dark theme, Lightweight Charts)
│   │   └── lightweight-charts.js  Bundled chart library (no CDN dependency)
│   └── logs/                 Trade logs written at runtime
│
└── Data/                     OHLCV CSVs — not tracked in git
```

---

## Commands

All commands assume you are in the **project root** with the venv active.

### Train

```bash
# Full 3-year training on all configured coins (~3–5 min)
./trader2/Scripts/python.exe train.py

# Use a different config
./trader2/Scripts/python.exe train.py --config config.yaml
```

Saves artifacts to `models/`: `xgb_model.json`, `feature_columns.json`, `thresholds.json`, `label_meta.json`.

---

### Evaluate

```bash
# Classification report + signal quality + plots
./trader2/Scripts/python.exe eval.py
```

Outputs to `models/`: `confusion_matrix.png`, `feature_importance.png`, `threshold_curve.png`, `calibration.png`, `signal_quality.csv`.

---

### Backtest

```bash
# Event-driven backtest on held-out test split
./trader2/Scripts/python.exe backtest.py
```

Outputs to `models/`: `backtest_report.json`, `backtest_trades.csv`, `backtest_equity.csv`, `backtest_equity.png`.

---

### Replay Simulation — Terminal

```bash
# Replay default CSV from config
./trader2/Scripts/python.exe sim/sim_replay.py

# Replay a specific file
./trader2/Scripts/python.exe sim/sim_replay.py --data Data/BTCUSDT/full_year/2025_1m.csv

# Max speed (no delay between bars)
./trader2/Scripts/python.exe sim/sim_replay.py --speed 0

# Custom date range
./trader2/Scripts/python.exe sim/sim_replay.py --start 2025-01-01 --end 2025-03-01
```

Logs trades to `sim/logs/replay_log.jsonl`.

---

### Replay Simulation — Browser Dashboard

```bash
# Default replay with browser UI
./trader2/Scripts/python.exe sim/sim_replay_ui.py

# Specific file
./trader2/Scripts/python.exe sim/sim_replay_ui.py --data Data/BTCUSDT/full_year/2025_1m.csv

# Max speed
./trader2/Scripts/python.exe sim/sim_replay_ui.py --speed 0

# Custom date range
./trader2/Scripts/python.exe sim/sim_replay_ui.py --start 2025-06-01 --end 2025-07-01

# Custom ports
./trader2/Scripts/python.exe sim/sim_replay_ui.py --port 8080 --ws-port 8765
```

Open **http://localhost:8080** in your browser. The replay waits for a browser connection before starting.

**Dashboard features:**
- Candlestick chart with VWAP overlay
- Volume histogram + CVD sub-chart
- Fixed Range Volume Profile (POC / VAH / VAL) toggle
- Volume spike bubbles on chart
- Trade entry/exit arrows with direction and probability
- In-trade candles highlighted blue
- Live running trade log with PnL and win rate
- Replay progress bar and final session summary

---

### Live Paper Trading — Terminal

```bash
# Default symbol (BTCUSDT from config)
./trader2/Scripts/python.exe sim/sim_live_binance.py

# Different symbol
./trader2/Scripts/python.exe sim/sim_live_binance.py --symbol ETHUSDT
```

Connects to Binance WebSocket, pre-warms the FeatureEngine from recent REST candles, then trades live 1m candles in paper mode. Logs to `sim/logs/live_log.jsonl`.

---

### Live Paper Trading — Browser Dashboard

```bash
# Default (BTCUSDT, port 8080)
./trader2/Scripts/python.exe sim/sim_live_ui.py

# Different symbol
./trader2/Scripts/python.exe sim/sim_live_ui.py --symbol ETHUSDT

# Custom ports
./trader2/Scripts/python.exe sim/sim_live_ui.py --port 8080 --ws-port 8765
```

Open **http://localhost:8080**. Pre-warms from the last 80 REST candles so the chart is populated immediately — no waiting for live data.

---

### Live Real Trading — Browser Dashboard ⚠️

> **WARNING: This places REAL orders with REAL money on Binance Futures.**
> Start with a small `position_size_pct` in `config.yaml` to verify behaviour.
> Requires Binance Futures API keys in `.env`.

```bash
# Default (BTCUSDT, port 8080)
./trader2/Scripts/python.exe trade_live.py

# Different symbol
./trader2/Scripts/python.exe trade_live.py --symbol ETHUSDT

# Custom ports
./trader2/Scripts/python.exe trade_live.py --port 8080 --ws-port 8765
```

Open **http://localhost:8080**. Same dashboard as paper trading, with a **LIVE** badge.

**What it does per bar:**
1. Receives closed 1m candle from Binance WebSocket
2. Updates FeatureEngine (incremental, causal — no lookahead)
3. Runs XGBoost inference → `p_up`, `p_down`, `p_no_break`
4. If `p_up > T_up` → MARKET BUY + server-side STOP_MARKET + TAKE_PROFIT_MARKET
5. If `p_dn > T_dn` → MARKET SELL + server-side STOP_MARKET + TAKE_PROFIT_MARKET
6. Polls Binance position endpoint each bar to detect SL/TP fills
7. Force-closes on time stop or Ctrl+C

**Setup:**
```bash
cp .env.example .env
# Fill in your Binance Futures API keys
```

**Key config values to set before going live** (`config.yaml`):
```yaml
trading:
  T_up:              0.67   # entry threshold — raise to reduce trade frequency
  T_down:            0.67
  sl_pct:            0.0015 # 0.15% stop-loss
  tp_pct:            0.0045 # 0.45% take-profit  (1:3 RR)
  time_stop:         20     # force-close after N bars
  position_size_pct: 1.0    # START LOW — this is your leverage multiplier
  initial_capital:   10.0   # your actual account balance in USDT
```

---

## Configuration Reference

All hyperparameters live in `config.yaml`. Everything is documented inline with ranges.

| Section | Key parameters |
|---|---|
| `data` | `symbol`, `years`, `coins` (multi-coin training list) |
| `labels` | `L_range` (range lookback), `H_horizon` (trade window), `sl_pct`, `tp_pct` |
| `features` | `atr_short/long`, `ema_periods`, `cvd_window`, `vwap_window` |
| `training` | `n_wf_splits`, `break_weight`, `xgb_params` (n_estimators, learning_rate, max_depth) |
| `trading` | `T_up`, `T_down`, `sl_pct`, `tp_pct`, `time_stop`, `cooldown`, `position_size_pct` |
| `simulation` | WebSocket URLs, log file paths, replay speed |

> `labels.sl_pct` and `trading.sl_pct` must match. Same for `tp_pct`.

---

## Label Design

Labels are computed using a **barrier method** over a 20-candle lookahead:

- **UP (2)** — price hits `entry + tp_pct` before `entry − sl_pct` within 20 bars
- **DOWN (0)** — price hits `entry − sl_pct` before `entry + tp_pct` within 20 bars
- **NO_BREAK (1)** — neither barrier reached within the horizon

Label distribution (4 years, 3 coins): UP ≈ 13.9%, DOWN ≈ 14.1%, NO_BREAK ≈ 72%.
Break classes are upweighted (`break_weight = 10.0`) during training.

---

## Notes

- **FeatureEngine warmup:** ~70 bars required before valid features. The REST pre-warm fills this instantly so live trading starts on bar 1.
- **XGBoost + NaN:** The model handles residual NaN natively — no imputation needed.
- **MTF features:** 5m and 15m features are derived by causal resampling of the 1m buffer inside `compute_features()` — no separate feeds needed, train/live stay in sync.
- **Single source of truth:** `compute_features()` is used identically in training, backtesting, and live inference.
- **Fakeout filtering:** A secondary `ema9_21_diff` regime filter skips entries in strongly trending bars to avoid chasing.
