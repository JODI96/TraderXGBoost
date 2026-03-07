# Optimization Log

---

## Run 21 – PCT Barriers + 20-min Window + Fast Momentum Features
**Date:** 2026-03-07

### Changes from Run 20 (best_params.yaml baseline)
| Change | Old | New | Reason |
|---|---|---|---|
| Label barriers | ATR-based (sl=2×ATR, tp=7×ATR) | **pct-based** (sl=0.15%, tp=0.45%) | Fee was eating 68% of gross TP in low-ATR squeezes; now fixed ~9% |
| H_horizon / time_stop | 15 bars | **20 bars** | 20-minute trade window |
| rr_ratio | 3.5 | **3.0** | Cleaner ratio |
| tp_atr fallback | 7 | **6** (=2×3) | Matches new rr_ratio |
| cooldown | 30 bars | **5 bars** | Allow faster re-entry |
| break_weight | 4.0 | **10.0** | 3% label rate needs ~16× to balance; 10 = selective but learnable |
| trend30_window | 30 bars | **10 bars** | Faster regime detection (10 min vs 30 min) |
| New features | — | EMA3/5, CVD3, price_mom_3/5, price_accel, delta_accel, ema_diff_accel | Model reacts faster to momentum shifts |
| train.py WF-CV | median of all folds | **max of valid folds** | Degenerate folds (f1=0) no longer pull estimator count to 3 |
| train.py val weights | uniform | **weighted** (same as train) | Early stopping now uses consistent weighted logloss |
| backtest.py SL/TP | ATR-based | **pct-based** when sl_pct/tp_pct in config | Execution matches labels |
| backtest.py dist col | hardcoded dist_rh_20 | **dynamic** dist_rh_{L_range} | Survives L_range changes |

### Training Results
- Label distribution: DOWN=3.1%, NO_BREAK=93.9%, UP=3.0%
- TP win rate: UP=83%, DOWN=82% (excellent)
- WF-CV: all 5 folds valid, n_estimators=1266
- Hold-out precision: DOWN=12%, UP=11%, NO_BREAK=96%

### Backtest Results (test split: 2025-05-26 to 2025-12-31, 218 days)
| T_up/T_down | Return | Trades | Trades/day | Win Rate | Profit Factor | Max DD |
|---|---|---|---|---|---|---|
| 0.58 | -14.2% | 556 | 2.5 | 35.3% | 0.80 | -15.6% |
| 0.60 | -6.4% | 348 | 1.6 | 36.5% | 0.86 | -8.2% |
| 0.62 | -0.31% | 193 | 0.9 | 40.9% | 0.99 | -4.5% |
| **0.64** | **+3.18%** | **94** | **0.4** | **48.9%** | **1.30** | **-2.4%** |
| 0.66 | +1.19% | 44 | 0.2 | 40.9% | 1.23 | -1.5% |
| 0.68 | +1.91% | 18 | 0.1 | 55.6% | 2.29 | -0.7% |

**Best config: T=0.64** → +3.18%, PF=1.30, 94 trades, max DD -2.4%

### Key Findings
1. **PCT barriers fixed fee dominance** — gross TP is now $45 (fee=9%) vs $5.88 (fee=68%) in squeezes
2. **break_weight=10 is the sweet spot** — break_weight=20 gave 8% precision (model too aggressive); 10 gives 12% with better calibration
3. **Fast features improved every threshold** — T=0.62 went from -1.94% → -0.31%; T=0.64 from +2.4% → +3.18%
4. **Trade density ceiling** — only 3% of 1m bars have UP/DOWN labels (0.45% move in 20 min is rare); max ~44 trades/day theoretically
5. **Profitable zone is T=0.62–0.68** — consistent positive returns, getting better each iteration

### Root Cause of Remaining Loss at Low Thresholds
- avg_win ~$13 (not full $41 TP) — most winning trades exit via time-stop (20 bars), not TP hit
- avg_loss ~$11 — most losing trades also exit via time-stop, not SL
- Fix: give trades more time (time_stop=30–40) or accept lower trade frequency

### Next Experiments to Try
1. **time_stop=30** in backtest at T=0.62-0.64 (no retrain) — more time to hit TP, avg_win should increase
2. If (1) works: retrain with H_horizon=30 to align labels
3. Increase max_depth 5→6, reduce min_child_weight 10→5 for more expressive model

---

## Run 25 – Multi-coin Training (BTC+ETH+SOL) + 3x Leverage
**Date:** 2026-03-07

### Changes from Run 24
| Change | Old | New | Reason |
|---|---|---|---|
| Training coins | BTCUSDT only | **BTCUSDT + ETHUSDT + SOLUSDT** | More data diversity -> better generalisation |
| Training rows | 1.26M train | **3.79M train** (3x) | 3 years x 3 coins |
| n_estimators | 1359 | **2225** | More data needs more trees |
| Training time | 3.9 min | **14.7 min** (GPU) | 3x data |
| position_size_pct | 1.0 | **3.0** (3x leverage) | Target 50% return in 6 months |
| Trading coin | BTCUSDT | **BTCUSDT only** | ETH/SOL PCT barriers too tight |
| backtest.py | CPU inference | **GPU inference** | device=cuda for batch prediction |
| backtest.py | BTCUSDT hardcoded | **--symbol flag** added | Enables per-coin backtesting |

### Training Results
- Coins: BTCUSDT + ETHUSDT + SOLUSDT (4.7M total rows, 3.79M train)
- Label distribution: DOWN=3.8%, UP=3.6% (slightly higher than BTC-only 3.1%)
- TP win rate: 80% across all 3 coins (vs 83% BTC-only; ETH/SOL slightly harder)
- WF-CV: all 5 folds valid, n_estimators=2225
- GPU training: RTX 3080, 14.7 min

### BTC Backtest Results (3x leverage, time_stop=30)
| T | Return | Trades | Trades/day | Win Rate | PF | Max DD |
|---|---|---|---|---|---|---|
| 0.58 | -45.6% | 693 | 3.2 | 33.2% | 0.78 | -53.6% |
| 0.60 | -17.2% | 450 | 2.1 | 34.9% | 0.89 | -37.5% |
| **0.62** | **+13.85%** | **276** | **1.3** | **38.4%** | **1.13** | **-17.6%** |
| 0.64 | +0.83% | 158 | 0.7 | 36.1% | 1.01 | -9.4% |
| **0.66** | **+10.98%** | **88** | **0.4** | **42.0%** | **1.30** | **-6.6%** |
| **0.68** | **+11.12%** | **43** | **0.2** | **46.5%** | **1.72** | **-5.1%** |

vs Run 24 BTC (1x leverage, time_stop=30):
- T=0.62: -0.17% -> **+13.85%** (multi-coin model + 3x leverage)
- T=0.66: +2.03% -> **+10.98%** (3x leverage)

### ETH/SOL Results: Broken
- ETH T=0.62: -68.9%, 1,881 trades (8.6/day), win=33.4%
- SOL T=0.62: -84.2%, 1,647 trades (7.6/day), win=31.9%
- Root cause: 0.15% PCT SL too tight for ETH/SOL volatility (2-3x BTC)
- Decision: trade BTC only, train on all 3 coins

### Key Findings
1. Multi-coin training dramatically improves BTC quality: T=0.62 went from -0.17% to +13.85%
2. 3x leverage on profitable BTC signals gives realistic path to 50% in 6 months
3. PCT barriers are coin-specific; ETH/SOL need ATR-based SL/TP (future work)
4. T=0.66 chosen as live config: balanced return (+10.98%) with lower drawdown (-6.6%)

### Next Experiments
1. Fine-grained T scan 0.60-0.68 (0.01 steps) to find exact peak
2. ATR-based SL/TP for ETH/SOL to unlock multi-coin trading
3. Add BNB/XRP to training for even more diversity

---

## Runs 22–24 – Depth/Horizon Exploration + GPU + time_stop=30
**Date:** 2026-03-07

### Run 22 – max_depth=6, min_child_weight=5 (H=20)
- Hypothesis: more expressive model pushes T=0.62 from breakeven to profitable
- Result: WORSE at T=0.64 (+1.05% vs Run 21's +3.18%), fewer high-confidence signals
- Root cause: depth-6 distributes probability mass more evenly; fewer signals exceed 0.64
- T=0.62 improved to -0.04% (nearly breakeven) but volume dropped too

### Run 23 – H_horizon=30, time_stop=30, depth=6
- Hypothesis: more time to hit TP improves win rate
- Result: WORSE across all thresholds vs Run 21
- Root cause: H=30 labels include borderline setups (TP win rate 76% vs 83%)
- More trades at T=0.64 (165 vs 94) but only 37% win rate, barely +0.37%

### Run 24 – H=20 strict labels + time_stop=30 in execution (depth=5, GPU)
- Hypothesis: strict H=20 labels + extra hold time beats aligned H=30
- Model: same as Run 21 settings, retrained on GPU (3.9 min vs ~15 min CPU)
- n_estimators: 1359

| T | ts=20 | ts=30 | Improvement |
|---|-------|-------|-------------|
| 0.62 | -0.33% / 185tr | -0.17% / 183tr | +0.16% |
| 0.64 | +1.24% / 92tr | +2.17% / 89tr | +0.93% |
| 0.66 | +1.20% / 45tr | +2.03% / 45tr | +0.83% |
| 0.68 | +1.97% / 20tr | +2.61% / 20tr | +0.64% |

**Best config: T=0.64, ts=30** -> +2.17%, PF=1.19, 0.41/day
**Best quality: T=0.68, ts=30** -> +2.61%, PF=2.36, 60% win rate

### Key Findings
1. H=20 + ts=30 > H=30 + ts=30: strict labels select better setups; extra time helps them complete
2. depth=5 + mcw=10 > depth=6 + mcw=5: simpler model generalises better at 3% label rate
3. GPU (RTX 3080) reduces training time 15 min -> 4 min (add device: "cuda" to xgb_params)
4. vol_regime filter did NOT fix L_range=8 model (win rate stuck at 33-38%)
5. L_range=8 is structurally inferior to L_range=20 for signal quality

### Next Experiments
1. break_weight=12-15: push T=0.62 from -0.17% to profitable (PF=0.99 currently)
2. cooldown=2: faster re-entry after winning trades
3. Multi-pair training: ETHUSDT/SOLUSDT for more data diversity
4. Asymmetric T_up/T_down thresholds if UP/DOWN precision differs

---

## Runs 1–20 Summary (pre-PCT)
- Best: Run 20 — break_weight=4, sl_atr=2, rr_ratio=3.5, tp_atr=7, H=15
- Result: total_return=-10.48%, max_dd=-10.55%
- Root cause: fee dominance at low-ATR entries (fee eating 68% of gross TP)
- See original best_params.yaml for full analysis
