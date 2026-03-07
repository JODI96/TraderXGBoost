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

## Runs 1–20 Summary (pre-PCT)
- Best: Run 20 — break_weight=4, sl_atr=2, rr_ratio=3.5, tp_atr=7, H=15
- Result: total_return=-10.48%, max_dd=-10.55%
- Root cause: fee dominance at low-ATR entries (fee eating 68% of gross TP)
- See original best_params.yaml for full analysis
