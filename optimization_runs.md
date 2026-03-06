# Optimization Run Log

## Baseline
- total_return: -73.84%  win_rate: 30.6%  profit_factor: 0.374  n_trades: 3037  sharpe: -23.93
- ROOT CAUSE: sl_atr=20 (20x ATR from entry), tp_atr=60 -> unreachable in 20 bars
  Most trades close by TIME STOP with near-random tiny P&L.

---

## Run 1
**Changes**: sl/tp tightened to achievable range, k_atr reduced, threshold raised, break_weight reduced
- labels.sl_atr: 20->2, labels.rr_ratio: 3.0 (keep), labels.k_atr: 30->5, labels.H_horizon: 20 (keep)
- trading.sl_atr: 20->2, trading.tp_atr: 60->6
- trading.T_up: 0.4->0.55, trading.T_down: 0.4->0.55
- training.break_weight: 40.0->15.0

**Result**: Improved but still losing (runs 1-15 progressively tuned)

---

## Runs 2-15 (summary)
Various tuning of: k_atr (5-30), sl_atr (2-8), rr_ratio (2.0-3.0), break_weight (5-40),
T_up/T_down (0.40-0.72), require_squeeze (true/false), max_depth (3-5),
L_range (20-30, learned MUST be 20 due to backtest dist_rh_20 filter).

Best result in this range: Run 11 at T=0.65: -3.75%, 76 trades, avg_win=$10.56 > avg_loss=$9.40
(small sample, statistically unreliable)

Key discoveries:
- L_range MUST be 20 (backtest.py hardcodes dist_rh_20 entry filter)
- L_atr=1 better than L_atr=7 initially (more labels)
- break_weight=4 gives BETTER precision than break_weight=10-40
- Model precision ceiling ~13-21% with break_weight >= 10

---

## Run 16
**Config**: k_atr=5, sl_atr=2, rr_ratio=2.0 (tp=4), break_weight=10, T=0.60, cooldown=50
- Labels: DOWN=1.5%, UP=1.9% (few labels), TP win rate=75%
- Model: DOWN precision=0.13, UP precision=0.21
- Backtest: total_return=-81.02%, n_trades=3963, win_rate=31.9%, avg_win=$5.39, avg_loss=$5.53
- ROOT CAUSE: avg_win < avg_loss. Most trades closed by time stop at ~$0 gross - $4 fee.
  The 2:1 RR doesn't show up in practice. tp_atr=4 too small (fee eats TP in low-vol).

---

## Run 17
**Changes**: k_atr=8, max_depth=5, min_child_weight=10, colsample=0.70, break_weight=15, T=0.68
- Labels: DOWN=3.6%, UP=3.8%, TP win rate=71%
- Model: DOWN precision=0.15, UP precision=0.16 (slight improvement)
- Backtest: total_return=-56.02%, n_trades=2158, win_rate=33.3%, avg_win=$5.23, avg_loss=$6.51
- avg_win still < avg_loss. More trades but same fundamental problem.

---

## Run 18
**Changes**: k_atr=12, rr_ratio=3.5 (tp=7), break_weight=25, T=0.65, cooldown=40
- Labels: DOWN=2.1%, UP=2.2%, TP win rate=66%
- Model: DOWN precision=0.12, UP precision=0.15 (worse - too few labels, model saturated)
- Backtest: total_return=-61.08%, n_trades=2351, win_rate=27.2%, avg_win=$6.28, avg_loss=$5.91
- FIRST RUN WHERE avg_win > avg_loss! RR improvement is working.
  But win_rate 27.2% still too low (needs ~49% for positive EV with these avg values).

---

## Run 19 (BEST MODEL BY PRECISION)
**Changes**: k_atr=5, rr_ratio=3.5 (tp=7), break_weight=4, n_estimators=5000,
            gamma=0, reg_alpha=1.0, reg_lambda=1.0, T=0.55, cooldown=10
- Labels: DOWN=5.6%, UP=5.8%, TP win rate=70%
- Model: DOWN precision=0.28, UP precision=0.29 (BEST EVER - break_weight=4 is key)
  WF-CV median n_estimators=649 (early stopping fired! vs 3000+ cap before)
- Backtest (T=0.55): total_return=-48.99%, n_trades=1723, win_rate=24.4%, avg_win=$4.08, avg_loss=$5.08
- Backtest (T=0.65): total_return=-12.62%, n_trades=356, win_rate=21.3%, avg_win=$3.06, avg_loss=$5.34
  Exit breakdown: SL=216 (60.7%), TP=102 (28.7%), TIME=38 (10.7%)
  TP exits: avg_net_pnl=+$1.88 (tiny due to low-vol squeeze entries)
  SL exits: avg_net_pnl=-$6.38 (full SL at higher ATR periods)

KEY INSIGHT: Model enters during squeezes (low ATR) for TP wins -> small dollar TP.
  Same model enters during high ATR for SL losses -> big dollar SL.
  This ATR selection bias explains why avg_win << expected despite 3.5:1 RR.

---

## Run 20 (BEST RESULT BY TOTAL RETURN = LOWEST LOSS)
**Changes from Run 19**: L_atr=1->7 (smooth label ATR barriers), T=0.58, cooldown=30
- Labels: DOWN=4.2%, UP=4.4%, TP win rate=82% (BEST LABEL QUALITY: L_atr=7 key)
- Model: DOWN precision=0.23, UP precision=0.26 (slightly lower than R19 but better labels)
  WF-CV n_estimators=766 (converged)
- Backtest (T=0.58): total_return=-10.48%, n_trades=277, win_rate=19.1%, avg_win=$2.97, avg_loss=$5.38
  Exit: SL=174, TP=78, TIME=25
  TP exits: avg_net_pnl=+$1.59, avg_raw_pnl=+$5.36
  SL exits: avg_net_pnl=-$6.49, avg_raw_pnl=-$2.71
- Drawdown well-controlled: -10.55% max (best in 20 runs)

CONCLUSION: Still could not achieve positive profit with config-only changes.
Best config saved to best_params.yaml.

---

## Summary: Why Profitability Was Not Achieved

1. FEE DOMINANCE during low-vol entries:
   Fee = $4/trade (fixed %). Gross TP at squeeze ATR ~ $5-6. Net TP = $1-2.
   Gross SL at higher ATR = $2-3. Net SL = $6-7.
   This produces avg_win ~ $2-3, avg_loss ~ $5-7 regardless of RR ratio in labels.

2. MODEL PRECISION CEILING at ~28-29%:
   No config combination pushed precision above 29%.
   At 29% precision * 82% label TP win rate = 23.8% actual TP hit rate.
   Barely positive EV in ATR units, negative after fee conversion.

3. REQUIRED CODE CHANGES:
   a) Minimum ATR filter (skip if ATR_usd < min_atr_usd)
   b) Fixed-dollar SL/TP independent of ATR
   c) Volatility regime filter (only trade high-ATR conditions)
