"""
liquidation_risk.py – Monte Carlo liquidation probability analysis.

Simulates N random trade paths for each (T, leverage) combination using
empirical win rate and PCT SL/TP from config. Reports probability of
liquidation (capital < liquidation_threshold) within 1, 3, 5, 10 years.

Usage
-----
    python liquidation_risk.py
    python liquidation_risk.py --sims 50000 --liq_threshold 0.10
"""

from __future__ import annotations

import argparse
import yaml
import numpy as np

N_SIMS_DEFAULT   = 10_000
LIQ_THRESHOLD    = 0.20   # liquidated when capital drops below 20% of initial
YEARS            = [1, 3, 5, 10]
TRADING_DAYS     = 365

# Trades/day per threshold — from backtest results (218-day test period)
# Format: T -> trades_per_day
TRADES_PER_DAY = {
    0.58: 3.18,
    0.60: 2.06,
    0.62: 1.27,
    0.64: 0.72,
    0.66: 0.40,
    0.68: 0.20,
    0.70: 0.06,
}

WIN_RATES = {
    0.58: 0.332,
    0.60: 0.349,
    0.62: 0.384,
    0.64: 0.361,
    0.66: 0.420,
    0.68: 0.465,
    0.70: 0.429,
}


def liquidation_prob(win_rate: float, win_pct: float, loss_pct: float,
                     trades: int, n_sims: int, liq_threshold: float,
                     rng: np.random.Generator) -> float:
    """Monte Carlo: fraction of paths that hit liquidation within `trades` steps."""
    capital = np.ones(n_sims, dtype=np.float64)
    liquidated = np.zeros(n_sims, dtype=bool)

    # Process in chunks for memory efficiency
    chunk = 512
    remaining = trades
    while remaining > 0:
        n = min(chunk, remaining)
        remaining -= n
        wins = rng.random((n_sims, n)) < win_rate
        # Multiply capital by per-trade factor
        for t in range(n):
            factor = np.where(wins[:, t], 1.0 + win_pct, 1.0 - loss_pct)
            capital = np.where(liquidated, capital, capital * factor)
            liquidated |= capital < liq_threshold

    return float(liquidated.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--sims",          type=int,   default=N_SIMS_DEFAULT)
    parser.add_argument("--liq_threshold", type=float, default=LIQ_THRESHOLD)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tc  = cfg["trading"]
    lc  = cfg["labels"]
    sc  = cfg.get("sweep", {})

    tp_pct    = lc.get("tp_pct", tc.get("tp_pct", 0.0045))
    sl_pct    = lc.get("sl_pct", tc.get("sl_pct", 0.0015))
    T_values  = sc.get("T_values",  [0.62, 0.66, 0.68])
    leverages = sc.get("leverages", [10.0, 20.0, 30.0])
    leverages = [l for l in leverages if l >= 5.0]  # skip low leverage (not interesting)

    rng = np.random.default_rng(42)

    year_cols = "  ".join(f"{y}yr%" for y in YEARS)
    header = f"{'T':>6}  {'Lev':>5}  {'WinRate':>8}  {'Trd/yr':>7}  {year_cols}"
    sep    = "=" * len(header)

    print(f"\n{sep}")
    print(f"LIQUIDATION PROBABILITY  (n={args.sims:,} sims, liq<{args.liq_threshold*100:.0f}% capital)")
    print(sep)
    print(header)
    print("-" * len(header))

    for lev in leverages:
        win_pct  = lev * tp_pct
        loss_pct = lev * sl_pct
        for T in T_values:
            if T not in WIN_RATES:
                continue
            wr          = WIN_RATES[T]
            tpd         = TRADES_PER_DAY.get(T, 0.5)
            trades_year = int(tpd * TRADING_DAYS)

            probs = []
            for yr in YEARS:
                p = liquidation_prob(
                    win_rate=wr, win_pct=win_pct, loss_pct=loss_pct,
                    trades=trades_year * yr,
                    n_sims=args.sims, liq_threshold=args.liq_threshold,
                    rng=rng,
                )
                probs.append(p)

            prob_str = "  ".join(f"{p*100:>6.1f}%" for p in probs)
            print(f"  {T:.2f}  {lev:>4.0f}x"
                  f"  {wr*100:>7.1f}%"
                  f"  {trades_year:>7}"
                  f"  {prob_str}")
        print()

    print(sep)
    print(f"Note: win_pct=+{win_pct*100:.1f}%/trade, loss_pct=-{lev*sl_pct*100:.1f}%/trade at {lev:.0f}x")
    print(f"      Liquidation = capital falls below {args.liq_threshold*100:.0f}% of starting capital")
    print(sep)


if __name__ == "__main__":
    main()
