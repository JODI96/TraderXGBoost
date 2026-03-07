"""
backtest.py – Event-driven vectorised backtest on historical data.

Uses the trained XGBoost model to generate signals bar-by-bar and
simulates paper trades with SL / TP / time-stop exit logic.

Usage
-----
    python backtest.py                       # default: test split
    python backtest.py --year 2025           # specific year
    python backtest.py --T_up 0.60 --T_down 0.60

Outputs
-------
    models/backtest_trades.csv
    models/backtest_equity.csv
    models/backtest_report.json
    models/backtest_equity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

import data as data_mod
import features as feat_mod
import labels as label_mod


# ─────────────────────────────────────────────────────────────────────────────
def _load_artifacts(art_dir: str = "models"):
    p = Path(art_dir)
    model = xgb.Booster()
    model.load_model(str(p / "xgb_model.json"))
    # Use GPU for batch prediction if model was trained with CUDA
    try:
        model.set_param({"device": "cuda"})
    except Exception:
        pass
    with open(p / "feature_columns.json") as f:
        feat_cols = json.load(f)
    with open(p / "thresholds.json") as f:
        thresholds = json.load(f)
    return model, feat_cols, thresholds


# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    df_feat: pd.DataFrame,
    probs: np.ndarray,
    cfg: dict,
    T_up:      float | None = None,
    T_down:    float | None = None,
    d_max:     float | None = None,
    time_stop: int   | None = None,
    min_vol:   float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Event-driven simulation; processes one candle at a time.

    Parameters
    ----------
    df_feat   : feature DataFrame (same index as probs)
    probs     : (n, 3) probability array from model
    feat_meta : DataFrame with at least: atr, range_high_{L}, range_low_{L}
    cfg       : full config dict
    T_up / T_down / d_max : override trading thresholds

    Returns
    -------
    trades_df : DataFrame of closed trades
    equity    : np.ndarray of equity curve ($ per bar)
    report    : dict of summary statistics
    """
    tc  = cfg["trading"]
    lc  = cfg["labels"]

    T_u  = T_up    or tc["T_up"]
    T_d  = T_down  or tc["T_down"]
    dmax = d_max   or tc["d_max_atr"]

    sl_atr    = tc.get("sl_atr", 2)
    tp_atr    = tc.get("tp_atr", 7)
    sl_pct    = tc.get("sl_pct")
    tp_pct    = tc.get("tp_pct")
    use_pct   = sl_pct is not None and tp_pct is not None
    time_stop = time_stop if time_stop is not None else tc["time_stop"]
    cooldown  = tc["cooldown"]
    req_sq    = tc.get("require_squeeze", False)
    capital0  = tc["initial_capital"]
    pos_pct   = tc["position_size_pct"]

    cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
               lc["slippage"]  + lc["spread"]) * 2

    n      = len(df_feat)
    close  = df_feat["close"].values
    # Use atr_short from features (same source as execution.py uses for SL/TP)
    atr    = df_feat["atr_short"].values if "atr_short" in df_feat.columns \
             else np.full(n, np.nan)
    sq_col    = df_feat["squeeze_flag"].values if "squeeze_flag" in df_feat.columns \
                else np.zeros(n)
    vol_regime = df_feat["vol_regime"].values if "vol_regime" in df_feat.columns \
                 else np.ones(n)
    min_vol    = min_vol if min_vol is not None else tc.get("min_vol_regime", 0.0)

    p_down = probs[:, 0]
    p_up   = probs[:, 2]

    # Use pre-computed distance features from df_feat (same as execution.py)
    L_range = lc.get("L_range", 20)
    rh_col  = f"dist_rh_{L_range}"
    rl_col  = f"dist_rl_{L_range}"
    dist_rh = df_feat[rh_col].values if rh_col in df_feat.columns else np.full(n, np.nan)
    dist_rl = df_feat[rl_col].values if rl_col in df_feat.columns else np.full(n, np.nan)

    capital      = capital0
    pos_dir      = 0          # 0=flat, 1=long, -1=short
    entry_price  = 0.0
    entry_idx    = 0
    sl_price     = 0.0
    tp_price     = 0.0
    pos_size     = 0.0        # units of base asset
    cooldown_rem = 0

    trades: list[dict] = []
    equity = np.full(n, capital0, dtype=np.float64)

    for i in range(1, n):
        c = close[i]
        if np.isnan(c) or np.isnan(atr[i]):
            equity[i] = capital
            continue

        # ── Mark-to-market running equity (unrealised) ────────────────────────
        if pos_dir == 1:
            equity[i] = capital + (c - entry_price) * pos_size
        elif pos_dir == -1:
            equity[i] = capital + (entry_price - c) * pos_size
        else:
            equity[i] = capital

        # ── Check exit conditions ─────────────────────────────────────────────
        if pos_dir != 0:
            time_held = i - entry_idx
            hit_sl    = (pos_dir == 1  and c <= sl_price) or \
                        (pos_dir == -1 and c >= sl_price)
            hit_tp    = (pos_dir == 1  and c >= tp_price) or \
                        (pos_dir == -1 and c <= tp_price)
            hit_time  = time_held >= time_stop

            if hit_sl or hit_tp or hit_time:
                exit_p  = c
                raw_pnl = ((exit_p - entry_price) * pos_dir * pos_size)
                cost_pnl = entry_price * pos_size * cost_rt
                net_pnl  = raw_pnl - cost_pnl
                capital += net_pnl
                equity[i] = capital

                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx":  i,
                    "entry_ts":  df_feat.index[entry_idx],
                    "exit_ts":   df_feat.index[i],
                    "direction": "LONG" if pos_dir == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price":  exit_p,
                    "size":        pos_size,
                    "raw_pnl":     round(raw_pnl,  2),
                    "cost":        round(cost_pnl, 2),
                    "net_pnl":     round(net_pnl,  2),
                    "time_held":   time_held,
                    "exit_reason": "SL" if hit_sl else ("TP" if hit_tp else "TIME"),
                })
                pos_dir      = 0
                cooldown_rem = cooldown

        if cooldown_rem > 0:
            cooldown_rem -= 1
            continue

        # ── Entry logic ───────────────────────────────────────────────────────
        if pos_dir == 0:
            sq_ok  = (not req_sq) or sq_col[i] == 1
            vol_ok = vol_regime[i] >= min_vol

            # LONG: imminent up-breakout
            if (p_up[i] >= T_u and
                    not np.isnan(dist_rh[i]) and dist_rh[i] <= dmax and
                    sq_ok and vol_ok):
                entry_price  = c
                pos_size     = (capital * pos_pct) / (c + 1e-9)
                if use_pct:
                    sl_price = entry_price * (1.0 - sl_pct)
                    tp_price = entry_price * (1.0 + tp_pct)
                else:
                    sl_price = entry_price - sl_atr * atr[i]
                    tp_price = entry_price + tp_atr * atr[i]
                pos_dir      = 1
                entry_idx    = i

            # SHORT: imminent down-breakout
            elif (p_down[i] >= T_d and
                    not np.isnan(dist_rl[i]) and dist_rl[i] <= dmax and
                    sq_ok and vol_ok):
                entry_price  = c
                pos_size     = (capital * pos_pct) / (c + 1e-9)
                if use_pct:
                    sl_price = entry_price * (1.0 + sl_pct)
                    tp_price = entry_price * (1.0 - tp_pct)
                else:
                    sl_price = entry_price + sl_atr * atr[i]
                    tp_price = entry_price - tp_atr * atr[i]
                pos_dir      = -1
                entry_idx    = i

    # Close any open trade at end
    if pos_dir != 0:
        exit_p   = close[-1]
        raw_pnl  = (exit_p - entry_price) * pos_dir * pos_size
        cost_pnl = entry_price * pos_size * cost_rt
        net_pnl  = raw_pnl - cost_pnl
        capital += net_pnl
        trades.append({
            "entry_idx": entry_idx, "exit_idx": n - 1,
            "direction": "LONG" if pos_dir == 1 else "SHORT",
            "entry_price": entry_price, "exit_price": exit_p,
            "size": pos_size, "net_pnl": round(net_pnl, 2),
            "exit_reason": "END",
        })

    trades_df = pd.DataFrame(trades)
    report    = _compute_report(trades_df, equity, capital0)
    return trades_df, equity, report


# ─────────────────────────────────────────────────────────────────────────────
def _compute_report(trades: pd.DataFrame, equity: np.ndarray,
                    capital0: float) -> dict:
    if trades.empty:
        return {"error": "no trades"}
    pnls = trades["net_pnl"].values
    wins = pnls[pnls > 0]
    loss = pnls[pnls < 0]

    total_ret  = (equity[-1] - capital0) / capital0 * 100
    profit_fac = wins.sum() / (-loss.sum() + 1e-9) if len(loss) else float("inf")
    sharpe     = _annualised_sharpe(equity, periods_per_year=525_600)

    return {
        "total_return_pct": round(total_ret, 2),
        "n_trades":         len(trades),
        "win_rate":         round(len(wins) / len(pnls) * 100, 1),
        "profit_factor":    round(profit_fac, 3),
        "sharpe_ratio":     round(sharpe, 3),
        "avg_win":          round(wins.mean(), 2) if len(wins) else 0,
        "avg_loss":         round(loss.mean(), 2) if len(loss) else 0,
        "max_drawdown_pct": round(_max_dd(equity), 2),
        "final_capital":    round(equity[-1], 2),
    }


def _max_dd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / (peak + 1e-9) * 100
    return float(dd.min())


def _annualised_sharpe(equity: np.ndarray,
                       periods_per_year: int = 525_600) -> float:
    rets = np.diff(equity) / (equity[:-1] + 1e-9)
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


# ─────────────────────────────────────────────────────────────────────────────
def _plot_equity(equity: np.ndarray, trades: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity, lw=0.8, color="steelblue", label="Equity")
    if not trades.empty and "exit_idx" in trades.columns:
        wins  = trades[trades["net_pnl"] > 0]["exit_idx"].values
        losss = trades[trades["net_pnl"] <= 0]["exit_idx"].values
        ax.scatter(wins,  equity[wins.astype(int)],  color="green", s=10, zorder=3)
        ax.scatter(losss, equity[losss.astype(int)], color="red",   s=10, zorder=3)
    ax.set_xlabel("Bar"); ax.set_ylabel("Equity ($)")
    ax.set_title("Backtest Equity Curve")
    ax.legend(); plt.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",      type=int,   default=None)
    parser.add_argument("--T_up",      type=float, default=None)
    parser.add_argument("--T_down",    type=float, default=None)
    parser.add_argument("--d_max",     type=float, default=None)
    parser.add_argument("--time_stop", type=int,   default=None)
    parser.add_argument("--min_vol",   type=float, default=None)
    parser.add_argument("--symbol",    default=None, help="Override symbol (e.g. ETHUSDT)")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--artifacts", default="models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("Loading model ...")
    model, feat_cols, thresholds = _load_artifacts(args.artifacts)

    # ── Load data ─────────────────────────────────────────────────────────────
    cfg_tmp = dict(cfg); cfg_tmp["data"] = dict(cfg["data"])
    if args.symbol:
        cfg_tmp["data"]["base_dir"] = (
            f"{cfg['data']['data_root']}/{args.symbol}/full_year"
        )
    if args.year:
        cfg_tmp["data"]["years"] = [args.year]
        df_raw = data_mod.load_all(cfg_tmp)
    else:
        df_raw = data_mod.load_all(cfg_tmp)
        split  = int(len(df_raw) * (1 - cfg["training"]["test_size"]))
        df_raw = df_raw.iloc[split:]

    # ── Features + Labels ─────────────────────────────────────────────────────
    print("Computing features ...")
    df_feat = feat_mod.compute_features(df_raw, cfg)
    df_lab  = label_mod.compute_labels(df_raw, cfg)

    avail   = [c for c in feat_cols if c in df_feat.columns]
    # df_lab columns: label, range_high, range_low, atr, thresh_high, thresh_low, …
    # run_backtest reads atr_short, dist_rh_20, dist_rl_20 directly from df_feat cols
    df_all  = df_feat.join(df_lab[["label"]], how="inner").dropna(subset=avail + ["label"])

    X = df_all[avail]

    # ── Predict ───────────────────────────────────────────────────────────────
    print("Predicting ...")
    dm    = xgb.DMatrix(X.values, feature_names=avail)
    probs = model.predict(dm).reshape(-1, 3)

    # ── Run backtest ──────────────────────────────────────────────────────────
    print("Running backtest ...")
    trades, equity, report = run_backtest(
        df_feat   = df_all,
        probs     = probs,
        cfg       = cfg,
        T_up      = args.T_up,
        T_down    = args.T_down,
        d_max     = args.d_max,
        time_stop = args.time_stop,
        min_vol   = args.min_vol,
    )

    print("\n--- Backtest Report ---")
    for k, v in report.items():
        print(f"  {k:25s}: {v}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    art = Path(args.artifacts)
    if not trades.empty:
        trades.to_csv(art / "backtest_trades.csv", index=False)
        print(f"\n  Trades -> {art / 'backtest_trades.csv'}  ({len(trades)} rows)")

    pd.Series(equity).to_csv(art / "backtest_equity.csv", index=False, header=["equity"])

    with open(art / "backtest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report -> {art / 'backtest_report.json'}")

    _plot_equity(equity, trades, str(art / "backtest_equity.png"))


if __name__ == "__main__":
    main()
