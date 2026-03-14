"""
backtest.py – Event-driven vectorised backtest on historical data.

Uses the trained XGBoost model to generate signals bar-by-bar and
simulates paper trades with SL / TP / time-stop exit logic.

Usage
-----
    python backtest.py                       # default: test split
    python backtest.py --year 2025           # specific year
    python backtest.py --T_up 0.60 --T_down 0.60
    python backtest.py --model trial24       # use a saved Fabio model
    python backtest.py --model jodi          # use Jodi's current model

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
import shutil
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
# Pre-configured model profiles for --model flag
# Each profile specifies the model file and config overrides.
# All models use the standard feature_columns.json / label_meta.json / thresholds.json.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PROFILES = {
    "trial60": {
        "model": "models/fabio_models/v2.1_trial60_PF7.05.json",
        "desc": "PF 7.05, WR 83.3%, 6 Trades, DD -3.9%",
        "config": {"T_up": 0.64, "T_down": 0.64, "sl_pct": 0.0015, "tp_pct": 0.0045,
                   "time_stop": 30},
        "labels": {"H_horizon": 20, "sl_pct": 0.0015, "tp_pct": 0.0045},
    },
    "trial51": {
        "model": "models/fabio_models/v2.1_trial51_PF5.42.json",
        "desc": "PF 5.42, WR 80.0%, 10 Trades, DD -3.7%",
        "config": {"T_up": 0.60, "T_down": 0.58, "sl_pct": 0.0015, "tp_pct": 0.0045,
                   "time_stop": 25},
        "labels": {"H_horizon": 15, "sl_pct": 0.0015, "tp_pct": 0.0045},
    },
    "trial75": {
        "model": "models/fabio_models/v2.1_trial75_PF3.63.json",
        "desc": "PF 3.63, WR 70.0%, 20 Trades, DD -7.5%",
        "config": {"T_up": 0.60, "T_down": 0.58, "sl_pct": 0.002, "tp_pct": 0.0045,
                   "time_stop": 25},
        "labels": {"H_horizon": 15, "sl_pct": 0.002, "tp_pct": 0.0045},
    },
    "trial33": {
        "model": "models/fabio_models/v2.1_trial33_PF3.42.json",
        "desc": "PF 3.42, WR 63.2%, 19 Trades, DD -5.9%",
        "config": {"T_up": 0.60, "T_down": 0.58, "sl_pct": 0.0015, "tp_pct": 0.0045,
                   "time_stop": 25},
        "labels": {"H_horizon": 15, "sl_pct": 0.0015, "tp_pct": 0.0045},
    },
    "trial24": {
        "model": "models/fabio_models/v2.1_trial24_PF2.69.json",
        "desc": "PF 2.69, WR 62.1%, 29 Trades, DD -8.4%",
        "config": {"T_up": 0.64, "T_down": 0.60, "sl_pct": 0.002, "tp_pct": 0.0045,
                   "time_stop": 30},
        "labels": {"H_horizon": 20, "sl_pct": 0.002, "tp_pct": 0.0045},
    },
    "jodi": {
        "model": "models/xgb_model.json",
        "desc": "Jodi's current model (as-is from config)",
        "config": {},
        "labels": {},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
def _liquidation_prob(win_rate: float, win_pct: float, loss_pct: float,
                      trades: int, n_sims: int, liq_threshold: float,
                      rng: np.random.Generator) -> float:
    """Monte Carlo: fraction of paths where capital drops below liq_threshold."""
    capital    = np.ones(n_sims, dtype=np.float64)
    liquidated = np.zeros(n_sims, dtype=bool)
    chunk = 512
    done  = 0
    while done < trades:
        n    = min(chunk, trades - done)
        wins = rng.random((n_sims, n)) < win_rate
        for t in range(n):
            factor  = np.where(wins[:, t], 1.0 + win_pct, 1.0 - loss_pct)
            capital = np.where(liquidated, capital, capital * factor)
            liquidated |= capital < liq_threshold
        done += n
    return float(liquidated.mean())


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
    T_up:           float | None = None,
    T_down:         float | None = None,
    d_max:          float | None = None,
    time_stop:      int   | None = None,
    min_vol:        float | None = None,
    min_ema9_21:    float | None = None,
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
    high   = df_feat["high"].values
    low    = df_feat["low"].values
    # Use atr_short from features (same source as execution.py uses for SL/TP)
    atr    = df_feat["atr_short"].values if "atr_short" in df_feat.columns \
             else np.full(n, np.nan)
    sq_col    = df_feat["squeeze_flag"].values if "squeeze_flag" in df_feat.columns \
                else np.zeros(n)
    vol_regime = df_feat["vol_regime"].values if "vol_regime" in df_feat.columns \
                 else np.ones(n)
    min_vol      = min_vol     if min_vol     is not None else tc.get("min_vol_regime",   0.0)
    min_ema9_21  = min_ema9_21 if min_ema9_21 is not None else tc.get("min_ema9_21_diff", -999.0)
    ema9_21_diff = df_feat["ema9_21_diff"].values if "ema9_21_diff" in df_feat.columns \
                   else np.full(n, 999.0)

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
            h, l      = high[i], low[i]
            # Use intrabar high/low so SL/TP fire as soon as price touches level
            hit_sl    = (pos_dir == 1  and l <= sl_price) or \
                        (pos_dir == -1 and h >= sl_price)
            hit_tp    = (pos_dir == 1  and h >= tp_price) or \
                        (pos_dir == -1 and l <= tp_price)
            hit_time  = time_held >= time_stop

            if hit_sl or hit_tp or hit_time:
                # Exit at exact SL/TP level; time-stop exits at close
                if hit_sl:
                    exit_p = sl_price
                elif hit_tp:
                    exit_p = tp_price
                else:
                    exit_p = c
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
            sq_ok   = (not req_sq) or sq_col[i] == 1
            vol_ok  = vol_regime[i] >= min_vol
            ema_ok  = ema9_21_diff[i] >= min_ema9_21

            # LONG: imminent up-breakout
            if (p_up[i] >= T_u and
                    not np.isnan(dist_rh[i]) and dist_rh[i] <= dmax and
                    sq_ok and vol_ok and ema_ok):
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
                    sq_ok and vol_ok and ema_ok):
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

    active_days = (trades["entry_ts"].dt.date.nunique()
                   if "entry_ts" in trades.columns else 0)

    return {
        "total_return_pct": round(total_ret, 2),
        "n_trades":         len(trades),
        "active_days":      active_days,
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
    parser.add_argument("--model",     default=None,
                        help="Model profile name (e.g. trial24, trial60, jodi). "
                             "Use --model list to see all available profiles.")
    parser.add_argument("--year",      type=int,   default=None)
    parser.add_argument("--T_up",      type=float, default=None)
    parser.add_argument("--T_down",    type=float, default=None)
    parser.add_argument("--d_max",     type=float, default=None)
    parser.add_argument("--time_stop", type=int,   default=None)
    parser.add_argument("--min_vol",   type=float, default=None)
    parser.add_argument("--symbol",    default=None, help="Override symbol (e.g. ETHUSDT)")
    parser.add_argument("--pos_size",  type=float, default=None, help="Override position_size_pct (leverage)")
    parser.add_argument("--sweep",     action="store_true", help="Run grid search over sweep.T_values x sweep.leverages")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--artifacts", default="models")
    args = parser.parse_args()

    # ── List available model profiles ─────────────────────────────────────────
    if args.model == "list":
        print("\nAvailable model profiles (use with --model <name>):\n")
        for name, prof in MODEL_PROFILES.items():
            print(f"  {name:<12} {prof['desc']}")
        print()
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Apply model profile if specified ──────────────────────────────────────
    if args.model:
        profile = MODEL_PROFILES.get(args.model)
        if profile is None:
            print(f"  ERROR: Unknown model '{args.model}'.")
            print(f"  Available: {', '.join(MODEL_PROFILES.keys())}")
            print(f"  Use --model list to see details.")
            return

        print(f"  Using model profile: {args.model}")
        print(f"  {profile['desc']}")

        # Copy model file to models/xgb_model.json so _load_artifacts picks it up
        model_src = Path(profile["model"])
        if not model_src.exists():
            print(f"  ERROR: Model file not found: {model_src}")
            return
        model_dst = Path(args.artifacts) / "xgb_model.json"
        if str(model_src) != str(model_dst):
            shutil.copy2(model_src, model_dst)
            print(f"  Copied {model_src.name} -> {model_dst}")

        # Apply config overrides
        for k, v in profile.get("config", {}).items():
            cfg["trading"][k] = v
        for k, v in profile.get("labels", {}).items():
            cfg["labels"][k] = v

        tc = profile.get("config", {})
        print(f"  Config: T_up={tc.get('T_up', '-')}, T_down={tc.get('T_down', '-')}, "
              f"sl={tc.get('sl_pct', '-')}, tp={tc.get('tp_pct', '-')}")

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

    # ── Apply pos_size override ───────────────────────────────────────────────
    if args.pos_size is not None:
        cfg["trading"]["position_size_pct"] = args.pos_size

    # ── Sweep mode ────────────────────────────────────────────────────────────
    if args.sweep:
        sc         = cfg.get("sweep", {})
        T_values   = sc.get("T_values",  [0.60, 0.62, 0.64, 0.66, 0.68])
        leverages  = sc.get("leverages", [cfg["trading"]["position_size_pct"]])
        liq_thresh = sc.get("liq_threshold", 0.20)
        n_sims     = sc.get("n_sims", 10_000)
        tp_pct     = cfg["labels"].get("tp_pct", cfg["trading"].get("tp_pct", 0.0045))
        sl_pct     = cfg["labels"].get("sl_pct", cfg["trading"].get("sl_pct", 0.0015))
        rng        = np.random.default_rng(42)
        test_days  = 292  # approximate length of test split in calendar days

        W = 100
        col_hdr = (
            f"  {'T':>5}  {'Ret%':>6}  {'Ann%':>6}  {'Trd':>4}  {'Days':>4}  "
            f"{'Trd/d':>5}  {'Win%':>5}  {'PF':>5}  {'MaxDD%':>7}  "
            f"  {'P(-30%)':>8}  {'P(-50%)':>8}  {'P(-80%)':>8}"
        )

        print(f"\n{'=' * W}")
        print(f"  SWEEP  |  test={test_days}d  |  MC risk: {n_sims:,} paths/scenario  |  "
              f"P(-X%) = prob capital drops >=X% at any point in 1 yr")
        print(f"{'=' * W}")
        print(col_hdr)

        for lev in leverages:
            cfg["trading"]["position_size_pct"] = lev
            win_pct  = lev * tp_pct
            loss_pct = lev * sl_pct
            print(f"\n  {'─' * (W - 2)}")
            print(f"  Leverage {lev:.0f}x  "
                  f"(TP/trade={win_pct*100:.2f}%  SL/trade={loss_pct*100:.3f}%)")
            print(f"  {'─' * (W - 2)}")

            for T in T_values:
                _, _, rep = run_backtest(
                    df_feat=df_all, probs=probs, cfg=cfg,
                    T_up=T, T_down=T,
                    d_max=args.d_max, time_stop=args.time_stop, min_vol=args.min_vol,
                )
                if "error" in rep:
                    print(f"  T={T:.2f}  (no trades)")
                    continue

                wr        = rep["win_rate"] / 100.0
                n_trades  = rep["n_trades"]
                tpd       = n_trades / test_days
                trades_yr = int(tpd * 365)
                ann_ret   = rep["total_return_pct"] * (365.0 / test_days)

                # Use ACTUAL observed avg_win / avg_loss from the backtest
                # (not theoretical TP/SL) so that time-stop exits are reflected.
                ic = cfg["trading"]["initial_capital"]
                win_pct_mc  = rep["avg_win"]  / ic        # avg win as fraction of starting capital
                loss_pct_mc = -rep["avg_loss"] / ic       # avg loss as fraction of starting capital (positive)
                # Fallback to theoretical if no wins/losses observed
                if win_pct_mc <= 0:
                    win_pct_mc = win_pct
                if loss_pct_mc <= 0:
                    loss_pct_mc = loss_pct

                # Monte Carlo: prob of capital dropping below threshold at any
                # point during 1 year.  thresholds: 0.70 (-30%), 0.50 (-50%),
                # 0.20 (-80% / near-ruin)
                p30 = _liquidation_prob(wr, win_pct_mc, loss_pct_mc, trades_yr, n_sims, 0.70, rng)
                p50 = _liquidation_prob(wr, win_pct_mc, loss_pct_mc, trades_yr, n_sims, 0.50, rng)
                p80 = _liquidation_prob(wr, win_pct_mc, loss_pct_mc, trades_yr, n_sims, liq_thresh, rng)

                flag = "  ***" if rep["profit_factor"] >= 1.20 and n_trades >= 50 else \
                       "  *"   if rep["profit_factor"] >= 1.05 and n_trades >= 30 else ""

                print(
                    f"  {T:.2f}  "
                    f"{rep['total_return_pct']:>+6.1f}%  "
                    f"{ann_ret:>+6.1f}%  "
                    f"{n_trades:>4}  "
                    f"{rep['active_days']:>4}  "
                    f"{tpd:>5.2f}  "
                    f"{rep['win_rate']:>5.1f}%  "
                    f"{rep['profit_factor']:>5.2f}  "
                    f"{rep['max_drawdown_pct']:>7.1f}%  "
                    f"  {p30*100:>6.1f}%  "
                    f"{p50*100:>6.1f}%  "
                    f"{p80*100:>6.1f}%"
                    f"{flag}"
                )

        print(f"\n{'=' * W}")
        print(f"  Ann%   = annualised return extrapolated from {test_days}-day test")
        print(f"  Trd/d  = trades per calendar day")
        print(f"  P(-X%) = Monte Carlo prob of experiencing a -30%/-50%/-80% drawdown at any point in 1 yr")
        print(f"  ***    = PF >= 1.20 and >= 50 trades  (statistically meaningful)")
        print(f"  *      = PF >= 1.05 and >= 30 trades  (moderate sample)")
        print(f"{'=' * W}")
        return

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
