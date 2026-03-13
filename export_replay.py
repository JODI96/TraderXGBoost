"""
export_replay.py – Run LOCALLY to generate replay_data.json.

Usage:
    python export_replay.py --data Data/BTCUSDT/monthly/2026-01_1m.csv

Output:
    sim/static/replay_data.json   ← commit this file to the repo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import data as data_mod
import features as feat_mod
from backtest import run_backtest

BASE      = Path(__file__).parent
MAX_BARS  = 5000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True)
    parser.add_argument("--max-bars", type=int, default=MAX_BARS)
    parser.add_argument("--out",      default="sim/static/replay_data.json")
    args = parser.parse_args()

    with open(BASE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # 1. Load data
    print("Loading data …")
    df = data_mod.load_csv(str(BASE / args.data))
    if len(df) > args.max_bars:
        df = df.iloc[-args.max_bars:].copy()
        print(f"Trimmed to last {args.max_bars} bars")

    # 2. Features
    print("Computing features …")
    df_feat = feat_mod.compute_features(df, cfg)
    df_feat = df_feat.dropna().copy()

    # 3. Model
    print("XGBoost inference …")
    model = xgb.Booster()
    model.load_model(str(BASE / "models" / "xgb_model.json"))
    feat_cols = feat_mod.get_feature_columns(cfg)
    feat_cols = [c for c in feat_cols if c in df_feat.columns]
    dmatrix   = xgb.DMatrix(df_feat[feat_cols].values, feature_names=feat_cols)
    probs     = model.predict(dmatrix)

    n_cls = int(cfg.get("model", {}).get("num_class", 5))
    if probs.ndim == 1:
        probs = probs.reshape(-1, n_cls)

    # 4. Backtest
    print("Running backtest …")
    trades_df, equity_curve, report = run_backtest(df_feat, probs, cfg)
    print(f"{len(trades_df)} trades found.")

    # 5. Index trades by timestamp
    def _col(frame, *names):
        return next((c for c in names if c in frame.columns), None)

    entry_col  = _col(trades_df, "entry_time",  "open_time")
    exit_col   = _col(trades_df, "exit_time",   "close_time")
    dir_col    = _col(trades_df, "side",         "direction")
    eprice_col = _col(trades_df, "entry_price",  "entry")
    pnl_col    = _col(trades_df, "pnl_pct",      "pnl", "return")
    sl_col     = _col(trades_df, "sl")
    tp_col     = _col(trades_df, "tp")
    reason_col = _col(trades_df, "reason",       "exit_reason")

    by_entry: dict = {}
    by_exit:  dict = {}
    if not trades_df.empty:
        if entry_col:
            for _, t in trades_df.iterrows():
                by_entry[str(pd.Timestamp(t[entry_col]))] = t.to_dict()
        if exit_col:
            for _, t in trades_df.iterrows():
                by_exit[str(pd.Timestamp(t[exit_col]))] = t.to_dict()

    # 6. Build event stream
    print("Building event stream …")
    init_cap = float(cfg.get("backtest", {}).get("initial_capital", 10.0))
    balance  = init_cap
    n_trades = n_wins = 0
    open_t   = None
    events: list[dict] = []

    events.append({"type": "replay_info", "ts": int(df_feat.index[0].timestamp() * 1000), "pct": 0})

    n_bars = len(df_feat)
    for i, (ts, row) in enumerate(df_feat.iterrows()):
        ts_ms  = int(ts.timestamp() * 1000)
        ts_str = str(ts)
        p_row  = probs[i] if i < len(probs) else np.zeros(n_cls)
        p_up   = float(p_row[2]) if len(p_row) > 2 else 0.0
        p_down = float(p_row[0]) if len(p_row) > 0 else 0.0

        if ts_str in by_entry:
            t = by_entry[ts_str]
            open_t = t
            events.append({
                "type":      "trade_open", "ts": ts_ms,
                "direction": str(t.get(dir_col,    "LONG")),
                "p_up":      round(p_up,   4),
                "p_down":    round(p_down, 4),
                "price":     float(row["close"]),
                "sl":        float(t[sl_col]) if sl_col and sl_col in t else round(float(row["close"]) * 0.9985, 2),
                "tp":        float(t[tp_col]) if tp_col and tp_col in t else round(float(row["close"]) * 1.0045, 2),
            })

        if ts_str in by_exit:
            t   = by_exit[ts_str]
            pnl = float(t[pnl_col]) if pnl_col and pnl_col in t else 0.0
            balance *= (1.0 + pnl)
            n_trades += 1
            if pnl > 0:
                n_wins += 1
            open_t = None
            events.append({
                "type":        "trade_close", "ts": ts_ms,
                "net_pnl":     round(pnl * balance, 6),
                "direction":   str(t.get(dir_col,    "LONG")),
                "entry_price": float(t[eprice_col]) if eprice_col and eprice_col in t else float(row["open"]),
                "exit_price":  float(row["close"]),
                "reason":      str(t.get(reason_col, "TP/SL")),
            })

        eq_val = float(equity_curve[i]) if i < len(equity_curve) else balance
        events.append({
            "type": "candle", "ts": ts_ms,
            "open":    float(row["open"]),  "high":    float(row["high"]),
            "low":     float(row["low"]),   "close":   float(row["close"]),
            "volume":  float(row["volume"]),
            "buy_vol": float(row["taker_buy_vol"]) if "taker_buy_vol" in row else float(row["volume"]) * 0.5,
        })

        events.append({
            "type": "stats", "ts": ts_ms,
            "balance":   round(balance, 4),
            "vwap":      float(row["vwap"])      if "vwap"      in row else float(row["close"]),
            "rel_vol":   float(row["rel_vol"])   if "rel_vol"   in row else 1.0,
            "buy_ratio": float(row["buy_ratio"]) if "buy_ratio" in row else 0.5,
            "equity":    round(eq_val, 4),
            "n_trades":  n_trades,
            "win_rate":  round(n_wins / n_trades if n_trades else 0, 3),
            "position":  str(open_t.get(dir_col, "LONG")) if open_t else None,
            "unreal":    0.0,
            "entry":     float(open_t[eprice_col]) if open_t and eprice_col and eprice_col in open_t else None,
            "sl":        float(open_t[sl_col])     if open_t and sl_col     and sl_col     in open_t else None,
            "tp":        float(open_t[tp_col])     if open_t and tp_col     and tp_col     in open_t else None,
            "p_up":  round(p_up,   4),
            "p_down": round(p_down, 4),
        })

        if i % 100 == 0:
            events.append({"type": "replay_info", "ts": ts_ms, "pct": round(i / n_bars * 100, 1)})

    final_eq  = float(equity_curve[-1]) if len(equity_curve) else balance
    total_ret = (final_eq / init_cap - 1.0) if init_cap else 0.0
    events.append({
        "type": "replay_done",
        "summary": {
            "n_trades":         len(trades_df),
            "win_rate_pct":     round(n_wins / max(n_trades, 1) * 100, 1),
            "net_pnl":          round(final_eq - init_cap, 4),
            "total_return_pct": round(total_ret * 100, 2),
            "max_drawdown_pct": round(float(report.get("max_drawdown", report.get("max_dd", 0))) * 100, 2),
            "profit_factor":    round(float(report.get("profit_factor", 0)), 2),
            "final_capital":    round(final_eq, 4),
        },
    })

    # 7. Save
    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"events": events}, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Saved {len(events)} events → {out_path}  ({size_mb:.1f} MB)")
    print("Next: git add sim/static/replay_data.json && git push")


if __name__ == "__main__":
    main()
