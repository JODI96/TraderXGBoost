"""
sim/sim_replay.py – Historical replay simulation using trained model.

Feeds bars from a CSV file through the FeatureEngine → XGBoost model →
ExecutionEngine → Portfolio, logging every bar and trade.

Usage
-----
    python sim/sim_replay.py                          # uses config.yaml defaults
    python sim/sim_replay.py --data Data/BTCUSDT/full_year/2025_1m.csv
    python sim/sim_replay.py --speed 60               # 1 bar per second
    python sim/sim_replay.py --start 2025-06-01
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Add project root to path when running as script from sim/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from tqdm import tqdm

import data as data_mod
import features as feat_mod
from sim.portfolio  import Portfolio
from sim.execution  import ExecutionEngine
from sim.replay_feed import ReplayFeed


# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts(art_dir: str = "models"):
    import json
    from pathlib import Path
    p = Path(art_dir)
    model = xgb.Booster()
    model.load_model(str(p / "xgb_model.json"))
    try:
        model.set_param({"device": "cuda"})
    except Exception:
        pass
    with open(p / "feature_columns.json") as f:
        feat_cols = json.load(f)
    return model, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
def run_replay(cfg: dict, args) -> None:
    sim_cfg  = cfg["simulation"]["replay"]
    data_file = args.data or sim_cfg["data_file"]
    speed     = args.speed if args.speed is not None else sim_cfg.get("speed_multiplier", 0)
    log_file  = sim_cfg.get("log_file", "sim/logs/replay_log.jsonl")

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("Loading model artifacts …")
    model, feat_cols = load_artifacts(cfg["training"]["artifacts_dir"])

    # ── Build engine objects ──────────────────────────────────────────────────
    lc      = cfg["labels"]
    tc      = cfg["trading"]
    cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
               lc["slippage"]  + lc["spread"]) * 2

    portfolio = Portfolio(
        initial_capital = tc["initial_capital"],
        cost_rt         = cost_rt,
        cooldown_bars   = tc["cooldown"],
    )
    engine   = ExecutionEngine(cfg, portfolio, log_file)
    start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end_ts   = pd.Timestamp(args.end,   tz="UTC") if args.end   else None

    last_price = 0.0
    last_ts    = None
    bar_count  = 0
    trade_count = 0
    open_event  = {}

    # ── PREBATCH MODE: load all → batch features → single GPU predict ─────────
    if getattr(args, "prebatch", False):
        print("Pre-batch mode: computing all features + GPU batch prediction …")
        df_raw = data_mod.load_csv(data_file)
        if start_ts:
            df_raw = df_raw[df_raw.index >= start_ts]
        if end_ts:
            df_raw = df_raw[df_raw.index <= end_ts]

        df_feat = feat_mod.compute_features(df_raw, cfg)
        avail   = [c for c in feat_cols if c in df_feat.columns]
        df_feat = df_feat.dropna(subset=avail).copy()

        print(f"  Running GPU batch prediction on {len(df_feat):,} bars …")
        dm        = xgb.DMatrix(df_feat[avail].values.astype(np.float32),
                                feature_names=avail)
        all_probs = model.predict(dm).reshape(-1, 3)

        print(f"\n{'='*60}")
        print(f" Replay Simulation  [PREBATCH / GPU]")
        print(f" Data:  {data_file}")
        print(f" Bars:  {len(df_feat):,}")
        print(f" Log:   {log_file}")
        print(f"{'='*60}\n")

        with tqdm(total=len(df_feat), desc="Replay", unit="bar") as pbar:
            for i, (ts, feat_row) in enumerate(df_feat.iterrows()):
                last_price = float(feat_row["close"])
                last_ts    = ts
                probs      = all_probs[i]
                event      = engine.on_bar(feat_row, probs, ts, last_price)

                if event and event.get("event") == "OPEN":
                    trade_count += 1
                    open_event = event
                elif event and event.get("event") == "CLOSE":
                    reason  = event.get("exit_reason", "")
                    net_pnl = event.get("net_pnl", 0.0)
                    outcome = "CORRECT" if reason == "TP" else ("WRONG" if reason == "SL" else "TIME")
                    sign    = "+" if net_pnl >= 0 else ""
                    pbar.write(
                        f"  #{trade_count:3d} | {open_event.get('direction', '?'):5s} | "
                        f"in {open_event.get('price', 0.0):.2f} -> out {event.get('exit_price', last_price):.2f} | "
                        f"p_up={open_event.get('p_up', 0.0):.3f} p_dn={open_event.get('p_down', 0.0):.3f} | "
                        f"PnL={sign}{net_pnl:.2f}$ [{reason}] {outcome:7s} | "
                        f"Balance={portfolio.capital:.2f}$"
                    )
                    open_event = {}
                bar_count += 1
                pbar.update(1)

    # ── STREAMING MODE: bar-by-bar FeatureEngine + per-bar predict ────────────
    else:
        feat_engine = feat_mod.FeatureEngine(cfg, feature_cols=feat_cols)

        def predict_probs(feat_row: pd.Series) -> np.ndarray:
            arr = feat_row.values.reshape(1, -1).astype(np.float32)
            dm  = xgb.DMatrix(arr, feature_names=feat_cols)
            return model.predict(dm)[0]

        feed = ReplayFeed(data_file, speed_multiplier=speed,
                          start_ts=start_ts, end_ts=end_ts)

        print(f"\n{'='*60}")
        print(f" Replay Simulation")
        print(f" Data:  {data_file}")
        print(f" Bars:  {len(feed):,}")
        print(f" Speed: {'max' if speed == 0 else f'{speed}x'}")
        print(f" Log:   {log_file}")
        print(f"{'='*60}\n")

        with tqdm(total=len(feed), desc="Replay", unit="bar") as pbar:
            for candle in feed:
                last_ts    = candle.name
                last_price = float(candle["close"])

                feat_row = feat_engine.update(candle)
                if feat_row is None:
                    pbar.update(1)
                    continue

                probs = predict_probs(feat_row)
                event = engine.on_bar(feat_row, probs, last_ts, last_price)

                if event and event.get("event") == "OPEN":
                    trade_count += 1
                    open_event = event
                elif event and event.get("event") == "CLOSE":
                    reason  = event.get("exit_reason", "")
                    net_pnl = event.get("net_pnl", 0.0)
                    outcome = "CORRECT" if reason == "TP" else ("WRONG" if reason == "SL" else "TIME")
                    sign    = "+" if net_pnl >= 0 else ""
                    pbar.write(
                        f"  #{trade_count:3d} | {open_event.get('direction', '?'):5s} | "
                        f"in {open_event.get('price', 0.0):.2f} -> out {event.get('exit_price', last_price):.2f} | "
                        f"p_up={open_event.get('p_up', 0.0):.3f} p_dn={open_event.get('p_down', 0.0):.3f} | "
                        f"PnL={sign}{net_pnl:.2f}$ [{reason}] {outcome:7s} | "
                        f"Balance={portfolio.capital:.2f}$"
                    )
                    open_event = {}
                bar_count += 1
                pbar.update(1)

    # ── Cleanup (shared) ──────────────────────────────────────────────────────
    if last_ts is not None:
        engine.force_close(last_price, last_ts)
    engine.close_log()

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = portfolio.summary()
    print(f"\n{'─'*50}")
    print(f" Replay Complete  |  {bar_count:,} bars processed")
    print(f"{'─'*50}")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")

    from pathlib import Path
    art = Path(cfg["training"]["artifacts_dir"])
    portfolio.save(str(art / "replay_portfolio.json"))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Historical replay simulation")
    parser.add_argument("--data",     default=None,  help="CSV data file path")
    parser.add_argument("--speed",    type=float, default=None,
                        help="Speed multiplier (0=max, 1=realtime, 60=1bar/s)")
    parser.add_argument("--start",    default=None,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default=None,  help="End date YYYY-MM-DD")
    parser.add_argument("--prebatch", action="store_true",
                        help="Pre-compute all features + run single GPU batch prediction (fastest mode)")
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_replay(cfg, args)


if __name__ == "__main__":
    main()
