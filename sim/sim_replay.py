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
    lc  = cfg["labels"]
    tc  = cfg["trading"]
    cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
               lc["slippage"]  + lc["spread"]) * 2

    portfolio = Portfolio(
        initial_capital = tc["initial_capital"],
        cost_rt         = cost_rt,
        cooldown_bars   = tc["cooldown"],
    )
    engine = ExecutionEngine(cfg, portfolio, log_file)
    feat_engine = feat_mod.FeatureEngine(cfg, feature_cols=feat_cols)

    # ── Build DMatrix for batch prediction on buffer snapshots ───────────────
    def predict_probs(feat_row: pd.Series) -> np.ndarray:
        arr = feat_row.values.reshape(1, -1).astype(np.float32)
        dm  = xgb.DMatrix(arr, feature_names=feat_cols)
        return model.predict(dm)[0]   # shape (3,)

    # ── Replay feed ───────────────────────────────────────────────────────────
    start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end_ts   = pd.Timestamp(args.end,   tz="UTC") if args.end   else None
    feed     = ReplayFeed(data_file, speed_multiplier=speed,
                          start_ts=start_ts, end_ts=end_ts)

    print(f"\n{'='*60}")
    print(f" Replay Simulation")
    print(f" Data:  {data_file}")
    print(f" Bars:  {len(feed):,}")
    print(f" Speed: {'max' if speed == 0 else f'{speed}x'}")
    print(f" Log:   {log_file}")
    print(f"{'='*60}\n")

    bar_count   = 0
    trade_count = 0

    with tqdm(total=len(feed), desc="Replay", unit="bar") as pbar:
        for candle in feed:
            ts    = candle.name   # pd.Timestamp
            price = float(candle["close"])

            # Update feature buffer
            feat_row = feat_engine.update(candle)
            if feat_row is None:
                pbar.update(1)
                continue   # still warming up

            # Predict
            probs = predict_probs(feat_row)

            # Execute
            event = engine.on_bar(feat_row, probs, ts, price)
            if event and event.get("event") == "OPEN":
                trade_count += 1
                pbar.write(f"  [{ts}] {event['direction']} @ {price:.2f}  "
                           f"p_up={event.get('p_up', 0):.3f}  "
                           f"p_down={event.get('p_down', 0):.3f}")
            bar_count += 1
            pbar.update(1)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    engine.force_close(price, ts)
    engine.close_log()

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = portfolio.summary()
    print(f"\n{'─'*50}")
    print(f" Replay Complete  |  {bar_count:,} bars processed")
    print(f"{'─'*50}")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")

    # Save portfolio state
    from pathlib import Path
    art = Path(cfg["training"]["artifacts_dir"])
    portfolio.save(str(art / "replay_portfolio.json"))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Historical replay simulation")
    parser.add_argument("--data",   default=None,  help="CSV data file path")
    parser.add_argument("--speed",  type=float, default=None,
                        help="Speed multiplier (0=max, 1=realtime, 60=1bar/s)")
    parser.add_argument("--start",  default=None,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=None,  help="End date YYYY-MM-DD")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_replay(cfg, args)


if __name__ == "__main__":
    main()
