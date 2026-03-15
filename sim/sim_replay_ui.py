"""
sim/sim_replay_ui.py – Historical replay simulation with browser dashboard.

Same chart UI as sim_live_ui.py, fed from a CSV file instead of Binance WS.

Opens two local servers:
  http://localhost:8080   Trading dashboard (candlestick chart, stats, trade log)
  ws://localhost:8765     Real-time data feed from the replay engine to the browser

Usage
-----
    python sim/sim_replay_ui.py
    python sim/sim_replay_ui.py --data Data/BTCUSDT/full_year/2025_1m.csv
    python sim/sim_replay_ui.py --speed 30        # 30 bars/sec
    python sim/sim_replay_ui.py --speed 0         # max speed
    python sim/sim_replay_ui.py --start 2025-06-01 --end 2025-07-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

try:
    import websockets
except ImportError:
    raise ImportError("websockets package required: pip install websockets>=10.0")

import logging
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

import data as data_mod
import features as feat_mod
from sim.portfolio   import Portfolio
from sim.execution   import ExecutionEngine


def _feat(feat_row, key):
    """Safely extract a float from a feature row; returns None if missing/NaN."""
    try:
        v = float(feat_row[key])
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 4)
    except Exception:
        return None


# ── Global browser-client registry ────────────────────────────────────────────
_clients: set = set()
_paused: bool = False


async def _broadcast(msg: dict) -> None:
    if not _clients:
        return
    payload = json.dumps(msg, default=str)
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send(payload)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


async def _ws_browser_handler(websocket, *_args) -> None:
    global _paused
    _clients.add(websocket)
    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
                cmd = data.get("cmd")
                if cmd == "pause":
                    _paused = True
                elif cmd == "resume":
                    _paused = False
            except Exception:
                pass
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


def _start_http_server(static_dir: str, port: int) -> None:
    class _QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=static_dir, **kwargs)
    httpd = HTTPServer(("", port), _QuietHandler)
    httpd.serve_forever()


def _load_artifacts(art_dir: str):
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


# ── Replay trading loop ────────────────────────────────────────────────────────
async def _replay_loop(cfg: dict, data_file: str, speed: float,
                       start_ts, end_ts, ws_port: int) -> None:
    art_dir  = cfg["training"]["artifacts_dir"]
    log_file = cfg["simulation"]["replay"].get("log_file", "sim/logs/replay_log.jsonl")
    tc  = cfg["trading"]
    lc  = cfg["labels"]
    cvd_col = f"cvd_{cfg['features']['cvd_window']}"

    print("[replay] Loading model artifacts ...")
    model, feat_cols = _load_artifacts(art_dir)

    portfolio = Portfolio(
        initial_capital = tc["initial_capital"],
        maker_fee       = lc["maker_fee"],
        taker_fee       = lc["taker_fee"],
        cooldown_bars   = tc["cooldown"],
    )
    engine = ExecutionEngine(cfg, portfolio, log_file)

    # ── Pre-batch: compute all features + single GPU prediction pass ──────────
    print("[replay] Loading data and computing features ...")
    df_raw = data_mod.load_csv(data_file)
    if start_ts:
        df_raw = df_raw[df_raw.index >= start_ts]
    if end_ts:
        df_raw = df_raw[df_raw.index <= end_ts]

    df_feat = feat_mod.compute_features(df_raw, cfg)
    avail   = [c for c in feat_cols if c in df_feat.columns]
    df_feat = df_feat.dropna(subset=avail).copy()
    df_raw  = df_raw.loc[df_feat.index]   # align raw OHLCV to post-warmup rows

    print(f"[replay] GPU batch prediction on {len(df_feat):,} bars ...")
    dm        = xgb.DMatrix(df_feat[avail].values.astype(np.float32),
                            feature_names=avail)
    all_probs = model.predict(dm).reshape(-1, 3)

    total   = len(df_feat)
    sleep_s = (1.0 / speed) if speed > 0 else 0.0

    print(f"[replay] {total:,} bars ready | speed={'max' if speed == 0 else f'{speed:.0f} bars/s'}")
    print(f"[replay] Dashboard -> http://localhost:8080\n")

    # Wait for browser before starting
    print("[replay] Waiting for browser to connect ...")
    while not _clients:
        await asyncio.sleep(0.2)
    print("[replay] Browser connected. Starting replay ...")

    last_price = 0.0
    last_ts    = None

    for bar_i, (ts, feat_row) in enumerate(df_feat.iterrows(), start=1):
        raw_row    = df_raw.iloc[bar_i - 1]
        last_price = float(raw_row["close"])
        last_ts    = ts
        probs      = all_probs[bar_i - 1]
        p_down, _, p_up = probs

        # ── Broadcast candle ──────────────────────────────────────────────────
        await _broadcast({
            "type":    "candle",
            "ts":      int(ts.timestamp() * 1000),
            "open":    float(raw_row["open"]),
            "high":    float(raw_row["high"]),
            "low":     float(raw_row["low"]),
            "close":   float(raw_row["close"]),
            "volume":  float(raw_row["volume"]),
            "buy_vol": float(raw_row.get("taker_buy_vol", raw_row["volume"] / 2)),
        })

        # ── Replay progress ───────────────────────────────────────────────────
        if bar_i % 50 == 0 or bar_i == total:
            await _broadcast({
                "type":  "replay_info",
                "bar":   bar_i,
                "total": total,
                "ts":    int(ts.timestamp() * 1000),
                "pct":   round(bar_i / total * 100, 1),
            })

        # ── Execute ───────────────────────────────────────────────────────────
        event   = engine.on_bar(feat_row, probs, ts, last_price)
        equity  = portfolio.mark_to_market(last_price)
        balance = portfolio.capital
        pos     = portfolio.position
        status  = "FLAT" if pos is None else ("LONG" if pos.direction == 1 else "SHORT")

        # ── Broadcast stats ───────────────────────────────────────────────────
        n_trades = len(portfolio.trade_log)
        wins     = sum(1 for t in portfolio.trade_log if t.net_pnl > 0)
        stats: dict = {
            "type":      "stats",
            "ts":        int(ts.timestamp() * 1000),
            "price":     round(last_price, 2),
            "p_up":      round(float(p_up),   4),
            "p_down":    round(float(p_down), 4),
            "balance":   round(balance, 2),
            "equity":    round(equity,  2),
            "position":  status,
            "n_trades":  n_trades,
            "win_rate":  round(wins / n_trades * 100, 1) if n_trades else 0,
            "vwap":      _feat(feat_row, "vwap"),
            "cvd":       _feat(feat_row, cvd_col),
            "rel_vol":   _feat(feat_row, "rel_vol"),
            "buy_ratio": _feat(feat_row, "taker_buy_ratio"),
        }
        if pos is not None:
            stats["sl"]     = round(pos.sl_price,    2)
            stats["tp"]     = round(pos.tp_price,    2)
            stats["entry"]  = round(pos.entry_price, 2)
            stats["unreal"] = round(equity - balance, 2)
        await _broadcast(stats)

        # ── Trade events ──────────────────────────────────────────────────────
        if event:
            ev_type = event.get("event", "")
            if ev_type == "OPEN" and portfolio.position is not None:
                pos2     = portfolio.position
                size_usd = pos2.size * pos2.entry_price
                await _broadcast({
                    "type":      "trade_open",
                    "ts":        int(ts.timestamp() * 1000),
                    "price":     round(pos2.entry_price, 2),
                    "direction": event.get("direction"),
                    "sl":        round(pos2.sl_price, 2),
                    "tp":        round(pos2.tp_price, 2),
                    "size_usd":  round(size_usd, 2),
                    "p_up":      round(float(p_up),   3),
                    "p_down":    round(float(p_down), 3),
                })
            elif ev_type == "CLOSE":
                pnl    = event.get("net_pnl", 0)
                reason = event.get("reason", "")
                nt     = len(portfolio.trade_log)
                wins2  = sum(1 for t in portfolio.trade_log if t.net_pnl > 0)
                last_t = portfolio.trade_log[-1] if portfolio.trade_log else None
                await _broadcast({
                    "type":        "trade_close",
                    "ts":          int(ts.timestamp() * 1000),
                    "price":       round(float(event.get("exit_price", last_price)), 2),
                    "net_pnl":     round(float(pnl), 2),
                    "reason":      reason,
                    "balance":     round(portfolio.capital, 2),
                    "win_rate":    round(wins2 / nt * 100, 1) if nt else 0,
                    "direction":   last_t.direction    if last_t else "",
                    "entry_price": round(last_t.entry_price, 2) if last_t else 0,
                    "exit_price":  round(last_t.exit_price,  2) if last_t else 0,
                })

        # ── Pacing ────────────────────────────────────────────────────────────
        while _paused:
            await asyncio.sleep(0.1)
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        elif bar_i % 100 == 0:
            await asyncio.sleep(0)   # yield to event loop so WS messages flush

    # ── Done ──────────────────────────────────────────────────────────────────
    engine.force_close(last_price, last_ts or "end")
    engine.close_log()

    summary = portfolio.summary()
    await _broadcast({"type": "replay_done", "summary": summary})

    print(f"\n{'='*50}")
    print(" Replay Complete")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")

    art = Path(art_dir)
    portfolio.save(str(art / "replay_portfolio.json"))


# ── Orchestrator ───────────────────────────────────────────────────────────────
async def _main(cfg: dict, data_file: str, speed: float,
                start_ts, end_ts, http_port: int, ws_port: int) -> None:
    static_dir = str(Path(__file__).parent / "static")
    os.makedirs(static_dir, exist_ok=True)

    t = threading.Thread(
        target=_start_http_server, args=(static_dir, http_port), daemon=True
    )
    t.start()
    print(f"[http] Dashboard  ->  http://localhost:{http_port}")
    print(f"[ws]   Data feed  ->  ws://localhost:{ws_port}")

    async with websockets.serve(_ws_browser_handler, "localhost", ws_port):
        await _replay_loop(cfg, data_file, speed, start_ts, end_ts, ws_port)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Replay simulation with browser dashboard")
    parser.add_argument("--data",    default=None,  help="CSV data file")
    parser.add_argument("--speed",   type=float, default=30,
                        help="Bars per second (0=max speed, default=30)")
    parser.add_argument("--start",   default=None,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None,  help="End date YYYY-MM-DD")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--ws-port", type=int, default=8765)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_file = args.data or cfg["simulation"]["replay"]["data_file"]
    start_ts  = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end_ts    = pd.Timestamp(args.end,   tz="UTC") if args.end   else None

    try:
        asyncio.run(_main(cfg, data_file, args.speed, start_ts, end_ts,
                          args.port, args.ws_port))
    except KeyboardInterrupt:
        print("\n[replay] Stopped.")


if __name__ == "__main__":
    main()
