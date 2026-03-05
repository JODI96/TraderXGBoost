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

import features as feat_mod
from sim.portfolio   import Portfolio
from sim.execution   import ExecutionEngine
from sim.replay_feed import ReplayFeed


def _feat(feat_row, key):
    """Safely extract a float from a feature row; returns None if missing/NaN."""
    try:
        v = float(feat_row[key])
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 4)
    except Exception:
        return None


# ── Global browser-client registry ────────────────────────────────────────────
_clients: set = set()


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
    _clients.add(websocket)
    try:
        async for _ in websocket:
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
    with open(p / "feature_columns.json") as f:
        feat_cols = json.load(f)
    return model, feat_cols


def _predict(model, feat_cols, feat_row):
    arr = feat_row.values.reshape(1, -1).astype(np.float32)
    dm  = xgb.DMatrix(arr, feature_names=feat_cols)
    return model.predict(dm)[0]


# ── Replay trading loop ────────────────────────────────────────────────────────
async def _replay_loop(cfg: dict, data_file: str, speed: float,
                       start_ts, end_ts, ws_port: int) -> None:
    art_dir  = cfg["training"]["artifacts_dir"]
    log_file = cfg["simulation"]["replay"].get("log_file", "sim/logs/replay_log.jsonl")
    tc = cfg["trading"]
    lc = cfg["labels"]

    print("[replay] Loading model artifacts ...")
    model, feat_cols = _load_artifacts(art_dir)

    cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
               lc["slippage"]  + lc["spread"]) * 2
    portfolio   = Portfolio(
        initial_capital = tc["initial_capital"],
        cost_rt         = cost_rt,
        cooldown_bars   = tc["cooldown"],
    )
    engine      = ExecutionEngine(cfg, portfolio, log_file)
    feat_engine = feat_mod.FeatureEngine(cfg, feature_cols=feat_cols)

    feed      = ReplayFeed(data_file, speed_multiplier=0,
                           start_ts=start_ts, end_ts=end_ts)
    total     = len(feed)
    bar_i     = 0
    last_price = 0.0
    last_ts    = None
    cvd_col   = f"cvd_{cfg['features']['cvd_window']}"   # e.g. "cvd_20"

    # sleep interval between bars (0 = yield only, no real sleep)
    sleep_s = (1.0 / speed) if speed > 0 else 0.0

    print(f"[replay] {total:,} bars | speed={'max' if speed == 0 else f'{speed:.0f} bars/s'}")
    print(f"[replay] Dashboard -> http://localhost:8080\n")

    # Wait for at least one browser client before starting
    print("[replay] Waiting for browser to connect ...")
    while not _clients:
        await asyncio.sleep(0.2)
    print("[replay] Browser connected. Starting replay ...")

    for candle in feed:
        ts    = candle.name
        price = float(candle["close"])
        vol   = float(candle["volume"])
        buy_v = float(candle.get("taker_buy_vol", vol / 2))
        last_price = price
        last_ts    = ts
        bar_i += 1

        # ── Broadcast candle ──────────────────────────────────────────────────
        await _broadcast({
            "type":    "candle",
            "ts":      int(ts.timestamp() * 1000),
            "open":    float(candle["open"]),
            "high":    float(candle["high"]),
            "low":     float(candle["low"]),
            "close":   float(candle["close"]),
            "volume":  vol,
            "buy_vol": buy_v,
        })

        # ── Replay progress ───────────────────────────────────────────────────
        if bar_i % 50 == 0 or bar_i == total:
            await _broadcast({
                "type":     "replay_info",
                "bar":      bar_i,
                "total":    total,
                "ts":       int(ts.timestamp() * 1000),
                "pct":      round(bar_i / total * 100, 1),
            })

        # ── Feature update ────────────────────────────────────────────────────
        feat_row = feat_engine.update(candle)
        if feat_row is None:
            # still warming up
            await _broadcast({"type": "warmup",
                               "remaining": feat_engine.min_warmup - len(feat_engine.buffer),
                               "price": price})
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            elif bar_i % 100 == 0:
                await asyncio.sleep(0)
            continue

        # ── Predict + execute ─────────────────────────────────────────────────
        probs  = _predict(model, feat_cols, feat_row)
        event  = engine.on_bar(feat_row, probs, ts, price)

        p_down, _, p_up = probs
        equity  = portfolio.mark_to_market(price)
        balance = portfolio.capital
        pos     = portfolio.position
        status  = "FLAT" if pos is None else ("LONG" if pos.direction == 1 else "SHORT")

        # ── Broadcast stats ───────────────────────────────────────────────────
        n_trades = len(portfolio.trade_log)
        wins     = sum(1 for t in portfolio.trade_log if t.net_pnl > 0)
        stats: dict = {
            "type":      "stats",
            "ts":        int(ts.timestamp() * 1000),
            "price":     round(price, 2),
            "p_up":      round(float(p_up),   4),
            "p_down":    round(float(p_down), 4),
            "balance":   round(balance, 2),
            "equity":    round(equity,  2),
            "position":  status,
            "n_trades":  n_trades,
            "win_rate":  round(wins / n_trades * 100, 1) if n_trades else 0,
            # ── Indicators ───────────────────────────────────────────────────
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
                    "price":       round(float(event.get("exit_price", price)), 2),
                    "net_pnl":     round(float(pnl), 2),
                    "reason":      reason,
                    "balance":     round(portfolio.capital, 2),
                    "win_rate":    round(wins2 / nt * 100, 1) if nt else 0,
                    "direction":   last_t.direction    if last_t else "",
                    "entry_price": round(last_t.entry_price, 2) if last_t else 0,
                    "exit_price":  round(last_t.exit_price,  2) if last_t else 0,
                })

        # ── Pacing ────────────────────────────────────────────────────────────
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
