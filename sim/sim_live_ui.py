"""
sim/sim_live_ui.py – Live paper-trading with browser-based dashboard.

Opens two local servers:
  http://localhost:8080   Trading dashboard (candlestick chart, stats, trade log)
  ws://localhost:8765     Real-time data feed from the trading engine to the browser

Pre-warms the FeatureEngine from the most recent historical CSV so that trading
begins on the very first live candle (no live warmup delay).

Usage
-----
    python sim/sim_live_ui.py
    python sim/sim_live_ui.py --symbol ETHUSDT
    python sim/sim_live_ui.py --port 8080 --ws-port 8765

Ctrl+C to stop; portfolio is saved on exit.
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

import urllib.request as _urllib_req

import features as feat_mod
from sim.portfolio       import Portfolio
from sim.execution       import ExecutionEngine
from sim.binance_ws_feed import BinanceWSFeed


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
    """Send a JSON message to all connected browser WebSocket clients."""
    if not _clients:
        return
    payload = json.dumps(msg, default=str)
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send(payload)
        except Exception:
            dead.add(ws)
    _clients -= dead


# ── WebSocket handler (browser side) ──────────────────────────────────────────
async def _ws_browser_handler(websocket, *_args) -> None:
    """Accept and hold browser WebSocket connections."""
    _clients.add(websocket)
    try:
        async for _ in websocket:
            pass        # browser sends nothing; loop keeps connection alive
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


# ── HTTP server (serves index.html + static assets) ───────────────────────────
def _start_http_server(static_dir: str, port: int) -> None:
    """Run a simple HTTP file server in a daemon thread."""
    class _QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass    # suppress per-request console noise

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=static_dir, **kwargs)

    httpd = HTTPServer(("", port), _QuietHandler)
    httpd.serve_forever()


# ── Model helpers ──────────────────────────────────────────────────────────────
def _load_artifacts(art_dir: str):
    p = Path(art_dir)
    model = xgb.Booster()
    model.load_model(str(p / "xgb_model.json"))
    with open(p / "feature_columns.json") as f:
        feat_cols = json.load(f)
    return model, feat_cols


def _predict(model: xgb.Booster, feat_cols: list[str],
             feat_row: pd.Series) -> np.ndarray:
    arr = feat_row.values.reshape(1, -1).astype(np.float32)
    dm  = xgb.DMatrix(arr, feature_names=feat_cols)
    return model.predict(dm)[0]   # shape (3,)


# ── Binance REST pre-warm ──────────────────────────────────────────────────────
def _prewarm_sync(feat_engine: feat_mod.FeatureEngine,
                  symbol: str, buf_size: int) -> int:
    """
    Fetch the most recent `buf_size` closed 1m candles from the Binance REST
    API and feed them into the FeatureEngine buffer.  No API key required.
    Returns the number of bars loaded (0 on failure).
    """
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={symbol}&interval=1m&limit={buf_size + 1}"
    )
    print(f"[prewarm] Fetching {buf_size} recent 1m candles from Binance ({symbol}) ...")
    try:
        with _urllib_req.urlopen(url, timeout=15) as resp:
            klines = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"[prewarm] WARNING: REST fetch failed ({exc}). Warmup will happen live.")
        return 0

    # Drop the last entry – it may still be the open (partial) candle
    klines = klines[:-1]

    for k in klines:
        ts  = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
        row = pd.Series({
            "open":          float(k[1]),
            "high":          float(k[2]),
            "low":           float(k[3]),
            "close":         float(k[4]),
            "volume":        float(k[5]),
            "taker_buy_vol": float(k[9]),
        }, name=ts)
        feat_engine.update(row)

    n = len(klines)
    print(f"[prewarm] Done – {n} live bars loaded. Feature engine ready.")
    return n


async def _prewarm_from_binance(feat_engine: feat_mod.FeatureEngine,
                                symbol: str, buf_size: int) -> int:
    """Async wrapper: runs the REST fetch in a thread so the event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _prewarm_sync, feat_engine, symbol, buf_size)


# ── Main trading loop ──────────────────────────────────────────────────────────
async def _trading_loop(cfg: dict, symbol: str, ws_port: int) -> None:
    sim_cfg  = cfg["simulation"]["live"]
    log_file = sim_cfg.get("log_file", "sim/logs/live_log.jsonl")
    art_dir  = cfg["training"]["artifacts_dir"]
    tc       = cfg["trading"]
    lc       = cfg["labels"]

    # Load model
    print("[live] Loading model artifacts ...")
    model, feat_cols = _load_artifacts(art_dir)

    # Portfolio + execution engine
    cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
               lc["slippage"]  + lc["spread"]) * 2
    portfolio   = Portfolio(
        initial_capital = tc["initial_capital"],
        cost_rt         = cost_rt,
        cooldown_bars   = tc["cooldown"],
    )
    engine      = ExecutionEngine(cfg, portfolio, log_file)
    feat_engine = feat_mod.FeatureEngine(cfg, feature_cols=feat_cols)

    # Pre-warm from Binance REST API
    await _prewarm_from_binance(feat_engine, symbol, cfg["features"]["live_buffer"])

    # Binance WebSocket feed
    ws_url = sim_cfg.get("ws_url", "wss://stream.binance.com:9443/ws")
    feed   = BinanceWSFeed(symbol=symbol, ws_url=ws_url)

    print(f"\n[live] Streaming {symbol} | ws_port={ws_port}")
    print(f"[live] Press Ctrl+C to stop.\n")

    last_price = 0.0
    last_ts    = None
    cvd_col    = f"cvd_{cfg['features']['cvd_window']}"

    try:
        async for candle in feed:
            ts    = candle.name
            price = float(candle["close"])
            vol   = float(candle["volume"])
            buy_v = float(candle.get("taker_buy_vol", vol / 2))
            last_price = price
            last_ts    = ts

            # ── Send candle to browser chart ──────────────────────────────────
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

            feat_row = feat_engine.update(candle)
            if feat_row is None:
                remaining = feat_engine.min_warmup - len(feat_engine.buffer)
                print(f"  [{ts}] Warming up ... ({remaining} bars left)")
                await _broadcast({"type": "warmup", "remaining": remaining,
                                  "price": price})
                continue

            probs  = _predict(model, feat_cols, feat_row)
            event  = engine.on_bar(feat_row, probs, ts, price)

            p_down, _, p_up = probs
            equity  = portfolio.mark_to_market(price)
            balance = portfolio.capital
            pos     = portfolio.position
            status  = "FLAT" if pos is None else \
                      ("LONG" if pos.direction == 1 else "SHORT")

            # ── Console line ──────────────────────────────────────────────────
            unreal_str = ""
            if pos is not None:
                unreal     = equity - balance
                unreal_str = f"  unreal={unreal:+.2f}"
            print(f"  [{ts}]  {price:>10.2f}  "
                  f"p_up={p_up:.3f}  p_dn={p_down:.3f}  "
                  f"pos={status:5s}  "
                  f"bal=${balance:>10,.2f}  eq=${equity:>10,.2f}{unreal_str}")

            # ── Broadcast stats to browser ────────────────────────────────────
            n_trades = len(portfolio.trade_log)
            wins     = sum(1 for t in portfolio.trade_log if t.net_pnl > 0)
            stats: dict = {
                "type":      "stats",
                "ts":        int(ts.timestamp() * 1000),
                "price":     round(price, 2),
                "p_up":      round(float(p_up),   4),
                "p_down":    round(float(p_down), 4),
                "balance":   round(balance, 2),
                "equity":    round(equity, 2),
                "position":  status,
                "n_trades":  n_trades,
                "win_rate":  round(wins / n_trades * 100, 1) if n_trades else 0,
                # ── Indicators ───────────────────────────────────────────────
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

            # ── Trade events ──────────────────────────────────────────────────
            if event:
                ev_type = event.get("event", "")

                if ev_type == "OPEN" and portfolio.position is not None:
                    pos2     = portfolio.position
                    size_usd = pos2.size * pos2.entry_price
                    print(f"\n  {'='*58}")
                    print(f"  ENTRY  {event.get('direction'):5s}  @ {pos2.entry_price:.2f}")
                    print(f"    SL   : {pos2.sl_price:.2f}  "
                          f"({abs(pos2.entry_price - pos2.sl_price):.2f} pts)")
                    print(f"    TP   : {pos2.tp_price:.2f}  "
                          f"({abs(pos2.tp_price - pos2.entry_price):.2f} pts)")
                    print(f"    Size : {pos2.size:.6f} {symbol[:3]}  (${size_usd:,.2f})")
                    print(f"  {'='*58}\n")

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
                    pnl      = event.get("net_pnl", 0)
                    reason   = event.get("reason", "")
                    pnl_sign = "+" if pnl >= 0 else ""
                    nt       = len(portfolio.trade_log)
                    wins2    = sum(1 for t in portfolio.trade_log if t.net_pnl > 0)
                    wr       = f"{wins2/nt*100:.0f}%" if nt else "n/a"
                    exit_p   = event.get("exit_price", price)
                    print(f"\n  {'='*58}")
                    print(f"  EXIT  {reason:6s} @ {exit_p:.2f}  "
                          f"PnL: {pnl_sign}${pnl:.2f}")
                    print(f"    New balance : ${portfolio.capital:,.2f}")
                    print(f"    Total trades: {nt}   Win rate: {wr}")
                    print(f"  {'='*58}\n")

                    last_t = portfolio.trade_log[-1] if portfolio.trade_log else None
                    await _broadcast({
                        "type":        "trade_close",
                        "ts":          int(ts.timestamp() * 1000),
                        "price":       round(float(exit_p), 2),
                        "net_pnl":     round(float(pnl), 2),
                        "reason":      reason,
                        "balance":     round(portfolio.capital, 2),
                        "win_rate":    round(wins2 / nt * 100, 1) if nt else 0,
                        "direction":   last_t.direction if last_t else "",
                        "entry_price": round(last_t.entry_price, 2) if last_t else 0,
                        "exit_price":  round(last_t.exit_price,  2) if last_t else 0,
                    })

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await feed.stop()
        engine.force_close(last_price, last_ts or "end")
        engine.close_log()

        summary = portfolio.summary()
        print(f"\n{'='*50}")
        print(" Live Session Summary")
        print(f"{'='*50}")
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")

        art = Path(art_dir)
        portfolio.save(str(art / "live_portfolio.json"))


# ── Orchestrator ───────────────────────────────────────────────────────────────
async def _main(cfg: dict, symbol: str, http_port: int, ws_port: int) -> None:
    static_dir = str(Path(__file__).parent / "static")
    os.makedirs(static_dir, exist_ok=True)

    # HTTP file server in a background thread
    t = threading.Thread(
        target=_start_http_server, args=(static_dir, http_port), daemon=True
    )
    t.start()
    print(f"[http] Dashboard  ->  http://localhost:{http_port}")
    print(f"[ws]   Data feed  ->  ws://localhost:{ws_port}")

    # Browser WebSocket server + trading loop (same asyncio event loop)
    async with websockets.serve(_ws_browser_handler, "localhost", ws_port):
        await _trading_loop(cfg, symbol, ws_port)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Live trading dashboard")
    parser.add_argument("--symbol",  default=None,
                        help="Symbol (default: from config)")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--port",    type=int, default=8080,
                        help="HTTP port for browser dashboard (default: 8080)")
    parser.add_argument("--ws-port", type=int, default=8765,
                        help="WebSocket port for real-time data (default: 8765)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    symbol = (args.symbol or
              cfg["simulation"]["live"].get("symbol",
              cfg["data"]["symbol"])).upper()

    asyncio.run(_main(cfg, symbol, args.port, args.ws_port))


if __name__ == "__main__":
    main()
