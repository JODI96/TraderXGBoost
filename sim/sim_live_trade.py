"""
sim/sim_live_trade.py – REAL Binance Futures trading with browser dashboard.

*** WARNING: THIS PLACES REAL ORDERS WITH REAL MONEY ***

Uses IDENTICAL signal logic to sim_live_ui.py (same model, features, filters)
but replaces the paper Portfolio with BinancePortfolio which sends live orders.

Setup
-----
  1. Copy .env.example to .env and fill in your Binance Futures API keys
  2. Adjust config.yaml:  trading.initial_capital, position_size_pct, T_up/T_down
  3. Start with a small position_size_pct (e.g. 1.0 = 1x leverage) to verify

Servers
-------
  http://localhost:8080   Trading dashboard
  ws://localhost:8765     Real-time data feed

Usage
-----
    python sim/sim_live_trade.py
    python sim/sim_live_trade.py --symbol BTCUSDT --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional – can set env vars manually

try:
    import websockets
except ImportError:
    raise ImportError("websockets required: pip install websockets>=10.0")

import logging as _logging
_logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

import urllib.request as _urllib_req

import features as feat_mod
from sim.execution          import ExecutionEngine
from sim.binance_portfolio  import BinancePortfolio
from sim.binance_ws_feed    import BinanceWSFeed

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _feat(feat_row, key):
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
    _clients -= dead


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
    HTTPServer(("", port), _QuietHandler).serve_forever()


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


def _predict(model, feat_cols, feat_row):
    arr = feat_row.values.reshape(1, -1).astype(np.float32)
    dm  = xgb.DMatrix(arr, feature_names=feat_cols)
    return model.predict(dm)[0]


def _prewarm_sync(feat_engine, symbol: str, buf_size: int) -> int:
    url = (f"https://fapi.binance.com/fapi/v1/klines"
           f"?symbol={symbol}&interval=1m&limit={buf_size + 1}")
    logger.info(f"[prewarm] Fetching {buf_size} recent 1m candles ({symbol}) ...")
    try:
        with _urllib_req.urlopen(url, timeout=15) as resp:
            klines = json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning(f"[prewarm] REST fetch failed: {exc}")
        return 0
    klines = klines[:-1]  # drop open candle
    for k in klines:
        ts  = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
        row = pd.Series({
            "open": float(k[1]), "high": float(k[2]),
            "low":  float(k[3]), "close": float(k[4]),
            "volume": float(k[5]), "taker_buy_vol": float(k[9]),
        }, name=ts)
        feat_engine.update(row)
    logger.info(f"[prewarm] Done – {len(klines)} bars loaded.")
    return len(klines)


async def _prewarm_async(feat_engine, symbol, buf_size):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _prewarm_sync, feat_engine, symbol, buf_size)


# ── Main trading loop ──────────────────────────────────────────────────────────
async def _trading_loop(cfg: dict, symbol: str, ws_port: int) -> None:
    tc      = cfg["trading"]
    lc      = cfg["labels"]
    art_dir = cfg["training"]["artifacts_dir"]

    # ── API keys from environment ─────────────────────────────────────────────
    api_key    = os.environ.get("BINANCE_API_KEY",    "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        print("\n  ERROR: BINANCE_API_KEY / BINANCE_API_SECRET not set.")
        print("  Copy .env.example to .env and fill in your keys.\n")
        return

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("[live] Loading model ...")
    model, feat_cols = _load_artifacts(art_dir)

    # ── Real portfolio (Binance Futures) ──────────────────────────────────────
    logger.info("[live] Connecting to Binance Futures ...")
    portfolio = BinancePortfolio(api_key, api_secret, symbol, cfg)

    log_file    = cfg["simulation"]["live"].get("log_file", "sim/logs/live_log.jsonl")
    engine      = ExecutionEngine(cfg, portfolio, log_file)
    feat_engine = feat_mod.FeatureEngine(cfg, feature_cols=feat_cols)

    # ── Pre-warm ──────────────────────────────────────────────────────────────
    await _prewarm_async(feat_engine, symbol, cfg["features"]["live_buffer"])

    # ── Binance WS feed ───────────────────────────────────────────────────────
    ws_url = cfg["simulation"]["live"].get("ws_url", "wss://fstream.binance.com/ws")
    feed   = BinanceWSFeed(symbol=symbol, ws_url=ws_url)

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  *** LIVE TRADING – REAL MONEY ON BINANCE FUTURES ***")
    print(f"  Symbol   : {symbol}")
    print(f"  Leverage : {portfolio.leverage}x  (position_size_pct in config)")
    print(f"  Balance  : ${portfolio.capital:,.2f} USDT")
    print(f"  T_up/dn  : {tc['T_up']} / {tc['T_down']}")
    print(f"  SL/TP    : {tc.get('sl_pct',0)*100:.2f}% / {tc.get('tp_pct',0)*100:.2f}%")
    print(f"  Dashboard: http://localhost:8080")
    print("=" * 62)
    print("  Ctrl+C to stop gracefully (open position will be closed)")
    print()

    last_price = 0.0
    last_ts    = None
    cvd_col    = f"cvd_{cfg['features']['cvd_window']}"

    # ── Background SL/TP guard (every 5 seconds between bars) ─────────────────
    _price_ref = [0.0]

    async def _background_guard():
        while True:
            await asyncio.sleep(5)
            if _price_ref[0] > 0:
                try:
                    # Fast-path: detect limit fill and place SL/TP without waiting for bar close
                    portfolio.handle_pending_fill_immediate(
                        sl_pct=tc.get("sl_pct", 0.002),
                        tp_pct=tc.get("tp_pct", 0.006),
                    )
                    # Verify existing position is still protected
                    portfolio.verify_protection(_price_ref[0])
                except Exception as exc:
                    logger.warning(f"[bg_guard] error: {exc}")

    bg_guard_task = asyncio.create_task(_background_guard())

    try:
        async for candle in feed:
            ts    = candle.name
            price = float(candle["close"])
            vol   = float(candle["volume"])
            buy_v = float(candle.get("taker_buy_vol", vol / 2))
            last_price = price
            last_ts    = ts
            _price_ref[0] = price

            # ── Send candle to browser ────────────────────────────────────────
            await _broadcast({
                "type":  "candle",
                "ts":    int(ts.timestamp() * 1000),
                "open":  float(candle["open"]),
                "high":  float(candle["high"]),
                "low":   float(candle["low"]),
                "close": price,
                "volume": vol,
                "buy_vol": buy_v,
            })

            feat_row = feat_engine.update(candle)
            if feat_row is None:
                remaining = feat_engine.min_warmup - len(feat_engine.buffer)
                logger.info(f"[{ts}] Warming up ... ({remaining} bars left)")
                await _broadcast({"type": "warmup", "remaining": remaining, "price": price})
                continue

            probs  = _predict(model, feat_cols, feat_row)
            event  = engine.on_bar(feat_row, probs, ts, price)

            p_down, _, p_up = probs
            equity  = portfolio.mark_to_market(price)
            balance = portfolio.capital
            pos     = portfolio.position
            status  = "FLAT" if pos is None else ("LONG" if pos.direction == 1 else "SHORT")

            # ── Terminal status line ──────────────────────────────────────────
            unreal_str = ""
            if pos is not None:
                unreal     = equity - balance
                unreal_str = f"  unreal={unreal:+.2f}"
            print(f"  [{ts}]  {price:>10.2f}  "
                  f"p_up={p_up:.3f}  p_dn={p_down:.3f}  "
                  f"pos={status:5s}  "
                  f"bal=${balance:>10,.2f}  eq=${equity:>10,.2f}{unreal_str}")

            # ── Browser stats ─────────────────────────────────────────────────
            n_trades = len(portfolio.trade_log)
            wins     = sum(1 for t in portfolio.trade_log if t.net_pnl > 0)
            stats: dict = {
                "type":     "stats",
                "ts":       int(ts.timestamp() * 1000),
                "price":    round(price, 2),
                "p_up":     round(float(p_up),   4),
                "p_down":   round(float(p_down), 4),
                "balance":  round(balance, 2),
                "equity":   round(equity,  2),
                "position": status,
                "n_trades": n_trades,
                "win_rate": round(wins / n_trades * 100, 1) if n_trades else 0,
                "vwap":     _feat(feat_row, "vwap"),
                "cvd":      _feat(feat_row, cvd_col),
                "rel_vol":  _feat(feat_row, "rel_vol"),
                "buy_ratio":_feat(feat_row, "taker_buy_ratio"),
                "live":     True,   # flag so UI can show LIVE badge
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
                        "live":      True,
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
                    print(f"  EXIT   {reason:6s} @ {exit_p:.2f}  "
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
                        "direction":   last_t.direction    if last_t else "",
                        "entry_price": round(last_t.entry_price, 2) if last_t else 0,
                        "exit_price":  round(last_t.exit_price,  2) if last_t else 0,
                        "live":        True,
                    })

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        bg_guard_task.cancel()
        try:
            await bg_guard_task
        except asyncio.CancelledError:
            pass
        await feed.stop()

        if portfolio.position is not None:
            print("\n  Closing open position before exit ...")
            engine.force_close(last_price, last_ts or "end")

        engine.close_log()

        summary = portfolio.summary()
        print(f"\n{'='*50}")
        print(" Live Trading Session Summary")
        print(f"{'='*50}")
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")

        art = Path(art_dir)
        portfolio.save(str(art / "live_trade_portfolio.json"))


# ── Orchestrator ───────────────────────────────────────────────────────────────
async def _main(cfg: dict, symbol: str, http_port: int, ws_port: int) -> None:
    static_dir = str(Path(__file__).parent / "static")
    os.makedirs(static_dir, exist_ok=True)

    threading.Thread(
        target=_start_http_server, args=(static_dir, http_port), daemon=True
    ).start()

    print(f"[http] Dashboard -> http://localhost:{http_port}")
    print(f"[ws]   Data feed -> ws://localhost:{ws_port}")

    async with websockets.serve(_ws_browser_handler, "localhost", ws_port):
        await _trading_loop(cfg, symbol, ws_port)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="LIVE Binance Futures trading")
    parser.add_argument("--symbol",  default=None)
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--ws-port", type=int, default=8765)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    symbol = (args.symbol or
              cfg["simulation"]["live"].get("symbol",
              cfg["data"]["symbol"])).upper()

    asyncio.run(_main(cfg, symbol, args.port, args.ws_port))


if __name__ == "__main__":
    main()
