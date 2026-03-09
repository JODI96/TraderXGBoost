"""
trade_live.py – REAL Binance Futures trading with browser dashboard.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# ── ANSI colours (work on Linux/macOS/Windows Terminal) ───────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    # text
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    # background
    BG_GREEN  = "\033[42m"
    BG_RED    = "\033[41m"
    BG_BLUE   = "\033[44m"
    BG_YELLOW = "\033[43m"

def _clr(text, *codes) -> str:
    return "".join(codes) + str(text) + C.RESET
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
_candle_buffer: list = []          # recent candles replayed to new browser tabs
_CANDLE_BUF_MAX = 500              # keep last N candles in memory


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
        for c in list(_candle_buffer):
            await websocket.send(json.dumps(c, default=str))
    except Exception:
        _clients.discard(websocket)
        return
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
    server = HTTPServer(("", port), _QuietHandler)
    server.allow_reuse_address = True
    server.serve_forever()


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
        _candle_buffer.append({
            "type":     "candle",
            "ts":       int(ts.timestamp() * 1000),
            "open":     float(k[1]),
            "high":     float(k[2]),
            "low":      float(k[3]),
            "close":    float(k[4]),
            "volume":   float(k[5]),
            "buy_vol":  float(k[9]),
        })
    # trim buffer to max size
    if len(_candle_buffer) > _CANDLE_BUF_MAX:
        del _candle_buffer[:-_CANDLE_BUF_MAX]
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
    sep = _clr("═" * 62, C.BLUE, C.BOLD)
    print(f"\n {sep}")
    print(f"  {_clr('LIVE TRADING', C.BOLD, C.BG_BLUE, C.WHITE)}  "
          f"{_clr('REAL MONEY · BINANCE FUTURES', C.YELLOW, C.BOLD)}")
    print(f"  {_clr('Symbol  ', C.GRAY)} {_clr(symbol, C.WHITE, C.BOLD)}"
          f"   {_clr('Leverage', C.GRAY)} {_clr(str(portfolio.leverage)+'x', C.YELLOW, C.BOLD)}"
          f"   {_clr('Balance', C.GRAY)} {_clr(f'${portfolio.capital:,.2f}', C.CYAN, C.BOLD)}")
    print(f"  {_clr('T_up/dn ', C.GRAY)} {tc['T_up']} / {tc['T_down']}"
          f"   {_clr('SL/TP', C.GRAY)} {tc.get('sl_pct',0)*100:.2f}% / {tc.get('tp_pct',0)*100:.2f}%"
          f"   {_clr('Dashboard', C.GRAY)} {_clr('http://localhost:8080', C.BLUE)}")
    print(f" {sep}")
    print(f"  {_clr('Ctrl+C', C.YELLOW)} to stop  ·  open position will be closed")
    print()

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

            # ── Send candle to browser ────────────────────────────────────────
            c_msg = {
                "type":    "candle",
                "ts":      int(ts.timestamp() * 1000),
                "open":    float(candle["open"]),
                "high":    float(candle["high"]),
                "low":     float(candle["low"]),
                "close":   price,
                "volume":  vol,
                "buy_vol": buy_v,
            }
            _candle_buffer.append(c_msg)
            if len(_candle_buffer) > _CANDLE_BUF_MAX:
                del _candle_buffer[:-_CANDLE_BUF_MAX]
            await _broadcast(c_msg)

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
            unreal     = equity - balance
            # position colour
            if pos is None:
                pos_str = _clr(f"{'FLAT':5s}", C.GRAY)
            elif pos.direction == 1:
                pos_str = _clr(f"{'LONG':5s}", C.GREEN, C.BOLD)
            else:
                pos_str = _clr(f"{'SHORT':5s}", C.RED, C.BOLD)

            # unrealised P&L colour
            if pos is not None:
                unreal_col = C.GREEN if unreal >= 0 else C.RED
                unreal_str = "  " + _clr(f"unreal={unreal:+.2f}", unreal_col)
            else:
                unreal_str = ""

            # probability colours
            pup_col = C.GREEN if p_up   >= engine.T_up   else C.GRAY
            pdn_col = C.RED   if p_down >= engine.T_down else C.GRAY

            # balance / equity
            eq_col = C.GREEN if equity >= balance else C.RED

            ts_str   = _clr(f"{str(ts)[11:16]}", C.GRAY)           # HH:MM only
            price_str= _clr(f"{price:>10,.2f}", C.WHITE, C.BOLD)
            pup_str  = _clr(f"p▲{p_up:.3f}", pup_col)
            pdn_str  = _clr(f"p▼{p_down:.3f}", pdn_col)
            bal_str  = _clr(f"bal ${balance:>9,.2f}", C.CYAN)
            eq_str   = _clr(f"eq ${equity:>9,.2f}", eq_col)
            skip_str = _clr(f"{engine.last_skip_reason}", C.DIM)

            print(f" {ts_str}  {price_str}  {pup_str}  {pdn_str}  "
                  f"{pos_str}  {bal_str}  {eq_str}{unreal_str}"
                  f"  {skip_str}")

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
                    d        = event.get("direction", "")
                    dir_col  = C.GREEN if d == "LONG" else C.RED
                    sep      = _clr("─" * 60, C.BLUE)
                    print(f"\n {sep}")
                    print(f"  {_clr('ENTRY', C.BOLD, C.BG_BLUE, C.WHITE)}  "
                          f"{_clr(d, dir_col, C.BOLD)}  "
                          f"@ {_clr(f'{pos2.entry_price:,.2f}', C.WHITE, C.BOLD)}  "
                          f"{_clr(f'size {pos2.size:.5f} {symbol[:3]} (${size_usd:,.2f})', C.GRAY)}")
                    print(f"  {_clr('SL', C.RED,   C.BOLD)} {pos2.sl_price:>10,.2f}  "
                          f"{_clr(f'-{abs(pos2.entry_price-pos2.sl_price):.2f} pts', C.RED)}")
                    print(f"  {_clr('TP', C.GREEN, C.BOLD)} {pos2.tp_price:>10,.2f}  "
                          f"{_clr(f'+{abs(pos2.tp_price-pos2.entry_price):.2f} pts', C.GREEN)}")
                    print(f" {sep}\n")

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
                    pnl    = event.get("net_pnl", 0)
                    reason = event.get("reason", "")
                    exit_p = event.get("exit_price", price)
                    nt     = len(portfolio.trade_log)
                    wins2  = sum(1 for t in portfolio.trade_log if t.net_pnl > 0)
                    wr     = f"{wins2/nt*100:.0f}%" if nt else "n/a"
                    pnl_col   = C.GREEN if pnl >= 0 else C.RED
                    pnl_str   = _clr(f"{'+'if pnl>=0 else ''}{pnl:,.2f}", pnl_col, C.BOLD)
                    rsn_col   = C.GREEN if reason == "TP" else (C.RED if reason == "SL" else C.YELLOW)
                    sep       = _clr("─" * 60, C.BLUE)
                    print(f"\n {sep}")
                    print(f"  {_clr('EXIT', C.BOLD, C.BG_RED if pnl<0 else C.BG_GREEN, C.WHITE)}  "
                          f"{_clr(reason, rsn_col, C.BOLD)}  "
                          f"@ {_clr(f'{exit_p:,.2f}', C.WHITE, C.BOLD)}  "
                          f"PnL {pnl_str}")
                    print(f"  {_clr('Balance', C.GRAY)} {_clr(f'${portfolio.capital:,.2f}', C.CYAN, C.BOLD)}  "
                          f"{_clr(f'trades {nt}', C.GRAY)}  "
                          f"{_clr(f'WR {wr}', C.GREEN if nt and wins2/nt>=0.6 else C.YELLOW)}")
                    print(f" {sep}\n")

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
        await feed.stop()

        print(f"\n {_clr('─'*62, C.YELLOW)}")
        if portfolio.position is not None:
            d = 'LONG' if portfolio.position.direction == 1 else 'SHORT'
            print(f"  {_clr('SHUTDOWN', C.BOLD, C.BG_YELLOW)} "
                  f"closing {_clr(d, C.BOLD)}  "
                  f"entry={portfolio.position.entry_price:.2f}  "
                  f"last={last_price:.2f}")
            if last_price <= 0:
                print(f"  {_clr('WARNING', C.YELLOW)} last_price=0, fetching from Binance ...")
                try:
                    ticker = portfolio._client.futures_symbol_ticker(symbol=symbol)
                    last_price = float(ticker["price"])
                    print(f"  {_clr('Price fetched', C.GREEN)} {last_price:.2f}")
                except Exception as fe:
                    print(f"  {_clr('Price fetch FAILED', C.RED)} {fe}")
            try:
                engine.force_close(last_price, last_ts or "shutdown")
                print(f"  {_clr('Position closed', C.GREEN, C.BOLD)}")
            except Exception as ce:
                print(f"  {_clr('force_close ERROR', C.RED)} {ce}")
        else:
            print(f"  {_clr('No open position on exit', C.GRAY)}")

        engine.close_log()

        summary = portfolio.summary()
        sep2 = _clr("═" * 50, C.BLUE, C.BOLD)
        print(f"\n {sep2}")
        print(f"  {_clr('SESSION SUMMARY', C.BOLD, C.WHITE)}")
        print(f" {_clr('─'*50, C.BLUE)}")
        for k, v in summary.items():
            val_col = C.GREEN if isinstance(v, (int,float)) and v > 0 else (C.RED if isinstance(v,(int,float)) and v < 0 else C.WHITE)
            print(f"  {_clr(k+' ', C.GRAY)}{_clr(str(v), val_col, C.BOLD)}")
        print(f" {sep2}")

        art = Path(art_dir)
        portfolio.save(str(art / "live_trade_portfolio.json"))


# ── Orchestrator ───────────────────────────────────────────────────────────────
async def _main(cfg: dict, symbol: str, http_port: int, ws_port: int) -> None:
    static_dir = str(Path(__file__).resolve().parent / "sim" / "static")
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
    # BinancePortfolio makes blocking requests calls on the asyncio thread which
    # corrupts Windows ProactorEventLoop and silently breaks the WS server socket.
    # SelectorEventLoop handles mixed sync/async I/O correctly on Windows.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
