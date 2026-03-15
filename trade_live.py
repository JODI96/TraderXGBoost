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
import datetime
import os
import shutil
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

# ── ANSI enable (Windows Terminal needs explicit flag) ────────────────────────
if sys.platform == "win32":
    import ctypes as _ct
    try:
        _k = _ct.windll.kernel32
        _h = _k.GetStdHandle(-11)
        _m = _ct.c_ulong()
        _k.GetConsoleMode(_h, _ct.byref(_m))
        _k.SetConsoleMode(_h, _m.value | 0x0004)
    except Exception:
        os.system("")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ── Terminal UI ───────────────────────────────────────────────────────────────
def _c(n: int) -> str:
    return f"\033[38;5;{n}m"

_R   = "\033[0m"
_BD  = "\033[1m"
_DM  = "\033[2m"
_BDR = _c(24)    # steel-blue borders
_HDR = _c(51)    # electric-cyan header
_WHT = _c(255)   # white values
_GRN = _c(82)    # neon green
_RED = _c(196)   # bright red
_GLD = _c(220)   # gold
_GRY = _c(244)   # label grey
_DGR = _c(238)   # dim grey
_CYN = _c(87)    # light cyan
_ORG = _c(208)   # orange

_W = 72   # inner width (between ║ chars)


class _S:
    """Styled segment. vw = visual (on-screen) width; defaults to len(text)."""
    def __init__(self, text: str, color: str = "", vw: int = -1):
        self.text  = str(text)
        self.color = color
        self.w     = len(self.text) if vw < 0 else vw
    def __str__(self) -> str:
        return f"{self.color}{self.text}{_R}" if self.color else self.text


def _sp(n: int = 1) -> _S:
    return _S(" " * n)


def _L(*segs: _S) -> str:
    """Build  ║ segments… ║  line, right-padded to _W."""
    used = sum(s.w for s in segs)
    pad  = max(0, _W - used)
    body = "".join(str(s) for s in segs) + " " * pad
    return f"{_BDR}║{_R}{body}{_BDR}║{_R}"


def _mkbar(val: float, width: int, on: int, off: int = 238) -> _S:
    filled = max(0, min(width, round(val * width)))
    ansi   = f"\033[38;5;{on}m{'█'*filled}\033[38;5;{off}m{'░'*(width-filled)}{_R}"
    return _S(ansi, vw=width)


_TOP = f"{_BDR}╔{'═'*_W}╗{_R}"
_DIV = f"{_BDR}╠{'═'*_W}╣{_R}"
_BOT = f"{_BDR}╚{'═'*_W}╝{_R}"
_BLK = f"{_BDR}║{' '*_W}║{_R}"


def _print_dashboard(ts, price: float, prev_price: float,
                     p_up: float, p_down: float,
                     balance: float, equity: float,
                     pos, status: str,
                     n_trades: int, wins: int, net_pnl: float,
                     skip_reason: str, bar_count: int, symbol: str) -> None:
    global _W

    # ── Adapt to terminal size ────────────────────────────────────────────────
    cols, rows = shutil.get_terminal_size((82, 30))
    _W = max(60, cols - 2)
    TOP = f"{_BDR}╔{'═'*_W}╗{_R}"
    DIV = f"{_BDR}╠{'═'*_W}╣{_R}"
    BOT = f"{_BDR}╚{'═'*_W}╝{_R}"
    BLK = f"{_BDR}║{' '*_W}║{_R}"

    out = [TOP]

    # ── Header ────────────────────────────────────────────────────────────────
    sym_text = f"  >> {symbol}  LIVE FUTURES"
    ts_text  = ts.strftime(" %Y-%m-%d  %H:%M:%S UTC  ")
    gap      = max(0, _W - len(sym_text) - len(ts_text))
    out.append(_L(_S(sym_text, _HDR + _BD), _sp(gap), _S(ts_text, _GRY)))
    out.append(DIV)
    out.append(BLK)

    # ── Price + Bar + Position status ─────────────────────────────────────────
    diff        = price - (prev_price if prev_price > 0 else price)
    price_text  = f"${price:>12,.2f}"
    if diff > 0:
        arr_text, arr_col = f"▲  +{diff:,.2f}", _GRN
    elif diff < 0:
        arr_text, arr_col = f"▼  {diff:,.2f}",  _RED
    else:
        arr_text, arr_col = "─",                 _GRY
    arr_text = f"{arr_text:<12}"

    bar_text = f"BAR #{bar_count:<5}"
    if status == "LONG":
        pos_text, pos_col = "◆  LONG   ", _GRN + _BD
    elif status == "SHORT":
        pos_text, pos_col = "◆  SHORT  ", _RED + _BD
    elif status == "PENDING_LONG":
        pos_text, pos_col = "◈  PEND L ", _GLD + _BD
    elif status == "PENDING_SHORT":
        pos_text, pos_col = "◈  PEND S ", _GLD + _BD
    else:
        pos_text, pos_col = "─  FLAT   ", _GRY

    left_w  = 4 + 13 + 2 + 12
    right_w = 10 + 3 + 10 + 2
    gap     = max(0, _W - left_w - right_w)
    out.append(_L(
        _sp(4), _S(price_text, _WHT + _BD), _sp(2), _S(arr_text, arr_col),
        _sp(gap),
        _S(bar_text, _DGR), _sp(3), _S(pos_text, pos_col), _sp(2),
    ))
    out.append(BLK)
    out.append(DIV)

    # ── Probability bars ──────────────────────────────────────────────────────
    # Fixed prefix: 4+5+3+6+3=21  Fixed suffix: 2+"NEXT "+4=11  BAR_W fills rest
    BAR_W     = max(16, _W - 21 - 11)
    up_pct    = f"{p_up   * 100:5.1f}%"
    dn_pct    = f"{p_down * 100:5.1f}%"
    _now      = datetime.datetime.utcnow()
    secs_left = 60 - _now.second
    timer_s   = f"0:{secs_left:02d}"
    out.append(_L(_sp(4), _S("P(UP)", _GRY), _sp(3), _S(up_pct, _GRN + _BD),
                  _sp(3), _mkbar(p_up,   BAR_W, 82),
                  _sp(2), _S("NEXT ", _GRY), _S(timer_s, _CYN + _BD)))
    out.append(_L(_sp(4), _S("P(DN)", _GRY), _sp(3), _S(dn_pct, _RED + _BD),
                  _sp(3), _mkbar(p_down, BAR_W, 196)))
    out.append(DIV)

    # ── Open position details ─────────────────────────────────────────────────
    if pos is not None:
        unreal  = equity - balance
        u_sign  = "+" if unreal >= 0 else ""
        u_col   = _GRN if unreal >= 0 else _RED
        entry_s = f"${pos.entry_price:,.2f}"
        sl_s    = f"${pos.sl_price:,.2f}"
        tp_s    = f"${pos.tp_price:,.2f}"
        pnl_s   = f"{u_sign}${abs(unreal):,.2f}"
        out.append(_L(
            _sp(4),
            _S("ENTRY ", _GRY), _S(entry_s, _WHT + _BD), _sp(3),
            _S("SL ",    _GRY), _S(sl_s,    _RED + _BD), _sp(3),
            _S("TP ",    _GRY), _S(tp_s,    _GRN + _BD), _sp(3),
            _S("PNL ",   _GRY), _S(pnl_s,   u_col + _BD), _sp(2),
        ))
        out.append(DIV)

    # ── Balance / Equity ──────────────────────────────────────────────────────
    unreal = equity - balance
    u_sign = "+" if unreal >= 0 else ""
    u_col  = _GRN if unreal >= 0 else _RED
    bal_s  = f"${balance:>10,.2f}"
    eq_s   = f"${equity:>10,.2f}"
    unr_s  = f"{u_sign}${abs(unreal):>8,.2f}"
    out.append(_L(
        _sp(4),
        _S("BALANCE ", _GRY), _S(bal_s, _WHT + _BD), _sp(3),
        _S("EQUITY  ", _GRY), _S(eq_s,  _WHT + _BD), _sp(3),
        _S("UNREAL  ", _GRY), _S(unr_s, u_col + _BD), _sp(2),
    ))
    out.append(DIV)

    # ── Session stats ─────────────────────────────────────────────────────────
    wr_s  = f"{wins / n_trades * 100:5.1f}%" if n_trades else "   ─  "
    pnl_s = f"{'+' if net_pnl >= 0 else ''}${net_pnl:,.2f}"
    p_col = _GRN if net_pnl >= 0 else _RED
    out.append(_L(
        _sp(4),
        _S(f"{n_trades} TRADES", _WHT + _BD), _sp(4),
        _S("WIN RATE ", _GRY), _S(wr_s,  _GLD + _BD), _sp(4),
        _S("NET PNL  ", _GRY), _S(pnl_s, p_col + _BD), _sp(2),
    ))
    out.append(DIV)

    # ── Signal status ─────────────────────────────────────────────────────────
    sk = skip_reason[:max(0, _W - 9)]
    out.append(_L(_sp(4), _S("► ", _GLD + _BD), _sp(1), _S(sk, _GRY)))

    # ── Event log – fills remaining terminal height ───────────────────────────
    used      = len(out) + 2           # +DIV before log section +BOT
    n_log     = max(0, rows - used - 1)
    logs      = list(_portfolio_log)[-n_log:] if n_log > 0 else []
    if logs:
        out.append(DIV)
        for entry in logs:
            if "[bg_fill]" in entry:
                col = _GRN
            elif "FAILED" in entry or "CRITICAL" in entry or "ERROR" in entry:
                col = _RED
            elif "[reconcile]" in entry:
                col = _ORG
            elif "[guard]" in entry or "[bg_guard]" in entry:
                col = _CYN
            elif "[PENDING]" in entry:
                col = _GLD
            elif "[close]" in entry or "[exit]" in entry or "[pos]" in entry:
                col = _GRY
            else:
                col = _DGR
            text = entry[:_W - 4]
            out.append(_L(_sp(2), _S(text, col)))

    # ── Fill remaining rows with blank lines then close box ───────────────────
    used  = len(out) + 1               # +1 for BOT
    fill  = max(0, rows - used - 1)
    for _ in range(fill):
        out.append(BLK)
    out.append(BOT)

    # ── Render: jump to top-left, clear to end, write atomically ─────────────
    sys.stdout.write("\033[H\033[J" + "\n".join(out) + "\n")
    sys.stdout.flush()


def _print_trade_open(pos, direction: str, p_up: float, p_down: float,
                      symbol: str) -> None:
    size_usd = pos.size * pos.entry_price
    sl_pts   = abs(pos.entry_price - pos.sl_price)
    tp_pts   = abs(pos.tp_price    - pos.entry_price)
    col      = _GRN if direction == "LONG" else _RED
    W2       = 46
    top      = f"{col}╔{'═'*W2}╗{_R}"
    div      = f"{col}╠{'═'*W2}╣{_R}"
    bot      = f"{col}╚{'═'*W2}╝{_R}"
    def row(label, value, vcol=""):
        lbl = f"{_GRY}{label:<8}{_R}"
        val = f"{vcol}{value}{_R}" if vcol else value
        inner = f"  {lbl}  {val}"
        pad = W2 - 2 - 8 - 2 - len(value)
        return f"{col}║{_R}{inner}{' '*max(0,pad)}{col}║{_R}"
    title = f"  {'▲' if direction=='LONG' else '▼'}  POSITION OPENED  {direction:<5}  "
    title_pad = W2 - len(title.replace(_BD,'').replace(col,'').replace(_R,''))
    print(f"\n{top}")
    print(f"{col}║{_R}{_BD}{col}{title}{_R}{' '*max(0, W2-len(title))}{col}║{_R}")
    print(div)
    print(row("ENTRY",  f"${pos.entry_price:,.2f}",  _WHT + _BD))
    print(row("SL",     f"${pos.sl_price:,.2f}  ({sl_pts:.2f} pts)", _RED))
    print(row("TP",     f"${pos.tp_price:,.2f}  ({tp_pts:.2f} pts)", _GRN))
    print(row("SIZE",   f"{pos.size:.6f} {symbol[:3]}  (${size_usd:,.2f})", _GRY))
    print(f"{bot}\n")


def _print_trade_close(reason: str, exit_p: float, pnl: float,
                       balance: float, n: int, wr: float) -> None:
    col     = _GRN if pnl >= 0 else _RED
    sign    = "+" if pnl >= 0 else ""
    sym     = {"TP": "✓ TAKE PROFIT", "SL": "✗ STOP LOSS",
               "TIME": "◷ TIME STOP", "FORCE": "■ FORCE CLOSE",
               "GUARD_FAIL": "! GUARD CLOSE", "LIMIT_CLOSE": "~ LIMIT CLOSE"
               }.get(reason, reason)
    W2 = 46
    top = f"{col}╔{'═'*W2}╗{_R}"
    bot = f"{col}╚{'═'*W2}╝{_R}"
    def row(label, value):
        inner = f"  {_GRY}{label:<12}{_R}  {value}"
        plain_len = 2 + 12 + 2 + len(value.replace(_BD,'').replace(_GRN,'').replace(_RED,'').replace(_WHT,'').replace(_GLD,'').replace(_R,''))
        pad = W2 - plain_len
        return f"{col}║{_R}{inner}{' '*max(0,pad)}{col}║{_R}"
    print(f"\n{top}")
    title = f"  {sym}  "
    print(f"{col}║{_R}{_BD}{col}{title:<{W2}}{_R}{col}║{_R}")
    print(f"{col}╠{'═'*W2}╣{_R}")
    print(row("EXIT PRICE",  f"{_WHT}{_BD}${exit_p:,.2f}{_R}"))
    print(row("PNL",         f"{col}{_BD}{sign}${abs(pnl):,.2f}{_R}"))
    print(row("BALANCE",     f"{_WHT}${balance:,.2f}{_R}"))
    print(row("TRADES",      f"{_WHT}{n}  {_GRY}WR {wr:.0f}%{_R}"))
    print(f"{bot}\n")

import features as feat_mod
from sim.execution          import ExecutionEngine
from sim.binance_portfolio  import BinancePortfolio, event_log as _portfolio_log
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
    cols, _ = shutil.get_terminal_size((82, 30))
    W2 = cols
    sys.stdout.write("\033[2J\033[H")   # clear full screen before banner
    sys.stdout.flush()
    print(f"{_BDR}{'═'*W2}{_R}")
    print(f"  {_HDR}{_BD}*** LIVE TRADING  –  REAL MONEY  –  BINANCE FUTURES ***{_R}")
    print(f"{_BDR}{'─'*W2}{_R}")
    print(f"  {_GRY}Symbol   {_R}  {_WHT}{_BD}{symbol}{_R}   "
          f"{_GRY}Leverage{_R}  {_WHT}{_BD}{portfolio.leverage}x{_R}   "
          f"{_GRY}Balance{_R}  {_GLD}{_BD}${portfolio.capital:,.2f}{_R}")
    print(f"  {_GRY}SL/TP    {_R}  {_RED}{tc.get('sl_pct',0)*100:.2f}%{_R} / "
          f"{_GRN}{tc.get('tp_pct',0)*100:.2f}%{_R}   "
          f"{_GRY}T up/dn  {_R}  {_WHT}{tc['T_up']}{_R} / {_WHT}{tc['T_down']}{_R}   "
          f"{_GRY}Dashboard{_R}  {_CYN}http://localhost:8080{_R}")
    print(f"{_BDR}{'═'*W2}{_R}\n")

    last_price = 0.0
    prev_price = 0.0
    last_ts    = None
    cvd_col    = f"cvd_{cfg['features']['cvd_window']}"

    # ── Shared dashboard state (updated each bar, read by background task) ──────
    _dash: dict = {
        "ts": None, "price": 0.0, "prev_price": 0.0,
        "p_up": 0.0, "p_down": 0.0,
        "skip": "", "bar": 0,
    }

    def _redraw():
        if _dash["ts"] is None:
            return
        pos     = portfolio.position
        balance = portfolio.capital
        equity  = portfolio.mark_to_market(_dash["price"])
        if pos is not None:
            status = "LONG" if pos.direction == 1 else "SHORT"
            skip   = "in_pos"
        elif portfolio.has_pending:
            status = "PENDING_LONG" if portfolio.pending_dir == 1 else "PENDING_SHORT"
            skip   = (f"pending_{'long' if portfolio.pending_dir == 1 else 'short'}"
                      f"  limit={portfolio.pending_price:.2f}")
        else:
            status = "FLAT"
            skip   = _dash["skip"]
        nt      = len(portfolio.trade_log)
        wins    = sum(1 for t in portfolio.trade_log if t.net_pnl > 0)
        net_pnl = sum(t.net_pnl for t in portfolio.trade_log)
        _print_dashboard(
            _dash["ts"], _dash["price"], _dash["prev_price"],
            _dash["p_up"], _dash["p_down"],
            balance, equity, pos, status,
            nt, wins, net_pnl,
            skip, _dash["bar"], symbol,
        )

    # ── Background SL/TP guard + dashboard refresh (every 5 s) ───────────────
    _price_ref = [0.0]

    async def _background_guard():
        while True:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                return
            if _price_ref[0] > 0:
                try:
                    portfolio.handle_pending_fill_immediate(
                        sl_pct=tc.get("sl_pct", 0.002),
                        tp_pct=tc.get("tp_pct", 0.006),
                    )
                    portfolio.verify_protection(_price_ref[0])
                except (Exception, KeyboardInterrupt) as exc:
                    logger.warning(f"[bg_guard] error: {exc}")
                _redraw()

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

            # ── Update shared state + redraw dashboard ────────────────────────
            _dash["ts"]         = ts
            _dash["prev_price"] = _dash["price"]
            _dash["price"]      = price
            _dash["p_up"]       = float(p_up)
            _dash["p_down"]     = float(p_down)
            _dash["skip"]       = engine.last_skip_reason
            _dash["bar"]        = engine.bar_count
            prev_price          = _dash["prev_price"]
            _redraw()

            # ── Browser stats ─────────────────────────────────────────────────
            pos     = portfolio.position
            balance = portfolio.capital
            equity  = portfolio.mark_to_market(price)
            status  = "FLAT" if pos is None else ("LONG" if pos.direction == 1 else "SHORT")
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
                    pos2 = portfolio.position
                    _redraw()   # refresh dashboard immediately to show new position

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
                    _redraw()   # refresh dashboard immediately to show closed position

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
        except (asyncio.CancelledError, KeyboardInterrupt, Exception):
            pass
        await feed.stop()

        if portfolio.has_pending:
            oid = portfolio._pending_entry_order_id
            print(f"\n{_GLD}  Cancelling pending limit entry order (id={oid}) ...{_R}")
            import time as _t
            for _att in range(1, 13):   # retry every 5 s for up to 1 min
                try:
                    if oid:
                        portfolio._client.futures_cancel_order(
                            symbol=portfolio.symbol, orderId=oid)
                    portfolio.pending_dir             = 0
                    portfolio.pending_price           = 0.0
                    portfolio.pending_bar             = 0
                    portfolio._pending_entry_order_id = None
                    portfolio._pending_qty            = 0.0
                    portfolio._limit_entry_filled     = False
                    print(f"  {_GRN}Limit order cancelled.{_R}")
                    break
                except Exception as _ce:
                    print(f"  {_RED}Cancel attempt {_att}/12 FAILED: {_ce} – retry in 5 s{_R}")
                    _t.sleep(5)

        if portfolio.position is not None:
            pos = portfolio.position
            dir_str = "LONG" if pos.direction == 1 else "SHORT"
            print(f"\n  Ctrl+C – closing {dir_str} position "
                  f"entry={pos.entry_price:.2f}  size={pos.size}")

            # Step 1: cancel all SL/TP orders so they don't race the close
            portfolio._cancel_all_orders()

            # Step 2: market close – retry every 5 s for up to 10 min
            side_close = "SELL" if pos.direction == 1 else "BUY"
            closed = False
            max_attempts = 120   # 120 × 5 s = 10 min
            for att in range(1, max_attempts + 1):
                try:
                    resp = portfolio._client.futures_create_order(
                        symbol    = portfolio.symbol,
                        side      = side_close,
                        type      = "MARKET",
                        quantity  = portfolio._rq(pos.size),
                        reduceOnly= "true",
                    )
                    fill = float(resp.get("avgPrice") or 0)
                    print(f"  Closed @ {fill:.2f}  (attempt {att})")
                    closed = True
                    break
                except Exception as ce:
                    # -2022 = reduceOnly rejected (position already gone – SL/TP fired)
                    if "-2022" in str(ce) or "reduceOnly" in str(ce).lower():
                        print(f"  Position already closed on exchange (SL/TP fired).")
                        closed = True
                        break
                    print(f"  Close attempt {att}/{max_attempts} FAILED: {ce}  – retrying in 5 s ...")
                    await asyncio.sleep(5)

            if closed:
                portfolio.position = None
                print("  Shutdown complete.")
            else:
                print("  WARNING: Could not close position after 10 min of retries!")
                print("  Check Binance manually – SL/TP orders were cancelled.")
        else:
            print("\n  No open position on exit.")

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
