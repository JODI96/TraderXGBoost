"""
sim/sim_live_binance.py – Live paper-trading simulation via Binance WebSocket.

Connects to Binance public WebSocket (no API key needed),
receives closed 1m candles, computes features, predicts, and trades.

Usage
-----
    python sim/sim_live_binance.py
    python sim/sim_live_binance.py --symbol ETHUSDT
    python sim/sim_live_binance.py --config config.yaml

Ctrl+C to stop; final summary and portfolio log are written on exit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

import urllib.request as _urllib_req

import features as feat_mod
from sim.portfolio       import Portfolio
from sim.execution       import ExecutionEngine
from sim.binance_ws_feed import BinanceWSFeed


# ─────────────────────────────────────────────────────────────────────────────
def _load_artifacts(art_dir: str):
    from pathlib import Path
    p = Path(art_dir)
    model = xgb.Booster()
    model.load_model(str(p / "xgb_model.json"))
    with open(p / "feature_columns.json") as f:
        feat_cols = json.load(f)
    return model, feat_cols


def _prewarm_sync(feat_engine, symbol: str, buf_size: int) -> int:
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

    # Drop the last entry – it may be the still-open current candle
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


async def _prewarm_from_binance(feat_engine, symbol: str, buf_size: int) -> int:
    """Async wrapper: runs the REST fetch in a thread so the event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _prewarm_sync, feat_engine, symbol, buf_size)


def _predict(model: xgb.Booster, feat_cols: list[str],
             feat_row: pd.Series) -> np.ndarray:
    arr = feat_row.values.reshape(1, -1).astype(np.float32)
    dm  = xgb.DMatrix(arr, feature_names=feat_cols)
    return model.predict(dm)[0]   # shape (3,)


# ─────────────────────────────────────────────────────────────────────────────
async def run_live(cfg: dict, symbol: str) -> None:
    sim_cfg  = cfg["simulation"]["live"]
    log_file = sim_cfg.get("log_file", "sim/logs/live_log.jsonl")
    art_dir  = cfg["training"]["artifacts_dir"]
    tc       = cfg["trading"]
    lc       = cfg["labels"]

    # ── Artifacts ─────────────────────────────────────────────────────────────
    print("Loading model artifacts …")
    model, feat_cols = _load_artifacts(art_dir)

    # ── Portfolio + engine ────────────────────────────────────────────────────
    cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
               lc["slippage"]  + lc["spread"]) * 2
    portfolio = Portfolio(
        initial_capital = tc["initial_capital"],
        cost_rt         = cost_rt,
        cooldown_bars   = tc["cooldown"],
    )
    engine      = ExecutionEngine(cfg, portfolio, log_file)
    feat_engine = feat_mod.FeatureEngine(cfg, feature_cols=feat_cols)

    # ── Pre-warm from Binance REST API ───────────────────────────────────────
    await _prewarm_from_binance(feat_engine, symbol, cfg["features"]["live_buffer"])

    # ── WebSocket feed ────────────────────────────────────────────────────────
    ws_url = sim_cfg.get("ws_url", "wss://stream.binance.com:9443/ws")
    feed   = BinanceWSFeed(symbol=symbol, ws_url=ws_url)

    print(f"\n{'='*60}")
    print(f" Live Simulation  |  {symbol}  |  1m bars")
    print(f" Log: {log_file}")
    print(f" Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    last_candle = None
    last_ts     = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        async for candle in feed:
            last_ts     = candle.name
            last_candle = candle
            price       = float(candle["close"])

            feat_row = feat_engine.update(candle)
            if feat_row is None:
                remaining = feat_engine.min_warmup - len(feat_engine.buffer)
                print(f"  [{last_ts}] Warming up... ({remaining} bars remaining)")
                continue

            probs  = _predict(model, feat_cols, feat_row)
            event  = engine.on_bar(feat_row, probs, last_ts, price)

            # ── Console output ────────────────────────────────────────────────
            p_down, _, p_up = probs
            equity   = portfolio.mark_to_market(price)
            balance  = portfolio.capital          # realised cash
            pos      = portfolio.position
            status   = "FLAT" if pos is None else \
                       ("LONG" if pos.direction == 1 else "SHORT")

            # Per-bar status line (always printed)
            unreal = equity - balance
            unreal_str = f"  unreal={unreal:+.2f}" if pos is not None else ""
            print(f"  [{last_ts}]  {price:>10.2f}  "
                  f"p_up={p_up:.3f}  p_dn={p_down:.3f}  "
                  f"pos={status:5s}  "
                  f"bal=${balance:>10,.2f}  eq=${equity:>10,.2f}{unreal_str}")

            # Trade event lines
            if event:
                ev_type = event.get("event", "")
                if ev_type == "OPEN" and pos is not None:
                    size_usd = pos.size * pos.entry_price
                    print(f"\n  {'='*60}")
                    print(f"  ENTRY  {event.get('direction'):5s}  "
                          f"@ {pos.entry_price:.2f}")
                    print(f"    SL      : {pos.sl_price:.2f}  "
                          f"({abs(pos.entry_price - pos.sl_price):.2f} pts)")
                    print(f"    TP      : {pos.tp_price:.2f}  "
                          f"({abs(pos.tp_price - pos.entry_price):.2f} pts)")
                    print(f"    Size    : {pos.size:.6f} BTC  (${size_usd:,.2f})")
                    print(f"    Balance : ${balance:,.2f}")
                    print(f"  {'='*60}\n")

                elif ev_type == "CLOSE":
                    pnl      = event.get("net_pnl", 0)
                    reason   = event.get("reason", "")
                    pnl_sign = "+" if pnl >= 0 else ""
                    trades   = portfolio.trade_log
                    n        = len(trades)
                    wins     = sum(1 for t in trades if t.net_pnl > 0)
                    wr       = f"{wins/n*100:.0f}%" if n else "n/a"
                    print(f"\n  {'='*60}")
                    print(f"  EXIT   {reason:6s} @ {event.get('exit_price', price):.2f}  "
                          f"PnL: {pnl_sign}${pnl:.2f}")
                    print(f"    New balance : ${portfolio.capital:,.2f}")
                    print(f"    Total trades: {n}   Win rate: {wr}")
                    print(f"  {'='*60}\n")

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        await feed.stop()
        if last_candle is not None and last_ts is not None:
            engine.force_close(float(last_candle["close"]), last_ts)
        engine.close_log()

        summary = portfolio.summary()
        print(f"\n{'='*50}")
        print(" Live Session Summary")
        print(f"{'='*50}")
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")

        from pathlib import Path
        art = Path(art_dir)
        portfolio.save(str(art / "live_portfolio.json"))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Live Binance paper-trading sim")
    parser.add_argument("--symbol", default=None,
                        help="Trading pair symbol (default: from config)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    symbol = (args.symbol or
              cfg["simulation"]["live"].get("symbol",
              cfg["data"]["symbol"])).upper()

    asyncio.run(run_live(cfg, symbol))


if __name__ == "__main__":
    main()
