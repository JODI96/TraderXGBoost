"""
render_sim.py – Render-compatible replay server.

Combines HTTP (static files) + WebSocket on a SINGLE port.
Precomputes the full backtest, then streams bar-by-bar to the browser.

Usage:
    python render_sim.py --data Data/BTCUSDT/monthly/2026-01_1m.csv --speed 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from aiohttp import web

import sys
sys.path.insert(0, str(Path(__file__).parent))

import data as data_mod
import features as feat_mod
from backtest import run_backtest

# ── Globals ───────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
STATIC_DIR = BASE / "sim" / "static"
SIM_DATA: dict = {}
SPEED: float = 30.0


# ── Static files with patched WebSocket URL ───────────────────────────────────

def _patch_html(html: str) -> str:
    """Replace hardcoded ws://localhost:8765 with dynamic host-relative URL."""
    return re.sub(
        r"const WS_URL\s*=\s*['\"]ws://localhost:\d+['\"];",
        "const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;",
        html,
    )


async def handle_index(request: web.Request) -> web.Response:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return web.Response(text=_patch_html(html), content_type="text/html")


async def handle_static(request: web.Request) -> web.Response:
    filename = request.match_info["filename"]
    filepath = STATIC_DIR / filename
    if filepath.exists():
        return web.FileResponse(filepath)
    return web.Response(status=404)


# ── WebSocket streaming ───────────────────────────────────────────────────────

async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    await _stream(ws)
    return ws


async def _stream(ws: web.WebSocketResponse) -> None:
    data       = SIM_DATA
    df         = data["df"]
    trades_df  = data["trades_df"]
    equity     = data["equity_curve"]
    probs      = data["probs"]
    cfg        = data["cfg"]
    report     = data["report"]
    n_bars     = len(df)
    delay      = 1.0 / SPEED if SPEED > 0 else 0.0

    # ── Index trades by bar timestamp ────────────────────────────────────────
    col = lambda df, *names: next((c for c in names if c in df.columns), None)

    entry_col  = col(trades_df, "entry_time", "open_time")
    exit_col   = col(trades_df, "exit_time",  "close_time")
    dir_col    = col(trades_df, "side",        "direction")
    eprice_col = col(trades_df, "entry_price", "entry")
    pnl_col    = col(trades_df, "pnl_pct",     "pnl", "return")
    sl_col     = col(trades_df, "sl")
    tp_col     = col(trades_df, "tp")
    reason_col = col(trades_df, "reason",      "exit_reason")

    by_entry: dict = {}
    by_exit:  dict = {}
    if not trades_df.empty:
        if entry_col:
            for _, t in trades_df.iterrows():
                by_entry[pd.Timestamp(t[entry_col])] = t
        if exit_col:
            for _, t in trades_df.iterrows():
                by_exit[pd.Timestamp(t[exit_col])] = t

    init_cap  = float(cfg.get("backtest", {}).get("initial_capital", 10.0))
    balance   = init_cap
    n_trades  = 0
    n_wins    = 0
    open_t    = None

    n_classes = int(cfg.get("model", {}).get("num_class", 5))

    await ws.send_str(json.dumps({
        "type": "replay_info",
        "ts":   int(df.index[0].timestamp() * 1000),
        "pct":  0,
    }))

    for i, (ts, row) in enumerate(df.iterrows()):
        if ws.closed:
            break

        ts_ms = int(ts.timestamp() * 1000)
        p_row = probs[i] if i < len(probs) else np.zeros(n_classes)
        if p_row.ndim == 0:
            p_up = p_down = 0.0
        else:
            p_up   = float(p_row[2]) if len(p_row) > 2 else 0.0
            p_down = float(p_row[0]) if len(p_row) > 0 else 0.0

        # trade open
        if ts in by_entry:
            t = by_entry[ts]
            open_t = t
            await ws.send_str(json.dumps({
                "type":      "trade_open",
                "ts":        ts_ms,
                "direction": str(t[dir_col]) if dir_col else "LONG",
                "p_up":      round(p_up,   4),
                "p_down":    round(p_down, 4),
                "price":     float(row["close"]),
                "sl":        float(t[sl_col]) if sl_col and sl_col in t else round(float(row["close"]) * 0.9985, 2),
                "tp":        float(t[tp_col]) if tp_col and tp_col in t else round(float(row["close"]) * 1.0045, 2),
            }))

        # trade close
        if ts in by_exit:
            t = by_exit[ts]
            pnl = float(t[pnl_col]) if pnl_col and pnl_col in t else 0.0
            balance *= (1.0 + pnl)
            n_trades += 1
            if pnl > 0:
                n_wins += 1
            open_t = None
            await ws.send_str(json.dumps({
                "type":        "trade_close",
                "ts":          ts_ms,
                "net_pnl":     round(pnl * balance, 6),
                "direction":   str(t[dir_col])    if dir_col    and dir_col    in t else "LONG",
                "entry_price": float(t[eprice_col]) if eprice_col and eprice_col in t else float(row["open"]),
                "exit_price":  float(row["close"]),
                "reason":      str(t[reason_col]) if reason_col and reason_col in t else "TP/SL",
            }))

        # candle
        await ws.send_str(json.dumps({
            "type":    "candle",
            "ts":      ts_ms,
            "open":    float(row["open"]),
            "high":    float(row["high"]),
            "low":     float(row["low"]),
            "close":   float(row["close"]),
            "volume":  float(row["volume"]),
            "buy_vol": float(row["taker_buy_vol"]) if "taker_buy_vol" in row else float(row["volume"]) * 0.5,
        }))

        # stats
        eq_val   = float(equity[i]) if i < len(equity) else balance
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        await ws.send_str(json.dumps({
            "type":     "stats",
            "ts":       ts_ms,
            "balance":  round(balance, 4),
            "vwap":     float(row["vwap"])     if "vwap"     in row else float(row["close"]),
            "rel_vol":  float(row["rel_vol"])  if "rel_vol"  in row else 1.0,
            "buy_ratio":float(row["buy_ratio"])if "buy_ratio"in row else 0.5,
            "equity":   round(eq_val, 4),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 3),
            "position": str(open_t[dir_col]) if open_t is not None and dir_col else None,
            "unreal":   0.0,
            "entry":    float(open_t[eprice_col]) if open_t is not None and eprice_col and eprice_col in open_t else None,
            "sl":       float(open_t[sl_col])     if open_t is not None and sl_col     and sl_col     in open_t else None,
            "tp":       float(open_t[tp_col])     if open_t is not None and tp_col     and tp_col     in open_t else None,
            "p_up":     round(p_up,   4),
            "p_down":   round(p_down, 4),
        }))

        # progress every 100 bars
        if i % 100 == 0:
            await ws.send_str(json.dumps({
                "type": "replay_info",
                "ts":   ts_ms,
                "pct":  round(i / n_bars * 100, 1),
            }))

        if delay > 0:
            await asyncio.sleep(delay)

    # done
    final_equity = float(equity[-1]) if len(equity) > 0 else balance
    total_ret    = (final_equity / init_cap - 1.0) if init_cap else 0.0

    await ws.send_str(json.dumps({
        "type": "replay_done",
        "summary": {
            "n_trades":          len(trades_df),
            "win_rate_pct":      round(n_wins / max(n_trades, 1) * 100, 1),
            "net_pnl":           round(final_equity - init_cap, 4),
            "total_return_pct":  round(total_ret * 100, 2),
            "max_drawdown_pct":  round(float(report.get("max_drawdown", report.get("max_dd", 0))) * 100, 2),
            "profit_factor":     round(float(report.get("profit_factor", 0)), 2),
            "final_capital":     round(final_equity, 4),
        },
    }))


# ── Precompute ────────────────────────────────────────────────────────────────

def precompute(data_file: str, cfg: dict) -> dict:
    print("[render_sim] Loading data …")
    df = data_mod.load_csv(data_file)

    print("[render_sim] Computing features …")
    df_feat = feat_mod.compute_features(df, cfg)
    df_feat = df_feat.dropna().copy()

    print("[render_sim] Running XGBoost inference …")
    model = xgb.Booster()
    model.load_model(str(BASE / "models" / "xgb_model.json"))

    feat_cols = feat_mod.get_feature_columns(cfg)
    feat_cols = [c for c in feat_cols if c in df_feat.columns]
    dmatrix   = xgb.DMatrix(df_feat[feat_cols].values, feature_names=feat_cols)
    probs     = model.predict(dmatrix)

    n_classes = int(cfg.get("model", {}).get("num_class", 5))
    if probs.ndim == 1:
        probs = probs.reshape(-1, n_classes)

    print("[render_sim] Running backtest …")
    trades_df, equity_curve, report = run_backtest(df_feat, probs, cfg)
    print(f"[render_sim] Done — {len(trades_df)} trades found.")

    return {
        "df":           df_feat,
        "trades_df":    trades_df,
        "equity_curve": equity_curve,
        "probs":        probs,
        "cfg":          cfg,
        "report":       report,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True, help="Path to CSV data file")
    parser.add_argument("--speed", type=float, default=30.0, help="Bars per second")
    parser.add_argument("--port",  type=int,   default=int(os.environ.get("PORT", 8080)))
    args = parser.parse_args()

    global SPEED
    SPEED = args.speed

    with open(BASE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    SIM_DATA.update(precompute(args.data, cfg))

    app = web.Application()
    app.router.add_get("/",           handle_index)
    app.router.add_get("/ws",         websocket_handler)
    app.router.add_get("/{filename}", handle_static)

    print(f"[render_sim] Server ready → http://0.0.0.0:{args.port}")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
