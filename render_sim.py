"""
render_sim.py – Render-compatible replay server.

Combines HTTP + WebSocket on a SINGLE port.
Server starts immediately (port detection), precomputes in background.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# ── Block heavy optional imports early ───────────────────────────────────────
import unittest.mock as _mock
sys.modules.setdefault("matplotlib",      _mock.MagicMock())
sys.modules.setdefault("matplotlib.pyplot", _mock.MagicMock())
sys.modules.setdefault("seaborn",         _mock.MagicMock())

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))
import data as data_mod
import features as feat_mod
from backtest import run_backtest

# ── Globals ───────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
STATIC_DIR = BASE / "sim" / "static"
SIM_DATA:  dict  = {}
SIM_READY: bool  = False
SPEED:     float = 30.0
MAX_BARS:  int   = 5000   # memory guard for free tier


# ── HTML patch ────────────────────────────────────────────────────────────────

def _patch_html(html: str) -> str:
    return re.sub(
        r"const WS_URL\s*=\s*['\"]ws://localhost:\d+['\"];",
        "const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;",
        html,
    )


# ── HTTP handlers ─────────────────────────────────────────────────────────────

async def handle_index(request: web.Request) -> web.Response:
    if not SIM_READY:
        return web.Response(
            text="""<!doctype html><html><head>
            <meta charset="utf-8"><meta http-equiv="refresh" content="5">
            <style>body{background:#0d1117;color:#8b949e;font-family:sans-serif;
            display:flex;align-items:center;justify-content:center;height:100vh;margin:0;flex-direction:column;gap:1rem;}
            .spinner{width:40px;height:40px;border:3px solid #30363d;border-top-color:#58a6ff;
            border-radius:50%;animation:spin .8s linear infinite;}
            @keyframes spin{to{transform:rotate(360deg);}}</style></head>
            <body><div class="spinner"></div><p>Model loading… page refreshes automatically</p></body></html>""",
            content_type="text/html",
        )
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return web.Response(text=_patch_html(html), content_type="text/html")


async def handle_static(request: web.Request) -> web.Response:
    filepath = STATIC_DIR / request.match_info["filename"]
    if filepath.exists():
        return web.FileResponse(filepath)
    return web.Response(status=404)


# ── WebSocket ─────────────────────────────────────────────────────────────────

async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    if not SIM_READY:
        await ws.close()
        return ws
    await _stream(ws)
    return ws


async def _stream(ws: web.WebSocketResponse) -> None:
    data      = SIM_DATA
    df        = data["df"]
    trades_df = data["trades_df"]
    equity    = data["equity_curve"]
    probs     = data["probs"]
    cfg       = data["cfg"]
    report    = data["report"]
    n_bars    = len(df)
    delay     = 1.0 / SPEED if SPEED > 0 else 0.0

    def _col(frame, *names):
        return next((c for c in names if c in frame.columns), None)

    entry_col  = _col(trades_df, "entry_time",  "open_time")
    exit_col   = _col(trades_df, "exit_time",   "close_time")
    dir_col    = _col(trades_df, "side",         "direction")
    eprice_col = _col(trades_df, "entry_price",  "entry")
    pnl_col    = _col(trades_df, "pnl_pct",      "pnl", "return")
    sl_col     = _col(trades_df, "sl")
    tp_col     = _col(trades_df, "tp")
    reason_col = _col(trades_df, "reason",       "exit_reason")

    by_entry: dict = {}
    by_exit:  dict = {}
    if not trades_df.empty:
        if entry_col:
            for _, t in trades_df.iterrows():
                by_entry[pd.Timestamp(t[entry_col])] = t
        if exit_col:
            for _, t in trades_df.iterrows():
                by_exit[pd.Timestamp(t[exit_col])] = t

    init_cap = float(cfg.get("backtest", {}).get("initial_capital", 10.0))
    balance  = init_cap
    n_trades = n_wins = 0
    open_t   = None
    n_cls    = int(cfg.get("model", {}).get("num_class", 5))

    await ws.send_str(json.dumps({"type": "replay_info", "ts": int(df.index[0].timestamp() * 1000), "pct": 0}))

    for i, (ts, row) in enumerate(df.iterrows()):
        if ws.closed:
            break

        ts_ms = int(ts.timestamp() * 1000)
        p_row = probs[i] if i < len(probs) else np.zeros(n_cls)
        p_up   = float(p_row[2]) if len(p_row) > 2 else 0.0
        p_down = float(p_row[0]) if len(p_row) > 0 else 0.0

        if ts in by_entry:
            open_t = by_entry[ts]
            await ws.send_str(json.dumps({
                "type": "trade_open", "ts": ts_ms,
                "direction": str(open_t[dir_col]) if dir_col else "LONG",
                "p_up": round(p_up, 4), "p_down": round(p_down, 4),
                "price": float(row["close"]),
                "sl": float(open_t[sl_col]) if sl_col and sl_col in open_t else round(float(row["close"]) * 0.9985, 2),
                "tp": float(open_t[tp_col]) if tp_col and tp_col in open_t else round(float(row["close"]) * 1.0045, 2),
            }))

        if ts in by_exit:
            t   = by_exit[ts]
            pnl = float(t[pnl_col]) if pnl_col and pnl_col in t else 0.0
            balance *= (1.0 + pnl)
            n_trades += 1
            if pnl > 0:
                n_wins += 1
            open_t = None
            await ws.send_str(json.dumps({
                "type": "trade_close", "ts": ts_ms,
                "net_pnl":     round(pnl * balance, 6),
                "direction":   str(t[dir_col])     if dir_col     and dir_col     in t else "LONG",
                "entry_price": float(t[eprice_col]) if eprice_col  and eprice_col  in t else float(row["open"]),
                "exit_price":  float(row["close"]),
                "reason":      str(t[reason_col])  if reason_col  and reason_col  in t else "TP/SL",
            }))

        await ws.send_str(json.dumps({
            "type": "candle", "ts": ts_ms,
            "open": float(row["open"]), "high": float(row["high"]),
            "low":  float(row["low"]),  "close": float(row["close"]),
            "volume":  float(row["volume"]),
            "buy_vol": float(row["taker_buy_vol"]) if "taker_buy_vol" in row else float(row["volume"]) * 0.5,
        }))

        eq_val = float(equity[i]) if i < len(equity) else balance
        await ws.send_str(json.dumps({
            "type": "stats", "ts": ts_ms,
            "balance":   round(balance, 4),
            "vwap":      float(row["vwap"])      if "vwap"      in row else float(row["close"]),
            "rel_vol":   float(row["rel_vol"])   if "rel_vol"   in row else 1.0,
            "buy_ratio": float(row["buy_ratio"]) if "buy_ratio" in row else 0.5,
            "equity":    round(eq_val, 4),
            "n_trades":  n_trades,
            "win_rate":  round(n_wins / n_trades if n_trades else 0, 3),
            "position":  str(open_t[dir_col])     if open_t is not None and dir_col     else None,
            "unreal":    0.0,
            "entry":     float(open_t[eprice_col]) if open_t is not None and eprice_col and eprice_col in open_t else None,
            "sl":        float(open_t[sl_col])     if open_t is not None and sl_col     and sl_col     in open_t else None,
            "tp":        float(open_t[tp_col])     if open_t is not None and tp_col     and tp_col     in open_t else None,
            "p_up":  round(p_up,   4),
            "p_down": round(p_down, 4),
        }))

        if i % 100 == 0:
            await ws.send_str(json.dumps({"type": "replay_info", "ts": ts_ms, "pct": round(i / n_bars * 100, 1)}))

        if delay > 0:
            await asyncio.sleep(delay)

    final_eq  = float(equity[-1]) if len(equity) else balance
    total_ret = (final_eq / init_cap - 1.0) if init_cap else 0.0
    await ws.send_str(json.dumps({
        "type": "replay_done",
        "summary": {
            "n_trades":         len(trades_df),
            "win_rate_pct":     round(n_wins / max(n_trades, 1) * 100, 1),
            "net_pnl":          round(final_eq - init_cap, 4),
            "total_return_pct": round(total_ret * 100, 2),
            "max_drawdown_pct": round(float(report.get("max_drawdown", report.get("max_dd", 0))) * 100, 2),
            "profit_factor":    round(float(report.get("profit_factor", 0)), 2),
            "final_capital":    round(final_eq, 4),
        },
    }))


# ── Precompute (runs as background task) ──────────────────────────────────────

async def _precompute_bg(data_file: str, cfg: dict) -> None:
    global SIM_DATA, SIM_READY

    loop = asyncio.get_event_loop()

    def _run() -> dict:
        print("[render_sim] Loading CSV …")
        df = data_mod.load_csv(data_file)

        # Limit bars to stay within free-tier RAM
        if MAX_BARS and len(df) > MAX_BARS:
            df = df.iloc[-MAX_BARS:].copy()
            print(f"[render_sim] Trimmed to last {MAX_BARS} bars")

        print("[render_sim] Computing features …")
        df_feat = feat_mod.compute_features(df, cfg)
        df_feat = df_feat.dropna().copy()

        # Use float32 to halve memory
        float_cols = df_feat.select_dtypes(include="float64").columns
        df_feat[float_cols] = df_feat[float_cols].astype("float32")

        print("[render_sim] XGBoost inference …")
        model = xgb.Booster()
        model.load_model(str(BASE / "models" / "xgb_model.json"))

        feat_cols = feat_mod.get_feature_columns(cfg)
        feat_cols = [c for c in feat_cols if c in df_feat.columns]
        dmatrix   = xgb.DMatrix(df_feat[feat_cols].values.astype("float32"), feature_names=feat_cols)
        probs     = model.predict(dmatrix)

        n_cls = int(cfg.get("model", {}).get("num_class", 5))
        if probs.ndim == 1:
            probs = probs.reshape(-1, n_cls)

        del dmatrix, model  # free RAM

        print("[render_sim] Running backtest …")
        trades_df, equity_curve, report = run_backtest(df_feat, probs, cfg)
        print(f"[render_sim] Ready — {len(trades_df)} trades.")

        return {"df": df_feat, "trades_df": trades_df,
                "equity_curve": equity_curve, "probs": probs,
                "cfg": cfg, "report": report}

    data = await loop.run_in_executor(None, _run)
    SIM_DATA.update(data)
    SIM_READY = True
    print("[render_sim] Simulation ready for streaming.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True)
    parser.add_argument("--speed",    type=float, default=30.0)
    parser.add_argument("--max-bars", type=int,   default=5000)
    parser.add_argument("--port",     type=int,   default=int(os.environ.get("PORT", 8080)))
    args = parser.parse_args()

    global SPEED, MAX_BARS
    SPEED    = args.speed
    MAX_BARS = args.max_bars

    with open(BASE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    app = web.Application()
    app.router.add_get("/",           handle_index)
    app.router.add_get("/ws",         websocket_handler)
    app.router.add_get("/{filename}", handle_static)

    async def _startup(app_: web.Application) -> None:
        asyncio.create_task(_precompute_bg(args.data, cfg))

    app.on_startup.append(_startup)

    print(f"[render_sim] Starting server on port {args.port} …")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
