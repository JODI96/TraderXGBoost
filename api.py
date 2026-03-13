"""
api.py – FastAPI wrapper for TraderXGBoost backtester.

Deploy on Render.com:
  Build:  pip install -r requirements.txt
  Start:  uvicorn api:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import features as feat_mod
from backtest import run_backtest

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="TraderXGBoost API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain after deployment
    allow_methods=["GET"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    with open(BASE / "config.yaml") as f:
        return yaml.safe_load(f)


async def _fetch_klines(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 1500) -> pd.DataFrame:
    """Fetch OHLCV from Binance public REST API – no API key required."""
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={symbol}&interval={interval}&limit={limit}"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Binance API returned {resp.status}")
            raw = await resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    for col in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[col] = df[col].astype(float)

    df = df.rename(columns={"taker_buy_base": "taker_buy_vol"})
    return df[["open", "high", "low", "close", "volume", "taker_buy_vol"]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/simulate")
async def simulate(bars: int = 1000):
    """
    Fetch the latest `bars` BTCUSDT 1m candles, run feature engineering,
    XGBoost inference, and the event-driven backtester.

    Returns equity curve, trade list, and performance report.
    """
    try:
        cfg = _load_cfg()

        # 1. Market data – fetch extra bars for feature warm-up (~200 bars)
        df = await _fetch_klines(limit=min(bars + 250, 1500))

        # 2. Feature engineering
        df_feat = feat_mod.compute_features(df, cfg)
        df_feat = df_feat.dropna().copy()

        if len(df_feat) < 50:
            raise RuntimeError("Not enough data after feature computation")

        # 3. Load trained model
        model = xgb.Booster()
        model.load_model(str(BASE / "models" / "xgb_model.json"))

        # 4. Inference – produce (n, 5) probability matrix
        feat_cols = feat_mod.get_feature_columns(cfg)
        feat_cols = [c for c in feat_cols if c in df_feat.columns]

        dmatrix = xgb.DMatrix(df_feat[feat_cols].values, feature_names=feat_cols)
        probs: np.ndarray = model.predict(dmatrix)          # shape (n,) or (n, 5)

        if probs.ndim == 1:
            # Multi-class output is flat when num_class > 2 → reshape
            n_classes = int(cfg.get("model", {}).get("num_class", 5))
            probs = probs.reshape(-1, n_classes)

        # 5. Backtest
        trades_df, equity_curve, report = run_backtest(df_feat, probs, cfg)

        # 6. Format equity curve for Lightweight Charts (UNIX timestamps)
        equity: list[dict] = [
            {"time": int(ts.timestamp()), "value": round(float(v), 6)}
            for ts, v in zip(df_feat.index, equity_curve)
        ]

        # 7. Format trades
        trades: list[dict] = []
        if not trades_df.empty:
            time_col  = next((c for c in ["entry_time", "open_time"] if c in trades_df.columns), None)
            side_col  = next((c for c in ["side", "direction"]       if c in trades_df.columns), None)
            pnl_col   = next((c for c in ["pnl_pct", "pnl", "return"] if c in trades_df.columns), None)

            for _, row in trades_df.iterrows():
                t_time = row[time_col] if time_col else row.name
                trades.append({
                    "time": int(pd.Timestamp(t_time).timestamp()),
                    "side": str(row[side_col]) if side_col else "—",
                    "pnl":  round(float(row[pnl_col]), 6) if pnl_col else 0.0,
                })

        # Normalise report keys to expected frontend names
        stats: dict = {
            "total_return":  report.get("total_return",  report.get("net_return", 0)),
            "win_rate":      report.get("win_rate",      report.get("win_pct",    0)),
            "profit_factor": report.get("profit_factor", 0),
            "max_drawdown":  report.get("max_drawdown",  report.get("max_dd",     0)),
            "sharpe":        report.get("sharpe",        report.get("sharpe_ratio", 0)),
            "n_trades":      report.get("n_trades",      report.get("num_trades", len(trades_df))),
        }

        return {"equity_curve": equity, "trades": trades, "stats": stats}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
