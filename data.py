"""
data.py – Load and concatenate Binance OHLCV CSVs.

Binance CSV columns (12 total):
  0  open_time          ms timestamp
  1  open
  2  high
  3  low
  4  close
  5  volume             base asset volume
  6  close_time         ms timestamp (unused)
  7  quote_vol          quote asset volume (unused)
  8  num_trades
  9  taker_buy_vol      taker buy base asset volume  ← real delta source
  10 taker_buy_quote_vol (unused)
  11 ignore
"""

import os
import glob
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


BINANCE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "num_trades",
    "taker_buy_vol", "taker_buy_quote_vol", "ignore",
]

KEEP_COLS = ["open", "high", "low", "close", "volume", "taker_buy_vol"]


# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load a single Binance OHLCV CSV. Returns DataFrame indexed by UTC datetime."""
    df = pd.read_csv(
        path,
        header=None,
        names=BINANCE_COLS,
        dtype={
            "open_time": np.int64,
            "open": np.float64, "high": np.float64,
            "low": np.float64,  "close": np.float64,
            "volume": np.float64, "taker_buy_vol": np.float64,
        },
    )
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index.name = "timestamp"
    df = df[KEEP_COLS].copy()
    # Sanity: remove duplicate timestamps, sort
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_year(base_dir: str, year: int, tf: str = "1m") -> pd.DataFrame:
    """Load a full-year CSV for a given timeframe."""
    path = os.path.join(base_dir, f"{year}_{tf}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return load_csv(path)


def load_all(cfg: dict, tf: str = "1m") -> pd.DataFrame:
    """Load and concatenate all configured years for the given timeframe."""
    base = cfg["data"]["base_dir"]
    years = cfg["data"]["years"]
    frames = []
    for yr in years:
        path = os.path.join(base, f"{yr}_{tf}.csv")
        if not os.path.exists(path):
            print(f"[data] WARNING: {path} not found, skipping.")
            continue
        print(f"[data] Loading {path} …")
        frames.append(load_csv(path))
    if not frames:
        raise RuntimeError("No data files found. Check config.data.base_dir and years.")
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"[data] Loaded {len(df):,} rows  "
          f"({df.index[0]} to {df.index[-1]})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def candle_from_dict(d: dict) -> pd.Series:
    """
    Convert a raw kline dict (e.g. from Binance WS) to a 1-row Series.
    Supports both WS message fields and CSV-loaded dicts.
    """
    return pd.Series({
        "open":          float(d.get("o", d.get("open", 0))),
        "high":          float(d.get("h", d.get("high", 0))),
        "low":           float(d.get("l", d.get("low",  0))),
        "close":         float(d.get("c", d.get("close", 0))),
        "volume":        float(d.get("v", d.get("volume", 0))),
        "taker_buy_vol": float(d.get("V", d.get("taker_buy_vol", 0))),
    })


# ─────────────────────────────────────────────────────────────────────────────
def load_all_coins(cfg: dict, tf: str = "1m") -> dict[str, pd.DataFrame]:
    """
    Load 1m data for all coins listed in cfg["data"]["coins"].
    Returns a dict {symbol: DataFrame} – one DataFrame per coin.
    Each DataFrame has the same 6-column format as load_all().
    """
    root  = cfg["data"]["data_root"]
    coins = cfg["data"].get("coins", [cfg["data"]["symbol"]])
    years = cfg["data"]["years"]

    result: dict[str, pd.DataFrame] = {}
    for coin in coins:
        base = os.path.join(root, coin, "full_year")
        frames = []
        for yr in years:
            path = os.path.join(base, f"{yr}_{tf}.csv")
            if not os.path.exists(path):
                print(f"[data] WARNING: {path} not found, skipping.")
                continue
            print(f"[data] Loading {path} ...")
            frames.append(load_csv(path))
        if not frames:
            print(f"[data] WARNING: no data found for {coin}, skipping.")
            continue
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        print(f"[data] {coin}: {len(df):,} rows  "
              f"({df.index[0]} to {df.index[-1]})")
        result[coin] = df
    return result


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    df = load_all(cfg)
    print(df.tail())
    print(df.dtypes)
