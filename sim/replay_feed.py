"""
sim/replay_feed.py – Historical CSV replay feed.

Iterates over a Binance 1m CSV row-by-row, yielding one pd.Series
per closed candle.  Optionally sleeps between bars to simulate real-time.

Usage (as iterator)
-------------------
    from sim.replay_feed import ReplayFeed
    for candle in ReplayFeed("Data/BTCUSDT/full_year/2025_1m.csv", speed=0):
        # candle is a pd.Series with index (timestamp, open, high, …)
        ...
"""

from __future__ import annotations

import time
from typing import Iterator

import pandas as pd

import data as data_mod


class ReplayFeed:
    """
    Iterator that yields historical 1m candles from a CSV file.

    Parameters
    ----------
    data_file        : path to Binance OHLCV CSV
    speed_multiplier : 0  → no sleep (as-fast-as-possible)
                       1  → real-time (sleep 60 s between bars)
                       60 → one bar per second
    start_ts         : optional start timestamp (pd.Timestamp, UTC)
    end_ts           : optional end timestamp
    """

    def __init__(
        self,
        data_file: str,
        speed_multiplier: float = 0,
        start_ts: pd.Timestamp | None = None,
        end_ts:   pd.Timestamp | None = None,
    ):
        self.data_file        = data_file
        self.speed_multiplier = speed_multiplier

        self._df = data_mod.load_csv(data_file)

        if start_ts is not None:
            self._df = self._df[self._df.index >= start_ts]
        if end_ts is not None:
            self._df = self._df[self._df.index <= end_ts]

        print(f"[ReplayFeed] Loaded {len(self._df):,} bars "
              f"from {self._df.index[0]} to {self._df.index[-1]}")

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[pd.Series]:
        sleep_s = 60.0 / max(self.speed_multiplier, 1e-9) \
                  if self.speed_multiplier > 0 else 0.0

        for ts, row in self._df.iterrows():
            yield row
            if sleep_s > 0:
                time.sleep(sleep_s)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full underlying DataFrame (for batch use)."""
        return self._df.copy()
