"""
sim/binance_ws_feed.py – Binance public WebSocket 1m kline feed.

Connects to wss://stream.binance.com:9443/ws/<symbol>@kline_1m
and yields a pd.Series for each CLOSED candle.

No API key required (public market data).

Usage (async)
-------------
    import asyncio
    from sim.binance_ws_feed import BinanceWSFeed

    async def main():
        feed = BinanceWSFeed("BTCUSDT")
        async for candle in feed:
            print(candle)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import pandas as pd

try:
    import websockets
except ImportError:
    raise ImportError("websockets package required: pip install websockets>=12.0")


class BinanceWSFeed:
    """
    Async iterator that yields closed 1m candles from Binance WebSocket.

    Each yielded item is a pd.Series with the same schema as the CSV feed:
        open, high, low, close, volume, taker_buy_vol
    The series name is a UTC-aware pd.Timestamp of the candle open time.

    Parameters
    ----------
    symbol       : e.g. "BTCUSDT" (case-insensitive)
    ws_url       : Binance WS base URL
    max_retries  : reconnect attempts on failure
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        ws_url: str = "wss://stream.binance.com:9443/ws",
        max_retries: int = 10,
    ):
        self.symbol      = symbol.lower()
        self.ws_url      = ws_url
        self.max_retries = max_retries
        self._stream_url = f"{ws_url}/{self.symbol}@kline_1m"
        self._queue: asyncio.Queue[pd.Series] = asyncio.Queue()
        self._running     = False
        self._recv_task   = None

    # ── Async iterator interface ──────────────────────────────────────────────
    def __aiter__(self) -> AsyncIterator[pd.Series]:
        return self

    async def __anext__(self) -> pd.Series:
        if not self._running:
            await self.start()
        return await self._queue.get()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Start the background WebSocket receiver."""
        self._running  = True
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def stop(self) -> None:
        """Gracefully stop the feed."""
        self._running = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

    # ── Internals ─────────────────────────────────────────────────────────────
    async def _receive_loop(self) -> None:
        retries = 0
        while self._running and retries <= self.max_retries:
            try:
                async with websockets.connect(
                    self._stream_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    print(f"[BinanceWSFeed] Connected → {self._stream_url}")
                    retries = 0  # reset on successful connect
                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        candle = self._parse(msg)
                        if candle is not None:
                            await self._queue.put(candle)

            except (websockets.exceptions.ConnectionClosed,
                    ConnectionError, OSError) as exc:
                retries += 1
                wait = min(2 ** retries, 60)
                print(f"[BinanceWSFeed] Disconnected ({exc}). "
                      f"Retry {retries}/{self.max_retries} in {wait}s …")
                await asyncio.sleep(wait)

        print("[BinanceWSFeed] Feed stopped.")

    @staticmethod
    def _parse(msg: dict) -> pd.Series | None:
        """
        Parse a Binance kline WS message.
        Returns a pd.Series only for CLOSED candles (k.x == True).
        """
        k = msg.get("k", {})
        if not k.get("x", False):   # not a closed candle
            return None

        ts = pd.Timestamp(int(k["t"]), unit="ms", tz="UTC")
        s  = pd.Series(
            {
                "open":          float(k["o"]),
                "high":          float(k["h"]),
                "low":           float(k["l"]),
                "close":         float(k["c"]),
                "volume":        float(k["v"]),
                "taker_buy_vol": float(k["V"]),
            },
            name=ts,
        )
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: synchronous wrapper for non-async contexts
# ─────────────────────────────────────────────────────────────────────────────

class SyncBinanceWSFeed:
    """
    Thin synchronous wrapper around BinanceWSFeed using a background thread.
    Yields candles via Python generator.

    Usage
    -----
        for candle in SyncBinanceWSFeed("BTCUSDT"):
            process(candle)
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        import threading, queue
        self.symbol = symbol
        self._q     = queue.Queue()
        self._stop  = threading.Event()

        def _runner():
            asyncio.run(self._async_run(self._q, self._stop))

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    async def _async_run(self, q, stop_event) -> None:
        feed = BinanceWSFeed(self.symbol)
        await feed.start()
        while not stop_event.is_set():
            try:
                candle = await asyncio.wait_for(feed._queue.get(), timeout=1.0)
                q.put(candle)
            except asyncio.TimeoutError:
                pass
        await feed.stop()

    def __iter__(self):
        import queue as _q
        while True:
            try:
                yield self._q.get(timeout=120)   # wait up to 2 min per bar
            except _q.Empty:
                print("[SyncBinanceWSFeed] No bar received in 120s – stopping.")
                break

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
