"""
sim/binance_portfolio.py – Real Binance USDT-M Futures execution.

Drop-in replacement for Portfolio. ExecutionEngine works unchanged.

Order flow per trade
--------------------
  1. MARKET order  → entry fill
  2. STOP_MARKET   → server-side stop-loss  (closePosition=true)
  3. TAKE_PROFIT_MARKET → server-side take-profit (closePosition=true)

On each bar check_exits() polls the Binance position endpoint.
If the position is gone, we detect which order filled and record the trade.
For TIME/FORCE exits we send a market CLOSE order and cancel the other leg.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RealTrade:
    direction:   str
    entry_ts:    str
    exit_ts:     str
    entry_price: float
    exit_price:  float
    size:        float
    raw_pnl:     float
    cost:        float
    net_pnl:     float
    exit_reason: str
    entry_bar:   int
    exit_bar:    int


@dataclass
class RealPosition:
    direction:    int        # +1 long, -1 short
    entry_price:  float
    sl_price:     float
    tp_price:     float
    size:         float      # base-asset qty (BTC)
    entry_ts:     str
    entry_bar:    int
    atr_at_entry: float
    sl_order_id:  Optional[int] = None
    tp_order_id:  Optional[int] = None

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.direction * self.size


# ─────────────────────────────────────────────────────────────────────────────
class BinancePortfolio:
    """
    Mirrors the Portfolio interface so ExecutionEngine can be used as-is.
    All trades go to Binance USDT-M Futures.

    Parameters
    ----------
    api_key / api_secret : Binance Futures API credentials
    symbol               : e.g. "BTCUSDT"
    cfg                  : full config dict
    """

    def __init__(self, api_key: str, api_secret: str, symbol: str, cfg: dict):
        from binance.client import Client
        self._client = Client(api_key, api_secret)
        self.symbol  = symbol.upper()

        tc = cfg["trading"]
        lc = cfg["labels"]

        self.initial_capital = tc["initial_capital"]
        self.cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
                        lc["slippage"]  + lc["spread"]) * 2
        self.cooldown_bars = tc["cooldown"]

        # leverage = position_size_pct (e.g. 30 → 30x)
        self.leverage = max(1, int(tc.get("position_size_pct", 10)))
        self._pos_pct = tc.get("position_size_pct", 10)

        # Runtime state
        self.capital:       float               = self.initial_capital
        self.position:      Optional[RealPosition] = None
        self.trade_log:     list[RealTrade]     = []
        self.equity_curve:  list[float]         = []
        self.bar_count:     int                 = 0
        self._cooldown_rem: int                 = 0

        # Exchange precision for the symbol
        self._qty_precision:   int = 3   # decimal places for quantity
        self._price_precision: int = 1   # decimal places for price

        self._init_exchange()

    # ── Setup ─────────────────────────────────────────────────────────────────
    def _init_exchange(self) -> None:
        """Set leverage and load precision info."""
        try:
            self._client.futures_change_leverage(
                symbol=self.symbol, leverage=self.leverage)
            logger.info(f"[Binance] Leverage set to {self.leverage}x on {self.symbol}")
        except Exception as exc:
            logger.warning(f"[Binance] Could not set leverage: {exc}")

        try:
            info = self._client.futures_exchange_info()
            for sym in info["symbols"]:
                if sym["symbol"] == self.symbol:
                    for filt in sym["filters"]:
                        if filt["filterType"] == "LOT_SIZE":
                            step = float(filt["stepSize"])
                            self._qty_precision = max(0, -int(math.log10(step)))
                        if filt["filterType"] == "PRICE_FILTER":
                            tick = float(filt["tickSize"])
                            self._price_precision = max(0, -int(math.log10(tick)))
                    break
        except Exception as exc:
            logger.warning(f"[Binance] Could not load exchange info: {exc}")

        self._sync_balance()

    def _sync_balance(self) -> None:
        try:
            account = self._client.futures_account()
            for asset in account["assets"]:
                if asset["asset"] == "USDT":
                    self.capital = float(asset["availableBalance"])
                    logger.info(f"[Binance] Balance synced: ${self.capital:,.2f} USDT")
                    return
        except Exception as exc:
            logger.warning(f"[Binance] Balance sync failed: {exc}")

    def _rq(self, qty: float) -> float:
        factor = 10 ** self._qty_precision
        return math.floor(qty * factor) / factor

    def _rp(self, price: float) -> str:
        return f"{price:.{self._price_precision}f}"

    # ── State properties (same as Portfolio) ─────────────────────────────────
    @property
    def is_flat(self) -> bool:
        return self.position is None

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown_rem > 0

    @property
    def can_enter(self) -> bool:
        return self.is_flat and not self.in_cooldown

    def mark_to_market(self, price: float) -> float:
        if self.position:
            return self.capital + self.position.unrealised_pnl(price)
        return self.capital

    # ── Position management ───────────────────────────────────────────────────
    def open_trade(
        self,
        direction:  int,
        price:      float,
        atr:        float,
        sl_atr:     float,
        tp_atr:     float,
        pos_pct:    float,
        timestamp:  str,
        sl_pct:     Optional[float] = None,
        tp_pct:     Optional[float] = None,
    ) -> None:
        if not self.can_enter:
            return

        # ── SL / TP prices ────────────────────────────────────────────────────
        if sl_pct is not None and tp_pct is not None:
            sl_price = price * (1.0 - direction * sl_pct)
            tp_price = price * (1.0 + direction * tp_pct)
        else:
            sl_price = price - direction * sl_atr * atr
            tp_price = price + direction * tp_atr * atr

        # ── Notional / quantity ───────────────────────────────────────────────
        # pos_pct here is the leverage multiplier (e.g. 30 = 30x)
        notional = self.capital * pos_pct        # e.g. $10 000 * 30 = $300 000
        qty      = self._rq(notional / price)

        side       = "BUY"  if direction == 1 else "SELL"
        side_close = "SELL" if direction == 1 else "BUY"

        try:
            # 1. Market entry ──────────────────────────────────────────────────
            entry_resp = self._client.futures_create_order(
                symbol   = self.symbol,
                side     = side,
                type     = "MARKET",
                quantity = qty,
            )
            # avgPrice may be "0" on some responses — fall back to last price
            fill_price = float(entry_resp.get("avgPrice") or 0) or price
            logger.info(f"[OPEN]  {side:4s} {qty} {self.symbol} @ {fill_price:.2f}")

            # 2. Stop-loss (server-side) ───────────────────────────────────────
            sl_resp = self._client.futures_create_order(
                symbol        = self.symbol,
                side          = side_close,
                type          = "STOP_MARKET",
                stopPrice     = self._rp(sl_price),
                closePosition = "true",
                timeInForce   = "GTE_GTC",
            )
            logger.info(f"  SL order #{sl_resp['orderId']}  stop={sl_price:.2f}")

            # 3. Take-profit (server-side) ─────────────────────────────────────
            tp_resp = self._client.futures_create_order(
                symbol        = self.symbol,
                side          = side_close,
                type          = "TAKE_PROFIT_MARKET",
                stopPrice     = self._rp(tp_price),
                closePosition = "true",
                timeInForce   = "GTE_GTC",
            )
            logger.info(f"  TP order #{tp_resp['orderId']}  stop={tp_price:.2f}")

            self.position = RealPosition(
                direction    = direction,
                entry_price  = fill_price,
                sl_price     = sl_price,
                tp_price     = tp_price,
                size         = qty,
                entry_ts     = str(timestamp),
                entry_bar    = self.bar_count,
                atr_at_entry = atr,
                sl_order_id  = sl_resp.get("orderId"),
                tp_order_id  = tp_resp.get("orderId"),
            )

        except Exception as exc:
            logger.error(f"[Binance] open_trade FAILED: {exc}")

    def check_exits(self, high: float, low: float) -> tuple[Optional[str], float]:
        """
        Poll Binance to detect whether SL or TP fired server-side.
        high/low are unused (Binance handles intrabar execution).
        Returns (reason, exit_price) or (None, 0.0).
        """
        if self.position is None:
            return None, 0.0

        try:
            positions = self._client.futures_position_information(symbol=self.symbol)
            for p in positions:
                if p["symbol"] == self.symbol:
                    if abs(float(p["positionAmt"])) < 1e-6:
                        # Position closed — find out why
                        return self._detect_exit()
        except Exception as exc:
            logger.warning(f"[Binance] check_exits poll failed: {exc}")

        return None, 0.0

    def _detect_exit(self) -> tuple[str, float]:
        """Determine which order filled (SL or TP) and the fill price."""
        pos = self.position
        for order_id, reason in [
            (pos.sl_order_id, "SL"),
            (pos.tp_order_id, "TP"),
        ]:
            if order_id is None:
                continue
            try:
                order = self._client.futures_get_order(
                    symbol=self.symbol, orderId=order_id)
                if order["status"] == "FILLED":
                    fill = float(order.get("avgPrice") or order.get("stopPrice") or 0)
                    if fill == 0:
                        fill = pos.sl_price if reason == "SL" else pos.tp_price
                    return reason, fill
            except Exception as exc:
                logger.warning(f"[Binance] order query failed: {exc}")

        # Fallback: infer from last account trade
        try:
            trades = self._client.futures_account_trades(
                symbol=self.symbol, limit=1)
            if trades:
                exit_p = float(trades[0]["price"])
                reason = ("TP" if (pos.direction == 1 and exit_p >= pos.tp_price)
                          or    (pos.direction == -1 and exit_p <= pos.tp_price)
                          else "SL")
                return reason, exit_p
        except Exception:
            pass

        # Ultra-fallback: assume SL (conservative)
        return "SL", pos.sl_price

    def close_trade(self, price: float, timestamp: str, reason: str) -> RealTrade:
        """
        For SL/TP: just cancel remaining orders and record.
        For TIME/FORCE: send market-close order first, then cancel.
        """
        if self.position is None:
            raise RuntimeError("No open position to close.")

        pos = self.position

        if reason in ("TIME", "FORCE"):
            # Send market close
            side_close = "SELL" if pos.direction == 1 else "BUY"
            try:
                resp = self._client.futures_create_order(
                    symbol      = self.symbol,
                    side        = side_close,
                    type        = "MARKET",
                    quantity    = pos.size,
                    reduceOnly  = "true",
                )
                fill = float(resp.get("avgPrice") or 0)
                if fill:
                    price = fill
            except Exception as exc:
                logger.error(f"[Binance] market close failed: {exc}")

        # Cancel all remaining open orders (the other SL or TP leg)
        self._cancel_all_orders()

        raw_pnl  = (price - pos.entry_price) * pos.direction * pos.size
        cost_pnl = pos.entry_price * pos.size * self.cost_rt
        net_pnl  = raw_pnl - cost_pnl

        # Re-sync balance from exchange
        self._sync_balance()

        trade = RealTrade(
            direction    = "LONG" if pos.direction == 1 else "SHORT",
            entry_ts     = pos.entry_ts,
            exit_ts      = str(timestamp),
            entry_price  = pos.entry_price,
            exit_price   = round(price, self._price_precision),
            size         = pos.size,
            raw_pnl      = round(raw_pnl,  4),
            cost         = round(cost_pnl, 4),
            net_pnl      = round(net_pnl,  4),
            exit_reason  = reason,
            entry_bar    = pos.entry_bar,
            exit_bar     = self.bar_count,
        )
        self.trade_log.append(trade)
        self.position      = None
        self._cooldown_rem = self.cooldown_bars
        logger.info(f"[CLOSE] {reason:5s} @ {price:.2f}  PnL={net_pnl:+.2f}$")
        return trade

    def _cancel_all_orders(self) -> None:
        try:
            self._client.futures_cancel_all_open_orders(symbol=self.symbol)
            logger.info("[Binance] All open orders cancelled.")
        except Exception as exc:
            logger.warning(f"[Binance] cancel_all_orders failed: {exc}")

    # ── Bar update ────────────────────────────────────────────────────────────
    def on_bar(self, price: float) -> None:
        self.bar_count += 1
        if self._cooldown_rem > 0:
            self._cooldown_rem -= 1
        self.equity_curve.append(self.mark_to_market(price))

    # ── Summary / persistence ─────────────────────────────────────────────────
    def summary(self) -> dict:
        pnls = [t.net_pnl for t in self.trade_log]
        if not pnls:
            return {"n_trades": 0, "capital": round(self.capital, 2)}
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p < 0]
        return {
            "n_trades":         len(pnls),
            "win_rate_pct":     round(len(wins) / len(pnls) * 100, 1),
            "profit_factor":    round(sum(wins) / (-sum(loss) + 1e-9), 3)
                                if loss else float("inf"),
            "net_pnl":          round(sum(pnls), 2),
            "total_return_pct": round((self.capital - self.initial_capital)
                                      / self.initial_capital * 100, 2),
            "final_capital":    round(self.capital, 2),
        }

    def save(self, path: str) -> None:
        data = {
            "summary": self.summary(),
            "trades":  [asdict(t) for t in self.trade_log],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[portfolio] Saved -> {path}")
