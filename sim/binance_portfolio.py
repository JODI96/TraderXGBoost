"""
sim/binance_portfolio.py – Real Binance USDT-M Futures execution.

Drop-in replacement for Portfolio. ExecutionEngine works unchanged.

Order flow per trade
--------------------
  1. MARKET order  -> entry fill
  2. STOP_MARKET   -> server-side stop-loss  (closePosition=true)
  3. TAKE_PROFIT_MARKET -> server-side take-profit (closePosition=true)

On each bar on_bar() polls the Binance position endpoint.
If the position is gone, check_exits() detects the exit via recent trades.
For TIME/FORCE exits we send a market CLOSE order and cancel the other leg.
"""

from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass, asdict
from typing import Optional

# Shared event log — trade_live.py reads this to render in the dashboard.
# Only important events go here (fills, guard failures, errors).
# Routine "confirmed alive" guard ticks are intentionally suppressed.
event_log: collections.deque = collections.deque(maxlen=8)


def _log(msg: str) -> None:
    """Append to event_log (shown in dashboard) without printing to stdout."""
    event_log.append(msg)


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
    """

    def __init__(self, api_key: str, api_secret: str, symbol: str, cfg: dict):
        from binance.client import Client
        self._client = Client(api_key, api_secret)
        self.symbol  = symbol.upper()

        tc = cfg["trading"]
        lc = cfg["labels"]

        self.initial_capital = tc["initial_capital"]
        self.maker_fee     = lc["maker_fee"]
        self.taker_fee     = lc["taker_fee"]
        self.cooldown_bars = tc["cooldown"]

        self.leverage = max(1, int(tc.get("position_size_pct", 10)))
        self._pos_pct = tc.get("position_size_pct", 10)

        # Runtime state
        self.capital:          float                  = self.initial_capital
        self.position:         Optional[RealPosition] = None
        self.trade_log:        list[RealTrade]        = []
        self.equity_curve:     list[float]            = []
        self.bar_count:        int                    = 0
        self._cooldown_rem:    int                    = 0
        self._exchange_closed: bool                   = False

        self._qty_precision:   int = 3
        self._price_precision: int = 1
        self._guard_failed:    bool = False
        self.cfg_sl_pct = tc.get("sl_pct", 0.002)
        self.cfg_tp_pct = tc.get("tp_pct", 0.006)

        # Pending limit open order
        self.pending_dir:   int   = 0
        self.pending_price: float = 0.0
        self.pending_bar:   int   = 0
        self._pending_entry_order_id              = None
        self._pending_qty:            float       = 0.0
        self._limit_entry_filled:     bool        = False

        # Pending limit close order (time-stop → limit before market fallback)
        self.pending_close_price:    float = 0.0
        self.pending_close_attempts: int   = 0
        self._pending_close_order_id              = None

        self._init_exchange()

    # ── Setup ─────────────────────────────────────────────────────────────────
    def _init_exchange(self) -> None:
        try:
            self._client.futures_change_leverage(
                symbol=self.symbol, leverage=self.leverage)
            print(f"[Binance] Leverage set to {self.leverage}x on {self.symbol}")
        except Exception as exc:
            print(f"[Binance] Could not set leverage: {exc}")

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
            print(f"[Binance] Could not load exchange info: {exc}")

        self._sync_balance()
        self._reconcile_on_startup()

    def _sync_balance(self) -> None:
        try:
            account = self._client.futures_account()
            for asset in account["assets"]:
                if asset["asset"] == "USDT":
                    self.capital = float(asset["availableBalance"])
                    print(f"[Binance] Balance synced: ${self.capital:,.2f} USDT")
                    return
        except Exception as exc:
            print(f"[Binance] Balance sync failed: {exc}")

    def _reconcile_on_startup(self) -> None:
        """On startup, detect orphaned Binance position and protect it."""
        try:
            account = self._client.futures_account()
        except Exception as exc:
            _log(f"[reconcile] Could not fetch account: {exc}")
            return

        pos_amt = entry_price = 0.0
        for p in account.get("positions", []):
            if p["symbol"] == self.symbol:
                pos_amt     = float(p.get("positionAmt", 0))
                entry_price = float(p.get("entryPrice",  0))
                break

        if abs(pos_amt) < 1e-6:
            _log("[reconcile] No open position – starting fresh.")
            return

        direction = 1 if pos_amt > 0 else -1
        size      = abs(pos_amt)
        dir_str   = "LONG" if direction == 1 else "SHORT"
        _log(f"[reconcile] ORPHANED POSITION: {dir_str} {size} @ {entry_price:.2f}")

        # Find existing SL/TP open orders
        sl_order_id = sl_stop = None
        tp_order_id = tp_stop = None
        try:
            for o in self._client.futures_get_open_orders(symbol=self.symbol):
                otype = o.get("type", "")
                if otype == "STOP_MARKET" and sl_order_id is None:
                    sl_order_id = o.get("orderId")
                    sl_stop     = float(o.get("stopPrice", 0))
                    _log(f"[reconcile] Found SL id={sl_order_id} stop={sl_stop:.2f}")
                elif otype == "TAKE_PROFIT_MARKET" and tp_order_id is None:
                    tp_order_id = o.get("orderId")
                    tp_stop     = float(o.get("stopPrice", 0))
                    _log(f"[reconcile] Found TP id={tp_order_id} stop={tp_stop:.2f}")
        except Exception as exc:
            _log(f"[reconcile] Could not fetch open orders: {exc}")

        # Use actual stop prices if orders found, else compute from config pct
        sl_price = sl_stop if sl_stop else entry_price * (1.0 - direction * self.cfg_sl_pct)
        tp_price = tp_stop if tp_stop else entry_price * (1.0 + direction * self.cfg_tp_pct)
        side_close = "SELL" if direction == 1 else "BUY"

        # Place any missing SL
        if sl_order_id is None:
            _log(f"[reconcile] SL missing – placing at {sl_price:.2f}")
            for att in range(1, 4):
                try:
                    resp = self._client.futures_create_order(
                        symbol=self.symbol, side=side_close, type="STOP_MARKET",
                        stopPrice=self._rp(sl_price), closePosition="true", timeInForce="GTE_GTC",
                    )
                    sl_order_id = resp.get("algoId") or resp.get("orderId")
                    _log(f"[reconcile] SL placed id={sl_order_id} attempt={att}")
                    break
                except Exception as exc:
                    _log(f"[reconcile] SL attempt {att} FAILED: {exc}")

        # Place any missing TP
        if tp_order_id is None:
            _log(f"[reconcile] TP missing – placing at {tp_price:.2f}")
            for att in range(1, 4):
                try:
                    resp = self._client.futures_create_order(
                        symbol=self.symbol, side=side_close, type="TAKE_PROFIT_MARKET",
                        stopPrice=self._rp(tp_price), closePosition="true", timeInForce="GTE_GTC",
                    )
                    tp_order_id = resp.get("algoId") or resp.get("orderId")
                    _log(f"[reconcile] TP placed id={tp_order_id} attempt={att}")
                    break
                except Exception as exc:
                    _log(f"[reconcile] TP attempt {att} FAILED: {exc}")

        # Still unprotected after 3 attempts each → emergency close
        if sl_order_id is None or tp_order_id is None:
            _log("[reconcile] CRITICAL: Cannot protect orphan – emergency close")
            try:
                self._client.futures_create_order(
                    symbol=self.symbol, side=side_close, type="MARKET",
                    quantity=self._rq(size), reduceOnly="true",
                )
                _log("[reconcile] Orphaned position closed.")
                return
            except Exception as ce:
                _log(f"[reconcile] Emergency close FAILED: {ce} – adopting with _guard_failed")
                self._guard_failed = True

        # Adopt into local state so guard loop can monitor it
        import datetime
        self.position = RealPosition(
            direction=direction, entry_price=entry_price,
            sl_price=sl_price, tp_price=tp_price, size=size,
            entry_ts=str(datetime.datetime.utcnow()),
            entry_bar=self.bar_count, atr_at_entry=0.0,
            sl_order_id=sl_order_id, tp_order_id=tp_order_id,
        )
        _log(f"[reconcile] Adopted: {dir_str} SL={sl_price:.2f} TP={tp_price:.2f}")

    def _rq(self, qty: float) -> float:
        return round(qty, self._qty_precision)

    def _rp(self, price: float) -> str:
        return f"{price:.{self._price_precision}f}"

    def handle_pending_fill_immediate(self, sl_pct: float, tp_pct: float) -> bool:
        """
        Poll Binance for a pending limit entry fill every ~5 s (called by background task).
        If filled, place SL/TP immediately without waiting for the next bar close.
        Returns True if a fill was handled.
        """
        if self.pending_dir == 0 or self._pending_entry_order_id is None:
            return False
        if self.position is not None:
            return False  # already adopted by execution engine

        try:
            order      = self._client.futures_get_order(
                symbol=self.symbol, orderId=self._pending_entry_order_id)
            status     = order.get("status")
            filled_qty = float(order.get("executedQty") or 0)
        except Exception as exc:
            _log(f"[bg_fill] Poll failed: {exc}")
            return False

        if status not in ("FILLED", "PARTIALLY_FILLED") or filled_qty < 1e-6:
            return False

        fill_price = float(order.get("avgPrice") or self.pending_price)
        direction  = self.pending_dir
        side_close = "SELL" if direction == 1 else "BUY"
        dir_str    = "LONG" if direction == 1 else "SHORT"
        qty        = self._rq(filled_qty)

        _log(f"[bg_fill] {dir_str} limit filled {qty} @ {fill_price:.2f} "
              f"– placing SL/TP immediately")

        # Cancel any unfilled remainder
        if status == "PARTIALLY_FILLED":
            try:
                self._client.futures_cancel_order(
                    symbol=self.symbol, orderId=self._pending_entry_order_id)
            except Exception:
                pass

        sl_price = fill_price * (1.0 - direction * sl_pct)
        tp_price = fill_price * (1.0 + direction * tp_pct)

        sl_order_id = None
        for att in range(1, 4):
            try:
                resp = self._client.futures_create_order(
                    symbol=self.symbol, side=side_close, type="STOP_MARKET",
                    stopPrice=self._rp(sl_price), closePosition="true",
                    timeInForce="GTE_GTC",
                )
                sl_order_id = resp.get("algoId") or resp.get("orderId")
                _log(f"[bg_fill] SL placed id={sl_order_id} stop={sl_price:.2f} att={att}")
                break
            except Exception as exc:
                _log(f"[bg_fill] SL attempt {att} FAILED: {exc}")

        tp_order_id = None
        for att in range(1, 4):
            try:
                resp = self._client.futures_create_order(
                    symbol=self.symbol, side=side_close, type="TAKE_PROFIT_MARKET",
                    stopPrice=self._rp(tp_price), closePosition="true",
                    timeInForce="GTE_GTC",
                )
                tp_order_id = resp.get("algoId") or resp.get("orderId")
                _log(f"[bg_fill] TP placed id={tp_order_id} stop={tp_price:.2f} att={att}")
                break
            except Exception as exc:
                _log(f"[bg_fill] TP attempt {att} FAILED: {exc}")

        # Clear pending state so execution engine won't try to open_trade() again
        self.pending_dir             = 0
        self.pending_price           = 0.0
        self.pending_bar             = 0
        self._pending_entry_order_id = None
        self._pending_qty            = 0.0
        self._limit_entry_filled     = False

        # If SL or TP could not be placed → emergency close
        if sl_order_id is None or tp_order_id is None:
            missing = []
            if sl_order_id is None: missing.append("SL")
            if tp_order_id is None: missing.append("TP")
            _log(f"[bg_fill] {'/'.join(missing)} FAILED after 3 attempts – emergency close")
            self._cancel_all_orders()
            emergency_closed = False
            try:
                self._client.futures_create_order(
                    symbol=self.symbol, side=side_close, type="MARKET",
                    quantity=qty, reduceOnly="true",
                )
                _log("[bg_fill] Position emergency-closed.")
                emergency_closed = True
            except Exception as ce:
                _log(f"[bg_fill] Emergency close FAILED: {ce} – _guard_failed=True")
                self._guard_failed = True

            if not emergency_closed:
                # Adopt into local state so guard loop can retry
                import datetime
                self.position = RealPosition(
                    direction=direction, entry_price=fill_price,
                    sl_price=sl_price, tp_price=tp_price, size=qty,
                    entry_ts=str(datetime.datetime.utcnow()),
                    entry_bar=self.bar_count, atr_at_entry=0.0,
                    sl_order_id=sl_order_id, tp_order_id=tp_order_id,
                )
            return True

        import datetime
        self.position = RealPosition(
            direction=direction, entry_price=fill_price,
            sl_price=sl_price, tp_price=tp_price, size=qty,
            entry_ts=str(datetime.datetime.utcnow()),
            entry_bar=self.bar_count, atr_at_entry=0.0,
            sl_order_id=sl_order_id, tp_order_id=tp_order_id,
        )
        _log(f"[bg_fill] Position adopted: {dir_str} @ {fill_price:.2f} "
              f"SL={sl_price:.2f} TP={tp_price:.2f}")
        return True

    # ── State properties ──────────────────────────────────────────────────────
    @property
    def is_flat(self) -> bool:
        return self.position is None

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown_rem > 0

    @property
    def has_pending(self) -> bool:
        return self.pending_dir != 0

    @property
    def limit_fill_confirmed(self) -> bool:
        """True when Binance confirmed the limit entry fill — bypass simulated check."""
        return self._limit_entry_filled

    @property
    def has_pending_close(self) -> bool:
        return self.pending_close_price != 0.0

    @property
    def can_enter(self) -> bool:
        return self.is_flat and not self.in_cooldown and not self.has_pending

    def mark_to_market(self, price: float) -> float:
        if self.position:
            return self.capital + self.position.unrealised_pnl(price)
        return self.capital

    # ── Pending limit open order ──────────────────────────────────────────────
    def place_pending_order(self, direction: int, limit_price: float) -> None:
        side     = "BUY" if direction == 1 else "SELL"
        notional = math.floor(self.capital * 0.95) * self._pos_pct
        qty      = self._rq(notional / limit_price)
        try:
            resp = self._client.futures_create_order(
                symbol=self.symbol, side=side, type="LIMIT",
                price=self._rp(limit_price), quantity=qty, timeInForce="GTC",
            )
            self._pending_entry_order_id = resp.get("orderId")
            self.pending_dir             = direction
            self.pending_price           = limit_price
            self.pending_bar             = self.bar_count
            self._pending_qty            = qty
            self._limit_entry_filled     = False
            _log(f"[PENDING] {side} LIMIT {qty} @ {limit_price:.2f}  "
                  f"id={self._pending_entry_order_id}")
        except Exception as exc:
            print(f"[Binance] place_pending_order FAILED: {exc} – no order placed, state unchanged")

    def cancel_pending_order(self) -> None:
        if self._pending_entry_order_id:
            try:
                self._client.futures_cancel_order(
                    symbol=self.symbol, orderId=self._pending_entry_order_id)
                _log(f"[PENDING] Entry order cancelled  id={self._pending_entry_order_id}")
            except Exception as exc:
                print(f"[Binance] cancel pending entry FAILED: {exc}")
        self.pending_dir             = 0
        self.pending_price           = 0.0
        self.pending_bar             = 0
        self._pending_entry_order_id = None
        self._pending_qty            = 0.0
        self._limit_entry_filled     = False

    # ── Pending limit close order ─────────────────────────────────────────────
    def place_pending_close(self, close_price: float) -> None:
        if self.position is None:
            return
        side = "SELL" if self.position.direction == 1 else "BUY"
        try:
            resp = self._client.futures_create_order(
                symbol=self.symbol, side=side, type="LIMIT",
                price=self._rp(close_price),
                quantity=self._rq(self.position.size),
                timeInForce="GTC", reduceOnly="true",
            )
            self._pending_close_order_id = resp.get("orderId")
            print(f"[PENDING CLOSE] {side} LIMIT @ {close_price:.2f}  "
                  f"id={self._pending_close_order_id}")
        except Exception as exc:
            print(f"[Binance] place_pending_close FAILED: {exc}")
            self._pending_close_order_id = None
        self.pending_close_price    = close_price
        self.pending_close_attempts = 1

    def update_pending_close(self, close_price: float) -> None:
        if self._pending_close_order_id:
            try:
                self._client.futures_cancel_order(
                    symbol=self.symbol, orderId=self._pending_close_order_id)
            except Exception:
                pass
            self._pending_close_order_id = None
        self.pending_close_attempts += 1
        if self.position is not None:
            side = "SELL" if self.position.direction == 1 else "BUY"
            try:
                resp = self._client.futures_create_order(
                    symbol=self.symbol, side=side, type="LIMIT",
                    price=self._rp(close_price),
                    quantity=self._rq(self.position.size),
                    timeInForce="GTC", reduceOnly="true",
                )
                self._pending_close_order_id = resp.get("orderId")
                print(f"[PENDING CLOSE] Updated @ {close_price:.2f}  "
                      f"id={self._pending_close_order_id}  attempt={self.pending_close_attempts}")
            except Exception as exc:
                print(f"[Binance] update_pending_close FAILED: {exc}")
        self.pending_close_price = close_price

    def clear_pending_close(self) -> None:
        if self._pending_close_order_id:
            try:
                self._client.futures_cancel_order(
                    symbol=self.symbol, orderId=self._pending_close_order_id)
            except Exception:
                pass
            self._pending_close_order_id = None
        self.pending_close_price    = 0.0
        self.pending_close_attempts = 0

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
        entry_bar:  Optional[int]   = None,
    ) -> None:
        if self.position is not None or self.in_cooldown:
            return

        side       = "BUY"  if direction == 1 else "SELL"
        side_close = "SELL" if direction == 1 else "BUY"

        try:
            if self._limit_entry_filled:
                # Limit order already filled on Binance — skip market order
                fill_price = self.pending_price
                qty        = self._pending_qty
                _log(f"[OPEN]  {side:4s} {qty} {self.symbol} @ {fill_price:.2f}  (limit fill)")
            else:
                # Market entry fallback
                notional   = math.floor(self.capital * 0.95) * pos_pct
                qty        = self._rq(notional / price)
                entry_resp = self._client.futures_create_order(
                    symbol=self.symbol, side=side, type="MARKET", quantity=qty)
                fill_price = float(entry_resp.get("avgPrice") or 0) or price
                _log(f"[OPEN]  {side:4s} {qty} {self.symbol} @ {fill_price:.2f}  "
                      f"(signal={price:.2f}, slip={fill_price - price:+.2f})")

            # 2. Recalculate SL/TP from actual fill price
            if sl_pct is not None and tp_pct is not None:
                sl_price = fill_price * (1.0 - direction * sl_pct)
                tp_price = fill_price * (1.0 + direction * tp_pct)
            else:
                sl_price = fill_price - direction * sl_atr * atr
                tp_price = fill_price + direction * tp_atr * atr

            # 3. Stop-loss – place with up to 3 attempts immediately
            sl_order_id = None
            sl_placed   = False
            for _att in range(1, 4):
                try:
                    sl_resp = self._client.futures_create_order(
                        symbol        = self.symbol,
                        side          = side_close,
                        type          = "STOP_MARKET",
                        stopPrice     = self._rp(sl_price),
                        closePosition = "true",
                        timeInForce   = "GTE_GTC",
                    )
                    sl_placed   = True
                    sl_order_id = sl_resp.get("algoId") or sl_resp.get("orderId")
                    print(f"  SL placed  id={sl_order_id}  stop={sl_price:.2f}  "
                          f"({abs(fill_price - sl_price):.2f} pts from fill)")
                    break
                except Exception as sl_exc:
                    print(f"  SL attempt {_att} FAILED: {sl_exc}")

            # 4. Take-profit – place with up to 3 attempts immediately
            tp_order_id = None
            tp_placed   = False
            for _att in range(1, 4):
                try:
                    tp_resp = self._client.futures_create_order(
                        symbol        = self.symbol,
                        side          = side_close,
                        type          = "TAKE_PROFIT_MARKET",
                        stopPrice     = self._rp(tp_price),
                        closePosition = "true",
                        timeInForce   = "GTE_GTC",
                    )
                    tp_placed   = True
                    tp_order_id = tp_resp.get("algoId") or tp_resp.get("orderId")
                    print(f"  TP placed  id={tp_order_id}  stop={tp_price:.2f}  "
                          f"({abs(tp_price - fill_price):.2f} pts from fill)")
                    break
                except Exception as tp_exc:
                    print(f"  TP attempt {_att} FAILED: {tp_exc}")

            self.position = RealPosition(
                direction    = direction,
                entry_price  = fill_price,
                sl_price     = sl_price,
                tp_price     = tp_price,
                size         = qty,
                entry_ts     = str(timestamp),
                entry_bar    = entry_bar if entry_bar is not None else self.bar_count,
                atr_at_entry = atr,
                sl_order_id  = sl_order_id,
                tp_order_id  = tp_order_id,
            )
            self.cancel_pending_order()

            # 5. Status summary
            print(f"  SL status: {'PLACED id=' + str(sl_order_id) if sl_placed else 'FAILED'}")
            print(f"  TP status: {'PLACED id=' + str(tp_order_id) if tp_placed else 'FAILED'}")

            # 6. If SL or TP placement failed after 3 attempts – close immediately
            if not sl_placed or not tp_placed:
                missing = []
                if not sl_placed: missing.append("SL")
                if not tp_placed: missing.append("TP")
                print(f"  {'/'.join(missing)} could not be placed after 3 attempts – closing position")
                self._cancel_all_orders()
                side_close2 = "SELL" if direction == 1 else "BUY"
                emergency_closed = False
                try:
                    self._client.futures_create_order(
                        symbol     = self.symbol,
                        side       = side_close2,
                        type       = "MARKET",
                        quantity   = qty,
                        reduceOnly = "true",
                    )
                    print(f"  Position emergency-closed on entry guard.")
                    emergency_closed = True
                except Exception as ce:
                    print(f"  Emergency close FAILED: {ce} – guard will retry next bar")
                if emergency_closed:
                    self.position = None
                else:
                    # Keep self.position alive so the guard loop can monitor and retry.
                    self._guard_failed = True

        except Exception as exc:
            print(f"[Binance] open_trade FAILED: {exc}")

    def _check_guard_orders(self, current_price: float) -> bool:
        """
        Verify SL and TP orders are still live on Binance.
        - If we have an order ID: query it directly via futures_get_order
        - If ID is None or query fails: attempt to place;
          -4130 = already exists on exchange (confirmed alive)
        Re-tries up to 2 times if genuinely missing.
        Returns False only if order is truly gone and re-placement fails.
        """
        pos = self.position
        if pos is None:
            return True

        side_close = "SELL" if pos.direction == 1 else "BUY"

        def _order_alive(order_id) -> bool:
            if order_id is None:
                return False
            try:
                # Try regular order first, then algo order
                o = self._client.futures_get_order(
                    symbol=self.symbol, orderId=order_id)
                return o.get("status") in ("NEW", "PARTIALLY_FILLED")
            except Exception:
                return False

        def _place_order(order_type: str, stop_price: float):
            """
            Try to place order. Returns (ok, new_order_id).
            -4130 = already exists -> ok=True, id=None (ID unknown but order is live).
            """
            try:
                resp = self._client.futures_create_order(
                    symbol        = self.symbol,
                    side          = side_close,
                    type          = order_type,
                    stopPrice     = self._rp(stop_price),
                    closePosition = "true",
                    timeInForce   = "GTE_GTC",
                )
                return True, resp.get("algoId") or resp.get("orderId")
            except Exception as exc:
                if "-4130" in str(exc):
                    return True, None   # order already exists on exchange
                return False, None

        # ── Check SL ──────────────────────────────────────────────────────────
        sl_ok = _order_alive(pos.sl_order_id)
        if not sl_ok:
            for attempt in range(1, 4):
                ok, new_id = _place_order("STOP_MARKET", pos.sl_price)
                if ok:
                    if new_id:
                        pos.sl_order_id = new_id
                        _log(f"[guard] SL re-placed  id={new_id}  stop={pos.sl_price:.2f}")
                    # "confirmed alive" via -4130: silent, no log spam
                    sl_ok = True
                    break
                _log(f"[guard] SL re-place FAILED att={attempt}  stop={pos.sl_price:.2f}")

        # ── Check TP ──────────────────────────────────────────────────────────
        tp_ok = _order_alive(pos.tp_order_id)
        if not tp_ok:
            for attempt in range(1, 4):
                ok, new_id = _place_order("TAKE_PROFIT_MARKET", pos.tp_price)
                if ok:
                    if new_id:
                        pos.tp_order_id = new_id
                        _log(f"[guard] TP re-placed  id={new_id}  stop={pos.tp_price:.2f}")
                    tp_ok = True
                    break
                _log(f"[guard] TP re-place FAILED att={attempt}  stop={pos.tp_price:.2f}")

        if sl_ok and tp_ok:
            return True

        _log(f"[guard] FAILED after 3 att – emergency close @ {current_price:.2f}")
        return False

    def verify_protection(self, current_price: float) -> bool:
        """Called by background guard task every 5 seconds."""
        if self.position is None:
            return True
        try:
            account = self._client.futures_account()
            for p in account.get("positions", []):
                if p["symbol"] == self.symbol:
                    if abs(float(p.get("positionAmt", 0))) < 1e-6:
                        self._exchange_closed = True
                        return True
                    break
        except Exception as exc:
            _log(f"[bg_guard] Position poll failed: {exc}")
        ok = self._check_guard_orders(current_price)
        if not ok:
            self._guard_failed = True
            _log(f"[bg_guard] Guard FAILED @ {current_price:.2f}")
        return ok

    def check_exits(self, high: float, low: float) -> tuple[Optional[str], float]:
        """
        Called by ExecutionEngine each bar. Relies on on_bar() having already
        polled Binance. Returns (reason, exit_price) if position was closed.
        """
        if self.position is None:
            return None, 0.0
        if self._guard_failed:
            return "GUARD_FAIL", high
        if not self._exchange_closed:
            return None, 0.0
        return self._detect_exit()

    def _detect_exit(self) -> tuple[str, float]:
        """Position is gone on exchange — get exit price from recent trades."""
        pos = self.position
        # If our limit close order was confirmed filled via on_bar polling
        if self.pending_close_price != 0.0 and self._pending_close_order_id is None:
            fill_price = self.pending_close_price
            _log(f"[exit] detected LIMIT_CLOSE @ {fill_price:.2f}")
            return "LIMIT_CLOSE", fill_price
        try:
            trades = self._client.futures_account_trades(symbol=self.symbol, limit=5)
            if trades:
                exit_p = float(trades[-1]["price"])
                reason = ("TP" if (pos.direction ==  1 and exit_p >= pos.tp_price)
                               or (pos.direction == -1 and exit_p <= pos.tp_price)
                          else "SL")
                _log(f"[exit] detected {reason} @ {exit_p:.2f}")
                return reason, exit_p
        except Exception as exc:
            print(f"[Binance] trade fetch failed: {exc}")

        return "SL", pos.sl_price

    def close_trade(self, price: float, timestamp: str, reason: str) -> RealTrade:
        if self.position is None:
            raise RuntimeError("No open position to close.")

        pos = self.position

        if reason in ("TIME", "FORCE", "GUARD_FAIL"):
            import time as _time
            side_close  = "SELL" if pos.direction == 1 else "BUY"
            max_attempts = 120   # retry every 5 s for up to 10 min
            market_closed = False
            for att in range(1, max_attempts + 1):
                try:
                    resp = self._client.futures_create_order(
                        symbol     = self.symbol,
                        side       = side_close,
                        type       = "MARKET",
                        quantity   = pos.size,
                        reduceOnly = "true",
                    )
                    fill = float(resp.get("avgPrice") or 0)
                    if fill:
                        price = fill
                    _log(f"[close] Market close OK @ {price:.2f}  (attempt {att})")
                    market_closed = True
                    break
                except Exception as exc:
                    if "-2022" in str(exc) or "reduceOnly" in str(exc).lower():
                        _log(f"[close] Position already closed on exchange (SL/TP fired).")
                        market_closed = True
                        break
                    _log(f"[close] Attempt {att}/{max_attempts} FAILED: {exc}"
                          f"  – retrying in 5 s ...")
                    _time.sleep(5)
            if not market_closed:
                _log(f"[close] CRITICAL: Could not close after 10 min – manual action required!")
        # LIMIT_CLOSE: position already closed by limit order — no market order needed

        self._cancel_all_orders()
        self.clear_pending_close()

        exit_fee = self.maker_fee if reason in ("TP", "SL", "LIMIT_CLOSE") else self.taker_fee
        raw_pnl  = (price - pos.entry_price) * pos.direction * pos.size
        cost_pnl = pos.entry_price * pos.size * (self.maker_fee + exit_fee)
        net_pnl  = raw_pnl - cost_pnl

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
        self.position         = None
        self._cooldown_rem    = self.cooldown_bars
        self._exchange_closed = False
        print(f"[CLOSE] {reason:5s} @ {price:.2f}  PnL={net_pnl:+.2f}$")
        return trade

    def _cancel_all_orders(self) -> None:
        try:
            self._client.futures_cancel_all_open_orders(symbol=self.symbol)
            print("[Binance] All open orders cancelled.")
        except Exception as exc:
            print(f"[Binance] cancel_all_orders failed: {exc}")

    # ── Bar update ────────────────────────────────────────────────────────────
    def on_bar(self, price: float) -> None:
        self.bar_count += 1
        if self._cooldown_rem > 0:
            self._cooldown_rem -= 1
        self.equity_curve.append(self.mark_to_market(price))
        self._exchange_closed = False
        self._guard_failed    = False

        # ── Poll pending limit entry order ────────────────────────────────────
        if self.pending_dir != 0 and self._pending_entry_order_id:
            try:
                order  = self._client.futures_get_order(
                    symbol=self.symbol, orderId=self._pending_entry_order_id)
                status    = order.get("status")
                filled_qty = float(order.get("executedQty") or 0)
                if status == "FILLED" or (status == "PARTIALLY_FILLED" and filled_qty > 0):
                    fill_p = float(order.get("avgPrice") or self.pending_price)
                    self.pending_price       = fill_p
                    self._pending_qty        = self._rq(filled_qty)  # use actual filled qty
                    self._limit_entry_filled = True
                    if status == "PARTIALLY_FILLED":
                        # Cancel unfilled remainder and proceed with what filled
                        try:
                            self._client.futures_cancel_order(
                                symbol=self.symbol, orderId=self._pending_entry_order_id)
                        except Exception:
                            pass
                        self._pending_entry_order_id = None
                        _log(f"[PENDING] Limit entry PARTIALLY FILLED {filled_qty} @ {fill_p:.2f} — remainder cancelled")
                    else:
                        _log(f"[PENDING] Limit entry FILLED {filled_qty} @ {fill_p:.2f}")
                elif status in ("CANCELED", "EXPIRED", "REJECTED"):
                    _log(f"[PENDING] Limit entry {status} — clearing")
                    self.cancel_pending_order()
            except Exception as exc:
                print(f"[Binance] pending entry poll FAILED: {exc}")

        # ── Poll pending limit close order ────────────────────────────────────
        if self.pending_close_price != 0.0 and self._pending_close_order_id:
            try:
                order  = self._client.futures_get_order(
                    symbol=self.symbol, orderId=self._pending_close_order_id)
                status = order.get("status")
                if status == "FILLED":
                    fill_p = float(order.get("avgPrice") or self.pending_close_price)
                    self.pending_close_price     = fill_p   # actual fill price
                    self._pending_close_order_id = None     # signals _detect_exit to use LIMIT_CLOSE
                    print(f"[PENDING CLOSE] Limit close FILLED @ {fill_p:.2f}")
                elif status in ("CANCELED", "EXPIRED", "REJECTED"):
                    print(f"[PENDING CLOSE] Order {status} — clearing")
                    self._pending_close_order_id = None
            except Exception as exc:
                print(f"[Binance] pending close poll FAILED: {exc}")

        if self.position is not None:
            # 1. Poll position size
            try:
                account = self._client.futures_account()
                for p in account.get("positions", []):
                    if p["symbol"] == self.symbol:
                        amt = float(p["positionAmt"])
                        if abs(amt) < 1e-6:
                            _log("[pos] Position closed on exchange")
                            self._exchange_closed = True
                        break
            except Exception as exc:
                _log(f"[pos] Poll failed: {exc}")

            # 2. Guard: verify SL/TP orders still live (only if position still open)
            if not self._exchange_closed:
                if not self._check_guard_orders(price):
                    self._guard_failed = True

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