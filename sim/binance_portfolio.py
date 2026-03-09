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

import json
import math
from dataclasses import dataclass, asdict
from typing import Optional


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
        self.cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
                        lc["slippage"]  + lc["spread"]) * 2
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

    def _rq(self, qty: float) -> float:
        return round(qty, self._qty_precision)

    def _rp(self, price: float) -> str:
        return f"{price:.{self._price_precision}f}"

    # ── State properties ──────────────────────────────────────────────────────
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

        notional = math.floor(self.capital) * pos_pct
        qty      = self._rq(notional / price)

        side       = "BUY"  if direction == 1 else "SELL"
        side_close = "SELL" if direction == 1 else "BUY"

        try:
            # 1. Market entry – get real fill price first
            entry_resp = self._client.futures_create_order(
                symbol   = self.symbol,
                side     = side,
                type     = "MARKET",
                quantity = qty,
            )
            fill_price = float(entry_resp.get("avgPrice") or 0) or price
            print(f"[OPEN]  {side:4s} {qty} {self.symbol} @ {fill_price:.2f}  "
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
                entry_bar    = self.bar_count,
                atr_at_entry = atr,
                sl_order_id  = sl_order_id,
                tp_order_id  = tp_order_id,
            )

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
                try:
                    self._client.futures_create_order(
                        symbol     = self.symbol,
                        side       = side_close2,
                        type       = "MARKET",
                        quantity   = qty,
                        reduceOnly = "true",
                    )
                    print(f"  Position closed (guard on entry)")
                except Exception as ce:
                    print(f"  Close FAILED: {ce}")
                self.position = None

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
            for attempt in range(1, 3):
                ok, new_id = _place_order("STOP_MARKET", pos.sl_price)
                if ok:
                    if new_id:
                        pos.sl_order_id = new_id
                        print(f"[guard] SL re-placed (attempt {attempt})  "
                              f"id={new_id}  stop={pos.sl_price:.2f}")
                    else:
                        print(f"[guard] SL confirmed alive (attempt {attempt})  "
                              f"id={pos.sl_order_id}  stop={pos.sl_price:.2f}")
                    sl_ok = True
                    break
                print(f"[guard] SL re-place failed (attempt {attempt})  "
                      f"stop={pos.sl_price:.2f}")

        # ── Check TP ──────────────────────────────────────────────────────────
        tp_ok = _order_alive(pos.tp_order_id)
        if not tp_ok:
            for attempt in range(1, 3):
                ok, new_id = _place_order("TAKE_PROFIT_MARKET", pos.tp_price)
                if ok:
                    if new_id:
                        pos.tp_order_id = new_id
                        print(f"[guard] TP re-placed (attempt {attempt})  "
                              f"id={new_id}  stop={pos.tp_price:.2f}")
                    else:
                        print(f"[guard] TP confirmed alive (attempt {attempt})  "
                              f"id={pos.tp_order_id}  stop={pos.tp_price:.2f}")
                    tp_ok = True
                    break
                print(f"[guard] TP re-place failed (attempt {attempt})  "
                      f"stop={pos.tp_price:.2f}")

        if sl_ok and tp_ok:
            return True

        print(f"[guard] FAILED after 2 attempts – closing position "
              f"at {current_price:.2f}  (sl_ok={sl_ok} tp_ok={tp_ok})")
        return False

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
        try:
            trades = self._client.futures_account_trades(symbol=self.symbol, limit=5)
            if trades:
                exit_p = float(trades[-1]["price"])
                reason = ("TP" if (pos.direction ==  1 and exit_p >= pos.tp_price)
                               or (pos.direction == -1 and exit_p <= pos.tp_price)
                          else "SL")
                print(f"[exit] detected {reason} @ {exit_p:.2f}")
                return reason, exit_p
        except Exception as exc:
            print(f"[Binance] trade fetch failed: {exc}")

        return "SL", pos.sl_price

    def close_trade(self, price: float, timestamp: str, reason: str) -> RealTrade:
        if self.position is None:
            raise RuntimeError("No open position to close.")

        pos = self.position

        if reason in ("TIME", "FORCE", "GUARD_FAIL"):
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
                print(f"[Binance] market close failed: {exc}")

        self._cancel_all_orders()

        raw_pnl  = (price - pos.entry_price) * pos.direction * pos.size
        cost_pnl = pos.entry_price * pos.size * self.cost_rt
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
        if self.position is not None:
            # 1. Poll position size
            try:
                account = self._client.futures_account()
                for p in account.get("positions", []):
                    if p["symbol"] == self.symbol:
                        amt = float(p["positionAmt"])
                        print(f"[pos check] positionAmt={amt}")
                        if abs(amt) < 1e-6:
                            print("[pos] Position closed on exchange")
                            self._exchange_closed = True
                        break
            except Exception as exc:
                print(f"[pos] Poll failed: {exc}")

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