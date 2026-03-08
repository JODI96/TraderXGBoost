"""
sim/portfolio.py – Portfolio state and P&L tracking for the paper-trading sim.

Designed for single-position trading (max_positions = 1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Trade:
    """A single completed trade record."""
    direction:   str        # 'LONG' or 'SHORT'
    entry_ts:    str        # ISO timestamp
    exit_ts:     str        # ISO timestamp
    entry_price: float
    exit_price:  float
    size:        float      # units (base asset)
    raw_pnl:     float
    cost:        float
    net_pnl:     float
    exit_reason: str        # 'SL' | 'TP' | 'TIME' | 'FORCE'
    entry_bar:   int
    exit_bar:    int


@dataclass
class Position:
    """Currently open position."""
    direction:   int        # +1 = long, -1 = short
    entry_price: float
    sl_price:    float
    tp_price:    float
    size:        float
    entry_ts:    str
    entry_bar:   int
    atr_at_entry: float

    def unrealised_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.direction * self.size


@dataclass
class Portfolio:
    """
    Tracks capital, open position, trade log, and equity curve.

    Parameters
    ----------
    initial_capital  : starting capital in USD
    cost_rt          : round-trip cost fraction (fee + slippage both sides)
    cooldown_bars    : bars to wait after a trade closes before next entry
    """
    initial_capital: float = 10_000.0
    cost_rt:         float = 0.0018
    cooldown_bars:   int   = 3

    # Runtime state (not __init__ params)
    capital:      float            = field(init=False)
    position:     Optional[Position] = field(init=False, default=None)
    trade_log:    list[Trade]      = field(init=False, default_factory=list)
    equity_curve: list[float]      = field(init=False, default_factory=list)
    bar_count:    int              = field(init=False, default=0)
    _cooldown_rem: int             = field(init=False, default=0)

    def __post_init__(self):
        self.capital = self.initial_capital

    # ── State queries ─────────────────────────────────────────────────────────
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
        """Current total value including unrealised P&L."""
        if self.position:
            return self.capital + self.position.unrealised_pnl(price)
        return self.capital

    # ── Position management ───────────────────────────────────────────────────
    def open_trade(
        self,
        direction: int,
        price: float,
        atr: float,
        sl_atr: float,
        tp_atr: float,
        pos_pct: float,
        timestamp: str,
        sl_pct: float | None = None,
        tp_pct: float | None = None,
    ) -> None:
        if not self.can_enter:
            raise RuntimeError("Cannot open trade: position open or in cooldown.")
        size = (self.capital * pos_pct) / (price + 1e-9)
        if sl_pct is not None and tp_pct is not None:
            sl_price = price * (1.0 - direction * sl_pct)
            tp_price = price * (1.0 + direction * tp_pct)
        else:
            sl_price = price - direction * sl_atr * atr
            tp_price = price + direction * tp_atr * atr
        self.position = Position(
            direction    = direction,
            entry_price  = price,
            sl_price     = sl_price,
            tp_price     = tp_price,
            size         = size,
            entry_ts     = str(timestamp),
            entry_bar    = self.bar_count,
            atr_at_entry = atr,
        )

    def close_trade(
        self,
        price: float,
        timestamp: str,
        reason: str,
    ) -> Trade:
        if self.position is None:
            raise RuntimeError("No open position to close.")
        pos        = self.position
        raw_pnl    = pos.unrealised_pnl(price)
        cost_pnl   = pos.entry_price * pos.size * self.cost_rt
        net_pnl    = raw_pnl - cost_pnl
        self.capital += net_pnl

        trade = Trade(
            direction    = "LONG" if pos.direction == 1 else "SHORT",
            entry_ts     = pos.entry_ts,
            exit_ts      = str(timestamp),
            entry_price  = pos.entry_price,
            exit_price   = price,
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
        return trade

    # ── Bar update (call once per new closed candle) ──────────────────────────
    def on_bar(self, price: float) -> None:
        self.bar_count += 1
        if self._cooldown_rem > 0:
            self._cooldown_rem -= 1
        self.equity_curve.append(self.mark_to_market(price))

    # ── Exit check ────────────────────────────────────────────────────────────
    def check_exits(self, high: float, low: float) -> tuple[Optional[str], float]:
        """
        Check intrabar SL/TP using the candle's high and low.
        Returns (reason, exit_price) or (None, 0.0).
        SL takes priority when both are hit in the same candle.
        """
        if self.position is None:
            return None, 0.0
        pos = self.position
        if pos.direction == 1:
            if low  <= pos.sl_price: return "SL", pos.sl_price
            if high >= pos.tp_price: return "TP", pos.tp_price
        else:
            if high >= pos.sl_price: return "SL", pos.sl_price
            if low  <= pos.tp_price: return "TP", pos.tp_price
        return None, 0.0

    # ── Summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        pnls = [t.net_pnl for t in self.trade_log]
        if not pnls:
            return {"n_trades": 0, "capital": round(self.capital, 2)}
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p < 0]
        eq   = np.array(self.equity_curve) if self.equity_curve \
               else np.array([self.capital])
        peak = np.maximum.accumulate(eq)
        dd   = ((eq - peak) / (peak + 1e-9) * 100).min()
        return {
            "n_trades":        len(pnls),
            "win_rate_pct":    round(len(wins) / len(pnls) * 100, 1),
            "profit_factor":   round(sum(wins) / (-sum(loss) + 1e-9), 3)
                               if loss else float("inf"),
            "net_pnl":         round(sum(pnls), 2),
            "total_return_pct": round((self.capital - self.initial_capital)
                                       / self.initial_capital * 100, 2),
            "max_drawdown_pct": round(float(dd), 2),
            "final_capital":    round(self.capital, 2),
        }

    def to_dict(self) -> dict:
        return {
            "summary":   self.summary(),
            "trades":    [asdict(t) for t in self.trade_log],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"[portfolio] Saved → {path}")
