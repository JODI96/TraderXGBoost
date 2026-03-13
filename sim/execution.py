"""
sim/execution.py – Signal generation and order routing for the paper-trading sim.

This module bridges the model predictions and the Portfolio:
  1. Receives a feature vector (pd.Series) + probability vector.
  2. Applies trading rules to determine if a trade should be entered/exited.
  3. Calls portfolio.open_trade / portfolio.close_trade.
  4. Logs every bar and every trade event to a JSONL file.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from sim.portfolio import Portfolio, Trade


# ─────────────────────────────────────────────────────────────────────────────
class ExecutionEngine:
    """
    Stateless signal evaluator + order router.

    Parameters
    ----------
    cfg       : full config dict
    portfolio : Portfolio instance (holds state)
    log_path  : path to JSONL log file (appended per bar)
    """

    def __init__(self, cfg: dict, portfolio: Portfolio, log_path: str):
        self.cfg   = cfg
        self.port  = portfolio
        self.log_path = log_path

        tc = cfg["trading"]
        lc = cfg["labels"]
        self.T_up      = tc["T_up"]
        self.T_down    = tc["T_down"]
        self.d_max     = tc["d_max_atr"]
        self.sl_atr    = tc["sl_atr"]
        self.tp_atr    = tc["tp_atr"]
        self.sl_pct    = tc.get("sl_pct")
        self.tp_pct    = tc.get("tp_pct")
        self.time_stop = tc["time_stop"]
        self.pos_pct   = tc["position_size_pct"]
        self.req_sq       = tc.get("require_squeeze", False)
        self.min_ema9_21  = tc.get("min_ema9_21_diff", -999.0)
        self.L_range      = lc.get("L_range", 20)

        self.cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
                        lc["slippage"]  + lc["spread"]) * 2

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self._log_fh = open(log_path, "a", buffering=1)   # line-buffered

        self.bar_count = 0
        self.last_skip_reason = ""
        self.last_skip_data:  dict = {}

    # ── Per-bar entry point ───────────────────────────────────────────────────
    def on_bar(
        self,
        features: pd.Series,
        probs:    np.ndarray,     # shape (3,): [p_down, p_no_break, p_up]
        timestamp,
        price:    float,
    ) -> Optional[dict]:
        """
        Process one completed 1m bar.
        Returns a trade event dict if a trade was opened/closed, else None.
        """
        self.bar_count += 1
        self.port.on_bar(price)

        # Extract feature scalars (safe get with fallback)
        def _f(col, default=np.nan):
            return float(features.get(col, default)) \
                   if hasattr(features, "get") else default

        atr          = _f("atr_short", _f("atr", 0.0))
        high         = _f("high",  price)
        low          = _f("low",   price)
        dist_rh      = _f(f"dist_rh_{self.L_range}", np.nan)
        dist_rl      = _f(f"dist_rl_{self.L_range}", np.nan)
        squeeze      = int(_f("squeeze_flag", 0))
        ema9_21_diff = _f("ema9_21_diff", 999.0)

        p_down, p_no_break, p_up = float(probs[0]), float(probs[1]), float(probs[2])
        pred_class = int(np.argmax(probs))

        trade_event = None

        # ── 1. Check SL / TP exit (intrabar high/low) ────────────────────────
        if self.port.position is not None:
            reason, exit_p = self.port.check_exits(high, low)
            if reason:
                trade = self.port.close_trade(exit_p, timestamp, reason)
                trade_event = {"event": "CLOSE", "reason": reason,
                               **self._trade_dict(trade)}

        # ── 2. Check time-stop exit (close price) ────────────────────────────
        if self.port.position is not None:
            time_held = self.port.bar_count - self.port.position.entry_bar
            if time_held >= self.time_stop:
                trade = self.port.close_trade(price, timestamp, "TIME")
                trade_event = {"event": "CLOSE", "reason": "TIME",
                               **self._trade_dict(trade)}

        # ── 3. Entry logic ────────────────────────────────────────────────────
        if not self.port.can_enter:
            if self.port.position is not None:
                self.last_skip_reason = "in_pos"
                self.last_skip_data   = {"status": "in_pos"}
            else:
                self.last_skip_reason = f"cooldown={self.port._cooldown_rem}"
                self.last_skip_data   = {"status": "cooldown",
                                         "bars": self.port._cooldown_rem}
        elif atr <= 0:
            self.last_skip_reason = "atr=0"
            self.last_skip_data   = {"status": "atr0"}
        else:
            sq_ok  = (not self.req_sq) or (squeeze == 1)
            ema_ok = ema9_21_diff >= self.min_ema9_21

            rh_ok  = (not np.isnan(dist_rh)) and dist_rh <= self.d_max
            rl_ok  = (not np.isnan(dist_rl)) and dist_rl <= self.d_max
            rh_str = f"{dist_rh:.2f}" if not np.isnan(dist_rh) else "nan"
            rl_str = f"{dist_rl:.2f}" if not np.isnan(dist_rl) else "nan"

            self.last_skip_data = {
                "status":   "eval",
                "p_up":     p_up,    "p_up_ok":  p_up   >= self.T_up,
                "rh":       rh_str,  "rh_ok":    rh_ok,
                "p_dn":     p_down,  "p_dn_ok":  p_down >= self.T_down,
                "rl":       rl_str,  "rl_ok":    rl_ok,
                "sq_ok":    sq_ok,   "ema_ok":   ema_ok,
            }
            _Y = lambda v: "(Y)" if v else "(N)"
            self.last_skip_reason = (
                f"[(p_up={p_up:.3f} {_Y(p_up>=self.T_up)} AND rh={rh_str} {_Y(rh_ok)})"
                f" OR (p_dn={p_down:.3f} {_Y(p_down>=self.T_down)} AND rl={rl_str} {_Y(rl_ok)})]"
                f" AND sq {_Y(sq_ok)} AND ema {_Y(ema_ok)}"
            )

            # LONG: up-break anticipated
            if (p_up >= self.T_up and
                    not np.isnan(dist_rh) and dist_rh <= self.d_max and
                    sq_ok and ema_ok):
                self.port.open_trade(
                    direction = 1,
                    price     = price,
                    atr       = atr,
                    sl_atr    = self.sl_atr,
                    tp_atr    = self.tp_atr,
                    pos_pct   = self.pos_pct,
                    timestamp = timestamp,
                    sl_pct    = self.sl_pct,
                    tp_pct    = self.tp_pct,
                )
                self.last_skip_reason = "ENTERED_LONG"
                trade_event = {"event": "OPEN", "direction": "LONG",
                               "price": price, "p_up": p_up,
                               "sl": self.port.position.sl_price,
                               "tp": self.port.position.tp_price}

            # SHORT: down-break anticipated
            elif (p_down >= self.T_down and rl_ok and sq_ok and ema_ok):
                self.port.open_trade(
                    direction = -1,
                    price     = price,
                    atr       = atr,
                    sl_atr    = self.sl_atr,
                    tp_atr    = self.tp_atr,
                    pos_pct   = self.pos_pct,
                    timestamp = timestamp,
                    sl_pct    = self.sl_pct,
                    tp_pct    = self.tp_pct,
                )
                self.last_skip_reason = "ENTERED_SHORT"
                trade_event = {"event": "OPEN", "direction": "SHORT",
                               "price": price, "p_down": p_down,
                               "sl": self.port.position.sl_price,
                               "tp": self.port.position.tp_price}

        # ── Log bar ───────────────────────────────────────────────────────────
        bar_log = {
            "bar":       self.bar_count,
            "ts":        str(timestamp),
            "price":     round(price, 4),
            "p_down":    round(p_down,     4),
            "p_no_brk":  round(p_no_break, 4),
            "p_up":      round(p_up,       4),
            "pred":      pred_class,
            "atr":       round(atr, 4),
            "dist_rh":   round(dist_rh, 3) if not np.isnan(dist_rh) else None,
            "dist_rl":   round(dist_rl, 3) if not np.isnan(dist_rl) else None,
            "squeeze":   squeeze,
            "in_pos":    not self.port.is_flat,
            "equity":    round(self.port.mark_to_market(price), 2),
            "trade":     trade_event,
        }
        self._log_fh.write(json.dumps(bar_log) + "\n")

        return trade_event

    # ── Force-close on shutdown ───────────────────────────────────────────────
    def force_close(self, price: float, timestamp) -> Optional[Trade]:
        if self.port.position is not None:
            trade = self.port.close_trade(price, timestamp, "FORCE")
            self._log_fh.write(json.dumps(
                {"event": "FORCE_CLOSE", **self._trade_dict(trade)}) + "\n")
            return trade
        return None

    def close_log(self) -> None:
        self._log_fh.flush()
        self._log_fh.close()

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _trade_dict(t: Trade) -> dict:
        return {
            "direction":   t.direction,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "net_pnl":     t.net_pnl,
            "exit_reason": t.exit_reason,
        }
