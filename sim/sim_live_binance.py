"""
sim/sim_live_binance.py – Live paper-trading simulation via Binance WebSocket.

Connects to Binance public WebSocket (no API key needed),
receives closed 1m candles, computes features, predicts, and trades.

Usage
-----
    python sim/sim_live_binance.py
    python sim/sim_live_binance.py --symbol ETHUSDT
    python sim/sim_live_binance.py --config config.yaml

Ctrl+C to stop; final summary and portfolio log are written on exit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

import urllib.request as _urllib_req

import features as feat_mod
from sim.portfolio       import Portfolio
from sim.execution       import ExecutionEngine
from sim.binance_ws_feed import BinanceWSFeed


# ── ANSI colours ──────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"

def _clr(text, *codes) -> str:
    return "".join(codes) + str(text) + C.RESET


# ─────────────────────────────────────────────────────────────────────────────
def _load_artifacts(art_dir: str):
    from pathlib import Path
    p = Path(art_dir)
    model = xgb.Booster()
    model.load_model(str(p / "xgb_model.json"))
    with open(p / "feature_columns.json") as f:
        feat_cols = json.load(f)
    return model, feat_cols


def _prewarm_sync(feat_engine, symbol: str, buf_size: int) -> int:
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={symbol}&interval=1m&limit={buf_size + 1}"
    )
    print(f"[prewarm] Fetching {buf_size} recent 1m candles from Binance ({symbol}) ...")
    try:
        with _urllib_req.urlopen(url, timeout=15) as resp:
            klines = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"[prewarm] WARNING: REST fetch failed ({exc}). Warmup will happen live.")
        return 0

    klines = klines[:-1]  # drop open candle

    for k in klines:
        ts  = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
        row = pd.Series({
            "open":          float(k[1]),
            "high":          float(k[2]),
            "low":           float(k[3]),
            "close":         float(k[4]),
            "volume":        float(k[5]),
            "taker_buy_vol": float(k[9]),
        }, name=ts)
        feat_engine.update(row)

    n = len(klines)
    print(f"[prewarm] Done  {n} bars loaded. Feature engine ready.")
    return n


async def _prewarm_from_binance(feat_engine, symbol: str, buf_size: int) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _prewarm_sync, feat_engine, symbol, buf_size)


def _predict(model: xgb.Booster, feat_cols: list[str],
             feat_row: pd.Series) -> np.ndarray:
    arr = feat_row.values.reshape(1, -1).astype(np.float32)
    dm  = xgb.DMatrix(arr, feature_names=feat_cols)
    return model.predict(dm)[0]


# ─────────────────────────────────────────────────────────────────────────────
async def run_live(cfg: dict, symbol: str) -> None:
    sim_cfg  = cfg["simulation"]["live"]
    log_file = sim_cfg.get("log_file", "sim/logs/live_log.jsonl")
    art_dir  = cfg["training"]["artifacts_dir"]
    tc       = cfg["trading"]
    lc       = cfg["labels"]

    print("Loading model artifacts ...")
    model, feat_cols = _load_artifacts(art_dir)

    cost_rt = (lc["maker_fee"] + lc["taker_fee"] +
               lc["slippage"]  + lc["spread"]) * 2
    portfolio = Portfolio(
        initial_capital = tc["initial_capital"],
        cost_rt         = cost_rt,
        cooldown_bars   = tc["cooldown"],
    )
    engine      = ExecutionEngine(cfg, portfolio, log_file)
    feat_engine = feat_mod.FeatureEngine(cfg, feature_cols=feat_cols)

    await _prewarm_from_binance(feat_engine, symbol, cfg["features"]["live_buffer"])

    ws_url = sim_cfg.get("ws_url", "wss://stream.binance.com:9443/ws")
    feed   = BinanceWSFeed(symbol=symbol, ws_url=ws_url)

    # ── Banner ────────────────────────────────────────────────────────────────
    _BW  = 66
    _BR  = lambda s: _clr(s, C.CYAN, C.BOLD)
    _btop = _BR("╔" + "═" * _BW + "╗")
    _bmid = _BR("╠" + "═" * _BW + "╣")
    _bbot = _BR("╚" + "═" * _BW + "╝")
    _bl   = _BR("║")
    lev   = int(tc.get("position_size_pct", 10))
    print(f"\n {_btop}")
    print(f" {_bl}  {_clr('PAPER TRADING', C.WHITE, C.BOLD)}  "
          f"{_clr('·', C.DIM)}  {_clr(symbol, C.CYAN, C.BOLD)}  "
          f"{_clr('·', C.DIM)}  {_clr('BINANCE (simulated)', C.YELLOW, C.BOLD)}")
    print(f" {_bmid}")
    print(f" {_bl}  {_clr('Leverage', C.GRAY)} {_clr(str(lev)+'x', C.YELLOW, C.BOLD)}"
          f"  {_clr('|', C.DIM)}  {_clr('Capital', C.GRAY)} {_clr(f'${portfolio.capital:,.2f}', C.CYAN, C.BOLD)}"
          f"  {_clr('|', C.DIM)}  {_clr('T', C.GRAY)} {_clr(str(tc['T_up']), C.WHITE)}"
          f"  {_clr('|', C.DIM)}  {_clr('SL', C.RED)} {tc.get('sl_pct',0)*100:.2f}%"
          f"  {_clr('/', C.DIM)}  {_clr('TP', C.GREEN)} {tc.get('tp_pct',0)*100:.2f}%")
    print(f" {_bl}  {_clr('Log', C.GRAY)} {_clr(log_file, C.BLUE)}"
          f"  {_clr('|', C.DIM)}  {_clr('Ctrl+C', C.YELLOW)} {_clr('to stop gracefully', C.GRAY)}")
    print(f" {_bbot}\n")

    last_candle = None
    last_ts     = None
    last_price  = 0.0

    try:
        async for candle in feed:
            last_ts     = candle.name
            last_candle = candle
            price       = float(candle["close"])
            last_price  = price

            feat_row = feat_engine.update(candle)
            if feat_row is None:
                remaining = feat_engine.min_warmup - len(feat_engine.buffer)
                print(f"  [{last_ts}] Warming up... ({remaining} bars remaining)")
                continue

            probs  = _predict(model, feat_cols, feat_row)
            event  = engine.on_bar(feat_row, probs, last_ts, price)

            p_down, _, p_up = probs
            equity  = portfolio.mark_to_market(price)
            balance = portfolio.capital
            pos     = portfolio.position

            # ── Per-bar status line ───────────────────────────────────────────
            unreal = equity - balance
            _D = _clr(" | ", C.GRAY, C.DIM)

            if pos is None:
                pos_str = _clr("o FLAT ", C.GRAY, C.DIM)
            elif pos.direction == 1:
                pos_str = _clr("* LONG ", C.GREEN, C.BOLD)
            else:
                pos_str = _clr("* SHORT", C.RED, C.BOLD)

            pup_col = C.GREEN if p_up   >= engine.T_up   else C.GRAY
            pdn_col = C.RED   if p_down >= engine.T_down else C.DIM
            pup_str = _clr(f"^{p_up:.3f}", pup_col,
                           C.BOLD if p_up >= engine.T_up else "")
            pdn_str = _clr(f"v{p_down:.3f}", pdn_col,
                           C.BOLD if p_down >= engine.T_down else "")

            bal_str = _clr(f"${balance:>9,.2f}", C.CYAN, C.BOLD)
            if pos is not None:
                unreal_col = C.GREEN if unreal >= 0 else C.RED
                fin_str = bal_str + "  " + _clr(f"{unreal:+.2f}", unreal_col, C.BOLD)
            else:
                fin_str = bal_str

            # ── Condition display ─────────────────────────────────────────────
            sd  = engine.last_skip_data
            AND = _clr(" AND ", C.YELLOW, C.BOLD)
            OR  = _clr(" OR ",  C.MAGENTA, C.BOLD)

            def _cv(val, ok, label="") -> str:
                col = C.GREEN if ok else C.RED
                return _clr(f"{label}{val}", col, C.BOLD)

            if sd.get("status") == "in_pos":
                cond_str = _clr("* IN POSITION", C.CYAN, C.BOLD)
            elif sd.get("status") == "cooldown":
                cond_str = _clr(f"~ COOLDOWN {sd['bars']}b", C.YELLOW, C.BOLD)
            elif sd.get("status") == "atr0":
                cond_str = _clr("! ATR=0", C.RED, C.BOLD)
            elif sd.get("status") == "eval":
                long_s  = (_cv(f"{sd['p_up']:.3f}", sd['p_up_ok'], "pu:")
                           + AND + _cv(sd['rh'], sd['rh_ok'], "rh:"))
                short_s = (_cv(f"{sd['p_dn']:.3f}", sd['p_dn_ok'], "pd:")
                           + AND + _cv(sd['rl'], sd['rl_ok'], "rl:"))
                shared  = (_cv("sq", sd['sq_ok']) + AND + _cv("ema", sd['ema_ok']))
                signal  = f"({long_s})" + OR + f"({short_s})"
                cond_str = f"({signal})" + AND + f"({shared})"
            else:
                cond_str = _clr(engine.last_skip_reason, C.DIM)

            ts_str    = _clr(str(last_ts)[11:16], C.GRAY)
            price_str = _clr(f"{price:>10,.2f}", C.WHITE, C.BOLD)

            print(f" {ts_str}{_D}{price_str}{_D}{pup_str} {pdn_str}{_D}{pos_str}{_D}{fin_str}  {_D}  {cond_str}")

            # ── Trade events ──────────────────────────────────────────────────
            if event:
                ev_type = event.get("event", "")

                if ev_type == "OPEN" and pos is not None:
                    size_usd = pos.size * pos.entry_price
                    d        = event.get("direction", "")
                    d_col    = C.GREEN if d == "LONG" else C.RED
                    d_sym    = "^" if d == "LONG" else "v"
                    _etop = _clr("\u250c" + "\u2500" * 60 + "\u2510", d_col, C.BOLD)
                    _ebot = _clr("\u2514" + "\u2500" * 60 + "\u2518", d_col, C.BOLD)
                    _el   = _clr("\u2502", d_col, C.BOLD)
                    print(f"\n {_etop}")
                    print(f" {_el}  {_clr(d_sym + ' ENTRY  ' + d, d_col, C.BOLD)}"
                          f"  @  {_clr(f'{pos.entry_price:,.2f}', C.WHITE, C.BOLD)}"
                          f"  {_clr('|', C.DIM)}  {_clr(f'{pos.size:.5f} {symbol[:3]}  ${size_usd:,.2f}', C.GRAY)}")
                    print(f" {_el}  {_clr('SL', C.RED, C.BOLD)}  {_clr(f'{pos.sl_price:,.2f}', C.RED, C.BOLD)}"
                          f"  {_clr(f'( -{abs(pos.entry_price - pos.sl_price):.2f} pts )', C.RED, C.DIM)}")
                    print(f" {_el}  {_clr('TP', C.GREEN, C.BOLD)}  {_clr(f'{pos.tp_price:,.2f}', C.GREEN, C.BOLD)}"
                          f"  {_clr(f'( +{abs(pos.tp_price - pos.entry_price):.2f} pts )', C.GREEN, C.DIM)}")
                    print(f" {_ebot}\n")

                elif ev_type == "CLOSE":
                    pnl    = event.get("net_pnl", 0)
                    reason = event.get("reason", "")
                    exit_p = event.get("exit_price", price)
                    trades = portfolio.trade_log
                    nt     = len(trades)
                    wins   = sum(1 for t in trades if t.net_pnl > 0)
                    wr     = f"{wins/nt*100:.0f}%" if nt else "n/a"
                    pnl_col  = C.GREEN if pnl >= 0 else C.RED
                    rsn_col  = C.GREEN if reason == "TP" else (C.RED if reason == "SL" else C.YELLOW)
                    rsn_sym  = "+" if reason == "TP" else ("-" if reason == "SL" else "~")
                    wr_col   = C.GREEN if nt and wins/nt >= 0.6 else C.YELLOW
                    pnl_sign = "+" if pnl >= 0 else ""
                    _xtop = _clr("\u250c" + "\u2500" * 60 + "\u2510", rsn_col, C.BOLD)
                    _xbot = _clr("\u2514" + "\u2500" * 60 + "\u2518", rsn_col, C.BOLD)
                    _xl   = _clr("\u2502", rsn_col, C.BOLD)
                    print(f"\n {_xtop}")
                    print(f" {_xl}  {_clr(rsn_sym + ' EXIT  ' + reason, rsn_col, C.BOLD)}"
                          f"  @  {_clr(f'{exit_p:,.2f}', C.WHITE, C.BOLD)}"
                          f"  {_clr('|', C.DIM)}  PnL  {_clr(f'{pnl_sign}{pnl:,.2f}', pnl_col, C.BOLD)}")
                    print(f" {_xl}  {_clr('Balance', C.GRAY)} {_clr(f'${portfolio.capital:,.2f}', C.CYAN, C.BOLD)}"
                          f"  {_clr('|', C.DIM)}  trades {_clr(str(nt), C.WHITE, C.BOLD)}"
                          f"  {_clr('|', C.DIM)}  WR {_clr(wr, wr_col, C.BOLD)}")
                    print(f" {_xbot}\n")

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await feed.stop()

        _SW   = 60
        _stop = _clr("╔" + "═" * _SW + "╗", C.YELLOW, C.BOLD)
        _smid = _clr("╠" + "═" * _SW + "╣", C.YELLOW, C.BOLD)
        _sbot = _clr("╚" + "═" * _SW + "╝", C.YELLOW, C.BOLD)
        _sl   = _clr("║", C.YELLOW, C.BOLD)
        print(f"\n {_stop}")
        if portfolio.position is not None and last_candle is not None:
            d     = "LONG" if portfolio.position.direction == 1 else "SHORT"
            d_col = C.GREEN if d == "LONG" else C.RED
            print(f" {_sl}  {_clr('SHUTDOWN', C.YELLOW, C.BOLD)}  "
                  f"closing {_clr(d, d_col, C.BOLD)}"
                  f"  {_clr('|', C.DIM)}  entry {_clr(f'{portfolio.position.entry_price:,.2f}', C.WHITE)}"
                  f"  last {_clr(f'{last_price:,.2f}', C.WHITE)}")
            try:
                engine.force_close(last_price, last_ts or "shutdown")
                print(f" {_sl}  {_clr('Position closed', C.GREEN, C.BOLD)}")
            except Exception as ce:
                print(f" {_sl}  {_clr('force_close ERROR', C.RED)}  {ce}")
        else:
            print(f" {_sl}  {_clr('No open position', C.GRAY)}")

        engine.close_log()

        summary = portfolio.summary()
        print(f" {_smid}")
        print(f" {_sl}  {_clr('SESSION SUMMARY', C.WHITE, C.BOLD)}")
        print(f" {_smid}")
        for k, v in summary.items():
            val_col = (C.GREEN if isinstance(v, (int, float)) and v > 0
                       else (C.RED if isinstance(v, (int, float)) and v < 0 else C.WHITE))
            print(f" {_sl}  {_clr(f'{k:<22}', C.GRAY)} {_clr(str(v), val_col, C.BOLD)}")
        print(f" {_sbot}")

        from pathlib import Path
        art = Path(art_dir)
        portfolio.save(str(art / "live_portfolio.json"))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # Enable ANSI colour processing on Windows 10+
        os.system("")
        # Force UTF-8 output so box-drawing chars render correctly
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)

    parser = argparse.ArgumentParser(description="Live Binance paper-trading sim")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    symbol = (args.symbol or
              cfg["simulation"]["live"].get("symbol",
              cfg["data"]["symbol"])).upper()

    asyncio.run(run_live(cfg, symbol))


if __name__ == "__main__":
    main()
