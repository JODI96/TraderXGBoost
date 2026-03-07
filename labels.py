"""
labels.py – Barrier-based breakout anticipation labels (1:RR aligned).

Label contract
--------------
  0 = DOWN_BREAK_SOON
  1 = NO_BREAK
  2 = UP_BREAK_SOON

A positive label at time t requires ALL of:
  1. Price is currently INSIDE the range  (pre-breakout filter)
  2. Price breaks the range boundary within H candles  (it IS a breakout)
  3. Take-profit (sl_atr * rr_ratio * ATR beyond entry) is hit within H candles
  4. TP is hit BEFORE SL  (the trade would have been profitable at the target RR)

This directly aligns training with execution: the model learns to predict
breakouts that would have paid out at the configured 1:3 reward/risk.

Fakeout column:
  Price broke the range but SL was hit before (or instead of) TP.
  Useful as a secondary label to explicitly learn to avoid fake breakouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────────────────────────────────────
def compute_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Parameters
    ----------
    df  : 1m OHLCV DataFrame.  Must not contain NaN in OHLCV columns.
    cfg : full config dict.

    Returns
    -------
    DataFrame with columns:
        label          float  (0, 1, 2 or NaN for last H rows / warmup)
        up_fakeout     float  broke up but SL hit before TP
        down_fakeout   float  broke down but SL hit before TP
        range_high     float
        range_low      float
        atr            float
        thresh_high    float
        thresh_low     float
        in_range       float
        tp_up          float  actual TP level for a long entry at close[t]
        tp_down        float  actual TP level for a short entry at close[t]
        sl_up          float  actual SL level for a long entry
        sl_down        float  actual SL level for a short entry
    """
    lc       = cfg["labels"]
    L        = lc["L_range"]        # range lookback
    H        = lc["H_horizon"]      # look-ahead horizon
    k        = lc["k_atr"]          # range breakout multiplier
    L_atr    = lc["L_atr"]          # ATR span
    sl_m     = lc["sl_atr"]         # stop-loss ATR multiplier
    rr       = lc["rr_ratio"]       # reward:risk  (e.g. 3.0 for 1:3)
    tp_m     = sl_m * rr            # take-profit ATR multiplier

    n = len(df)

    # ── Causal ATR ───────────────────────────────────────────────────────────
    prev_c = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_c).abs(),
         (df["low"]  - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=L_atr, min_periods=L_atr, adjust=False).mean()

    # ── Causal range ─────────────────────────────────────────────────────────
    range_high = df["high"].rolling(L, min_periods=L).max()
    range_low  = df["low"].rolling(L,  min_periods=L).min()
    thresh_high = range_high + k * atr
    thresh_low  = range_low  - k * atr

    # ── Pre-breakout filter ───────────────────────────────────────────────────
    in_range = (df["close"] < thresh_high) & (df["close"] > thresh_low)

    # ── Barrier levels (entry assumed at close[t]) ────────────────────────────
    close_v = df["close"].values
    atr_v   = atr.values

    sl_pct = lc.get("sl_pct")
    tp_pct = lc.get("tp_pct")
    if sl_pct is not None and tp_pct is not None:
        # Fixed-percentage barriers: TP/SL in dollar terms is stable across ATR regimes,
        # so fee overhead stays low even during squeeze entries.
        tp_up_v   = close_v * (1.0 + tp_pct)
        sl_up_v   = close_v * (1.0 - sl_pct)
        tp_down_v = close_v * (1.0 - tp_pct)
        sl_down_v = close_v * (1.0 + sl_pct)
    else:
        # ATR-based barriers (legacy fallback)
        tp_up_v   = close_v + tp_m * atr_v
        sl_up_v   = close_v - sl_m * atr_v
        tp_down_v = close_v - tp_m * atr_v
        sl_down_v = close_v + sl_m * atr_v

    # ── Future high/low matrices  shape (n, H) ────────────────────────────────
    # h_shifts[t, i] = high[t + i + 1]  (i = 0 → bar t+1)
    h_shifts = np.stack(
        [df["high"].shift(-i).values for i in range(1, H + 1)], axis=1
    )
    l_shifts = np.stack(
        [df["low"].shift(-i).values  for i in range(1, H + 1)], axis=1
    )

    # ── Range breakout hit matrices ───────────────────────────────────────────
    broke_up   = h_shifts > thresh_high.values[:, None]   # (n, H)
    broke_down = l_shifts < thresh_low.values[:, None]

    # ── Barrier hit matrices ──────────────────────────────────────────────────
    tp_up_hit   = h_shifts >= tp_up_v[:, None]     # long TP touched
    sl_up_hit   = l_shifts <= sl_up_v[:, None]     # long SL touched
    tp_down_hit = l_shifts <= tp_down_v[:, None]   # short TP touched
    sl_down_hit = h_shifts >= sl_down_v[:, None]   # short SL touched

    # ── First-touch step index (0-based; H = "never within horizon") ──────────
    def _first(mat: np.ndarray) -> np.ndarray:
        return np.where(mat.any(axis=1), mat.argmax(axis=1), H)

    first_broke_up   = _first(broke_up)
    first_broke_down = _first(broke_down)
    first_tp_up      = _first(tp_up_hit)
    first_sl_up      = _first(sl_up_hit)
    first_tp_down    = _first(tp_down_hit)
    first_sl_down    = _first(sl_down_hit)

    in_r = in_range.values

    # ── UP label conditions ───────────────────────────────────────────────────
    # 1. currently inside range
    # 2. price breaks above thresh_high within H
    # 3. TP is reached within H
    # 4. TP is reached before SL
    up_cond = (
        in_r
        & (first_broke_up < H)
        & (first_tp_up    < H)
        & (first_tp_up    < first_sl_up)
    )

    # ── DOWN label conditions ─────────────────────────────────────────────────
    down_cond = (
        in_r
        & (first_broke_down < H)
        & (first_tp_down    < H)
        & (first_tp_down    < first_sl_down)
    )

    # ── Assign labels ─────────────────────────────────────────────────────────
    label = np.ones(n, dtype=np.float32)   # default: NO_BREAK

    label[up_cond   & ~down_cond] = 2.0
    label[~up_cond  &  down_cond] = 0.0

    # Both directions qualify → whichever TP hits first wins; tie → NO_BREAK
    both = up_cond & down_cond
    label[both & (first_tp_up   < first_tp_down)] = 2.0
    label[both & (first_tp_down < first_tp_up)]   = 0.0

    # NaN for last H rows and ATR warmup
    label[-H:] = np.nan
    label[atr.isna().values] = np.nan

    label_s = pd.Series(label, index=df.index, name="label")

    # ── Fakeout labels ────────────────────────────────────────────────────────
    # Broke out but SL hit before (or without) TP — the trade was a loser
    up_fakeout = (
        in_r
        & (first_broke_up < H)
        & (first_sl_up    < H)
        & (first_sl_up   <= first_tp_up)   # SL at same time or earlier than TP
    )
    down_fakeout = (
        in_r
        & (first_broke_down < H)
        & (first_sl_down    < H)
        & (first_sl_down   <= first_tp_down)
    )

    return pd.DataFrame(
        {
            "label":        label_s,
            "up_fakeout":   up_fakeout.astype(np.float32),
            "down_fakeout": down_fakeout.astype(np.float32),
            "range_high":   range_high,
            "range_low":    range_low,
            "atr":          atr,
            "thresh_high":  thresh_high,
            "thresh_low":   thresh_low,
            "in_range":     in_range.astype(np.float32),
            "tp_up":        pd.Series(tp_up_v,   index=df.index),
            "tp_down":      pd.Series(tp_down_v, index=df.index),
            "sl_up":        pd.Series(sl_up_v,   index=df.index),
            "sl_down":      pd.Series(sl_down_v, index=df.index),
        },
        index=df.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
def label_stats(labels_df: pd.DataFrame, cfg: dict | None = None) -> None:
    """Print class distribution and fakeout summary."""
    s = labels_df["label"].dropna()
    n = len(s)
    counts = s.value_counts().sort_index()
    names  = {0: "DOWN_BREAK_SOON", 1: "NO_BREAK", 2: "UP_BREAK_SOON"}

    rr_str = ""
    if cfg:
        sl = cfg["labels"].get("sl_atr", "?")
        rr = cfg["labels"].get("rr_ratio", "?")
        rr_str = f"  [SL={sl}xATR  TP={float(sl)*float(rr):.1f}xATR  RR=1:{rr}]"

    print(f"\nLabel distribution (n={n:,}){rr_str}")
    for cls, cnt in counts.items():
        print(f"  {int(cls)} {names.get(int(cls), '?'):20s}  "
              f"{cnt:8,}  ({cnt/n*100:.1f}%)")

    fk_up      = int(labels_df["up_fakeout"].sum())
    fk_down    = int(labels_df["down_fakeout"].sum())
    tp_up      = int((s == 2).sum())
    tp_down    = int((s == 0).sum())
    # Of all bars that broke out upward: TP wins vs SL wins
    broke_up   = tp_up + fk_up
    broke_down = tp_down + fk_down
    tp_up_pct   = tp_up   / broke_up   * 100 if broke_up   else 0
    tp_down_pct = tp_down / broke_down * 100 if broke_down else 0
    print(f"\n  Of breakouts that occurred (within H bars):")
    print(f"    UP   breakouts: {broke_up:,}  |  TP win: {tp_up:,} "
          f"({tp_up_pct:.0f}%)   SL loss: {fk_up:,} ({100-tp_up_pct:.0f}%)")
    print(f"    DOWN breakouts: {broke_down:,}  |  TP win: {tp_down:,} "
          f"({tp_down_pct:.0f}%)   SL loss: {fk_down:,} ({100-tp_down_pct:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import data as data_mod

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = data_mod.load_all(cfg)
    lab = compute_labels(df, cfg)
    label_stats(lab, cfg)
    print(lab[["label", "range_high", "range_low", "tp_up", "sl_up"]].dropna().head(5))
