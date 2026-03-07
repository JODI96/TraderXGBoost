"""
features.py – Single source of truth for feature computation.

All features are CAUSAL: feature at index t uses only data up to t.
MTF (5m / 15m) features are derived by resampling the 1m buffer,
then forward-filling to 1m resolution with a 1-period shift to
ensure the completed bar is used.

Usage
-----
Batch (training):
    df_feat = compute_features(df_1m, cfg)

Live (single-step):
    engine = FeatureEngine(cfg)
    for candle in feed:
        row = engine.update(candle)   # returns pd.Series or None (warming up)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional
import yaml

# Suppress pandas PerformanceWarning caused by adding many columns one-by-one.
# The warning is cosmetic — computation is correct. A full pd.concat refactor
# would fix it properly but isn't worth the complexity here.
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented",
                        category=pd.errors.PerformanceWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ewm_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             span: int) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_c).abs(),
         (low  - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=span, min_periods=span, adjust=False).mean(), tr


def _slope(series: pd.Series, n: int = 5) -> pd.Series:
    """Normalised n-step slope: (x[t] - x[t-n]) / x[t-n]."""
    prev = series.shift(n)
    return (series - prev) / (prev.abs() + 1e-9)


def _rank_pct(series: pd.Series, window: int = 200) -> pd.Series:
    return series.rolling(window, min_periods=max(window // 4, 10)).rank(pct=True)


def _trend30_features(close: pd.Series, high: pd.Series, low: pd.Series,
                      atr: pd.Series, window: int,
                      slope_thresh: float, range_thresh: float) -> pd.DataFrame:
    """
    30-bar trend filter features.

    trend30_slope   : (close - close[t-window]) / (window * ATR)  — normalised momentum
    trend30_range   : rolling(window) high-low range / ATR         — how wide the move was
    trend30_regime  : +1 up-trending / -1 down-trending / 0 consolidating
    trend30_consol  : 1 if consolidating, else 0  (convenient binary signal)
    """
    slope = (close - close.shift(window)) / (window * atr + 1e-9)
    rng   = (high.rolling(window, min_periods=window // 2).max() -
             low.rolling(window,  min_periods=window // 2).min()) / (atr + 1e-9)

    regime = pd.Series(0, index=close.index, dtype=np.int8)
    regime[slope >  slope_thresh] =  1
    regime[slope < -slope_thresh] = -1
    # Override to 0 (consolidation) when range is compressed regardless of slope
    regime[rng < range_thresh] = 0

    return pd.DataFrame({
        "trend30_slope":  slope,
        "trend30_range":  rng,
        "trend30_regime": regime,
        "trend30_consol": (regime == 0).astype(np.int8),
    }, index=close.index)


def _compute_vp(df: pd.DataFrame, lookback: int,
                atr: pd.Series, n_buckets: int = 40) -> tuple:
    """
    Rolling Volume Profile: for each bar compute POC, VAH, VAL over
    the last `lookback` bars using numpy bincount (vectorised inner loop).

    Returns 5 arrays (length = len(df)), NaN before warmup:
        poc_dist      : (close - POC)  / ATR
        vah_dist      : (close - VAH)  / ATR
        val_dist      : (close - VAL)  / ATR
        poc_vol_ratio : POC bucket volume / total volume  (concentration)
        in_value_area : 1.0 if VAL <= close <= VAH, else 0.0
    """
    n       = len(df)
    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    volumes = df["volume"].values
    atrs    = atr.values

    poc_dist = np.full(n, np.nan)
    vah_dist = np.full(n, np.nan)
    val_dist = np.full(n, np.nan)
    poc_vr   = np.full(n, np.nan)
    in_va    = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        sl   = slice(i - lookback + 1, i + 1)
        h, l, v, c = highs[sl], lows[sl], volumes[sl], closes[sl]
        pmin, pmax = l.min(), h.max()
        atr_i = atrs[i]
        if pmax <= pmin or atr_i < 1e-9:
            continue

        bsize = (pmax - pmin) / n_buckets
        bidx  = np.clip(((c - pmin) / bsize).astype(int), 0, n_buckets - 1)
        bvol  = np.bincount(bidx, weights=v, minlength=n_buckets)
        total = bvol.sum()
        if total < 1e-9:
            continue

        poc_i     = int(bvol.argmax())
        poc_price = pmin + (poc_i + 0.5) * bsize

        # Value area: greedily add highest-volume buckets until 70% reached
        order  = np.argsort(bvol)[::-1]
        cum    = 0.0
        va_idx = []
        for idx in order:
            cum += bvol[idx]
            va_idx.append(int(idx))
            if cum >= total * 0.70:
                break
        vah_price = pmin + (max(va_idx) + 1) * bsize
        val_price = pmin + min(va_idx) * bsize

        cl           = closes[i]
        poc_dist[i]  = (cl - poc_price) / atr_i
        vah_dist[i]  = (cl - vah_price) / atr_i
        val_dist[i]  = (cl - val_price)  / atr_i
        poc_vr[i]    = bvol[poc_i] / total
        in_va[i]     = 1.0 if val_price <= cl <= vah_price else 0.0

    return poc_dist, vah_dist, val_dist, poc_vr, in_va


# ─────────────────────────────────────────────────────────────────────────────
# MTF resampling (used inside compute_features)
# ─────────────────────────────────────────────────────────────────────────────

_RESAMPLE_AGG = {
    "open":          "first",
    "high":          "max",
    "low":           "min",
    "close":         "last",
    "volume":        "sum",
    "taker_buy_vol": "sum",
}


def _mtf_features(df_1m: pd.DataFrame, rule: str, suffix: str,
                  atr_span: int = 14, rng_L: int = 20,
                  cvd_w: int = 10) -> pd.DataFrame:
    """
    Resample df_1m to `rule`, compute a set of HTF features, then
    forward-fill back to the 1m index.

    Critical: shift(1) on the HTF series so that at 1m bar t we only
    see the COMPLETED HTF bar, never the partially-open current one.
    """
    df_tf = df_1m[list(_RESAMPLE_AGG)].resample(rule).agg(_RESAMPLE_AGG).dropna()
    df_tf = df_tf.shift(1)                       # causal: use completed bar

    atr_tf, _ = _ewm_atr(df_tf["high"], df_tf["low"], df_tf["close"], atr_span)
    ema9  = df_tf["close"].ewm(span=9,  min_periods=9,  adjust=False).mean()
    ema21 = df_tf["close"].ewm(span=21, min_periods=21, adjust=False).mean()
    rh = df_tf["high"].rolling(rng_L, min_periods=1).max()
    rl = df_tf["low"].rolling(rng_L, min_periods=1).min()
    delta_tf = 2 * df_tf["taker_buy_vol"] - df_tf["volume"]
    cvd_tf   = delta_tf.rolling(cvd_w, min_periods=1).sum()

    feats = pd.DataFrame({
        f"ema9_{suffix}":        ema9,
        f"ema21_{suffix}":       ema21,
        f"ema9_slope_{suffix}":  _slope(ema9, 3),
        f"trend_{suffix}":       (ema9 > ema21).astype(int) - (ema9 < ema21).astype(int),
        f"atr_{suffix}":         atr_tf,
        f"rh_{suffix}":          rh,
        f"rl_{suffix}":          rl,
        f"dist_rh_{suffix}":     (rh - df_tf["close"]) / (atr_tf + 1e-9),
        f"dist_rl_{suffix}":     (df_tf["close"] - rl) / (atr_tf + 1e-9),
        f"cvd_{suffix}":         cvd_tf,
        f"delta_ratio_{suffix}": delta_tf / (df_tf["volume"] + 1e-9),
    }, index=df_tf.index)

    # Reindex to 1m and forward-fill
    return feats.reindex(df_1m.index).ffill()


# ─────────────────────────────────────────────────────────────────────────────
# Main feature function
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute all features on a 1-minute OHLCV DataFrame.

    Required input columns:
        open, high, low, close, volume, taker_buy_vol

    Index must be a pd.DatetimeIndex (UTC).

    Returns the same DataFrame with feature columns appended.
    NaN rows at the start (warmup period) are preserved; callers should
    drop them after aligning with labels.
    """
    fc    = cfg["features"]
    atr_s  = fc["atr_short"]          # 7
    atr_l  = fc["atr_long"]           # 50
    bb_p   = fc["bb_period"]          # 20
    bb_sd  = fc["bb_std"]             # 2.0
    kc_m   = fc["keltner_mult"]       # 1.5
    ema_ps = fc["ema_periods"]        # [9, 21, 50, 200]
    cvd_f  = fc["cvd_window"]         # 20  (fast)
    cvd_sl = fc["cvd_window_slow"]    # 60  (slow)
    vwap_w = fc["vwap_window"]        # 390
    rlbs   = fc["range_lookbacks"]    # [20, 60]
    vol_lb = fc["vol_lookback"]       # 20

    df = df.copy()

    # ── A: ATR / Volatility Contraction ──────────────────────────────────────
    atr_short, tr = _ewm_atr(df["high"], df["low"], df["close"], atr_s)
    atr_long,  _  = _ewm_atr(df["high"], df["low"], df["close"], atr_l)

    df["atr_short"]    = atr_short
    df["atr_long"]     = atr_long
    df["atr_ratio"]    = atr_short / (atr_long + 1e-9)
    df["tr_slope"]     = _slope(tr.rolling(5, min_periods=1).mean(), 5)

    # Bollinger Bands
    bb_mid  = df["close"].rolling(bb_p, min_periods=bb_p).mean()
    bb_std_ = df["close"].rolling(bb_p, min_periods=bb_p).std()
    bb_up   = bb_mid + bb_sd * bb_std_
    bb_lo   = bb_mid - bb_sd * bb_std_
    df["bb_width"]      = (bb_up - bb_lo) / (bb_mid.abs() + 1e-9)
    df["bb_pct"]        = (df["close"] - bb_lo) / (bb_up - bb_lo + 1e-9)
    df["bb_width_rank"] = _rank_pct(df["bb_width"], 200)

    # Keltner Channel – squeeze: BB inside KC
    ema20 = df["close"].ewm(span=20, min_periods=20, adjust=False).mean()
    kc_up = ema20 + kc_m * atr_short
    kc_lo = ema20 - kc_m * atr_short
    df["squeeze_flag"] = ((bb_up < kc_up) & (bb_lo > kc_lo)).astype(np.int8)
    df["vol_regime"]   = _rank_pct(atr_short, 200)   # 0=quiet … 1=explosive

    # ── B: Volume / Energy Build-up ──────────────────────────────────────────
    vol_mean = df["volume"].rolling(vol_lb, min_periods=5).mean()
    vol_std  = df["volume"].rolling(vol_lb, min_periods=5).std().clip(lower=1e-9)

    df["rel_vol"]         = df["volume"] / (vol_mean + 1e-9)
    df["vol_zscore"]      = (df["volume"] - vol_mean) / vol_std
    df["vol_rank"]        = _rank_pct(df["volume"], 200)
    df["taker_buy_ratio"] = df["taker_buy_vol"] / (df["volume"] + 1e-9)

    # Volume-squeeze composite: high volume while range is compressed
    rng_tight = (df["bb_width"] < df["bb_width"].rolling(100, min_periods=20).quantile(0.25)).astype(int)
    df["vol_squeeze_buildup"] = df["rel_vol"] * rng_tight

    # ── C: Orderflow / CVD (REAL delta from taker_buy_vol) ───────────────────
    delta = 2.0 * df["taker_buy_vol"] - df["volume"]   # positive = net buy

    df["delta"]       = delta
    df["delta_ratio"] = delta / (df["volume"] + 1e-9)   # -1 to +1

    df[f"cvd_{cvd_f}"]  = delta.rolling(cvd_f,  min_periods=1).sum()
    df[f"cvd_{cvd_sl}"] = delta.rolling(cvd_sl, min_periods=1).sum()
    df["cvd_slope"]     = _slope(df[f"cvd_{cvd_f}"], 5)

    # Delta divergence: price direction vs CVD direction  (0=confirm, ±2=diverge)
    p_dir  = df["close"].pct_change(cvd_f).apply(np.sign)
    cvd_dir = df[f"cvd_{cvd_f}"].diff(cvd_f).apply(np.sign)
    df["delta_divergence"] = p_dir - cvd_dir   # -2,0,+2

    # Wick geometry
    body_top   = df[["open", "close"]].max(axis=1)
    body_bot   = df[["open", "close"]].min(axis=1)
    upper_wick = df["high"] - body_top
    lower_wick = body_bot   - df["low"]
    c_range    = (df["high"] - df["low"]).clip(lower=1e-9)

    df["wick_imbalance"]   = (upper_wick - lower_wick) / c_range  # +→bearish pressure
    df["wick_vol_product"] = df["wick_imbalance"] * df["volume"]

    # Accumulation / Distribution (Money Flow Multiplier × Volume)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / c_range
    ad  = mfm * df["volume"]
    df["ad"]           = ad
    df["cum_ad_20"]    = ad.rolling(20, min_periods=1).sum()
    df["cum_ad_60"]    = ad.rolling(60, min_periods=1).sum()
    df["ad_slope"]     = _slope(df["cum_ad_20"], 5)
    ad_dir             = df["cum_ad_20"].diff(cvd_f).apply(np.sign)
    df["ad_price_div"] = p_dir - ad_dir

    # Delta Volume Bubble proxy (Hoss-style): big volume + extreme delta ratio
    df["delta_bubble"] = (
        (df["vol_zscore"] > 1.5).astype(int) * df["delta_ratio"].abs()
    )

    # CVD candle magnitude: percentile rank of |delta| over recent window (0=tiny, 1=huge)
    # Interaction with trend direction is handled in Section K after trend30 is computed.
    cvd_sw = fc.get("cvd_strength_window", 50)
    abs_delta = delta.abs()
    df["cvd_candle_pct"] = abs_delta.rolling(cvd_sw, min_periods=max(cvd_sw // 4, 5)).rank(pct=True)

    # ── D: VWAP Context ──────────────────────────────────────────────────────
    tp     = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = tp * df["volume"]
    df["vwap"]       = (tp_vol.rolling(vwap_w, min_periods=10).sum() /
                        df["volume"].rolling(vwap_w, min_periods=10).sum())
    df["dist_vwap"]  = (df["close"] - df["vwap"]) / (atr_short + 1e-9)
    df["vwap_slope"] = _slope(df["vwap"], 5)

    # Hoss VWAP-AD: volume-weighted bias relative to VWAP
    above_vwap = (df["close"] > df["vwap"]).astype(float) * 2 - 1  # +1 above, -1 below
    vwap_ad    = above_vwap * delta
    df["vwap_ad_cum"] = vwap_ad.rolling(cvd_f, min_periods=1).sum() / (
        df["volume"].rolling(cvd_f, min_periods=1).sum() + 1e-9
    )

    # ── E: Level Proximity + Range Features ──────────────────────────────────
    for L in rlbs:
        rh = df["high"].rolling(L, min_periods=L).max()
        rl = df["low"].rolling(L,  min_periods=L).min()
        rw = (rh - rl) / (df["close"] + 1e-9)

        df[f"range_high_{L}"]   = rh
        df[f"range_low_{L}"]    = rl
        df[f"range_width_{L}"]  = rw
        df[f"range_pctile_{L}"] = _rank_pct(rw, 200)

        df[f"dist_rh_{L}"]  = (rh - df["close"]) / (atr_short + 1e-9)
        df[f"dist_rl_{L}"]  = (df["close"] - rl)  / (atr_short + 1e-9)

        # Touch counts near boundary (within 0.3 ATR)
        near_h = ((rh - df["high"]) / (atr_short + 1e-9) < 0.3).astype(int)
        near_l = ((df["low"] - rl)  / (atr_short + 1e-9) < 0.3).astype(int)
        df[f"touches_h_{L}"] = near_h.rolling(L, min_periods=1).sum()
        df[f"touches_l_{L}"] = near_l.rolling(L, min_periods=1).sum()

        # Rejection wicks near boundary
        mask_h = ((rh - df["close"]) / (atr_short + 1e-9) < 0.8).astype(int)
        mask_l = ((df["close"] - rl)  / (atr_short + 1e-9) < 0.8).astype(int)
        df[f"reject_wick_h_{L}"] = (
            (upper_wick * mask_h).rolling(L, min_periods=1).mean() /
            (atr_short + 1e-9)
        )
        df[f"reject_wick_l_{L}"] = (
            (lower_wick * mask_l).rolling(L, min_periods=1).mean() /
            (atr_short + 1e-9)
        )

    # Causal pivot points (confirmed 2 bars back)
    is_ph = (df["high"].shift(2) > df["high"].shift(3)) & \
            (df["high"].shift(2) > df["high"].shift(1))
    is_pl = (df["low"].shift(2) < df["low"].shift(3)) & \
            (df["low"].shift(2) < df["low"].shift(1))
    ph = df["high"].shift(2).where(is_ph).ffill()
    pl = df["low"].shift(2).where(is_pl).ffill()
    df["dist_pivot_h"] = (ph - df["close"]) / (atr_short + 1e-9)
    df["dist_pivot_l"] = (df["close"] - pl) / (atr_short + 1e-9)

    # ── F: Trend (EMAs) ──────────────────────────────────────────────────────
    emas = {}
    for p in ema_ps:
        e = df["close"].ewm(span=p, min_periods=p, adjust=False).mean()
        emas[p] = e
        df[f"ema{p}_slope"] = _slope(e, 3)
        df[f"dist_ema{p}"]  = (df["close"] - e) / (atr_short + 1e-9)

    df["ema9_21_diff"]   = (emas[9]  - emas[21]) / (atr_short + 1e-9)
    df["ema21_50_diff"]  = (emas[21] - emas[50]) / (atr_short + 1e-9)
    df["ema_trend_flag"] = (
        ((emas[9] > emas[21]) & (emas[21] > emas[50])).astype(int) -
        ((emas[9] < emas[21]) & (emas[21] < emas[50])).astype(int)
    )

    # ── F2: Fast Momentum Features (react within 2-5 bars of trend change) ───
    # These give the model early warning of trend acceleration/deceleration,
    # letting it raise UP/DOWN probability faster than the slower EMA/trend30 features.

    # Ultra-fast EMAs: react to price direction within 2-3 bars
    ema3 = df["close"].ewm(span=3, min_periods=2, adjust=False).mean()
    ema5 = df["close"].ewm(span=5, min_periods=3, adjust=False).mean()
    df["ema3_slope"]  = _slope(ema3, 2)           # 2-bar normalised slope
    df["ema5_slope"]  = _slope(ema5, 3)           # 3-bar normalised slope
    df["ema3_5_diff"] = (ema3 - ema5) / (atr_short + 1e-9)  # fast crossover

    # Price momentum & acceleration: how fast is price moving RIGHT NOW
    df["price_mom_3"] = (df["close"] - df["close"].shift(3)) / (df["close"].shift(3) + 1e-9)
    df["price_mom_5"] = (df["close"] - df["close"].shift(5)) / (df["close"].shift(5) + 1e-9)
    # Acceleration: is the momentum speeding up or slowing down?
    df["price_accel"] = df["price_mom_3"] - df["price_mom_3"].shift(3)

    # Fast CVD: 3-bar delta captures immediate order flow pressure
    df["cvd_3"]        = delta.rolling(3, min_periods=1).sum()
    df["cvd_3_slope"]  = _slope(df["cvd_3"], 3)
    # Delta acceleration: is buying/selling pressure intensifying?
    df["delta_accel"]  = df["delta_ratio"] - df["delta_ratio"].shift(3)

    # Trend acceleration: is the EMA9-21 spread widening or narrowing?
    # Positive = trend strengthening; negative = trend fading
    df["ema_diff_accel"] = df["ema9_21_diff"] - df["ema9_21_diff"].shift(5)

    # ── G: MTF Features (5m & 15m, derived from 1m via causal resampling) ───
    if isinstance(df.index, pd.DatetimeIndex) and len(df) >= 20:
        for rule, suffix in [("5min", "5m"), ("15min", "15m")]:
            try:
                mtf = _mtf_features(df, rule=rule, suffix=suffix,
                                     atr_span=14, rng_L=20, cvd_w=10)
                for col in mtf.columns:
                    df[col] = mtf[col]
            except Exception as exc:
                print(f"[features] MTF {suffix} failed: {exc}")

    # ── H: Volume Profile (POC / VAH / VAL) ──────────────────────────────────
    for vp_lb in fc.get("range_lookbacks", [20, 60]):
        _pd, _vhd, _vld, _pvr, _iva = _compute_vp(df, vp_lb, atr_short)
        df[f"poc_dist_{vp_lb}"]      = _pd    # (close - POC)  / ATR
        df[f"vah_dist_{vp_lb}"]      = _vhd   # (close - VAH)  / ATR
        df[f"val_dist_{vp_lb}"]      = _vld   # (close - VAL)  / ATR
        df[f"poc_vol_ratio_{vp_lb}"] = _pvr   # volume concentration at POC
        df[f"in_value_area_{vp_lb}"] = _iva   # 1 if price inside value area

    # ── J: 30-min Trend Filter ────────────────────────────────────────────────
    t30w = fc.get("trend30_window",       30)
    t30s = fc.get("trend30_slope_thresh", 0.002)
    t30r = fc.get("trend30_range_thresh", 3.0)
    t30  = _trend30_features(df["close"], df["high"], df["low"],
                             atr_short, t30w, t30s, t30r)
    for col in t30.columns:
        df[col] = t30[col]

    # ── K: CVD × Trend Interaction ────────────────────────────────────────────
    # Explicit product features so the model gets a single axis encoding
    # "CVD confirms trend direction" without needing deep cross-feature splits.
    #
    #   cvd_trend_align   : delta_ratio × trend30_regime
    #       > 0  →  CVD direction matches trend (bullish CVD in uptrend, or bearish in downtrend)
    #       < 0  →  CVD contradicts trend
    #       = 0  →  consolidating (regime=0) or no net delta
    #
    #   cvd_trend_strength: cvd_candle_pct × trend30_regime
    #       high +  →  large bullish CVD candle during uptrend  (prefer LONG)
    #       high -  →  large bearish CVD candle during downtrend (prefer SHORT)
    #       ≈ 0     →  consolidation or small CVD candle
    t30_regime = df["trend30_regime"].astype(float)
    df["cvd_trend_align"]    = df["delta_ratio"] * t30_regime
    df["cvd_trend_strength"] = df["cvd_candle_pct"] * t30_regime

    # ── I: Time-of-Day & Day-of-Week (cyclic encoding) ───────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        hr  = df.index.hour + df.index.minute / 60.0
        dow = df.index.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * hr  / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hr  / 24.0)
        df["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Live Incremental Engine
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Maintains a rolling deque of 1m candles and calls compute_features
    on each update.  Single source of truth: same code path as training.
    """

    def __init__(self, cfg: dict, feature_cols: list[str] | None = None):
        self.cfg         = cfg
        self.buf_size    = cfg["features"]["live_buffer"]
        self.buffer: deque = deque(maxlen=self.buf_size)
        self.feature_cols = feature_cols  # if set, only return these columns
        # Warmup: need enough bars for ATR + EMA + range features.
        # VWAP and slow EMAs have min_periods set low so they populate quickly.
        # XGBoost handles NaN natively; we just need enough bars for core signals.
        self.min_warmup = max(
            cfg["features"]["atr_long"],          # 50
            max(cfg["features"]["range_lookbacks"]),  # 60
            max(cfg["features"]["ema_periods"][:2]),  # 21
        ) + 10   # = 70 bars (~70 minutes of warmup)

    def update(self, candle: pd.Series | dict) -> Optional[pd.Series]:
        """
        Add one new closed 1m candle and return the current feature vector.
        Returns None during warmup period.

        Parameters
        ----------
        candle : pd.Series or dict with keys: open, high, low, close,
                 volume, taker_buy_vol.  Index/timestamp is the candle's
                 open_time as a UTC-aware pd.Timestamp.
        """
        if isinstance(candle, dict):
            ts = candle.get("timestamp", candle.get("open_time", None))
            candle = pd.Series(candle)
            if ts is not None:
                candle.name = pd.Timestamp(ts, unit="ms", tz="UTC") \
                    if isinstance(ts, (int, float)) else pd.Timestamp(ts, tz="UTC")
        self.buffer.append(candle)

        if len(self.buffer) < self.min_warmup:
            return None

        df = pd.DataFrame(list(self.buffer))
        # Reconstruct DatetimeIndex from the series names (timestamps)
        if all(isinstance(s.name, pd.Timestamp) for s in self.buffer):
            df.index = pd.DatetimeIndex(
                [s.name for s in self.buffer], name="timestamp"
            )

        df_feat = compute_features(df, self.cfg)
        row = df_feat.iloc[-1]

        if self.feature_cols:
            # Fill any missing columns with NaN (XGBoost handles NaN natively)
            row = pd.Series(
                {c: row.get(c, np.nan) for c in self.feature_cols},
                index=self.feature_cols,
            )

        return row


# ─────────────────────────────────────────────────────────────────────────────
def get_feature_columns(cfg: dict) -> list[str]:
    """
    Return the ordered list of feature column names that will be produced
    by compute_features (excluding raw OHLCV and label cols).
    Build a tiny synthetic DataFrame to get the exact column list.
    """
    n = 500  # enough for all rolling windows
    rng = np.random.default_rng(0)
    price = 30000 + rng.normal(0, 100, n).cumsum()
    price = np.clip(price, 1000, 1e6)
    ts    = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df_tmp = pd.DataFrame({
        "open":          price,
        "high":          price * (1 + rng.uniform(0, 0.002, n)),
        "low":           price * (1 - rng.uniform(0, 0.002, n)),
        "close":         price,
        "volume":        rng.uniform(10, 200, n),
        "taker_buy_vol": rng.uniform(5,  100, n),
    }, index=ts)
    df_feat = compute_features(df_tmp, cfg)
    raw = {"open", "high", "low", "close", "volume", "taker_buy_vol"}
    return [c for c in df_feat.columns if c not in raw]


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cols = get_feature_columns(cfg)
    print(f"Total features: {len(cols)}")
    for c in cols:
        print(" ", c)
