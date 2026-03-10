"""
EMA direction analysis: do wins/losses differ by EMA alignment per trade direction?
Uses real backtest pipeline. No model changes needed.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from pathlib import Path

import data as data_mod
import features as feat_mod
import labels as label_mod

cfg = yaml.safe_load(open("config.yaml"))
tc  = cfg["trading"]
lc  = cfg["labels"]

# Load artifacts
model = xgb.Booster()
model.load_model("models/xgb_model.json")
try:
    model.set_param({"device": "cuda"})
except Exception:
    pass
feat_cols = json.load(open("models/feature_columns.json"))

T_u  = tc["T_up"]
T_d  = tc["T_down"]
dmax = tc["d_max_atr"]
L    = lc.get("L_range", 20)
rh_c = f"dist_rh_{L}"
rl_c = f"dist_rl_{L}"

req_sq    = tc.get("require_squeeze", False)
min_ema   = tc.get("min_ema9_21_diff", -999.0)
max_cvd   = tc.get("max_cvd_slope",    999.0)
max_atr_r = tc.get("max_atr_ratio",    999.0)
sl_pct    = tc.get("sl_pct")
tp_pct    = tc.get("tp_pct")
sl_atr    = tc.get("sl_atr", 2)
tp_atr    = tc.get("tp_atr", 6)
use_pct   = sl_pct and tp_pct
time_stop = tc["time_stop"]
cooldown  = tc["cooldown"]
pos_pct   = tc["position_size_pct"]
capital0  = tc["initial_capital"]
cost_rt   = (lc["maker_fee"] + lc["taker_fee"] + lc["slippage"] + lc["spread"]) * 2

def get_col(df, name, default):
    return df[name].values if name in df.columns else np.full(len(df), default)

def run_year(year):
    raw = data_mod.load_csv(f"Data/BTCUSDT/full_year/{year}_1m.csv")
    df  = feat_mod.compute_features(raw, cfg)
    X   = xgb.DMatrix(df[feat_cols].fillna(0).values, feature_names=feat_cols)
    prob = model.predict(X)

    p_down = prob[:, 0]
    p_up   = prob[:, 2]

    close     = df["close"].values
    high      = df["high"].values
    low       = df["low"].values
    atr       = get_col(df, "atr_short", np.nan)
    sq_col    = get_col(df, "squeeze_flag", 0)
    ema_diff  = get_col(df, "ema9_21_diff", 999.0)
    cvd_sl    = get_col(df, "cvd_slope", 0.0)
    atr_rat   = get_col(df, "atr_ratio", 0.0)
    dist_rh   = get_col(df, rh_c, np.nan)
    dist_rl   = get_col(df, rl_c, np.nan)

    n = len(df)
    capital   = capital0
    pos_dir   = 0
    entry_idx = 0
    sl_price  = tp_price = pos_size = 0.0
    cdrem     = 0
    trades    = []

    for i in range(1, n):
        c = close[i]
        if np.isnan(c) or np.isnan(atr[i]):
            continue

        if pos_dir != 0:
            time_held = i - entry_idx
            h, l = high[i], low[i]
            hit_sl   = (pos_dir == 1 and l <= sl_price) or (pos_dir == -1 and h >= sl_price)
            hit_tp   = (pos_dir == 1 and h >= tp_price) or (pos_dir == -1 and l <= tp_price)
            hit_time = time_held >= time_stop
            if hit_sl or hit_tp or hit_time:
                exit_p = sl_price if hit_sl else (tp_price if hit_tp else c)
                raw_pnl = (exit_p - entry_price) * pos_dir * pos_size
                cost_pnl = entry_price * pos_size * cost_rt
                net_pnl  = raw_pnl - cost_pnl
                capital += net_pnl
                trades[-1].update({
                    "exit_reason": "SL" if hit_sl else ("TP" if hit_tp else "TIME"),
                    "net_pnl": net_pnl,
                    "win": 1 if net_pnl > 0 else 0,
                })
                pos_dir = 0
                cdrem   = cooldown

        if cdrem > 0:
            cdrem -= 1
            continue

        if pos_dir == 0:
            sq_ok  = (not req_sq) or sq_col[i] == 1
            ema_ok = ema_diff[i] >= min_ema
            cvd_ok = cvd_sl[i]  <= max_cvd
            atr_ok = atr_rat[i] <= max_atr_r

            if (p_up[i] >= T_u and not np.isnan(dist_rh[i]) and dist_rh[i] <= dmax
                    and sq_ok and ema_ok and cvd_ok and atr_ok):
                entry_price = c
                pos_size    = (capital * pos_pct) / (c + 1e-9)
                sl_price = c * (1 - sl_pct) if use_pct else c - sl_atr * atr[i]
                tp_price = c * (1 + tp_pct) if use_pct else c + tp_atr * atr[i]
                pos_dir  = 1
                entry_idx = i
                trades.append({"direction": "LONG", "ema_diff": ema_diff[i],
                               "p": p_up[i], "entry_idx": i})

            elif (p_down[i] >= T_d and not np.isnan(dist_rl[i]) and dist_rl[i] <= dmax
                    and sq_ok and ema_ok and cvd_ok and atr_ok):
                entry_price = c
                pos_size    = (capital * pos_pct) / (c + 1e-9)
                sl_price = c * (1 + sl_pct) if use_pct else c + sl_atr * atr[i]
                tp_price = c * (1 - tp_pct) if use_pct else c - tp_atr * atr[i]
                pos_dir  = -1
                entry_idx = i
                trades.append({"direction": "SHORT", "ema_diff": ema_diff[i],
                               "p": p_down[i], "entry_idx": i})

    # Drop any open trade at end
    trades = [t for t in trades if "win" in t]
    return pd.DataFrame(trades)


def show(df_t, year):
    if len(df_t) == 0:
        print(f"  {year}: no trades")
        return

    print(f"\n{'='*65}")
    print(f"  {year}  |  Trades: {len(df_t)}  |  WR: {df_t['win'].mean()*100:.1f}%")
    print(f"{'='*65}")
    print(f"  {'':8s}  {'All':>18s}  |  {'EMA>0 (bullish)':>18s}  |  {'EMA<0 (bearish)':>18s}")
    print(f"  {'-'*62}")

    for direction in ["LONG", "SHORT"]:
        sub = df_t[df_t["direction"] == direction]
        if len(sub) == 0:
            continue
        s_pos = sub[sub["ema_diff"] > 0]
        s_neg = sub[sub["ema_diff"] < 0]

        def fmt(s):
            if len(s) == 0: return "     N/A"
            return f"n={len(s):3d} WR={s['win'].mean()*100:.0f}%"

        print(f"  {direction:8s}  {fmt(sub):>18s}  |  {fmt(s_pos):>18s}  |  {fmt(s_neg):>18s}")

    # Finer buckets
    print(f"\n  Bucket breakdown (LONG vs SHORT separately):")
    bins   = [-999, -1.5, -0.5, 0.0, 0.5, 1.5, 999]
    blabels = ["<-1.5", "-1.5:-0.5", "-0.5:0", "0:0.5", "0.5:1.5", ">1.5"]
    df_t = df_t.copy()
    df_t["bucket"] = pd.cut(df_t["ema_diff"], bins=bins, labels=blabels)

    for direction in ["LONG", "SHORT"]:
        sub = df_t[df_t["direction"] == direction]
        if len(sub) == 0:
            continue
        g = sub.groupby("bucket", observed=True).agg(n=("win","count"), wr=("win","mean"))
        print(f"\n  {direction}")
        print(f"  {'Bucket':12s}  {'n':>4}  {'WR':>6}  Chart")
        for bkt, row in g.iterrows():
            if row["n"] == 0: continue
            bar = "#" * int(row["wr"] * 20)
            print(f"  {str(bkt):12s}  {int(row['n']):>4}  {row['wr']*100:>5.1f}%  {bar}")


for yr in [2023, 2024, 2025]:
    df_t = run_year(yr)
    show(df_t, yr)
