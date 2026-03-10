"""
Regime feature analysis: capture all feature values at trade entry,
compare wins vs losses across years. Goal: find features that
consistently separate wins from losses in ALL 3 years.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

import data as data_mod
import features as feat_mod

cfg = yaml.safe_load(open("config.yaml"))
tc  = cfg["trading"]
lc  = cfg["labels"]

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

CAPTURE_FEATURES = [
    "trend30_regime", "trend30_consol", "trend30_slope", "trend30_range",
    "cvd_trend_align", "cvd_trend_strength",
    "vol_regime", "rel_vol", "vol_zscore", "atr_ratio",
    "bb_pct", "bb_width", "squeeze_flag",
    "ema9_21_diff", "ema21_50_diff", "ema_trend_flag",
    "cvd_slope", "delta_ratio", "taker_buy_ratio",
    "dist_vwap", "price_mom_3", "price_mom_5",
]

def get_col(df, name, default):
    return df[name].values if name in df.columns else np.full(len(df), default)

def run_year(year):
    raw  = data_mod.load_csv(f"Data/BTCUSDT/full_year/{year}_1m.csv")
    df   = feat_mod.compute_features(raw, cfg)
    X    = xgb.DMatrix(df[feat_cols].fillna(0).values, feature_names=feat_cols)
    prob = model.predict(X)

    p_down = prob[:, 0]
    p_up   = prob[:, 2]

    close    = df["close"].values
    high     = df["high"].values
    low      = df["low"].values
    atr      = get_col(df, "atr_short", np.nan)
    sq_col   = get_col(df, "squeeze_flag", 0)
    ema_diff = get_col(df, "ema9_21_diff", 999.0)
    cvd_sl   = get_col(df, "cvd_slope", 0.0)
    atr_rat  = get_col(df, "atr_ratio", 0.0)
    dist_rh  = get_col(df, rh_c, np.nan)
    dist_rl  = get_col(df, rl_c, np.nan)

    n = len(df)
    capital = capital0
    pos_dir = 0
    entry_idx = 0
    sl_price = tp_price = pos_size = entry_price = 0.0
    cdrem = 0
    trades = []

    for i in range(1, n):
        c = close[i]
        if np.isnan(c) or np.isnan(atr[i]):
            continue

        if pos_dir != 0:
            time_held = i - entry_idx
            h, l = high[i], low[i]
            hit_sl   = (pos_dir ==  1 and l <= sl_price) or (pos_dir == -1 and h >= sl_price)
            hit_tp   = (pos_dir ==  1 and h >= tp_price) or (pos_dir == -1 and l <= tp_price)
            hit_time = time_held >= time_stop
            if hit_sl or hit_tp or hit_time:
                exit_p   = sl_price if hit_sl else (tp_price if hit_tp else c)
                raw_pnl  = (exit_p - entry_price) * pos_dir * pos_size
                net_pnl  = raw_pnl - entry_price * pos_size * cost_rt
                capital += net_pnl
                trades[-1]["win"]  = 1 if net_pnl > 0 else 0
                trades[-1]["exit"] = "SL" if hit_sl else ("TP" if hit_tp else "TIME")
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
            long_ok  = (p_up[i]   >= T_u and not np.isnan(dist_rh[i]) and dist_rh[i] <= dmax)
            short_ok = (p_down[i] >= T_d and not np.isnan(dist_rl[i]) and dist_rl[i] <= dmax)

            if (long_ok or short_ok) and sq_ok and ema_ok and cvd_ok and atr_ok:
                direction = "LONG" if long_ok else "SHORT"
                entry_price = c
                pos_size    = (capital * pos_pct) / (c + 1e-9)
                if direction == "LONG":
                    sl_price = c * (1 - sl_pct) if use_pct else c - sl_atr * atr[i]
                    tp_price = c * (1 + tp_pct) if use_pct else c + tp_atr * atr[i]
                    pos_dir  = 1
                else:
                    sl_price = c * (1 + sl_pct) if use_pct else c + sl_atr * atr[i]
                    tp_price = c * (1 - tp_pct) if use_pct else c - tp_atr * atr[i]
                    pos_dir  = -1
                entry_idx = i
                row = {"direction": direction, "year": year}
                for feat in CAPTURE_FEATURES:
                    row[feat] = float(df[feat].iloc[i]) if feat in df.columns else np.nan
                trades.append(row)

    return pd.DataFrame([t for t in trades if "win" in t])


# ── Run all years ─────────────────────────────────────────────────────────────
all_trades = []
for yr in [2023, 2024, 2025]:
    print(f"  Loading {yr}...")
    df_t = run_year(yr)
    all_trades.append(df_t)
    print(f"  {yr}: {len(df_t)} trades, WR={df_t['win'].mean()*100:.1f}%")

df_all = pd.concat(all_trades, ignore_index=True)

# ── Per-feature: wins vs losses median, and WR at best threshold ──────────────
print(f"\n{'='*80}")
print(f"  FEATURE ANALYSIS: median at entry for WINS vs LOSSES per year")
print(f"  Direction = which way to filter (higher=better or lower=better for WR)")
print(f"{'='*80}")
print(f"\n  {'Feature':<22} | {'23W':>6} {'23L':>6} | {'24W':>6} {'24L':>6} | {'25W':>6} {'25L':>6} | Dir")
print(f"  {'-'*75}")

for feat in CAPTURE_FEATURES:
    if feat not in df_all.columns:
        continue
    row_str = f"  {feat:<22} |"
    dirs = []
    for yr in [2023, 2024, 2025]:
        sub = df_all[df_all["year"] == yr]
        w = sub[sub["win"] == 1][feat].median()
        l = sub[sub["win"] == 0][feat].median()
        row_str += f" {w:>6.3f} {l:>6.3f} |"
        dirs.append("H" if w > l else "L")  # H=wins have higher values
    consistent = len(set(dirs)) == 1
    marker = " ***" if consistent else ""
    row_str += f" {'H' if dirs[0]=='H' else 'L'}{marker}"
    print(row_str)

# ── For consistent features: find threshold that blocks losses, keeps 2025 wins ─
print(f"\n{'='*80}")
print(f"  THRESHOLD SEARCH: keep 95%+ of 2025 wins, block max losses across ALL years")
print(f"{'='*80}")

wins25 = df_all[df_all["year"] == 2025]

for feat in CAPTURE_FEATURES:
    if feat not in df_all.columns:
        continue

    # Check consistency of direction
    dirs = []
    for yr in [2023, 2024, 2025]:
        sub = df_all[df_all["year"] == yr]
        w = sub[sub["win"] == 1][feat].median()
        l = sub[sub["win"] == 0][feat].median()
        dirs.append(1 if w > l else -1)
    if len(set(dirs)) > 1:
        continue  # inconsistent direction

    direction_up = dirs[0] == 1  # wins have higher values

    vals25_win = wins25[wins25["win"] == 1][feat].dropna()
    if len(vals25_win) < 5:
        continue

    thresholds = np.percentile(df_all[feat].dropna(), np.linspace(2, 98, 100))

    best = None
    best_score = 0

    for thresh in thresholds:
        if direction_up:
            # wins have higher values → keep when feat >= thresh
            w25_kept = (vals25_win >= thresh).mean()
            if w25_kept < 0.95:
                continue
            blk = {}
            for yr in [2023, 2024, 2025]:
                sub = df_all[df_all["year"] == yr]
                losses = sub[sub["win"] == 0][feat].dropna()
                blk[yr] = float((losses < thresh).mean()) if len(losses) else 0
            op = ">="
        else:
            # wins have lower values → keep when feat <= thresh
            w25_kept = (vals25_win <= thresh).mean()
            if w25_kept < 0.95:
                continue
            blk = {}
            for yr in [2023, 2024, 2025]:
                sub = df_all[df_all["year"] == yr]
                losses = sub[sub["win"] == 0][feat].dropna()
                blk[yr] = float((losses > thresh).mean()) if len(losses) else 0
            op = "<="

        score = blk[2023] * 1.5 + blk[2024] * 1.5 + blk[2025]
        if score > best_score:
            best_score = score
            best = {"thresh": thresh, "op": op, "w25_kept": w25_kept,
                    "blk23": blk[2023], "blk24": blk[2024], "blk25": blk[2025],
                    "score": score}

    if best and best_score > 0.1:
        print(f"\n  {feat} {best['op']} {best['thresh']:.4f}")
        print(f"    2025 wins kept: {best['w25_kept']:.1%}")
        print(f"    Losses blocked: 2023={best['blk23']:.1%}  2024={best['blk24']:.1%}  2025={best['blk25']:.1%}  score={best['score']:.3f}")

        # Show actual WR if filter applied
        print(f"    WR after filter:")
        for yr in [2023, 2024, 2025]:
            sub = df_all[df_all["year"] == yr]
            if best["op"] == ">=":
                kept = sub[sub[feat] >= best["thresh"]]
            else:
                kept = sub[sub[feat] <= best["thresh"]]
            if len(kept) == 0:
                continue
            print(f"      {yr}: n={len(kept):3d}  WR={kept['win'].mean()*100:.1f}%  "
                  f"(removed {len(sub)-len(kept)} trades, "
                  f"of which {(sub[sub['win']==0][feat].notna()).sum() - len(kept[kept['win']==0])} losses)")
