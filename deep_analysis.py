"""
deep_analysis.py - Deep statistical analysis of losing trades.
Goal: find filters that produce HUGE improvement in profit factor and win rate.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

import data as data_mod
import features as feat_mod
import labels as label_mod


# ─────────────────────────────────────────────────────────────────────────────
def compute_profit_factor(pnls):
    wins  = pnls[pnls > 0].sum()
    losses = (-pnls[pnls < 0]).sum()
    if losses < 1e-9:
        return float("inf")
    return wins / losses


def compute_return_pct(pnls, initial_capital=10.0):
    capital = initial_capital
    for p in pnls:
        capital += p
    return (capital - initial_capital) / initial_capital * 100.0


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("DEEP LOSING TRADE ANALYSIS")
    print("=" * 80)

    # ── Load config ──────────────────────────────────────────────────────────
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # ── Load trades ──────────────────────────────────────────────────────────
    trades = pd.read_csv("models/backtest_trades.csv")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["exit_ts"]  = pd.to_datetime(trades["exit_ts"],  utc=True)
    trades["outcome"]  = (trades["net_pnl"] > 0).astype(int)
    n_trades = len(trades)
    n_wins   = trades["outcome"].sum()
    n_losses = n_trades - n_wins
    base_wr  = n_wins / n_trades
    base_pf  = compute_profit_factor(trades["net_pnl"].values)

    print(f"\nBaseline: {n_trades} trades | wins={n_wins} losses={n_losses}")
    print(f"Win rate: {base_wr*100:.1f}%  |  PF: {base_pf:.3f}")

    # ── Load feature data for test split ─────────────────────────────────────
    print("\nLoading feature data for test split...")
    df_raw  = data_mod.load_all(cfg)
    split   = int(len(df_raw) * (1 - cfg["training"]["test_size"]))
    df_raw  = df_raw.iloc[split:]

    df_feat = feat_mod.compute_features(df_raw, cfg)
    df_lab  = label_mod.compute_labels(df_raw, cfg)

    with open("models/feature_columns.json") as f:
        feat_cols = json.load(f)

    avail   = [c for c in feat_cols if c in df_feat.columns]
    df_all  = df_feat.join(df_lab[["label"]], how="inner").dropna(subset=avail + ["label"])
    df_all  = df_all.sort_index()

    print(f"Feature data: {len(df_all)} bars")

    # ── Match each trade to entry bar features ───────────────────────────────
    print("\nMatching trades to feature bars...")
    trade_features = []

    # Normalize trade timestamps to match feature index
    feat_index_set = set(df_all.index)

    matched = 0
    unmatched = 0
    for _, row in trades.iterrows():
        ts = row["entry_ts"]
        # Try exact match first
        if ts in feat_index_set:
            feat_row = df_all.loc[ts]
            if isinstance(feat_row, pd.DataFrame):
                feat_row = feat_row.iloc[0]
            rec = feat_row[avail].to_dict()
            rec["entry_ts"]   = ts
            rec["net_pnl"]    = row["net_pnl"]
            rec["outcome"]    = row["outcome"]
            rec["direction"]  = row["direction"]
            rec["exit_reason"]= row["exit_reason"]
            rec["time_held"]  = row["time_held"]
            rec["entry_price"]= row["entry_price"]
            rec["entry_idx"]  = row["entry_idx"]
            trade_features.append(rec)
            matched += 1
        else:
            # Try nearest bar within 1 minute
            diffs = abs(df_all.index - ts)
            nearest_pos = diffs.argmin()
            if diffs[nearest_pos].total_seconds() <= 60:
                feat_row = df_all.iloc[nearest_pos]
                rec = feat_row[avail].to_dict()
                rec["entry_ts"]   = ts
                rec["net_pnl"]    = row["net_pnl"]
                rec["outcome"]    = row["outcome"]
                rec["direction"]  = row["direction"]
                rec["exit_reason"]= row["exit_reason"]
                rec["time_held"]  = row["time_held"]
                rec["entry_price"]= row["entry_price"]
                rec["entry_idx"]  = row["entry_idx"]
                trade_features.append(rec)
                matched += 1
            else:
                unmatched += 1

    print(f"Matched: {matched} / {n_trades}  (unmatched: {unmatched})")

    df_t = pd.DataFrame(trade_features)
    df_t["hour_utc"] = pd.to_datetime(df_t["entry_ts"]).dt.hour

    wins_df   = df_t[df_t["outcome"] == 1]
    losses_df = df_t[df_t["outcome"] == 0]

    print(f"\nIn analysis set: {len(df_t)} trades | wins={len(wins_df)} | losses={len(losses_df)}")

    # =========================================================================
    # A) Feature distributions: Cohen's d, rank by discrimination power
    # =========================================================================
    print("\n" + "=" * 80)
    print("A) TOP 20 MOST DISCRIMINATING FEATURES (Cohen's d)")
    print("=" * 80)

    feature_disc = []
    for col in avail:
        if col not in df_t.columns:
            continue
        w_vals = wins_df[col].dropna()
        l_vals = losses_df[col].dropna()
        if len(w_vals) < 3 or len(l_vals) < 3:
            continue
        w_mean = w_vals.mean()
        l_mean = l_vals.mean()
        # Pooled std (Cohen's d)
        n_w, n_l = len(w_vals), len(l_vals)
        pooled_std = np.sqrt(
            ((n_w - 1) * w_vals.std()**2 + (n_l - 1) * l_vals.std()**2) / (n_w + n_l - 2 + 1e-9)
        )
        if pooled_std < 1e-9:
            continue
        cohens_d = abs(w_mean - l_mean) / pooled_std
        feature_disc.append({
            "feature": col,
            "win_mean": w_mean,
            "loss_mean": l_mean,
            "diff": w_mean - l_mean,
            "cohens_d": cohens_d,
            "win_median": w_vals.median(),
            "loss_median": l_vals.median(),
        })

    feature_disc_df = pd.DataFrame(feature_disc).sort_values("cohens_d", ascending=False)

    print(f"\n{'Feature':<35} {'Win Mean':>10} {'Loss Mean':>10} {'Diff':>10} {'Cohen d':>10}")
    print("-" * 80)
    for _, r in feature_disc_df.head(20).iterrows():
        print(f"  {r['feature']:<33} {r['win_mean']:>10.4f} {r['loss_mean']:>10.4f} "
              f"{r['diff']:>10.4f} {r['cohens_d']:>10.4f}")

    # =========================================================================
    # B) Time of day analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("B) TIME OF DAY ANALYSIS")
    print("=" * 80)

    hour_stats = []
    for hr in sorted(df_t["hour_utc"].unique()):
        mask = df_t["hour_utc"] == hr
        sub  = df_t[mask]
        n    = len(sub)
        if n == 0:
            continue
        wr   = sub["outcome"].mean()
        avg_pnl = sub["net_pnl"].mean()
        hour_stats.append({"hour": hr, "n": n, "win_rate": wr, "avg_pnl": avg_pnl})

    hour_df = pd.DataFrame(hour_stats).sort_values("win_rate")

    print(f"\n{'Hour UTC':>8} {'N':>4} {'Win%':>6} {'Avg PnL':>10}")
    print("-" * 35)
    for _, r in hour_df.iterrows():
        flag = "  <-- BAD" if r["win_rate"] < 0.40 and r["n"] >= 3 else ""
        print(f"  {int(r['hour']):>6}   {int(r['n']):>4}  {r['win_rate']*100:>5.1f}%  "
              f"{r['avg_pnl']:>10.2f}{flag}")

    bad_hours = hour_df[(hour_df["win_rate"] < 0.40) & (hour_df["n"] >= 3)]["hour"].tolist()
    print(f"\nBad hours (WR<40%, n>=3): {bad_hours}")
    if bad_hours:
        mask_bh  = df_t["hour_utc"].isin(bad_hours)
        wins_blocked_bh   = (wins_df["hour_utc"].isin(bad_hours)).sum()
        losses_blocked_bh = (losses_df["hour_utc"].isin(bad_hours)).sum()
        remain = df_t[~mask_bh]
        new_wr = remain["outcome"].mean() if len(remain) else 0
        new_pf = compute_profit_factor(remain["net_pnl"].values) if len(remain) else 0
        print(f"  Time filter (block bad hours): trades remaining={len(remain)}")
        print(f"  Wins blocked: {wins_blocked_bh}/{len(wins_df)} ({wins_blocked_bh/len(wins_df)*100:.1f}%)")
        print(f"  Losses blocked: {losses_blocked_bh}/{len(losses_df)} ({losses_blocked_bh/len(losses_df)*100:.1f}%)")
        print(f"  New WR: {new_wr*100:.1f}%  |  New PF: {new_pf:.3f}")

    # =========================================================================
    # C) Threshold scan for top 5 discriminating features
    # =========================================================================
    print("\n" + "=" * 80)
    print("C) THRESHOLD SCAN FOR TOP FEATURES")
    print("=" * 80)

    top5 = feature_disc_df.head(5)["feature"].tolist()

    best_filters = []

    for feat in top5:
        if feat not in df_t.columns:
            continue
        feat_vals = df_t[feat].dropna()
        if len(feat_vals) < 10:
            continue

        w_mean  = wins_df[feat].dropna().mean()
        l_mean  = losses_df[feat].dropna().mean()

        # Determine direction: higher = good for wins? or lower?
        direction = ">" if w_mean > l_mean else "<"

        # Scan percentile thresholds
        percentiles = np.arange(10, 91, 5)
        thresholds  = np.percentile(feat_vals.dropna(), percentiles)

        best_score = -999
        best_thresh = None
        best_stats  = None

        for thresh in thresholds:
            if direction == ">":
                mask_keep = df_t[feat] >= thresh
            else:
                mask_keep = df_t[feat] <= thresh

            sub = df_t[mask_keep.fillna(False)]
            if len(sub) < 10:
                continue

            sub_wins   = (sub["outcome"] == 1).sum()
            sub_losses = (sub["outcome"] == 0).sum()
            all_wins   = (df_t["outcome"] == 1).sum()
            all_losses = (df_t["outcome"] == 0).sum()

            wins_kept_pct    = sub_wins   / (all_wins   + 1e-9)
            losses_kept_pct  = sub_losses / (all_losses + 1e-9)
            losses_blocked_pct = 1 - losses_kept_pct
            wins_blocked_pct   = 1 - wins_kept_pct

            # Score: maximize (losses_blocked - wins_blocked) with penalty for too few trades
            score = losses_blocked_pct - wins_blocked_pct
            if len(sub) < 15:
                score -= 0.5

            if score > best_score:
                best_score  = score
                best_thresh = thresh
                pf_new = compute_profit_factor(sub["net_pnl"].values)
                wr_new = sub["outcome"].mean() if len(sub) else 0
                best_stats = {
                    "feature":         feat,
                    "threshold":       round(thresh, 6),
                    "direction":       direction,
                    "n_remain":        len(sub),
                    "wins_blocked":    int(all_wins - sub_wins),
                    "losses_blocked":  int(all_losses - sub_losses),
                    "wins_blocked_pct":   round(wins_blocked_pct * 100, 1),
                    "losses_blocked_pct": round(losses_blocked_pct * 100, 1),
                    "new_wr":          round(wr_new * 100, 1),
                    "new_pf":          round(pf_new, 3),
                    "score":           round(best_score, 3),
                }

        if best_stats:
            best_filters.append(best_stats)
            print(f"\n  Feature: {feat}  ({direction} {best_stats['threshold']})")
            print(f"    Trades remaining: {best_stats['n_remain']}")
            print(f"    Wins blocked:     {best_stats['wins_blocked']} / {len(wins_df)} "
                  f"({best_stats['wins_blocked_pct']}%)")
            print(f"    Losses blocked:   {best_stats['losses_blocked']} / {len(losses_df)} "
                  f"({best_stats['losses_blocked_pct']}%)")
            print(f"    New WR: {best_stats['new_wr']}%  |  New PF: {best_stats['new_pf']}")

    # =========================================================================
    # C2) Extended scan over ALL discriminating features (top 30)
    # =========================================================================
    print("\n" + "=" * 80)
    print("C2) EXTENDED THRESHOLD SCAN (top 30 features)")
    print("=" * 80)

    top30 = feature_disc_df.head(30)["feature"].tolist()
    all_filter_candidates = []

    for feat in top30:
        if feat not in df_t.columns:
            continue
        feat_vals = df_t[feat].dropna()
        if len(feat_vals) < 10:
            continue

        w_vals = wins_df[feat].dropna()
        l_vals = losses_df[feat].dropna()
        if len(w_vals) < 3 or len(l_vals) < 3:
            continue

        w_mean  = w_vals.mean()
        l_mean  = l_vals.mean()
        direction = ">" if w_mean > l_mean else "<"

        # Fine scan
        percentiles = np.arange(5, 96, 2)
        thresholds  = np.percentile(feat_vals.dropna(), percentiles)

        for thresh in thresholds:
            if direction == ">":
                mask_keep = df_t[feat] >= thresh
            else:
                mask_keep = df_t[feat] <= thresh

            sub = df_t[mask_keep.fillna(False)]
            if len(sub) < 30:  # Must keep >= 30 trades
                continue

            sub_wins   = (sub["outcome"] == 1).sum()
            sub_losses = (sub["outcome"] == 0).sum()
            all_wins   = (df_t["outcome"] == 1).sum()
            all_losses = (df_t["outcome"] == 0).sum()

            wins_blocked_pct   = 1 - sub_wins   / (all_wins   + 1e-9)
            losses_blocked_pct = 1 - sub_losses / (all_losses + 1e-9)

            # Quality filter criteria
            if losses_blocked_pct < 0.30:
                continue
            if wins_blocked_pct > 0.15:
                continue

            pf_new = compute_profit_factor(sub["net_pnl"].values)
            wr_new = sub["outcome"].mean()

            all_filter_candidates.append({
                "feature":         feat,
                "threshold":       round(thresh, 6),
                "direction":       direction,
                "n_remain":        len(sub),
                "wins_blocked":    int(all_wins - sub_wins),
                "losses_blocked":  int(all_losses - sub_losses),
                "wins_blocked_pct":   round(wins_blocked_pct * 100, 1),
                "losses_blocked_pct": round(losses_blocked_pct * 100, 1),
                "new_wr":          round(wr_new * 100, 1),
                "new_pf":          round(pf_new, 3),
                "pf_improvement":  round(pf_new - base_pf, 3),
            })

    if all_filter_candidates:
        cand_df = pd.DataFrame(all_filter_candidates).sort_values("new_pf", ascending=False)
        # Deduplicate by feature (keep best per feature)
        cand_df_best = cand_df.groupby("feature").first().reset_index().sort_values("new_pf", ascending=False)

        print(f"\nFilters meeting criteria (blocks >=30% losses, <=15% wins, >=30 trades):")
        print(f"{'Feature':<35} {'Dir':>3} {'Thresh':>10} {'Remain':>7} {'W_blk%':>7} "
              f"{'L_blk%':>7} {'NewWR':>6} {'NewPF':>7}")
        print("-" * 95)
        for _, r in cand_df_best.head(20).iterrows():
            print(f"  {r['feature']:<33} {r['direction']:>3} {r['threshold']:>10.4f} "
                  f"{r['n_remain']:>7} {r['wins_blocked_pct']:>6.1f}% "
                  f"{r['losses_blocked_pct']:>6.1f}% {r['new_wr']:>5.1f}% "
                  f"{r['new_pf']:>7.3f}")
    else:
        cand_df_best = pd.DataFrame()
        print("No single filters met strict criteria. Relaxing for interaction analysis...")

    # =========================================================================
    # D) Interaction analysis: pairs of top features
    # =========================================================================
    print("\n" + "=" * 80)
    print("D) INTERACTION ANALYSIS (2-filter combos, PF>2.0 target)")
    print("=" * 80)

    # Use best single-filter thresholds found
    # Build candidate filter list for combinations
    # Use top-10 from discriminating features with their best thresholds

    # First build a set of (feature, direction, threshold) combos from C2 or fall back to C
    if len(all_filter_candidates) > 0:
        filter_pool_df = pd.DataFrame(all_filter_candidates).sort_values("pf_improvement", ascending=False)
        filter_pool_df = filter_pool_df.groupby("feature").first().reset_index()
    else:
        # Fallback: use top features with median as threshold
        rows = []
        for feat in feature_disc_df.head(15)["feature"].tolist():
            if feat not in df_t.columns:
                continue
            w_mean = wins_df[feat].dropna().mean()
            l_mean = losses_df[feat].dropna().mean()
            direction = ">" if w_mean > l_mean else "<"
            thresh = df_t[feat].dropna().median()
            rows.append({"feature": feat, "direction": direction, "threshold": thresh})
        filter_pool_df = pd.DataFrame(rows)

    filter_pool = filter_pool_df.head(10).to_dict("records")

    combo_results = []
    for i, f1 in enumerate(filter_pool):
        for f2 in filter_pool[i+1:]:
            feat1, dir1, thresh1 = f1["feature"], f1["direction"], f1["threshold"]
            feat2, dir2, thresh2 = f2["feature"], f2["direction"], f2["threshold"]

            if feat1 not in df_t.columns or feat2 not in df_t.columns:
                continue

            if dir1 == ">":
                mask1 = df_t[feat1] >= thresh1
            else:
                mask1 = df_t[feat1] <= thresh1

            if dir2 == ">":
                mask2 = df_t[feat2] >= thresh2
            else:
                mask2 = df_t[feat2] <= thresh2

            sub = df_t[mask1.fillna(False) & mask2.fillna(False)]
            if len(sub) < 30:
                continue

            pf_new = compute_profit_factor(sub["net_pnl"].values)
            wr_new = sub["outcome"].mean()
            n_remain = len(sub)

            if pf_new > 1.5:
                combo_results.append({
                    "feat1": feat1, "dir1": dir1, "thresh1": thresh1,
                    "feat2": feat2, "dir2": dir2, "thresh2": thresh2,
                    "n_remain": n_remain,
                    "win_rate": round(wr_new * 100, 1),
                    "pf": round(pf_new, 3),
                })

    combo_df = pd.DataFrame(combo_results).sort_values("pf", ascending=False) if combo_results else pd.DataFrame()

    if not combo_df.empty:
        print(f"\nTop combos with PF > 1.5 and >= 30 trades:")
        print(f"{'Feature1':<30} {'Dir':>3} {'Thresh1':>10} | {'Feature2':<30} {'Dir':>3} {'Thresh2':>10} "
              f"{'Rem':>5} {'WR%':>6} {'PF':>7}")
        print("-" * 120)
        for _, r in combo_df.head(15).iterrows():
            print(f"  {r['feat1']:<28} {r['dir1']:>3} {r['thresh1']:>10.4f} | "
                  f"{r['feat2']:<28} {r['dir2']:>3} {r['thresh2']:>10.4f} "
                  f"{r['n_remain']:>5} {r['win_rate']:>6.1f}% {r['pf']:>7.3f}")
    else:
        print("No combos found with PF > 1.5 and >= 30 trades.")

    # =========================================================================
    # E) Exit analysis: losing trades - immediate reversal vs slow grind
    # =========================================================================
    print("\n" + "=" * 80)
    print("E) EXIT ANALYSIS - Losing Trade Patterns")
    print("=" * 80)

    loss_trades = trades[trades["net_pnl"] <= 0].copy()

    print(f"\nLosing trades breakdown by exit reason:")
    for reason, grp in loss_trades.groupby("exit_reason"):
        avg_pnl   = grp["net_pnl"].mean()
        avg_held  = grp["time_held"].mean()
        print(f"  {reason:<8}: {len(grp):>3} trades | avg_pnl={avg_pnl:.2f} | avg_held={avg_held:.1f} bars")

    print(f"\nLosing trades by time held:")
    time_bins = [(1, 2, "1-2 bars (instant reversal)"),
                 (3, 5, "3-5 bars"),
                 (6, 10, "6-10 bars"),
                 (11, 20, "11-20 bars"),
                 (21, 30, "21-30 bars (slow grind)")]
    for lo, hi, label in time_bins:
        sub = loss_trades[(loss_trades["time_held"] >= lo) & (loss_trades["time_held"] <= hi)]
        if len(sub) > 0:
            print(f"  {label:<35}: {len(sub):>3} ({len(sub)/len(loss_trades)*100:.1f}%) "
                  f"avg_pnl={sub['net_pnl'].mean():.2f}")

    # Feature analysis: instant reversal vs slow grind
    instant_loss_idx = set(loss_trades[loss_trades["time_held"] <= 2]["entry_ts"].tolist())
    slow_loss_idx    = set(loss_trades[loss_trades["time_held"] >= 10]["entry_ts"].tolist())

    inst_feat = df_t[df_t["entry_ts"].isin(instant_loss_idx)]
    slow_feat = df_t[df_t["entry_ts"].isin(slow_loss_idx)]

    if len(inst_feat) >= 2 and len(slow_feat) >= 2:
        print(f"\nInstant reversals (held<=2): {len(inst_feat)} | Slow grind (held>=10): {len(slow_feat)}")
        print("Top features differentiating instant vs slow losses:")
        diff_rows = []
        for col in ["atr_ratio", "vol_zscore", "rel_vol", "bb_pct", "dist_rh_20", "dist_rl_20",
                    "ema9_21_diff", "trend30_slope", "cvd_10", "delta_ratio", "vol_regime",
                    "squeeze_flag", "bb_width_rank"]:
            if col not in df_t.columns:
                continue
            i_vals = inst_feat[col].dropna()
            s_vals = slow_feat[col].dropna()
            if len(i_vals) < 2 or len(s_vals) < 2:
                continue
            diff_rows.append({"feature": col,
                               "instant_mean": round(i_vals.mean(), 4),
                               "slow_mean":    round(s_vals.mean(), 4)})
        for r in diff_rows:
            print(f"    {r['feature']:<25}: instant={r['instant_mean']:>8.4f}  slow={r['slow_mean']:>8.4f}")

    # =========================================================================
    # F) Consecutive losses / clustering
    # =========================================================================
    print("\n" + "=" * 80)
    print("F) CONSECUTIVE LOSS CLUSTERING")
    print("=" * 80)

    trades_sorted = trades.sort_values("entry_ts").reset_index(drop=True)
    streak = 0
    max_streak = 0
    streak_starts = []
    current_streak_start = None
    clusters = []

    for idx, row in trades_sorted.iterrows():
        if row["net_pnl"] <= 0:
            streak += 1
            if streak == 1:
                current_streak_start = row["entry_ts"]
            if streak > max_streak:
                max_streak = streak
        else:
            if streak >= 3:
                clusters.append({
                    "start": current_streak_start,
                    "length": streak,
                    "total_loss": trades_sorted.iloc[max(0,idx-streak):idx]["net_pnl"].sum()
                })
            streak = 0
            current_streak_start = None

    if streak >= 3:
        clusters.append({"start": current_streak_start, "length": streak,
                          "total_loss": trades_sorted.tail(streak)["net_pnl"].sum()})

    print(f"\nMax consecutive losses: {max_streak}")
    print(f"Loss clusters (>= 3 in a row): {len(clusters)}")
    for c in clusters:
        print(f"  Start: {c['start']}  |  Length: {c['length']}  |  Total loss: {c['total_loss']:.2f}")

    # Look at market condition features before a loss cluster
    if clusters and len(df_t) > 0:
        print("\nFeature state before/during loss clusters:")
        for c in clusters[:3]:
            ts_start = pd.Timestamp(c["start"])
            near = df_t[df_t["entry_ts"] <= ts_start].tail(1)
            if len(near):
                r = near.iloc[0]
                feat_report = []
                for col in ["atr_ratio", "vol_regime", "trend30_regime", "ema9_21_diff",
                            "bb_width_rank", "squeeze_flag", "vol_zscore"]:
                    if col in r.index:
                        feat_report.append(f"{col}={r[col]:.3f}")
                print(f"  {ts_start}  len={c['length']}: " + "  ".join(feat_report))

    # =========================================================================
    # SUMMARY: Top 3 filters
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: TOP 3 SINGLE FILTERS RANKED BY PF IMPROVEMENT")
    print("=" * 80)

    # Build comprehensive candidate list from all analyses
    all_candidates = []

    # From extended scan
    if len(all_filter_candidates) > 0:
        for c in all_filter_candidates:
            all_candidates.append(c)

    # Also add time filter if it qualifies
    if bad_hours:
        mask_bh = ~df_t["hour_utc"].isin(bad_hours)
        sub_bh  = df_t[mask_bh]
        if len(sub_bh) >= 30:
            w_blk = (wins_df["hour_utc"].isin(bad_hours)).sum()
            l_blk = (losses_df["hour_utc"].isin(bad_hours)).sum()
            pf_bh = compute_profit_factor(sub_bh["net_pnl"].values)
            wr_bh = sub_bh["outcome"].mean()
            all_candidates.append({
                "feature": f"hour_utc NOT IN {bad_hours}",
                "threshold": None,
                "direction": "time",
                "n_remain": len(sub_bh),
                "wins_blocked": w_blk,
                "losses_blocked": l_blk,
                "wins_blocked_pct": round(w_blk / len(wins_df) * 100, 1),
                "losses_blocked_pct": round(l_blk / len(losses_df) * 100, 1),
                "new_wr": round(wr_bh * 100, 1),
                "new_pf": round(pf_bh, 3),
                "pf_improvement": round(pf_bh - base_pf, 3),
            })

    # From C (best per feature, any threshold)
    if len(best_filters) > 0:
        for c in best_filters:
            sub_tmp = df_t
            if c["direction"] == ">":
                sub_tmp = df_t[df_t[c["feature"]] >= c["threshold"]]
            elif c["direction"] == "<":
                sub_tmp = df_t[df_t[c["feature"]] <= c["threshold"]]
            if len(sub_tmp) >= 30:
                pf_tmp = compute_profit_factor(sub_tmp["net_pnl"].values)
                wr_tmp = sub_tmp["outcome"].mean()
                sub_w  = sub_tmp["outcome"].sum()
                sub_l  = (sub_tmp["outcome"] == 0).sum()
                all_wins   = n_wins
                all_losses = n_losses
                all_candidates.append({
                    "feature":         c["feature"],
                    "threshold":       c["threshold"],
                    "direction":       c["direction"],
                    "n_remain":        len(sub_tmp),
                    "wins_blocked":    all_wins - sub_w,
                    "losses_blocked":  all_losses - sub_l,
                    "wins_blocked_pct":   round((all_wins - sub_w) / (all_wins + 1e-9) * 100, 1),
                    "losses_blocked_pct": round((all_losses - sub_l) / (all_losses + 1e-9) * 100, 1),
                    "new_wr":          round(wr_tmp * 100, 1),
                    "new_pf":          round(pf_tmp, 3),
                    "pf_improvement":  round(pf_tmp - base_pf, 3),
                })

    if not all_candidates:
        # Fall back: just rank all single thresholds
        print("Insufficient data for strict criteria. Showing best available filters:")
        rows2 = []
        for feat in feature_disc_df.head(20)["feature"].tolist():
            if feat not in df_t.columns:
                continue
            w_mean  = wins_df[feat].dropna().mean()
            l_mean  = losses_df[feat].dropna().mean()
            direction = ">" if w_mean > l_mean else "<"
            for pct in [25, 30, 35, 40, 45]:
                thresh = np.percentile(df_t[feat].dropna(), pct if direction == ">" else 100-pct)
                if direction == ">":
                    sub = df_t[df_t[feat] >= thresh]
                else:
                    sub = df_t[df_t[feat] <= thresh]
                if len(sub) < 20:
                    continue
                pf_new = compute_profit_factor(sub["net_pnl"].values)
                wr_new = sub["outcome"].mean()
                sub_w  = sub["outcome"].sum()
                sub_l  = (sub["outcome"] == 0).sum()
                rows2.append({
                    "feature": feat, "threshold": thresh, "direction": direction,
                    "n_remain": len(sub),
                    "wins_blocked": n_wins - sub_w,
                    "losses_blocked": n_losses - sub_l,
                    "wins_blocked_pct": round((n_wins - sub_w) / (n_wins + 1e-9) * 100, 1),
                    "losses_blocked_pct": round((n_losses - sub_l) / (n_losses + 1e-9) * 100, 1),
                    "new_wr": round(wr_new * 100, 1),
                    "new_pf": round(pf_new, 3),
                    "pf_improvement": round(pf_new - base_pf, 3),
                })
        all_candidates = rows2

    all_cand_df = pd.DataFrame(all_candidates).sort_values("new_pf", ascending=False)
    # Deduplicate by feature name
    all_cand_best = all_cand_df.drop_duplicates(subset=["feature"]).head(20)

    print(f"\nBaseline PF: {base_pf:.3f}  |  Baseline WR: {base_wr*100:.1f}%  |  N={n_trades}")
    print()

    top3 = all_cand_best.head(3).to_dict("records")
    for rank, r in enumerate(top3, 1):
        if_str = f"{r['direction']} {r['threshold']}" if r['threshold'] is not None else f"(time filter)"
        print(f"  FILTER #{rank}: {r['feature']} {if_str}")
        print(f"    Trades remaining:  {r['n_remain']} / {n_trades}")
        print(f"    Wins blocked:      {r['wins_blocked']} / {n_wins} ({r['wins_blocked_pct']}%)")
        print(f"    Losses blocked:    {r['losses_blocked']} / {n_losses} ({r['losses_blocked_pct']}%)")
        print(f"    New win rate:      {r['new_wr']}%")
        print(f"    New profit factor: {r['new_pf']}")
        print(f"    PF improvement:    +{r['pf_improvement']}")
        # Estimate return
        sub = df_t
        if r["direction"] == ">":
            sub = df_t[df_t[r["feature"]] >= r["threshold"]]
        elif r["direction"] == "<":
            sub = df_t[df_t[r["feature"]] <= r["threshold"]]
        new_ret = compute_return_pct(sub["net_pnl"].values) if len(sub) > 0 else 0
        print(f"    Estimated return:  {new_ret:.1f}%  (baseline 14461%)")
        print()

    # Show full top-20 candidates table
    print("\nFull top-20 candidates:")
    print(f"{'#':<3} {'Feature':<35} {'Dir':>3} {'Thresh':>10} {'Rem':>5} "
          f"{'W_blk%':>7} {'L_blk%':>7} {'NewWR%':>7} {'NewPF':>7} {'dPF':>7}")
    print("-" * 100)
    for i, (_, r) in enumerate(all_cand_best.iterrows(), 1):
        thresh_str = f"{r['threshold']:.4f}" if r["threshold"] is not None else "N/A"
        print(f"  {i:<3} {r['feature']:<33} {r['direction']:>3} {thresh_str:>10} "
              f"{r['n_remain']:>5} {r['wins_blocked_pct']:>6.1f}% "
              f"{r['losses_blocked_pct']:>6.1f}% {r['new_wr']:>6.1f}% "
              f"{r['new_pf']:>7.3f} {r['pf_improvement']:>+7.3f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
