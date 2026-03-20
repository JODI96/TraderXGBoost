"""
auto_tune.py – Deep Analysis + Auto-Tuner for the BTC 1-min Scalping System.

Phases
------
1. Data loading + feature/label computation (disk-cached per coin per year)
2. Feature importance analysis (gain / weight / cover)
3. Year-by-year baseline (prediction-only, no execution filters)
4. Sensitivity analysis (T_up / T_down sweep)
5. Auto-tuner  -- fast mode: only T_up/T_down (seconds/trial, probs cached)
               -- full mode: XGBoost params + retrain (uses cached training data)
6. Final report

Cache files
-----------
  models/tuning/cache/{coin}_{year}_{feat_label_hash}.pkl   feature+label DataFrame
  models/tuning/cache/train_{coins}_{years}_{hash}.pkl      assembled X/y splits
  models/tuning/cache/probs_{year}_{model_mtime}.npy        pre-computed probabilities

Cache hash only covers features + labels config sections.  Changing trading params
(T_up, break_weight, etc.) does NOT invalidate feature caches.

Usage
-----
python auto_tune.py --analysis-only
python auto_tune.py --mode fast --trials 100
python auto_tune.py --mode full --trials 20
python auto_tune.py --resume --trials 100
python auto_tune.py --no-cache
python auto_tune.py --year 2024 --analysis-only
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import features as feat_mod
from backtest import run_backtest
from train import walk_forward_cv, train_final
from cache import (
    load_coin_year,
    load_years_parallel,
    load_training_data,
    feat_label_hash as _feat_label_hash,
)

# ── Output dirs ────────────────────────────────────────────────────────────────
TUNING_DIR = ROOT / "models" / "tuning"
CACHE_DIR  = TUNING_DIR / "cache"
BEST_DIR   = ROOT / "models" / "best_tuned"

for _d in (TUNING_DIR, CACHE_DIR, BEST_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _model_mtime(art_dir: str = "models") -> int:
    """Integer mtime of xgb_model.json — used to key the probability cache."""
    p = Path(art_dir) / "xgb_model.json"
    return int(p.stat().st_mtime) if p.exists() else 0


# =============================================================================
# Model helpers
# =============================================================================

def _load_model(art_dir: str = "models") -> tuple[xgb.Booster, list[str], dict]:
    p = Path(art_dir)
    model = xgb.Booster()
    model.load_model(str(p / "xgb_model.json"))
    try:
        model.set_param({"device": "cuda"})
    except Exception:
        pass
    with open(p / "feature_columns.json") as f:
        feat_cols = json.load(f)
    with open(p / "thresholds.json") as f:
        thresholds = json.load(f)
    return model, feat_cols, thresholds


def _predict(model: xgb.Booster, df: pd.DataFrame,
             feat_cols: list[str]) -> np.ndarray:
    avail = [c for c in feat_cols if c in df.columns]
    dm = xgb.DMatrix(df[avail].values, feature_names=avail)
    return model.predict(dm).reshape(-1, 3)


def _prediction_only_cfg(cfg: dict) -> dict:
    """Deep copy of cfg with all execution filters disabled."""
    c = copy.deepcopy(cfg)
    c["trading"]["require_squeeze"] = False
    c["trading"]["cooldown"]        = 0
    return c


# =============================================================================
# Phase 1: Data loading (delegates to cache.py)
# =============================================================================


# =============================================================================
# Phase 1c: Probability cache (fast mode — reuse across restarts)
# =============================================================================

def get_probs_cached(model: xgb.Booster, df_all: pd.DataFrame,
                     feat_cols: list[str], coin: str, year: int,
                     mtime: int, no_cache: bool = False) -> np.ndarray:
    """Return cached probabilities or compute + cache them."""
    cache_path = CACHE_DIR / f"probs_{coin}_{year}_{mtime}.npy"
    if not no_cache and cache_path.exists():
        print(f"  [cache] Probs {coin} {year} <- {cache_path.name}")
        return np.load(cache_path)
    print(f"  [pred]  Computing probs for {coin} {year} ...")
    probs = _predict(model, df_all, feat_cols)
    np.save(cache_path, probs)
    return probs


# =============================================================================
# Phase 2: Feature Importance
# =============================================================================

def analyze_features(model: xgb.Booster,
                     feature_cols: list[str]) -> pd.DataFrame:
    gain   = model.get_score(importance_type="gain")
    weight = model.get_score(importance_type="weight")
    cover  = model.get_score(importance_type="cover")

    rows = [{"feature": f,
             "gain":    gain.get(f, 0.0),
             "weight":  weight.get(f, 0.0),
             "cover":   cover.get(f, 0.0)}
            for f in feature_cols]

    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    total = df["gain"].sum()
    df["gain_pct"]     = df["gain"] / (total + 1e-12) * 100
    df["cum_gain_pct"] = df["gain_pct"].cumsum()

    out = TUNING_DIR / "feature_importance.csv"
    df.to_csv(out, index=False, float_format="%.4f")

    W = 72
    print(f"\n{'=' * W}")
    print("  FEATURE IMPORTANCE (sorted by gain)")
    print(f"{'=' * W}")
    print(f"  {'#':>3}  {'Feature':<35}  {'Gain%':>6}  {'CumGain%':>9}  {'Weight':>7}")
    print(f"  {'-' * 68}")
    print("  -- TOP 20 --")
    for i, row in df.head(20).iterrows():
        mark = " <" if row["gain_pct"] < 0.5 else ""
        print(f"  {i+1:>3}  {row['feature']:<35}  {row['gain_pct']:>6.2f}%"
              f"  {row['cum_gain_pct']:>8.1f}%  {int(row['weight']):>7}{mark}")

    removable = df[df["gain_pct"] < 0.5]["feature"].tolist()
    if removable:
        print(f"\n  -- REMOVAL CANDIDATES (gain < 0.5%, {len(removable)} features) --")
        for i, row in df[df["gain_pct"] < 0.5].iterrows():
            print(f"  {i+1:>3}  {row['feature']:<35}  {row['gain_pct']:>6.2f}%")

    print(f"\n  Saved: {out}")
    print(f"{'=' * W}")
    return df


# =============================================================================
# Phase 3: Year-by-year backtest helper
# =============================================================================

def run_year_backtest(df_all: pd.DataFrame, probs: np.ndarray,
                      cfg: dict, T_up: float, T_down: float) -> dict:
    po_cfg = _prediction_only_cfg(cfg)
    _, _, report = run_backtest(
        df_feat     = df_all,
        probs       = probs,
        cfg         = po_cfg,
        T_up        = T_up,
        T_down      = T_down,
        d_max       = 99.0,
        min_vol     = 0.0,
        min_ema9_21 = -99.0,
    )
    return report


def print_baseline(years_results: dict[int, dict], label: str = "BASELINE") -> None:
    W = 72
    print(f"\n{'=' * W}")
    print(f"  {label}")
    print(f"{'=' * W}")
    print(f"  {'Year':>4}  {'Trades':>6}  {'WR%':>5}  {'PF':>5}  "
          f"{'Sharpe':>6}  {'MaxDD%':>7}  {'Ret%':>7}")
    print(f"  {'-' * 60}")
    for yr, rep in sorted(years_results.items()):
        if "error" in rep:
            print(f"  {yr:>4}  (no trades)")
            continue
        print(f"  {yr:>4}  {rep['n_trades']:>6}  {rep['win_rate']:>5.1f}  "
              f"{rep['profit_factor']:>5.2f}  {rep['sharpe_ratio']:>6.2f}  "
              f"{rep['max_drawdown_pct']:>7.1f}  {rep['total_return_pct']:>+7.1f}%")
    print(f"{'=' * W}")


# =============================================================================
# Phase 4: Sensitivity analysis
# =============================================================================

def sensitivity_analysis(years_data: dict[int, tuple],
                          fast_probs: dict[int, np.ndarray],
                          cfg: dict) -> None:
    T_values = [0.50, 0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.67,
                0.68, 0.70, 0.75, 0.80]
    years    = sorted(years_data.keys())

    rows = []
    W = 72
    print(f"\n{'=' * W}")
    print("  SENSITIVITY: T_up / T_down sweep  (symmetric, prediction-only)")
    print(f"{'=' * W}")
    hdr = "  " + "  ".join(
        f"{'PF_'+str(y):>6}  {'N_'+str(y):>5}  {'Sh_'+str(y):>5}"
        for y in years
    )
    print(f"  {'T':>5}  " + hdr.strip())
    print(f"  {'-' * 66}")

    for T in T_values:
        row: dict = {"T": T}
        parts = [f"  {T:.2f}  "]
        for yr in years:
            df_all = years_data[yr][0]
            probs  = fast_probs[yr]
            rep = run_year_backtest(df_all, probs, cfg, T_up=T, T_down=T)
            if "error" in rep:
                pf, n, sh = 0.0, 0, 0.0
            else:
                pf = rep["profit_factor"]
                n  = rep["n_trades"]
                sh = rep["sharpe_ratio"]
            row[f"PF_{yr}"] = pf
            row[f"N_{yr}"]  = n
            row[f"Sh_{yr}"] = sh
            parts.append(f"  {pf:>6.2f}  {n:>5}  {sh:>5.2f}")
        print("".join(parts))
        rows.append(row)

    out = TUNING_DIR / "sensitivity_T.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print(f"\n  Saved: {out}")
    print(f"{'=' * W}")


# =============================================================================
# Phase 5: Auto-Tuner
# =============================================================================

TRIAL_CSV = TUNING_DIR / "trial_history.csv"
BEST_JSON = TUNING_DIR / "best_params.json"

_TRIAL_FIELDS = [
    "trial", "score", "T_up", "T_down",
    "PF_2023", "PF_2024", "PF_2025",
    "N_2023",  "N_2024",  "N_2025",
    "Sh_2023", "Sh_2024", "Sh_2025",
    "break_weight", "max_depth", "min_child_weight",
    "subsample", "colsample_bytree", "gamma",
    "reg_alpha", "reg_lambda", "learning_rate",
    "k_atr", "H_horizon", "L_range", "L_atr", "sl_pct", "rr_ratio",
]


def _score(reports: dict[int, dict], min_trades: int = 60,
           max_pf_spread: float = 1.0) -> float:
    """Average profit factor across all years (equal weight).
    Zero if any year fails PF < 1.0 or trades < min.
    Zero if max(PF) - min(PF) across years exceeds max_pf_spread — keeps all
    years in a consistent range rather than letting one year dominate."""
    pfs = []
    for yr, rep in reports.items():
        if "error" in rep:
            return 0.0
        pf = rep["profit_factor"]
        n  = rep["n_trades"]
        if pf < 1.0 or n < min_trades:
            return 0.0
        pfs.append(pf)
    if not pfs:
        return 0.0
    if max(pfs) - min(pfs) > max_pf_spread:
        return 0.0
    return sum(pfs) / len(pfs)


def _append_trial(trial_num: int, params: dict, score: float,
                  reports: dict[int, dict]) -> None:
    write_header = not TRIAL_CSV.exists()
    with open(TRIAL_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TRIAL_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        row: dict = {
            "trial": trial_num, "score": round(score, 6),
            "T_up": params.get("T_up"), "T_down": params.get("T_down"),
        }
        for yr in (2023, 2024, 2025):
            rep = reports.get(yr, {})
            row[f"PF_{yr}"] = rep.get("profit_factor", 0.0)
            row[f"N_{yr}"]  = rep.get("n_trades", 0)
            row[f"Sh_{yr}"] = rep.get("sharpe_ratio", 0.0)
        for k in ("break_weight", "max_depth", "min_child_weight",
                  "subsample", "colsample_bytree", "gamma",
                  "reg_alpha", "reg_lambda", "learning_rate"):
            row[k] = params.get(k, "")
        w.writerow(row)


def _save_best(model: xgb.Booster, feat_cols: list[str], params: dict,
               score: float, reports: dict[int, dict]) -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(BEST_DIR / "xgb_model.json"))
    with open(BEST_DIR / "feature_columns.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    with open(BEST_DIR / "thresholds.json", "w") as f:
        json.dump({
            "T_up":    params["T_up"],
            "T_down":  params["T_down"],
            "sl_pct":  params.get("sl_pct"),
            "tp_pct":  params.get("sl_pct", 0) * params.get("rr_ratio", 3.0)
                       if params.get("sl_pct") else None,
            "classes": {0: "DOWN_BREAK_SOON", 1: "NO_BREAK", 2: "UP_BREAK_SOON"},
        }, f, indent=2)
    with open(BEST_JSON, "w") as f:
        json.dump({"score": score, "params": params,
                   "year_reports": {str(yr): r for yr, r in reports.items()}},
                  f, indent=2, default=str)
    print(f"  [best]  score={score:.4f}  -> {BEST_DIR}")


def auto_tune(years_data: dict[int, tuple],
              fast_probs: dict[int, np.ndarray],
              model_base: xgb.Booster,
              feat_cols: list[str],
              cfg: dict,
              n_trials: int = 100,
              mode: str = "fast",
              resume: bool = False,
              no_cache: bool = False,
              n_jobs: int = 1) -> dict:
    """
    Optuna-based tuner (random search fallback if Optuna not installed).
    fast: tune T_up/T_down only — probs pre-computed, ~seconds/trial.
    full: also tune XGBoost params and retrain — uses cached training data.
    """
    years = sorted(years_data.keys())

    # Full mode loads training data inside each trial (label params vary per trial)

    best_score  = -1.0
    best_params: dict = {}
    best_model  = model_base
    trial_counter = [0]
    _lock = threading.Lock()   # protects shared state in parallel trials

    # ── Objectives ────────────────────────────────────────────────────────────
    def objective_fast(trial) -> float:
        T_up   = trial.suggest_float("T_up",   0.50, 0.90)
        T_down = trial.suggest_float("T_down", 0.50, 0.90)
        reports: dict[int, dict] = {}
        for yr in years:
            rep = run_year_backtest(years_data[yr][0], fast_probs[yr],
                                    cfg, T_up, T_down)
            reports[yr] = rep
            if rep.get("profit_factor", 0.0) < 0.85 or rep.get("n_trades", 0) < 40:
                try:
                    import optuna
                    raise optuna.exceptions.TrialPruned()
                except ImportError:
                    break
        spread = cfg.get("tuning", {}).get("max_pf_spread", 1.0)
        return _score(reports, max_pf_spread=spread), reports

    def objective_full(trial) -> tuple[float, dict, xgb.Booster]:
        # ── XGBoost params ────────────────────────────────────────────────────
        T_up   = trial.suggest_float("T_up",            0.50, 0.90)
        T_down = trial.suggest_float("T_down",          0.50, 0.90)
        bw     = trial.suggest_float("break_weight",    4.0,  20.0)
        md     = trial.suggest_int  ("max_depth",       3,    7)
        mcw    = trial.suggest_int  ("min_child_weight",5,    50)
        sub    = trial.suggest_float("subsample",       0.4,  1.0)
        col    = trial.suggest_float("colsample_bytree",0.3,  1.0)
        gam    = trial.suggest_float("gamma",           0.0,  5.0)
        al     = trial.suggest_float("reg_alpha",       0.0,  10.0)
        la     = trial.suggest_float("reg_lambda",      0.1,  10.0)
        lr     = trial.suggest_float("learning_rate",   0.005,0.05)
        # ── Label params ──────────────────────────────────────────────────────
        k_atr     = trial.suggest_int  ("k_atr",      1,    15)
        H_horizon = trial.suggest_int  ("H_horizon",  10,   40)
        L_range   = trial.suggest_int  ("L_range",    10,   40)
        L_atr     = trial.suggest_int  ("L_atr",      3,    14)
        sl_pct    = trial.suggest_float("sl_pct",     0.001, 0.003)
        rr_ratio  = trial.suggest_float("rr_ratio",   2.0,   6.0)
        tp_pct    = sl_pct * rr_ratio

        cfg_trial = copy.deepcopy(cfg)
        # XGBoost
        cfg_trial["training"]["break_weight"]                   = bw
        cfg_trial["training"]["xgb_params"]["max_depth"]        = md
        cfg_trial["training"]["xgb_params"]["min_child_weight"] = mcw
        cfg_trial["training"]["xgb_params"]["subsample"]        = sub
        cfg_trial["training"]["xgb_params"]["colsample_bytree"] = col
        cfg_trial["training"]["xgb_params"]["gamma"]            = gam
        cfg_trial["training"]["xgb_params"]["reg_alpha"]        = al
        cfg_trial["training"]["xgb_params"]["reg_lambda"]       = la
        cfg_trial["training"]["xgb_params"]["learning_rate"]    = lr
        # Label params
        cfg_trial["labels"]["k_atr"]                = k_atr
        cfg_trial["labels"]["H_horizon"]            = H_horizon
        cfg_trial["labels"]["L_range"]              = L_range
        cfg_trial["labels"]["L_atr"]                = L_atr
        cfg_trial["labels"]["sl_pct"]               = sl_pct
        cfg_trial["labels"]["tp_pct"]               = tp_pct
        cfg_trial["training"]["gap_candles"]        = H_horizon  # must be >= H_horizon
        # Sync trading sl/tp with labels so backtest uses same barriers
        cfg_trial["trading"]["sl_pct"] = sl_pct
        cfg_trial["trading"]["tp_pct"] = tp_pct

        # Load training data for this trial's label config (cached by config hash)
        print(f"\n  [full] Loading data "
              f"(k_atr={k_atr}, H={H_horizon}, L_range={L_range}, "
              f"sl={sl_pct:.4f}, rr={rr_ratio:.1f}) ...")
        train_data      = load_training_data(cfg_trial, no_cache=no_cache)
        X_train         = train_data["X_train"]
        y_train         = train_data["y_train"]
        X_val           = train_data["X_val"]
        y_val           = train_data["y_val"]
        trial_feat_cols = train_data["feat_cols"]
        print(f"  [full] CV ...")
        best_n, _ = walk_forward_cv(X_train, y_train, cfg_trial)
        print(f"  [full] Training (n={best_n}) ...")
        m = train_final(X_train, y_train, X_val, y_val, best_n, cfg_trial)

        reports: dict[int, dict] = {}
        for yr in years:
            probs = _predict(m, years_data[yr][0], trial_feat_cols)
            rep   = run_year_backtest(years_data[yr][0], probs, cfg_trial, T_up, T_down)
            reports[yr] = rep
            if rep.get("profit_factor", 0.0) < 0.85 or rep.get("n_trades", 0) < 40:
                break
        spread = cfg.get("tuning", {}).get("max_pf_spread", 1.0)
        return _score(reports, max_pf_spread=spread), reports, m

    # ── Wrapped objective (logging + best tracking, thread-safe) ─────────────
    def wrapped_objective(trial) -> float:
        nonlocal best_score, best_params, best_model
        t0 = time.time()

        if mode == "fast":
            sc, reports = objective_fast(trial)
            params = {"T_up": trial.params["T_up"], "T_down": trial.params["T_down"]}
        else:
            sc, reports, m = objective_full(trial)
            params = dict(trial.params)

        elapsed = time.time() - t0

        with _lock:
            trial_counter[0] += 1
            tn = trial_counter[0]
            _append_trial(tn, params, sc, reports)

            def _pf(yr): return reports.get(yr, {}).get("profit_factor", 0.0)
            print(f"  trial {tn:>4}  score={sc:.4f}  "
                  f"PF 23={_pf(2023):.2f}  24={_pf(2024):.2f}  25={_pf(2025):.2f}  "
                  f"T_up={params.get('T_up', 0):.3f}  T_dn={params.get('T_down', 0):.3f}"
                  f"  ({elapsed:.1f}s)")

            if sc > best_score:
                best_score  = sc
                best_params = dict(params)
                if mode == "full":
                    best_model = m
                _save_best(best_model, feat_cols, params, sc, reports)

        return sc

    # ── Optuna (with random-search fallback) ─────────────────────────────────
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        db_path = TUNING_DIR / "optuna.db"
        study   = optuna.create_study(
            direction      = "maximize",
            storage        = f"sqlite:///{db_path}",
            study_name     = "btc_scalper",
            load_if_exists = True,
            pruner         = optuna.pruners.MedianPruner(
                                 n_startup_trials=10, n_warmup_steps=1),
        )
        done = len(study.trials)
        if resume and done > 0:
            print(f"  [optuna] Resuming — {done} trials already done.")
        else:
            print(f"  [optuna] Starting study (DB: {db_path.name}).")

        study.optimize(wrapped_objective, n_trials=n_trials,
                       n_jobs=n_jobs, show_progress_bar=False)
        best_params = study.best_trial.params
        best_score  = study.best_trial.value

    except ImportError:
        print("  [warn] optuna not installed — using random search.")
        _random_search(wrapped_objective, n_trials, mode)

    return best_params


def _random_search(objective_fn, n_trials: int, mode: str) -> None:
    class _FakeTrial:
        def __init__(self):
            self.params: dict = {}
        def suggest_float(self, name, lo, hi):
            v = random.uniform(lo, hi); self.params[name] = v; return v
        def suggest_int(self, name, lo, hi):
            v = random.randint(lo, hi); self.params[name] = v; return v
        def report(self, *_): pass

    for _ in range(n_trials):
        try:
            objective_fn(_FakeTrial())
        except Exception as exc:
            print(f"  [warn] trial error: {exc}")


# =============================================================================
# Phase 6: Final Report
# =============================================================================

def print_final_report(best_params: dict, feat_df: pd.DataFrame,
                        years_data: dict[int, tuple],
                        fast_probs: dict[int, np.ndarray],
                        model: xgb.Booster,
                        feat_cols: list[str],
                        cfg: dict) -> None:
    W = 72
    lines: list[str] = []

    def p(*args):
        s = " ".join(str(a) for a in args)
        print(s); lines.append(s)

    p(f"\n{'=' * W}")
    p("  FINAL REPORT")
    p(f"{'=' * W}")

    if best_params:
        p("\n  Best Parameters:")
        for k, v in best_params.items():
            p(f"    {k:<25}: {v}")

        T_up   = best_params.get("T_up",   cfg["trading"]["T_up"])
        T_down = best_params.get("T_down", cfg["trading"]["T_down"])
        p(f"\n  Performance (T_up={T_up:.3f}, T_down={T_down:.3f}):")
        p(f"  {'Year':>4}  {'Trades':>6}  {'WR%':>5}  {'PF':>5}  "
          f"{'Sharpe':>6}  {'MaxDD%':>7}  {'Ret%':>7}")
        p(f"  {'-' * 60}")
        for yr, (df_all, _) in sorted(years_data.items()):
            probs = fast_probs.get(yr) or _predict(model, df_all, feat_cols)
            rep   = run_year_backtest(df_all, probs, cfg, T_up, T_down)
            if "error" in rep:
                p(f"  {yr:>4}  (no trades)")
            else:
                p(f"  {yr:>4}  {rep['n_trades']:>6}  {rep['win_rate']:>5.1f}  "
                  f"{rep['profit_factor']:>5.2f}  {rep['sharpe_ratio']:>6.2f}  "
                  f"{rep['max_drawdown_pct']:>7.1f}  "
                  f"{rep['total_return_pct']:>+7.1f}%")

    if not feat_df.empty:
        removable = feat_df[feat_df["gain_pct"] < 0.5]["feature"].tolist()
        p(f"\n  Top 20 features to KEEP:")
        for i, f in enumerate(feat_df.head(20)["feature"], 1):
            p(f"    {i:>2}. {f}")
        if removable:
            p(f"\n  Removal candidates (gain < 0.5%, {len(removable)} features):")
            p("    " + ", ".join(removable))

    p(f"\n{'=' * W}")
    out = TUNING_DIR / "analysis_report.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n  Report -> {out}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep analysis + auto-tuner for BTC 1-min scalping system")
    parser.add_argument("--mode",          choices=["fast", "full"], default="fast")
    parser.add_argument("--trials",        type=int, default=100)
    parser.add_argument("--resume",        action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--year",          type=int, default=None,
                        help="backtest years to analyze (default: 2023 2024 2025)")
    parser.add_argument("--jobs",          type=int, default=1,
                        help="parallel Optuna trials (fast mode only, default: 1)")
    parser.add_argument("--config",        default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bt_years = [args.year] if args.year else [2023, 2024, 2025]
    coin     = cfg["data"]["symbol"]   # BTCUSDT — backtest coin

    W = 72
    print(f"\n{'=' * W}")
    print("  AUTO_TUNE  --  BTC 1-min Scalping System")
    print(f"  Mode: {args.mode}  |  Trials: {args.trials}  |  "
          f"Backtest years: {bt_years}  |  Train coin: {coin}")
    coins_cfg = cfg["data"].get("coins", [coin])
    print(f"  Training coins: {coins_cfg}  |  "
          f"Training years: {cfg['data']['years']}")
    print(f"{'=' * W}")

    # ── Phase 1: Load backtest years in parallel (BTCUSDT) ────────────────────
    print(f"\n[Phase 1] Loading backtest data ({coin}, {bt_years}) in parallel ...")
    t0 = time.time()
    years_data = load_years_parallel(coin, bt_years, cfg, no_cache=args.no_cache)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Phase 2: Load model + feature importance ──────────────────────────────
    print(f"\n[Phase 2] Loading model + analyzing features ...")
    model, feat_cols, thresholds = _load_model("models")
    mtime  = _model_mtime("models")
    feat_df = analyze_features(model, feat_cols)

    # ── Phase 2b: Pre-compute / load cached probabilities ────────────────────
    # Probs are always cached on the FULL year (cache reuse across runs).
    # 2025 is then trimmed to the unseen test portion after caching.
    print(f"\n  Pre-computing probabilities ...")
    fast_probs: dict[int, np.ndarray] = {}
    for yr in bt_years:
        fast_probs[yr] = get_probs_cached(
            model, years_data[yr][0], feat_cols,
            coin, yr, mtime, no_cache=args.no_cache)

    # ── Phase 1b: Trim 2025 to unseen test portion only ──────────────────────
    if 2025 in years_data:
        print(f"\n  Determining test split cutoff (loading training data) ...")
        train_data = load_training_data(cfg, no_cache=args.no_cache)
        test_start = train_data["X_val"].index[0]
        df_2025, fc_2025 = years_data[2025]
        mask_2025        = df_2025.index >= test_start
        years_data[2025] = (df_2025[mask_2025], fc_2025)
        fast_probs[2025] = fast_probs[2025][mask_2025]
        print(f"  2025 trimmed to unseen: {mask_2025.sum():,} rows "
              f"(from {test_start})")

    # ── Phase 3: Year-by-year baseline ───────────────────────────────────────
    print(f"\n[Phase 3] Year-by-year baseline (prediction-only) ...")
    T_base = thresholds.get("T_up", cfg["trading"]["T_up"])
    baseline = {yr: run_year_backtest(years_data[yr][0], fast_probs[yr],
                                      cfg, T_base, T_base)
                for yr in bt_years}
    print_baseline(baseline,
                   f"YEAR-BY-YEAR BASELINE (prediction-only, T={T_base:.2f})")

    # ── Phase 4: Sensitivity analysis ────────────────────────────────────────
    print(f"\n[Phase 4] Sensitivity analysis ...")
    sensitivity_analysis(years_data, fast_probs, cfg)

    if args.analysis_only:
        print("\n  --analysis-only flag set. Stopping before auto-tuner.")
        print_final_report({}, feat_df, years_data, fast_probs, model, feat_cols, cfg)
        return

    # ── Phase 5: Auto-tuner ───────────────────────────────────────────────────
    print(f"\n[Phase 5] Auto-tuner ({args.mode} mode, {args.trials} trials) ...")
    best_params = auto_tune(
        years_data = years_data,
        fast_probs = fast_probs,
        model_base = model,
        feat_cols  = feat_cols,
        cfg        = cfg,
        n_trials   = args.trials,
        mode       = args.mode,
        resume     = args.resume,
        no_cache   = args.no_cache,
        n_jobs     = args.jobs if args.mode == "fast" else 1,
    )

    # ── Phase 6: Final report ─────────────────────────────────────────────────
    print(f"\n[Phase 6] Final report ...")
    print_final_report(best_params, feat_df, years_data, fast_probs,
                       model, feat_cols, cfg)

    print(f"\nDone. Artifacts:")
    print(f"  {TUNING_DIR}")
    print(f"  {BEST_DIR}")


if __name__ == "__main__":
    main()
