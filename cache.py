"""
cache.py – Shared feature/label cache for train.py, backtest.py, auto_tune.py.

Cache files (in models/tuning/cache/)
--------------------------------------
  {coin}_{year}_{hash}.pkl
      Full feature DataFrame + 'label' column for one coin-year.
      Hash covers only features + labels config — changing trading or
      training params does NOT bust the cache.

  train_{coins}_{years}_{hash}.pkl
      Assembled X_train / y_train / X_val / y_val for all training
      coins × years (same split as train.py).

The cache directory is created automatically on first use.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import sys

ROOT      = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "models" / "tuning" / "cache"

# Add project root so this module can be imported from anywhere
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def feat_label_hash(cfg: dict) -> str:
    """
    MD5 hash of only the features + labels config sections.
    Trading / training params do NOT affect feature computation,
    so tuning T_up, break_weight etc. will NOT bust this cache.
    """
    relevant = {
        "features": cfg.get("features", {}),
        "labels":   cfg.get("labels",   {}),
    }
    blob = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
def load_coin_year(coin: str, year: int, cfg: dict,
                   no_cache: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    Load + compute features/labels for one coin-year.  Returns:
        df_all    – DataFrame with all feature columns + 'label'
        feat_cols – ordered list of model feature column names

    Cache key: {coin}_{year}_{feat_label_hash}.pkl
    First call:  ~30-40 s (feature computation)
    Subsequent:  <1 s
    """
    import data     as data_mod
    import features as feat_mod
    import labels   as label_mod

    _ensure_cache_dir()
    h          = feat_label_hash(cfg)
    cache_path = CACHE_DIR / f"{coin}_{year}_{h}.pkl"

    if not no_cache and cache_path.exists():
        print(f"  [cache] {coin} {year} <- {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Build a per-coin-year config that points to the right CSV path
    cfg_cy = copy.deepcopy(cfg)
    cfg_cy["data"]["years"]    = [year]
    cfg_cy["data"]["base_dir"] = (
        f"{cfg['data']['data_root']}/{coin}/full_year"
    )

    print(f"  [load]  {coin} {year} ...")
    df_raw  = data_mod.load_all(cfg_cy)
    print(f"  [feat]  {coin} {year} ...")
    df_feat = feat_mod.compute_features(df_raw, cfg)
    print(f"  [lab]   {coin} {year} ...")
    df_lab  = label_mod.compute_labels(df_raw, cfg)

    feat_cols = feat_mod.get_feature_columns(cfg)
    avail     = [c for c in feat_cols if c in df_feat.columns]

    # Join all feature cols (including atr_short, dist_rh_20 etc.) + label.
    # Drop only rows where the label is NaN (last H_horizon bars).
    df_all = (df_feat[avail]
              .join(df_lab[["label"]], how="inner")
              .dropna(subset=["label"]))
    df_all["label"] = df_all["label"].astype(int)

    print(f"  [done]  {coin} {year}: {len(df_all):,} rows")
    result = (df_all, avail)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


# ─────────────────────────────────────────────────────────────────────────────
def load_years_parallel(coin: str, years: list[int], cfg: dict,
                        no_cache: bool = False,
                        max_workers: int = 4) -> dict[int, tuple]:
    """
    Load multiple years for one coin in parallel threads.
    Returns {year: (df_all, feat_cols)}.
    """
    results: dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=min(len(years), max_workers)) as pool:
        futures = {
            pool.submit(load_coin_year, coin, yr, cfg, no_cache): yr
            for yr in years
        }
        for fut in as_completed(futures):
            yr = futures[fut]
            results[yr] = fut.result()
    return results


# ─────────────────────────────────────────────────────────────────────────────
def load_training_data(cfg: dict, no_cache: bool = False) -> dict:
    """
    Load feature+label data for ALL training coins and ALL years
    (matching train.py exactly) and assemble X/y train-val splits.

    Returns dict:
        X_train, y_train, X_val, y_val  – numpy/pandas ready for XGBoost
        feat_cols                        – feature column list

    Cache key: train_{coins}_{years}_{hash}.pkl
    First call:  ~2 min with parallel loading  (~6 min without)
    Subsequent:  ~5 s
    """
    import features as feat_mod

    _ensure_cache_dir()
    h     = feat_label_hash(cfg)
    coins = cfg["data"].get("coins", [cfg["data"]["symbol"]])
    years = cfg["data"]["years"]

    coin_str   = "-".join(sorted(str(c) for c in coins))
    year_str   = "-".join(str(y) for y in sorted(years))
    cache_path = CACHE_DIR / f"train_{coin_str}_{year_str}_{h}.pkl"

    if not no_cache and cache_path.exists():
        print(f"  [cache] Training data <- {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # ── Load all coin×year pairs in parallel ─────────────────────────────────
    tasks = [(coin, yr) for coin in coins for yr in years]
    print(f"  Loading {len(tasks)} coin-year datasets "
          f"({len(coins)} coins x {len(years)} years) in parallel ...")

    pieces: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as pool:
        futures = {
            pool.submit(load_coin_year, coin, yr, cfg, no_cache): (coin, yr)
            for coin, yr in tasks
        }
        for fut in as_completed(futures):
            coin_f, yr_f = futures[fut]
            try:
                df_all, _ = fut.result()
                pieces.append(df_all)
            except Exception as exc:
                print(f"  [warn]  {coin_f} {yr_f} failed: {exc}")

    if not pieces:
        raise RuntimeError("No training data loaded — check data paths.")

    df_combined = pd.concat(pieces).sort_index()

    feat_cols_all = feat_mod.get_feature_columns(cfg)
    fc_avail      = [c for c in feat_cols_all if c in df_combined.columns]

    # Drop rows with any NaN in feature columns
    df_combined = df_combined.dropna(subset=fc_avail)

    X_all = df_combined[fc_avail]
    y_all = df_combined["label"].astype(int).values

    split  = int(len(X_all) * (1 - cfg["training"]["test_size"]))
    result = {
        "X_train":   X_all.iloc[:split],
        "y_train":   y_all[:split],
        "X_val":     X_all.iloc[split:],
        "y_val":     y_all[split:],
        "feat_cols": fc_avail,
    }
    print(f"  Train: {len(result['X_train']):,}  Val: {len(result['X_val']):,}  "
          f"Features: {len(fc_avail)}")

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  [cache] Saved -> {cache_path.name}")
    return result
