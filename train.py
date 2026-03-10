"""
train.py – XGBoost training pipeline.

Steps:
  1. Load full 3-year 1m dataset.
  2. Compute features + labels (single aligned DataFrame).
  3. Temporal 80/20 train/test split.
  4. Walk-forward CV on training set (early stopping each fold).
  5. Final model trained on full train set with best n_estimators.
  6. Save artifacts to models/ directory.

Artifacts
---------
  models/xgb_model.json         – trained model
  models/feature_columns.json   – ordered list of feature names
  models/thresholds.json        – default probability thresholds + metadata
  models/label_meta.json        – class counts, warmup length, etc.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, classification_report

import data as data_mod
import features as feat_mod
import labels as label_mod


# ─────────────────────────────────────────────────────────────────────────────
def build_sample_weights(y: np.ndarray, break_weight: float) -> np.ndarray:
    """
    Upweight minority break classes relative to NO_BREAK.
    Class 1 (NO_BREAK) -> weight 1.0
    Class 0 / 2        -> weight break_weight
    """
    w = np.where(y == 1, 1.0, break_weight).astype(np.float32)
    return w


def make_dmatrix(X: pd.DataFrame, y: np.ndarray,
                 w: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(X.values, label=y,
                       weight=w, feature_names=list(X.columns))


# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_cv(X_train: pd.DataFrame, y_train: np.ndarray,
                    cfg: dict) -> tuple[int, list[dict]]:
    """
    Rolling walk-forward cross-validation on the training set.
    Returns (best_n_estimators, fold_metrics_list).
    """
    tc      = cfg["training"]
    xp      = dict(tc["xgb_params"])
    n_split = tc["n_wf_splits"]
    gap     = tc["gap_candles"]
    es      = tc["early_stopping_rounds"]
    bw      = tc["break_weight"]

    tscv = TimeSeriesSplit(n_splits=n_split, gap=gap)

    best_rounds_per_fold: list[int] = []
    fold_metrics: list[dict]        = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        Xtr = X_train.iloc[tr_idx]
        ytr = y_train[tr_idx]
        Xvl = X_train.iloc[val_idx]
        yvl = y_train[val_idx]

        wtr = build_sample_weights(ytr, bw)
        wvl = build_sample_weights(yvl, bw)   # keep consistent with training weights

        dtr  = make_dmatrix(Xtr, ytr, wtr)
        dvl  = make_dmatrix(Xvl, yvl, wvl)

        params = {k: v for k, v in xp.items()
                  if k not in ("n_estimators",)}

        evals_result: dict = {}
        model = xgb.train(
            params,
            dtr,
            num_boost_round=xp["n_estimators"],
            evals=[(dvl, "val")],
            early_stopping_rounds=es,
            evals_result=evals_result,
            verbose_eval=False,
        )

        n_rounds = model.best_iteration + 1
        best_rounds_per_fold.append(n_rounds)

        # Evaluate
        probs = model.predict(dvl).reshape(-1, xp["num_class"])
        preds = probs.argmax(axis=1)
        ll    = log_loss(yvl, probs)

        rep = classification_report(yvl, preds,
                                    target_names=["DOWN", "NO_BREAK", "UP"],
                                    output_dict=True, zero_division=0)
        metrics = {
            "fold": fold,
            "n_rounds": n_rounds,
            "val_logloss": round(ll, 4),
            "f1_DOWN":    round(rep["DOWN"]["f1-score"], 4),
            "f1_UP":      round(rep["UP"]["f1-score"],   4),
            "f1_macro":   round(rep["macro avg"]["f1-score"], 4),
        }
        fold_metrics.append(metrics)
        print(f"  Fold {fold}: n_rounds={n_rounds}  logloss={ll:.4f}  "
              f"f1_macro={metrics['f1_macro']:.4f}  "
              f"f1_DOWN={metrics['f1_DOWN']:.4f}  "
              f"f1_UP={metrics['f1_UP']:.4f}")

    # Use max of non-degenerate folds (those where model actually learned UP/DOWN).
    # With high break_weight, a few folds may have degenerate val loss curves and
    # stop at 1-3 rounds — don't let those pull the estimate down.
    valid_rounds = [
        m["n_rounds"] for m in fold_metrics
        if m["f1_DOWN"] > 0 or m["f1_UP"] > 0
    ]
    if valid_rounds:
        best_n = int(np.max(valid_rounds))
        print(f"\n  WF-CV best_n_estimators (max of {len(valid_rounds)} valid folds): {best_n}")
    else:
        best_n = int(np.max(best_rounds_per_fold))
        print(f"\n  WF-CV best_n_estimators (max, all folds degenerate): {best_n}")
    return best_n, fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
def train_final(X_train: pd.DataFrame, y_train: np.ndarray,
                X_val: pd.DataFrame, y_val: np.ndarray,
                n_estimators: int, cfg: dict) -> xgb.Booster:
    """Train final model on full training set with the given n_estimators."""
    tc  = cfg["training"]
    xp  = dict(tc["xgb_params"])
    bw  = tc["break_weight"]

    params = {k: v for k, v in xp.items() if k not in ("n_estimators",)}

    wtr = build_sample_weights(y_train, bw)
    dtr = make_dmatrix(X_train, y_train, wtr)
    dvl = make_dmatrix(X_val,   y_val)

    evals_result: dict = {}
    model = xgb.train(
        params,
        dtr,
        num_boost_round=n_estimators,
        evals=[(dtr, "train"), (dvl, "val")],
        evals_result=evals_result,
        verbose_eval=100,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_test(model: xgb.Booster, X_test: pd.DataFrame,
                     y_test: np.ndarray, cfg: dict) -> dict:
    """Quick evaluation on hold-out test set."""
    n_classes = cfg["training"]["xgb_params"]["num_class"]
    dtest = make_dmatrix(X_test, y_test)
    probs = model.predict(dtest).reshape(-1, n_classes)
    preds = probs.argmax(axis=1)
    ll    = log_loss(y_test, probs)
    rep   = classification_report(y_test, preds,
                                  target_names=["DOWN", "NO_BREAK", "UP"],
                                  output_dict=True, zero_division=0)
    print("\n--- Hold-out Test Results ---")
    print(f"  Log-loss : {ll:.4f}")
    print(classification_report(y_test, preds,
                                target_names=["DOWN", "NO_BREAK", "UP"],
                                zero_division=0))
    return {"logloss": ll, "report": rep, "probs": probs}


# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(model: xgb.Booster, feature_cols: list[str],
                   cfg: dict, meta: dict) -> None:
    art_dir = Path(cfg["training"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(art_dir / "xgb_model.json"))
    print(f"  Model   -> {art_dir / 'xgb_model.json'}")

    with open(art_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Features-> {art_dir / 'feature_columns.json'}")

    thresholds = {
        "T_up":   cfg["trading"]["T_up"],
        "T_down": cfg["trading"]["T_down"],
        "d_max_atr": cfg["trading"]["d_max_atr"],
        "classes": {0: "DOWN_BREAK_SOON", 1: "NO_BREAK", 2: "UP_BREAK_SOON"},
    }
    with open(art_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  Thresholds-> {art_dir / 'thresholds.json'}")

    with open(art_dir / "label_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Meta    -> {art_dir / 'label_meta.json'}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()

    # ── 1. Load data (all coins) ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading data for all coins")
    print("=" * 60)
    coin_data = data_mod.load_all_coins(cfg)
    print(f"\n  Coins loaded: {list(coin_data.keys())}")

    # ── 2. Compute features + labels per coin, then concat ────────────────────
    print("\nSTEP 2 & 3: Computing features + labels per coin")
    feat_cols = feat_mod.get_feature_columns(cfg)

    all_pieces:     list[pd.DataFrame] = []
    all_label_dfs:  list[pd.DataFrame] = []
    for coin, df_raw in coin_data.items():
        print(f"\n  [{coin}] features ...")
        df_feat = feat_mod.compute_features(df_raw, cfg)
        print(f"  [{coin}] labels ...")
        df_lab  = label_mod.compute_labels(df_raw, cfg)

        # Keep only known feature columns that exist for this coin
        fc = [c for c in feat_cols if c in df_feat.columns]
        piece = df_feat[fc].join(df_lab[["label"]], how="inner").dropna()

        # ── Near-range training filter ────────────────────────────────────────
        # Only train on bars where price is near a range boundary — matching
        # the actual trading entry filter (d_max_atr=1.5). This focuses the
        # model on near-breakout scenarios instead of general market prediction.
        d_train  = cfg["trading"].get("d_max_atr", 1.5) * 3.0  # 3x entry filter
        L_range  = cfg["labels"].get("L_range", 20)
        rh_c_tr  = f"dist_rh_{L_range}"
        rl_c_tr  = f"dist_rl_{L_range}"
        if rh_c_tr in piece.columns and rl_c_tr in piece.columns:
            near = (piece[rh_c_tr] <= d_train) | (piece[rl_c_tr] <= d_train)
            piece = piece[near]

        print(f"  [{coin}] usable rows: {len(piece):,}")
        all_pieces.append(piece)
        all_label_dfs.append(df_lab.loc[piece.index])  # aligned fakeout cols

    df_all = pd.concat(all_pieces).sort_index()
    df_all_labels = pd.concat(all_label_dfs).sort_index()

    # Recalculate feat_cols to only include columns present across all coins
    feat_cols = [c for c in feat_cols if c in df_all.columns]
    df_all = df_all[feat_cols + ["label"]].dropna()
    df_all_labels = df_all_labels.loc[df_all.index]

    label_mod.label_stats(df_all_labels)
    print(f"\n  Total usable rows (all coins): {len(df_all):,}")

    X = df_all[feat_cols]
    y = df_all["label"].astype(int).values

    # ── 5. Temporal train / test split ────────────────────────────────────────
    test_size  = cfg["training"]["test_size"]
    split_idx  = int(len(X) * (1 - test_size))
    X_train    = X.iloc[:split_idx]
    y_train    = y[:split_idx]
    X_test     = X.iloc[split_idx:]
    y_test     = y[split_idx:]
    print(f"\n  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Train period: {X_train.index[0]} -> {X_train.index[-1]}")
    print(f"  Test  period: {X_test.index[0]}  -> {X_test.index[-1]}")

    # ── 6. Walk-forward CV ────────────────────────────────────────────────────
    print("\nSTEP 4: Walk-forward cross-validation")
    best_n, fold_metrics = walk_forward_cv(X_train, y_train, cfg)

    # ── 7. Train final model ──────────────────────────────────────────────────
    print(f"\nSTEP 5: Training final model  (n_estimators={best_n})")
    model = train_final(X_train, y_train, X_test, y_test, best_n, cfg)

    # ── 8. Evaluate on test ───────────────────────────────────────────────────
    print("\nSTEP 6: Evaluating on hold-out test set")
    test_results = evaluate_on_test(model, X_test, y_test, cfg)

    # ── 9. Save artifacts ─────────────────────────────────────────────────────
    print("\nSTEP 7: Saving artifacts")
    meta = {
        "train_rows": int(len(X_train)),
        "test_rows":  int(len(X_test)),
        "train_start": str(X_train.index[0]),
        "train_end":   str(X_train.index[-1]),
        "test_start":  str(X_test.index[0]),
        "test_end":    str(X_test.index[-1]),
        "n_estimators": best_n,
        "n_features":   len(feat_cols),
        "test_logloss": test_results["logloss"],
        "fold_metrics": fold_metrics,
        "label_config": cfg["labels"],
    }
    save_artifacts(model, feat_cols, cfg, meta)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
