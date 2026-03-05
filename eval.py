"""
eval.py – Model evaluation: classification metrics + trading-oriented analysis.

Usage
-----
    python eval.py                    # uses models/ artifacts + config.yaml
    python eval.py --year 2025        # evaluate only on a specific year's data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    classification_report, confusion_matrix,
    log_loss, roc_auc_score, precision_recall_curve,
)

import data as data_mod
import features as feat_mod
import labels as label_mod


CLASS_NAMES = ["DOWN_BREAK", "NO_BREAK", "UP_BREAK"]


# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts(art_dir: str = "models") -> tuple:
    p = Path(art_dir)
    model = xgb.Booster()
    model.load_model(str(p / "xgb_model.json"))
    with open(p / "feature_columns.json") as f:
        feat_cols = json.load(f)
    with open(p / "thresholds.json") as f:
        thresholds = json.load(f)
    return model, feat_cols, thresholds


# ─────────────────────────────────────────────────────────────────────────────
def predict(model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    dm = xgb.DMatrix(X.values, feature_names=list(X.columns))
    probs = model.predict(dm).reshape(-1, 3)
    return probs


# ─────────────────────────────────────────────────────────────────────────────
def signal_quality(probs: np.ndarray, y_true: np.ndarray,
                   thresholds: dict) -> pd.DataFrame:
    """
    For a range of probability thresholds, compute:
      - Signal count (how many trades triggered)
      - Precision (fraction of triggered signals that were correct)
    Returns a DataFrame for plotting.
    """
    rows = []
    for t in np.arange(0.40, 0.90, 0.02):
        for cls_idx, cls_name in [(0, "DOWN"), (2, "UP")]:
            mask = probs[:, cls_idx] >= t
            if mask.sum() == 0:
                continue
            precision = (y_true[mask] == cls_idx).mean()
            rows.append({
                "threshold": round(t, 2),
                "class":     cls_name,
                "signals":   int(mask.sum()),
                "precision": round(precision, 3),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion(y_true, preds, out_path: str) -> None:
    cm = confusion_matrix(y_true, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i,j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_threshold_curve(sq: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cls in zip(axes, ["UP", "DOWN"]):
        sub = sq[sq["class"] == cls]
        ax.plot(sub["threshold"], sub["precision"], "o-", label="Precision")
        ax2 = ax.twinx()
        ax2.plot(sub["threshold"], sub["signals"],  "s--", color="orange",
                 label="# Signals")
        ax.set_xlabel("Probability Threshold"); ax.set_ylabel("Precision")
        ax2.set_ylabel("Signal Count")
        ax.set_title(f"{cls}_BREAK_SOON Threshold Analysis")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_feature_importance(model: xgb.Booster, out_path: str,
                            top_n: int = 30) -> None:
    imp = model.get_score(importance_type="gain")
    imp_s = pd.Series(imp).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    imp_s.plot.barh(ax=ax)
    ax.set_xlabel("Gain Importance")
    ax.set_title(f"Top-{top_n} Feature Importances (Gain)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_prob_calibration(probs: np.ndarray, y_true: np.ndarray,
                          out_path: str) -> None:
    """Reliability diagram for each class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(0, 1, 11)
    for cls_idx, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
        p   = probs[:, cls_idx]
        act = (y_true == cls_idx).astype(float)
        bin_idx = np.digitize(p, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
        mean_p = [p[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                  for b in range(len(bins) - 1)]
        mean_a = [act[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                  for b in range(len(bins) - 1)]
        bx = (bins[:-1] + bins[1:]) / 2
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(bx, mean_p, "o-", label="Model prob")
        ax.plot(bx, mean_a, "s-", label="Actual rate")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Actual fraction")
        ax.set_title(f"Calibration: {cls_name}"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        help="Evaluate only on this year (default: use test split)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--artifacts", default="models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("Loading model artifacts …")
    model, feat_cols, thresholds = load_artifacts(args.artifacts)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.year:
        cfg_tmp = dict(cfg)
        cfg_tmp["data"] = dict(cfg["data"])
        cfg_tmp["data"]["years"] = [args.year]
        df_raw = data_mod.load_all(cfg_tmp)
    else:
        df_raw = data_mod.load_all(cfg)
        # Use test split
        split = int(len(df_raw) * (1 - cfg["training"]["test_size"]))
        df_raw = df_raw.iloc[split:]

    # ── Features + Labels ─────────────────────────────────────────────────────
    print("Computing features …")
    df_feat = feat_mod.compute_features(df_raw, cfg)
    df_lab  = label_mod.compute_labels(df_raw, cfg)

    avail_feat_cols = [c for c in feat_cols if c in df_feat.columns]
    df_all  = df_feat[avail_feat_cols].join(df_lab[["label"]], how="inner").dropna()
    X       = df_all[avail_feat_cols]
    y_true  = df_all["label"].astype(int).values

    # ── Predict ───────────────────────────────────────────────────────────────
    print("Predicting …")
    probs  = predict(model, X)
    preds  = probs.argmax(axis=1)

    # ── Classification metrics ────────────────────────────────────────────────
    print("\n─── Classification Report ───")
    print(classification_report(y_true, preds, target_names=CLASS_NAMES,
                                zero_division=0))
    print(f"Log-loss:  {log_loss(y_true, probs):.4f}")

    # ── Signal quality ────────────────────────────────────────────────────────
    sq = signal_quality(probs, y_true, thresholds)
    print("\n─── Signal Quality (subset) ───")
    print(sq[sq["threshold"].isin([0.50, 0.55, 0.60, 0.65, 0.70])].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    art_dir = Path(args.artifacts)
    plot_confusion(y_true, preds, str(art_dir / "confusion_matrix.png"))
    plot_threshold_curve(sq,    str(art_dir / "threshold_curve.png"))
    plot_feature_importance(model, str(art_dir / "feature_importance.png"))
    plot_prob_calibration(probs, y_true, str(art_dir / "calibration.png"))

    # ── Save signal-quality table ─────────────────────────────────────────────
    sq.to_csv(art_dir / "signal_quality.csv", index=False)
    print(f"\n  Signal quality → {art_dir / 'signal_quality.csv'}")


if __name__ == "__main__":
    main()
