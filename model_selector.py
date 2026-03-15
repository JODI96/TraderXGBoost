"""
Model Selector for TraderXGBoost
================================
Standalone script that sets up a model + config, then runs backtest.py.
Does NOT modify backtest.py itself.

Usage:
  python model_selector.py --model jodi_original
  python model_selector.py --model v22_trial32
  python model_selector.py --model list
  python model_selector.py --model all     (tests all models, shows comparison)
"""
import os, sys, json, shutil, subprocess, argparse
from pathlib import Path

import yaml

# All models use the same 133 standard feature columns.
# Only the model file and trading config differ.
MODELS_REGISTRY = {
    "jodi_original": {
        "desc": "Jodis Original (PF 5.82, WR 75%, 28 Trades)",
        "model": "models/xgb_model_JODI_ORIGINAL_DO_NOT_OVERWRITE.json",
        "config": {"T_up": 0.67, "T_down": 0.67, "sl_pct": 0.0015, "tp_pct": 0.0045, "time_stop": 30},
    },
    "jodi_sl2tp6": {
        "desc": "Jodis Original + sl2/tp6 (PF 2.58, WR 58.3%, 48 Trades)",
        "model": "models/xgb_model_JODI_ORIGINAL_DO_NOT_OVERWRITE.json",
        "config": {"T_up": 0.67, "T_down": 0.67, "sl_pct": 0.002, "tp_pct": 0.006, "time_stop": 30},
    },
    "v2_trial41": {
        "desc": "Fabio v2 Trial 41 (PF 4.64, WR 62.5%, 24 Trades, DD -10.1%)",
        "model": "models/fabio_models/v2_trial41_PF4.64.json",
        "config": {"T_up": 0.67, "T_down": 0.67, "sl_pct": 0.0015, "tp_pct": 0.0055, "time_stop": 30},
    },
    "v21_trial24": {
        "desc": "Fabio v2.1 Trial 24 (PF 2.69, WR 62.1%, 29 Trades, DD -8.4%)",
        "model": "models/fabio_models/v2.1_trial24_PF2.69.json",
        "config": {"T_up": 0.64, "T_down": 0.60, "sl_pct": 0.002, "tp_pct": 0.0045, "time_stop": 30},
    },
    "v21_trial60": {
        "desc": "Fabio v2.1 Trial 60 (PF 7.05, WR 83.3%, 6 Trades, DD -3.9%)",
        "model": "models/fabio_models/v2.1_trial60_PF7.05.json",
        "config": {"T_up": 0.64, "T_down": 0.64, "sl_pct": 0.0015, "tp_pct": 0.0045, "time_stop": 30},
    },
    "v21_trial51": {
        "desc": "Fabio v2.1 Trial 51 (PF 5.42, WR 80.0%, 10 Trades, DD -3.7%)",
        "model": "models/fabio_models/v2.1_trial51_PF5.42.json",
        "config": {"T_up": 0.60, "T_down": 0.58, "sl_pct": 0.0015, "tp_pct": 0.0045, "time_stop": 25},
    },
    "v21_trial75": {
        "desc": "Fabio v2.1 Trial 75 (PF 3.63, WR 70.0%, 20 Trades, DD -7.5%)",
        "model": "models/fabio_models/v2.1_trial75_PF3.63.json",
        "config": {"T_up": 0.60, "T_down": 0.58, "sl_pct": 0.002, "tp_pct": 0.0045, "time_stop": 25},
    },
    "v21_trial33": {
        "desc": "Fabio v2.1 Trial 33 (PF 3.42, WR 63.2%, 19 Trades, DD -5.9%)",
        "model": "models/fabio_models/v2.1_trial33_PF3.42.json",
        "config": {"T_up": 0.60, "T_down": 0.58, "sl_pct": 0.0015, "tp_pct": 0.0045, "time_stop": 25},
    },
    "v3_trial36": {
        "desc": "Fabio v3 Trial 36 (PF 2.87, WR 54.5%, 33 Trades, DD -12.8%)",
        "model": "models/fabio_models/v3_trial36_PF2.87.json",
        "config": {"T_up": 0.67, "T_down": 0.67, "sl_pct": 0.0015, "tp_pct": 0.0055, "time_stop": 30},
    },
    "v3_trial53": {
        "desc": "Fabio v3 Trial 53 (PF 3.27, WR 60%, 15 Trades, DD -8.5%)",
        "model": "models/fabio_models/v3_trial53_PF3.27.json",
        "config": {"T_up": 0.67, "T_down": 0.67, "sl_pct": 0.0015, "tp_pct": 0.0055, "time_stop": 30},
    },
    "v22_trial32": {
        "desc": "Fabio v2.2 Trial 32 (PF 2.71, WR 61.8%, 34 Trades, DD -6.5%)",
        "model": "models/fabio_models/v22_trial32_PF2.71.json",
        "config": {"T_up": 0.55, "T_down": 0.64, "sl_pct": 0.0015, "tp_pct": 0.0045, "time_stop": 20},
    },
}


def setup_model(name):
    """Copy model file and patch config for the selected model."""
    m = MODELS_REGISTRY[name]
    print(f"\n{'='*60}")
    print(f"  Setting up: {name}")
    print(f"  {m['desc']}")
    print(f"{'='*60}")

    if not os.path.exists(m["model"]):
        print(f"  ERROR: {m['model']} not found!")
        return False

    # Copy model
    shutil.copy2(m["model"], "models/xgb_model.json")
    print(f"  Copied {Path(m['model']).name} -> models/xgb_model.json")

    # Patch config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    for key, val in m["config"].items():
        if key in ("sl_pct", "tp_pct"):
            cfg["labels"][key] = val
            cfg["trading"][key] = val
        elif key in ("T_up", "T_down", "time_stop"):
            cfg["trading"][key] = val

    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    tc = m["config"]
    print(f"  Config: T_up={tc.get('T_up')}, T_down={tc.get('T_down')}, "
          f"sl={tc.get('sl_pct')}, tp={tc.get('tp_pct')}, ts={tc.get('time_stop')}")
    return True


def run_backtest():
    """Run backtest.py and return the report dict."""
    result = subprocess.run(
        [sys.executable, "-u", "backtest.py"],
        capture_output=True, text=True, timeout=600,
    )
    # Print last part of output (the report)
    for line in result.stdout.split("\n"):
        if line.strip():
            print(f"  {line.strip()}")
    if result.returncode != 0:
        print(f"  BACKTEST ERROR: {result.stderr[-500:]}")
        return None
    try:
        with open("models/backtest_report.json") as f:
            return json.load(f)
    except Exception:
        return None


def list_models():
    """Print all available models."""
    print(f"\n{'='*70}")
    print(f"  Available Models for TraderXGBoost")
    print(f"{'='*70}")
    for name, m in MODELS_REGISTRY.items():
        exists = "OK" if os.path.exists(m["model"]) else "MISSING"
        print(f"  [{exists:>7}] {name:<18} {m['desc']}")
    print(f"\nUsage:")
    print(f"  python model_selector.py --model <name>")
    print(f"  python model_selector.py --model all   (compare all)")


def run_all():
    """Test all models and show comparison table."""
    # Save original config and model
    shutil.copy2("config.yaml", "config.yaml.bak")
    if os.path.exists("models/xgb_model.json"):
        shutil.copy2("models/xgb_model.json", "models/xgb_model_selector_backup.json")

    results = {}
    for name in MODELS_REGISTRY:
        if not os.path.exists(MODELS_REGISTRY[name]["model"]):
            print(f"\n  SKIP {name} (model file not found)")
            continue
        if setup_model(name):
            r = run_backtest()
            if r and "error" not in r:
                results[name] = r

    # Restore originals
    shutil.copy2("config.yaml.bak", "config.yaml")
    os.remove("config.yaml.bak")
    if os.path.exists("models/xgb_model_selector_backup.json"):
        shutil.copy2("models/xgb_model_selector_backup.json", "models/xgb_model.json")
        os.remove("models/xgb_model_selector_backup.json")

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"  MODEL COMPARISON (sorted by Profit Factor)")
    print(f"{'='*80}")
    print(f"  {'Model':<18} {'Return':>8} {'Trades':>7} {'WR':>6} {'PF':>6} "
          f"{'MaxDD':>7} {'Sharpe':>7}")
    print(f"  {'-'*63}")

    sorted_results = sorted(results.items(),
                            key=lambda x: x[1].get("profit_factor", 0),
                            reverse=True)
    for name, r in sorted_results:
        print(f"  {name:<18} {r.get('total_return_pct',0):>+7.1f}% "
              f"{r.get('n_trades',0):>7} {r.get('win_rate',0):>5.1f}% "
              f"{r.get('profit_factor',0):>5.2f} {r.get('max_drawdown_pct',0):>6.1f}% "
              f"{r.get('sharpe_ratio',0):>6.2f}")

    print(f"\n  {len(results)} models tested. Original model + config restored.")


def main():
    parser = argparse.ArgumentParser(description="Model Selector for TraderXGBoost")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. jodi_original, v22_trial32, list, all)")
    args = parser.parse_args()

    if args.model == "list":
        list_models()
    elif args.model == "all":
        run_all()
    else:
        if args.model not in MODELS_REGISTRY:
            print(f"  ERROR: Model '{args.model}' not found.")
            print(f"  Available: {', '.join(MODELS_REGISTRY.keys())}")
            print(f"  Use --model list for details.")
            return

        # Save original config
        shutil.copy2("config.yaml", "config.yaml.bak")
        if os.path.exists("models/xgb_model.json"):
            shutil.copy2("models/xgb_model.json", "models/xgb_model_selector_backup.json")

        if setup_model(args.model):
            run_backtest()

        # Restore originals
        shutil.copy2("config.yaml.bak", "config.yaml")
        os.remove("config.yaml.bak")
        if os.path.exists("models/xgb_model_selector_backup.json"):
            shutil.copy2("models/xgb_model_selector_backup.json", "models/xgb_model.json")
            os.remove("models/xgb_model_selector_backup.json")
        print(f"\n  Original model + config restored.")


if __name__ == "__main__":
    main()
