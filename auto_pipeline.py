"""
auto_pipeline.py v2.1 – Optuna wrapper around Jodi's train.py + backtest.py.

Benchmark: Jodi Master March 2026 (48 trades, PF 2.17, +61.2%, 10% pos size)
Goal: Beat PF and/or trade count while keeping DD controlled.

Each trial:
  1. Patches config.yaml with trial hyperparameters
  2. Runs Jodi's train.py (walk-forward CV + final model)
  3. Runs Jodi's backtest.py (event-driven backtest with all filters)
  4. Reads models/backtest_report.json
  5. Saves model as pipeline_results/trial_{n}_model.json
  6. Restores original config.yaml

NEVER touches models/xgb_model_JODI_MASTER_NEW.json
"""
import json, shutil, subprocess, sys, time, copy
import numpy as np
from pathlib import Path
from datetime import datetime

import yaml
import optuna

# ============================================================
# PATHS
# ============================================================
PROJECT   = Path(".")
CONFIG    = PROJECT / "config.yaml"
RESULTS   = PROJECT / "pipeline_results"
RESULTS.mkdir(exist_ok=True)

MODEL_OUT     = PROJECT / "models" / "xgb_model.json"
REPORT_OUT    = PROJECT / "models" / "backtest_report.json"
PYTHON        = str(PROJECT / "trader_venv" / "Scripts" / "python.exe")

BENCHMARK = {
    "win_rate": 54.2,
    "profit_factor": 2.17,
    "n_trades": 48,
    "max_drawdown_pct": -13.38,
    "total_return_pct": 61.22,
    "sharpe_ratio": 3.302,
}

N_TRIALS = 100

# ============================================================
# HELPERS
# ============================================================
def load_config():
    with open(CONFIG) as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open(CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

def run_cmd(args, timeout=1200):
    """Run a command, return (returncode, stdout, stderr). Catches timeout."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT)
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"

def read_report():
    """Read backtest_report.json."""
    if not REPORT_OUT.exists():
        return None
    with open(REPORT_OUT) as f:
        return json.load(f)


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def objective(trial):
    t0 = time.time()

    # ── Sample hyperparameters ──
    # XGBoost (tight around Jodi's proven params)
    max_depth        = trial.suggest_int("max_depth", 4, 7)
    learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
    n_estimators     = trial.suggest_int("n_estimators", 1000, 5000, step=500)
    min_child_weight = trial.suggest_int("min_child_weight", 5, 50)
    subsample        = trial.suggest_float("subsample", 0.5, 0.8)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 0.8)
    gamma            = trial.suggest_float("gamma", 0.0, 3.0)
    reg_alpha        = trial.suggest_float("reg_alpha", 0.1, 10.0, log=True)
    reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 10.0)
    break_weight     = trial.suggest_float("break_weight", 5.0, 25.0)

    # Labels (tight around Jodi's proven values)
    sl_pct    = trial.suggest_categorical("sl_pct", [0.0010, 0.0015, 0.0020])
    tp_pct    = trial.suggest_categorical("tp_pct", [0.0035, 0.0045, 0.0055])
    H_horizon = trial.suggest_categorical("H_horizon", [15, 20, 25])

    # Trading thresholds
    T_up   = trial.suggest_categorical("T_up", [0.58, 0.60, 0.62, 0.64, 0.67, 0.70])
    T_down = trial.suggest_categorical("T_down", [0.58, 0.60, 0.62, 0.64, 0.67, 0.70])

    print(f"\n{'='*60}", flush=True)
    print(f"  Trial #{trial.number}", flush=True)
    print(f"  depth={max_depth} lr={learning_rate:.4f} est={n_estimators} "
          f"mcw={min_child_weight} sub={subsample:.2f} col={colsample_bytree:.2f}", flush=True)
    print(f"  gamma={gamma:.2f} alpha={reg_alpha:.4f} lambda={reg_lambda:.2f} "
          f"bw={break_weight:.1f}", flush=True)
    print(f"  sl={sl_pct:.4f} tp={tp_pct:.4f} H={H_horizon} "
          f"T_up={T_up:.2f} T_down={T_down:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Patch config ──
    cfg = load_config()
    original_cfg = copy.deepcopy(cfg)

    # XGBoost params
    xp = cfg["training"]["xgb_params"]
    xp["max_depth"]        = max_depth
    xp["learning_rate"]    = learning_rate
    xp["n_estimators"]     = n_estimators
    xp["min_child_weight"] = min_child_weight
    xp["subsample"]        = subsample
    xp["colsample_bytree"] = colsample_bytree
    xp["gamma"]            = gamma
    xp["reg_alpha"]        = reg_alpha
    xp["reg_lambda"]       = reg_lambda
    cfg["training"]["break_weight"] = break_weight

    # Label params
    cfg["labels"]["sl_pct"]    = sl_pct
    cfg["labels"]["tp_pct"]    = tp_pct
    cfg["labels"]["H_horizon"] = H_horizon

    # Trading params (must match labels)
    cfg["trading"]["sl_pct"]  = sl_pct
    cfg["trading"]["tp_pct"]  = tp_pct
    cfg["trading"]["T_up"]    = T_up
    cfg["trading"]["T_down"]  = T_down
    cfg["trading"]["time_stop"] = H_horizon + 10  # always H + 10

    save_config(cfg)

    try:
        # ── Run train.py ──
        print("  [TRAIN] Starting...", flush=True)
        rc, stdout, stderr = run_cmd([PYTHON, "-u", "train.py"], timeout=1800)
        if rc != 0:
            print(f"  [TRAIN] FAILED (rc={rc})", flush=True)
            print(stderr[-500:] if stderr else "no stderr", flush=True)
            return 0.0

        # Grab key info from output
        for line in stdout.split("\n"):
            if "best_n_estimators" in line or "Log-loss" in line:
                print(f"  {line.strip()}", flush=True)

        # ── Save trial model ──
        trial_model = RESULTS / f"trial_{trial.number:04d}_model.json"
        if MODEL_OUT.exists():
            shutil.copy2(MODEL_OUT, trial_model)

        # ── Run backtest.py ──
        print("  [BACKTEST] Starting...", flush=True)
        rc, stdout, stderr = run_cmd([PYTHON, "-u", "backtest.py"], timeout=120)
        if rc != 0:
            print(f"  [BACKTEST] FAILED (rc={rc})", flush=True)
            print(stderr[-500:] if stderr else "no stderr", flush=True)
            return 0.0

        # Print backtest output
        for line in stdout.split("\n"):
            if line.strip():
                print(f"  {line.strip()}", flush=True)

        # ── Read results ──
        report = read_report()
        if report is None:
            print("  [ERROR] No backtest_report.json found", flush=True)
            return 0.0

        ret    = report.get("total_return_pct", 0)
        nt     = report.get("n_trades", 0)
        wr     = report.get("win_rate", 0)
        pf     = report.get("profit_factor", 0)
        sharpe = report.get("sharpe_ratio", 0)
        max_dd = report.get("max_drawdown_pct", 0)
        final  = report.get("final_capital", 0)

        # Store in trial
        trial.set_user_attr("report", report)

        # ── Score: PF-focused, penalize losses, reward trades logarithmically ──
        # Goal: beat Jodi's PF=2.17 with >= 48 trades
        if nt < 5:
            score = 0.0
        elif pf < 1.0:
            score = pf * 0.5  # losing strategies capped at 0.5
        else:
            score = pf * (1 + np.log1p(nt) * 0.3) - abs(max_dd) * 0.02
            # Bonus for beating benchmark trade count
            if nt >= 48:
                score *= 1.1
            # Halve score if drawdown is catastrophic
            if max_dd < -40:
                score *= 0.5

        elapsed = time.time() - t0

        # ── Save trial report ──
        save_trial_report(trial.number, trial.params, report, elapsed)

        print(f"\n  RESULT: ret={ret:+.1f}%  trades={nt}  WR={wr:.1f}%  "
              f"PF={pf:.2f}  DD={max_dd:.1f}%  sharpe={sharpe:.2f}  "
              f"score={score:.2f}  ({elapsed:.0f}s)", flush=True)

        return score

    finally:
        # ── ALWAYS restore original config ──
        save_config(original_cfg)


# ============================================================
# REPORTING
# ============================================================
def save_trial_report(trial_num, params, report, elapsed):
    ret    = report.get("total_return_pct", 0)
    nt     = report.get("n_trades", 0)
    wr     = report.get("win_rate", 0)
    pf     = report.get("profit_factor", 0)
    sharpe = report.get("sharpe_ratio", 0)
    max_dd = report.get("max_drawdown_pct", 0)
    final  = report.get("final_capital", 0)
    bm     = BENCHMARK

    text = f"""
{'='*60}
Trial #{trial_num} | {datetime.now().strftime('%Y-%m-%d %H:%M')} | {elapsed:.0f}s
{'='*60}

PARAMETERS:
  max_depth:        {params.get('max_depth')}
  learning_rate:    {params.get('learning_rate', 0):.4f}
  n_estimators:     {params.get('n_estimators')}
  min_child_weight: {params.get('min_child_weight')}
  subsample:        {params.get('subsample', 0):.2f}
  colsample_bytree: {params.get('colsample_bytree', 0):.2f}
  gamma:            {params.get('gamma', 0):.2f}
  reg_alpha:        {params.get('reg_alpha', 0):.4f}
  reg_lambda:       {params.get('reg_lambda', 0):.2f}
  break_weight:     {params.get('break_weight', 0):.1f}
  sl_pct:           {params.get('sl_pct', 0):.4f}
  tp_pct:           {params.get('tp_pct', 0):.4f}
  H_horizon:        {params.get('H_horizon')}
  T_up:             {params.get('T_up', 0):.2f}
  T_down:           {params.get('T_down', 0):.2f}

BACKTEST RESULTS (10% position size):
  Return:         {ret:+.1f}%
  Trades:         {nt}
  Win Rate:       {wr:.1f}%
  Profit Factor:  {pf:.2f}
  Sharpe:         {sharpe:.2f}
  Max Drawdown:   {max_dd:.1f}%
  Final Capital:  ${final:.2f}

vs JODI BENCHMARK (master, 10% pos size):
  Return:  {ret:+.1f}% vs {bm['total_return_pct']:+.1f}%  {'BETTER' if ret > bm['total_return_pct'] else 'WORSE'}
  WR:      {wr:.1f}% vs {bm['win_rate']:.1f}%  {'BETTER' if wr > bm['win_rate'] else 'WORSE'}
  PF:      {pf:.2f} vs {bm['profit_factor']:.2f}  {'BETTER' if pf > bm['profit_factor'] else 'WORSE'}
  Trades:  {nt} vs {bm['n_trades']}  {'BETTER' if nt > bm['n_trades'] else 'WORSE'}
  DD:      {max_dd:.1f}% vs {bm['max_drawdown_pct']:.1f}%  {'BETTER' if max_dd > bm['max_drawdown_pct'] else 'WORSE'}
  Sharpe:  {sharpe:.2f} vs {bm['sharpe_ratio']:.2f}  {'BETTER' if sharpe > bm['sharpe_ratio'] else 'WORSE'}
"""
    filepath = RESULTS / f"trial_{trial_num:04d}.txt"
    with open(filepath, "w") as f:
        f.write(text)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  TraderXGBoost Auto-Pipeline v2.1")
    print("  Wrapper around Jodi's train.py + backtest.py")
    print(f"  {N_TRIALS} Optuna trials | GPU: RTX 3090")
    print(f"  Benchmark: PF={BENCHMARK['profit_factor']:.2f}, "
          f"{BENCHMARK['n_trades']} trades, "
          f"+{BENCHMARK['total_return_pct']:.1f}%")
    print("=" * 60)

    # Verify backup is safe
    jodi_backup = PROJECT / "models" / "xgb_model_JODI_MASTER_NEW.json"
    if jodi_backup.exists():
        print(f"\n  Jodi backup OK: {jodi_backup} ({jodi_backup.stat().st_size / 1e6:.1f} MB)")
    else:
        print("\n  WARNING: No Jodi backup found!")

    # Delete old Optuna DB to start fresh
    db_path = RESULTS / "optuna.db"
    if db_path.exists():
        db_path.unlink()
        print("  Deleted old optuna.db - starting fresh")

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="pipeline_v2.1",
        storage=f"sqlite:///{RESULTS}/optuna.db",
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )

    print(f"\n  Starting {N_TRIALS} trials...\n")
    study.optimize(objective, n_trials=N_TRIALS)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    best = study.best_trial
    report = best.user_attrs.get("report", {})
    print(f"\n  Best Trial: #{best.number}")
    print(f"  Score: {best.value:.2f}")
    print(f"  Return: {report.get('total_return_pct', 0):+.1f}%")
    print(f"  Trades: {report.get('n_trades', 0)}")
    print(f"  Win Rate: {report.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {report.get('profit_factor', 0):.2f}")
    print(f"  Sharpe: {report.get('sharpe_ratio', 0):.2f}")
    print(f"  Max DD: {report.get('max_drawdown_pct', 0):.1f}%")
    print(f"\n  Best model: {RESULTS}/trial_{best.number:04d}_model.json")
    print(f"  All results: {RESULTS}/")

    # Restore Jodi's model
    if jodi_backup.exists():
        shutil.copy2(jodi_backup, MODEL_OUT)
        print(f"\n  Restored Jodi's model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
