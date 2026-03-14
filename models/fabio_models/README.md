# Fabio's Trial 24 Model

## Quick Start
```bash
git fetch origin
git checkout Nexapon
cp models/fabio_models/v2.1_trial24_PF2.69.json models/xgb_model.json
cp models/fabio_models/feature_columns_trial24.json models/feature_columns.json
python backtest.py
```

## Config (in config.yaml setzen)
```yaml
trading:
  T_up: 0.64
  T_down: 0.60
  sl_pct: 0.002
  tp_pct: 0.0045
```

## Backtest Ergebnisse
- Return: +42.4% (10x Leverage) / +3.72% (1x)
- Trades: 29
- Win Rate: 62.1%
- Profit Factor: 2.69
- Max Drawdown: -8.4%
- Sharpe: 2.85

## Parameter (XGBoost)
- depth=5, lr=0.0095, est=3500, mcw=13
- sub=0.58, col=0.80, gamma=0.97
- alpha=0.58, lambda=4.69, bw=7.0
- sl=0.002, tp=0.0045, H=20

## Wichtig
- Braucht 133 Features (experimental Branch features.py)
- NICHT mit master features.py verwenden (nur 113 Features)
- feature_columns_trial24.json muss als models/feature_columns.json kopiert werden
