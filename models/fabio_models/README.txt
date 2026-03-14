Fabio's Pipeline Models (March 2026)
=====================================

Quick Test (from project root):
  python backtest.py --model trial60      # PF 7.05, WR 83.3%,  6 Trades
  python backtest.py --model trial51      # PF 5.42, WR 80.0%, 10 Trades
  python backtest.py --model trial75      # PF 3.63, WR 70.0%, 20 Trades
  python backtest.py --model trial33      # PF 3.42, WR 63.2%, 19 Trades
  python backtest.py --model trial24      # PF 2.69, WR 62.1%, 29 Trades
  python backtest.py --model jodi         # Jodi's current model (baseline)
  python backtest.py --model list         # Show all profiles

Pipeline v2.1 Results (100 Optuna trials, Jodi's train.py + backtest.py):
  Benchmark: Jodi Master (PF 2.17, 48 Trades, +61.2%, DD -13.4%)

  Rank  Trial   Return  Trades   WR     PF    MaxDD   Sharpe
  ----  ------  ------  ------  -----  -----  ------  ------
   1    #60     +11.5%     6    83.3%  7.05   -3.9%    2.39
   2    #51     +17.2%    10    80.0%  5.42   -3.7%    2.48
   3    #75     +33.2%    20    70.0%  3.63   -7.5%    2.94
   4    #33     +31.6%    19    63.2%  3.42   -5.9%    3.36
   5    #24     +42.4%    29    62.1%  2.69   -8.4%    2.85

Legacy Models (Pipeline v2/v3, old backtest code):
  v2_trial41: Return +114.7%, 24 Trades, WR 62.5%, PF 4.64, DD -10.1%
  v3_trial36: Return +123.5%, 33 Trades, WR 54.5%, PF 2.87, DD -12.8%
  v3_trial53: Return +49.7%, 15 Trades, WR 60.0%, PF 3.27, DD -8.5%

Notes:
  - All v2.1 models trained with Jodi's master code (March 2026)
  - All use 133 features (standard feature_columns.json)
  - Backtest on test split (20%) with 10% position size
  - All models beat Jodi's PF but with fewer trades
