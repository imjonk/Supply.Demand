# v0.37.4 target/time-filter diagnostics

Adds two requested backtest/reporting features:

1. Entry start-time cutoff
   - New replay argument: `--max-entry-time HH:MM`
   - Example: `--max-entry-time 13:00`
   - Candidates whose detected entry time is after the cutoff are rejected with `entry_after_max_entry_time`.
   - This lets the replay simulate taking no new trades after 1 PM New York time.

2. Target model comparison
   - Adds intraday ATR14 calculation from 5-minute RTH bars.
   - Adds 1x ATR target diagnostics from entry.
   - Adds target comparison columns in `reports/backtest/trades.csv`:
     - `reached_atr_1x`
     - `entry_atr_14`
     - `atr_1x_r`
     - `target_1r_result`
     - `target_2r_result`
     - `target_3r_result`
     - `target_atr_1x_result`
   - Adds `analytics/target_model_comparison.csv` and dashboard table.

3. Dashboard filters
   - Adds `entry_time_bucket` to the embedded trade data.
   - Adds an interactive filter for entry time bucket:
     - `09:30-10:29`
     - `10:30-11:59`
     - `12:00-13:00`
     - `After 13:00`

Suggested run:

```bash
python build_backtest_snapshots.py --start 2026-01-01 --end 2026-06-30
python replay_backtest.py --start 2026-01-01 --end 2026-06-30 --preset balanced --rr 3 --max-entry-time 13:00
python analyze_backtest.py
```

For comparison against the old behavior, run once without `--max-entry-time` and once with `--max-entry-time 13:00`, then compare `summary.csv`, `analytics/target_model_comparison.csv`, and `strategy_dashboard.html`.
