# v0.36.5 Exit Path Audit

Purpose: audit whether the replay engine is scoring trades in the same order and with the same protection logic that would happen in a live trade.

This is an audit build. It does not change watchlist generation and does not promote new live filters.

## Added files / code paths

- `audit_exit_paths.py`
  - Enriches an existing `reports/backtest/trades.csv` with exit-path audit fields without rerunning entries.
  - Uses subprocess chunking for large trade files to keep memory stable.
- `replay_backtest.py`
  - Adds exit-path audit fields directly to newly replayed trades.
  - Preserves baseline conservative intrabar behavior: stop is processed before target when both occur in the same 5M candle.
  - Adds reversal/rejection management simulations.
- `analyze_backtest.py`
  - Adds an `Exit Path Audit` dashboard section.
  - Writes new CSVs under `reports/backtest/analytics/`.

## New analytics CSVs

- `exit_path_summary.csv`
- `exit_path_first_event.csv`
- `exit_path_flags.csv`
- `exit_path_management_variants.csv`

## New trade fields

Key audit fields include:

- `first_event_after_entry`
- `first_1r_time`, `first_2r_time`, `first_3r_time`
- `first_stop_time`, `first_target_3r_time`, `first_ema_protection_time`
- `mfe_r_until_exit`, `mae_r_until_exit`
- `mfe_r_full_day`, `mae_r_full_day`
- `target_and_stop_same_candle`
- `same_candle_ambiguity`
- `stop_after_reached_1r`, `stop_after_reached_2r`, `stop_after_reached_3r`
- `target_available_but_not_taken`
- `mfe_after_exit_detected`, `mae_after_exit_detected`
- `unrealistic_r_outlier`

## Reversal/rejection management simulations

Added diagnostic-only simulated R fields:

- `target_priority_3r_exit_r`
- `breakeven_after_1r_exit_r`
- `ema_protect_05r_exit_r`
- `boundary_loss_1_close_exit_r`
- `boundary_loss_2_closes_exit_r`
- `optimistic_intrabar_result`
- `neutral_intrabar_result`

## Current full-sample read from the included regenerated report

The included report keeps the same baseline summary as the prior run:

- Trades: 3,322
- Average R: 0.015
- Total R: 49.148
- Reversal/rejection baseline average R: -0.117

The exit-path audit found that reversal/rejection trades include:

- 235 stop-outs after reaching +1R
- 127 stop-outs after reaching +2R
- 77 stop-outs after reaching +3R
- 77 target/stop same-candle cases

The management simulation table shows that intrabar target-priority would materially change the reversal/rejection read, moving that subset from -0.117R average to about +0.026R average. This does not mean the optimistic assumption is correct; it means same-candle ordering is a major fidelity issue that should be audited before deciding reversals are genuinely weak.

## Recommended run order

After a fresh replay:

```bash
python replay_backtest.py --preset balanced
python analyze_backtest.py
```

To enrich an existing `trades.csv` without rerunning entries:

```bash
python audit_exit_paths.py
python analyze_backtest.py
```

Then open:

```text
reports/backtest/strategy_dashboard.html
```
