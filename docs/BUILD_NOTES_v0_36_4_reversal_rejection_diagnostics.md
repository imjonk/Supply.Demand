# v0.36.4 / v0.37.2 Reversal-Rejection Diagnostics

Purpose: audit whether Demand Reversal / Hold — Calls and Supply Rejection — Puts are genuinely weak, or whether the replay/backtest is entering or managing them differently than a live-style reversal trade would be handled.

## What changed

### replay_backtest.py
- Preserves the existing watchlist-first and baseline replay behavior.
- Keeps continuation logic permissive.
- Adds live-style reversal/rejection diagnostics for:
  - zone edge tap
  - zone tap depth
  - boundary reclaim/rejection
  - 9EMA/VWAP relationship
  - entry close location relative to zone
  - reaction candle body, range, and close strength
  - bad entry-side wick vs favorable rejection wick
  - VPA buckets, absorption, and follow-through volume
  - 1-candle and 2-candle confirmation
  - momentum/structure confirmation
  - backtest realism diagnosis
- Adds alternate reversal-only management simulation:
  - `ema_protect_05r_exit_r`
  - `ema_protect_05r_exit_reason`
  - related MFE/MAE/exit fields
- Fixes extracted-project replay by falling back from Windows absolute snapshot paths to local `reports/backtest/snapshots/` paths.
- Caches 5M symbol data to reduce repeated CSV reads during replay.

### analyze_backtest.py
- Adds `Reversal/Rejection Diagnostics` dashboard section.
- Writes new CSVs:
  - `reports/backtest/analytics/reversal_rejection_breakdowns.csv`
  - `reports/backtest/analytics/reversal_rejection_rule_variants.csv`
  - `reports/backtest/analytics/reversal_rejection_best_filters.csv`
- Compares baseline replay behavior against live-style diagnostic variants:
  - exclude B/C grades
  - Final only
  - A/A+ Final only
  - require zone edge tap
  - require strong instant reaction candle
  - require good wick quality
  - require VPA confirmation
  - require 2-candle follow-through
  - require 9EMA or VWAP break
  - require structure confirmation
  - require boundary reclaim/rejection
  - require confirmation score >= 4
  - require confirmation score >= 5
  - manage reversals with 9EMA protection after +0.5R

## Important interpretation

This build does not automatically change the live watchlist rules. It is an audit layer. The dashboard is meant to answer whether reversal/rejection losses are caused by true setup weakness, premature backtest entries, missing live-style confirmation, or management that is too loose for reversal trades.

## Run sequence

```bash
python replay_backtest.py --preset balanced
python analyze_backtest.py
```

For a faster smoke test:

```bash
python replay_backtest.py --start 2026-02-03 --end 2026-02-05 --preset balanced
python analyze_backtest.py
```

Open:

```text
reports/backtest/strategy_dashboard.html
```

## Notes on included outputs

The included `reports/backtest/trades.csv` has been enriched with reversal/rejection diagnostics using the existing full replay output that came with the uploaded project. The replay script itself can regenerate these fields from scratch.
