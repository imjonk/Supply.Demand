# v0.37.2_preopen_context_backtest

## Purpose

Simulate the new movement-context watchlist as it would appear before the market opens, using:

- prior regular-session zones only
- an 8:00 AM America/New_York premarket/current price context
- gap direction/size relative to the prior RTH close
- gap-vs-zone context
- movement, EMA/VWAP, volume/VPA, and historical zone-reaction context

This creates a closer historical comparison to how the live watchlist is intended to be used before the RTH session begins.

## Core rule preserved

Premarket and after-hours data remain informational context only. Extended-hours prices do **not** create, merge, break, refresh, or invalidate zones.

RTH-only logic remains in place for:

- zone creation
- zone merging
- active/broken zone state
- zone freshness/test count
- target ladders
- historical zone ledger state

## Main code changes

### `build_backtest_snapshots.py`

Adds preopen snapshot exports alongside the existing prior-close snapshots.

New default behavior:

```bash
python build_backtest_snapshots.py
```

Now writes normal snapshot files to:

```text
reports/backtest/snapshots/
```

and preopen context files to:

```text
reports/backtest/preopen_snapshots/
```

New files per test date:

```text
YYYY-MM-DD_preopen_context.csv
YYYY-MM-DD_movement_context_watchlist.csv
YYYY-MM-DD_preopen_final_watchlist.csv
```

The normal prior-close files are still produced:

```text
YYYY-MM-DD_active_zones.csv
YYYY-MM-DD_merged_zones.csv
YYYY-MM-DD_scenarios.csv
YYYY-MM-DD_final_watchlist.csv
```

The manifest now includes:

```text
preopen_context_time
preopen_scenarios
preopen_final_setups
preopen_context_file
preopen_scenario_file
preopen_final_file
```

New arguments:

```bash
python build_backtest_snapshots.py --preopen-time 08:00
python build_backtest_snapshots.py --no-preopen-context
```

### `watchlist.py`

`build_watchlist_from_zone_snapshot()` can now accept a `symbol_movement_context` dictionary through its `meta` argument. This allows historical/preopen snapshots to be enriched with the same movement-context fields used by the live watchlist.

### `replay_backtest.py`

Adds:

```bash
--snapshot-mode close
--snapshot-mode preopen
```

Default remains `close` for backward compatibility.

To replay the 8:00 AM movement-context watchlist:

```bash
python replay_backtest.py --preset balanced --snapshot-mode preopen
```

With final-only rows:

```bash
python replay_backtest.py --preset balanced --snapshot-mode preopen --use-final-only
```

The trade and candidate CSVs now carry through preopen/movement fields such as:

```text
snapshot_mode
snapshot_context_time
snapshot_context_type
current_price_as_of
current_price_session
gap_direction
gap_pct
gap_zone_context
zone_thesis
zone_movement_state
movement_watchlist_bucket
observation_score
observation_reason
watch_for
volume_state
vpa_state
historical_zone_tendency
```

### `analyze_backtest.py`

Adds preopen/movement-context analytics tables when those columns are present:

```text
performance_by_snapshot_mode.csv
performance_by_current_price_session.csv
performance_by_gap_direction.csv
performance_by_gap_zone_context.csv
performance_by_zone_thesis.csv
performance_by_zone_movement_state.csv
performance_by_movement_watchlist_bucket.csv
performance_by_volume_state.csv
performance_by_vpa_state.csv
```

The dashboard includes a new section:

```text
Preopen / Movement Context Analysis
```

## Recommended workflow

Build historical snapshots with preopen context:

```bash
python build_backtest_snapshots.py
```

Replay the preopen movement-context watchlist:

```bash
python replay_backtest.py --preset balanced --snapshot-mode preopen
```

Run exit-path audit and dashboard analysis:

```bash
python audit_exit_paths.py
python analyze_backtest.py
```

Open:

```text
reports/backtest/strategy_dashboard.html
```

## Smoke test performed

Validated with a small sample:

```bash
python build_backtest_snapshots.py --start 2026-01-14 --end 2026-01-17 --max-days 3 --symbols AAPL,AMD,AMZN
python replay_backtest.py --start 2026-01-15 --end 2026-01-16 --symbols AAPL,AMD,AMZN --preset balanced --snapshot-mode preopen
python analyze_backtest.py
```

The test produced preopen context files and carried gap/movement fields into replay output.
