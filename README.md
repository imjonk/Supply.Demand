# Jonathan Supply/Demand Scanner + Backtester

Clean project package: `v37.6_clean_project`.

This build keeps the current watchlist-first scanner and the current backtest/audit workflow, while removing old root-level generated files, backups, patch archives, and duplicate outputs that had accumulated across prior iterations.

## Current workflow

### 1. Refresh/download data

```powershell
python download_alpaca_bars.py
```

### 2. Generate the current watchlist

```powershell
python watchlist.py
```

Main outputs:

```text
reports/watchlist.html
reports/watchlist_zone_map.html
reports/watchlist.csv
reports/watchlist_zone_map.csv
reports/watchlist_all_candidates.csv
reports/detected_zones.csv
reports/active_zones.csv
reports/merged_zones.csv
```

### 3. Build/replay/analyze the backtest

```powershell
python build_backtest_snapshots.py
python replay_backtest.py --preset balanced
python audit_exit_paths.py
python analyze_backtest.py
```

Main outputs:

```text
reports/backtest/trades.csv
reports/backtest/entry_candidates.csv
reports/backtest/strategy_dashboard.html
reports/backtest/analytics/
```

## Project rules preserved

- Watchlist-first logic is preserved.
- Backtesting remains separate and diagnostic.
- Regular trading hours only: 9:30 AM–4:00 PM New York time.
- 5M lookback default: 6 months.
- 1D lookback default: 1 year.
- Zone detection uses body-based departure: departure candle body must be greater than 2x basing candle body.
- Broken zones are audit-only and excluded from active watchlist/merge/confluence/target/scenario logic.

## Cleanup result

The active project root now contains only source scripts, `data/`, `reports/`, `docs/`, and this README. Generated CSV/HTML/MD outputs belong under `reports/`, not the root folder.

See `docs/CLEANUP_MANIFEST_v0_36_8.md` for the exact cleanup decisions.

## v0.37.2 Preopen context backtest

Historical snapshot building now creates both the original prior-close snapshots and simulated 8:00 AM preopen movement-context snapshots.

Recommended preopen backtest sequence:

```bash
python build_backtest_snapshots.py
python replay_backtest.py --preset balanced --snapshot-mode preopen
python audit_exit_paths.py
python analyze_backtest.py
```

The zones remain regular-session-only. Premarket/after-hours prices are used only for current-price, gap, volume, and movement context in the simulated watchlist.
