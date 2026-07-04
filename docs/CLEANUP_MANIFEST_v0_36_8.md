# v0.36.8 Cleanup Manifest

Generated: 2026-06-29T02:23:10

## Goal

Clean up files that were abandoned, duplicated, or carried forward across prior iterations while preserving the active scanner, visual watchlist, backtest replay, exit-path audit, analytics dashboard, candle data, and current reports.

## Kept active source files

```text
config.py
data_loader.py
zone_detector.py
watchlist.py
generate_watchlist_zone_map.py
download_alpaca_bars.py
build_backtest_snapshots.py
replay_backtest.py
audit_exit_paths.py
analyze_backtest.py
clean_project.py
requirements.txt
README.md
```

## Kept folders

```text
data/      input OHLCV files
reports/   current generated watchlist/backtest outputs
docs/      build notes and cleanup manifest
```

## Removed from the clean package

- Root-level generated watchlist/backtest CSV/HTML/MD outputs that duplicate reports/ outputs.
- Dated one-off snapshot exports at project root.
- Old backup files: analyze_backtest(backup).py and watchlist_pre_v0366_backup.py.
- Old patch archive folders: archive/ and reports/archive/.
- Duplicate/old docs: README (1).md and stale root CLEAN_BUILD_MANIFEST.md.
- Legacy standalone backtest.py; current backtest path is build_backtest_snapshots.py -> replay_backtest.py -> audit_exit_paths.py -> analyze_backtest.py.

## Notes

- No scanner, zone detection, watchlist scoring, backtest, or dashboard logic was intentionally changed.
- Generated outputs now live under `reports/` only.
- Historical backtest snapshots remain under `reports/backtest/snapshots/` because `replay_backtest.py` uses them.
- Current backtest analytics remain under `reports/backtest/analytics/`.
- Current visual watchlist outputs remain under `reports/watchlist.html` and `reports/watchlist_zone_map.html`.
```
