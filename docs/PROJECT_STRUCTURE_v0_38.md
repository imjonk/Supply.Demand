# v0.38 Clean Project Structure

Use this as the target structure for the project folder.

## Keep these source files

```text
config.py
data_loader.py
zone_detector.py
watchlist.py
download_alpaca_bars.py
build_backtest_snapshots.py
replay_backtest.py
analyze_backtest.py
requirements.txt
README.md
.env.example
```

## Keep these folders

```text
data/      # input OHLCV files
reports/   # generated watchlist/backtest outputs
archive/   # old files moved by the cleanup script
```

## Defunct / safe to archive after patching

```text
apply_v0_37_confirmation_breakdown.py
README_v0_37_confirmation_breakdown.md
*.py.v036_backup
*.py.bak
__pycache__/
.pytest_cache/
```

## Generated outputs that should not live at project root anymore

These should be regenerated into `reports/`, not kept as source files:

```text
watchlist.html
watchlist.csv
watchlist.md
watchlist_rejections.html
watchlist_all_candidates.csv
scenario_watchlist.csv
zone_map.csv
detected_zones.csv
active_zones.csv
merged_zones.csv
backtest_*.csv
backtest.md
```

## Recommended cleanup commands

Dry run first:

```powershell
python clean_project.py
```

Apply safely by archiving old files:

```powershell
python clean_project.py --apply
```

Also archive reports if you want a blank output folder:

```powershell
python clean_project.py --apply --include-reports
```

Permanent deletion is available but not recommended:

```powershell
python clean_project.py --apply --delete
```
