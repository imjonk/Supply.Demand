# v0.37.5a Watchlist Inside-Zone Exclusion Patch

Purpose: keep the watchlist focused on pre-trade preparation candidates, not stocks already sitting inside supply/demand zones.

## Changes

- `watchlist.py`
  - Added `_price_inside_zone(...)`.
  - Added `_symbols_inside_any_zone(...)`.
  - Live watchlist generation now excludes a symbol entirely if its current price is inside any active mapped zone.
  - Historical snapshot watchlist generation uses the same `build_watchlist_from_zone_snapshot(...)` path, so backtesting snapshots receive the same exclusion.
  - Updated report language to state that inside-zone candidates are excluded because they represent unresolved consolidation/chop.

## Intended behavior

- If current price is inside any active supply/demand zone, the symbol is excluded from the watchlist.
- No candidate should appear with `distance_pct == 0.00` due to being inside the candidate zone.
- The symbol can become a candidate again after price is outside all active zones and is approaching a mapped zone.

## Validation

- `python -m py_compile watchlist.py build_backtest_snapshots.py` passed.
- A synthetic unit check verified that symbols inside demand/supply zones are not returned by `build_watchlist_from_zone_snapshot(...)`.
