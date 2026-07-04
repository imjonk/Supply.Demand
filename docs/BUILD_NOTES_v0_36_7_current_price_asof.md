# v0.36.7_current_price_asof

Purpose: make watchlist HTML price references timestamped so the viewer can distinguish the displayed current price from the later real-time market price.

Changes:
- Added `current_price_as_of` to watchlist rows.
- If a manual/current price override file includes a timestamp column, the watchlist uses that timestamp per symbol.
- If the override file has no timestamp column, the watchlist uses the file modified time as the quote snapshot time.
- If no override file is used, the watchlist uses the latest completed regular-session OHLCV candle time.
- `watchlist.html` now shows `as of ...` under each Current price value.
- `watchlist_zone_map.html` now shows `as of ...` in each price box.
- `watchlist.csv` and `watchlist_zone_map.csv` include `current_price_as_of`.

This change is display/audit-only and does not affect watchlist selection, grading, zone detection, or backtest logic.
