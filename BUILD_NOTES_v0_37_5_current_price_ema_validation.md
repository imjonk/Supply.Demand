# v0.37.5_current_price_ema_validation

## Purpose
This build fixes current-price source handling for the watchlist and tightens backtest validation around reversal entries and EMA protection exits.

## Current price / data download changes
- `download_alpaca_bars.py` still downloads the normal 5M and 1D OHLCV files.
- After the 5M download, it now writes `data/latest_market_prices.csv` from the newest downloaded 5M bar per symbol.
- This latest snapshot is intended for watchlist current price, zone proximity, and watchlist ranking.
- RTH-only logic is still used for zone construction and historical zone quality.
- Extended-hours/latest market data is allowed for current price/proximity, including weekend runs where the latest available market bar may be from the prior after-hours session.

## Watchlist changes
- `watchlist.py` now treats `data/latest_market_prices.csv` as the preferred current-price source.
- Manual `current_prices.csv` / scanner quote files are still supported, but they are only used if their as-of timestamp is at least as recent as the latest downloaded bar.
- This prevents stale premarket quote files from overriding newer downloaded bars.
- The report header now clarifies that current price uses latest downloaded market data while zones remain RTH-only.

## Backtest changes
- Reversal entries now require a completed zone rejection:
  - Demand reversal calls must close above the demand zone before entry.
  - Supply rejection puts must close below the supply zone before entry.
  - Inside-zone reversal confirmations are no longer valid entries.
- EMA protection now runs continuously after entry:
  - Calls/longs exit after 2 closes below 9EMA.
  - Puts/shorts exit after 2 closes above 9EMA.
  - The rule is no longer gated behind +1R or any profit threshold.
- `--ema-exit-after-r` is left as a deprecated compatibility argument but no longer gates EMA protection.

## Suggested run sequence
```bash
python download_alpaca_bars.py
python watchlist.py
python build_backtest_snapshots.py --start 2026-01-01 --end 2026-06-30
python replay_backtest.py --start 2026-01-01 --end 2026-06-30 --preset balanced --rr 3 --max-entry-time 13:00
python analyze_backtest.py
```

## Validation checks
- Confirm `data/latest_market_prices.csv` is created and has current `as_of` timestamps.
- Confirm watchlist cards show current price as-of timestamps from the latest available market data, not stale 8:00 AM snapshots when newer bars exist.
- Confirm reversal trades no longer have `entry_close_zone_location` inside the zone.
- Confirm EOD exits drop materially if EMA protection was previously being missed.
