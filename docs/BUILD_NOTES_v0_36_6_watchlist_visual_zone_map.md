# v0.36.6_watchlist_visual_zone_map

## Purpose
Add watchlist-facing visual context without changing the scanner's selection logic.

This build adds a chart-style zone-map view for Final / Actionable candidates and labels price structure using recent higher-high/higher-low or lower-high/lower-low movement.

## What changed

### New output

- `reports/watchlist_zone_map.html`
  - One chart-style card per Final / Actionable candidate.
  - Uses recent regular-session 5M candles.
  - Shades nearest active demand and supply zones.
  - Marks current price with a horizontal line.
  - Adds 9EMA and VWAP overlays.
  - Shows price position status, nearest demand/supply, structure alignment, and trade-vs-structure read.

- `reports/watchlist_zone_map.csv`
  - Same Final / Actionable candidates with added visual/context columns.

### Updated output

- `reports/watchlist.html`
  - Candidate cards now include mini vertical chart-style zone visuals.
  - Cards show structure alignment badges and trade-vs-structure labels.

- `reports/watchlist.csv`
  - Adds context columns such as:
    - `closest_demand_bottom`
    - `closest_demand_top`
    - `closest_supply_bottom`
    - `closest_supply_top`
    - `price_position_status`
    - `structure_bias_5m`
    - `structure_bias_15m`
    - `structure_bias_1d`
    - `structure_alignment`
    - `structure_trade_alignment`

### New script

- `generate_watchlist_zone_map.py`
  - Fast add-on script that enriches existing watchlist outputs and regenerates the zone-map visuals.
  - Use after `watchlist.py` if you want to refresh visuals without rerunning the full scanner.

## Structure-bias labels

Structure is computed from completed regular-session candles only.

- `Bullish HH/HL`
  - last confirmed swing high > prior swing high
  - last confirmed swing low > prior swing low

- `Bearish LH/LL`
  - last confirmed swing high < prior swing high
  - last confirmed swing low < prior swing low

- `Mixed / Transition`
  - conflicting structure, such as higher high with lower low or lower high with higher low

- `Range-bound`
  - recent swing highs/lows are effectively flat

- `Insufficient structure`
  - not enough confirmed swings available

## Important strategy note

This is visual context only.

The build does **not** change:

- zone detection
- final watchlist eligibility
- R:R filtering
- setup grading
- broken-zone exclusion logic
- backtest logic

## How to run

Full watchlist refresh:

```bash
python watchlist.py
```

Fast visual refresh from existing watchlist outputs:

```bash
python generate_watchlist_zone_map.py
```

Open:

```text
reports/watchlist_zone_map.html
reports/watchlist.html
```

## Current generated read from included sample

The included generated sample has 3 Final / Actionable candidates visualized:

- `AMZN — Demand Breakdown / Continuation — Puts`
  - structure: aligned bearish
  - trade vs structure: with structure

- `AAPL — Supply Breakout / Continuation — Calls`
  - structure: mixed, with 15M bullish context
  - price is inside nearest demand zone

- `AMZN — Supply Breakout / Continuation — Calls`
  - structure: aligned bearish
  - trade vs structure: counter-structure

This is useful as watchlist context: the bullish AMZN supply breakout is not automatically rejected, but it is clearly labeled as counter-structure.
