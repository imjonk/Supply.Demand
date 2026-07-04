# v0.37.1 Movement Context Watchlist

## Purpose

Shift the watchlist from proximity-first prioritization toward movement-context prioritization while keeping the trusted RTH zone engine intact.

The watchlist still uses mapped supply/demand zones, but the new movement-context output asks what price is doing around those zones:

- approaching supply/demand with strength or weakness
- rejecting away from a zone
- breaking through a zone
- trend/structure alignment using HH/HL and LH/LL context
- price versus 9EMA and VWAP
- recent volume/VPA behavior
- premarket/aftermarket gap context
- historical symbol-level zone reaction tendencies when available

## Important principles

- Zone creation, merging, broken-zone logic, target ladders, and backtesting remain RTH-only.
- Extended-hours price is used only as displayed/current context and must be labeled with timestamp/session.
- Movement-context scoring is informational. It does not rewrite the underlying zone detector.
- Historical zone tendencies come from `audit_zone_reactions.py` when available.

## New/updated files

- `movement_context.py`
- `audit_zone_reactions.py`
- `watchlist.py`
- `generate_watchlist_zone_map.py`

## New outputs

- `reports/movement_context_watchlist.html`
- `reports/movement_context_watchlist.csv`
- `reports/zone_reaction_events.csv`
- `reports/zone_reaction_summary.csv`
- `reports/zone_reaction_by_symbol.csv`
- `reports/zone_reaction_by_context.csv`

## New watchlist columns

- `zone_thesis`
- `zone_movement_state`
- `recent_move_direction`
- `recent_move_pct`
- `recent_move_strength`
- `price_vs_9ema`
- `price_vs_vwap`
- `volume_state`
- `vpa_state`
- `gap_direction`
- `gap_pct`
- `gap_zone_context`
- `current_price_session`
- `historical_zone_tendency`
- `historical_reaction_score`
- `observation_score`
- `observation_reason`
- `watch_for`
- `movement_watchlist_bucket`

## Usage

Normal watchlist flow:

```bash
python watchlist.py
```

Refresh visual/movement context from existing watchlist outputs:

```bash
python generate_watchlist_zone_map.py
```

Run raw zone reaction audit from existing snapshots:

```bash
python audit_zone_reactions.py
```

For a faster smoke test:

```bash
python audit_zone_reactions.py --max-snapshots 5
```

## Notes

The included package was regenerated with the current available reports and a 5-snapshot zone reaction audit sample so the movement-context watchlist has historical tendency columns available immediately. For fuller historical tendency statistics, run `python audit_zone_reactions.py` locally; it may take longer because it walks historical merged-zone snapshots.
