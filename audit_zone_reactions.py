"""Audit raw price reactions to historical merged-zone snapshots.

v0.37.0 / v0.37.1 support utility.

This script reuses reports/backtest/snapshots/*_merged_zones.csv as the
historical source of truth. It does not recreate zones and does not edit the
snapshot files. It asks: when price touched a previously mapped zone during the
next RTH session, did supply reject/break or demand hold/break?
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from config import DATA_DIR, REPORT_DIR
from data_loader import load_symbol_csv, regular_session_only, MARKET_TZ
from movement_context import compute_symbol_movement_context

SNAP_DIR = REPORT_DIR / "backtest" / "snapshots"


def _safe_float(x, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _session_bars(symbol: str, date_str: str, cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = symbol.upper()
    if key not in cache:
        p = DATA_DIR / f"{key}_5M.csv"
        if not p.exists():
            cache[key] = (pd.DataFrame(), pd.DataFrame())
        else:
            raw = load_symbol_csv(p)
            rth = regular_session_only(raw)
            cache[key] = (raw, rth)
    raw, rth = cache[key]
    if rth.empty:
        return raw, rth
    local = rth.tz_convert(MARKET_TZ) if getattr(rth.index, 'tz', None) is not None else rth
    day = local[local.index.strftime('%Y-%m-%d') == date_str].copy()
    return raw, day


def _first_touch_index(bars: pd.DataFrame, zone_type: str, bottom: float, top: float):
    if bars.empty:
        return None
    if zone_type == 'supply':
        touch = bars[pd.to_numeric(bars['high'], errors='coerce') >= bottom]
    else:
        touch = bars[pd.to_numeric(bars['low'], errors='coerce') <= top]
    if touch.empty:
        return None
    return touch.index[0]


def _context_before_touch(symbol: str, raw: pd.DataFrame, bars: pd.DataFrame, touch_ts, price: float):
    try:
        prior_raw = raw.loc[raw.index <= pd.Timestamp(touch_ts).tz_convert('UTC')].copy() if getattr(raw.index, 'tz', None) is not None else raw.copy()
        prior_rth = regular_session_only(prior_raw)
        return compute_symbol_movement_context(symbol, prior_raw, prior_rth, price, touch_ts)
    except Exception:
        return {}


def classify_zone_reaction(symbol: str, date_str: str, z: pd.Series, cache: dict, reaction_r: float = 1.0) -> dict | None:
    raw, bars = _session_bars(symbol, date_str, cache)
    if bars.empty:
        return None
    bottom = _safe_float(z.get('zone_bottom'))
    top = _safe_float(z.get('zone_top'))
    if bottom is None or top is None or top <= bottom:
        return None
    zone_type = str(z.get('zone_type', '')).lower()
    if zone_type not in ['supply', 'demand']:
        return None
    # Cheap prefilter by daily high/low.
    if zone_type == 'supply' and float(bars['high'].max()) < bottom:
        return None
    if zone_type == 'demand' and float(bars['low'].min()) > top:
        return None
    touch_ts = _first_touch_index(bars, zone_type, bottom, top)
    if touch_ts is None:
        return None
    touch_pos = bars.index.get_loc(touch_ts)
    after = bars.iloc[touch_pos:].copy()
    if after.empty:
        return None
    zone_h = max(top - bottom, 0.01)
    touch_row = after.iloc[0]
    price_at_touch = _safe_float(touch_row.get('close'))
    same_bar_ambiguous = False
    bars_to_resolution = None
    raw_outcome = 'unresolved'
    max_reaction = 0.0
    max_break = 0.0

    if zone_type == 'supply':
        broke_mask = pd.to_numeric(after['close'], errors='coerce') > top
        reject_mask = pd.to_numeric(after['low'], errors='coerce') <= bottom - zone_h * reaction_r
        first_broke = after[broke_mask].index[0] if broke_mask.any() else None
        first_reject = after[reject_mask].index[0] if reject_mask.any() else None
        max_reaction = max(0.0, (top - float(after['low'].min())) / zone_h)
        max_break = max(0.0, (float(after['high'].max()) - top) / zone_h)
        if first_broke is not None and first_reject is not None and first_broke == first_reject:
            raw_outcome = 'supply_ambiguous_same_bar'
            same_bar_ambiguous = True
            bars_to_resolution = int(after.index.get_loc(first_broke))
        elif first_broke is not None and (first_reject is None or after.index.get_loc(first_broke) < after.index.get_loc(first_reject)):
            raw_outcome = 'supply_broken'
            bars_to_resolution = int(after.index.get_loc(first_broke))
        elif first_reject is not None:
            raw_outcome = 'supply_rejected'
            bars_to_resolution = int(after.index.get_loc(first_reject))
        elif len(after) >= 6:
            raw_outcome = 'supply_chop'
    else:
        broke_mask = pd.to_numeric(after['close'], errors='coerce') < bottom
        hold_mask = pd.to_numeric(after['high'], errors='coerce') >= top + zone_h * reaction_r
        first_broke = after[broke_mask].index[0] if broke_mask.any() else None
        first_hold = after[hold_mask].index[0] if hold_mask.any() else None
        max_reaction = max(0.0, (float(after['high'].max()) - bottom) / zone_h)
        max_break = max(0.0, (bottom - float(after['low'].min())) / zone_h)
        if first_broke is not None and first_hold is not None and first_broke == first_hold:
            raw_outcome = 'demand_ambiguous_same_bar'
            same_bar_ambiguous = True
            bars_to_resolution = int(after.index.get_loc(first_broke))
        elif first_broke is not None and (first_hold is None or after.index.get_loc(first_broke) < after.index.get_loc(first_hold)):
            raw_outcome = 'demand_broken'
            bars_to_resolution = int(after.index.get_loc(first_broke))
        elif first_hold is not None:
            raw_outcome = 'demand_held'
            bars_to_resolution = int(after.index.get_loc(first_hold))
        elif len(after) >= 6:
            raw_outcome = 'demand_chop'

    # Keep this first-pass audit fast. Detailed structure/VPA context can be
    # joined later; the watchlist mainly needs symbol-level zone tendencies.
    ctx = {}
    return {
        'snapshot_date': date_str,
        'symbol': symbol,
        'zone_type': zone_type,
        'zone_timeframe': z.get('timeframe', ''),
        'zone_bottom': round(bottom, 3),
        'zone_top': round(top, 3),
        'quality_score': z.get('quality_score'),
        'freshness': z.get('freshness'),
        'tests': z.get('tests'),
        'base_time': z.get('base_time'),
        'touch_time': pd.Timestamp(touch_ts).strftime('%Y-%m-%d %H:%M %Z'),
        'price_at_touch': round(price_at_touch, 3) if price_at_touch is not None else None,
        'raw_zone_outcome': raw_outcome,
        'bars_to_resolution': bars_to_resolution,
        'same_bar_ambiguous': same_bar_ambiguous,
        'max_reaction_zone_r': round(max_reaction, 3),
        'max_break_zone_r': round(max_break, 3),
        'structure_or_move_context': ctx.get('recent_move_direction', ''),
        'recent_move_pct': ctx.get('recent_move_pct'),
        'price_vs_9ema': ctx.get('price_vs_9ema'),
        'price_vs_vwap': ctx.get('price_vs_vwap'),
        'volume_ratio_recent': ctx.get('volume_ratio_recent'),
        'vpa_state': ctx.get('vpa_state'),
    }


def build_summary(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    overall = events.groupby(['zone_type', 'raw_zone_outcome'], dropna=False).size().reset_index(name='events')
    sym_rows = []
    for sym, g in events.groupby('symbol'):
        supply = g[g['zone_type'].eq('supply')]
        demand = g[g['zone_type'].eq('demand')]
        row = {'symbol': sym, 'events': len(g), 'supply_events': len(supply), 'demand_events': len(demand)}
        if len(supply):
            row['supply_rejection_rate_pct'] = round(supply['raw_zone_outcome'].eq('supply_rejected').mean()*100, 2)
            row['supply_break_rate_pct'] = round(supply['raw_zone_outcome'].eq('supply_broken').mean()*100, 2)
        else:
            row['supply_rejection_rate_pct'] = 0.0; row['supply_break_rate_pct'] = 0.0
        if len(demand):
            row['demand_hold_rate_pct'] = round(demand['raw_zone_outcome'].eq('demand_held').mean()*100, 2)
            row['demand_break_rate_pct'] = round(demand['raw_zone_outcome'].eq('demand_broken').mean()*100, 2)
        else:
            row['demand_hold_rate_pct'] = 0.0; row['demand_break_rate_pct'] = 0.0
        sym_rows.append(row)
    by_symbol = pd.DataFrame(sym_rows).sort_values(['events','symbol'], ascending=[False, True])
    by_context = events.groupby(['zone_type','structure_or_move_context','vpa_state','raw_zone_outcome'], dropna=False).size().reset_index(name='events')
    return overall, by_symbol, by_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-snapshots', type=int, default=0, help='Optional cap for smoke tests; 0 = all snapshots')
    args = ap.parse_args()
    out_dir = REPORT_DIR
    out_dir.mkdir(exist_ok=True)
    files = sorted(SNAP_DIR.glob('*_merged_zones.csv'))
    if args.max_snapshots and args.max_snapshots > 0:
        files = files[:args.max_snapshots]
    cache = {}
    rows = []
    for path in files:
        date_str = path.name[:10]
        try:
            zones = pd.read_csv(path)
        except Exception:
            continue
        if zones.empty or 'symbol' not in zones.columns:
            continue
        zones = zones[zones.get('broken', False).astype(str).str.lower().isin(['false','0','nan','none',''])].copy() if 'broken' in zones.columns else zones
        for sym, g in zones.groupby(zones['symbol'].astype(str).str.upper()):
            raw, bars = _session_bars(sym, date_str, cache)
            if bars.empty:
                continue
            day_high = float(bars['high'].max()); day_low = float(bars['low'].min())
            g = g.copy()
            g['zb'] = pd.to_numeric(g['zone_bottom'], errors='coerce')
            g['zt'] = pd.to_numeric(g['zone_top'], errors='coerce')
            g = g.dropna(subset=['zb','zt'])
            # Only zones that the day could possibly touch.
            g = g[((g['zone_type'].eq('supply')) & (day_high >= g['zb'])) | ((g['zone_type'].eq('demand')) & (day_low <= g['zt']))]
            for _, z in g.iterrows():
                rec = classify_zone_reaction(sym, date_str, z, cache)
                if rec:
                    rows.append(rec)
    events = pd.DataFrame(rows)
    events.to_csv(out_dir / 'zone_reaction_events.csv', index=False)
    overall, by_symbol, by_context = build_summary(events)
    overall.to_csv(out_dir / 'zone_reaction_summary.csv', index=False)
    by_symbol.to_csv(out_dir / 'zone_reaction_by_symbol.csv', index=False)
    by_context.to_csv(out_dir / 'zone_reaction_by_context.csv', index=False)
    print(f'Wrote {out_dir / "zone_reaction_events.csv"} ({len(events)} events)')
    print(f'Wrote {out_dir / "zone_reaction_by_symbol.csv"}')


if __name__ == '__main__':
    main()
