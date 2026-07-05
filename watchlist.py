from pathlib import Path
from datetime import datetime, time
from zoneinfo import ZoneInfo
import html
import argparse
import pandas as pd

from config import (
    DATA_DIR, REPORT_DIR, WATCHLIST, TIMEFRAMES, SOURCE_SUFFIX_PRIORITY,
    FINAL_REPORT_MIN_GRADE, MAX_SETUPS_PER_SECTION, MIN_FINAL_RR,
    MERGE_OVERLAPPING_ZONES, ZONE_MERGE_TOLERANCE_PCT, MAX_MERGED_ZONE_WIDTH_PCT,
    REQUIRE_5M_SOURCE_FOR_WATCHLIST, STRICT_DEDUPLICATE_FINAL_WATCHLIST, FINAL_DEDUPE_TOLERANCE_PCT,
    TARGET_ZONE_MAX_TESTS, TARGET_ZONE_MIN_QUALITY_SCORE, TARGET_SELECTION_MIN_RR,
    MAX_FINAL_ZONE_TESTS, MAX_RESEARCH_ZONE_TESTS, ELIMINATE_ZONE_TESTS_AT,
    ALLOW_TWO_TEST_ZONES_WITH_CONFLUENCE, TWO_TEST_MIN_CONFLUENCE, TWO_TEST_MIN_QUALITY_SCORE,
    PREFERRED_RR_MIN, MAX_MODELED_TARGET_RR,
    TARGET_LADDER_MAX_LEVELS, TARGET_LADDER_SHOW_SOFT_OBSTACLES,
    WATCHLIST_SCENARIO_MAX_DISTANCE_PCT, WATCHLIST_DEVELOPING_MAX_DISTANCE_PCT,
    WATCHLIST_READY_DISTANCE_PCT, WATCHLIST_NEEDS_CONFIRMATION_DISTANCE_PCT,
    WATCHLIST_INCLUDE_DEVELOPING_IN_HTML, WATCHLIST_INCLUDE_ZONE_MAP_IN_HTML,
    WATCHLIST_MIN_DEVELOPING_GRADE_RANK,
    EXCLUDE_BROKEN_ZONES_FROM_WATCHLIST_CALCULATIONS,
)
from data_loader import load_symbol_csv, aggregate_bars, regular_session_only, MARKET_TZ
from zone_detector import ZONE_METADATA_FIELDS, detect_zones
from movement_context import (
    compute_symbol_movement_context, classify_market_session,
    load_zone_reaction_history, enrich_movement_context,
    generate_movement_context_html,
)


def _report_datestamps():
    """Return archive-friendly date stamps in New York market time."""
    now = datetime.now(ZoneInfo("America/New_York"))
    return now, now.strftime("%Y-%m-%d"), now.strftime("%Y%m%d_%H%M")


def _clean_float(value):
    if pd.isna(value):
        return None
    s = str(value).strip().replace('$', '').replace(',', '').replace('%', '')
    try:
        return float(s)
    except Exception:
        return None


def _read_csv_with_optional_header(path: Path) -> pd.DataFrame:
    """Read normal CSVs or Thinkorswim scanner exports that have title rows above the header."""
    text = path.read_text(encoding='utf-8-sig', errors='ignore').splitlines()
    header_idx = 0
    for i, line in enumerate(text):
        cols = [c.strip().lower() for c in line.split(',')]
        if 'symbol' in cols and ('last' in cols or 'close' in cols or 'price' in cols):
            header_idx = i
            break
    return pd.read_csv(path, skiprows=header_idx, encoding='utf-8-sig')


def _current_price_source_candidates() -> list[Path]:
    candidates = [
        # Written by download_alpaca_bars.py. This should be the default current-price
        # source for watchlist proximity because it comes from the newest downloaded
        # 5M market bar, including extended hours when available.
        DATA_DIR / 'latest_market_prices.csv',
        Path(__file__).resolve().parent / 'latest_market_prices.csv',
        # Manual/quote overrides remain supported, but build_watchlist only uses them
        # when their as-of timestamp is at least as recent as the downloaded bars.
        DATA_DIR / 'current_prices.csv',
        Path(__file__).resolve().parent / 'current_prices.csv',
    ]
    candidates.extend(sorted(DATA_DIR.glob('*WatchListScanner*.csv')))
    candidates.extend(sorted(Path(__file__).resolve().parent.glob('*WatchListScanner*.csv')))
    return candidates


def _format_price_timestamp(value) -> str:
    """Return a compact New York-time timestamp for price-as-of labels."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ''
        if ts.tzinfo is None:
            ts = ts.tz_localize(MARKET_TZ)
        else:
            ts = ts.tz_convert(MARKET_TZ)
        return ts.strftime('%Y-%m-%d %I:%M %p %Z')
    except Exception:
        text = str(value).strip()
        return text




def _parse_price_timestamp(value):
    """Parse a current-price timestamp into New York time, returning None on failure."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize(MARKET_TZ)
        else:
            ts = ts.tz_convert(MARKET_TZ)
        return ts
    except Exception:
        return None


def _market_session_label(value) -> str:
    ts = _parse_price_timestamp(value)
    if ts is None:
        return ''
    t = ts.time()
    if pd.Timestamp(ts.date()).dayofweek >= 5:
        return 'weekend_latest_available'
    if t >= pd.Timestamp('09:30').time() and t < pd.Timestamp('16:00').time():
        return 'RTH'
    if t >= pd.Timestamp('04:00').time() and t < pd.Timestamp('09:30').time():
        return 'premarket'
    if t >= pd.Timestamp('16:00').time() and t < pd.Timestamp('20:00').time():
        return 'aftermarket'
    return 'outside_extended_hours'

def load_current_prices() -> tuple[dict, str]:
    """
    Optional override. Put one of these files in either project root or data/:
      - current_prices.csv
      - WatchListScanner.csv / any csv with Symbol and Last columns

    Expected columns can be: symbol,last OR Symbol,Last OR symbol,price OR symbol,close.
    """
    for path in _current_price_source_candidates():
        if not path.exists():
            continue
        try:
            df = _read_csv_with_optional_header(path)
            cols = {c.lower().strip(): c for c in df.columns}
            if 'symbol' not in cols:
                continue
            price_col = None
            for candidate in ['last', 'price', 'close', 'mark']:
                if candidate in cols:
                    price_col = cols[candidate]
                    break
            if price_col is None:
                continue
            symbol_col = cols['symbol']
            prices = {}
            for _, row in df.iterrows():
                sym = str(row[symbol_col]).strip().upper()
                px = _clean_float(row[price_col])
                if sym and px is not None:
                    prices[sym] = px
            if prices:
                return prices, str(path.name)
        except Exception:
            continue
    return {}, 'latest delayed OHLCV close'


def load_current_price_as_of_overrides() -> dict:
    """
    Best-effort timestamp lookup for manual/Thinkorswim current price files.

    If the file has a timestamp column, use the per-row value. Otherwise, use
    the file modified time as the quote snapshot time. This is informational
    only and does not affect watchlist eligibility.
    """
    for path in _current_price_source_candidates():
        if not path.exists():
            continue
        try:
            df = _read_csv_with_optional_header(path)
            cols = {c.lower().strip(): c for c in df.columns}
            if 'symbol' not in cols:
                continue
            price_col = None
            for candidate in ['last', 'price', 'close', 'mark']:
                if candidate in cols:
                    price_col = cols[candidate]
                    break
            if price_col is None:
                continue
            symbol_col = cols['symbol']
            asof_col = None
            for candidate in ['as_of', 'asof', 'timestamp', 'datetime', 'date time', 'time', 'quote_time', 'last_time', 'updated', 'updated_at']:
                if candidate in cols:
                    asof_col = cols[candidate]
                    break
            fallback_asof = _format_price_timestamp(datetime.fromtimestamp(path.stat().st_mtime, tz=MARKET_TZ))
            out = {}
            for _, row in df.iterrows():
                sym = str(row[symbol_col]).strip().upper()
                px = _clean_float(row[price_col])
                if not sym or px is None:
                    continue
                if asof_col:
                    out[sym] = _format_price_timestamp(row.get(asof_col)) or fallback_asof
                else:
                    out[sym] = fallback_asof
            if out:
                return out
        except Exception:
            continue
    return {}



def _timeframe_rank(tf: str) -> int:
    return {'1D': 6, '4H': 5, '3H': 4, '2H': 3, '90m': 2, '1H': 1}.get(str(tf), 0)


def _zone_metadata_payload(row: pd.Series) -> dict:
    return {field: row.get(field, '') for field in ZONE_METADATA_FIELDS}


def _best_freshness(values) -> str:
    order = {'fresh': 3, 'one_test': 2, 'multiple_tests': 1, 'broken': 0}
    vals = [str(v) for v in values if str(v) in order]
    if not vals:
        return str(list(values)[0]) if len(values) else 'unknown'
    return max(vals, key=lambda v: order[v])


def _merge_zone_cluster(cluster: list[pd.Series]) -> dict:
    """Combine overlapping same-symbol/same-type zones into one watchlist zone.

    We keep raw zones in detected_zones.csv, but the final watchlist should not show
    multiple duplicate cards just because 1H/2H/3H/4H created nearly identical zones.
    The merged zone uses the full price span of the overlapping cluster and preserves
    confluence details in extra columns.
    """
    df = pd.DataFrame([c.to_dict() for c in cluster])
    df['tf_rank'] = df['timeframe'].apply(_timeframe_rank)
    df = df.sort_values(['tf_rank', 'quality_score'], ascending=[False, False])
    primary = df.iloc[0].to_dict()

    bottom = float(df['zone_bottom'].min())
    top = float(df['zone_top'].max())
    tfs = sorted(df['timeframe'].unique(), key=_timeframe_rank, reverse=True)
    patterns = sorted(df['pattern'].unique())
    freshness = _best_freshness(df['freshness'].tolist())
    tests = int(df['tests'].min()) if 'tests' in df else 0
    q = float(df['quality_score'].max()) + min(1.5, 0.5 * (len(df) - 1))

    primary.update({
        'timeframe': '/'.join(tfs),
        'zone_timeframe': '/'.join(tfs),
        'pattern': '/'.join(patterns),
        'zone_bottom': round(bottom, 2),
        'zone_top': round(top, 2),
        'freshness': freshness,
        'tests': tests,
        'quality_score': round(q, 2),
        'confluence_count': int(len(df)),
        'confluence_timeframes': ','.join(tfs),
        'nested_inside_higher_tf': bool(len(df) > 1),
        'higher_tf_zone_count': int(len(tfs)),
        'overlapping_zone_count': int(len(df)),
        'confluence_score': round(float(min(10, len(df) * 2)), 2),
        'merged_from': '; '.join(
            f"{r['timeframe']} {r['zone_type']} {float(r['zone_bottom']):.2f}-{float(r['zone_top']):.2f} {r['pattern']}" 
            for _, r in df.iterrows()
        ),
        'primary_timeframe': df.iloc[0]['timeframe'],
        'primary_base_time': df.iloc[0].get('base_time', ''),
    })
    primary.pop('tf_rank', None)
    return primary


def _active_zones_for_watchlist(zones_df: pd.DataFrame) -> pd.DataFrame:
    """Return only zones valid for watchlist/merge/target calculations.

    Broken/expired zones are useful for audit or future backtesting, but they
    should not participate in merge calculations, confluence scoring, target
    ladders, R:R calculations, or scenario generation.
    """
    if zones_df.empty or not EXCLUDE_BROKEN_ZONES_FROM_WATCHLIST_CALCULATIONS:
        return zones_df.copy()
    out = zones_df.copy()
    if 'broken' in out.columns:
        # Handles booleans and CSV/string variants.
        broken = out['broken'].astype(str).str.lower().isin(['true', '1', 'yes'])
        out = out.loc[~broken].copy()
    if 'freshness' in out.columns:
        out = out.loc[out['freshness'].astype(str).str.lower().ne('broken')].copy()
    return out


def merge_overlapping_zones(zones_df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping/nearby active zones by symbol/type for cleaner output.

    v0.32: broken zones are filtered before merge so an expired child zone cannot
    contaminate merged freshness, tests, quality, confluence, or target logic.
    """
    zones_df = _active_zones_for_watchlist(zones_df)
    if zones_df.empty or not MERGE_OVERLAPPING_ZONES:
        out = zones_df.copy()
        if not out.empty and 'confluence_count' not in out.columns:
            out['confluence_count'] = 1
            out['confluence_timeframes'] = out['timeframe'].astype(str)
            out['merged_from'] = ''
            out['primary_timeframe'] = out['timeframe'].astype(str)
        return out

    merged = []
    for (symbol, zone_type), group in zones_df.groupby(['symbol', 'zone_type']):
        group = group.sort_values(['zone_bottom', 'zone_top']).copy()
        cluster = []
        cluster_bottom = cluster_top = None

        for _, row in group.iterrows():
            bottom = float(row['zone_bottom'])
            top = float(row['zone_top'])
            midpoint = max((bottom + top) / 2, 0.01)
            tolerance = midpoint * (ZONE_MERGE_TOLERANCE_PCT / 100.0)

            if not cluster:
                cluster = [row]
                cluster_bottom, cluster_top = bottom, top
                continue

            proposed_bottom = min(cluster_bottom, bottom)
            proposed_top = max(cluster_top, top)
            proposed_mid = max((proposed_bottom + proposed_top) / 2, 0.01)
            proposed_width_pct = (proposed_top - proposed_bottom) / proposed_mid * 100.0

            overlaps_or_near = bottom <= (cluster_top + tolerance)
            width_ok = proposed_width_pct <= MAX_MERGED_ZONE_WIDTH_PCT

            if overlaps_or_near and width_ok:
                cluster.append(row)
                cluster_bottom, cluster_top = proposed_bottom, proposed_top
            else:
                merged.append(_merge_zone_cluster(cluster))
                cluster = [row]
                cluster_bottom, cluster_top = bottom, top

        if cluster:
            merged.append(_merge_zone_cluster(cluster))

    return pd.DataFrame(merged)


def _distance_to_zone(price: float, zone_type: str, top: float, bottom: float) -> float:
    if zone_type == 'demand':
        if bottom <= price <= top:
            return 0.0
        return abs(price - top if price > top else bottom - price) / price * 100
    else:
        if bottom <= price <= top:
            return 0.0
        return abs(bottom - price if price < bottom else price - top) / price * 100


def _status(distance: float) -> str:
    if distance <= 0.25:
        return 'Immediate'
    if distance <= 2.0:
        return 'Near-Term'
    if distance <= 5.0:
        return 'Future'
    return 'Too Far'


def _estimate_rr_to_level(price: float, row: pd.Series, target: float):
    """Estimate reward:risk to a concrete target level."""
    zone_height = abs(float(row['zone_top']) - float(row['zone_bottom']))
    if zone_height <= 0:
        return None

    if row['zone_type'] == 'demand':
        risk = max(price - float(row['zone_bottom']), zone_height)
        reward = max(0, target - price)
    else:
        risk = max(float(row['zone_top']) - price, zone_height)
        reward = max(0, price - target)

    if risk <= 0:
        return None
    return round(float(reward / risk), 2)


def _estimate_rr_to_level_for_side(price: float, row: pd.Series, target: float, side: str):
    """Estimate reward:risk to a concrete target level using trade direction.

    v0.23 fixes breakout scenarios. A bullish supply breakout uses the broken supply
    zone as the risk area but still needs long-side R:R. A bearish demand breakdown
    uses the broken demand zone as the risk area but still needs short-side R:R.
    """
    zone_height = abs(float(row['zone_top']) - float(row['zone_bottom']))
    if zone_height <= 0:
        return None

    side = str(side).lower()
    if side == 'long':
        risk = max(price - float(row['zone_bottom']), zone_height)
        reward = max(0, target - price)
    else:
        risk = max(float(row['zone_top']) - price, zone_height)
        reward = max(0, price - target)

    if risk <= 0:
        return None
    return round(float(reward / risk), 2)


def _distance_to_trigger_pct(price: float, trigger: float, in_zone: bool = False) -> float:
    if in_zone:
        return 0.0
    return abs(float(trigger) - float(price)) / max(float(price), 0.01) * 100.0


def _price_inside_zone(price: float, top: float, bottom: float) -> bool:
    """Return True when the current price is inside the zone boundaries.

    Watchlist candidates are pre-trade preparation items. A stock already
    inside the candidate zone is unresolved/choppy and should not be shown as
    Immediate/Actionable. This applies to live watchlists and historical
    watchlist snapshots.
    """
    try:
        price_f = float(price)
        top_f = float(top)
        bottom_f = float(bottom)
    except Exception:
        return False
    lo = min(bottom_f, top_f)
    hi = max(bottom_f, top_f)
    return lo <= price_f <= hi


def _hard_exclusion_reason(row: pd.Series) -> str:
    price = _safe_float(row.get('current_price'))
    top = _safe_float(row.get('zone_top'))
    bottom = _safe_float(row.get('zone_bottom'))
    if price is None or top is None or bottom is None:
        return ''
    if _price_inside_zone(price, top, bottom):
        return 'inside_zone_consolidation'
    zt = str(row.get('zone_type', '')).lower()
    lo, hi = min(bottom, top), max(bottom, top)
    state = str(row.get('zone_movement_state', ''))
    scenario = str(row.get('scenario', ''))

    if scenario == 'demand_hold' and zt == 'demand' and price < lo:
        return 'zone_already_resolved_breakout'
    if scenario == 'supply_reject' and zt == 'supply' and price > hi:
        return 'zone_already_resolved_breakout'
    if scenario == 'demand_hold' and state == 'bouncing_from_demand':
        return 'zone_already_resolved_rejection'
    if scenario == 'supply_reject' and state == 'rejecting_from_supply':
        return 'zone_already_resolved_rejection'
    return ''


def _apply_hard_exclusions(watch_df: pd.DataFrame) -> pd.DataFrame:
    if watch_df is None or watch_df.empty:
        return watch_df
    reasons = watch_df.apply(_hard_exclusion_reason, axis=1)
    out = watch_df.loc[reasons.eq('')].copy()
    out.attrs['hard_excluded_candidates'] = int(reasons.ne('').sum())
    out.attrs['hard_exclusion_counts'] = reasons[reasons.ne('')].value_counts().to_dict()
    return out


def _target_zone_quality(row: pd.Series) -> float:
    """Score an opposing target zone.

    We prefer targets that are still meaningful institutional zones: fresh, unbroken
    by the detector, high-quality departure/volume, and higher-timeframe/confluence.
    Zones with 4+ tests are excluded before scoring.
    """
    freshness = str(row.get('freshness', ''))
    tests = int(row.get('tests', 0) or 0)
    if freshness == 'broken' or tests > int(TARGET_ZONE_MAX_TESTS):
        return -999.0
    q = float(row.get('quality_score', 0) or 0)
    fresh_bonus = {'fresh': 3.0, 'one_test': 2.0, 'multiple_tests': 0.25}.get(freshness, 0.0)
    test_penalty = max(0, tests) * 0.75
    confluence_bonus = min(2.0, 0.45 * max(0, int(row.get('confluence_count', 1) or 1) - 1))
    tf_text = str(row.get('timeframe', ''))
    tf_bonus = 0.0
    for tf in tf_text.split('/'):
        tf_bonus = max(tf_bonus, {'1D': 3.0, '4H': 2.0, '3H': 1.5, '2H': 1.0, '90m': 0.75, '1H': 0.25}.get(tf.strip(), 0.0))
    return round(q + fresh_bonus + confluence_bonus + tf_bonus - test_penalty, 3)


def _target_ladder_label(t: pd.Series, target_level: float, rr: float, t_quality: float, role: str, obstacle_reason: str | None = None) -> str:
    tests = int(t.get('tests', 0) or 0)
    freshness = str(t.get('freshness', 'unknown'))
    base = f'{t["timeframe"]} {t["zone_type"]} @ {target_level:.2f}'
    details = f'RR {rr:.2f}, {freshness}, tests {tests}, quality {t_quality:.1f}'
    if role == 'soft_obstacle':
        details += f', soft obstacle{(": " + obstacle_reason) if obstacle_reason else ""}'
    return f'{base} ({details})'


def _build_target_ladder(zones: pd.DataFrame, symbol: str, zone_type: str, price: float, source_zone: pd.Series, side: str | None = None) -> dict:
    """Build a path-based opposing-zone ladder.

    v0.22 fixes an R:R inflation problem: a short should not skip nearby unbroken
    demand zones and claim a far demand zone as the only target. Broken zones are
    eliminated. Every unbroken opposing zone in the trade path is considered either
    a hard target or a soft obstacle. Final modeled R:R uses T1, the nearest
    unbroken opposing zone in the path. Farther zones are shown as an R:R range and
    ladder for discretion.
    """
    trade_side = str(side or ('long' if zone_type == 'demand' else 'short')).lower()
    # Exclude the source zone itself, but do not exclude all zones of the same type.
    # For bullish supply breakouts, the next target is another supply zone above.
    # For bearish demand breakdowns, the next target is another demand zone below.
    subset = zones[zones['symbol'] == symbol].copy()
    if subset.empty:
        return {
            'target': None, 'estimated_rr': None, 'target_quality': None, 'target_rr_raw': None,
            'target_level': None, 'target_reject_reason': 'no_opposing_zone',
            'target_rr_range': None, 'target_ladder': None, 'target_1_zone': None,
            'target_1_rr': None, 'target_1_freshness': None, 'target_1_tests': None,
            'target_1_role': None, 'target_2_zone': None, 'target_2_rr': None,
            'target_2_freshness': None, 'target_2_tests': None, 'target_2_role': None,
            'target_3_zone': None, 'target_3_rr': None, 'target_3_freshness': None,
            'target_3_tests': None, 'target_3_role': None, 'intervening_zone_count': 0,
            'nearest_obstacle_zone': None, 'target_selection_reason': 'no_opposing_zone',
        }

    rows = []
    reject_counts = {}
    def reject(reason):
        reject_counts[reason] = reject_counts.get(reason, 0) + 1

    source_key = (
        source_zone.get('symbol'), source_zone.get('timeframe'), source_zone.get('zone_type'),
        source_zone.get('base_time'), source_zone.get('zone_top'), source_zone.get('zone_bottom')
    )

    for _, t in subset.iterrows():
        t_key = (
            t.get('symbol'), t.get('timeframe'), t.get('zone_type'),
            t.get('base_time'), t.get('zone_top'), t.get('zone_bottom')
        )
        if t_key == source_key:
            continue
        freshness = str(t.get('freshness', ''))
        tests = int(t.get('tests', 0) or 0)
        if freshness == 'broken':
            reject('target_broken')
            continue

        if trade_side == 'long':
            # Long setup: first contact with supply is its bottom.
            if str(t.get('zone_type')) != 'supply':
                continue
            target_level = float(t['zone_bottom'])
            if target_level <= price:
                reject('target_wrong_side')
                continue
            distance = target_level - price
        else:
            # Short setup: first contact with demand is its top.
            if str(t.get('zone_type')) != 'demand':
                continue
            target_level = float(t['zone_top'])
            if target_level >= price:
                reject('target_wrong_side')
                continue
            distance = price - target_level

        rr_raw = _estimate_rr_to_level_for_side(price, source_zone, target_level, trade_side)
        if rr_raw is None:
            reject('target_rr_missing')
            continue

        t_quality = _target_zone_quality(t)
        obstacle_reason = None
        role = 'hard_target'
        if tests >= int(ELIMINATE_ZONE_TESTS_AT):
            role = 'soft_obstacle'
            obstacle_reason = '4plus_tests'
        elif tests > int(TARGET_ZONE_MAX_TESTS):
            role = 'soft_obstacle'
            obstacle_reason = 'too_many_tests'
        elif t_quality < float(TARGET_ZONE_MIN_QUALITY_SCORE):
            role = 'soft_obstacle'
            obstacle_reason = 'low_quality'

        if role == 'soft_obstacle' and not TARGET_LADDER_SHOW_SOFT_OBSTACLES:
            reject('target_soft_obstacle_hidden')
            continue

        rows.append({
            'target_level': target_level,
            'rr_raw': round(float(rr_raw), 2),
            'rr_modeled': round(min(float(rr_raw), float(MAX_MODELED_TARGET_RR)), 2),
            'target_quality': t_quality,
            'distance': distance,
            'role': role,
            'obstacle_reason': obstacle_reason,
            'freshness': freshness,
            'tests': tests,
            'timeframe': str(t.get('timeframe', '')),
            'zone_type': str(t.get('zone_type', '')),
            'zone_bottom': float(t.get('zone_bottom')),
            'zone_top': float(t.get('zone_top')),
            'label': _target_ladder_label(t, target_level, float(rr_raw), t_quality, role, obstacle_reason),
        })

    if not rows:
        reason = max(reject_counts.items(), key=lambda kv: kv[1])[0] if reject_counts else 'no_valid_target'
        return {
            'target': None, 'estimated_rr': None, 'target_quality': None, 'target_rr_raw': None,
            'target_level': None, 'target_reject_reason': reason,
            'target_rr_range': None, 'target_ladder': None, 'target_1_zone': None,
            'target_1_rr': None, 'target_1_freshness': None, 'target_1_tests': None,
            'target_1_role': None, 'target_2_zone': None, 'target_2_rr': None,
            'target_2_freshness': None, 'target_2_tests': None, 'target_2_role': None,
            'target_3_zone': None, 'target_3_rr': None, 'target_3_freshness': None,
            'target_3_tests': None, 'target_3_role': None, 'intervening_zone_count': 0,
            'nearest_obstacle_zone': None, 'target_selection_reason': reason,
        }

    # Path order: for longs, closest supply above first; for shorts, closest demand below first.
    rows.sort(key=lambda r: r['distance'])
    ladder = rows[:int(TARGET_LADDER_MAX_LEVELS)]
    primary = ladder[0]

    rr_values = [float(r['rr_raw']) for r in ladder]
    rr_range = f"1:{min(rr_values):.2f}–1:{max(rr_values):.2f}" if len(rr_values) > 1 else f"1:{rr_values[0]:.2f}"
    target_ladder = ' | '.join(f"T{i+1}: {r['label']}" for i, r in enumerate(ladder))
    nearest_obstacle_zone = primary['label'] if primary['role'] == 'soft_obstacle' else None

    # Official target is T1. This prevents far clean zones from overinflating R:R
    # when there are nearer unbroken opposing zones in the trade path.
    result = {
        'target': primary['label'],
        'estimated_rr': primary['rr_modeled'],
        'target_quality': primary['target_quality'],
        'target_rr_raw': primary['rr_raw'],
        'target_level': primary['target_level'],
        'target_reject_reason': None,
        'target_rr_range': rr_range,
        'target_ladder': target_ladder,
        'intervening_zone_count': max(0, len(ladder) - 1),
        'nearest_obstacle_zone': nearest_obstacle_zone,
        'target_selection_reason': 't1_nearest_unbroken_opposing_zone',
    }
    for i in range(3):
        prefix = f'target_{i+1}'
        if i < len(ladder):
            r = ladder[i]
            result[f'{prefix}_zone'] = r['label']
            result[f'{prefix}_rr'] = r['rr_raw']
            result[f'{prefix}_freshness'] = r['freshness']
            result[f'{prefix}_tests'] = r['tests']
            result[f'{prefix}_role'] = r['role']
        else:
            result[f'{prefix}_zone'] = None
            result[f'{prefix}_rr'] = None
            result[f'{prefix}_freshness'] = None
            result[f'{prefix}_tests'] = None
            result[f'{prefix}_role'] = None
    return result


def _select_best_target(zones: pd.DataFrame, symbol: str, zone_type: str, price: float, source_zone: pd.Series):
    """Compatibility wrapper for older callers; v0.22 uses _build_target_ladder."""
    d = _build_target_ladder(zones, symbol, zone_type, price, source_zone)
    return d['target'], d['estimated_rr'], d['target_quality'], d['target_rr_raw'], d['target_level'], d['target_reject_reason']

def _entry_zone_rejection_reasons(row: pd.Series) -> list[str]:
    """Explain why a candidate is not eligible for the final watchlist."""
    reasons = []
    freshness = str(row.get('freshness', ''))
    tests = int(row.get('tests', 0) or 0)
    quality = float(row.get('quality_score', 0) or 0)
    confluence = int(row.get('confluence_count', 1) or 1)

    if freshness == 'broken':
        reasons.append('rejected_broken_entry_zone')
    if tests >= int(ELIMINATE_ZONE_TESTS_AT):
        reasons.append('rejected_4plus_entry_tests')
    elif tests > int(MAX_RESEARCH_ZONE_TESTS):
        reasons.append('rejected_entry_tests_above_research_limit')
    elif tests > int(MAX_FINAL_ZONE_TESTS):
        reasons.append('research_only_3_tests')
    elif tests == 2:
        allowed_two_test = bool(ALLOW_TWO_TEST_ZONES_WITH_CONFLUENCE) and (
            confluence >= int(TWO_TEST_MIN_CONFLUENCE) or quality >= float(TWO_TEST_MIN_QUALITY_SCORE)
        )
        if not allowed_two_test:
            reasons.append('rejected_2_tests_without_confluence_or_quality')
    return reasons


def _setup_quality_score(row: pd.Series) -> float:
    """Balanced setup score: entry quality plus target/R:R/freshness/confluence."""
    entry_q = min(10.0, float(row.get('quality_score', 0) or 0))
    target_q = min(10.0, float(row.get('target_quality', 0) or 0)) if pd.notna(row.get('target_quality', None)) else 0.0
    rr = float(row.get('estimated_rr', 0) or 0) if pd.notna(row.get('estimated_rr', None)) else 0.0
    rr_component = min(10.0, (rr / max(float(MIN_FINAL_RR), 0.01)) * 7.5)
    tests = int(row.get('tests', 0) or 0)
    fresh_component = {0: 10.0, 1: 8.0, 2: 6.0, 3: 3.0}.get(tests, 0.0)
    confluence = int(row.get('confluence_count', 1) or 1)
    confluence_component = min(10.0, 5.0 + 2.0 * max(0, confluence - 1))
    distance = float(row.get('distance_pct', 99) or 99)
    distance_component = max(0.0, 10.0 - distance * 2.0)

    return round(
        entry_q * 0.25
        + fresh_component * 0.15
        + confluence_component * 0.15
        + target_q * 0.20
        + rr_component * 0.20
        + distance_component * 0.05,
        3,
    )

def _estimate_rr(price: float, row: pd.Series, target_text: str | None):
    # Kept for compatibility with older files/reports.
    if not target_text:
        return None
    try:
        target = float(target_text.split('@')[-1].split()[0].strip())
    except Exception:
        return None
    return _estimate_rr_to_level(price, row, target)


def _departure_grade(x: float) -> str:
    x = abs(float(x))
    if x >= 2.0:
        return 'Excellent'
    if x >= 1.5:
        return 'Strong'
    if x >= 1.0:
        return 'Moderate'
    return 'Weak'


def _volume_grade(x: float) -> str:
    x = float(x)
    if x >= 1.5:
        return 'Excellent'
    if x >= 1.2:
        return 'Strong'
    if x >= 0.9:
        return 'Average'
    return 'Weak'


def _setup_grade(score: float, rr=None) -> str:
    s = float(score)
    if s >= 8.8:
        return 'A+'
    if s >= 7.5:
        return 'A'
    if s >= 6.5:
        return 'B+'
    if s >= 5.0:
        return 'B'
    return 'C'


def _grade_rank_value(grade: str) -> int:
    return {'A+': 5, 'A': 4, 'B+': 3, 'B': 2, 'C': 1}.get(str(grade), 0)


def _option_contract_label(side: str) -> str:
    return 'Calls' if str(side).lower() == 'long' else 'Puts'


def _scenario_display_label(scenario: str, side: str | None = None) -> str:
    labels = {
        'demand_hold': 'Demand Reversal / Hold',
        'demand_break': 'Demand Breakdown / Continuation',
        'supply_reject': 'Supply Rejection',
        'supply_break': 'Supply Breakout / Continuation',
        'broken_supply_retest': 'Broken Supply Retest',
        'broken_demand_retest': 'Broken Demand Retest',
    }
    base = labels.get(str(scenario), str(scenario).replace('_', ' ').title())
    opt = _option_contract_label(side or ('long' if 'hold' in str(scenario) or 'breakout' in str(scenario) else 'short'))
    return f'{base} — {opt}'


def _rr_tier(rr) -> str:
    try:
        x = float(rr)
    except Exception:
        return 'No target'
    if x >= 4.0:
        return 'Excellent 1:4.0+'
    if x >= 2.5:
        return 'Strong 1:2.5–1:3.99'
    if x >= 2.0:
        return 'Acceptable 1:2.0–1:2.49'
    if x >= 1.5:
        return 'Watch only 1:1.5–1:1.99'
    return 'Poor <1:1.5'


def _rr_tier_class(rr) -> str:
    try:
        x = float(rr)
    except Exception:
        return 'rr-none'
    if x >= 4.0:
        return 'rr-excellent'
    if x >= 2.5:
        return 'rr-strong'
    if x >= 2.0:
        return 'rr-acceptable'
    if x >= 1.5:
        return 'rr-watch'
    return 'rr-poor'


def _rr_range_summary(row: pd.Series) -> str:
    vals = []
    for i in range(1, 4):
        v = row.get(f'target_{i}_rr')
        try:
            if pd.notna(v):
                vals.append(float(v))
        except Exception:
            pass
    if not vals:
        v = row.get('estimated_rr')
        try:
            if pd.notna(v):
                vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return 'No valid target'
    return f'T1 {_rr_text(vals[0])} ({_rr_tier(vals[0])}); range 1:{min(vals):.2f}–1:{max(vals):.2f}'


def _target_ladder_html(row: pd.Series) -> str:
    items = []
    for i in range(1, 4):
        zone = row.get(f'target_{i}_zone')
        rr = row.get(f'target_{i}_rr')
        if not isinstance(zone, str) or not zone:
            continue
        freshness = row.get(f'target_{i}_freshness')
        tests = row.get(f'target_{i}_tests')
        role = row.get(f'target_{i}_role')
        cls = _rr_tier_class(rr)
        items.append(
            f"<div class='target-row {cls}'><strong>T{i}</strong> <span>{_html_escape(zone)}</span> "
            f"<b>{_rr_text(rr)}</b><em>{_html_escape(_rr_tier(rr))}</em>"
            f"<small>{_html_escape(freshness)} • {tests} tests • {_html_escape(role)}</small></div>"
        )
    return ''.join(items) if items else '<div class="target-row rr-none">No valid target ladder</div>'


def _rr_tier_legend_html() -> str:
    return """
    <div class="rr-legend">
      <span class="rr-pill rr-excellent">Excellent: 1:4+</span>
      <span class="rr-pill rr-strong">Strong: 1:2.5–1:3.99</span>
      <span class="rr-pill rr-acceptable">Acceptable: 1:2.0–1:2.49</span>
      <span class="rr-pill rr-watch">Watch only: 1:1.5–1:1.99</span>
      <span class="rr-pill rr-poor">Poor: &lt;1:1.5</span>
    </div>
    """


def _checklist_html(text: str) -> str:
    raw = str(text or '')
    if ':' in raw:
        prefix, rest = raw.split(':', 1)
        pieces = [p.strip() for p in rest.replace(';', ',').split(',') if p.strip()]
        heading = f"<strong>{_html_escape(prefix.strip())} confirmation checklist</strong>"
    else:
        pieces = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
        heading = '<strong>Confirmation checklist</strong>'
    if not pieces:
        return heading
    lis = ''.join(f"<li>□ {_html_escape(p)}</li>" for p in pieces)
    return f"{heading}<ul class='checklist'>{lis}</ul>"


def _scenario_readiness(row: pd.Series) -> str:
    """Classify scenario readiness separately from final eligibility.

    v0.29 keeps scenario prep broad. Final eligibility means it is already
    strong enough to be on the actionable final list. Developing means the
    zone/scenario is worth watching, but it still needs price movement,
    confirmation, or better R:R. Research keeps useful map context.
    """
    if bool(row.get('final_eligible', False)):
        return 'Final / Actionable'
    reasons = str(row.get('rejection_reasons', ''))
    if 'rejected_broken_entry_zone' in reasons or 'rejected_4plus_entry_tests' in reasons:
        return 'Zone Map / Research'
    distance = float(row.get('distance_pct', 99) or 99)
    grade_rank = _grade_rank_value(str(row.get('grade', 'C')))
    rr = pd.to_numeric(pd.Series([row.get('estimated_rr')]), errors='coerce').iloc[0]
    has_target = isinstance(row.get('target'), str)
    if distance <= float(WATCHLIST_DEVELOPING_MAX_DISTANCE_PCT) and grade_rank >= int(WATCHLIST_MIN_DEVELOPING_GRADE_RANK):
        if (has_target and pd.notna(rr)) or 'rejected_low_rr' in reasons or 'rejected_low_rr_to_nearest_opposing_zone' in reasons:
            return 'Developing Scenario'
    return 'Zone Map / Research'


def _scenario_status(row: pd.Series) -> str:
    """Human-readable scenario status used in the watchlist report."""
    distance = float(row.get('distance_pct', 99) or 99)
    reasons = str(row.get('rejection_reasons', ''))
    if bool(row.get('final_eligible', False)):
        return 'Ready: wait for 5M trigger'
    if 'rejected_low_rr_to_nearest_opposing_zone' in reasons:
        return 'Blocked: nearest zone limits R:R'
    if 'rejected_no_valid_target' in reasons or 'no_opposing_zone' in reasons:
        return 'Needs target ladder'
    if distance <= float(WATCHLIST_READY_DISTANCE_PCT):
        return 'At zone: needs confirmation'
    if distance <= float(WATCHLIST_NEEDS_CONFIRMATION_DISTANCE_PCT):
        return 'Near zone: developing'
    return 'Future scenario'


def _confirmation_needed(row: pd.Series) -> str:
    scenario = str(row.get('scenario', ''))
    if scenario == 'supply_break':
        return 'Calls: 5M close above supply; hold/retest above zone; volume expands on break or continuation; above 9EMA/VWAP; higher-high or higher-low structure'
    if scenario == 'demand_break':
        return 'Puts: 5M close below demand; hold/retest below zone; volume expands on break or continuation; below 9EMA/VWAP; lower-low or lower-high structure'
    if scenario == 'demand_hold':
        return 'Calls: demand test; selling volume fades or stabilizes into zone; reversal candle; reclaim/hold 9EMA and VWAP; bullish continuation/RBR; market context improving'
    if scenario == 'supply_reject':
        return 'Puts: supply test; rejection candle; lose/hold below 9EMA and VWAP; bearish continuation/DBD; downside volume response; market context weakening'
    return 'Wait for 5M price action, volume behavior, 9EMA/VWAP, structure, and R:R confirmation'


def _zones_overlap_or_near(a: pd.Series, b: pd.Series, tolerance_pct: float) -> bool:
    """Return True when two final watchlist cards represent essentially the same zone."""
    a_bottom, a_top = float(a["zone_bottom"]), float(a["zone_top"])
    b_bottom, b_top = float(b["zone_bottom"]), float(b["zone_top"])
    midpoint = max((a_bottom + a_top + b_bottom + b_top) / 4.0, 0.01)
    tolerance = midpoint * (tolerance_pct / 100.0)
    return b_bottom <= (a_top + tolerance) and a_bottom <= (b_top + tolerance)


def _dedupe_final_watchlist(final_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate final cards that survived raw-zone merging.

    Raw zone merging happens before scenario generation, but duplicates can still survive
    when several higher-timeframe zones have nearly identical levels or when targets/R:R
    are recomputed afterward. This pass works on the actual final watchlist rows and keeps
    the best card per symbol + bias + zone_type + overlapping price area.
    """
    if final_df.empty or not STRICT_DEDUPLICATE_FINAL_WATCHLIST:
        return final_df.copy()

    status_order = {"Immediate": 0, "Near-Term": 1, "Future": 2}
    rows_out = []

    for (_symbol, _bias, _zone_type, _scenario), group in final_df.groupby(["symbol", "bias", "zone_type", "scenario"], sort=False):
        g = group.copy()
        g["status_rank"] = g["status"].map(status_order).fillna(99)
        g["grade_rank"] = g["grade"].apply(_grade_rank_value)
        g["rr_num"] = pd.to_numeric(g["estimated_rr"], errors="coerce").fillna(-999)
        g["confluence_num"] = pd.to_numeric(g["confluence_count"] if "confluence_count" in g.columns else pd.Series(1, index=g.index), errors="coerce").fillna(1)
        g = g.sort_values(
            ["zone_bottom", "zone_top", "status_rank", "grade_rank", "rr_num", "confluence_num", "quality_score"],
            ascending=[True, True, True, False, False, False, False],
        )

        cluster = []
        for _, row in g.iterrows():
            if not cluster:
                cluster = [row]
                continue

            # Compare against the current cluster's total zone span.
            cluster_bottom = min(float(r["zone_bottom"]) for r in cluster)
            cluster_top = max(float(r["zone_top"]) for r in cluster)
            temp = row.copy()
            span = pd.Series({"zone_bottom": cluster_bottom, "zone_top": cluster_top})
            if _zones_overlap_or_near(span, temp, FINAL_DEDUPE_TOLERANCE_PCT):
                cluster.append(row)
            else:
                rows_out.append(_best_final_cluster_row(cluster))
                cluster = [row]

        if cluster:
            rows_out.append(_best_final_cluster_row(cluster))

    out = pd.DataFrame(rows_out)
    if out.empty:
        return out
    for c in ["status_rank", "grade_rank", "rr_num", "confluence_num"]:
        out = out.drop(columns=[c], errors="ignore")
    return out


def _best_final_cluster_row(cluster: list[pd.Series]) -> dict:
    """Pick the strongest row from a duplicate cluster and annotate confluence."""
    df = pd.DataFrame([r.to_dict() for r in cluster]).copy()
    status_order = {"Immediate": 0, "Near-Term": 1, "Future": 2}
    df["status_rank"] = df["status"].map(status_order).fillna(99)
    df["grade_rank"] = df["grade"].apply(_grade_rank_value)
    df["rr_num"] = pd.to_numeric(df["estimated_rr"], errors="coerce").fillna(-999)
    df["confluence_num"] = pd.to_numeric(df["confluence_count"] if "confluence_count" in df.columns else pd.Series(1, index=df.index), errors="coerce").fillna(1)
    df = df.sort_values(
        ["status_rank", "grade_rank", "rr_num", "confluence_num", "quality_score", "distance_pct"],
        ascending=[True, False, False, False, False, True],
    )
    best = df.iloc[0].to_dict()

    # Show all contributing timeframes without duplicating cards.
    tfs = []
    for val in df.get("confluence_timeframes", df["timeframe"]).astype(str).tolist() + df["timeframe"].astype(str).tolist():
        for piece in val.replace("/", ",").split(","):
            piece = piece.strip()
            if piece and piece not in tfs:
                tfs.append(piece)
    tfs = sorted(tfs, key=_timeframe_rank, reverse=True)

    patterns = []
    for val in df["pattern"].astype(str).tolist():
        for piece in val.replace("/", ",").split(","):
            piece = piece.strip()
            if piece and piece not in patterns:
                patterns.append(piece)

    best["timeframe"] = "/".join(tfs) if tfs else best.get("timeframe", "")
    best["pattern"] = "/".join(patterns) if patterns else best.get("pattern", "")
    best["confluence_count"] = int(max(float(best.get("confluence_count", 1)), len(df)))
    best["confluence_timeframes"] = ",".join(tfs)
    best["duplicate_rows_collapsed"] = int(len(df))
    if len(df) > 1:
        best["merged_from"] = "; ".join(
            f"{r['timeframe']} {r['zone_type']} {float(r['zone_bottom']):.2f}-{float(r['zone_top']):.2f} RR {float(r['estimated_rr']):.2f}"
            for _, r in df.iterrows()
        )
    return best

def _filter_final_report(watch_df: pd.DataFrame, min_grade: str = FINAL_REPORT_MIN_GRADE) -> pd.DataFrame:
    """Return the cleaner final report view while preserving all candidates separately.

    Final report rules:
    - Grade must be at least FINAL_REPORT_MIN_GRADE.
    - Estimated R:R must be at least MIN_FINAL_RR.

    Lower-grade or lower-R:R setups are still saved to watchlist_all_candidates.csv
    so they can be reviewed while tuning.
    """
    if watch_df.empty:
        return watch_df.copy()

    min_rank = _grade_rank_value(min_grade)
    rr_values = pd.to_numeric(watch_df['estimated_rr'], errors='coerce')
    if 'final_eligible' in watch_df.columns:
        out = watch_df[watch_df['final_eligible'].astype(bool)].copy()
    else:
        out = watch_df[
            (watch_df['grade'].apply(_grade_rank_value) >= min_rank)
            & (rr_values >= MIN_FINAL_RR)
        ].copy()

    if out.empty:
        return out

    status_order = {'Immediate': 0, 'Near-Term': 1, 'Future': 2}
    out['status_rank'] = out['status'].map(status_order)
    out['grade_rank'] = out['grade'].apply(lambda g: -_grade_rank_value(g))
    out = out.sort_values(
        ['status_rank', 'grade_rank', 'distance_pct', 'quality_score'],
        ascending=[True, True, True, False]
    )

    out = out.drop(columns=['status_rank', 'grade_rank'], errors='ignore')
    out = _dedupe_final_watchlist(out)

    if MAX_SETUPS_PER_SECTION and MAX_SETUPS_PER_SECTION > 0 and not out.empty:
        out['status_rank'] = out['status'].map(status_order)
        out['grade_rank'] = out['grade'].apply(lambda g: -_grade_rank_value(g))
        out = out.sort_values(
            ['status_rank', 'grade_rank', 'distance_pct', 'quality_score'],
            ascending=[True, True, True, False]
        ).groupby('status', group_keys=False).head(MAX_SETUPS_PER_SECTION)
        out = out.drop(columns=['status_rank', 'grade_rank'], errors='ignore')

    return out


def _freshness_label(value: str, tests) -> str:
    label = {
        'fresh': 'Fresh / untested',
        'one_test': 'One successful retest',
        'multiple_tests': 'Multiple retests',
    }.get(str(value), str(value).replace('_', ' ').title())
    try:
        return f'{label} ({int(tests)} tests)'
    except Exception:
        return label


def _rr_text(rr) -> str:
    if rr is None or pd.isna(rr):
        return 'n/a'
    return f'1:{float(rr):.2f}'


def _fmt_money(x) -> str:
    if x is None or pd.isna(x):
        return 'n/a'
    return f'${float(x):,.2f}'


def _fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return 'n/a'
    return f'{float(x):.2f}%'



# -----------------------------------------------------------------------------
# v0.36.6 watchlist visual zone map + price-structure bias
# -----------------------------------------------------------------------------

def _safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _find_structure_swings(bars: pd.DataFrame, left: int = 2, right: int = 2, max_bars: int = 180) -> tuple[list[dict], list[dict]]:
    """Return simple pivot swing highs/lows from completed RTH bars.

    This is intentionally descriptive context for the watchlist, not a hard entry
    filter. It only looks backward from the current/report snapshot.
    """
    if bars is None or bars.empty or len(bars) < (left + right + 8):
        return [], []
    b = bars.tail(max_bars).copy()
    highs = []
    lows = []
    for i in range(left, len(b) - right):
        window = b.iloc[i-left:i+right+1]
        row = b.iloc[i]
        ts = b.index[i]
        hi = float(row['high'])
        lo = float(row['low'])
        # Require the pivot to be at least tied for the local extreme. Duplicate
        # equal highs/lows are okay; the last two confirmed swings carry the read.
        if hi >= float(window['high'].max()):
            highs.append({'time': ts, 'price': hi})
        if lo <= float(window['low'].min()):
            lows.append({'time': ts, 'price': lo})
    return highs, lows


def _structure_label_from_swings(highs: list[dict], lows: list[dict], reference_price: float | None = None) -> dict:
    """Classify recent structure as HH/HL, LH/LL, mixed, range, or insufficient."""
    if len(highs) < 2 or len(lows) < 2:
        return {
            'bias': 'Insufficient structure',
            'detail': 'Need at least two confirmed swing highs and lows',
            'last_high': None, 'prior_high': None, 'last_low': None, 'prior_low': None,
        }
    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]
    prior_high, last_high = float(h1['price']), float(h2['price'])
    prior_low, last_low = float(l1['price']), float(l2['price'])

    ref = max(abs(reference_price or last_high or 1.0), 0.01)
    flat_tol = ref * 0.0015  # 0.15%; avoids over-labeling tiny wiggles as trend.
    high_delta = last_high - prior_high
    low_delta = last_low - prior_low

    higher_high = high_delta > flat_tol
    lower_high = high_delta < -flat_tol
    higher_low = low_delta > flat_tol
    lower_low = low_delta < -flat_tol
    flat_high = abs(high_delta) <= flat_tol
    flat_low = abs(low_delta) <= flat_tol

    if higher_high and higher_low:
        bias = 'Bullish HH/HL'
    elif lower_high and lower_low:
        bias = 'Bearish LH/LL'
    elif flat_high and flat_low:
        bias = 'Range-bound'
    else:
        bias = 'Mixed / Transition'

    high_word = 'HH' if higher_high else ('LH' if lower_high else 'flat high')
    low_word = 'HL' if higher_low else ('LL' if lower_low else 'flat low')
    detail = f"{high_word}: {prior_high:.2f} → {last_high:.2f}; {low_word}: {prior_low:.2f} → {last_low:.2f}"
    return {
        'bias': bias,
        'detail': detail,
        'last_high': round(last_high, 2),
        'prior_high': round(prior_high, 2),
        'last_low': round(last_low, 2),
        'prior_low': round(prior_low, 2),
    }


def _structure_context_for_bars(bars: pd.DataFrame, reference_price: float | None = None, max_bars: int = 180) -> dict:
    highs, lows = _find_structure_swings(bars, max_bars=max_bars)
    return _structure_label_from_swings(highs, lows, reference_price)


def _compute_symbol_structure_context(raw_5m: pd.DataFrame, daily_raw: pd.DataFrame | None, reference_price: float | None = None) -> dict:
    """Compute 5M, 15M, and daily HH/HL vs LH/LL context from RTH-only data."""
    out = {}
    try:
        rth_5m = regular_session_only(raw_5m)
        out['5m'] = _structure_context_for_bars(rth_5m, reference_price, max_bars=156)
        bars_15m = aggregate_bars(raw_5m, '15min')
        out['15m'] = _structure_context_for_bars(bars_15m, reference_price, max_bars=120)
    except Exception:
        out['5m'] = {'bias': 'Insufficient structure', 'detail': '5M structure unavailable'}
        out['15m'] = {'bias': 'Insufficient structure', 'detail': '15M structure unavailable'}

    try:
        if daily_raw is not None and not daily_raw.empty:
            out['1d'] = _structure_context_for_bars(daily_raw, reference_price, max_bars=90)
        else:
            daily_from_5m = aggregate_bars(raw_5m, '1D')
            out['1d'] = _structure_context_for_bars(daily_from_5m, reference_price, max_bars=90)
    except Exception:
        out['1d'] = {'bias': 'Insufficient structure', 'detail': 'Daily structure unavailable'}

    labels = [out.get('5m', {}).get('bias', ''), out.get('15m', {}).get('bias', ''), out.get('1d', {}).get('bias', '')]
    bullish = sum('Bullish' in x for x in labels)
    bearish = sum('Bearish' in x for x in labels)
    if bullish >= 2 and bearish == 0:
        alignment = 'Aligned bullish'
    elif bearish >= 2 and bullish == 0:
        alignment = 'Aligned bearish'
    elif 'Bullish' in labels[0] and 'Bearish' in labels[1]:
        alignment = 'Short-term bullish / 15M bearish'
    elif 'Bearish' in labels[0] and 'Bullish' in labels[1]:
        alignment = 'Short-term bearish / 15M bullish'
    elif bullish or bearish:
        alignment = 'Mixed structure'
    else:
        alignment = 'Range-bound / insufficient'
    out['alignment'] = alignment
    return out


def _closest_zone_context(zones_df: pd.DataFrame, symbol: str, price: float) -> dict:
    """Find closest active demand/supply context for a current-price zone map."""
    base = {
        'closest_demand_bottom': None, 'closest_demand_top': None, 'closest_demand_timeframe': '', 'closest_demand_relation': '', 'distance_to_demand_pct': None,
        'closest_supply_bottom': None, 'closest_supply_top': None, 'closest_supply_timeframe': '', 'closest_supply_relation': '', 'distance_to_supply_pct': None,
        'price_position_status': 'no_zone_context',
    }
    if zones_df is None or zones_df.empty or pd.isna(price):
        return base
    z = zones_df[zones_df['symbol'].astype(str).str.upper().eq(str(symbol).upper())].copy()
    if z.empty:
        return base

    def pick_zone(kind: str):
        sub = z[z['zone_type'].astype(str).str.lower().eq(kind)].copy()
        if sub.empty:
            return None, '', None
        sub['bottom_f'] = pd.to_numeric(sub['zone_bottom'], errors='coerce')
        sub['top_f'] = pd.to_numeric(sub['zone_top'], errors='coerce')
        sub = sub.dropna(subset=['bottom_f', 'top_f'])
        if sub.empty:
            return None, '', None
        containing = sub[(sub['bottom_f'] <= price) & (sub['top_f'] >= price)].copy()
        if not containing.empty:
            containing['dist'] = 0.0
            row = containing.sort_values(['top_f', 'bottom_f']).iloc[0]
            return row, 'containing_price', 0.0
        if kind == 'demand':
            below = sub[sub['top_f'] < price].copy()
            if not below.empty:
                below['dist'] = price - below['top_f']
                row = below.sort_values('dist').iloc[0]
                return row, 'below_price', float(row['dist'] / max(price, 0.01) * 100)
            sub['dist'] = (sub['bottom_f'] - price).abs()
            row = sub.sort_values('dist').iloc[0]
            return row, 'above_price', float(row['dist'] / max(price, 0.01) * 100)
        above = sub[sub['bottom_f'] > price].copy()
        if not above.empty:
            above['dist'] = above['bottom_f'] - price
            row = above.sort_values('dist').iloc[0]
            return row, 'above_price', float(row['dist'] / max(price, 0.01) * 100)
        sub['dist'] = (price - sub['top_f']).abs()
        row = sub.sort_values('dist').iloc[0]
        return row, 'below_price', float(row['dist'] / max(price, 0.01) * 100)

    d, d_rel, d_pct = pick_zone('demand')
    s, s_rel, s_pct = pick_zone('supply')
    if d is not None:
        base.update({
            'closest_demand_bottom': round(float(d['bottom_f']), 2),
            'closest_demand_top': round(float(d['top_f']), 2),
            'closest_demand_timeframe': str(d.get('timeframe', '')),
            'closest_demand_relation': d_rel,
            'distance_to_demand_pct': round(float(d_pct), 2) if d_pct is not None else None,
        })
    if s is not None:
        base.update({
            'closest_supply_bottom': round(float(s['bottom_f']), 2),
            'closest_supply_top': round(float(s['top_f']), 2),
            'closest_supply_timeframe': str(s.get('timeframe', '')),
            'closest_supply_relation': s_rel,
            'distance_to_supply_pct': round(float(s_pct), 2) if s_pct is not None else None,
        })

    if d_rel == 'containing_price':
        status = 'inside_demand'
    elif s_rel == 'containing_price':
        status = 'inside_supply'
    elif d is not None and s is not None and d_rel == 'below_price' and s_rel == 'above_price':
        status = 'between_nearest_demand_and_supply'
    elif s is not None and s_rel == 'below_price':
        status = 'above_nearest_supply'
    elif d is not None and d_rel == 'above_price':
        status = 'below_nearest_demand'
    else:
        status = 'zone_context_mixed'
    base['price_position_status'] = status
    return base


def _trade_structure_alignment(row: pd.Series) -> str:
    bias = str(row.get('bias', ''))
    s5 = str(row.get('structure_bias_5m', ''))
    s15 = str(row.get('structure_bias_15m', ''))
    bullish_votes = sum('Bullish' in x for x in [s5, s15])
    bearish_votes = sum('Bearish' in x for x in [s5, s15])
    if bullish_votes > bearish_votes:
        structure_side = 'Bullish'
    elif bearish_votes > bullish_votes:
        structure_side = 'Bearish'
    else:
        structure_side = 'Mixed'
    if structure_side == 'Mixed':
        return 'Mixed / no clear structure edge'
    return 'With structure' if structure_side == bias else 'Counter-structure'


def _add_watchlist_visual_context(watch_df: pd.DataFrame, zones_df: pd.DataFrame, structure_context: dict, latest_prices: dict | None = None, price_as_of: dict | None = None) -> pd.DataFrame:
    """Add zone-map and HH/HL/LH/LL context without affecting final eligibility."""
    if watch_df is None or watch_df.empty:
        return watch_df
    out = watch_df.copy()
    context_rows = []
    for _, row in out.iterrows():
        sym = str(row.get('symbol', '')).upper()
        price = _safe_float(row.get('current_price'), (latest_prices or {}).get(sym))
        zone_ctx = _closest_zone_context(zones_df, sym, price)
        sc = structure_context.get(sym, {}) if isinstance(structure_context, dict) else {}
        merged = dict(zone_ctx)
        existing_as_of = str(row.get('current_price_as_of', '')).strip() if row.get('current_price_as_of') is not None else ''
        merged['current_price_as_of'] = existing_as_of or (price_as_of or {}).get(sym, '')
        for tf_key, prefix in [('5m', '5m'), ('15m', '15m'), ('1d', '1d')]:
            tf_ctx = sc.get(tf_key, {}) if isinstance(sc, dict) else {}
            merged[f'structure_bias_{prefix}'] = tf_ctx.get('bias', 'Insufficient structure')
            merged[f'structure_detail_{prefix}'] = tf_ctx.get('detail', '')
            merged[f'last_swing_high_{prefix}'] = tf_ctx.get('last_high')
            merged[f'prior_swing_high_{prefix}'] = tf_ctx.get('prior_high')
            merged[f'last_swing_low_{prefix}'] = tf_ctx.get('last_low')
            merged[f'prior_swing_low_{prefix}'] = tf_ctx.get('prior_low')
        merged['structure_alignment'] = sc.get('alignment', 'Range-bound / insufficient') if isinstance(sc, dict) else 'Range-bound / insufficient'
        context_rows.append(merged)
    ctx_df = pd.DataFrame(context_rows, index=out.index)
    # Re-running the visual-zone-map add-on should be idempotent. Drop prior
    # visual/context columns before concatenating the refreshed context to avoid
    # duplicate column names that can make pandas return a DataFrame for out.get().
    out = out.drop(columns=[c for c in ctx_df.columns if c in out.columns], errors='ignore')
    out = pd.concat([out, ctx_df], axis=1)
    out['price_structure_bias'] = out.get('structure_alignment', 'Range-bound / insufficient')
    out['structure_trade_alignment'] = out.apply(_trade_structure_alignment, axis=1)
    return out


def _chart_bars_for_symbol(symbol: str, max_bars: int = 78) -> pd.DataFrame:
    path = DATA_DIR / f'{str(symbol).upper()}_5M.csv'
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = load_symbol_csv(path)
        rth = regular_session_only(raw)
        return rth.tail(max_bars).copy()
    except Exception:
        return pd.DataFrame()


def _zone_chart_svg(row: pd.Series, width: int = 720, height: int = 300, max_bars: int = 78) -> str:
    """Inline SVG mini chart with candles, nearest zones, current price, 9EMA, VWAP."""
    symbol = str(row.get('symbol', '')).upper()
    bars = _chart_bars_for_symbol(symbol, max_bars=max_bars)
    price = _safe_float(row.get('current_price'))
    if bars.empty or price is None:
        return "<div class='zone-chart-empty'>No 5M chart data available for this symbol.</div>"

    demand_bottom = _safe_float(row.get('closest_demand_bottom'))
    demand_top = _safe_float(row.get('closest_demand_top'))
    supply_bottom = _safe_float(row.get('closest_supply_bottom'))
    supply_top = _safe_float(row.get('closest_supply_top'))
    y_values = list(pd.to_numeric(bars['high'], errors='coerce').dropna()) + list(pd.to_numeric(bars['low'], errors='coerce').dropna()) + [price]
    for v in [demand_bottom, demand_top, supply_bottom, supply_top]:
        if v is not None:
            y_values.append(v)
    low = min(y_values)
    high = max(y_values)
    pad = max((high - low) * 0.08, price * 0.002, 0.01)
    y_min = low - pad
    y_max = high + pad

    left, right, top_m, bottom_m = 48, 96, 18, 34
    plot_w = width - left - right
    plot_h = height - top_m - bottom_m
    def x_at(i):
        if len(bars) <= 1:
            return left + plot_w / 2
        return left + (i / (len(bars) - 1)) * plot_w
    def y_at(v):
        return top_m + (y_max - float(v)) / max((y_max - y_min), 1e-9) * plot_h

    parts = [f"<svg viewBox='0 0 {width} {height}' class='zone-svg' role='img' aria-label='{_html_escape(symbol)} price chart with nearest supply and demand zones'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' rx='14' class='chart-bg'/>")
    # light horizontal grid
    for j in range(5):
        y = top_m + j * plot_h / 4
        val = y_max - j * (y_max - y_min) / 4
        parts.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' class='chart-grid'/>")
        parts.append(f"<text x='{width-right+8}' y='{y+4:.1f}' class='chart-axis'>{val:.2f}</text>")

    def band(y1, y2, cls, label):
        yy1, yy2 = y_at(y1), y_at(y2)
        y = min(yy1, yy2)
        h = max(2, abs(yy2 - yy1))
        parts.append(f"<rect x='{left}' y='{y:.1f}' width='{plot_w}' height='{h:.1f}' class='{cls}'/>")
        parts.append(f"<text x='{left+8}' y='{y+14:.1f}' class='zone-label'>{_html_escape(label)}</text>")

    if demand_bottom is not None and demand_top is not None:
        band(demand_bottom, demand_top, 'demand-band', f"Demand {demand_bottom:.2f}–{demand_top:.2f}")
    if supply_bottom is not None and supply_top is not None:
        band(supply_bottom, supply_top, 'supply-band', f"Supply {supply_bottom:.2f}–{supply_top:.2f}")

    # Candles
    candle_w = max(2.0, min(7.0, plot_w / max(len(bars), 1) * 0.55))
    for i, (_, b) in enumerate(bars.iterrows()):
        o, h, l, c = map(float, [b['open'], b['high'], b['low'], b['close']])
        x = x_at(i)
        cls = 'up-candle' if c >= o else 'down-candle'
        parts.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{y_at(h):.1f}' y2='{y_at(l):.1f}' class='{cls} wick'/>")
        y_body = min(y_at(o), y_at(c))
        h_body = max(1.5, abs(y_at(c) - y_at(o)))
        parts.append(f"<rect x='{x-candle_w/2:.1f}' y='{y_body:.1f}' width='{candle_w:.1f}' height='{h_body:.1f}' class='{cls}'/>")

    # 9EMA and session VWAP overlays.
    try:
        ema = bars['close'].ewm(span=9, adjust=False).mean()
        pts = ' '.join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(ema))
        parts.append(f"<polyline points='{pts}' class='ema-line'/>")
    except Exception:
        pass
    try:
        if 'vwap' in bars.columns:
            vwap_series = bars['vwap']
        else:
            vol = bars['volume'].replace(0, pd.NA)
            typical = (bars['high'] + bars['low'] + bars['close']) / 3.0
            vwap_series = (typical * vol).cumsum() / vol.cumsum()
        pts = ' '.join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(vwap_series.dropna()))
        if pts:
            parts.append(f"<polyline points='{pts}' class='vwap-line'/>")
    except Exception:
        pass

    py = y_at(price)
    parts.append(f"<line x1='{left}' x2='{width-right}' y1='{py:.1f}' y2='{py:.1f}' class='price-line'/>")
    parts.append(f"<text x='{width-right+8}' y='{py-5:.1f}' class='price-label'>Current {price:.2f}</text>")
    # x-axis labels
    try:
        first_ts = bars.index[0].strftime('%m/%d %H:%M')
        last_ts = bars.index[-1].strftime('%m/%d %H:%M')
        parts.append(f"<text x='{left}' y='{height-10}' class='chart-axis'>{_html_escape(first_ts)}</text>")
        parts.append(f"<text x='{width-right-70}' y='{height-10}' class='chart-axis'>{_html_escape(last_ts)}</text>")
    except Exception:
        pass
    parts.append("<g class='legend'><text x='52' y='16'>9EMA</text><text x='102' y='16'>VWAP</text></g>")
    parts.append("</svg>")
    return ''.join(parts)


def generate_watchlist_zone_map_html(watch_df: pd.DataFrame, meta: dict) -> str:
    report_date = _html_escape(meta.get('report_date', ''))
    report_dt = _html_escape(meta.get('report_datetime', ''))
    if watch_df is None or watch_df.empty:
        cards = "<p>No Final / Actionable candidates available for chart-style zone maps.</p>"
    else:
        card_parts = []
        for _, r in watch_df.iterrows():
            bias_class = 'bullish' if str(r.get('bias')) == 'Bullish' else 'bearish'
            svg = _zone_chart_svg(r, width=760, height=320, max_bars=78)
            card_parts.append(f"""
            <article class='zone-map-card {bias_class}'>
              <div class='zone-map-head'>
                <div><h2>{_html_escape(r.get('symbol',''))} — {_html_escape(r.get('scenario_label', r.get('setup','')))}</h2>
                <div class='muted'>{_html_escape(r.get('option_contract',''))} • Grade {_html_escape(r.get('setup_quality_grade', r.get('grade','')))} • {_html_escape(r.get('status',''))}</div></div>
                <div class='price-box'>Current<br><strong>{_fmt_money(r.get('current_price'))}</strong><small>as of {_html_escape(r.get('current_price_as_of',''))}</small></div>
              </div>
              {svg}
              <div class='zone-map-grid'>
                <div><span>Position</span><strong>{_html_escape(str(r.get('price_position_status','')).replace('_',' '))}</strong></div>
                <div><span>Demand</span><strong>{_fmt_money(r.get('closest_demand_bottom'))}–{_fmt_money(r.get('closest_demand_top'))}</strong><small>{_html_escape(r.get('closest_demand_timeframe',''))} • {_fmt_pct(r.get('distance_to_demand_pct'))}</small></div>
                <div><span>Supply</span><strong>{_fmt_money(r.get('closest_supply_bottom'))}–{_fmt_money(r.get('closest_supply_top'))}</strong><small>{_html_escape(r.get('closest_supply_timeframe',''))} • {_fmt_pct(r.get('distance_to_supply_pct'))}</small></div>
                <div><span>Structure</span><strong>{_html_escape(r.get('structure_alignment',''))}</strong><small>5M: {_html_escape(r.get('structure_bias_5m',''))}; 15M: {_html_escape(r.get('structure_bias_15m',''))}</small></div>
                <div><span>Trade vs structure</span><strong>{_html_escape(r.get('structure_trade_alignment',''))}</strong><small>{_html_escape(r.get('structure_detail_5m',''))}</small></div>
                <div><span>T1 R:R</span><strong>{_rr_text(r.get('estimated_rr'))}</strong><small>{_html_escape(r.get('rr_range_summary',''))}</small></div>
              </div>
            </article>
            """)
        cards = ''.join(card_parts)
    style = """
    <style>
      :root{--bg:#0f172a;--panel:#111827;--panel2:#1f2937;--line:#374151;--text:#e5e7eb;--muted:#9ca3af;--bull:#16a34a;--bear:#dc2626;--blue:#38bdf8;--gold:#f59e0b}
      body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,Helvetica,sans-serif}.wrap{max-width:1160px;margin:0 auto;padding:28px}h1{margin:0 0 6px}h2{margin:0}.muted{color:var(--muted);font-size:13px}.zone-map-card{background:var(--panel);border:1px solid var(--line);border-left:6px solid var(--blue);border-radius:18px;padding:18px;margin:18px 0;box-shadow:0 8px 22px rgba(0,0,0,.22)}.zone-map-card.bullish{border-left-color:var(--bull)}.zone-map-card.bearish{border-left-color:var(--bear)}.zone-map-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:14px}.price-box{text-align:right;background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:10px 14px;color:var(--muted)}.price-box strong{color:var(--text);font-size:20px}.price-box small{display:block;color:var(--muted);font-size:11px;margin-top:3px;max-width:210px}.zone-map-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin-top:12px}.zone-map-grid div{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:12px;padding:10px}.zone-map-grid span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}.zone-map-grid strong{display:block}.zone-map-grid small{display:block;color:var(--muted);margin-top:3px}.zone-svg{width:100%;height:auto;display:block}.chart-bg{fill:#0b1224}.chart-grid{stroke:#263244;stroke-width:1}.chart-axis,.zone-label{fill:#9ca3af;font-size:11px}.zone-label{font-weight:700}.demand-band{fill:rgba(22,163,74,.20);stroke:rgba(22,163,74,.55);stroke-width:1}.supply-band{fill:rgba(220,38,38,.19);stroke:rgba(220,38,38,.55);stroke-width:1}.up-candle{fill:#22c55e;stroke:#22c55e}.down-candle{fill:#ef4444;stroke:#ef4444}.wick{stroke-width:1.2}.ema-line{fill:none;stroke:#f59e0b;stroke-width:1.5}.vwap-line{fill:none;stroke:#38bdf8;stroke-width:1.5;stroke-dasharray:4 3}.price-line{stroke:#93c5fd;stroke-width:1.4;stroke-dasharray:5 4}.price-label{fill:#bfdbfe;font-size:11px;font-weight:700}.legend text{fill:#cbd5e1;font-size:11px}.legend text:first-child{fill:#fbbf24}.zone-chart-empty{color:var(--muted);border:1px dashed var(--line);border-radius:12px;padding:16px}
    </style>
    """
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Watchlist Visual Zone Map {report_date}</title>{style}</head><body><main class='wrap'><h1>Watchlist Visual Zone Map — {report_date}</h1><p class='muted'>Generated: {report_dt} America/New_York. Shows Final / Actionable candidates as chart-style panels using recent regular-session 5M candles, nearest active supply/demand zones, current price with per-symbol as-of time, 9EMA/VWAP overlays, and HH/HL vs LH/LL structure labels. This is visual context only; it does not change watchlist inclusion logic.</p>{cards}</main></body></html>"""


def build_watchlist_from_zone_snapshot(zones_df: pd.DataFrame, latest_prices: dict, meta: dict | None = None) -> pd.DataFrame:
    """Build scenario watchlist rows from an already-computed historical zone snapshot.

    This is used by the daily snapshot backtester so it does not need to rerun
    the full watchlist scanner for every historical day. The caller is
    responsible for making sure `zones_df` reflects only information known as of
    the snapshot date. Broken zones are still filtered here before merging.
    """
    meta = meta or {}
    latest_price_as_of = meta.get('current_price_as_of') or {}
    if not isinstance(latest_price_as_of, dict):
        latest_price_as_of = {}
    default_price_as_of = str(meta.get('snapshot_as_of') or meta.get('as_of_date') or '')

    if zones_df is None or zones_df.empty:
        return pd.DataFrame()

    active_zones_df = _active_zones_for_watchlist(zones_df)
    if active_zones_df.empty:
        return pd.DataFrame()
    watch_zones_df = merge_overlapping_zones(active_zones_df)

    candidates = []
    for _, z in watch_zones_df.iterrows():
        sym = str(z['symbol']).upper()
        if sym not in latest_prices:
            continue
        price = float(latest_prices[sym])
        top = float(z['zone_top'])
        bottom = float(z['zone_bottom'])
        in_zone = _price_inside_zone(price, top, bottom)

        scenario_defs = []
        if z['zone_type'] == 'demand':
            if price is not None:
                scenario_defs.append({
                    'scenario': 'demand_hold',
                    'setup': 'Demand Zone Test / Hold',
                    'bias': 'Bullish',
                    'side': 'long',
                    'trigger': bottom,
                    'confirmation': top,
                    'invalidation': bottom,
                    'distance': _distance_to_zone(price, z['zone_type'], top, bottom),
                })
            if price is not None:
                scenario_defs.append({
                    'scenario': 'demand_break',
                    'setup': 'Demand Zone Breakdown',
                    'bias': 'Bearish',
                    'side': 'short',
                    'trigger': bottom,
                    'confirmation': bottom,
                    'invalidation': top,
                    'distance': _distance_to_trigger_pct(price, bottom, in_zone=in_zone),
                })
        else:
            if price is not None:
                scenario_defs.append({
                    'scenario': 'supply_reject',
                    'setup': 'Supply Zone Rejection',
                    'bias': 'Bearish',
                    'side': 'short',
                    'trigger': top,
                    'confirmation': bottom,
                    'invalidation': top,
                    'distance': _distance_to_zone(price, z['zone_type'], top, bottom),
                })
            if price is not None:
                scenario_defs.append({
                    'scenario': 'supply_break',
                    'setup': 'Supply Zone Breakout',
                    'bias': 'Bullish',
                    'side': 'long',
                    'trigger': top,
                    'confirmation': top,
                    'invalidation': bottom,
                    'distance': _distance_to_trigger_pct(price, top, in_zone=in_zone),
                })

        for sc in scenario_defs:
            status = _status(float(sc['distance']))
            if float(sc['distance']) > float(WATCHLIST_SCENARIO_MAX_DISTANCE_PCT):
                continue

            target_info = _build_target_ladder(
                watch_zones_df, z['symbol'], z['zone_type'], price, z, side=sc['side']
            )

            candidate = {
                'status': status,
                'symbol': z['symbol'],
                'current_price': round(price, 2),
                'current_price_as_of': latest_price_as_of.get(z['symbol'], default_price_as_of),
                'setup': sc['setup'],
                'scenario': sc['scenario'],
                'scenario_label': _scenario_display_label(sc['scenario'], sc['side']),
                'option_contract': _option_contract_label(sc['side']),
                'bias': sc['bias'],
                'side': sc['side'],
                'timeframe': z['timeframe'],
                'pattern': z['pattern'],
                'zone_type': z['zone_type'],
                'zone_top': z['zone_top'],
                'zone_bottom': z['zone_bottom'],
                'trigger_level': round(float(sc['trigger']), 2),
                'confirmation_level': round(float(sc['confirmation']), 2),
                'invalidation_level': round(float(sc['invalidation']), 2),
                'distance_pct': round(float(sc['distance']), 2),
                'freshness': z['freshness'],
                'freshness_label': _freshness_label(z['freshness'], z['tests']),
                'tests': z['tests'],
                'departure_atr': z['departure_atr'],
                'departure_grade': _departure_grade(z['departure_atr']),
                'departure_volume_ratio': z['departure_volume_ratio'],
                'departure_body_vs_base_body': z.get('departure_body_vs_base_body', z.get('departure_move_vs_base_range', None)),
                'volume_grade': _volume_grade(z['departure_volume_ratio']),
                'quality_score': z['quality_score'],
                'target': target_info.get('target'),
                'estimated_rr': target_info.get('estimated_rr'),
                'target_rr_raw': target_info.get('target_rr_raw'),
                'target_quality': target_info.get('target_quality'),
                'target_level': target_info.get('target_level'),
                'target_reject_reason': target_info.get('target_reject_reason'),
                'target_rr_range': target_info.get('target_rr_range'),
                'target_ladder': target_info.get('target_ladder'),
                'target_1_zone': target_info.get('target_1_zone'),
                'target_1_rr': target_info.get('target_1_rr'),
                'target_1_freshness': target_info.get('target_1_freshness'),
                'target_1_tests': target_info.get('target_1_tests'),
                'target_1_role': target_info.get('target_1_role'),
                'target_2_zone': target_info.get('target_2_zone'),
                'target_2_rr': target_info.get('target_2_rr'),
                'target_2_freshness': target_info.get('target_2_freshness'),
                'target_2_tests': target_info.get('target_2_tests'),
                'target_2_role': target_info.get('target_2_role'),
                'target_3_zone': target_info.get('target_3_zone'),
                'target_3_rr': target_info.get('target_3_rr'),
                'target_3_freshness': target_info.get('target_3_freshness'),
                'target_3_tests': target_info.get('target_3_tests'),
                'target_3_role': target_info.get('target_3_role'),
                'intervening_zone_count': target_info.get('intervening_zone_count'),
                'nearest_obstacle_zone': target_info.get('nearest_obstacle_zone'),
                'target_selection_reason': target_info.get('target_selection_reason'),
                'grade': 'PENDING',
                'confluence_count': int(z.get('confluence_count', 1)),
                'confluence_timeframes': z.get('confluence_timeframes', z['timeframe']),
                'merged_from': z.get('merged_from', ''),
                'primary_timeframe': z.get('primary_timeframe', z['timeframe']),
            }
            candidate.update(_zone_metadata_payload(z))
            candidates.append(candidate)

    watch_df = pd.DataFrame(candidates)
    watch_df = _add_watchlist_visual_context(watch_df, watch_zones_df, {}, latest_prices, latest_price_as_of)
    symbol_movement_context = meta.get('symbol_movement_context') or {}
    if isinstance(symbol_movement_context, dict) and symbol_movement_context:
        watch_df = enrich_movement_context(watch_df, symbol_movement_context, load_zone_reaction_history(REPORT_DIR))
        watch_df['snapshot_context_time'] = meta.get('snapshot_context_time', '')
        watch_df['snapshot_context_type'] = meta.get('snapshot_context_type', 'historical')
    watch_df = _apply_hard_exclusions(watch_df)
    if watch_df.empty:
        return watch_df

    reasons_col = []
    for _, r in watch_df.iterrows():
        reasons = _entry_zone_rejection_reasons(r)
        if not isinstance(r.get('target'), str):
            reasons.append(r.get('target_reject_reason') or 'rejected_no_valid_target')
        elif pd.isna(r.get('target_quality')) or float(r.get('target_quality') or 0) < float(TARGET_ZONE_MIN_QUALITY_SCORE):
            reasons.append('rejected_low_target_quality')
        if pd.isna(r.get('estimated_rr')) or float(r.get('estimated_rr') or 0) < float(MIN_FINAL_RR):
            reasons.append('rejected_low_rr_to_nearest_opposing_zone')
        reasons_col.append(';'.join(dict.fromkeys([str(x) for x in reasons if x])) or 'eligible_pre_grade')
    watch_df['rejection_reasons'] = reasons_col
    watch_df['setup_quality_score'] = watch_df.apply(_setup_quality_score, axis=1)
    watch_df['setup_quality_grade'] = watch_df['setup_quality_score'].apply(_setup_grade)
    watch_df['grade'] = watch_df['setup_quality_grade']
    watch_df['rr_tier'] = watch_df['estimated_rr'].apply(_rr_tier)
    watch_df['rr_tier_class'] = watch_df['estimated_rr'].apply(_rr_tier_class)
    watch_df['rr_range_summary'] = watch_df.apply(_rr_range_summary, axis=1)
    min_rank = _grade_rank_value(FINAL_REPORT_MIN_GRADE)
    watch_df['final_eligible'] = (
        (watch_df['rejection_reasons'] == 'eligible_pre_grade')
        & (watch_df['grade'].apply(_grade_rank_value) >= min_rank)
        & (pd.to_numeric(watch_df['estimated_rr'], errors='coerce') >= float(MIN_FINAL_RR))
    )
    watch_df.loc[(watch_df['rejection_reasons'] == 'eligible_pre_grade') & (~watch_df['final_eligible']), 'rejection_reasons'] = 'rejected_grade_below_final_min'
    watch_df['watchlist_bucket'] = watch_df.apply(_scenario_readiness, axis=1)
    watch_df['scenario_status'] = watch_df.apply(_scenario_status, axis=1)
    watch_df['confirmation_needed'] = watch_df.apply(_confirmation_needed, axis=1)

    status_order = {'Immediate': 0, 'Near-Term': 1, 'Future': 2}
    watch_df['status_rank'] = watch_df['status'].map(status_order)
    watch_df['grade_rank'] = watch_df['grade'].apply(lambda g: -_grade_rank_value(g))
    watch_df = watch_df.sort_values(
        ['status_rank', 'grade_rank', 'distance_pct', 'quality_score'],
        ascending=[True, True, True, False]
    ).drop(columns=['status_rank', 'grade_rank'])
    return watch_df

def build_watchlist(as_of_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    all_zones = []
    latest_prices = {}
    latest_price_as_of = {}
    latest_bar_times = {}
    structure_context = {}
    symbol_movement_context = {}
    price_overrides, price_source = load_current_prices()
    price_override_as_of = load_current_price_as_of_overrides()

    as_of_cutoff = None
    if as_of_date:
        # Mentor-style comparison mode: use only regular-session data through
        # the close of this New York market date.
        as_of_cutoff = pd.Timestamp(f"{as_of_date} 16:00", tz=MARKET_TZ).tz_convert("UTC")
        price_source = f"regular-session close as of {as_of_date}"

    skipped_symbols = []

    for symbol in WATCHLIST:
        path = None
        source_suffix = None

        if REQUIRE_5M_SOURCE_FOR_WATCHLIST:
            candidate = DATA_DIR / f'{symbol}_5M.csv'
            if candidate.exists():
                path = candidate
                source_suffix = '5M'
            else:
                skipped_symbols.append(f'{symbol}: missing 5M source')
                continue
        else:
            for suffix in SOURCE_SUFFIX_PRIORITY:
                candidate = DATA_DIR / f'{symbol}_{suffix}.csv'
                if candidate.exists():
                    path = candidate
                    source_suffix = suffix
                    break
            if path is None:
                skipped_symbols.append(f'{symbol}: no source file')
                continue

        raw = load_symbol_csv(path)
        if as_of_cutoff is not None:
            raw = raw.loc[raw.index <= as_of_cutoff].copy()
        rth = regular_session_only(raw)
        if raw.empty or rth.empty:
            skipped_symbols.append(f'{symbol}: no regular-session data before cutoff')
            continue
        bar_as_of = _format_price_timestamp(rth.index[-1])
        latest_bar_times[symbol] = f'{bar_as_of} ({source_suffix} regular-session source)'
        # Zones remain RTH-only, but watchlist proximity must use the most current
        # market price available. Use the newest downloaded bar by default. A manual
        # quote/current-price file may override only if its timestamp is at least as
        # recent as the downloaded bar; this prevents stale premarket snapshots from
        # overriding a later RTH/aftermarket close.
        current_asof_for_context = rth.index[-1]
        raw_local = raw.tz_convert(MARKET_TZ) if as_of_cutoff is None and not raw.empty else None
        raw_latest_ts = raw_local.index[-1] if raw_local is not None and not raw_local.empty else rth.index[-1]
        raw_latest_price = float(raw_local['close'].iloc[-1]) if raw_local is not None and not raw_local.empty else float(rth['close'].iloc[-1])
        override_asof_text = price_override_as_of.get(symbol, '')
        override_ts = _parse_price_timestamp(override_asof_text)
        use_override = (
            as_of_cutoff is None
            and symbol in price_overrides
            and (override_ts is not None)
            and (override_ts >= raw_latest_ts)
        )
        if use_override:
            latest_prices[symbol] = float(price_overrides[symbol])
            latest_price_as_of[symbol] = _format_price_timestamp(override_ts)
            current_asof_for_context = override_ts
        elif as_of_cutoff is None:
            latest_prices[symbol] = raw_latest_price
            latest_price_as_of[symbol] = _format_price_timestamp(raw_latest_ts)
            current_asof_for_context = raw_latest_ts
        else:
            latest_prices[symbol] = float(rth['close'].iloc[-1])
            latest_price_as_of[symbol] = bar_as_of
            current_asof_for_context = rth.index[-1]

        symbol_movement_context[symbol] = compute_symbol_movement_context(
            symbol, raw, rth, latest_prices[symbol], current_asof_for_context
        )
        symbol_movement_context[symbol]['current_price_session'] = symbol_movement_context[symbol].get('latest_price_session', '') or _market_session_label(latest_price_as_of.get(symbol, ''))

        # v0.36.6: descriptive price-structure labels for the watchlist.
        # These are informational only and do not affect watchlist eligibility.
        daily_for_structure = None
        daily_structure_path = DATA_DIR / f"{symbol}_1D.csv"
        if daily_structure_path.exists():
            try:
                daily_for_structure = load_symbol_csv(daily_structure_path)
                if as_of_cutoff is not None:
                    daily_for_structure = daily_for_structure.loc[daily_for_structure.index <= as_of_cutoff].copy()
            except Exception:
                daily_for_structure = None
        structure_context[symbol] = _compute_symbol_structure_context(raw, daily_for_structure, latest_prices[symbol])

        for label, rule in TIMEFRAMES.items():
            # Use the separate 1D download when available so daily zones can use
            # a longer lookback than the 5M intraday file. Intraday timeframes
            # continue to be built from regular-session 5M bars.
            if label == "1D":
                daily_path = DATA_DIR / f"{symbol}_1D.csv"
                if daily_path.exists():
                    daily_raw = load_symbol_csv(daily_path)
                    if as_of_cutoff is not None:
                        daily_raw = daily_raw.loc[daily_raw.index <= as_of_cutoff].copy()
                    bars = daily_raw
                else:
                    bars = aggregate_bars(raw, rule)
            else:
                bars = aggregate_bars(raw, rule)
            if bars.empty:
                continue
            zones = detect_zones(bars, symbol, label)
            all_zones.extend(zones)

    zones_df = pd.DataFrame(all_zones)
    meta = {
        'price_source': 'latest downloaded market price; timestamp-aware overrides only',
        'latest_bar_times': latest_bar_times,
        'current_price_as_of': latest_price_as_of,
        'symbol_movement_context': symbol_movement_context,
        'price_overrides_loaded': bool(price_overrides),
        'require_5m_source': REQUIRE_5M_SOURCE_FOR_WATCHLIST,
        'skipped_symbols': skipped_symbols,
        'as_of_date': as_of_date or '',
    }
    if zones_df.empty:
        meta['active_zone_count'] = 0
        return zones_df, pd.DataFrame(), meta

    active_zones_df = _active_zones_for_watchlist(zones_df)
    meta['active_zone_count'] = int(len(active_zones_df))
    meta['expired_zone_count'] = int(len(zones_df) - len(active_zones_df))
    watch_zones_df = merge_overlapping_zones(active_zones_df)

    candidates = []
    for _, z in watch_zones_df.iterrows():
        sym = str(z['symbol']).upper()
        if sym not in latest_prices:
            continue
        price = latest_prices[sym]
        top = float(z['zone_top'])
        bottom = float(z['zone_bottom'])
        in_zone = _price_inside_zone(price, top, bottom)

        scenario_defs = []
        if z['zone_type'] == 'demand':
            # Bullish reaction from demand.
            if price is not None:
                scenario_defs.append({
                    'scenario': 'demand_hold',
                    'setup': 'Demand Zone Test / Hold',
                    'bias': 'Bullish',
                    'side': 'long',
                    'trigger': bottom,
                    'confirmation': top,
                    'invalidation': bottom,
                    'distance': _distance_to_zone(price, z['zone_type'], top, bottom),
                })
            # Bearish continuation through demand.
            if price is not None:
                scenario_defs.append({
                    'scenario': 'demand_break',
                    'setup': 'Demand Zone Breakdown',
                    'bias': 'Bearish',
                    'side': 'short',
                    'trigger': bottom,
                    'confirmation': bottom,
                    'invalidation': top,
                    'distance': _distance_to_trigger_pct(price, bottom, in_zone=in_zone),
                })
        else:
            # Bearish reaction from supply.
            if price is not None:
                scenario_defs.append({
                    'scenario': 'supply_reject',
                    'setup': 'Supply Zone Rejection',
                    'bias': 'Bearish',
                    'side': 'short',
                    'trigger': top,
                    'confirmation': bottom,
                    'invalidation': top,
                    'distance': _distance_to_zone(price, z['zone_type'], top, bottom),
                })
            # Bullish continuation through supply.
            if price is not None:
                scenario_defs.append({
                    'scenario': 'supply_break',
                    'setup': 'Supply Zone Breakout',
                    'bias': 'Bullish',
                    'side': 'long',
                    'trigger': top,
                    'confirmation': top,
                    'invalidation': bottom,
                    'distance': _distance_to_trigger_pct(price, top, in_zone=in_zone),
                })

        for sc in scenario_defs:
            status = _status(float(sc['distance']))
            if float(sc['distance']) > float(WATCHLIST_SCENARIO_MAX_DISTANCE_PCT):
                continue

            target_info = _build_target_ladder(
                watch_zones_df, z['symbol'], z['zone_type'], price, z, side=sc['side']
            )

            candidate = {
                'status': status,
                'symbol': z['symbol'],
                'current_price': round(price, 2),
                'current_price_as_of': latest_price_as_of.get(z['symbol'], ''),
                'setup': sc['setup'],
                'scenario': sc['scenario'],
                'scenario_label': _scenario_display_label(sc['scenario'], sc['side']),
                'option_contract': _option_contract_label(sc['side']),
                'bias': sc['bias'],
                'side': sc['side'],
                'timeframe': z['timeframe'],
                'pattern': z['pattern'],
                'zone_type': z['zone_type'],
                'zone_top': z['zone_top'],
                'zone_bottom': z['zone_bottom'],
                'trigger_level': round(float(sc['trigger']), 2),
                'confirmation_level': round(float(sc['confirmation']), 2),
                'invalidation_level': round(float(sc['invalidation']), 2),
                'distance_pct': round(float(sc['distance']), 2),
                'freshness': z['freshness'],
                'freshness_label': _freshness_label(z['freshness'], z['tests']),
                'tests': z['tests'],
                'departure_atr': z['departure_atr'],
                'departure_grade': _departure_grade(z['departure_atr']),
                'departure_volume_ratio': z['departure_volume_ratio'],
                'departure_body_vs_base_body': z.get('departure_body_vs_base_body', z.get('departure_move_vs_base_range', None)),
                'volume_grade': _volume_grade(z['departure_volume_ratio']),
                'quality_score': z['quality_score'],
                'target': target_info.get('target'),
                'estimated_rr': target_info.get('estimated_rr'),
                'target_rr_raw': target_info.get('target_rr_raw'),
                'target_quality': target_info.get('target_quality'),
                'target_level': target_info.get('target_level'),
                'target_reject_reason': target_info.get('target_reject_reason'),
                'target_rr_range': target_info.get('target_rr_range'),
                'target_ladder': target_info.get('target_ladder'),
                'target_1_zone': target_info.get('target_1_zone'),
                'target_1_rr': target_info.get('target_1_rr'),
                'target_1_freshness': target_info.get('target_1_freshness'),
                'target_1_tests': target_info.get('target_1_tests'),
                'target_1_role': target_info.get('target_1_role'),
                'target_2_zone': target_info.get('target_2_zone'),
                'target_2_rr': target_info.get('target_2_rr'),
                'target_2_freshness': target_info.get('target_2_freshness'),
                'target_2_tests': target_info.get('target_2_tests'),
                'target_2_role': target_info.get('target_2_role'),
                'target_3_zone': target_info.get('target_3_zone'),
                'target_3_rr': target_info.get('target_3_rr'),
                'target_3_freshness': target_info.get('target_3_freshness'),
                'target_3_tests': target_info.get('target_3_tests'),
                'target_3_role': target_info.get('target_3_role'),
                'intervening_zone_count': target_info.get('intervening_zone_count'),
                'nearest_obstacle_zone': target_info.get('nearest_obstacle_zone'),
                'target_selection_reason': target_info.get('target_selection_reason'),
                'grade': 'PENDING',
                'confluence_count': int(z.get('confluence_count', 1)),
                'confluence_timeframes': z.get('confluence_timeframes', z['timeframe']),
                'merged_from': z.get('merged_from', ''),
                'primary_timeframe': z.get('primary_timeframe', z['timeframe']),
            }
            candidate.update(_zone_metadata_payload(z))
            candidates.append(candidate)

    watch_df = pd.DataFrame(candidates)
    watch_df = _add_watchlist_visual_context(watch_df, watch_zones_df, structure_context, latest_prices, latest_price_as_of)
    watch_df = enrich_movement_context(watch_df, symbol_movement_context, load_zone_reaction_history(REPORT_DIR))
    watch_df = _apply_hard_exclusions(watch_df)
    if not watch_df.empty:
        # Explain every candidate's eligibility. This keeps raw detection broad while
        # making the final filter transparent and mentor-style.
        reasons_col = []
        for _, r in watch_df.iterrows():
            reasons = _entry_zone_rejection_reasons(r)
            if not isinstance(r.get('target'), str):
                reasons.append(r.get('target_reject_reason') or 'rejected_no_valid_target')
            elif pd.isna(r.get('target_quality')) or float(r.get('target_quality') or 0) < float(TARGET_ZONE_MIN_QUALITY_SCORE):
                reasons.append('rejected_low_target_quality')
            if pd.isna(r.get('estimated_rr')) or float(r.get('estimated_rr') or 0) < float(MIN_FINAL_RR):
                reasons.append('rejected_low_rr_to_nearest_opposing_zone')
            reasons_col.append(';'.join(dict.fromkeys([str(x) for x in reasons if x])) or 'eligible_pre_grade')
        watch_df['rejection_reasons'] = reasons_col
        watch_df['setup_quality_score'] = watch_df.apply(_setup_quality_score, axis=1)
        watch_df['setup_quality_grade'] = watch_df['setup_quality_score'].apply(_setup_grade)
        # Keep legacy `grade` as the setup-quality grade for downstream sorting/reporting.
        # Entry-confirmation scores are intentionally reserved for backtesting/trade review.
        watch_df['grade'] = watch_df['setup_quality_grade']
        watch_df['rr_tier'] = watch_df['estimated_rr'].apply(_rr_tier)
        watch_df['rr_tier_class'] = watch_df['estimated_rr'].apply(_rr_tier_class)
        watch_df['rr_range_summary'] = watch_df.apply(_rr_range_summary, axis=1)
        min_rank = _grade_rank_value(FINAL_REPORT_MIN_GRADE)
        watch_df['final_eligible'] = (
            (watch_df['rejection_reasons'] == 'eligible_pre_grade')
            & (watch_df['grade'].apply(_grade_rank_value) >= min_rank)
            & (pd.to_numeric(watch_df['estimated_rr'], errors='coerce') >= float(MIN_FINAL_RR))
        )
        watch_df.loc[(watch_df['rejection_reasons'] == 'eligible_pre_grade') & (~watch_df['final_eligible']), 'rejection_reasons'] = 'rejected_grade_below_final_min'

        watch_df['watchlist_bucket'] = watch_df.apply(_scenario_readiness, axis=1)
        watch_df['scenario_status'] = watch_df.apply(_scenario_status, axis=1)
        watch_df['confirmation_needed'] = watch_df.apply(_confirmation_needed, axis=1)

        status_order = {'Immediate': 0, 'Near-Term': 1, 'Future': 2}
        watch_df['status_rank'] = watch_df['status'].map(status_order)
        watch_df['grade_rank'] = watch_df['grade'].apply(lambda g: -_grade_rank_value(g))
        watch_df = watch_df.sort_values(
            ['status_rank', 'grade_rank', 'distance_pct', 'quality_score'],
            ascending=[True, True, True, False]
        ).drop(columns=['status_rank', 'grade_rank'])

    return zones_df, watch_df, meta


def _summary_table(watch_df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append('## Dashboard')
    lines.append('')
    if watch_df.empty:
        lines.append('| Metric | Value |')
        lines.append('|---|---:|')
        lines.append('| Final setups | 0 |')
        lines.append('')
        return lines

    bullish = int((watch_df['bias'] == 'Bullish').sum())
    bearish = int((watch_df['bias'] == 'Bearish').sum())
    avg_dist = watch_df['distance_pct'].mean()
    lines.append('| Metric | Value |')
    lines.append('|---|---:|')
    lines.append(f'| Final setups | {len(watch_df)} |')
    lines.append(f'| Bullish / bearish | {bullish} / {bearish} |')
    lines.append(f'| Average distance to zone | {avg_dist:.2f}% |')
    lines.append(f'| Best grade | {watch_df.iloc[0]["grade"]} |')
    lines.append('')

    lines.append('## Quick View')
    lines.append('')
    lines.append('| Section | Symbol | Bias | Grade | Zone | Distance | T1 R:R | R:R Range | Target Ladder |')
    lines.append('|---|---|---|---:|---|---:|---:|---:|---|')
    for _, r in watch_df.head(20).iterrows():
        zone = f'{r["timeframe"]} {r["zone_type"]} {_fmt_money(r["zone_bottom"])}–{_fmt_money(r["zone_top"])}'
        target = r.get('target_ladder') if isinstance(r.get('target_ladder'), str) else (r['target'] if isinstance(r['target'], str) else 'n/a')
        rr_range = r.get('target_rr_range') if isinstance(r.get('target_rr_range'), str) else _rr_text(r.get('estimated_rr'))
        lines.append(f'| {r["status"]} | {r["symbol"]} | {r["bias"]} | {r["grade"]} | {zone} | {_fmt_pct(r["distance_pct"])} | {_rr_text(r["estimated_rr"])} | {rr_range} | {target} |')
    lines.append('')
    return lines


def generate_markdown(watch_df: pd.DataFrame, meta: dict) -> str:
    report_date = meta.get('report_date', '')
    title = f'# Supply & Demand Scenario Watchlist — {report_date}' if report_date else '# Supply & Demand Watchlist'
    lines = [title, '']
    lines.append(f'_Price source: {meta.get("price_source", "latest delayed OHLCV close")}_')
    lines.append(f'_Final report filter: grades {FINAL_REPORT_MIN_GRADE} and above, minimum T1 R:R 1:{MIN_FINAL_RR:.2f}_')
    if meta.get('latest_bar_times'):
        last_seen = max(meta['latest_bar_times'].values())
        lines.append(f'_Latest OHLCV bar seen: {last_seen}_')
    lines.append('')

    if watch_df.empty:
        lines.append(f'No setups met the final report filter of grade {FINAL_REPORT_MIN_GRADE} or better with minimum T1 R:R 1:{MIN_FINAL_RR:.2f}. Check `watchlist_all_candidates.csv` for lower-grade or lower-R:R candidates.')
        return '\n'.join(lines)

    lines.extend(_summary_table(watch_df))

    for section in ['Immediate', 'Near-Term', 'Future']:
        sub = watch_df[watch_df['status'] == section]
        if sub.empty:
            continue
        lines.append(f'## {section}')
        lines.append('')
        for _, r in sub.iterrows():
            emoji = '📈' if r['bias'] == 'Bullish' else '📉'
            lines.append(f"### 📌 {r['symbol']} — {r['bias']} {r['setup']} `{r['grade']}`")
            lines.append('')
            lines.append(f"**Snapshot:** current {_fmt_money(r['current_price'])} • distance {_fmt_pct(r['distance_pct'])} • T1 R:R {_rr_text(r['estimated_rr'])} • range {r.get('target_rr_range', _rr_text(r['estimated_rr']))}")
            lines.append('')
            lines.append('| Field | Detail |')
            lines.append('|---|---|')
            lines.append(f"| Zone | **{r['timeframe']} {r['zone_type']}** {_fmt_money(r['zone_bottom'])}–{_fmt_money(r['zone_top'])} |")
            if int(r.get('confluence_count', 1)) > 1:
                lines.append(f"| Confluence | {int(r['confluence_count'])} overlapping zones: {r.get('confluence_timeframes', '')} |")
            if r['zone_type'] == 'demand':
                lines.append(f"| Failure level | {_fmt_money(r['invalidation_level'])} demand bottom |")
                lines.append(f"| Reclaim / hold area | {_fmt_money(r['confirmation_level'])} demand top |")
            else:
                lines.append(f"| Failure level | {_fmt_money(r['invalidation_level'])} supply top |")
                lines.append(f"| Rejection area | {_fmt_money(r['confirmation_level'])} supply bottom |")
            lines.append(f"| Pattern | {r['pattern']} |")
            lines.append(f"| Freshness | {r['freshness_label']} |")
            lines.append(f"| Departure | {r['departure_grade']} ({r['departure_atr']} ATR) |")
            if pd.notna(r.get('departure_body_vs_base_body')):
                lines.append(f"| Departure body vs. base body | {r['departure_body_vs_base_body']}x |")
            lines.append(f"| Volume | {r['volume_grade']} ({r['departure_volume_ratio']}x average) |")
            lines.append(f"| Quality score | {r['quality_score']}/10 |")
            if isinstance(r['target'], str):
                lines.append(f"| T1 target / nearest opposing zone | {r['target']} |")
            if isinstance(r.get('target_rr_range'), str):
                lines.append(f"| R:R range | {r.get('target_rr_range')} |")
            if isinstance(r.get('target_ladder'), str):
                lines.append(f"| Target ladder | {r.get('target_ladder')} |")
            lines.append('')
            scenario = str(r.get('scenario', ''))
            if scenario == 'supply_break':
                lines.append(f"{emoji} **Plan:** Watch for a confirmed 5m close above the {r['timeframe']} supply breakout level near {_fmt_money(r['trigger_level'])}, followed by continuation above 9EMA/VWAP with volume. The broken supply zone becomes the risk area; targets are the next supply zones above.")
            elif scenario == 'demand_break':
                lines.append(f"{emoji} **Plan:** Watch for a confirmed 5m close below the {r['timeframe']} demand breakdown level near {_fmt_money(r['trigger_level'])}, followed by continuation below 9EMA/VWAP with volume. The broken demand zone becomes the risk area; targets are the next demand zones below.")
            elif scenario == 'demand_hold':
                lines.append(f"{emoji} **Plan:** If price tests and holds above the {r['timeframe']} demand zone near {_fmt_money(r['invalidation_level'])}, stays above the open, 9EMA, and VWAP, and forms bullish 5m candles with increasing volume, monitor for a long toward the next supply zone.")
            else:
                lines.append(f"{emoji} **Plan:** If price tests and rejects the {r['timeframe']} supply zone near {_fmt_money(r['invalidation_level'])}, stays below the open, 9EMA, and VWAP, and forms lower lows on the 5m with downside volume, monitor for a short toward the next demand zone.")
            lines.append('')
    return '\n'.join(lines)


def _html_escape(x) -> str:
    return html.escape('' if x is None or pd.isna(x) else str(x))


def _badge_class(value: str) -> str:
    v = str(value).lower().replace('+', 'plus')
    return ''.join(ch for ch in v if ch.isalnum() or ch == '-')


def generate_html_report(watch_df: pd.DataFrame, all_watch_df: pd.DataFrame, meta: dict) -> str:
    price_source = _html_escape(meta.get('price_source', 'latest delayed OHLCV close'))
    last_seen = _html_escape(max(meta.get('latest_bar_times', {'n/a': 'n/a'}).values()))
    total_candidates = len(all_watch_df)
    final_count = len(watch_df)
    bullish = int((watch_df['bias'] == 'Bullish').sum()) if not watch_df.empty else 0
    bearish = int((watch_df['bias'] == 'Bearish').sum()) if not watch_df.empty else 0
    avg_dist = f"{watch_df['distance_pct'].mean():.2f}%" if not watch_df.empty else 'n/a'
    report_dt = _html_escape(meta.get('report_datetime', ''))
    report_date = _html_escape(meta.get('report_date', ''))

    style = """
    <style>
      :root { --bg:#0f172a; --panel:#111827; --panel2:#1f2937; --text:#e5e7eb; --muted:#9ca3af; --line:#374151; --bull:#16a34a; --bear:#dc2626; --gold:#f59e0b; --blue:#38bdf8; }
      body { margin:0; font-family: Arial, Helvetica, sans-serif; background:var(--bg); color:var(--text); }
      .wrap { max-width:1180px; margin:0 auto; padding:28px; }
      h1 { margin:0 0 6px 0; font-size:32px; }
      h2 { margin:32px 0 16px; border-bottom:1px solid var(--line); padding-bottom:8px; }
      .sub { color:var(--muted); line-height:1.5; }
      .stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin:22px 0; }
      .stat { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px; }
      .stat .label { color:var(--muted); font-size:13px; }
      .stat .value { font-size:24px; font-weight:700; margin-top:6px; }
      table { width:100%; border-collapse:collapse; background:var(--panel); border-radius:14px; overflow:hidden; }
      th, td { padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; font-size:14px; }
      th { color:var(--muted); background:var(--panel2); font-weight:600; }
      tr:last-child td { border-bottom:none; }
      .cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap:16px; }
      .card { background:var(--panel); border:1px solid var(--line); border-left:6px solid var(--blue); border-radius:16px; padding:18px; box-shadow: 0 8px 20px rgba(0,0,0,.18); }
      .card.bullish { border-left-color:var(--bull); }
      .card.bearish { border-left-color:var(--bear); }
      .card h3 { margin:0 0 8px; font-size:20px; }
      .badges { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 14px; }
      .badge { display:inline-block; padding:4px 9px; border-radius:999px; background:var(--panel2); color:var(--text); font-size:12px; border:1px solid var(--line); }
      .badge.grade-aplus, .badge.grade-a { background:rgba(245,158,11,.18); border-color:rgba(245,158,11,.5); color:#fde68a; }
      .badge.bullish { background:rgba(22,163,74,.16); border-color:rgba(22,163,74,.5); color:#bbf7d0; }
      .badge.bearish { background:rgba(220,38,38,.16); border-color:rgba(220,38,38,.5); color:#fecaca; }
      .grid { display:grid; grid-template-columns: 1fr 1fr; gap:10px 16px; margin:12px 0; }
      .item .k { color:var(--muted); font-size:12px; margin-bottom:2px; }
      .item .v { font-weight:600; }
      .bar { height:9px; background:#374151; border-radius:999px; overflow:hidden; margin-top:6px; }
      .fill { height:100%; background:linear-gradient(90deg, var(--blue), var(--gold)); }
      .plan { color:#d1d5db; background:rgba(255,255,255,.04); border:1px solid var(--line); border-radius:12px; padding:12px; line-height:1.5; margin-top:14px; }
      .small { color:var(--muted); font-size:12px; }
      .rr-legend { display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 22px; }
      .rr-pill { font-size:12px; border-radius:999px; padding:5px 10px; border:1px solid var(--line); background:var(--panel); }
      .rr-excellent { color:#bbf7d0; border-color:rgba(22,163,74,.65); background:rgba(22,163,74,.14); }
      .rr-strong { color:#d9f99d; border-color:rgba(132,204,22,.65); background:rgba(132,204,22,.14); }
      .rr-acceptable { color:#fde68a; border-color:rgba(245,158,11,.65); background:rgba(245,158,11,.14); }
      .rr-watch { color:#fed7aa; border-color:rgba(249,115,22,.65); background:rgba(249,115,22,.14); }
      .rr-poor, .rr-none { color:#fecaca; border-color:rgba(220,38,38,.65); background:rgba(220,38,38,.12); }
      .target-ladder { display:grid; gap:8px; margin-top:8px; }
      .target-row { padding:8px 10px; border-radius:10px; border:1px solid var(--line); display:grid; grid-template-columns: 34px 1fr auto; gap:8px; align-items:center; }
      .target-row em { display:block; font-style:normal; font-size:11px; color:var(--muted); }
      .target-row small { grid-column:2 / span 2; color:var(--muted); font-size:11px; }
      .checklist { margin:8px 0 0; padding-left:18px; line-height:1.55; }
      .checklist li { margin:2px 0; }
      .zone-visual { margin:12px 0 14px; }
      .zone-svg { width:100%; height:auto; display:block; border-radius:14px; }
      .chart-bg{fill:#0b1224}.chart-grid{stroke:#263244;stroke-width:1}.chart-axis,.zone-label{fill:#9ca3af;font-size:11px}.zone-label{font-weight:700}.demand-band{fill:rgba(22,163,74,.20);stroke:rgba(22,163,74,.55);stroke-width:1}.supply-band{fill:rgba(220,38,38,.19);stroke:rgba(220,38,38,.55);stroke-width:1}.up-candle{fill:#22c55e;stroke:#22c55e}.down-candle{fill:#ef4444;stroke:#ef4444}.wick{stroke-width:1.2}.ema-line{fill:none;stroke:#f59e0b;stroke-width:1.5}.vwap-line{fill:none;stroke:#38bdf8;stroke-width:1.5;stroke-dasharray:4 3}.price-line{stroke:#93c5fd;stroke-width:1.4;stroke-dasharray:5 4}.price-label{fill:#bfdbfe;font-size:11px;font-weight:700}.legend text{fill:#cbd5e1;font-size:11px}.legend text:first-child{fill:#fbbf24}.zone-chart-empty{color:var(--muted);border:1px dashed var(--line);border-radius:12px;padding:16px}
      a { color:#93c5fd; }
    </style>
    """

    rows = []
    if not watch_df.empty:
        for _, r in watch_df.head(20).iterrows():
            target = _html_escape(r.get('target_ladder') if isinstance(r.get('target_ladder'), str) else (r['target'] if isinstance(r['target'], str) else 'n/a'))
            rr_range = _html_escape(r.get('rr_range_summary') if isinstance(r.get('rr_range_summary'), str) else (r.get('target_rr_range') if isinstance(r.get('target_rr_range'), str) else _rr_text(r.get('estimated_rr'))))
            rr_cls = _rr_tier_class(r.get('estimated_rr'))
            rows.append(
                f"<tr><td>{_html_escape(r['status'])}</td><td><strong>{_html_escape(r['symbol'])}</strong></td>"
                f"<td>{_html_escape(r.get('option_contract',''))}</td><td>{_html_escape(r.get('scenario_label', r.get('setup','')))}</td><td>{_html_escape(r.get('setup_quality_grade', r['grade']))}</td>"
                f"<td>{_html_escape(r['timeframe'])} {_html_escape(r['zone_type'])} {_fmt_money(r['zone_bottom'])}–{_fmt_money(r['zone_top'])}</td>"
                f"<td>{_fmt_pct(r['distance_pct'])}</td><td class='{rr_cls}'>{_rr_text(r['estimated_rr'])}<br><small>{_html_escape(r.get('rr_tier',''))}</small></td><td>{rr_range}</td><td>{target}</td></tr>"
            )

    if rows:
        quick_table = _rr_tier_legend_html() + """
        <h2>Quick View</h2>
        <table><thead><tr><th>Section</th><th>Symbol</th><th>Contract</th><th>Scenario</th><th>Setup Grade</th><th>Zone</th><th>Distance</th><th>T1 R:R Tier</th><th>R:R Range</th><th>Target Ladder</th></tr></thead><tbody>
        """ + "\n".join(rows) + "</tbody></table>"
    else:
        quick_table = _rr_tier_legend_html() + f"<p>No setups met the final report filter of grade {FINAL_REPORT_MIN_GRADE} or better with minimum T1 R:R 1:{MIN_FINAL_RR:.2f}. Review <code>watchlist_all_candidates.csv</code> for lower-grade or lower-R:R candidates.</p>"

    def _scenario_table(df: pd.DataFrame, title: str, max_rows: int = 30) -> str:
        if df.empty:
            return ''
        rows2 = []
        view = df.copy()
        if 'distance_pct' in view.columns:
            view = view.sort_values(['distance_pct'], ascending=[True])
        for _, r in view.head(max_rows).iterrows():
            rr_cls = _rr_tier_class(r.get('estimated_rr'))
            rows2.append(
                f"<tr><td><strong>{_html_escape(r.get('symbol',''))}</strong></td>"
                f"<td>{_html_escape(r.get('option_contract',''))}</td><td>{_html_escape(r.get('scenario_label', r.get('setup','')))}</td>"
                f"<td>{_html_escape(r.get('scenario_status',''))}</td><td>{_html_escape(r.get('setup_quality_grade', r.get('grade','')))}</td>"
                f"<td>{_fmt_pct(r.get('distance_pct'))}</td><td class='{rr_cls}'>{_rr_text(r.get('estimated_rr'))}<br><small>{_html_escape(r.get('rr_tier',''))}</small></td>"
                f"<td>{_html_escape(r.get('rr_range_summary', r.get('target_rr_range') if isinstance(r.get('target_rr_range'), str) else ''))}</td>"
                f"<td>{_html_escape(r.get('freshness_label',''))}</td>"
                f"<td>{_html_escape(r.get('confirmation_needed',''))}</td></tr>"
            )
        return f"<h2>{_html_escape(title)}</h2><table><thead><tr><th>Symbol</th><th>Contract</th><th>Scenario</th><th>Status</th><th>Setup Grade</th><th>Distance</th><th>T1 R:R Tier</th><th>R:R Range Summary</th><th>Freshness</th><th>Confirmation Checklist</th></tr></thead><tbody>{''.join(rows2)}</tbody></table>"

    developing_html = ''
    zone_map_html = ''
    if WATCHLIST_INCLUDE_DEVELOPING_IN_HTML and not all_watch_df.empty and 'watchlist_bucket' in all_watch_df.columns:
        developing_df = all_watch_df[all_watch_df['watchlist_bucket'].eq('Developing Scenario')].copy()
        developing_html = _scenario_table(developing_df, 'Developing Scenario Watchlist', 40)
    if WATCHLIST_INCLUDE_ZONE_MAP_IN_HTML and not all_watch_df.empty and 'watchlist_bucket' in all_watch_df.columns:
        zone_map_df = all_watch_df[all_watch_df['watchlist_bucket'].eq('Zone Map / Research')].copy()
        # Keep zone map compact: nearest and highest quality research scenarios first.
        if not zone_map_df.empty:
            zone_map_df = zone_map_df.sort_values(['distance_pct', 'quality_score'], ascending=[True, False])
        zone_map_html = _scenario_table(zone_map_df, 'Zone Map / Research Context', 30)

    sections_html = []
    for section in ['Immediate', 'Near-Term', 'Future']:
        sub = watch_df[watch_df['status'] == section] if not watch_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        cards = []
        for _, r in sub.iterrows():
            bias_class = 'bullish' if r['bias'] == 'Bullish' else 'bearish'
            grade_class = 'grade-' + _badge_class(r['grade'])
            score = max(0, min(10, float(r['quality_score'])))
            fill = score * 10
            scenario = str(r.get('scenario', ''))
            if scenario == 'supply_break':
                plan = f"Watch for a confirmed 5m close above the {r['timeframe']} supply breakout level near {_fmt_money(r['trigger_level'])}, followed by continuation above 9EMA/VWAP with volume. The broken supply zone becomes the risk area; targets are the next supply zones above."
                confirm_label = 'Supply top / breakout'
                fail_label = 'Supply bottom / risk area'
            elif scenario == 'demand_break':
                plan = f"Watch for a confirmed 5m close below the {r['timeframe']} demand breakdown level near {_fmt_money(r['trigger_level'])}, followed by continuation below 9EMA/VWAP with volume. The broken demand zone becomes the risk area; targets are the next demand zones below."
                confirm_label = 'Demand bottom / breakdown'
                fail_label = 'Demand top / risk area'
            elif scenario == 'demand_hold':
                plan = f"If price tests and holds above the {r['timeframe']} demand zone near {_fmt_money(r['invalidation_level'])}, stays above the open, 9EMA, and VWAP, and forms bullish 5m candles with increasing volume, monitor for a long toward the next supply zone."
                confirm_label = 'Demand top / hold area'
                fail_label = 'Demand bottom / failure'
            else:
                plan = f"If price tests and rejects the {r['timeframe']} supply zone near {_fmt_money(r['invalidation_level'])}, stays below the open, 9EMA, and VWAP, and forms lower lows on the 5m with downside volume, monitor for a short toward the next demand zone."
                confirm_label = 'Supply bottom / rejection'
                fail_label = 'Supply top / failure'
            rr_cls = _rr_tier_class(r.get('estimated_rr'))
            target_ladder_html = _target_ladder_html(r)
            checklist_html = _checklist_html(r.get('confirmation_needed', ''))
            chart_html = _zone_chart_svg(r, width=620, height=270, max_bars=78)
            cards.append(f"""
            <article class="card {bias_class}">
              <h3>{'📈' if r['bias'] == 'Bullish' else '📉'} {_html_escape(r['symbol'])} — {_html_escape(r.get('scenario_label', r['setup']))}</h3>
              <div class="badges">
                <span class="badge {grade_class}">Setup Grade {_html_escape(r.get('setup_quality_grade', r['grade']))}</span>
                <span class="badge {bias_class}">{_html_escape(r.get('option_contract',''))}</span>
                <span class="badge {bias_class}">{_html_escape(r['bias'])}</span>
                <span class="badge">{_html_escape(r['status'])}</span>
                <span class="badge">{_html_escape(r['pattern'])}</span>
                <span class="badge">{_html_escape(r.get('structure_alignment',''))}</span>
                <span class="badge">{_html_escape(r.get('structure_trade_alignment',''))}</span>
                <span class="badge">Obs {_html_escape(r.get('observation_score',''))}</span>
                {('<span class="badge">Confluence ' + str(int(r.get('confluence_count', 1))) + '</span>') if int(r.get('confluence_count', 1)) > 1 else ''}
              </div>
              <div class="zone-visual">{chart_html}</div>
              <div class="grid">
                <div class="item"><div class="k">Current price</div><div class="v">{_fmt_money(r['current_price'])}<br><small>as of {_html_escape(r.get('current_price_as_of',''))} — {_html_escape(r.get('current_price_session',''))}</small></div></div>
                <div class="item"><div class="k">Distance to zone</div><div class="v">{_fmt_pct(r['distance_pct'])}</div></div>
                <div class="item"><div class="k">Price position</div><div class="v">{_html_escape(str(r.get('price_position_status','')).replace('_',' '))}</div></div>
                <div class="item"><div class="k">Structure</div><div class="v">{_html_escape(r.get('structure_alignment',''))}<br><small>{_html_escape(r.get('structure_bias_5m',''))} / {_html_escape(r.get('structure_bias_15m',''))}</small></div></div>
                <div class="item"><div class="k">Trade vs structure</div><div class="v">{_html_escape(r.get('structure_trade_alignment',''))}</div></div>
                <div class="item"><div class="k">Movement context</div><div class="v">{_html_escape(str(r.get('zone_movement_state','')).replace('_',' '))}<br><small>{_html_escape(r.get('vpa_state',''))} • {_html_escape(str(r.get('gap_zone_context','')).replace('_',' '))}</small></div></div>
                <div class="item"><div class="k">Watch for</div><div class="v">{_html_escape(r.get('watch_for',''))}</div></div>
                <div class="item"><div class="k">Nearest demand / supply</div><div class="v">D {_fmt_money(r.get('closest_demand_bottom'))}–{_fmt_money(r.get('closest_demand_top'))}<br>S {_fmt_money(r.get('closest_supply_bottom'))}–{_fmt_money(r.get('closest_supply_top'))}</div></div>
                <div class="item"><div class="k">Scenario</div><div class="v">{_html_escape(r.get('scenario_label', r['setup']))}</div></div>
                <div class="item"><div class="k">Zone</div><div class="v">{_html_escape(r['timeframe'])} {_html_escape(r['zone_type'])}<br>{_fmt_money(r['zone_bottom'])}–{_fmt_money(r['zone_top'])}</div></div>
                <div class="item"><div class="k">T1 R:R tier</div><div class="v {rr_cls}">{_rr_text(r['estimated_rr'])}<br><small>{_html_escape(r.get('rr_tier',''))}</small></div></div>
                <div class="item"><div class="k">R:R range</div><div class="v">{_html_escape(r.get('rr_range_summary', r.get('target_rr_range') if isinstance(r.get('target_rr_range'), str) else _rr_text(r.get('estimated_rr'))))}</div></div>
                <div class="item"><div class="k">{fail_label}</div><div class="v">{_fmt_money(r['invalidation_level'])}</div></div>
                <div class="item"><div class="k">{confirm_label}</div><div class="v">{_fmt_money(r['confirmation_level'])}</div></div>
                <div class="item"><div class="k">Freshness</div><div class="v">{_html_escape(r['freshness_label'])}</div></div>
                <div class="item"><div class="k">Confluence</div><div class="v">{_html_escape(r.get('confluence_timeframes', r['timeframe']))}</div></div>
                <div class="item"><div class="k">Departure</div><div class="v">{_html_escape(r['departure_grade'])} ({_html_escape(r['departure_atr'])} ATR)</div></div>
                <div class="item"><div class="k">Departure volume</div><div class="v">{_html_escape(r['volume_grade'])} ({_html_escape(r['departure_volume_ratio'])}x avg)</div></div>
              </div>
              <div class="item"><div class="k">Setup quality score</div><div class="v">{score:.1f}/10</div><div class="bar"><div class="fill" style="width:{fill:.0f}%"></div></div></div>
              <div class="plan"><strong>Target ladder and R:R tiers:</strong><div class="target-ladder">{target_ladder_html}</div></div>
              <div class="plan">{checklist_html}</div>
              <div class="plan"><strong>Plan:</strong> {_html_escape(plan)}</div>
            </article>
            """)
        sections_html.append(f"<h2>{section}</h2><div class='cards'>{''.join(cards)}</div>")

    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Supply & Demand Scenario Watchlist {report_date}</title>{style}</head>
<body><main class="wrap">
  <h1>Supply & Demand Scenario Watchlist — {report_date}</h1>
  <div class="sub">Generated: {report_dt} America/New_York<br>Price source: {price_source}<br>Final report filter: setup grades {FINAL_REPORT_MIN_GRADE} and above, minimum T1 R:R 1:{MIN_FINAL_RR:.2f}<br>Entry-confirmation score is intentionally reserved for backtesting/trade review, not watchlist inclusion.<br>Latest OHLCV bar seen: {last_seen}<br>Current price uses the latest downloaded market bar/quote snapshot for proximity; zones are built from regular-session candles only. Candidates are excluded entirely while current price is inside any active zone because that state is unresolved consolidation/chop. Stale quote overrides are ignored when downloaded bars are newer.</div>
  <section class="stats">
    <div class="stat"><div class="label">Final setups</div><div class="value">{final_count}</div></div>
    <div class="stat"><div class="label">All candidates</div><div class="value">{total_candidates}</div></div>
    <div class="stat"><div class="label">Bullish / bearish</div><div class="value">{bullish} / {bearish}</div></div>
    <div class="stat"><div class="label">Avg. distance</div><div class="value">{avg_dist}</div></div>
  </section>
  {quick_table}
  {''.join(sections_html)}
  {developing_html}
  {zone_map_html}
  <p class="small">v0.29 separates setup quality from entry confirmation. The watchlist prepares Calls/Puts scenarios, target ladders, R:R tiers, and confirmation checklists; entry-confirmation scoring belongs to backtesting/trade review.</p>
</main></body></html>"""


def generate_zone_inventory(zones_df: pd.DataFrame) -> str:
    if zones_df.empty:
        return '# Detected Zone Inventory\n\nNo zones detected.'
    lines = ['# Detected Zone Inventory', '']
    for symbol in sorted(zones_df['symbol'].unique()):
        lines.append(f'## {symbol}')
        sub = zones_df[zones_df['symbol'] == symbol].sort_values(['timeframe', 'zone_type', 'zone_bottom'])
        for _, z in sub.iterrows():
            lines.append(
                f"- **{z['timeframe']} {z['zone_type']}** {z['zone_bottom']:.2f}–{z['zone_top']:.2f} | "
                f"{z['pattern']} | freshness: {_freshness_label(z['freshness'], z['tests'])} | "
                f"departure: {z['departure_atr']} ATR | body/base: {z.get('departure_body_vs_base_body', 'n/a')}x | vol: {z['departure_volume_ratio']}x | base: {z['base_time']}"
            )
        lines.append('')
    return '\n'.join(lines)



def _split_rejection_reasons(value) -> list[str]:
    raw = str(value or '').strip()
    if not raw or raw in {'nan', 'None'}:
        return []
    return [x.strip() for x in raw.split(';') if x.strip()]


def _reason_label(reason: str) -> str:
    labels = {
        'rejected_low_rr_to_nearest_opposing_zone': 'Low T1 R:R / nearest zone too close',
        'rejected_grade_below_final_min': 'Setup grade below final minimum',
        'rejected_broken_entry_zone': 'Entry zone already broken',
        'rejected_4plus_entry_tests': 'Entry zone has 4+ tests',
        'rejected_entry_tests_above_research_limit': 'Entry zone above research test limit',
        'research_only_3_tests': '3-test zone / research only',
        'rejected_2_tests_without_confluence_or_quality': '2-test zone lacks confluence/quality',
        'rejected_low_target_quality': 'Target quality too low',
        'rejected_no_valid_target': 'No valid target ladder',
        'eligible_pre_grade': 'Eligible before grade filter',
    }
    return labels.get(str(reason), str(reason).replace('_', ' ').title())


def _reason_explanation(reason: str) -> str:
    explanations = {
        'rejected_low_rr_to_nearest_opposing_zone': 'The scenario may be technically interesting, but the nearest unbroken opposing zone is too close to justify the modeled risk/reward. It may still be useful as watch-only or developing if price improves.',
        'rejected_grade_below_final_min': 'The zone/scenario did not reach the required A/A+ setup quality threshold after freshness, confluence, target quality, distance, and R:R were considered.',
        'rejected_broken_entry_zone': 'The source zone has already been invalidated by price action, so it should not be used as a fresh decision area.',
        'rejected_4plus_entry_tests': 'The zone has been tested too many times. Even if price reacts there again, it is no longer considered a clean final-list setup.',
        'research_only_3_tests': 'The zone may matter as map context, but it is too tested for the final watchlist.',
        'rejected_2_tests_without_confluence_or_quality': 'Two-test zones need extra evidence, such as higher-timeframe confluence or strong quality, before they can qualify.',
        'rejected_low_target_quality': 'The target ladder exists, but the nearest usable opposing target is weak or stale.',
        'rejected_no_valid_target': 'No acceptable unbroken opposing zone was found in the trade path, so a clean R:R ladder could not be built.',
    }
    return explanations.get(str(reason), 'This reason came from the candidate diagnostic rules in watchlist_all_candidates.csv.')


def _rejection_summary_html(all_watch_df: pd.DataFrame, final_watch_df: pd.DataFrame, meta: dict) -> str:
    report_date = meta.get('report_date', '')
    report_dt = meta.get('report_datetime', '')
    if all_watch_df is None or all_watch_df.empty:
        total = final_count = rejected_count = 0
        body = '<h2>No candidates</h2><p>No watchlist candidates were generated.</p>'
    else:
        df = all_watch_df.copy()
        if 'final_eligible' not in df.columns:
            df['final_eligible'] = False
        total = len(df)
        final_count = int(df['final_eligible'].astype(bool).sum())
        rejected_count = total - final_count

        reason_rows = []
        for _, row in df[~df['final_eligible'].astype(bool)].iterrows():
            reasons = _split_rejection_reasons(row.get('rejection_reasons', ''))
            if not reasons:
                reasons = ['no_rejection_reason_recorded']
            for reason in reasons:
                reason_rows.append({
                    'reason': reason,
                    'label': _reason_label(reason),
                    'symbol': row.get('symbol', ''),
                    'scenario': row.get('scenario_label', row.get('setup', '')),
                    'bucket': row.get('watchlist_bucket', ''),
                    'grade': row.get('setup_quality_grade', row.get('grade', '')),
                    'rr': row.get('estimated_rr', None),
                    'distance_pct': row.get('distance_pct', None),
                })
        reasons_df = pd.DataFrame(reason_rows)
        if reasons_df.empty:
            reason_table = '<p>No rejected candidates found.</p>'
        else:
            counts = reasons_df.groupby(['reason','label'], dropna=False).size().reset_index(name='count').sort_values('count', ascending=False)
            reason_table_rows = []
            for _, r in counts.iterrows():
                reason_table_rows.append(
                    f"<tr><td><strong>{_html_escape(r['label'])}</strong><br><small>{_html_escape(r['reason'])}</small></td>"
                    f"<td class='num'>{int(r['count'])}</td>"
                    f"<td>{_html_escape(_reason_explanation(r['reason']))}</td></tr>"
                )
            reason_table = "<table><thead><tr><th>Rejection reason</th><th>Count</th><th>What it means</th></tr></thead><tbody>" + ''.join(reason_table_rows) + '</tbody></table>'

        if 'watchlist_bucket' in df.columns:
            bucket_counts = df.groupby('watchlist_bucket', dropna=False).size().reset_index(name='count')
        else:
            bucket_counts = pd.DataFrame()
        bucket_rows = ''.join(
            f"<tr><td>{_html_escape(r['watchlist_bucket'])}</td><td class='num'>{int(r['count'])}</td></tr>" for _, r in bucket_counts.iterrows()
        ) or '<tr><td colspan="2">No bucket data</td></tr>'

        symbol_counts = df[~df['final_eligible'].astype(bool)].groupby('symbol', dropna=False).size().reset_index(name='rejected').sort_values('rejected', ascending=False).head(20)
        sym_rows = ''.join(
            f"<tr><td>{_html_escape(r['symbol'])}</td><td class='num'>{int(r['rejected'])}</td></tr>" for _, r in symbol_counts.iterrows()
        ) or '<tr><td colspan="2">No rejected symbols</td></tr>'

        df['_rr_num'] = pd.to_numeric(df.get('estimated_rr'), errors='coerce')
        df['_score_num'] = pd.to_numeric(df.get('setup_quality_score'), errors='coerce')
        near = df[~df['final_eligible'].astype(bool)].copy()
        near = near.sort_values(['_score_num','_rr_num','distance_pct'], ascending=[False, False, True]).head(25)
        near_rows = []
        for _, r in near.iterrows():
            near_rows.append(
                f"<tr><td><strong>{_html_escape(r.get('symbol',''))}</strong><br><small>{_html_escape(r.get('scenario_label', r.get('setup','')))}</small></td>"
                f"<td>{_html_escape(r.get('watchlist_bucket',''))}</td>"
                f"<td>{_html_escape(r.get('setup_quality_grade', r.get('grade','')))}<br><small>{float(r.get('_score_num') or 0):.2f}/10</small></td>"
                f"<td>{_rr_text(r.get('estimated_rr'))}<br><small>{_html_escape(r.get('rr_tier',''))}</small></td>"
                f"<td>{_fmt_pct(r.get('distance_pct'))}</td>"
                f"<td>{_html_escape(r.get('freshness_label',''))}</td>"
                f"<td>{_html_escape(str(r.get('target_1_zone','')))}<br><small>{_html_escape(r.get('rr_range_summary',''))}</small></td>"
                f"<td>{_html_escape(str(r.get('rejection_reasons','')).replace(';', '; '))}</td></tr>"
            )
        near_table = "<table><thead><tr><th>Candidate</th><th>Bucket</th><th>Grade</th><th>T1 R:R</th><th>Distance</th><th>Freshness</th><th>T1 / range</th><th>Why not final?</th></tr></thead><tbody>" + ''.join(near_rows) + '</tbody></table>' if near_rows else '<p>No near-miss candidates.</p>'

        body = f"""
        <section class="stats">
          <div class="stat"><div class="label">All candidates</div><div class="value">{total}</div></div>
          <div class="stat"><div class="label">Final setups</div><div class="value">{final_count}</div></div>
          <div class="stat"><div class="label">Not final</div><div class="value">{rejected_count}</div></div>
          <div class="stat"><div class="label">Final pass rate</div><div class="value">{(final_count / total * 100 if total else 0):.1f}%</div></div>
        </section>
        <h2>Why candidates did not make the final watchlist</h2>
        {reason_table}
        <div class="two-col">
          <section><h2>Bucket counts</h2><table><thead><tr><th>Bucket</th><th>Count</th></tr></thead><tbody>{bucket_rows}</tbody></table></section>
          <section><h2>Most rejected symbols</h2><table><thead><tr><th>Symbol</th><th>Rejected candidates</th></tr></thead><tbody>{sym_rows}</tbody></table></section>
        </div>
        <h2>Highest-quality near misses</h2>
        <p>These are the candidates to inspect first when tuning. They had the best setup scores but missed the final watchlist because of R:R, target quality, zone freshness/tests, or grade threshold.</p>
        {near_table}
        """

    style = """
    <style>
      body{font-family:Inter,Segoe UI,Arial,sans-serif;background:#0b1020;color:#e9eefc;margin:0;padding:24px}
      .wrap{max-width:1200px;margin:0 auto}.sub{color:#aeb8d6;margin-bottom:20px;line-height:1.5}
      h1{margin:0 0 8px;font-size:30px}h2{margin-top:28px;color:#ffffff}
      .stats{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px;margin:18px 0}.stat{background:#121a33;border:1px solid #25345f;border-radius:14px;padding:14px}.label{color:#9faad0;font-size:12px;text-transform:uppercase}.value{font-size:28px;font-weight:800;margin-top:4px}
      table{width:100%;border-collapse:collapse;background:#121a33;border:1px solid #25345f;border-radius:12px;overflow:hidden;margin:12px 0 20px}th,td{border-bottom:1px solid #25345f;padding:10px;text-align:left;vertical-align:top}th{background:#182344;color:#dbe5ff}.num{text-align:right;font-weight:800}small{color:#aeb8d6}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px}p{color:#cbd5f4;line-height:1.5}
      @media(max-width:800px){.stats,.two-col{grid-template-columns:1fr}}
    </style>
    """
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Watchlist Rejection Summary {report_date}</title>{style}</head><body><main class="wrap"><h1>Watchlist Rejection Summary — {report_date}</h1><div class="sub">Generated: {report_dt} America/New_York<br>This report explains why scenario candidates stayed out of the strict final watchlist. Use it with watchlist_all_candidates.csv when tuning thresholds.</div>{body}<p class="sub">v0.30 watchlist reset: final watchlist inclusion is based on setup quality, zone context, valid T1 target ladder, and modeled R:R. Entry-confirmation scoring is intentionally separate from watchlist inclusion.</p></main></body></html>"""

def main():
    parser = argparse.ArgumentParser(description="Generate the scenario-prep watchlist.")
    parser.add_argument("--as-of-date", default=None, help="Use only regular-session data through YYYY-MM-DD 16:00 New York time.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)
    archive_dir = REPORT_DIR / 'archive'
    archive_dir.mkdir(exist_ok=True)
    report_now, report_date, report_stamp = _report_datestamps()
    zones_df, watch_df, meta = build_watchlist(as_of_date=args.as_of_date)
    meta['report_date'] = report_date
    meta['report_datetime'] = report_now.strftime('%Y-%m-%d %I:%M %p %Z')

    zones_path = REPORT_DIR / 'detected_zones.csv'
    active_zones_path = REPORT_DIR / 'active_zones.csv'
    merged_zones_path = REPORT_DIR / 'merged_zones.csv'
    all_watch_path = REPORT_DIR / 'watchlist_all_candidates.csv'
    scenario_watch_path = REPORT_DIR / 'scenario_watchlist.csv'
    zone_map_path = REPORT_DIR / 'zone_map.csv'
    watch_path = REPORT_DIR / 'watchlist.csv'
    md_path = REPORT_DIR / 'watchlist.md'
    html_path = REPORT_DIR / 'watchlist.html'
    rejection_html_path = REPORT_DIR / 'watchlist_rejections.html'
    visual_zone_map_path = REPORT_DIR / 'watchlist_zone_map.html'
    visual_zone_map_csv_path = REPORT_DIR / 'watchlist_zone_map.csv'
    movement_context_path = REPORT_DIR / 'movement_context_watchlist.html'
    movement_context_csv_path = REPORT_DIR / 'movement_context_watchlist.csv'
    html_archive_path = archive_dir / f'watchlist_{report_stamp}.html'
    rejection_html_archive_path = archive_dir / f'watchlist_rejections_{report_stamp}.html'
    visual_zone_map_archive_path = archive_dir / f'watchlist_zone_map_{report_stamp}.html'
    movement_context_archive_path = archive_dir / f'movement_context_watchlist_{report_stamp}.html'
    md_archive_path = archive_dir / f'watchlist_{report_stamp}.md'
    csv_archive_path = archive_dir / f'watchlist_{report_stamp}.csv'
    inventory_path = REPORT_DIR / 'detected_zones.md'

    final_watch_df = _filter_final_report(watch_df)
    active_zones_df = _active_zones_for_watchlist(zones_df)
    merged_zones_df = merge_overlapping_zones(active_zones_df)

    zones_df.to_csv(zones_path, index=False)
    active_zones_df.to_csv(active_zones_path, index=False)
    merged_zones_df.to_csv(merged_zones_path, index=False)
    watch_df.to_csv(all_watch_path, index=False)
    watch_df.to_csv(movement_context_csv_path, index=False)
    if 'watchlist_bucket' in watch_df.columns:
        watch_df[watch_df['watchlist_bucket'].isin(['Final / Actionable', 'Developing Scenario'])].to_csv(scenario_watch_path, index=False)
        watch_df[watch_df['watchlist_bucket'].eq('Zone Map / Research')].to_csv(zone_map_path, index=False)
    else:
        watch_df.to_csv(scenario_watch_path, index=False)
        pd.DataFrame().to_csv(zone_map_path, index=False)
    final_watch_df.to_csv(watch_path, index=False)
    final_watch_df.to_csv(csv_archive_path, index=False)
    final_watch_df.to_csv(visual_zone_map_csv_path, index=False)
    md_report = generate_markdown(final_watch_df, meta)
    html_report = generate_html_report(final_watch_df, watch_df, meta)
    rejection_html_report = _rejection_summary_html(watch_df, final_watch_df, meta)
    visual_zone_map_html = generate_watchlist_zone_map_html(final_watch_df, meta)
    movement_context_html = generate_movement_context_html(watch_df, meta)
    md_path.write_text(md_report, encoding='utf-8')
    md_archive_path.write_text(md_report, encoding='utf-8')
    html_path.write_text(html_report, encoding='utf-8')
    rejection_html_path.write_text(rejection_html_report, encoding='utf-8')
    visual_zone_map_path.write_text(visual_zone_map_html, encoding='utf-8')
    movement_context_path.write_text(movement_context_html, encoding='utf-8')
    html_archive_path.write_text(html_report, encoding='utf-8')
    rejection_html_archive_path.write_text(rejection_html_report, encoding='utf-8')
    visual_zone_map_archive_path.write_text(visual_zone_map_html, encoding='utf-8')
    movement_context_archive_path.write_text(movement_context_html, encoding='utf-8')
    inventory_path.write_text(generate_zone_inventory(zones_df), encoding='utf-8')

    print(f'Wrote {zones_path}')
    print(f'Wrote {active_zones_path}')
    print(f'Wrote {merged_zones_path}')
    print(f'Wrote {all_watch_path}')
    print(f'Wrote {scenario_watch_path}')
    print(f'Wrote {zone_map_path}')
    print(f'Wrote {watch_path}')
    print(f'Wrote {md_path}')
    print(f'Wrote {html_path}')
    print(f'Wrote {rejection_html_path}')
    print(f'Wrote {visual_zone_map_path}')
    print(f'Wrote {visual_zone_map_csv_path}')
    print(f'Wrote {movement_context_path}')
    print(f'Wrote {movement_context_csv_path}')
    print(f'Wrote {html_archive_path}')
    print(f'Wrote {rejection_html_archive_path}')
    print(f'Wrote {visual_zone_map_archive_path}')
    print(f'Wrote {movement_context_archive_path}')
    print(f'Wrote {inventory_path}')
    print(f'All candidates: {len(watch_df)}')
    print(f'Final report candidates: {len(final_watch_df)}')
    print(f"Price source: {meta.get('price_source')}")


if __name__ == '__main__':
    main()
