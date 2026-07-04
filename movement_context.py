"""Movement, gap, volume, and historical-zone context for the watchlist.

v0.37.1 add-on. These helpers are intentionally informational/scoring
context. They do not change RTH zone creation rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import time
import math
import html
import re
import pandas as pd

from data_loader import MARKET_TZ


def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_money(x) -> str:
    v = safe_float(x)
    return "n/a" if v is None else f"${v:,.2f}"


def format_pct(x) -> str:
    v = safe_float(x)
    return "n/a" if v is None else f"{v:.2f}%"


def esc(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return html.escape(str(x))



def _clean_ts_input(ts):
    if isinstance(ts, str):
        text = ts.strip()
        # Pandas does not reliably parse abbreviations like EDT/EST. Strip them
        # and localize below to America/New_York for display/session labels.
        text = re.sub(r'\s+(EDT|EST|ET)$', '', text, flags=re.IGNORECASE)
        return text
    return ts


def classify_market_session(ts) -> str:
    """Classify a timestamp into Premarket/RTH/Aftermarket/Closed."""
    try:
        t = pd.Timestamp(_clean_ts_input(ts))
        if pd.isna(t):
            return "Unknown"
        if t.tzinfo is None:
            t = t.tz_localize(MARKET_TZ)
        else:
            t = t.tz_convert(MARKET_TZ)
        tod = t.time()
        if time(9, 30) <= tod < time(16, 0):
            return "RTH"
        if time(4, 0) <= tod < time(9, 30):
            return "Premarket"
        if time(16, 0) <= tod < time(20, 0):
            return "Aftermarket"
        return "Closed / overnight"
    except Exception:
        return "Unknown"


def _fmt_ts(ts) -> str:
    try:
        t = pd.Timestamp(_clean_ts_input(ts))
        if pd.isna(t):
            return ""
        if t.tzinfo is None:
            t = t.tz_localize(MARKET_TZ)
        else:
            t = t.tz_convert(MARKET_TZ)
        return t.strftime("%Y-%m-%d %I:%M %p %Z")
    except Exception:
        return str(ts) if ts is not None else ""


def compute_symbol_movement_context(symbol: str, raw_5m: pd.DataFrame, rth_5m: pd.DataFrame, current_price: float | None = None, current_asof=None) -> dict:
    """Compute symbol-level movement, gap, EMA/VWAP, and volume context.

    raw_5m may include extended-hours bars. rth_5m must be regular-session only.
    Zones remain RTH-only; this context is descriptive for watchlist ranking.
    """
    out = {
        "latest_price_session": "Unknown",
        "latest_price_time": "",
        "prior_rth_close": None,
        "gap_abs": None,
        "gap_pct": None,
        "gap_direction": "unknown",
        "gap_size_bucket": "unknown",
        "recent_move_direction": "unknown",
        "recent_move_pct": None,
        "recent_move_strength": "unknown",
        "price_vs_9ema": "unknown",
        "price_vs_vwap": "unknown",
        "ema9": None,
        "session_vwap": None,
        "volume_ratio_recent": None,
        "volume_state": "unknown",
        "vpa_state": "unknown",
    }
    if raw_5m is None or raw_5m.empty or rth_5m is None or rth_5m.empty:
        return out

    raw_local = raw_5m.tz_convert(MARKET_TZ) if getattr(raw_5m.index, 'tz', None) is not None else raw_5m.copy()
    rth = rth_5m.copy()
    if getattr(rth.index, 'tz', None) is None:
        rth = rth.tz_localize(MARKET_TZ)
    else:
        rth = rth.tz_convert(MARKET_TZ)

    if current_price is None:
        current_price = safe_float(raw_local['close'].iloc[-1])
    out['latest_price_time'] = _fmt_ts(current_asof if current_asof is not None else raw_local.index[-1])
    out['latest_price_session'] = classify_market_session(current_asof if current_asof is not None else raw_local.index[-1])

    # Prior/anchor RTH close is the latest completed RTH close known in the file.
    prior_close = safe_float(rth['close'].iloc[-1])
    out['prior_rth_close'] = prior_close
    if current_price is not None and prior_close:
        gap_abs = float(current_price) - float(prior_close)
        gap_pct = gap_abs / max(abs(float(prior_close)), 0.01) * 100
        out['gap_abs'] = round(gap_abs, 3)
        out['gap_pct'] = round(gap_pct, 3)
        if gap_pct >= 0.75:
            out['gap_direction'] = 'Gap up'
        elif gap_pct <= -0.75:
            out['gap_direction'] = 'Gap down'
        else:
            out['gap_direction'] = 'Flat / no meaningful gap'
        ag = abs(gap_pct)
        if ag >= 3:
            out['gap_size_bucket'] = 'large_gap'
        elif ag >= 1.5:
            out['gap_size_bucket'] = 'moderate_gap'
        elif ag >= 0.75:
            out['gap_size_bucket'] = 'small_gap'
        else:
            out['gap_size_bucket'] = 'flat'

    recent = rth.tail(24).copy()
    if len(recent) >= 6 and current_price is not None:
        anchor = safe_float(recent['close'].iloc[-min(12, len(recent))])
        if anchor:
            mv = (float(current_price) - anchor) / max(abs(anchor), 0.01) * 100
            out['recent_move_pct'] = round(mv, 3)
            if mv >= 0.5:
                out['recent_move_direction'] = 'Bullish'
            elif mv <= -0.5:
                out['recent_move_direction'] = 'Bearish'
            else:
                out['recent_move_direction'] = 'Sideways'
            amv = abs(mv)
            if amv >= 2.0:
                out['recent_move_strength'] = 'impulsive'
            elif amv >= 0.75:
                out['recent_move_strength'] = 'directional'
            elif amv >= 0.25:
                out['recent_move_strength'] = 'grinding'
            else:
                out['recent_move_strength'] = 'sideways'

    close = safe_float(current_price)
    if close is not None and len(rth) >= 9:
        ema9 = rth['close'].ewm(span=9, adjust=False).mean().iloc[-1]
        out['ema9'] = round(float(ema9), 3)
        out['price_vs_9ema'] = 'above_9ema' if close >= float(ema9) else 'below_9ema'
    if close is not None and 'vwap' in rth.columns and not rth['vwap'].dropna().empty:
        # Prefer source VWAP on the latest RTH candle when present.
        vwap = safe_float(rth['vwap'].dropna().iloc[-1])
        if vwap is not None:
            out['session_vwap'] = round(vwap, 3)
            out['price_vs_vwap'] = 'above_vwap' if close >= vwap else 'below_vwap'
    elif close is not None and 'volume' in rth.columns and len(rth) > 2:
        vol = pd.to_numeric(rth['volume'], errors='coerce').fillna(0)
        vwap = ((rth['close'] * vol).cumsum() / vol.cumsum().replace(0, pd.NA)).iloc[-1]
        if not pd.isna(vwap):
            out['session_vwap'] = round(float(vwap), 3)
            out['price_vs_vwap'] = 'above_vwap' if close >= float(vwap) else 'below_vwap'

    if len(rth) >= 30:
        recent_vol = pd.to_numeric(rth['volume'].tail(3), errors='coerce').mean()
        avg_vol = pd.to_numeric(rth['volume'].tail(30), errors='coerce').mean()
        if avg_vol and not pd.isna(avg_vol):
            ratio = float(recent_vol) / float(avg_vol)
            out['volume_ratio_recent'] = round(ratio, 3)
            if ratio >= 1.75:
                out['volume_state'] = 'high_volume'
            elif ratio >= 1.20:
                out['volume_state'] = 'elevated_volume'
            elif ratio < 0.75:
                out['volume_state'] = 'low_volume'
            else:
                out['volume_state'] = 'normal_volume'

            bodies = (rth['close'] - rth['open']).abs().tail(3)
            ranges = (rth['high'] - rth['low']).replace(0, pd.NA).tail(3)
            body_ratio = safe_float((bodies / ranges).mean())
            if ratio >= 1.5 and body_ratio is not None and body_ratio <= 0.35:
                out['vpa_state'] = 'absorption_possible'
            elif ratio >= 1.2 and body_ratio is not None and body_ratio >= 0.60:
                out['vpa_state'] = 'volume_expansion_with_direction'
            elif ratio < 0.75 and out['recent_move_direction'] in ['Bullish', 'Bearish']:
                out['vpa_state'] = 'low_volume_move_warning'
            else:
                out['vpa_state'] = out['volume_state']
    return out


def _historical_summary_from_row(row: pd.Series, history: dict) -> tuple[str, float]:
    sym = str(row.get('symbol', '')).upper()
    zone_type = str(row.get('zone_type', '')).lower()
    side = str(row.get('side', '')).lower()
    h = history.get(sym, {}) if isinstance(history, dict) else {}
    if not h:
        return 'No historical zone-reaction audit yet', 0.0
    if zone_type == 'supply':
        rej = safe_float(h.get('supply_rejection_rate_pct'), 0.0) or 0.0
        brk = safe_float(h.get('supply_break_rate_pct'), 0.0) or 0.0
        n = int(safe_float(h.get('supply_events'), 0) or 0)
        text = f"Supply history: rejects {rej:.0f}%, breaks {brk:.0f}% ({n} tests)"
        score = (brk - rej) / 5.0 if side == 'long' else (rej - brk) / 5.0
        return text, max(-15, min(15, score))
    if zone_type == 'demand':
        hold = safe_float(h.get('demand_hold_rate_pct'), 0.0) or 0.0
        brk = safe_float(h.get('demand_break_rate_pct'), 0.0) or 0.0
        n = int(safe_float(h.get('demand_events'), 0) or 0)
        text = f"Demand history: holds {hold:.0f}%, breaks {brk:.0f}% ({n} tests)"
        score = (hold - brk) / 5.0 if side == 'long' else (brk - hold) / 5.0
        return text, max(-15, min(15, score))
    return 'No matching zone-history context', 0.0


def load_zone_reaction_history(report_dir: Path) -> dict:
    """Load symbol-level zone tendency summaries, if the zone audit has been run."""
    candidates = [
        report_dir / 'zone_reaction_by_symbol.csv',
        report_dir / 'zone_reaction_summary_by_symbol.csv',
        report_dir / 'backtest' / 'analytics' / 'zone_reaction_by_symbol.csv',
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if 'symbol' not in df.columns:
                continue
            out = {}
            for _, r in df.iterrows():
                out[str(r['symbol']).upper()] = r.to_dict()
            if out:
                return out
        except Exception:
            continue
    return {}


def _gap_zone_label(row: pd.Series) -> str:
    gap = safe_float(row.get('gap_pct'))
    if gap is None:
        return 'gap_unknown'
    price = safe_float(row.get('current_price'))
    prior = safe_float(row.get('prior_rth_close'))
    if price is None or prior is None:
        return 'gap_unknown'
    d_bot, d_top = safe_float(row.get('closest_demand_bottom')), safe_float(row.get('closest_demand_top'))
    s_bot, s_top = safe_float(row.get('closest_supply_bottom')), safe_float(row.get('closest_supply_top'))
    if s_bot is not None and s_top is not None:
        if prior < s_bot and price > s_top:
            return 'gap_through_supply'
        if s_bot <= price <= s_top:
            return 'gap_into_supply'
        if price > s_top and prior <= s_top:
            return 'gap_above_supply'
    if d_bot is not None and d_top is not None:
        if prior > d_top and price < d_bot:
            return 'gap_through_demand'
        if d_bot <= price <= d_top:
            return 'gap_into_demand'
        if price < d_bot and prior >= d_bot:
            return 'gap_below_demand'
    if abs(gap) < 0.75:
        return 'no_meaningful_gap'
    return 'gap_between_zones'


def _zone_movement_state(row: pd.Series) -> str:
    zt = str(row.get('zone_type', '')).lower()
    side = str(row.get('side', '')).lower()
    move = str(row.get('recent_move_direction', 'unknown'))
    strength = str(row.get('recent_move_strength', 'unknown'))
    price = safe_float(row.get('current_price'))
    top, bottom = safe_float(row.get('zone_top')), safe_float(row.get('zone_bottom'))
    if price is None or top is None or bottom is None:
        return 'unknown_movement_state'
    inside = bottom <= price <= top
    if zt == 'supply':
        if price > top:
            return 'breaking_above_supply' if side == 'long' else 'supply_broken_risk'
        if inside:
            return 'testing_supply'
        if move == 'Bullish':
            return 'approaching_supply_with_strength' if strength in ['directional', 'impulsive'] else 'approaching_supply_grinding'
        if move == 'Bearish':
            return 'rejecting_from_supply'
        return 'below_supply_no_trigger'
    if zt == 'demand':
        if price < bottom:
            return 'breaking_below_demand' if side == 'short' else 'demand_broken_risk'
        if inside:
            return 'testing_demand'
        if move == 'Bearish':
            return 'approaching_demand_with_weakness' if strength in ['directional', 'impulsive'] else 'approaching_demand_grinding'
        if move == 'Bullish':
            return 'bouncing_from_demand'
        return 'above_demand_no_trigger'
    return 'unknown_movement_state'


def _zone_thesis(row: pd.Series) -> str:
    scenario = str(row.get('scenario', ''))
    mapping = {
        'supply_break': 'Supply Breakout Watch',
        'supply_reject': 'Supply Rejection Watch',
        'demand_hold': 'Demand Hold/Reversal Watch',
        'demand_break': 'Demand Breakdown Watch',
    }
    return mapping.get(scenario, str(row.get('scenario_label', row.get('setup', 'Zone Watch'))))


def _alignment_score(row: pd.Series) -> tuple[float, str]:
    side = str(row.get('bias', ''))
    s = str(row.get('structure_trade_alignment', ''))
    if s == 'With structure':
        return 18, 'structure aligned'
    if s == 'Counter-structure':
        return -8, 'counter-structure'
    return 0, 'structure mixed'


def enrich_movement_context(watch_df: pd.DataFrame, symbol_context: dict, zone_history: dict | None = None) -> pd.DataFrame:
    if watch_df is None or watch_df.empty:
        return watch_df
    out = watch_df.copy()
    rows = []
    for _, row in out.iterrows():
        sym = str(row.get('symbol', '')).upper()
        sc = symbol_context.get(sym, {}) if isinstance(symbol_context, dict) else {}
        merged = {}
        for k, v in sc.items():
            if isinstance(v, (str, int, float, type(None))):
                merged[k] = v
        # Do not overwrite explicit watchlist timestamp if already present.
        if not str(row.get('current_price_as_of', '')).strip():
            merged['current_price_as_of'] = sc.get('latest_price_time', '')
        if 'current_price_session' not in merged:
            merged['current_price_session'] = sc.get('latest_price_session', '')
        merged['gap_zone_context'] = _gap_zone_label(pd.concat([row, pd.Series(merged)]))
        rows.append(merged)
    ctx = pd.DataFrame(rows, index=out.index)
    out = out.drop(columns=[c for c in ctx.columns if c in out.columns], errors='ignore')
    out = pd.concat([out, ctx], axis=1)

    history = zone_history or {}
    states = []
    theses = []
    hist_texts = []
    hist_scores = []
    scores = []
    reasons = []
    watch_for = []
    for _, r in out.iterrows():
        state = _zone_movement_state(r)
        thesis = _zone_thesis(r)
        hist_text, hist_score = _historical_summary_from_row(r, history)
        align_score, align_reason = _alignment_score(r)
        score = 50.0
        reason_parts = [align_reason]
        # Movement-state scoring.
        if state in ['approaching_supply_with_strength', 'breaking_above_supply', 'approaching_demand_with_weakness', 'breaking_below_demand', 'rejecting_from_supply', 'bouncing_from_demand']:
            score += 14
            reason_parts.append(state.replace('_', ' '))
        elif 'grinding' in state or 'testing' in state:
            score += 6
            reason_parts.append(state.replace('_', ' '))
        elif 'risk' in state:
            score -= 6
            reason_parts.append(state.replace('_', ' '))
        score += align_score
        # Volume / VPA.
        vpa = str(r.get('vpa_state', 'unknown'))
        if vpa == 'volume_expansion_with_direction':
            score += 12
            reason_parts.append('volume expansion')
        elif vpa == 'absorption_possible':
            score += 8
            reason_parts.append('possible absorption')
        elif vpa == 'low_volume_move_warning':
            score -= 6
            reason_parts.append('low-volume move warning')
        # Gap context.
        gap_ctx = str(r.get('gap_zone_context', ''))
        gap_dir = str(r.get('gap_direction', ''))
        if gap_ctx in ['gap_through_supply', 'gap_through_demand']:
            score += 8
            reason_parts.append(gap_ctx.replace('_', ' '))
        elif gap_ctx in ['gap_into_supply', 'gap_into_demand']:
            score += 5
            reason_parts.append(gap_ctx.replace('_', ' '))
        if gap_dir in ['Gap up', 'Gap down']:
            reason_parts.append(f"{gap_dir.lower()} {format_pct(r.get('gap_pct'))}")
        # Historical reaction context.
        score += hist_score
        if hist_text and 'No historical' not in hist_text:
            reason_parts.append(hist_text)
        score = max(0, min(100, round(score, 1)))
        states.append(state)
        theses.append(thesis)
        hist_texts.append(hist_text)
        hist_scores.append(round(hist_score, 2))
        scores.append(score)
        reasons.append('; '.join(dict.fromkeys([p for p in reason_parts if p])))
        watch_for.append(_watch_for_text(pd.concat([r, pd.Series({'zone_movement_state': state, 'zone_thesis': thesis})])))
    out['zone_thesis'] = theses
    out['zone_movement_state'] = states
    out['historical_zone_tendency'] = hist_texts
    out['historical_reaction_score'] = hist_scores
    out['observation_score'] = scores
    out['observation_reason'] = reasons
    out['watch_for'] = watch_for
    out['movement_watchlist_bucket'] = pd.cut(
        pd.to_numeric(out['observation_score'], errors='coerce').fillna(0),
        bins=[-0.1, 54.9, 69.9, 100.1],
        labels=['Research / monitor only', 'Developing movement watch', 'Priority movement watch']
    ).astype(str)
    return out


def _watch_for_text(row: pd.Series) -> str:
    thesis = str(row.get('zone_thesis', ''))
    if thesis == 'Supply Breakout Watch':
        return 'Watch for a strong RTH close above supply, opening-volume confirmation, and hold/retest above the zone.'
    if thesis == 'Supply Rejection Watch':
        return 'Watch for failure to close above supply, upper-wick/absorption behavior, then bearish follow-through away from the zone.'
    if thesis == 'Demand Hold/Reversal Watch':
        return 'Watch for demand tap, absorption or lower-wick rejection, reclaim of demand top, then bullish follow-through.'
    if thesis == 'Demand Breakdown Watch':
        return 'Watch for strong RTH close below demand, opening-volume confirmation, and hold/retest below the zone.'
    return 'Watch for clean RTH confirmation before treating this as actionable.'


def generate_movement_context_html(watch_df: pd.DataFrame, meta: dict) -> str:
    report_date = esc(meta.get('report_date', ''))
    report_dt = esc(meta.get('report_datetime', ''))
    if watch_df is None or watch_df.empty:
        cards = '<p>No movement-context candidates available.</p>'
    else:
        df = watch_df.copy()
        if 'observation_score' in df.columns:
            df['_obs'] = pd.to_numeric(df['observation_score'], errors='coerce').fillna(0)
            df = df.sort_values(['_obs', 'setup_quality_score'], ascending=[False, False]).head(60)
        parts = []
        for _, r in df.iterrows():
            score = safe_float(r.get('observation_score'), 0) or 0
            score_class = 'high' if score >= 70 else ('mid' if score >= 55 else 'low')
            gap_line = f"{esc(r.get('gap_direction',''))} {format_pct(r.get('gap_pct'))} • {esc(str(r.get('gap_zone_context','')).replace('_',' '))}"
            px_session = esc(r.get('current_price_session', r.get('latest_price_session','')))
            parts.append(f"""
            <article class='card {score_class}'>
              <div class='head'>
                <div><h2>{esc(r.get('symbol',''))} — {esc(r.get('zone_thesis', r.get('scenario_label','')))}</h2>
                <p>{esc(r.get('option_contract',''))} • Grade {esc(r.get('setup_quality_grade', r.get('grade','')))} • {esc(r.get('movement_watchlist_bucket',''))}</p></div>
                <div class='score'><span>Observation</span><strong>{score:.1f}</strong></div>
              </div>
              <div class='grid'>
                <div><span>Current price</span><strong>{format_money(r.get('current_price'))}</strong><small>as of {esc(r.get('current_price_as_of',''))} — {px_session}</small></div>
                <div><span>Movement state</span><strong>{esc(str(r.get('zone_movement_state','')).replace('_',' '))}</strong><small>{esc(r.get('recent_move_direction',''))} {format_pct(r.get('recent_move_pct'))} • {esc(r.get('recent_move_strength',''))}</small></div>
                <div><span>Structure</span><strong>{esc(r.get('structure_alignment',''))}</strong><small>{esc(r.get('structure_bias_5m',''))} / {esc(r.get('structure_bias_15m',''))}</small></div>
                <div><span>EMA / VWAP</span><strong>{esc(r.get('price_vs_9ema',''))} / {esc(r.get('price_vs_vwap',''))}</strong><small>9EMA {format_money(r.get('ema9'))}; VWAP {format_money(r.get('session_vwap'))}</small></div>
                <div><span>Volume / VPA</span><strong>{esc(r.get('vpa_state',''))}</strong><small>{esc(r.get('volume_state',''))}; recent vol {safe_float(r.get('volume_ratio_recent'), 0) or 0:.2f}x</small></div>
                <div><span>Gap context</span><strong>{gap_line}</strong><small>Prior RTH close {format_money(r.get('prior_rth_close'))}</small></div>
                <div><span>Zone</span><strong>{esc(r.get('timeframe',''))} {esc(r.get('zone_type',''))}</strong><small>{format_money(r.get('zone_bottom'))}–{format_money(r.get('zone_top'))}</small></div>
                <div><span>Historical tendency</span><strong>{esc(r.get('historical_zone_tendency',''))}</strong><small>Historical score {safe_float(r.get('historical_reaction_score'), 0) or 0:.1f}</small></div>
              </div>
              <p class='reason'><strong>Why:</strong> {esc(r.get('observation_reason',''))}</p>
              <p class='watch'><strong>Watch for:</strong> {esc(r.get('watch_for',''))}</p>
            </article>
            """)
        cards = ''.join(parts)
    style = """
    <style>
      :root{--bg:#0f172a;--panel:#111827;--panel2:#1f2937;--line:#374151;--text:#e5e7eb;--muted:#9ca3af;--good:#22c55e;--mid:#f59e0b;--low:#64748b;--blue:#38bdf8}
      body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,Helvetica,sans-serif}.wrap{max-width:1240px;margin:0 auto;padding:28px}h1{margin:0 0 6px}h2{margin:0}.muted,p{color:#cbd5e1;line-height:1.45}.card{background:var(--panel);border:1px solid var(--line);border-left:6px solid var(--low);border-radius:18px;padding:18px;margin:18px 0;box-shadow:0 8px 22px rgba(0,0,0,.22)}.card.high{border-left-color:var(--good)}.card.mid{border-left-color:var(--mid)}.head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}.head p{margin:5px 0 0;color:var(--muted)}.score{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:10px 14px;text-align:right}.score span{display:block;color:var(--muted);font-size:12px}.score strong{font-size:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:10px;margin-top:14px}.grid div{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:12px;padding:10px}.grid span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}.grid strong{display:block}.grid small{display:block;color:var(--muted);margin-top:3px}.reason,.watch{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-radius:12px;padding:10px 12px;margin:12px 0 0}.watch{background:rgba(34,197,94,.07);border-color:rgba(34,197,94,.22)}
    </style>
    """
    intro = "Movement context ranks candidates by what price is doing around mapped RTH zones: recent direction, structure, volume/VPA, extended-hours gap context, and historical zone behavior. Zones remain RTH-only; extended-hours prices are context only."
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Movement Context Watchlist {report_date}</title>{style}</head><body><main class='wrap'><h1>Movement Context Watchlist — {report_date}</h1><p class='muted'>Generated: {report_dt} America/New_York. {intro}</p>{cards}</main></body></html>"""
