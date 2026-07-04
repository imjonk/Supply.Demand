from __future__ import annotations

import argparse
from pathlib import Path
from datetime import time
import pandas as pd
import numpy as np

from config import DATA_DIR, REPORT_DIR
from data_loader import load_symbol_csv, regular_session_only


def _parse_symbols(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {s.strip().upper() for s in value.replace(';', ',').split(',') if s.strip()}


_SYMBOL_5M_CACHE: dict[str, pd.DataFrame] = {}


def _load_symbol_5m_all(symbol: str) -> pd.DataFrame:
    symbol = str(symbol).upper()
    if symbol in _SYMBOL_5M_CACHE:
        return _SYMBOL_5M_CACHE[symbol]
    path = DATA_DIR / f"{symbol}_5M.csv"
    if not path.exists():
        _SYMBOL_5M_CACHE[symbol] = pd.DataFrame()
        return _SYMBOL_5M_CACHE[symbol]
    _SYMBOL_5M_CACHE[symbol] = regular_session_only(load_symbol_csv(path)).copy()
    return _SYMBOL_5M_CACHE[symbol]


def _load_day_5m(symbol: str, day: str) -> pd.DataFrame:
    df_all = _load_symbol_5m_all(symbol)
    if df_all.empty:
        return pd.DataFrame()
    target = pd.to_datetime(day).date()
    df = df_all[df_all.index.date == target].copy()
    if df.empty:
        return df
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    if "vwap" not in df.columns or df["vwap"].isna().all():
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].replace(0, np.nan).cumsum()
    df["avg_volume20"] = df["volume"].rolling(20, min_periods=3).mean()
    df["volume_ratio"] = df["volume"] / df["avg_volume20"].replace(0, np.nan)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = df["body"] / df["range"]
    df["avg_range20"] = df["range"].rolling(20, min_periods=3).mean()
    df["range_ratio"] = df["range"] / df["avg_range20"].replace(0, np.nan)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=3).mean()
    return df



def _resolve_snapshot_file(manifest_row: pd.Series, *, snapshot_mode: str, use_final_only: bool) -> tuple[Path, str]:
    """Resolve the frozen watchlist snapshot file for one replay day.

    Replay must consume daily snapshot rows only. It must not create or discover
    new watchlist candidates intraday. This helper centralizes snapshot-file
    selection and keeps path fallback behavior for moved Windows projects.
    """
    if snapshot_mode == "preopen":
        scenario_col = "preopen_final_file" if use_final_only else "preopen_scenario_file"
        fallback_dir = REPORT_DIR / "backtest" / "preopen_snapshots"
    else:
        scenario_col = "final_file" if use_final_only else "scenario_file"
        fallback_dir = REPORT_DIR / "backtest" / "snapshots"

    scenario_file = Path(str(manifest_row.get(scenario_col, "")))
    if not scenario_file.exists():
        snap_name = str(scenario_file).replace("\\", "/").split("/")[-1]
        fallback = fallback_dir / snap_name
        scenario_file = fallback if fallback.exists() else scenario_file
    return scenario_file, scenario_col


def _prepare_snapshot_rows(scenarios: pd.DataFrame, *, day: str, scenario_file: Path, snapshot_mode: str) -> pd.DataFrame:
    """Validate and stamp snapshot rows before replay.

    This is a guardrail against accidentally replaying dynamically-created
    candidates. If older snapshot files lack candidate ids, assign deterministic
    legacy ids so every candidate/trade can still be traced to a row in the
    frozen watchlist file.
    """
    out = scenarios.copy()
    out["replay_snapshot_file"] = str(scenario_file)
    out["replay_snapshot_mode"] = snapshot_mode
    if "snapshot_candidate_id" not in out.columns:
        out["snapshot_candidate_id"] = [f"{day}|{snapshot_mode}|legacy_snapshot_row_{i}" for i in range(len(out))]
    if "snapshot_test_date" not in out.columns:
        out["snapshot_test_date"] = day
    return out

def _scenario_side(row) -> str:
    side = str(row.get("side", "")).lower()
    if side in {"long", "short"}:
        return side
    bias = str(row.get("bias", "")).lower()
    opt = str(row.get("option_contract", "")).lower()
    return "long" if "bull" in bias or "call" in opt else "short"


def _scenario_kind(row) -> str:
    s = str(row.get("scenario", "")).lower()
    label = str(row.get("scenario_label", "")).lower()
    full = s + " " + label
    if "supply" in full and "break" in full:
        return "supply_breakout"
    if "demand" in full and "break" in full:
        return "demand_breakdown"
    if "supply" in full and "reject" in full:
        return "supply_rejection"
    if "demand" in full and ("hold" in full or "reversal" in full or "bounce" in full):
        return "demand_reversal"
    zt = str(row.get("zone_type", ""))
    side = _scenario_side(row)
    if zt == "demand" and side == "long":
        return "demand_reversal"
    if zt == "demand" and side == "short":
        return "demand_breakdown"
    if zt == "supply" and side == "short":
        return "supply_rejection"
    return "supply_breakout"



def _safe_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _volume_bucket(volume_ratio: float) -> str:
    if not np.isfinite(volume_ratio):
        return "unknown"
    if volume_ratio < 0.80:
        return "low_volume"
    if volume_ratio < 1.20:
        return "normal_volume"
    if volume_ratio < 1.75:
        return "elevated_volume"
    return "high_volume"


def _close_location_vs_zone(close: float, bottom: float, top: float) -> str:
    if not all(np.isfinite(x) for x in [close, bottom, top]) or top <= bottom:
        return "unknown"
    if close > top:
        return "above_zone"
    if close < bottom:
        return "below_zone"
    pct = (close - bottom) / max(top - bottom, 1e-9)
    if pct >= 0.67:
        return "inside_upper_third"
    if pct >= 0.33:
        return "inside_middle_third"
    return "inside_lower_third"


def _score_bucket(score: int) -> str:
    if score >= 6:
        return "ideal_confirmation"
    if score >= 4:
        return "strong_confirmation"
    if score >= 2:
        return "weak_confirmation"
    return "no_real_confirmation"


def _time_of_day_bucket(ts: pd.Timestamp) -> str:
    t = ts.time()
    if t < time(10, 30):
        return "09:30-10:30 open"
    if t < time(12, 0):
        return "10:30-12:00 morning"
    if t < time(14, 0):
        return "12:00-14:00 midday"
    if t < time(15, 30):
        return "14:00-15:30 afternoon"
    return "15:30-16:00 close"


def _is_reversal_rejection_kind(kind: str) -> bool:
    return kind in {"demand_reversal", "supply_rejection"}


def _directional_followthrough(nxt: pd.Series, kind: str, ref_close: float, ref_high: float, ref_low: float) -> bool:
    close = _safe_float(nxt.get("close"))
    open_ = _safe_float(nxt.get("open"))
    high = _safe_float(nxt.get("high"))
    low = _safe_float(nxt.get("low"))
    if kind == "demand_reversal":
        return close > open_ and close > ref_close and high > ref_high
    if kind == "supply_rejection":
        return close < open_ and close < ref_close and low < ref_low
    return False


def _live_style_reversal_diagnostics(day_df: pd.DataFrame, scenario: pd.Series, ts: pd.Timestamp) -> dict:
    """Return live-style reversal/rejection diagnostics for the candidate/entry candle.

    These fields are diagnostic only. They do not change the baseline watchlist or baseline replay entry.
    The intent is to see whether reversal/rejection trades are being entered/managed differently than a live trader would.
    """
    kind = _scenario_kind(scenario)
    bottom = _safe_float(scenario.get("zone_bottom"))
    top = _safe_float(scenario.get("zone_top"))
    out = {
        "time_of_day_bucket": _time_of_day_bucket(ts),
        "entry_close_zone_location": "unknown",
        "entry_9ema_relation": "unknown",
        "entry_vwap_relation": "unknown",
        "entry_9ema_confirmed": False,
        "entry_vwap_confirmed": False,
        "zone_edge_tapped": False,
        "zone_edge_tap_depth_pct": np.nan,
        "reclaimed_demand_top": False,
        "rejected_below_supply_bottom": False,
        "has_boundary_reclaim_reject": False,
        "reaction_body_pct": np.nan,
        "reaction_range_vs_avg": np.nan,
        "reaction_close_strength": np.nan,
        "strong_instant_reaction": False,
        "entry_side_wick_pct": np.nan,
        "rejection_wick_pct": np.nan,
        "has_bad_entry_side_wick": False,
        "has_favorable_rejection_wick": False,
        "wick_quality_bucket": "unknown",
        "entry_volume_bucket": "unknown",
        "has_breakout_volume": False,
        "has_absorption_signature": False,
        "has_followthrough_volume": False,
        "vpa_confirmation_bucket": "unknown",
        "vpa_confirmed": False,
        "one_candle_followthrough": False,
        "two_candle_followthrough": False,
        "three_candle_followthrough": False,
        "breaks_9ema": False,
        "breaks_vwap": False,
        "ema_or_vwap_break": False,
        "structure_confirmation": False,
        "momentum_confirmation_bucket": "unknown",
        "live_confirmation_score": 0,
        "live_confirmation_bucket": "unknown",
        "has_1c_confirmation": False,
        "has_2c_confirmation": False,
        "backtest_realism_diagnosis": "not_reversal_rejection",
    }
    if not all(np.isfinite(x) for x in [bottom, top]) or top <= bottom or ts not in day_df.index:
        return out

    idx = day_df.index.get_loc(ts)
    if isinstance(idx, slice):
        idx = idx.start
    row = day_df.iloc[int(idx)]
    prev = day_df.iloc[int(idx) - 1] if int(idx) > 0 else row
    next3 = day_df.iloc[int(idx) + 1:int(idx) + 4]

    open_ = _safe_float(row.get("open"))
    high = _safe_float(row.get("high"))
    low = _safe_float(row.get("low"))
    close = _safe_float(row.get("close"))
    ema = _safe_float(row.get("ema9"))
    vwap = _safe_float(row.get("vwap"))
    volume_ratio = _safe_float(row.get("volume_ratio"))
    body = abs(close - open_)
    rng = max(high - low, 1e-9)
    body_pct = body / rng
    upper_wick = max(0.0, high - max(open_, close))
    lower_wick = max(0.0, min(open_, close) - low)
    upper_wick_pct = upper_wick / rng
    lower_wick_pct = lower_wick / rng
    avg_range = _safe_float(row.get("avg_range20"))
    range_vs_avg = rng / avg_range if np.isfinite(avg_range) and avg_range > 0 else _safe_float(row.get("range_ratio"))
    close_strength = (close - low) / rng

    out["entry_close_zone_location"] = _close_location_vs_zone(close, bottom, top)
    out["entry_9ema_relation"] = "above_9ema" if close > ema else "below_9ema" if np.isfinite(ema) else "unknown"
    out["entry_vwap_relation"] = "above_vwap" if close > vwap else "below_vwap" if np.isfinite(vwap) else "unknown"
    out["reaction_body_pct"] = round(body_pct, 3)
    out["reaction_range_vs_avg"] = round(float(range_vs_avg), 3) if np.isfinite(range_vs_avg) else np.nan
    out["reaction_close_strength"] = round(close_strength, 3)
    out["entry_volume_bucket"] = _volume_bucket(volume_ratio)

    if not _is_reversal_rejection_kind(kind):
        return out

    touched_zone = low <= top and high >= bottom
    zone_height = max(top - bottom, 1e-9)
    if kind == "demand_reversal":
        tap_depth = (top - low) / zone_height if low <= top else 0.0
        strong_close = close_strength >= 0.75
        bad_entry_side_wick = upper_wick_pct >= 0.35
        favorable_rejection_wick = lower_wick_pct >= 0.25
        boundary_confirmed = close > top
        ema_confirmed = close > ema
        vwap_confirmed = close > vwap
        structure_now = close > open_ and close >= _safe_float(prev.get("close")) and high >= _safe_float(prev.get("high"))
        one_candle_ft = boundary_confirmed and close > open_
    else:
        tap_depth = (high - bottom) / zone_height if high >= bottom else 0.0
        strong_close = close_strength <= 0.25
        bad_entry_side_wick = lower_wick_pct >= 0.35
        favorable_rejection_wick = upper_wick_pct >= 0.25
        boundary_confirmed = close < bottom
        ema_confirmed = close < ema
        vwap_confirmed = close < vwap
        structure_now = close < open_ and close <= _safe_float(prev.get("close")) and low <= _safe_float(prev.get("low"))
        one_candle_ft = boundary_confirmed and close < open_

    next_flags = [_directional_followthrough(nr, kind, close, high, low) for _, nr in next3.iterrows()]
    two_candle_ft = bool(one_candle_ft or any(next_flags[:1]))
    three_candle_ft = bool(two_candle_ft or any(next_flags[:3]))
    breakout_volume = bool(np.isfinite(volume_ratio) and volume_ratio >= 1.20 and body_pct >= 0.55)
    absorption = bool(np.isfinite(volume_ratio) and volume_ratio >= 1.50 and body_pct <= 0.35)
    followthrough_volume = False
    for _, nr in next3.head(2).iterrows():
        nr_body_ratio = _safe_float(nr.get("body_ratio"))
        nr_volume_ratio = _safe_float(nr.get("volume_ratio"))
        if _directional_followthrough(nr, kind, close, high, low) and np.isfinite(nr_body_ratio) and nr_body_ratio >= 0.55 and np.isfinite(nr_volume_ratio) and nr_volume_ratio >= 1.20:
            followthrough_volume = True
            break

    if breakout_volume:
        vpa_bucket = "breakout_volume_confirmed"
    elif absorption and followthrough_volume:
        vpa_bucket = "absorption_plus_followthrough"
    elif absorption:
        vpa_bucket = "absorption_no_followthrough"
    elif np.isfinite(volume_ratio) and volume_ratio < 0.80:
        vpa_bucket = "low_volume_reaction"
    elif np.isfinite(volume_ratio):
        vpa_bucket = "normal_or_elevated_volume_reaction"
    else:
        vpa_bucket = "unknown"
    vpa_confirmed = breakout_volume or (absorption and followthrough_volume)

    strong_reaction = bool(touched_zone and body_pct >= 0.60 and strong_close and (not np.isfinite(range_vs_avg) or range_vs_avg >= 1.10))
    good_wick_quality = not bad_entry_side_wick
    ema_or_vwap = bool(ema_confirmed or vwap_confirmed)
    structure_confirmed = bool(structure_now or two_candle_ft)
    score = int(touched_zone) + int(strong_reaction) + int(good_wick_quality) + int(vpa_confirmed) + int(ema_or_vwap) + int(structure_confirmed)
    has_1c = score >= 4
    has_2c = has_1c or bool(two_candle_ft and ema_or_vwap and (vpa_confirmed or followthrough_volume or breakout_volume))

    if kind == "demand_reversal":
        out["reclaimed_demand_top"] = bool(boundary_confirmed)
        out["entry_side_wick_pct"] = round(upper_wick_pct, 3)
        out["rejection_wick_pct"] = round(lower_wick_pct, 3)
    else:
        out["rejected_below_supply_bottom"] = bool(boundary_confirmed)
        out["entry_side_wick_pct"] = round(lower_wick_pct, 3)
        out["rejection_wick_pct"] = round(upper_wick_pct, 3)

    if bad_entry_side_wick:
        wick_bucket = "bad_entry_side_wick"
    elif favorable_rejection_wick:
        wick_bucket = "favorable_rejection_wick"
    else:
        wick_bucket = "acceptable_wick"

    if has_1c:
        momentum_bucket = "confirmed_same_candle"
    elif has_2c:
        momentum_bucket = "confirmed_within_2_candles"
    elif three_candle_ft:
        momentum_bucket = "partial_followthrough_within_3_candles"
    elif touched_zone and not one_candle_ft:
        momentum_bucket = "failed_or_stalled_after_tap"
    else:
        momentum_bucket = "never_confirmed"

    if boundary_confirmed and ema_or_vwap and structure_confirmed:
        realism = "valid_live_style_entry"
    elif not boundary_confirmed and has_2c:
        realism = "confirmation_arrived_late_or_without_boundary_reclaim"
    elif not boundary_confirmed:
        realism = "entered_before_boundary_reclaim_reject"
    elif not vpa_confirmed:
        realism = "entered_without_vpa_confirmation"
    elif not good_wick_quality:
        realism = "entered_with_bad_entry_side_wick"
    else:
        realism = "review"

    out.update({
        "zone_edge_tapped": bool(touched_zone),
        "zone_edge_tap_depth_pct": round(max(0.0, float(tap_depth)) * 100.0, 2),
        "has_boundary_reclaim_reject": bool(boundary_confirmed),
        "entry_9ema_confirmed": bool(ema_confirmed),
        "entry_vwap_confirmed": bool(vwap_confirmed),
        "strong_instant_reaction": bool(strong_reaction),
        "has_bad_entry_side_wick": bool(bad_entry_side_wick),
        "has_favorable_rejection_wick": bool(favorable_rejection_wick),
        "wick_quality_bucket": wick_bucket,
        "has_breakout_volume": bool(breakout_volume),
        "has_absorption_signature": bool(absorption),
        "has_followthrough_volume": bool(followthrough_volume),
        "vpa_confirmation_bucket": vpa_bucket,
        "vpa_confirmed": bool(vpa_confirmed),
        "one_candle_followthrough": bool(one_candle_ft),
        "two_candle_followthrough": bool(two_candle_ft),
        "three_candle_followthrough": bool(three_candle_ft),
        "breaks_9ema": bool(ema_confirmed),
        "breaks_vwap": bool(vwap_confirmed),
        "ema_or_vwap_break": bool(ema_or_vwap),
        "structure_confirmation": bool(structure_confirmed),
        "momentum_confirmation_bucket": momentum_bucket,
        "live_confirmation_score": int(score),
        "live_confirmation_bucket": _score_bucket(score),
        "has_1c_confirmation": bool(has_1c),
        "has_2c_confirmation": bool(has_2c),
        "backtest_realism_diagnosis": realism,
    })
    return out

def _entry_signal(day_df: pd.DataFrame, scenario: pd.Series, min_entry_time: time, preset: str) -> tuple[pd.Timestamp | None, dict]:
    """Find a 5M entry signal and return detailed confirmation diagnostics.

    Baseline entry behavior is intentionally preserved. The added fields are diagnostics used to audit
    whether reversal/rejection trades were entered before a live-style confirmation pattern completed.
    """
    try:
        bottom = float(scenario["zone_bottom"])
        top = float(scenario["zone_top"])
    except Exception:
        return None, {"rejection_reason": "invalid_zone_levels", "confirmation_state": "invalid"}

    kind = _scenario_kind(scenario)
    is_rev = _is_reversal_rejection_kind(kind)
    min_vol = {"balanced": 0.75, "strict": 1.0, "exploratory": 0.5}.get(preset, 0.75)
    require_body = {"balanced": 0.45, "strict": 0.55, "exploratory": 0.30}.get(preset, 0.45)

    idx = day_df.index
    open_a = day_df["open"].to_numpy(dtype=float)
    high_a = day_df["high"].to_numpy(dtype=float)
    low_a = day_df["low"].to_numpy(dtype=float)
    close_a = day_df["close"].to_numpy(dtype=float)
    ema_a = day_df["ema9"].to_numpy(dtype=float)
    vwap_a = day_df["vwap"].to_numpy(dtype=float)
    vr_a = day_df["volume_ratio"].fillna(0.0).to_numpy(dtype=float) if "volume_ratio" in day_df.columns else np.zeros(len(day_df))
    br_a = day_df["body_ratio"].fillna(0.0).to_numpy(dtype=float) if "body_ratio" in day_df.columns else np.zeros(len(day_df))

    touched_any = False
    touched_after_min = False
    touched_before_min = False
    best = {
        "rejection_reason": "price_never_touched_zone",
        "confirmation_state": "not_triggered",
        "best_confirmation_score": 0,
    }

    def _update_best(reason: str, state: str, score: int, i: int, missing: list[str], diag: dict):
        nonlocal best
        current_best = int(best.get("best_confirmation_score", -1) or -1)
        live_score = int(diag.get("live_confirmation_score", 0) or 0)
        combined_score = max(int(score), live_score)
        if combined_score >= current_best:
            best = {
                **diag,
                "rejection_reason": reason,
                "confirmation_state": state,
                "best_confirmation_score": int(combined_score),
                "best_candidate_time": idx[i].isoformat(),
                "best_candidate_close": round(float(close_a[i]), 4),
                "best_candidate_ema9": round(float(ema_a[i]), 4),
                "best_candidate_vwap": round(float(vwap_a[i]), 4),
                "best_candidate_volume_ratio": round(float(vr_a[i]), 2),
                "best_candidate_body_ratio": round(float(br_a[i]), 2),
                "missing_confirmation_components": ";".join(missing),
            }

    for i in range(1, len(day_df)):
        ts = idx[i]
        close = close_a[i]
        high = high_a[i]
        low = low_a[i]
        open_ = open_a[i]
        ema = ema_a[i]
        vwap = vwap_a[i]
        vr = vr_a[i]
        br = br_a[i]

        bullish_structure = close > open_ and close >= close_a[i-1] and high >= high_a[i-1]
        bearish_structure = close < open_ and close <= close_a[i-1] and low <= low_a[i-1]
        touched_zone = (low <= top) and (high >= bottom)

        if touched_zone:
            touched_any = True
            if ts.time() < min_entry_time:
                touched_before_min = True
            else:
                touched_after_min = True

        if ts.time() < min_entry_time:
            continue

        score = 0
        missing = []
        ok_touch = ok_direction = ok_ema = ok_vwap = ok_structure = False

        if kind == "demand_reversal":
            ok_touch = touched_zone
            # Reversal entries must occur after a completed zone rejection: price
            # has entered demand and closed back above the demand zone. Do not
            # treat inside-zone candles as entry confirmations.
            ok_direction = touched_zone and close > top
            ok_ema = close > ema
            ok_vwap = close > vwap
            ok_structure = bullish_structure
            components = ["tested_demand", "closed_above_demand_zone", "above_9ema", "above_vwap", "bullish_structure"]
            if not ok_touch: missing.append("zone_touch")
            if ok_touch and not ok_direction: missing.append("closed_above_demand_zone")
            if not ok_ema: missing.append("above_9ema")
            if not ok_vwap: missing.append("above_vwap")
            if not ok_structure: missing.append("bullish_structure")
        elif kind == "supply_rejection":
            ok_touch = touched_zone
            # Reversal entries must occur after a completed zone rejection: price
            # has entered supply and closed back below the supply zone. Do not
            # treat inside-zone candles as entry confirmations.
            ok_direction = touched_zone and close < bottom
            ok_ema = close < ema
            ok_vwap = close < vwap
            ok_structure = bearish_structure
            components = ["tested_supply", "closed_below_supply_zone", "below_9ema", "below_vwap", "bearish_structure"]
            if not ok_touch: missing.append("zone_touch")
            if ok_touch and not ok_direction: missing.append("closed_below_supply_zone")
            if not ok_ema: missing.append("below_9ema")
            if not ok_vwap: missing.append("below_vwap")
            if not ok_structure: missing.append("bearish_structure")
        elif kind == "supply_breakout":
            broke = close > top and close_a[i-1] <= top
            held = close > top and low >= bottom
            ok_touch = touched_zone or broke or held
            ok_direction = broke or held
            ok_ema = close > ema
            ok_vwap = close > vwap
            ok_structure = bullish_structure
            components = ["close_above_supply", "above_9ema", "above_vwap", "bullish_structure"]
            if not ok_direction: missing.append("close_above_supply")
            if not ok_ema: missing.append("above_9ema")
            if not ok_vwap: missing.append("above_vwap")
            if not ok_structure: missing.append("bullish_structure")
        elif kind == "demand_breakdown":
            broke = close < bottom and close_a[i-1] >= bottom
            held = close < bottom and high <= top
            ok_touch = touched_zone or broke or held
            ok_direction = broke or held
            ok_ema = close < ema
            ok_vwap = close < vwap
            ok_structure = bearish_structure
            components = ["close_below_demand", "below_9ema", "below_vwap", "bearish_structure"]
            if not ok_direction: missing.append("close_below_demand")
            if not ok_ema: missing.append("below_9ema")
            if not ok_vwap: missing.append("below_vwap")
            if not ok_structure: missing.append("bearish_structure")
        else:
            continue

        for flag in [ok_touch, ok_direction, ok_ema, ok_vwap, ok_structure]:
            if flag:
                score += 1

        core_ok = ok_touch and ok_direction and ok_ema and ok_vwap and ok_structure
        volume_ok = vr >= min_vol
        body_ok = br >= require_body
        diag = _live_style_reversal_diagnostics(day_df, scenario, ts) if is_rev and (ok_touch or core_ok) else {}

        if core_ok and volume_ok and body_ok:
            return ts, {
                **diag,
                "entry_price": float(close),
                "entry_volume_ratio": round(float(vr), 2),
                "entry_body_ratio": round(float(br), 2),
                "entry_ema9": round(float(ema), 2),
                "entry_vwap": round(float(vwap), 2),
                "entry_components": ";".join(components),
                "entry_kind": kind,
                "confirmation_state": "confirmed",
                "rejection_reason": "",
            }

        if core_ok and (not volume_ok or not body_ok):
            weak = []
            if not volume_ok:
                weak.append("volume_below_preset_threshold")
            if not body_ok:
                weak.append("body_below_preset_threshold")
            _update_best("confirmation_pending_low_volume_or_small_body", "developing", score, i, weak, diag)
        elif ok_touch:
            if not ok_direction:
                reason = "zone_touched_no_directional_rejection_or_break"
            elif not ok_ema and not ok_vwap:
                reason = "zone_touched_wrong_ema_and_vwap_side"
            elif not ok_ema:
                reason = "zone_touched_wrong_ema_side"
            elif not ok_vwap:
                reason = "zone_touched_wrong_vwap_side"
            elif not ok_structure:
                reason = "zone_touched_no_structure_confirmation"
            else:
                reason = "zone_touched_incomplete_confirmation"
            _update_best(reason, "not_confirmed", score, i, missing, diag)

    if touched_before_min and not touched_after_min:
        best["rejection_reason"] = "confirmation_or_zone_touch_before_min_entry_time"
        best["confirmation_state"] = "too_early"
    elif touched_any and not touched_after_min:
        best["rejection_reason"] = "zone_touched_before_min_entry_time_only"
        best["confirmation_state"] = "too_early"
    elif touched_any and best.get("rejection_reason") == "price_never_touched_zone":
        best["rejection_reason"] = "zone_touched_no_usable_5m_confirmation"
        best["confirmation_state"] = "not_confirmed"

    return None, best



def _iso_or_blank(ts):
    if ts is None:
        return ""
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return ""


def _exit_row(ts, price, reason, r_mult, mfe, mae, reached_1r, reached_2r, reached_3r, target, stop, risk, **extra):
    row = {
        "exit_time": ts.isoformat(),
        "exit_price": round(float(price), 4),
        "exit_reason": reason,
        "r_multiple": round(float(r_mult), 3),
        "mfe_r": round(float(mfe), 3),
        "mae_r": round(float(mae), 3),
        "reached_1r": bool(reached_1r),
        "reached_2r": bool(reached_2r),
        "reached_3r": bool(reached_3r),
        "target_price": round(float(target), 4),
        "stop_price": round(float(stop), 4),
        "risk_per_share": round(float(risk), 4),
    }
    row.update(extra)
    return row


def _event_audit_payload(
    *,
    side: str,
    entry_price: float,
    risk: float,
    rr: float,
    exit_ts,
    exit_reason: str,
    r_mult: float,
    mfe_until_exit: float,
    mae_until_exit: float,
    full_day_mfe: float,
    full_day_mae: float,
    first_event: str | None,
    first_1r_time,
    first_2r_time,
    first_3r_time,
    first_stop_time,
    first_target_time,
    first_ema_time,
    target_and_stop_same_candle: bool,
    one_r_and_stop_same_candle: bool,
    two_r_and_stop_same_candle: bool,
    three_r_and_stop_same_candle: bool,
):
    """Build diagnostic-only trade-path audit fields.

    Baseline replay behavior remains conservative inside a 5M candle: zone stop is processed before target.
    These fields make that assumption visible so we can diagnose whether live management would differ.
    """
    exit_ts = pd.Timestamp(exit_ts)
    first_target_ts = pd.Timestamp(first_target_time) if first_target_time is not None else None
    first_ema_ts = pd.Timestamp(first_ema_time) if first_ema_time is not None else None
    first_1r_ts = pd.Timestamp(first_1r_time) if first_1r_time is not None else None
    first_2r_ts = pd.Timestamp(first_2r_time) if first_2r_time is not None else None
    first_3r_ts = pd.Timestamp(first_3r_time) if first_3r_time is not None else None

    target_available = first_target_ts is not None and first_target_ts <= exit_ts and exit_reason != "target_3r"
    ema_available = first_ema_ts is not None and first_ema_ts <= exit_ts and not str(exit_reason).startswith("ema_protection")
    stop_exit = exit_reason == "stop_zone"
    eps = 1e-9
    ambiguous = bool(target_and_stop_same_candle)
    if ambiguous:
        conservative_result = -1.0
        optimistic_result = float(rr)
        neutral_result = np.nan
    else:
        conservative_result = float(r_mult)
        optimistic_result = float(r_mult)
        neutral_result = float(r_mult)

    risk_pct = abs(risk / entry_price) if entry_price else np.nan
    unrealistic = bool(
        (np.isfinite(risk_pct) and risk_pct < 0.0005)
        or (np.isfinite(full_day_mfe) and full_day_mfe > 10.0)
        or (np.isfinite(full_day_mae) and full_day_mae < -10.0)
    )

    return {
        "first_event_after_entry": first_event or ("end_of_day_first" if exit_reason == "end_of_day" else f"{exit_reason}_first"),
        "first_event_r_result": round(float(conservative_result), 3) if np.isfinite(conservative_result) else np.nan,
        "first_1r_time": _iso_or_blank(first_1r_time),
        "first_2r_time": _iso_or_blank(first_2r_time),
        "first_3r_time": _iso_or_blank(first_3r_time),
        "first_stop_time": _iso_or_blank(first_stop_time),
        "first_target_3r_time": _iso_or_blank(first_target_time),
        "first_ema_protection_time": _iso_or_blank(first_ema_time),
        "actual_exit_event_time": _iso_or_blank(exit_ts),
        "mfe_r_until_exit": round(float(mfe_until_exit), 3),
        "mae_r_until_exit": round(float(mae_until_exit), 3),
        "mfe_r_full_day": round(float(full_day_mfe), 3),
        "mae_r_full_day": round(float(full_day_mae), 3),
        "target_and_stop_same_candle": bool(target_and_stop_same_candle),
        "one_r_and_stop_same_candle": bool(one_r_and_stop_same_candle),
        "two_r_and_stop_same_candle": bool(two_r_and_stop_same_candle),
        "three_r_and_stop_same_candle": bool(three_r_and_stop_same_candle),
        "same_candle_ambiguity": bool(target_and_stop_same_candle or one_r_and_stop_same_candle or two_r_and_stop_same_candle or three_r_and_stop_same_candle),
        "conservative_intrabar_result": round(float(conservative_result), 3) if np.isfinite(conservative_result) else np.nan,
        "optimistic_intrabar_result": round(float(optimistic_result), 3) if np.isfinite(optimistic_result) else np.nan,
        "neutral_intrabar_result": round(float(neutral_result), 3) if np.isfinite(neutral_result) else np.nan,
        "intrabar_policy_used": "conservative_stop_before_target",
        "stop_after_reached_1r": bool(stop_exit and first_1r_ts is not None and first_1r_ts <= exit_ts),
        "stop_after_reached_2r": bool(stop_exit and first_2r_ts is not None and first_2r_ts <= exit_ts),
        "stop_after_reached_3r": bool(stop_exit and first_3r_ts is not None and first_3r_ts <= exit_ts),
        "target_available_but_not_taken": bool(target_available),
        "ema_protection_available_but_not_taken": bool(ema_available),
        "mfe_after_exit_detected": bool(np.isfinite(full_day_mfe) and np.isfinite(mfe_until_exit) and full_day_mfe > mfe_until_exit + eps),
        "mae_after_exit_detected": bool(np.isfinite(full_day_mae) and np.isfinite(mae_until_exit) and full_day_mae < mae_until_exit - eps),
        "unrealistic_r_outlier": unrealistic,
        "risk_pct_of_entry": round(float(risk_pct) * 100.0, 4) if np.isfinite(risk_pct) else np.nan,
    }


def _simulate_exit(day_df: pd.DataFrame, entry_time, scenario: pd.Series, entry: dict, rr: float, ema_exit_after_r: float, ema_confirm_bars: int):
    side = _scenario_side(scenario)
    entry_price = float(entry["entry_price"])
    bottom = float(scenario["zone_bottom"])
    top = float(scenario["zone_top"])

    if side == "long":
        stop = bottom
        risk = entry_price - stop
        target = entry_price + rr * risk
    else:
        stop = top
        risk = stop - entry_price
        target = entry_price - rr * risk

    if not np.isfinite(risk) or risk <= 0:
        return {"skip_reason": "invalid_risk"}

    entry_bar = day_df.loc[entry_time] if entry_time in day_df.index else None
    entry_atr_14 = float(entry_bar.get("atr14", np.nan)) if entry_bar is not None else np.nan
    atr_1x_r = entry_atr_14 / risk if np.isfinite(entry_atr_14) and entry_atr_14 > 0 else np.nan

    future = day_df[day_df.index > entry_time].copy()
    if future.empty:
        return {"skip_reason": "no_future_bars"}

    if side == "long":
        full_day_mfe = float(((future["high"] - entry_price) / risk).max())
        full_day_mae = float(((future["low"] - entry_price) / risk).min())
    else:
        full_day_mfe = float(((entry_price - future["low"]) / risk).max())
        full_day_mae = float(((entry_price - future["high"]) / risk).min())

    mfe = 0.0
    mae = 0.0
    adverse_ema_count = 0
    reached_1r = reached_2r = reached_3r = False
    reached_atr_1x = False
    first_event = None
    first_1r_time = first_2r_time = first_3r_time = None
    first_atr_1x_time = None
    first_stop_time = first_target_time = first_ema_time = None
    target_and_stop_same_candle = False
    one_r_and_stop_same_candle = False
    two_r_and_stop_same_candle = False
    three_r_and_stop_same_candle = False

    def mark_first_event(label: str):
        nonlocal first_event
        if first_event is None:
            first_event = label

    for ts, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        ema = float(row["ema9"])

        if side == "long":
            favorable_r = (high - entry_price) / risk
            adverse_r = (low - entry_price) / risk
            stop_hit = low <= stop
            target_hit = high >= target
            hit_1r = favorable_r >= 1.0
            hit_2r = favorable_r >= 2.0
            hit_3r = favorable_r >= 3.0
            hit_atr_1x = np.isfinite(atr_1x_r) and favorable_r >= atr_1x_r
            adverse_ema = close < ema
            close_r = (close - entry_price) / risk
        else:
            favorable_r = (entry_price - low) / risk
            adverse_r = (entry_price - high) / risk
            stop_hit = high >= stop
            target_hit = low <= target
            hit_1r = favorable_r >= 1.0
            hit_2r = favorable_r >= 2.0
            hit_3r = favorable_r >= 3.0
            hit_atr_1x = np.isfinite(atr_1x_r) and favorable_r >= atr_1x_r
            adverse_ema = close > ema
            close_r = (entry_price - close) / risk

        mfe = max(mfe, favorable_r)
        mae = min(mae, adverse_r)

        if hit_1r and first_1r_time is None:
            first_1r_time = ts
        if hit_2r and first_2r_time is None:
            first_2r_time = ts
        if hit_3r and first_3r_time is None:
            first_3r_time = ts
        if hit_atr_1x and first_atr_1x_time is None:
            first_atr_1x_time = ts
        if stop_hit and first_stop_time is None:
            first_stop_time = ts
        if target_hit and first_target_time is None:
            first_target_time = ts

        reached_1r = reached_1r or hit_1r
        reached_2r = reached_2r or hit_2r
        reached_3r = reached_3r or hit_3r
        reached_atr_1x = reached_atr_1x or hit_atr_1x

        if stop_hit and target_hit:
            target_and_stop_same_candle = True
        if stop_hit and hit_1r:
            one_r_and_stop_same_candle = True
        if stop_hit and hit_2r:
            two_r_and_stop_same_candle = True
        if stop_hit and hit_3r:
            three_r_and_stop_same_candle = True

        if first_event is None:
            if stop_hit and target_hit:
                mark_first_event("same_candle_target_and_stop")
            elif stop_hit and hit_3r:
                mark_first_event("same_candle_3r_and_stop")
            elif stop_hit and hit_2r:
                mark_first_event("same_candle_2r_and_stop")
            elif stop_hit and hit_1r:
                mark_first_event("same_candle_1r_and_stop")
            elif stop_hit:
                mark_first_event("stop_zone_first")
            elif target_hit:
                mark_first_event("target_3r_first")
            elif hit_3r:
                mark_first_event("first_3r")
            elif hit_2r:
                mark_first_event("first_2r")
            elif hit_1r:
                mark_first_event("first_1r")

        def finish(price, reason, r_mult):
            audit = _event_audit_payload(
                side=side,
                entry_price=entry_price,
                risk=risk,
                rr=rr,
                exit_ts=ts,
                exit_reason=reason,
                r_mult=r_mult,
                mfe_until_exit=mfe,
                mae_until_exit=mae,
                full_day_mfe=full_day_mfe,
                full_day_mae=full_day_mae,
                first_event=first_event,
                first_1r_time=first_1r_time,
                first_2r_time=first_2r_time,
                first_3r_time=first_3r_time,
                first_stop_time=first_stop_time,
                first_target_time=first_target_time,
                first_ema_time=first_ema_time,
                target_and_stop_same_candle=target_and_stop_same_candle,
                one_r_and_stop_same_candle=one_r_and_stop_same_candle,
                two_r_and_stop_same_candle=two_r_and_stop_same_candle,
                three_r_and_stop_same_candle=three_r_and_stop_same_candle,
            )
            return _exit_row(
                ts, price, reason, r_mult, mfe, mae, reached_1r, reached_2r, reached_3r, target, stop, risk,
                entry_atr_14=round(float(entry_atr_14), 4) if np.isfinite(entry_atr_14) else np.nan,
                atr_1x_r=round(float(atr_1x_r), 3) if np.isfinite(atr_1x_r) else np.nan,
                reached_atr_1x=bool(reached_atr_1x),
                first_atr_1x_time=_iso_or_blank(first_atr_1x_time),
                target_1r_result=1.0 if reached_1r else round(float(r_mult), 3),
                target_2r_result=2.0 if reached_2r else round(float(r_mult), 3),
                target_3r_result=3.0 if reached_3r else round(float(r_mult), 3),
                target_atr_1x_result=round(float(atr_1x_r), 3) if reached_atr_1x and np.isfinite(atr_1x_r) else round(float(r_mult), 3),
                **audit
            )

        # Preserve current baseline scoring: conservative stop-before-target inside a 5M candle.
        if stop_hit:
            return finish(stop, "stop_zone", -1.0)
        if target_hit:
            return finish(target, "target_3r", rr)

        # EMA protection is a continuous trade-management rule, not a profit-gated
        # rule: calls exit after two closes below 9EMA; puts exit after two closes
        # above 9EMA. Evaluate every candle after entry.
        adverse_ema_count = adverse_ema_count + 1 if adverse_ema else 0
        if adverse_ema_count >= ema_confirm_bars:
            if first_ema_time is None:
                first_ema_time = ts
            if first_event is None:
                mark_first_event(f"ema_protection_{ema_confirm_bars}_closes_first")
            return finish(close, f"ema_protection_{ema_confirm_bars}_closes", close_r)

    last_ts = future.index[-1]
    last_close = float(future.iloc[-1]["close"])
    r_mult = (last_close - entry_price) / risk if side == "long" else (entry_price - last_close) / risk
    ts = last_ts
    audit = _event_audit_payload(
        side=side,
        entry_price=entry_price,
        risk=risk,
        rr=rr,
        exit_ts=last_ts,
        exit_reason="end_of_day",
        r_mult=r_mult,
        mfe_until_exit=mfe,
        mae_until_exit=mae,
        full_day_mfe=full_day_mfe,
        full_day_mae=full_day_mae,
        first_event=first_event or "end_of_day_first",
        first_1r_time=first_1r_time,
        first_2r_time=first_2r_time,
        first_3r_time=first_3r_time,
        first_stop_time=first_stop_time,
        first_target_time=first_target_time,
        first_ema_time=first_ema_time,
        target_and_stop_same_candle=target_and_stop_same_candle,
        one_r_and_stop_same_candle=one_r_and_stop_same_candle,
        two_r_and_stop_same_candle=two_r_and_stop_same_candle,
        three_r_and_stop_same_candle=three_r_and_stop_same_candle,
    )
    return _exit_row(
        last_ts, last_close, "end_of_day", r_mult, mfe, mae, reached_1r, reached_2r, reached_3r, target, stop, risk,
        entry_atr_14=round(float(entry_atr_14), 4) if np.isfinite(entry_atr_14) else np.nan,
        atr_1x_r=round(float(atr_1x_r), 3) if np.isfinite(atr_1x_r) else np.nan,
        reached_atr_1x=bool(reached_atr_1x),
        first_atr_1x_time=_iso_or_blank(first_atr_1x_time),
        target_1r_result=1.0 if reached_1r else round(float(r_mult), 3),
        target_2r_result=2.0 if reached_2r else round(float(r_mult), 3),
        target_3r_result=3.0 if reached_3r else round(float(r_mult), 3),
        target_atr_1x_result=round(float(atr_1x_r), 3) if reached_atr_1x and np.isfinite(atr_1x_r) else round(float(r_mult), 3),
        **audit
    )


def _variant_exit_row(prefix: str, ts, price, reason, r_mult):
    return {
        f"{prefix}_exit_time": _iso_or_blank(ts),
        f"{prefix}_exit_price": round(float(price), 4),
        f"{prefix}_exit_reason": reason,
        f"{prefix}_exit_r": round(float(r_mult), 3),
    }


def _simulate_reversal_management_variant(day_df: pd.DataFrame, entry_time, scenario: pd.Series, entry: dict, rr: float, variant: str, ema_confirm_bars: int = 2):
    side = _scenario_side(scenario)
    kind = _scenario_kind(scenario)
    entry_price = float(entry["entry_price"])
    bottom = float(scenario["zone_bottom"])
    top = float(scenario["zone_top"])
    if side == "long":
        base_stop = bottom
        risk = entry_price - base_stop
        target = entry_price + rr * risk
    else:
        base_stop = top
        risk = base_stop - entry_price
        target = entry_price - rr * risk
    if not np.isfinite(risk) or risk <= 0:
        return {}
    future = day_df[day_df.index > entry_time].copy()
    if future.empty:
        return {}

    be_active = False
    inside_count = 0
    adverse_ema_count = 0
    prefix = variant

    for ts, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        ema = float(row.get("ema9", np.nan))
        if side == "long":
            target_hit = high >= target
            stop_hit = low <= base_stop
            be_hit = be_active and low <= entry_price
            favorable_r = (high - entry_price) / risk
            close_r = (close - entry_price) / risk
            boundary_lost = close <= top if kind == "demand_reversal" else False
            adverse_ema = close < ema
        else:
            target_hit = low <= target
            stop_hit = high >= base_stop
            be_hit = be_active and high >= entry_price
            favorable_r = (entry_price - low) / risk
            close_r = (entry_price - close) / risk
            boundary_lost = close >= bottom if kind == "supply_rejection" else False
            adverse_ema = close > ema

        if variant == "target_priority_3r":
            if target_hit:
                return _variant_exit_row(prefix, ts, target, "target_3r", rr)
            if stop_hit:
                return _variant_exit_row(prefix, ts, base_stop, "stop_zone", -1.0)
        elif variant == "breakeven_after_1r":
            if be_hit:
                return _variant_exit_row(prefix, ts, entry_price, "breakeven_after_1r", 0.0)
            if stop_hit:
                return _variant_exit_row(prefix, ts, base_stop, "stop_zone", -1.0)
            if target_hit:
                return _variant_exit_row(prefix, ts, target, "target_3r", rr)
            if favorable_r >= 1.0:
                be_active = True
        elif variant == "boundary_loss_1_close":
            if stop_hit:
                return _variant_exit_row(prefix, ts, base_stop, "stop_zone", -1.0)
            if target_hit:
                return _variant_exit_row(prefix, ts, target, "target_3r", rr)
            if boundary_lost:
                return _variant_exit_row(prefix, ts, close, "boundary_loss_1_close", close_r)
        elif variant == "boundary_loss_2_closes":
            if stop_hit:
                return _variant_exit_row(prefix, ts, base_stop, "stop_zone", -1.0)
            if target_hit:
                return _variant_exit_row(prefix, ts, target, "target_3r", rr)
            inside_count = inside_count + 1 if boundary_lost else 0
            if inside_count >= 2:
                return _variant_exit_row(prefix, ts, close, "boundary_loss_2_closes", close_r)
        elif variant == "ema_protect_05r":
            if stop_hit:
                return _variant_exit_row(prefix, ts, base_stop, "stop_zone", -1.0)
            if target_hit:
                return _variant_exit_row(prefix, ts, target, "target_3r", rr)
            if favorable_r >= 0.5:
                adverse_ema_count = adverse_ema_count + 1 if adverse_ema else 0
                if adverse_ema_count >= ema_confirm_bars:
                    return _variant_exit_row(prefix, ts, close, f"ema_protection_{ema_confirm_bars}_closes_after_0_5r", close_r)

    last_ts = future.index[-1]
    last_close = float(future.iloc[-1]["close"])
    r_mult = (last_close - entry_price) / risk if side == "long" else (entry_price - last_close) / risk
    return _variant_exit_row(prefix, last_ts, last_close, "end_of_day", r_mult)


def _simulate_reversal_management_variants(day_df: pd.DataFrame, entry_time, scenario: pd.Series, entry: dict, rr: float, ema_confirm_bars: int):
    if not _is_reversal_rejection_kind(_scenario_kind(scenario)):
        return {}
    out = {}
    for variant in [
        "target_priority_3r",
        "breakeven_after_1r",
        "boundary_loss_1_close",
        "boundary_loss_2_closes",
        "ema_protect_05r",
    ]:
        out.update(_simulate_reversal_management_variant(day_df, entry_time, scenario, entry, rr, variant, ema_confirm_bars))
    return out

def main():
    parser = argparse.ArgumentParser(description="Replay historical days from daily watchlist snapshots.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols.")
    parser.add_argument("--rr", type=float, default=3.0, help="Primary target in R multiples.")
    parser.add_argument("--preset", choices=["exploratory", "balanced", "strict"], default="balanced")
    parser.add_argument("--min-entry-time", default="09:40", help="NY time, default avoids first 5-minute candle.")
    parser.add_argument("--max-entry-time", default=None, help="NY time cutoff for new entries. Example: 13:00 blocks entries after 1 PM.")
    parser.add_argument("--ema-exit-after-r", type=float, default=0.0, help="Deprecated compatibility option. 9EMA protection now runs continuously after entry.")
    parser.add_argument("--ema-exit-confirm-bars", type=int, default=2)
    parser.add_argument("--use-final-only", action="store_true", help="Replay only strict final watchlist rows. Default uses final + developing scenario rows.")
    parser.add_argument("--snapshot-mode", choices=["close", "preopen"], default="close", help="Use prior-close snapshots or simulated preopen movement-context snapshots. Default: close.")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    min_h, min_m = [int(x) for x in args.min_entry_time.split(":")]
    min_entry_time = time(min_h, min_m)
    max_entry_time = None
    if args.max_entry_time:
        max_h, max_m = [int(x) for x in args.max_entry_time.split(":")]
        max_entry_time = time(max_h, max_m)

    manifest_path = REPORT_DIR / "backtest" / "snapshot_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit("Run build_backtest_snapshots.py first.")
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"].eq("ok")].copy()
    if args.start:
        manifest = manifest[pd.to_datetime(manifest["test_date"]) >= pd.to_datetime(args.start)]
    if args.end:
        manifest = manifest[pd.to_datetime(manifest["test_date"]) <= pd.to_datetime(args.end)]

    trades = []
    candidates = []

    for _, m in manifest.iterrows():
        day = str(m["test_date"])
        scenario_file, scenario_col = _resolve_snapshot_file(
            m,
            snapshot_mode=args.snapshot_mode,
            use_final_only=args.use_final_only,
        )
        if not scenario_file.exists():
            candidates.append({
                "test_date": day,
                "as_of_date": m.get("as_of_date", ""),
                "snapshot_mode": args.snapshot_mode,
                "scenario_file": str(scenario_file),
                "entry_eligible": False,
                "rejection_reason": "missing_snapshot_file",
            })
            continue
        scenarios = pd.read_csv(scenario_file)
        scenarios = _prepare_snapshot_rows(
            scenarios,
            day=day,
            scenario_file=scenario_file,
            snapshot_mode=args.snapshot_mode,
        )
        if scenarios.empty:
            continue
        if symbols:
            scenarios = scenarios[scenarios["symbol"].astype(str).str.upper().isin(symbols)].copy()
        if scenarios.empty:
            continue
        if "watchlist_bucket" in scenarios.columns and not args.use_final_only:
            scenarios = scenarios[scenarios["watchlist_bucket"].isin(["Final / Actionable", "Developing Scenario"])].copy()
        if scenarios.empty:
            continue

        day_data_cache = {}
        symbol_open_until = {}
        for _, sc in scenarios.sort_values(["symbol", "distance_pct", "setup_quality_score"], ascending=[True, True, False]).iterrows():
            sym = str(sc["symbol"]).upper()
            if sym not in day_data_cache:
                day_data_cache[sym] = _load_day_5m(sym, day)
            day_df = day_data_cache[sym]
            cand = {
                "test_date": day,
                "as_of_date": m["as_of_date"],
                "snapshot_mode": args.snapshot_mode,
                "scenario_file": str(scenario_file),
                "snapshot_candidate_id": sc.get("snapshot_candidate_id", ""),
                "snapshot_test_date": sc.get("snapshot_test_date", day),
                "snapshot_as_of_date": sc.get("snapshot_as_of_date", m.get("as_of_date", "")),
                "snapshot_source_file": sc.get("snapshot_source_file", str(scenario_file)),
                "replay_snapshot_file": sc.get("replay_snapshot_file", str(scenario_file)),
                "symbol": sym,
                "scenario": sc.get("scenario_label", sc.get("scenario", "")),
                "side": _scenario_side(sc),
                "zone_type": sc.get("zone_type"),
                "zone_bottom": sc.get("zone_bottom"),
                "zone_top": sc.get("zone_top"),
                "timeframe": sc.get("timeframe", sc.get("primary_timeframe", "")),
                "freshness": sc.get("freshness", sc.get("freshness_label", "")),
                "tests": sc.get("tests", ""),
                "watchlist_bucket": sc.get("watchlist_bucket", ""),
                "movement_watchlist_bucket": sc.get("movement_watchlist_bucket", ""),
                "zone_thesis": sc.get("zone_thesis", ""),
                "zone_movement_state": sc.get("zone_movement_state", ""),
                "observation_score": sc.get("observation_score", ""),
                "observation_reason": sc.get("observation_reason", ""),
                "watch_for": sc.get("watch_for", ""),
                "current_price": sc.get("current_price", ""),
                "current_price_as_of": sc.get("current_price_as_of", ""),
                "current_price_session": sc.get("current_price_session", ""),
                "snapshot_context_time": sc.get("snapshot_context_time", m.get("preopen_context_time", "")),
                "snapshot_context_type": sc.get("snapshot_context_type", args.snapshot_mode),
                "gap_direction": sc.get("gap_direction", ""),
                "gap_pct": sc.get("gap_pct", ""),
                "gap_zone_context": sc.get("gap_zone_context", ""),
                "recent_move_direction": sc.get("recent_move_direction", ""),
                "recent_move_strength": sc.get("recent_move_strength", ""),
                "price_vs_9ema": sc.get("price_vs_9ema", ""),
                "price_vs_vwap": sc.get("price_vs_vwap", ""),
                "volume_state": sc.get("volume_state", ""),
                "vpa_state": sc.get("vpa_state", ""),
                "historical_zone_tendency": sc.get("historical_zone_tendency", ""),
                "historical_reaction_score": sc.get("historical_reaction_score", ""),
                "setup_quality_grade": sc.get("setup_quality_grade", sc.get("grade", "")),
                "setup_quality_score": sc.get("setup_quality_score", sc.get("quality_score", "")),
            }
            if day_df.empty:
                cand.update({"entry_eligible": False, "rejection_reason": "missing_5m_day_data"})
                candidates.append(cand)
                continue
            entry_time, entry = _entry_signal(day_df, sc, min_entry_time, args.preset)
            if entry_time is None:
                cand.update({"entry_eligible": False, **entry})
                candidates.append(cand)
                continue
            if max_entry_time is not None and entry_time.time() > max_entry_time:
                cand.update({"entry_eligible": False, **entry, "entry_time": entry_time.isoformat(), "rejection_reason": "entry_after_max_entry_time"})
                candidates.append(cand)
                continue
            if sym in symbol_open_until and entry_time <= symbol_open_until[sym]:
                cand.update({"entry_eligible": False, "rejection_reason": "overlap_same_symbol"})
                candidates.append(cand)
                continue
            exit_info = _simulate_exit(day_df, entry_time, sc, entry, args.rr, args.ema_exit_after_r, args.ema_exit_confirm_bars)
            if "skip_reason" in exit_info:
                cand.update({"entry_eligible": False, "rejection_reason": exit_info["skip_reason"]})
                candidates.append(cand)
                continue

            management_variants = {}
            if _is_reversal_rejection_kind(entry.get("entry_kind", "")):
                management_variants = _simulate_reversal_management_variants(
                    day_df, entry_time, sc, entry, args.rr, args.ema_exit_confirm_bars
                )

            symbol_open_until[sym] = pd.to_datetime(exit_info["exit_time"])
            row = {**cand, **entry, **exit_info, **management_variants, "entry_time": entry_time.isoformat(), "entry_eligible": True,
                   "target_ladder": sc.get("target_ladder", "")}
            trades.append(row)
            candidates.append({**cand, **entry, "entry_time": entry_time.isoformat(), "entry_eligible": True})

    out_dir = REPORT_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades)
    cand_df = pd.DataFrame(candidates)
    trades_path = out_dir / "trades.csv"
    cand_path = out_dir / "entry_candidates.csv"
    summary_path = out_dir / "summary.csv"
    trades_df.to_csv(trades_path, index=False)
    cand_df.to_csv(cand_path, index=False)

    if trades_df.empty:
        summary = pd.DataFrame([{"trades": 0}])
    else:
        summary = pd.DataFrame([{
            "trades": len(trades_df),
            "profit_win_rate": round((trades_df["r_multiple"] > 0).mean() * 100, 2),
            "target_hit_rate": round((trades_df["exit_reason"] == "target_3r").mean() * 100, 2),
            "avg_r": round(trades_df["r_multiple"].mean(), 3),
            "total_r": round(trades_df["r_multiple"].sum(), 3),
            "reached_1r_rate": round(trades_df["reached_1r"].mean() * 100, 2),
            "reached_2r_rate": round(trades_df["reached_2r"].mean() * 100, 2),
            "reached_3r_rate": round(trades_df["reached_3r"].mean() * 100, 2),
            "avg_mfe_r": round(trades_df["mfe_r"].mean(), 3),
            "avg_mae_r": round(trades_df["mae_r"].mean(), 3),
        }])
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {trades_path}")
    print(f"Wrote {cand_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
