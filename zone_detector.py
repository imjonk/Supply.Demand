from dataclasses import asdict
import pandas as pd
import numpy as np

from config import RULES

ZONE_METADATA_FIELDS = [
    "zone_id", "zone_timeframe", "zone_created_time", "zone_age_days", "zone_age_bars",
    "zone_width", "zone_width_pct", "zone_height_atr", "touch_count_before_snapshot",
    "first_touch_time", "last_touch_time", "fresh_zone", "departure_body_ratio",
    "departure_candle_count", "departure_atr_expansion", "departure_strength_score",
    "nested_inside_higher_tf", "higher_tf_zone_count", "overlapping_zone_count",
    "confluence_score", "trend_alignment", "distance_to_next_supply",
    "distance_to_next_demand", "gap_created_zone", "gap_into_zone",
    "gap_away_from_zone", "base_quality_score", "historical_reaction_count",
]


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_top"] = out[["open", "close"]].max(axis=1)
    out["body_bottom"] = out[["open", "close"]].min(axis=1)
    out["avg_range"] = out["range"].rolling(RULES.lookback_for_averages, min_periods=8).mean()
    out["avg_volume"] = out["volume"].rolling(RULES.lookback_for_averages, min_periods=8).mean()
    return out


def _zone_freshness(df: pd.DataFrame, start_idx: int, zone_type: str, top: float, bottom: float) -> tuple[str, int, bool, str, str]:
    # Start checking after the departure window so the departure itself doesn't count as a retest.
    tests = 0
    broken = False
    first_touch_time = ""
    last_touch_time = ""
    subsequent = df.iloc[start_idx + RULES.departure_window + 1:]

    for _, row in subsequent.iterrows():
        overlaps = (row["low"] <= top) and (row["high"] >= bottom)

        if zone_type == "demand":
            if row["close"] < bottom:
                broken = True
                break
            if overlaps:
                tests += 1
                first_touch_time = first_touch_time or row.name.isoformat()
                last_touch_time = row.name.isoformat()
        else:
            if row["close"] > top:
                broken = True
                break
            if overlaps:
                tests += 1
                first_touch_time = first_touch_time or row.name.isoformat()
                last_touch_time = row.name.isoformat()

    if broken:
        return "broken", tests, True, first_touch_time, last_touch_time
    if tests == 0:
        return "fresh", tests, False, first_touch_time, last_touch_time
    if tests == 1:
        return "one_test", tests, False, first_touch_time, last_touch_time
    return "multiple_tests", tests, False, first_touch_time, last_touch_time


def _classify_prior_move(d: pd.DataFrame, i: int, row: pd.Series) -> tuple[str | None, float]:
    prior_start = max(0, i - RULES.prior_window)
    prior_move = row["close"] - d.iloc[prior_start]["close"]
    prior_atr = prior_move / row["avg_range"] if row["avg_range"] > 0 else 0

    if prior_atr >= RULES.min_prior_move_atr:
        return "rally", prior_atr
    if prior_atr <= -RULES.min_prior_move_atr:
        return "drop", prior_atr

    # If the larger two-candle lookback is too flat, use the immediate prior candle
    # as a softer classification instead of rejecting the base outright.
    prev = d.iloc[i - 1] if i > 0 else None
    if prev is not None:
        immediate = row["close"] - prev["close"]
        if immediate > 0:
            return "rally", prior_atr
        if immediate < 0:
            return "drop", prior_atr

    return None, prior_atr


def _classify_departure(d: pd.DataFrame, i: int, row: pd.Series) -> tuple[str | None, float, float, float]:
    """Return departure_dir, departure_atr, departure_vol_ratio, departure_body_multiple.

    Current zone validation rule:
    The candle immediately after the basing candle must have a body greater than
    RULES.next_candle_move_multiple times the basing candle body, in the departure
    direction.

    This intentionally uses candle BODY, not full candle range or wick movement.
    """
    dep_end = min(len(d) - 1, i + RULES.departure_window)
    departure_slice = d.iloc[i + 1: dep_end + 1]
    if departure_slice.empty:
        return None, 0.0, 0.0, 0.0

    departure_vol_ratio = departure_slice["volume"].mean() / row["avg_volume"] if row["avg_volume"] > 0 else 0

    if RULES.use_next_candle_multiplier:
        nxt = departure_slice.iloc[0]
        base_body = max(float(row["body"]), 0.01)
        next_body = abs(float(nxt["close"] - nxt["open"]))
        body_multiple = next_body / base_body

        up_ok = (nxt["close"] > nxt["open"]) and (body_multiple > RULES.next_candle_move_multiple)
        down_ok = (nxt["close"] < nxt["open"]) and (body_multiple > RULES.next_candle_move_multiple)

        # Keep departure_atr as a broad strength metric for scoring, but do not use
        # ATR/range/wicks to decide whether this is a valid base/departure pair.
        departure_move = float(nxt["close"] - row["close"])
        departure_atr = departure_move / row["avg_range"] if row["avg_range"] > 0 else 0

        if up_ok:
            return "rally", departure_atr, departure_vol_ratio, body_multiple
        if down_ok:
            return "drop", departure_atr, departure_vol_ratio, body_multiple

    # Fallback: old ATR-style rule over the departure window.
    departure_move = d.iloc[dep_end]["close"] - row["close"]
    departure_atr = departure_move / row["avg_range"] if row["avg_range"] > 0 else 0
    if departure_atr >= RULES.min_departure_move_atr:
        return "rally", departure_atr, departure_vol_ratio, abs(departure_move) / max(float(row["body"]), 0.01)
    if departure_atr <= -RULES.min_departure_move_atr:
        return "drop", departure_atr, departure_vol_ratio, abs(departure_move) / max(float(row["body"]), 0.01)
    return None, departure_atr, departure_vol_ratio, abs(departure_move) / max(float(row["body"]), 0.01)


def detect_zones(df: pd.DataFrame, symbol: str, timeframe: str) -> list[dict]:
    d = _prep_features(df)
    rows = []
    if len(d) < RULES.lookback_for_averages + RULES.prior_window + RULES.departure_window + 5:
        return rows

    for i in range(RULES.lookback_for_averages, len(d) - RULES.departure_window):
        row = d.iloc[i]
        if not np.isfinite(row["avg_range"]) or not np.isfinite(row["avg_volume"]):
            continue
        if row["avg_range"] <= 0 or row["avg_volume"] <= 0 or row["range"] <= 0:
            continue

        is_small_range = row["range"] <= row["avg_range"] * RULES.base_range_max_vs_avg
        is_small_body = row["body"] <= row["range"] * RULES.base_body_max_of_range
        vol_ratio = row["volume"] / row["avg_volume"]
        is_avg_volume = RULES.volume_min_vs_avg <= vol_ratio <= RULES.volume_max_vs_avg

        if not (is_small_range and is_small_body and is_avg_volume):
            continue

        prior_dir, prior_atr = _classify_prior_move(d, i, row)
        if prior_dir is None:
            continue

        departure_dir, departure_atr, departure_vol_ratio, departure_body_multiple = _classify_departure(d, i, row)
        if departure_dir is None:
            continue
        if departure_vol_ratio < RULES.min_departure_volume_vs_avg:
            continue

        if departure_dir == "rally":
            zone_type = "demand"
            pattern = "RBR" if prior_dir == "rally" else "DBR"
            top = float(row["body_top"])
            bottom = float(row["low"])
        else:
            zone_type = "supply"
            pattern = "RBD" if prior_dir == "rally" else "DBD"
            top = float(row["high"])
            bottom = float(row["body_bottom"])

        freshness, tests, broken, first_touch_time, last_touch_time = _zone_freshness(d, i, zone_type, top, bottom)

        departure_strength = abs(departure_atr)
        quality_score = 0
        quality_score += min(4, departure_strength)
        quality_score += min(3, departure_vol_ratio)
        quality_score += {"fresh": 3, "one_test": 1.5, "multiple_tests": 0.5}.get(freshness, 0)
        quality_score += {"1D": 3.0, "4H": 2.0, "3H": 1.5, "2H": 1.0, "90m": 0.75, "1H": 0.5}.get(timeframe, 0)
        zone_width = max(0.0, top - bottom)
        midpoint = max((top + bottom) / 2.0, 0.01)
        base_time = d.index[i]
        age_days = (d.index[-1] - base_time).total_seconds() / 86400.0
        zone_id = f"{symbol}|{timeframe}|{zone_type}|{base_time.isoformat()}|{bottom:.2f}|{top:.2f}"

        rows.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "zone_id": zone_id,
            "zone_timeframe": timeframe,
            "zone_type": zone_type,
            "zone_created_time": base_time.isoformat(),
            "pattern": pattern,
            "base_time": base_time.isoformat(),
            "zone_top": round(top, 2),
            "zone_bottom": round(bottom, 2),
            "zone_age_days": round(float(age_days), 2),
            "zone_age_bars": int(max(0, len(d) - i - 1)),
            "zone_width": round(float(zone_width), 2),
            "zone_width_pct": round(float(zone_width / midpoint * 100.0), 3),
            "zone_height_atr": round(float(zone_width / row["avg_range"]), 3),
            "base_open": round(float(row["open"]), 2),
            "base_high": round(float(row["high"]), 2),
            "base_low": round(float(row["low"]), 2),
            "base_close": round(float(row["close"]), 2),
            "base_range": round(float(row["range"]), 2),
            "base_volume_ratio": round(float(vol_ratio), 2),
            "prior_atr": round(float(prior_atr), 2),
            "departure_atr": round(float(departure_atr), 2),
            "departure_atr_expansion": round(float(abs(departure_atr)), 3),
            "departure_strength_score": round(float(min(4, departure_strength) + min(3, departure_vol_ratio)), 3),
            "departure_body_ratio": round(float(departure_body_multiple), 2),
            "departure_candle_count": int(RULES.departure_window),
            "departure_body_vs_base_body": round(float(departure_body_multiple), 2),
            "departure_volume_ratio": round(float(departure_vol_ratio), 2),
            "freshness": freshness,
            "tests": int(tests),
            "touch_count_before_snapshot": int(tests),
            "first_touch_time": first_touch_time,
            "last_touch_time": last_touch_time,
            "fresh_zone": bool(freshness == "fresh" and not broken),
            "broken": bool(broken),
            "nested_inside_higher_tf": False,
            "higher_tf_zone_count": 1,
            "overlapping_zone_count": 1,
            "confluence_score": 0,
            "trend_alignment": "",
            "distance_to_next_supply": "",
            "distance_to_next_demand": "",
            "gap_created_zone": False,
            "gap_into_zone": False,
            "gap_away_from_zone": False,
            "base_quality_score": round(float(quality_score), 2),
            "historical_reaction_count": "",
            "quality_score": round(float(quality_score), 2),
        })

    return rows
