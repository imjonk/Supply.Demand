from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
from datetime import time
import pandas as pd

from config import DATA_DIR, REPORT_DIR, WATCHLIST, TIMEFRAMES, RULES
from data_loader import load_symbol_csv, regular_session_only, aggregate_bars, MARKET_TZ
from zone_detector import detect_zones
from core.watchlist_engine import (
    build_watchlist_from_zone_snapshot,
    _filter_final_report,
    _active_zones_for_watchlist,
    merge_overlapping_zones,
)
from movement_context import compute_symbol_movement_context, classify_market_session


def _parse_symbols(value: str | None) -> list[str]:
    if not value:
        return list(WATCHLIST)
    return [s.strip().upper() for s in value.replace(';', ',').split(',') if s.strip()]


def _available_trading_dates(symbols: list[str]) -> list:
    dates = set()
    for sym in symbols:
        path = DATA_DIR / f"{sym}_5M.csv"
        if not path.exists():
            continue
        try:
            df = regular_session_only(load_symbol_csv(path))
        except Exception:
            continue
        for d in df.index.normalize().unique():
            dates.add(pd.Timestamp(d).date())
    return sorted(dates)


def _in_range(dates, start: str | None, end: str | None):
    start_d = pd.to_datetime(start).date() if start else None
    end_d = pd.to_datetime(end).date() if end else None
    out = []
    for d in dates:
        if start_d and d < start_d:
            continue
        if end_d and d > end_d:
            continue
        out.append(d)
    return out


def _asof_ts(day) -> pd.Timestamp:
    return pd.Timestamp(f"{day} 16:00", tz=MARKET_TZ)


def _load_symbol_sources(symbol: str, max_zone_asof: pd.Timestamp, max_context_asof: pd.Timestamp | None = None):
    """Load symbol bars for two separate purposes.

    raw_zone/rth_zone are clipped to the prior RTH as-of and are the only data
    used to create/state zones. raw_context may extend into the next session's
    premarket so snapshot watchlists can simulate an 8:00 AM current price
    without letting extended-hours data mutate zones.
    """
    path = DATA_DIR / f"{symbol}_5M.csv"
    if not path.exists():
        return None, None, None
    raw_all = load_symbol_csv(path)
    raw_zone = raw_all.loc[raw_all.index <= max_zone_asof.tz_convert("UTC")].copy()
    raw_context = raw_all
    if max_context_asof is not None:
        raw_context = raw_all.loc[raw_all.index <= max_context_asof.tz_convert("UTC")].copy()
    rth_zone = regular_session_only(raw_zone)
    if rth_zone.empty:
        return None, None, None
    return raw_zone, rth_zone, raw_context


def _precompute_bars_and_zones(symbols: list[str], max_asof: pd.Timestamp, max_context_asof: pd.Timestamp | None = None):
    """Load/resample once and detect the complete zone universe once.

    Snapshot generation then only updates each zone's lifecycle as of the daily
    cutoff. This avoids the v0.34 behavior of repeatedly redetecting zones from
    scratch for every historical day.
    """
    bars_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    five_min_cache: Dict[str, pd.DataFrame] = {}
    raw_5m_cache: Dict[str, pd.DataFrame] = {}
    all_zone_rows = []
    skipped = []

    for symbol in symbols:
        raw, rth, raw_context = _load_symbol_sources(symbol, max_asof, max_context_asof)
        if raw is None or rth is None or rth.empty:
            skipped.append(f"{symbol}: missing/empty 5M data")
            continue
        five_min_cache[symbol] = rth
        raw_5m_cache[symbol] = raw_context if raw_context is not None else raw

        for label, rule in TIMEFRAMES.items():
            if label == "1D":
                daily_path = DATA_DIR / f"{symbol}_1D.csv"
                if daily_path.exists():
                    daily_raw = load_symbol_csv(daily_path)
                    daily_raw = daily_raw.loc[daily_raw.index <= max_asof.tz_convert("UTC")].copy()
                    bars = daily_raw.tz_convert(MARKET_TZ) if daily_raw.index.tz is not None else daily_raw
                else:
                    bars = aggregate_bars(raw, rule)
            else:
                bars = aggregate_bars(raw, rule)
            if bars.empty:
                continue
            bars_cache[(symbol, label)] = bars
            zones = detect_zones(bars, symbol, label)
            for z in zones:
                z = dict(z)
                # A detected zone becomes actionable only after the departure
                # candle has closed. With the current rule this is normally the
                # candle immediately after the base, but keep it derived from
                # RULES.departure_window for consistency.
                try:
                    base_time = pd.Timestamp(z["base_time"])
                    loc = bars.index.get_loc(base_time)
                    dep_loc = min(len(bars.index) - 1, int(loc) + int(RULES.departure_window))
                    z["active_after_time"] = bars.index[dep_loc].isoformat()
                except Exception:
                    z["active_after_time"] = z.get("base_time", "")
                all_zone_rows.append(z)

    zones_df = pd.DataFrame(all_zone_rows)
    return bars_cache, five_min_cache, raw_5m_cache, zones_df, skipped


def _state_one_zone(row: pd.Series, bars: pd.DataFrame, asof: pd.Timestamp) -> dict | None:
    """Recalculate tests/broken/freshness using only candles known by as-of."""
    try:
        base_time = pd.Timestamp(row["base_time"])
        active_after = pd.Timestamp(row.get("active_after_time") or row["base_time"])
    except Exception:
        return None
    if active_after > asof:
        return None

    top = float(row["zone_top"])
    bottom = float(row["zone_bottom"])
    zone_type = str(row["zone_type"])

    subsequent = bars[(bars.index > active_after) & (bars.index <= asof)]
    tests = 0
    broken = False
    broken_time = ""
    last_test_time = ""
    for ts, b in subsequent.iterrows():
        overlaps = (float(b["low"]) <= top) and (float(b["high"]) >= bottom)
        if zone_type == "demand":
            if float(b["close"]) < bottom:
                broken = True
                broken_time = ts.isoformat()
                break
            if overlaps:
                tests += 1
                last_test_time = ts.isoformat()
        else:
            if float(b["close"]) > top:
                broken = True
                broken_time = ts.isoformat()
                break
            if overlaps:
                tests += 1
                last_test_time = ts.isoformat()

    if broken:
        freshness = "broken"
    elif tests == 0:
        freshness = "fresh"
    elif tests == 1:
        freshness = "one_test"
    else:
        freshness = "multiple_tests"

    tf_bonus = {"1D": 3.0, "4H": 2.0, "3H": 1.5, "2H": 1.0, "90m": 0.75, "1H": 0.5}.get(str(row.get("timeframe", "")), 0)
    fresh_bonus = {"fresh": 3, "one_test": 1.5, "multiple_tests": 0.5, "broken": 0}.get(freshness, 0)
    try:
        departure_strength = abs(float(row.get("departure_atr", 0) or 0))
    except Exception:
        departure_strength = 0.0
    try:
        dep_vol = float(row.get("departure_volume_ratio", 0) or 0)
    except Exception:
        dep_vol = 0.0
    quality_score = min(4, departure_strength) + min(3, dep_vol) + fresh_bonus + tf_bonus

    out = row.to_dict()
    out.update({
        "tests": int(tests),
        "freshness": freshness,
        "broken": bool(broken),
        "broken_time": broken_time,
        "last_test_time": last_test_time,
        "quality_score": round(float(quality_score), 2),
        "snapshot_as_of": asof.isoformat(),
    })
    return out


def _zones_as_of(all_zones: pd.DataFrame, bars_cache: Dict[Tuple[str, str], pd.DataFrame], asof: pd.Timestamp) -> pd.DataFrame:
    if all_zones.empty:
        return all_zones.copy()
    rows = []
    # Cheap prefilter before the per-zone lifecycle scan.
    z = all_zones.copy()
    z["_active_after"] = pd.to_datetime(z["active_after_time"], errors="coerce", utc=True).dt.tz_convert(MARKET_TZ)
    z = z[z["_active_after"].notna() & (z["_active_after"] <= asof)]
    for _, row in z.iterrows():
        key = (str(row["symbol"]), str(row["timeframe"]))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            continue
        state = _state_one_zone(row, bars, asof)
        if state is not None:
            rows.append(state)
    return pd.DataFrame(rows)


def _latest_prices_as_of(five_min_cache: Dict[str, pd.DataFrame], asof: pd.Timestamp) -> dict:
    prices = {}
    for sym, df in five_min_cache.items():
        sub = df[df.index <= asof]
        if not sub.empty:
            prices[sym] = float(sub["close"].iloc[-1])
    return prices


def _parse_preopen_time(value: str) -> time:
    try:
        hh, mm = str(value).split(":", 1)
        return time(int(hh), int(mm))
    except Exception as exc:
        raise SystemExit(f"Invalid --preopen-time {value!r}; expected HH:MM, e.g. 08:00") from exc


def _preopen_ts(day, preopen_time: time) -> pd.Timestamp:
    return pd.Timestamp.combine(pd.Timestamp(day).date(), preopen_time).tz_localize(MARKET_TZ)


def _latest_prices_and_times_as_of(raw_cache: Dict[str, pd.DataFrame], asof: pd.Timestamp, fallback_rth_prices: dict | None = None, fallback_asof: pd.Timestamp | None = None) -> tuple[dict, dict]:
    """Return latest available extended-hours/current price as of a simulated context time.

    If no extended-hours bar is available by the context timestamp for a symbol,
    fall back to the prior RTH close so historical snapshots remain complete.
    """
    prices: dict[str, float] = {}
    asofs: dict[str, str] = {}
    fallback_rth_prices = fallback_rth_prices or {}
    for sym, df in raw_cache.items():
        local = df.tz_convert(MARKET_TZ) if getattr(df.index, "tz", None) is not None else df.copy()
        sub = local[local.index <= asof]
        if not sub.empty:
            prices[sym] = float(sub["close"].iloc[-1])
            asofs[sym] = sub.index[-1].isoformat()
        elif sym in fallback_rth_prices:
            prices[sym] = float(fallback_rth_prices[sym])
            asofs[sym] = (fallback_asof.isoformat() if fallback_asof is not None else "")
    return prices, asofs


def _symbol_context_as_of(raw_cache: Dict[str, pd.DataFrame], rth_cache: Dict[str, pd.DataFrame], latest_prices: dict, latest_price_as_of: dict, context_asof: pd.Timestamp) -> dict:
    ctx = {}
    for sym, price in latest_prices.items():
        raw = raw_cache.get(sym)
        rth = rth_cache.get(sym)
        if raw is None or rth is None or raw.empty or rth.empty:
            continue
        raw_local = raw.tz_convert(MARKET_TZ) if getattr(raw.index, "tz", None) is not None else raw.copy()
        raw_prior = raw_local[raw_local.index <= context_asof]
        rth_prior = rth[rth.index <= context_asof]
        if raw_prior.empty or rth_prior.empty:
            continue
        asof_value = latest_price_as_of.get(sym, context_asof.isoformat())
        sc = compute_symbol_movement_context(sym, raw_prior, rth_prior, float(price), asof_value)
        sc["current_price_session"] = sc.get("latest_price_session", classify_market_session(asof_value))
        sc["snapshot_context_time"] = context_asof.isoformat()
        sc["snapshot_context_type"] = "preopen"
        ctx[sym] = sc
    return ctx


def _write_preopen_context_snapshot(path: Path, symbols: list[str], prior_prices: dict, preopen_prices: dict, preopen_asofs: dict, symbol_ctx: dict, test_date, prior_asof: pd.Timestamp, preopen_asof: pd.Timestamp) -> None:
    rows = []
    for sym in symbols:
        if sym not in preopen_prices and sym not in prior_prices:
            continue
        ctx = symbol_ctx.get(sym, {})
        rows.append({
            "test_date": str(test_date),
            "zone_snapshot_as_of": prior_asof.isoformat(),
            "preopen_context_as_of": preopen_asof.isoformat(),
            "symbol": sym,
            "prior_rth_close": prior_prices.get(sym),
            "preopen_price": preopen_prices.get(sym),
            "preopen_price_as_of": preopen_asofs.get(sym, ""),
            "preopen_price_session": classify_market_session(preopen_asofs.get(sym, preopen_asof.isoformat())),
            "preopen_gap_pct": ctx.get("gap_pct"),
            "preopen_gap_direction": ctx.get("gap_direction"),
            "preopen_gap_abs": ctx.get("gap_abs"),
            "preopen_move_direction": ctx.get("recent_move_direction"),
            "preopen_move_strength": ctx.get("recent_move_strength"),
            "preopen_volume_state": ctx.get("volume_state"),
            "preopen_vpa_state": ctx.get("vpa_state"),
            "preopen_price_vs_9ema": ctx.get("price_vs_9ema"),
            "preopen_price_vs_vwap": ctx.get("price_vs_vwap"),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Build daily historical watchlist snapshots using a cached forward zone ledger.")
    parser.add_argument("--start", default=None, help="First trading date to test, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Last trading date to test, YYYY-MM-DD.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols. Defaults to config WATCHLIST.")
    parser.add_argument("--max-days", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--save-raw-zones", action="store_true", help="Also save each day's raw zone snapshot. This is slower and uses more disk.")
    parser.add_argument("--preopen-time", default="08:00", help="Historical preopen context time in New York time, HH:MM. Default: 08:00.")
    parser.add_argument("--no-preopen-context", action="store_true", help="Disable 8:00 AM preopen/movement-context snapshot exports.")
    args = parser.parse_args()
    preopen_time = _parse_preopen_time(args.preopen_time)

    symbols = _parse_symbols(args.symbols)
    dates = _in_range(_available_trading_dates(symbols), args.start, args.end)
    if len(dates) < 2:
        raise SystemExit("Not enough 5M regular-session trading dates available to build snapshots.")
    if args.max_days:
        dates = dates[: args.max_days]

    max_asof = _asof_ts(dates[-2] if len(dates) > 1 else dates[-1])
    max_context_asof = _preopen_ts(dates[-1], preopen_time) if not args.no_preopen_context else max_asof
    print(f"Precomputing candles and zone universe once through {max_asof.date()}...")
    if not args.no_preopen_context:
        print(f"Loading extended-hours context through simulated preopen {max_context_asof}...")
    bars_cache, five_min_cache, raw_5m_cache, all_zones, skipped = _precompute_bars_and_zones(symbols, max_asof, max_context_asof)
    print(f"Detected {len(all_zones):,} historical raw zones across {len(bars_cache):,} symbol/timeframe series.")

    base = REPORT_DIR / "backtest" / "snapshots"
    base.mkdir(parents=True, exist_ok=True)
    preopen_base = REPORT_DIR / "backtest" / "preopen_snapshots"
    if not args.no_preopen_context:
        preopen_base.mkdir(parents=True, exist_ok=True)
    rows = []

    # Test date uses the previous trading day's close as the as-of snapshot.
    for idx in range(1, len(dates)):
        test_date = dates[idx]
        as_of_date = dates[idx - 1]
        asof = _asof_ts(as_of_date)
        print(f"Exporting snapshot for {test_date} using cached zone ledger as-of {as_of_date}...")
        try:
            zones_df = _zones_as_of(all_zones, bars_cache, asof)
            latest_prices = _latest_prices_as_of(five_min_cache, asof)
            active_zones = _active_zones_for_watchlist(zones_df)
            merged_zones = merge_overlapping_zones(active_zones)
            latest_price_as_of = {sym: asof.isoformat() for sym in latest_prices}
            watch_df = build_watchlist_from_zone_snapshot(
                zones_df,
                latest_prices,
                {
                    "as_of_date": str(as_of_date),
                    "snapshot_as_of": asof.isoformat(),
                    "current_price_as_of": latest_price_as_of,
                },
            )
            final_watch = _filter_final_report(watch_df)
        except Exception as exc:
            rows.append({"test_date": str(test_date), "as_of_date": str(as_of_date), "status": "error", "error": str(exc)})
            print(f"  ERROR: {exc}")
            continue

        prefix = base / str(test_date)
        if args.save_raw_zones:
            zones_path = prefix.with_name(prefix.name + "_zones.csv")
            zones_df.to_csv(zones_path, index=False)
        else:
            zones_path = ""
        active_path = prefix.with_name(prefix.name + "_active_zones.csv")
        merged_path = prefix.with_name(prefix.name + "_merged_zones.csv")
        scenarios_path = prefix.with_name(prefix.name + "_scenarios.csv")
        final_path = prefix.with_name(prefix.name + "_final_watchlist.csv")

        preopen_context_path = ""
        preopen_scenarios_path = ""
        preopen_final_path = ""
        preopen_count = 0
        preopen_final_count = 0
        if not args.no_preopen_context:
            preopen_asof = _preopen_ts(test_date, preopen_time)
            preopen_prices, preopen_price_as_of = _latest_prices_and_times_as_of(raw_5m_cache, preopen_asof, latest_prices, asof)
            preopen_symbol_context = _symbol_context_as_of(raw_5m_cache, five_min_cache, preopen_prices, preopen_price_as_of, preopen_asof)
            preopen_watch_df = build_watchlist_from_zone_snapshot(
                zones_df,
                preopen_prices,
                {
                    "as_of_date": str(as_of_date),
                    "test_date": str(test_date),
                    "snapshot_as_of": asof.isoformat(),
                    "snapshot_context_time": preopen_asof.isoformat(),
                    "snapshot_context_type": "preopen",
                    "current_price_as_of": preopen_price_as_of,
                    "symbol_movement_context": preopen_symbol_context,
                },
            )
            preopen_final_watch = _filter_final_report(preopen_watch_df)
            preopen_prefix = preopen_base / str(test_date)
            preopen_context_path = str(preopen_prefix.with_name(preopen_prefix.name + "_preopen_context.csv"))
            preopen_scenarios_path = str(preopen_prefix.with_name(preopen_prefix.name + "_movement_context_watchlist.csv"))
            preopen_final_path = str(preopen_prefix.with_name(preopen_prefix.name + "_preopen_final_watchlist.csv"))
            _write_preopen_context_snapshot(Path(preopen_context_path), symbols, latest_prices, preopen_prices, preopen_price_as_of, preopen_symbol_context, test_date, asof, preopen_asof)
            preopen_watch_df.to_csv(preopen_scenarios_path, index=False)
            preopen_final_watch.to_csv(preopen_final_path, index=False)
            preopen_count = len(preopen_watch_df)
            preopen_final_count = len(preopen_final_watch)

        active_zones.to_csv(active_path, index=False)
        merged_zones.to_csv(merged_path, index=False)
        watch_df.to_csv(scenarios_path, index=False)
        final_watch.to_csv(final_path, index=False)

        rows.append({
            "test_date": str(test_date),
            "as_of_date": str(as_of_date),
            "status": "ok",
            "raw_zones": len(zones_df),
            "active_zones": len(active_zones),
            "merged_zones": len(merged_zones),
            "scenarios": len(watch_df),
            "final_setups": len(final_watch),
            "preopen_context_time": (_preopen_ts(test_date, preopen_time).isoformat() if not args.no_preopen_context else ""),
            "preopen_scenarios": preopen_count,
            "preopen_final_setups": preopen_final_count,
            "scenario_file": str(scenarios_path),
            "final_file": str(final_path),
            "active_zones_file": str(active_path),
            "merged_zones_file": str(merged_path),
            "raw_zones_file": str(zones_path),
            "preopen_context_file": preopen_context_path,
            "preopen_scenario_file": preopen_scenarios_path,
            "preopen_final_file": preopen_final_path,
        })

    manifest = REPORT_DIR / "backtest" / "snapshot_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Wrote {manifest}")
    if skipped:
        skipped_path = REPORT_DIR / "backtest" / "snapshot_skipped_symbols.txt"
        skipped_path.write_text("\n".join(skipped))
        print(f"Skipped symbols written to {skipped_path}")


if __name__ == "__main__":
    main()
