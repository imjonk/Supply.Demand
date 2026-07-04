"""Generate chart-style watchlist zone maps and HH/HL vs LH/LL labels.

v0.36.6 add-on. This script reads the current watchlist outputs and adds
visual context without changing scanner selection logic.

Outputs:
  reports/watchlist_zone_map.html
  reports/watchlist_zone_map.csv
  reports/watchlist.html               (regenerated with mini chart cards)
  reports/watchlist.csv                (same final rows plus visual/context columns)
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from config import DATA_DIR, REPORT_DIR
from data_loader import load_symbol_csv, regular_session_only
from watchlist import (
    _add_watchlist_visual_context,
    _format_price_timestamp,
    _compute_symbol_structure_context,
    generate_html_report,
    generate_watchlist_zone_map_html,
)
from movement_context import (
    compute_symbol_movement_context, load_zone_reaction_history,
    enrich_movement_context, generate_movement_context_html,
)


def _read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _structure_context_for_symbols(symbols, price_lookup):
    out = {}
    for sym in sorted(set(str(s).upper() for s in symbols if str(s).strip())):
        path_5m = DATA_DIR / f"{sym}_5M.csv"
        if not path_5m.exists():
            continue
        try:
            raw_5m = load_symbol_csv(path_5m)
            daily_raw = None
            path_1d = DATA_DIR / f"{sym}_1D.csv"
            if path_1d.exists():
                try:
                    daily_raw = load_symbol_csv(path_1d)
                except Exception:
                    daily_raw = None
            out[sym] = _compute_symbol_structure_context(raw_5m, daily_raw, price_lookup.get(sym))
        except Exception as exc:
            out[sym] = {
                "5m": {"bias": "Insufficient structure", "detail": f"5M unavailable: {exc}"},
                "15m": {"bias": "Insufficient structure", "detail": "15M unavailable"},
                "1d": {"bias": "Insufficient structure", "detail": "Daily unavailable"},
                "alignment": "Range-bound / insufficient",
            }
    return out



def _movement_context_for_symbols(symbols, price_lookup, price_asof_lookup):
    out = {}
    for sym in sorted(set(str(s).upper() for s in symbols if str(s).strip())):
        path_5m = DATA_DIR / f"{sym}_5M.csv"
        if not path_5m.exists():
            continue
        try:
            raw = load_symbol_csv(path_5m)
            rth = regular_session_only(raw)
            asof = price_asof_lookup.get(sym) or (raw.tz_convert('America/New_York').index[-1] if not raw.empty else None)
            out[sym] = compute_symbol_movement_context(sym, raw, rth, price_lookup.get(sym), asof)
            out[sym]['current_price_session'] = out[sym].get('latest_price_session', '')
        except Exception:
            pass
    return out


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    final_path = REPORT_DIR / "watchlist.csv"
    all_path = REPORT_DIR / "watchlist_all_candidates.csv"
    merged_zones_path = REPORT_DIR / "merged_zones.csv"

    final_df = _read_csv(final_path)
    all_df = _read_csv(all_path)
    zones_df = _read_csv(merged_zones_path)

    if final_df.empty and all_df.empty:
        raise SystemExit("No watchlist rows found. Run watchlist.py first.")

    source_df = pd.concat([final_df, all_df], ignore_index=True) if not all_df.empty else final_df.copy()
    price_lookup = {}
    price_asof_lookup = {}
    if not source_df.empty and "symbol" in source_df.columns and "current_price" in source_df.columns:
        for _, row in source_df.dropna(subset=["symbol", "current_price"]).iterrows():
            try:
                sym = str(row["symbol"]).upper()
                price_lookup[sym] = float(row["current_price"])
                existing_as_of = str(row.get("current_price_as_of", "")).strip() if row.get("current_price_as_of") is not None else ""
                if existing_as_of and existing_as_of.lower() != "nan":
                    price_asof_lookup[sym] = existing_as_of
            except Exception:
                pass

    symbols = source_df["symbol"].dropna().astype(str).str.upper().tolist() if "symbol" in source_df.columns else []
    for sym in sorted(set(symbols)):
        if sym in price_asof_lookup:
            continue
        path_5m = DATA_DIR / f"{sym}_5M.csv"
        if not path_5m.exists():
            continue
        try:
            bars = regular_session_only(load_symbol_csv(path_5m))
            if not bars.empty:
                price_asof_lookup[sym] = _format_price_timestamp(bars.index[-1])
        except Exception:
            pass

    structure_context = _structure_context_for_symbols(symbols, price_lookup)

    enriched_final = _add_watchlist_visual_context(final_df, zones_df, structure_context, price_lookup, price_asof_lookup)
    enriched_all = _add_watchlist_visual_context(all_df, zones_df, structure_context, price_lookup, price_asof_lookup)
    movement_context = _movement_context_for_symbols(symbols, price_lookup, price_asof_lookup)
    history = load_zone_reaction_history(REPORT_DIR)
    enriched_final = enrich_movement_context(enriched_final, movement_context, history)
    enriched_all = enrich_movement_context(enriched_all, movement_context, history)

    # Preserve existing primary outputs, but with added visual/context columns.
    enriched_final.to_csv(final_path, index=False)
    if not enriched_all.empty:
        enriched_all.to_csv(all_path, index=False)

    zone_map_csv = REPORT_DIR / "watchlist_zone_map.csv"
    zone_map_html = REPORT_DIR / "watchlist_zone_map.html"
    html_path = REPORT_DIR / "watchlist.html"
    movement_html = REPORT_DIR / "movement_context_watchlist.html"
    movement_csv = REPORT_DIR / "movement_context_watchlist.csv"

    now = datetime.now(ZoneInfo("America/New_York"))
    meta = {
        "report_date": now.strftime("%Y-%m-%d"),
        "report_datetime": now.strftime("%Y-%m-%d %I:%M %p %Z"),
        "price_source": "watchlist current_price column + latest regular-session 5M candles; per-symbol current_price_as_of shown in HTML",
        "latest_bar_times": {"n/a": "see data/*_5M.csv"},
    }

    enriched_final.to_csv(zone_map_csv, index=False)
    enriched_all.to_csv(movement_csv, index=False)
    zone_map_html.write_text(generate_watchlist_zone_map_html(enriched_final, meta), encoding="utf-8")
    html_path.write_text(generate_html_report(enriched_final, enriched_all, meta), encoding="utf-8")
    movement_html.write_text(generate_movement_context_html(enriched_all, meta), encoding="utf-8")

    print(f"Wrote {zone_map_csv}")
    print(f"Wrote {zone_map_html}")
    print(f"Updated {final_path}")
    print(f"Updated {html_path}")
    print(f"Wrote {movement_html}")
    print(f"Wrote {movement_csv}")
    print(f"Final candidates visualized: {len(enriched_final)}")


if __name__ == "__main__":
    main()
