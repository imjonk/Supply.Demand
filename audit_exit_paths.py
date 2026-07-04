
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd

from config import REPORT_DIR
from replay_backtest import (
    _SYMBOL_5M_CACHE,
    _load_symbol_5m_all,
    _simulate_exit,
    _simulate_reversal_management_variants,
    _scenario_kind,
    _is_reversal_rejection_kind,
)


_SYMBOL_DAY_GROUP_CACHE: dict[str, dict[object, pd.DataFrame]] = {}
_DAY_DF_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def _prepare_day_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    if "vwap" not in df.columns or df["vwap"].isna().all():
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].replace(0, pd.NA).cumsum()
    df["avg_volume20"] = df["volume"].rolling(20, min_periods=3).mean()
    df["volume_ratio"] = df["volume"] / df["avg_volume20"].replace(0, pd.NA)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).replace(0, pd.NA)
    df["body_ratio"] = df["body"] / df["range"]
    df["avg_range20"] = df["range"].rolling(20, min_periods=3).mean()
    df["range_ratio"] = df["range"] / df["avg_range20"].replace(0, pd.NA)
    return df


def _fast_load_day_5m(symbol: str, day: str) -> pd.DataFrame:
    symbol = str(symbol).upper()
    key = (symbol, str(day))
    if key in _DAY_DF_CACHE:
        return _DAY_DF_CACHE[key]
    if symbol not in _SYMBOL_DAY_GROUP_CACHE:
        all_df = _load_symbol_5m_all(symbol)
        if all_df.empty:
            _SYMBOL_DAY_GROUP_CACHE[symbol] = {}
        else:
            _SYMBOL_DAY_GROUP_CACHE[symbol] = {d: g.copy() for d, g in all_df.groupby(all_df.index.date)}
    target = pd.to_datetime(day).date()
    raw = _SYMBOL_DAY_GROUP_CACHE.get(symbol, {}).get(target, pd.DataFrame())
    _DAY_DF_CACHE[key] = _prepare_day_5m(raw) if raw is not None and not raw.empty else pd.DataFrame()
    return _DAY_DF_CACHE[key]

BASE_EXIT_COLS = {
    "exit_time", "exit_price", "exit_reason", "r_multiple", "mfe_r", "mae_r",
    "reached_1r", "reached_2r", "reached_3r", "target_price", "stop_price", "risk_per_share",
}

AUDIT_COPY_COLS = [
    "first_event_after_entry", "first_event_r_result", "first_1r_time", "first_2r_time", "first_3r_time",
    "first_stop_time", "first_target_3r_time", "first_ema_protection_time", "actual_exit_event_time",
    "mfe_r_until_exit", "mae_r_until_exit", "mfe_r_full_day", "mae_r_full_day",
    "target_and_stop_same_candle", "one_r_and_stop_same_candle", "two_r_and_stop_same_candle",
    "three_r_and_stop_same_candle", "same_candle_ambiguity", "conservative_intrabar_result",
    "optimistic_intrabar_result", "neutral_intrabar_result", "intrabar_policy_used",
    "stop_after_reached_1r", "stop_after_reached_2r", "stop_after_reached_3r",
    "target_available_but_not_taken", "ema_protection_available_but_not_taken",
    "mfe_after_exit_detected", "mae_after_exit_detected", "unrealistic_r_outlier", "risk_pct_of_entry",
]


def _row_date(row) -> str:
    if "entry_time" in row and pd.notna(row["entry_time"]):
        return str(pd.to_datetime(row["entry_time"]).date())
    if "test_date" in row and pd.notna(row["test_date"]):
        return str(pd.to_datetime(row["test_date"]).date())
    raise ValueError("Trade row has no entry_time or test_date")


def enrich_exit_paths(trades: pd.DataFrame, rr: float = 3.0, ema_exit_after_r: float = 1.0, ema_confirm_bars: int = 2) -> pd.DataFrame:
    out = trades.copy()
    updates = []
    for pos, (idx, row) in enumerate(out.iterrows()):
        if pos and pos % 500 == 0:
            # Keep long full-sample enrichments stable on memory-constrained machines.
            _DAY_DF_CACHE.clear()
            _SYMBOL_DAY_GROUP_CACHE.clear()
            _SYMBOL_5M_CACHE.clear()
        update = {}
        try:
            sym = str(row.get("symbol", "")).upper()
            day = _row_date(row)
            day_df = _fast_load_day_5m(sym, day)
            if day_df.empty:
                update["exit_path_audit_status"] = "missing_5m_day_data"
                updates.append((idx, update))
                continue
            entry_time = pd.to_datetime(row.get("entry_time"))
            if entry_time.tzinfo is None and day_df.index.tz is not None:
                entry_time = entry_time.tz_localize(day_df.index.tz)
            elif entry_time.tzinfo is not None and day_df.index.tz is not None:
                entry_time = entry_time.tz_convert(day_df.index.tz)
            scenario = pd.Series(row)
            entry = {
                "entry_price": float(row.get("entry_price")),
                "entry_kind": row.get("entry_kind", _scenario_kind(scenario)),
            }
            info = _simulate_exit(day_df, entry_time, scenario, entry, rr, ema_exit_after_r, ema_confirm_bars)
            if "skip_reason" in info:
                update["exit_path_audit_status"] = info["skip_reason"]
                updates.append((idx, update))
                continue
            update["exit_path_audit_status"] = "ok"
            for c in AUDIT_COPY_COLS:
                update[c] = info.get(c)
            update["audit_recalc_exit_time"] = info.get("exit_time")
            update["audit_recalc_exit_reason"] = info.get("exit_reason")
            update["audit_recalc_r_multiple"] = info.get("r_multiple")
            update["audit_recalc_exit_price"] = info.get("exit_price")
            update["audit_exit_reason_matches_baseline"] = str(info.get("exit_reason")) == str(row.get("exit_reason"))
            try:
                update["audit_r_delta_vs_baseline"] = round(float(info.get("r_multiple")) - float(row.get("r_multiple")), 3)
            except Exception:
                update["audit_r_delta_vs_baseline"] = None
            kind = row.get("entry_kind", _scenario_kind(scenario))
            if _is_reversal_rejection_kind(str(kind)):
                update.update(_simulate_reversal_management_variants(day_df, entry_time, scenario, entry, rr, ema_confirm_bars))
        except Exception as exc:
            update["exit_path_audit_status"] = f"error:{type(exc).__name__}"
            update["exit_path_audit_error"] = str(exc)[:200]
        updates.append((idx, update))
    if updates:
        upd_df = pd.DataFrame([u for _, u in updates], index=[i for i, _ in updates])
        for col in upd_df.columns:
            out[col] = upd_df[col]
    return out


def main():
    parser = argparse.ArgumentParser(description="Enrich existing backtest trades with exit-path audit fields without rerunning entries.")
    parser.add_argument("--trades", default=str(REPORT_DIR / "backtest" / "trades.csv"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--rr", type=float, default=3.0)
    parser.add_argument("--ema-exit-after-r", type=float, default=1.0)
    parser.add_argument("--ema-exit-confirm-bars", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=500, help="For large files, audit in fresh subprocess chunks to keep memory stable.")
    parser.add_argument("--no-chunk", action="store_true", help="Internal option used by chunk subprocesses.")
    args = parser.parse_args()
    path = Path(args.trades)
    out_path = Path(args.out) if args.out else path
    trades = pd.read_csv(path)

    if not args.no_chunk and args.chunk_size and len(trades) > args.chunk_size:
        with tempfile.TemporaryDirectory(prefix="exit_path_audit_") as tmp:
            tmpdir = Path(tmp)
            audited_files = []
            for part, start in enumerate(range(0, len(trades), args.chunk_size)):
                chunk = trades.iloc[start:start + args.chunk_size].copy()
                chunk_in = tmpdir / f"chunk_{part:03d}.csv"
                chunk_out = tmpdir / f"chunk_{part:03d}_audited.csv"
                chunk.to_csv(chunk_in, index=False)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--trades", str(chunk_in),
                    "--out", str(chunk_out),
                    "--rr", str(args.rr),
                    "--ema-exit-after-r", str(args.ema_exit_after_r),
                    "--ema-exit-confirm-bars", str(args.ema_exit_confirm_bars),
                    "--no-chunk",
                ]
                print(f"Auditing chunk {part + 1} ({len(chunk)} rows)...")
                subprocess.run(cmd, check=True)
                audited_files.append(chunk_out)
            enriched = pd.concat([pd.read_csv(f) for f in audited_files], ignore_index=True)
            enriched.to_csv(out_path, index=False)
    else:
        enriched = enrich_exit_paths(trades, args.rr, args.ema_exit_after_r, args.ema_exit_confirm_bars)
        enriched.to_csv(out_path, index=False)

    ok = int((enriched.get("exit_path_audit_status") == "ok").sum()) if "exit_path_audit_status" in enriched.columns else 0
    print(f"Wrote {out_path} ({ok}/{len(enriched)} rows audited)")


if __name__ == "__main__":
    main()
