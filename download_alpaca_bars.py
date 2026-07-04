import argparse
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import WATCHLIST, DATA_DIR

MARKET_TZ = ZoneInfo("America/New_York")


def parse_symbols(value: str | None):
    if not value:
        return WATCHLIST
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def parse_dt(value: str | None):
    if not value:
        return None
    # Accept YYYY-MM-DD or full ISO strings.
    if len(value) == 10:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def export_bars(client, symbols, timeframe, start, end, suffix: str) -> tuple[int, pd.DataFrame]:
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        print(f"WARNING: Alpaca returned no {suffix} bars. Check dates, symbols, or market-data permissions.")
        return 0, df

    exported = 0
    for symbol in df.index.get_level_values("symbol").unique():
        symbol_df = df.xs(symbol, level="symbol")
        out_path = DATA_DIR / f"{symbol}_{suffix}.csv"
        symbol_df.to_csv(out_path)
        exported += 1
        print(f"Wrote {out_path} ({len(symbol_df)} bars)")
    return exported, df


def _classify_market_session(ts) -> str:
    try:
        local = pd.Timestamp(ts)
        if local.tzinfo is None:
            local = local.tz_localize("UTC")
        local = local.tz_convert(MARKET_TZ)
        t = local.time()
        if t >= datetime.strptime("09:30", "%H:%M").time() and t < datetime.strptime("16:00", "%H:%M").time():
            return "RTH"
        if t >= datetime.strptime("04:00", "%H:%M").time() and t < datetime.strptime("09:30", "%H:%M").time():
            return "premarket"
        if t >= datetime.strptime("16:00", "%H:%M").time() and t < datetime.strptime("20:00", "%H:%M").time():
            return "aftermarket"
        return "outside_extended_hours"
    except Exception:
        return "unknown"


def write_latest_market_prices(symbols) -> int:
    """Write latest_market_prices.csv from the newest downloaded 5M bar per symbol.

    Zones should still be built from regular-session candles only. This snapshot is
    strictly for current price/proximity/watchlist fabrication, so it may use
    premarket or aftermarket bars when those are the latest available market data.
    """
    rows = []
    for symbol in symbols:
        path = DATA_DIR / f"{symbol}_5M.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            df = df.dropna(subset=[ts_col, "close"]).sort_values(ts_col)
            if df.empty:
                continue
            last = df.iloc[-1]
            ts = pd.Timestamp(last[ts_col])
            rows.append({
                "symbol": symbol,
                "price": float(last["close"]),
                "as_of": ts.tz_convert(MARKET_TZ).isoformat(),
                "session": _classify_market_session(ts),
                "source": "latest_downloaded_5M_bar",
            })
        except Exception as exc:
            print(f"WARNING: could not derive latest price for {symbol}: {exc}")
    if not rows:
        print("WARNING: no latest market price snapshot rows were written")
        return 0
    out = DATA_DIR / "latest_market_prices.csv"
    pd.DataFrame(rows).sort_values("symbol").to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} symbols) for watchlist current-price/proximity use")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Download Alpaca OHLCV bars for the trading scanner.")
    parser.add_argument("--days-5m", type=int, default=180, help="Number of calendar days of 5-minute bars to download. Default: 180 (~6 months).")
    parser.add_argument("--days-1d", type=int, default=365, help="Number of calendar days of daily bars to download. Default: 365 (~1 year).")
    parser.add_argument("--days", type=int, default=None, help="Backward-compatible alias for --days-5m. If provided, overrides --days-5m only.")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols, e.g. AMZN,NVDA,META")
    parser.add_argument("--start", type=str, default=None, help="Optional start date/time, e.g. 2026-03-01")
    parser.add_argument("--end", type=str, default=None, help="Optional end date/time. If omitted, uses now UTC minus --delay-minutes.")
    parser.add_argument("--delay-minutes", type=int, default=20, help="Delay the end time to avoid live-data permission errors. Default: 20")
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("ALPACA_KEY")
    secret_key = os.getenv("ALPACA_SECRET")

    if not api_key or not secret_key:
        raise RuntimeError("Missing ALPACA_KEY or ALPACA_SECRET in .env")

    symbols = parse_symbols(args.symbols)
    end = parse_dt(args.end) or (datetime.now(timezone.utc) - timedelta(minutes=args.delay_minutes))
    days_5m = args.days if args.days is not None else args.days_5m
    start_5m = parse_dt(args.start) or (end - timedelta(days=days_5m))
    start_1d = parse_dt(args.start) or (end - timedelta(days=args.days_1d))

    client = StockHistoricalDataClient(api_key, secret_key)
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Downloading bars for {len(symbols)} symbols")
    print(f"5M start: {start_5m.isoformat()}  (~{days_5m} calendar days)")
    print(f"1D start: {start_1d.isoformat()}  (~{args.days_1d} calendar days)")
    print(f"End:      {end.isoformat()}")
    if args.end is None:
        print(f"Delay:    {args.delay_minutes} minutes behind current UTC time")

    exported_5m, _bars_5m = export_bars(
        client=client,
        symbols=symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_5m,
        end=end,
        suffix="5M",
    )
    exported_1d, _bars_1d = export_bars(
        client=client,
        symbols=symbols,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start_1d,
        end=end,
        suffix="1D",
    )
    snapshot_rows = write_latest_market_prices(symbols)

    print(f"Exported {exported_5m} symbols with 5M bars and {exported_1d} symbols with 1D bars to {DATA_DIR}")
    print(f"Latest current-price snapshot rows: {snapshot_rows}")
    print("Next: python watchlist.py")


if __name__ == "__main__":
    main()
