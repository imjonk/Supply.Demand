from pathlib import Path
from datetime import time
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
MARKET_TZ = "America/New_York"


def load_symbol_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in ["trade_count", "vwap"]:
        if col in df.columns:
            numeric_cols.append(col)

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


def regular_session_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-session bars only, using New York time.

    All scanner price calculations intentionally ignore premarket and after-hours
    data. Returned index is timezone-aware America/New_York.
    """
    local = df.tz_convert(MARKET_TZ)
    mask = (local.index.time >= time(9, 30)) & (local.index.time < time(16, 0))
    return local.loc[mask]


def _regular_session_only(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for regular-session filtering."""
    return regular_session_only(df)


def aggregate_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate source bars into higher timeframes.

    Important for 90m: bars are anchored to 9:30 AM New York time, so 90m candles align like:
    9:30-11:00, 11:00-12:30, 12:30-2:00, 2:00-3:30, etc.

    Alpaca timestamps are UTC; output remains indexed in New York time for readable reports.
    """
    local = _regular_session_only(df)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    if "trade_count" in local.columns:
        agg["trade_count"] = "sum"

    out = local.resample(
        rule,
        label="left",
        closed="left",
        origin="start_day",
        offset="9h30min",
    ).agg(agg).dropna()

    if "vwap" in local.columns:
        # Approximate aggregate VWAP using candle volume.
        vwap_num = (local["vwap"] * local["volume"]).resample(
            rule, label="left", closed="left", origin="start_day", offset="9h30min"
        ).sum()
        vol = local["volume"].resample(
            rule, label="left", closed="left", origin="start_day", offset="9h30min"
        ).sum()
        out["vwap"] = vwap_num / vol.replace(0, pd.NA)

    return out
