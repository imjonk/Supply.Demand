from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import REPORT_DIR


BOOL_TRUE = {"true", "1", "yes", "y", "t"}
FUNNEL_GROUPS = {"funnel_by_scenario": "scenario_family", "funnel_by_symbol": "symbol", "funnel_by_timeframe": "timeframe", "funnel_by_grade": "watchlist_grade", "funnel_by_hour": "hour_of_day", "funnel_by_quality_score_bucket": "quality_score_bucket", "funnel_by_distance_bucket": "distance_bucket", "funnel_by_gap_direction": "gap_direction", "funnel_by_gap_size_bucket": "gap_size_bucket", "funnel_by_recent_movement_direction": "recent_move_direction", "funnel_by_recent_movement_strength": "recent_move_strength"}


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(BOOL_TRUE)


def _safe_pct(x: float) -> float:
    try:
        if pd.isna(x):
            return 0.0
        return round(float(x) * 100.0, 2)
    except Exception:
        return 0.0


def _fmt_num(x, places: int = 2) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.{places}f}"
    except Exception:
        return str(x)


def _table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows.</p>"
    shown = df.copy()
    if max_rows:
        shown = shown.head(max_rows)
    return shown.to_html(index=False, escape=False, classes="tbl")


def _card(label: str, value, sub: str = "") -> str:
    return f"<div class='card'><div class='label'>{label}</div><div class='value'>{value}</div><div class='subtle'>{sub}</div></div>"




def _quality_note(trades: pd.DataFrame) -> str:
    n = len(trades)
    if n < 30:
        return "Very small sample. Treat all percentages as directional only."
    if n < 100:
        return "Small sample. Use the results to form hypotheses, not final rules."
    return "Sample is large enough to compare buckets, but still validate changes out-of-sample."


def _edge_label(avg_r, profit_win_rate, trades) -> tuple[str, str]:
    try:
        ar = float(avg_r)
        wr = float(profit_win_rate)
        n = int(trades)
    except Exception:
        return ("Unknown", "neutral")
    if n < 20:
        return ("Low sample", "warn")
    if ar >= 0.20 and wr >= 52:
        return ("Positive edge", "good")
    if ar >= 0.05 and wr >= 48:
        return ("Slight edge", "ok")
    if ar > -0.05:
        return ("Flat / noisy", "warn")
    return ("Negative edge", "bad")


def _edge_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in ["avg_r", "profit_win_rate", "trades", "target_hit_rate", "reached_1r_rate", "reached_2r_rate"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    labels = out.apply(lambda r: _edge_label(r.get("avg_r"), r.get("profit_win_rate"), r.get("trades"))[0], axis=1)
    out.insert(min(len(group_cols), len(out.columns)), "edge_read", labels)
    return out


def _simple_bars(df: pd.DataFrame, label_col: str, value_col: str, title: str, sub: str = "", max_rows: int = 8, places: int = 2, suffix: str = "") -> str:
    if df is None or df.empty or label_col not in df.columns or value_col not in df.columns:
        return "<p class='muted'>No chart data.</p>"
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
    d = d.sort_values(value_col, ascending=False).head(max_rows)
    vals = d[value_col].tolist()
    max_abs = max([abs(v) for v in vals] + [1])
    rows = []
    for _, r in d.iterrows():
        label = str(r[label_col])
        val = float(r[value_col])
        w = max(2, min(100, abs(val) / max_abs * 100))
        cls = "pos" if val >= 0 else "neg"
        rows.append(f"<div class='barrow'><div class='barlabel'>{label}</div><div class='bartrack'><div class='barfill {cls}' style='width:{w:.1f}%'></div></div><div class='barval'>{val:.{places}f}{suffix}</div></div>")
    return f"<div class='viz'><h3>{title}</h3><div class='muted'>{sub}</div>{''.join(rows)}</div>"


def _two_col_definition(term: str, meaning: str) -> str:
    return f"<tr><td><strong>{term}</strong></td><td>{meaning}</td></tr>"


def _glossary_html() -> str:
    rows = []
    rows.append(_two_col_definition("Profit win rate", "Percent of simulated trades that ended with <code>r_multiple &gt; 0</code>. A +0.20R 9EMA exit counts as profitable; it does not require the 3R target."))
    rows.append(_two_col_definition("Target hit rate", "Percent of trades that reached the modeled full target, usually +3R in this backtest."))
    rows.append(_two_col_definition("Average R", "Average trade result measured in risk units. +0.25R means the average trade made one quarter of the initial risk."))
    rows.append(_two_col_definition("Total R", "Sum of all R-multiples. Useful for overall strategy contribution, but affected by trade count."))
    rows.append(_two_col_definition("MFE", "Maximum favorable excursion: the best unrealized R the trade reached after entry. This shows whether the idea moved in our favor even if the exit did not capture it."))
    rows.append(_two_col_definition("MAE", "Maximum adverse excursion: the worst unrealized R against the trade. Large negative MAE means the stop/position management needs review."))
    rows.append(_two_col_definition("Reached +1R/+2R/+3R", "Whether price touched that R milestone at any point after entry, not necessarily where the trade exited."))
    rows.append(_two_col_definition("9EMA protection", "Risk-management exit after the trade has moved enough to justify protection. This is separate from the original thesis target."))
    rows.append(_two_col_definition("Edge read", "A plain-English bucket from average R, profit win rate, and sample size. It is a guide, not a statistical proof."))
    return "<table class='tbl'>" + "<thead><tr><th>Variable</th><th>Meaning</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _strategy_read_html(tables: dict[str, pd.DataFrame], trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<div class='note warn'><strong>No trades to interpret.</strong></div>"
    scenario = tables.get("performance_by_scenario", pd.DataFrame()).copy()
    bucket = tables.get("performance_by_watchlist_bucket", pd.DataFrame()).copy()
    exit_s = tables.get("exit_reason_summary", pd.DataFrame()).copy()
    parts = []
    if not scenario.empty:
        pos = scenario[pd.to_numeric(scenario.get("avg_r"), errors="coerce") > 0]
        neg = scenario[pd.to_numeric(scenario.get("avg_r"), errors="coerce") <= 0]
        if not pos.empty:
            names = ", ".join(pos.sort_values("avg_r", ascending=False)["scenario"].astype(str).head(3))
            parts.append(f"<li><strong>Not all setups are equal:</strong> positive average-R buckets currently include {names}.</li>")
        if not neg.empty:
            names = ", ".join(neg.sort_values("avg_r")["scenario"].astype(str).head(3))
            parts.append(f"<li><strong>Potential problem areas:</strong> negative average-R buckets currently include {names}. These may need stricter entry confirmation or separate rules.</li>")
    if not bucket.empty and "watchlist_bucket" in bucket.columns:
        final = bucket[bucket["watchlist_bucket"].astype(str).str.contains("Final", case=False, na=False)]
        dev = bucket[bucket["watchlist_bucket"].astype(str).str.contains("Develop", case=False, na=False)]
        if not final.empty and not dev.empty:
            parts.append(f"<li><strong>Watchlist quality matters:</strong> Final/Actionable scenarios averaged {float(final.iloc[0].get('avg_r',0)):.3f}R versus Developing scenarios at {float(dev.iloc[0].get('avg_r',0)):.3f}R.</li>")
    if not exit_s.empty and "exit_reason" in exit_s.columns:
        stop = exit_s[exit_s["exit_reason"].astype(str).str.contains("stop", case=False, na=False)]
        ema = exit_s[exit_s["exit_reason"].astype(str).str.contains("ema", case=False, na=False)]
        if not stop.empty and not ema.empty:
            parts.append(f"<li><strong>Exit logic is separating outcomes:</strong> EMA protection averaged {float(ema.iloc[0].get('avg_r',0)):.3f}R while zone stops averaged {float(stop.iloc[0].get('avg_r',0)):.3f}R.</li>")
    if not parts:
        parts.append("<li>The current run does not show a clear separation yet. Increase sample size or inspect the funnel.</li>")
    return "<div class='note'><strong>Strategy read:</strong><ul>" + "".join(parts) + "</ul></div>"


def _performance_agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or not all(c in df.columns for c in group_cols):
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        trades=("symbol", "count"),
        profit_wins=("r_multiple", lambda s: int((s > 0).sum())),
        profit_win_rate=("r_multiple", lambda s: round((s > 0).mean() * 100, 2)),
        target_hits=("reached_3r", lambda s: int(_bool_series(s).sum()) if s.name else 0),
        target_hit_rate=("reached_3r", lambda s: round(_bool_series(s).mean() * 100, 2)),
        reached_1r_rate=("reached_1r", lambda s: round(_bool_series(s).mean() * 100, 2)),
        reached_2r_rate=("reached_2r", lambda s: round(_bool_series(s).mean() * 100, 2)),
        avg_r=("r_multiple", "mean"),
        median_r=("r_multiple", "median"),
        total_r=("r_multiple", "sum"),
        avg_mfe_r=("mfe_r", "mean"),
        avg_mae_r=("mae_r", "mean"),
    ).reset_index()
    for c in ["avg_r", "median_r", "total_r", "avg_mfe_r", "avg_mae_r"]:
        if c in out.columns:
            out[c] = out[c].round(3)
    return out.sort_values(["total_r", "avg_r", "trades"], ascending=[False, False, False])


def _exit_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "exit_reason" not in trades.columns:
        return pd.DataFrame()
    out = trades.groupby("exit_reason", dropna=False).agg(
        trades=("symbol", "count"),
        avg_r=("r_multiple", "mean"),
        total_r=("r_multiple", "sum"),
        avg_mfe_r=("mfe_r", "mean"),
        avg_mae_r=("mae_r", "mean"),
    ).reset_index()
    for c in ["avg_r", "total_r", "avg_mfe_r", "avg_mae_r"]:
        out[c] = out[c].round(3)
    return out.sort_values("trades", ascending=False)


def _rejection_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    eligible = _bool_series(candidates.get("entry_eligible", pd.Series(False, index=candidates.index)))
    rej = candidates.loc[~eligible].copy()
    if rej.empty:
        return pd.DataFrame()
    if "rejection_reason" not in rej.columns:
        rej["rejection_reason"] = "unknown"
    out = rej["rejection_reason"].fillna("unknown").value_counts().reset_index()
    out.columns = ["rejection_reason", "count"]
    out["pct_of_rejections"] = (out["count"] / out["count"].sum() * 100).round(2)
    return out


def _funnel(candidates: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    eligible = _bool_series(candidates.get("entry_eligible", pd.Series(False, index=candidates.index)))
    rows = [
        {"stage": "entry_candidates", "count": len(candidates), "pct_of_candidates": 100.0},
        {"stage": "entry_eligible", "count": int(eligible.sum()), "pct_of_candidates": round(eligible.mean() * 100, 2)},
        {"stage": "simulated_trades", "count": len(trades), "pct_of_candidates": round(len(trades) / max(len(candidates), 1) * 100, 2)},
    ]
    for reason, n in candidates.loc[~eligible, "rejection_reason"].fillna("unknown").value_counts().items() if "rejection_reason" in candidates.columns else []:
        rows.append({"stage": f"rejected: {reason}", "count": int(n), "pct_of_candidates": round(n / max(len(candidates), 1) * 100, 2)})
    return pd.DataFrame(rows)


def _col(df: pd.DataFrame, name: str, default="") -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def _nonblank(s: pd.Series) -> pd.Series:
    text = s.fillna("").astype(str).str.strip()
    return text.ne("") & ~text.str.lower().isin({"nan", "none"})


def _rate(n, d) -> float:
    return round(float(n) / max(float(d), 1.0) * 100.0, 2)


def _lifecycle_dataset(lifecycle: pd.DataFrame, candidates: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty or "snapshot_candidate_id" not in lifecycle.columns:
        return pd.DataFrame()
    df = lifecycle.copy()
    cand_cols = ["snapshot_candidate_id", "entry_eligible", "side", "timeframe", "setup_quality_grade", "setup_quality_score", "gap_direction", "gap_pct", "recent_move_direction", "recent_move_strength", "watchlist_bucket"]
    if not candidates.empty and "snapshot_candidate_id" in candidates.columns:
        use = [c for c in cand_cols if c in candidates.columns and (c == "snapshot_candidate_id" or c not in df.columns)]
        df = df.merge(candidates[use].drop_duplicates("snapshot_candidate_id"), on="snapshot_candidate_id", how="left")
    if not trades.empty and "snapshot_candidate_id" in trades.columns:
        use = [c for c in ["snapshot_candidate_id", "r_multiple", "reached_1r", "reached_2r", "reached_3r"] if c in trades.columns]
        df = df.merge(trades[use].drop_duplicates("snapshot_candidate_id"), on="snapshot_candidate_id", how="left")

    state = _col(df, "lifecycle_state").fillna("").astype(str)
    df["reached_zone_flag"] = _nonblank(_col(df, "first_zone_touch_time"))
    df["exited_zone_flag"] = _nonblank(_col(df, "first_zone_exit_time"))
    df["confirmed_flag"] = _nonblank(_col(df, "confirmation_time"))
    df["entry_eligible_flag"] = _bool_series(_col(df, "entry_eligible", False))
    df["trade_entered_flag"] = state.eq("entered_trade") | _col(df, "r_multiple", np.nan).notna()
    df["winning_trade_flag"] = pd.to_numeric(_col(df, "r_multiple", np.nan), errors="coerce") > 0
    for c in ["reached_1r", "reached_2r", "reached_3r"]:
        df[f"{c}_flag"] = _bool_series(_col(df, c, False))

    text = (_col(df, "scenario").astype(str) + " " + _col(df, "zone_type").astype(str) + " " + _col(df, "side").astype(str)).str.lower()
    df["scenario_family"] = np.select([text.str.contains("demand") & text.str.contains("break"), text.str.contains("demand"), text.str.contains("supply") & text.str.contains("break"), text.str.contains("supply")], ["Demand Break", "Demand Reversal", "Supply Break", "Supply Reversal"], default="Unknown")
    ts_base = _col(df, "first_zone_touch_time").where(df["reached_zone_flag"], _col(df, "entry_time"))
    ts = pd.to_datetime(ts_base, errors="coerce", utc=True)
    try:
        ts = ts.dt.tz_convert("America/New_York")
    except Exception:
        pass
    df["hour_of_day"] = ts.dt.strftime("%H:00").fillna("unknown")
    df["watchlist_grade"] = _col(df, "setup_quality_grade", "unknown").fillna("unknown").replace("", "unknown")
    q = pd.to_numeric(_col(df, "setup_quality_score", np.nan), errors="coerce")
    dist = pd.to_numeric(_col(df, "distance_pct", np.nan), errors="coerce").abs()
    gap = pd.to_numeric(_col(df, "gap_pct", np.nan), errors="coerce").abs()
    df["quality_score_bucket"] = pd.cut(q, [-np.inf, 5, 7, 8, 9, np.inf], labels=["<5", "5-6.9", "7-7.9", "8-8.9", "9+"]).astype("object").fillna("unknown")
    df["distance_bucket"] = pd.cut(dist, [-np.inf, 1, 2, 5, 10, np.inf], labels=["<=1%", "1-2%", "2-5%", "5-10%", ">10%"]).astype("object").fillna("unknown")
    df["gap_size_bucket"] = pd.cut(gap, [-np.inf, .5, 1, 2, 5, np.inf], labels=["<=0.5%", "0.5-1%", "1-2%", "2-5%", ">5%"]).astype("object").fillna("unknown")
    return df


def _funnel_summary_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["stage", "count", "pct_of_candidates", "pct_from_prior"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    stages = [("Snapshot Candidates", len(df)), ("Zone Reached", int(df["reached_zone_flag"].sum())), ("Zone Exited", int(df["exited_zone_flag"].sum())), ("Entry Eligible", int(df["entry_eligible_flag"].sum())), ("Trade Entered", int(df["trade_entered_flag"].sum())), ("Winning Trade", int(df["winning_trade_flag"].sum()))]
    return pd.DataFrame([{"stage": s, "count": c, "pct_of_candidates": _rate(c, len(df)), "pct_from_prior": 100.0 if i == 0 else _rate(c, stages[i - 1][1])} for i, (s, c) in enumerate(stages)], columns=cols)


def _funnel_by_lifecycle(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    cols = [group_col, "candidates", "reached_zone", "zone_exited", "entry_eligible", "trades", "winners", "reached_pct", "trade_pct", "win_pct"]
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, g in df.groupby(_col(df, group_col, "unknown").fillna("unknown").replace("", "unknown"), dropna=False):
        n = len(g); trades = int(g["trade_entered_flag"].sum())
        rows.append({
            group_col: key, "candidates": n, "reached_zone": int(g["reached_zone_flag"].sum()),
            "zone_exited": int(g["exited_zone_flag"].sum()), "entry_eligible": int(g["entry_eligible_flag"].sum()),
            "trades": trades, "winners": int(g["winning_trade_flag"].sum()),
            "reached_pct": _rate(g["reached_zone_flag"].sum(), n), "trade_pct": _rate(trades, n),
            "win_pct": _rate(g["winning_trade_flag"].sum(), trades),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(["candidates", "trades"], ascending=[False, False])


def _rejection_breakdown_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rejection_reason", "count", "pct_of_candidates", "pct_of_rejections"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    lost = df.loc[~df["trade_entered_flag"]].copy()
    reason = _col(lost, "rejection_reason").where(_nonblank(_col(lost, "rejection_reason")), _col(lost, "lifecycle_state", "unknown"))
    out = reason.fillna("unknown").replace("", "unknown").value_counts().reset_index()
    out.columns = ["rejection_reason", "count"]
    out["pct_of_candidates"] = out["count"].apply(lambda n: _rate(n, len(df)))
    out["pct_of_rejections"] = out["count"].apply(lambda n: _rate(n, len(lost)))
    return out[cols]


def _conversion_rates_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["transition", "numerator", "denominator", "conversion_pct"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    pairs = [("Snapshot -> Zone Reached", df["reached_zone_flag"], pd.Series(True, index=df.index)), ("Zone Reached -> Trade", df["trade_entered_flag"] & df["reached_zone_flag"], df["reached_zone_flag"]), ("Zone Exited -> Confirmation", df["confirmed_flag"] & df["exited_zone_flag"], df["exited_zone_flag"]), ("Confirmation -> Trade", df["trade_entered_flag"] & df["confirmed_flag"], df["confirmed_flag"]), ("Trade -> Winner", df["winning_trade_flag"], df["trade_entered_flag"]), ("Trade -> 1R", df["reached_1r_flag"], df["trade_entered_flag"]), ("Trade -> 2R", df["reached_2r_flag"], df["trade_entered_flag"]), ("Trade -> 3R", df["reached_3r_flag"], df["trade_entered_flag"])]
    return pd.DataFrame([{"transition": name, "numerator": int(num.sum()), "denominator": int(den.sum()), "conversion_pct": _rate(num.sum(), den.sum())} for name, num, den in pairs], columns=cols)


def _time_of_day(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "entry_time" not in trades.columns:
        return pd.DataFrame()
    df = trades.copy()
    ts = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    # Normalize mixed timezone strings to New York time for time-of-day grouping.
    try:
        ts = ts.dt.tz_convert("America/New_York")
    except Exception:
        pass
    df["entry_hour"] = ts.dt.strftime("%H:00")
    return _performance_agg(df.dropna(subset=["entry_hour"]), ["entry_hour"])


def _confirmation_components(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "entry_components" not in trades.columns:
        return pd.DataFrame()
    all_components = sorted({c.strip() for x in trades["entry_components"].dropna().astype(str) for c in x.split(";") if c.strip()})
    rows = []
    for comp in all_components:
        has = trades["entry_components"].fillna("").astype(str).apply(lambda x, c=comp: c in [part.strip() for part in x.split(";") if part.strip()])
        subset = trades.loc[has]
        without = trades.loc[~has]
        if subset.empty:
            continue
        rows.append({
            "confirmation": comp,
            "trades_with": len(subset),
            "win_rate_with": round((subset["r_multiple"] > 0).mean() * 100, 2),
            "avg_r_with": round(subset["r_multiple"].mean(), 3),
            "reached_1r_with": round(_bool_series(subset.get("reached_1r", pd.Series(False, index=subset.index))).mean() * 100, 2),
            "reached_2r_with": round(_bool_series(subset.get("reached_2r", pd.Series(False, index=subset.index))).mean() * 100, 2),
            "trades_without": len(without),
            "win_rate_without": round((without["r_multiple"] > 0).mean() * 100, 2) if not without.empty else np.nan,
            "avg_r_without": round(without["r_multiple"].mean(), 3) if not without.empty else np.nan,
            "avg_r_lift": round(subset["r_multiple"].mean() - (without["r_multiple"].mean() if not without.empty else 0), 3),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["avg_r_lift", "avg_r_with", "trades_with"], ascending=[False, False, False])


def _target_progress(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for label, col in [("Reached +1R", "reached_1r"), ("Reached +2R", "reached_2r"), ("Reached +3R / target", "reached_3r"), ("Reached 1x ATR", "reached_atr_1x")]:
        if col in trades.columns:
            n = int(_bool_series(trades[col]).sum())
            rows.append({"milestone": label, "count": n, "pct_trades": round(n / max(len(trades), 1) * 100, 2)})
    if "mfe_r" in trades.columns:
        for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            n = int((trades["mfe_r"] >= r).sum())
            rows.append({"milestone": f"MFE reached +{r:g}R", "count": n, "pct_trades": round(n / max(len(trades), 1) * 100, 2)})
    return pd.DataFrame(rows)


def _target_model_comparison(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    candidates = [
        ("Baseline actual exits", "r_multiple", None),
        ("Take profit at +1R", "target_1r_result", "reached_1r"),
        ("Take profit at +2R", "target_2r_result", "reached_2r"),
        ("Take profit at +3R", "target_3r_result", "reached_3r"),
        ("Take profit at 1x ATR", "target_atr_1x_result", "reached_atr_1x"),
    ]
    rows = []
    for label, col, hit_col in candidates:
        if col not in trades.columns:
            continue
        r = pd.to_numeric(trades[col], errors="coerce").dropna()
        if r.empty:
            continue
        rows.append({
            "target_model": label,
            "trades": int(len(r)),
            "profit_win_rate": round(float((r > 0).mean() * 100.0), 2),
            "target_hit_rate": round(float(_bool_series(trades[hit_col]).mean() * 100.0), 2) if hit_col and hit_col in trades.columns else np.nan,
            "avg_r": round(float(r.mean()), 3),
            "total_r": round(float(r.sum()), 3),
            "median_r": round(float(r.median()), 3),
        })
    return pd.DataFrame(rows).sort_values(["avg_r", "profit_win_rate"], ascending=[False, False])


def _opportunity_cost(candidates: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    # This first-pass report is intentionally conservative: it explains why candidates did not become entries.
    # True post-rejection price movement requires storing MFE for rejected events during replay.
    if candidates.empty:
        return pd.DataFrame()
    eligible = _bool_series(candidates.get("entry_eligible", pd.Series(False, index=candidates.index)))
    rejected = candidates.loc[~eligible].copy()
    if rejected.empty or "rejection_reason" not in rejected.columns:
        return pd.DataFrame()
    out = rejected.groupby("rejection_reason", dropna=False).agg(
        rejected_candidates=("symbol", "count"),
        symbols=("symbol", lambda s: ", ".join(sorted(set(s.astype(str)))[:12])),
        avg_volume_ratio=("entry_volume_ratio", "mean") if "entry_volume_ratio" in rejected.columns else ("symbol", "count"),
        avg_body_ratio=("entry_body_ratio", "mean") if "entry_body_ratio" in rejected.columns else ("symbol", "count"),
    ).reset_index()
    for c in ["avg_volume_ratio", "avg_body_ratio"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out.sort_values("rejected_candidates", ascending=False)





def _reversal_rejection_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(False, index=getattr(df, "index", []))
    entry = df.get("entry_kind", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    scenario = df.get("scenario", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    return (
        entry.isin(["demand_reversal", "supply_rejection"])
        | scenario.str.contains("demand", na=False) & (scenario.str.contains("reversal", na=False) | scenario.str.contains("hold", na=False))
        | scenario.str.contains("supply", na=False) & scenario.str.contains("reject", na=False)
    )


def _sample_flag(n: int) -> str:
    if n <= 0:
        return "no_sample"
    if n < 10:
        return "very_low_sample"
    if n < 25:
        return "low_sample"
    return "ok"


def _as_bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return _bool_series(df[col])


def _as_num_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _grade_is_a_family(v) -> bool:
    s = str(v).strip().upper()
    return s.startswith("A")


def _reversal_perf_metrics(df: pd.DataFrame, r_col: str = "r_multiple") -> dict:
    if df is None or df.empty:
        return {
            "trades": 0,
            "sample_flag": "no_sample",
            "profit_win_rate": 0.0,
            "target_hit_rate": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
        }
    r = _as_num_col(df, r_col if r_col in df.columns else "r_multiple").fillna(0.0)
    target_col = "reached_3r" if "reached_3r" in df.columns else "exit_reason"
    if target_col == "reached_3r":
        target_hit = _bool_series(df[target_col]).mean() * 100.0
    else:
        target_hit = df[target_col].astype(str).str.contains("target", case=False, na=False).mean() * 100.0
    n = len(df)
    return {
        "trades": int(n),
        "sample_flag": _sample_flag(n),
        "profit_win_rate": round(float((r > 0).mean() * 100.0), 2),
        "target_hit_rate": round(float(target_hit), 2),
        "avg_r": round(float(r.mean()), 3),
        "total_r": round(float(r.sum()), 3),
        "avg_mfe_r": round(float(_as_num_col(df, "mfe_r").mean()), 3) if "mfe_r" in df.columns else np.nan,
        "avg_mae_r": round(float(_as_num_col(df, "mae_r").mean()), 3) if "mae_r" in df.columns else np.nan,
    }


REVERSAL_DIAGNOSTIC_FIELDS = [
    "scenario",
    "setup_quality_grade",
    "watchlist_bucket",
    "timeframe",
    "freshness",
    "tests",
    "entry_kind",
    "entry_9ema_relation",
    "entry_vwap_relation",
    "entry_close_zone_location",
    "has_boundary_reclaim_reject",
    "reclaimed_demand_top",
    "rejected_below_supply_bottom",
    "has_1c_confirmation",
    "has_2c_confirmation",
    "time_of_day_bucket",
    "exit_reason",
    "live_confirmation_score",
    "live_confirmation_bucket",
    "wick_quality_bucket",
    "vpa_confirmation_bucket",
    "momentum_confirmation_bucket",
    "backtest_realism_diagnosis",
]


def _build_reversal_rejection_breakdowns(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rev = trades.loc[_reversal_rejection_mask(trades)].copy()
    if rev.empty:
        return pd.DataFrame()
    rows = []
    for field in REVERSAL_DIAGNOSTIC_FIELDS:
        if field not in rev.columns:
            continue
        for value, group in rev.groupby(field, dropna=False):
            rows.append({
                "diagnostic_scope": "all_reversal_rejection",
                "breakdown": field,
                "value": "unknown" if pd.isna(value) or value == "" else value,
                **_reversal_perf_metrics(group),
            })
        if field != "scenario" and "scenario" in rev.columns:
            for (scenario, value), group in rev.groupby(["scenario", field], dropna=False):
                rows.append({
                    "diagnostic_scope": scenario,
                    "breakdown": field,
                    "value": "unknown" if pd.isna(value) or value == "" else value,
                    **_reversal_perf_metrics(group),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["breakdown", "diagnostic_scope", "total_r", "avg_r", "trades"], ascending=[True, True, False, False, False])


def _build_reversal_rule_variants(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rev = trades.loc[_reversal_rejection_mask(trades)].copy()
    if rev.empty:
        return pd.DataFrame()

    grade = rev.get("setup_quality_grade", pd.Series("", index=rev.index))
    bucket = rev.get("watchlist_bucket", pd.Series("", index=rev.index)).fillna("").astype(str)
    score = _as_num_col(rev, "live_confirmation_score", default=0).fillna(0)
    good_wick = ~_as_bool_col(rev, "has_bad_entry_side_wick")
    variants = [
        ("Existing backtest behavior", pd.Series(True, index=rev.index), "r_multiple"),
        ("Live-style: exclude B/C grades", grade.apply(_grade_is_a_family), "r_multiple"),
        ("Live-style: Final only", bucket.str.contains("Final|Action", case=False, na=False), "r_multiple"),
        ("Live-style: A/A+ Final only", bucket.str.contains("Final|Action", case=False, na=False) & grade.apply(_grade_is_a_family), "r_multiple"),
        ("Require zone edge tap", _as_bool_col(rev, "zone_edge_tapped"), "r_multiple"),
        ("Require strong instant reaction candle", _as_bool_col(rev, "strong_instant_reaction"), "r_multiple"),
        ("Require good wick quality", good_wick, "r_multiple"),
        ("Require VPA confirmation", _as_bool_col(rev, "vpa_confirmed"), "r_multiple"),
        ("Require 2-candle follow-through window", _as_bool_col(rev, "has_2c_confirmation"), "r_multiple"),
        ("Require 9EMA or VWAP break", _as_bool_col(rev, "ema_or_vwap_break"), "r_multiple"),
        ("Require structure confirmation", _as_bool_col(rev, "structure_confirmation"), "r_multiple"),
        ("Require boundary reclaim/rejection", _as_bool_col(rev, "has_boundary_reclaim_reject"), "r_multiple"),
        ("Require confirmation score >= 4", score >= 4, "r_multiple"),
        ("Require confirmation score >= 5", score >= 5, "r_multiple"),
        ("Manage reversals: 9EMA protection after +0.5R", pd.Series(True, index=rev.index), "ema_protect_05r_exit_r" if "ema_protect_05r_exit_r" in rev.columns else "r_multiple"),
    ]

    baseline = _reversal_perf_metrics(rev)
    baseline_trades = max(int(baseline["trades"]), 1)
    baseline_avg_r = float(baseline["avg_r"])
    rows = []
    for name, mask, r_col in variants:
        mask = mask.fillna(False) if hasattr(mask, "fillna") else mask
        subset = rev.loc[mask].copy()
        metrics = _reversal_perf_metrics(subset, r_col=r_col)
        rows.append({
            "variant": name,
            **metrics,
            "sample_retention_pct": round(metrics["trades"] / baseline_trades * 100.0, 2),
            "avg_r_delta_vs_baseline": round(float(metrics["avg_r"]) - baseline_avg_r, 3),
            "r_source": r_col,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["diagnostic_read"] = np.select(
        [
            (out["avg_r_delta_vs_baseline"] > 0) & (out["sample_retention_pct"] >= 40) & (out["trades"] >= 25),
            (out["avg_r_delta_vs_baseline"] > 0) & (out["trades"] < 25),
            (out["avg_r_delta_vs_baseline"] > 0) & (out["sample_retention_pct"] < 40),
            (out["avg_r_delta_vs_baseline"] <= 0),
        ],
        [
            "candidate_filter_or_management_change",
            "improves_but_low_sample",
            "improves_but_too_selective",
            "does_not_improve_expectancy",
        ],
        default="review",
    )
    return out.sort_values(["avg_r", "sample_retention_pct", "trades"], ascending=[False, False, False])


def _identify_reversal_best_filters(rule_variants: pd.DataFrame) -> pd.DataFrame:
    if rule_variants is None or rule_variants.empty:
        return pd.DataFrame()
    out = rule_variants.copy()
    out = out[out["variant"].ne("Existing backtest behavior")].copy()
    if out.empty:
        return out
    out["usefulness_rank"] = np.select(
        [
            out["diagnostic_read"].eq("candidate_filter_or_management_change"),
            out["diagnostic_read"].eq("improves_but_too_selective"),
            out["diagnostic_read"].eq("improves_but_low_sample"),
        ],
        [1, 2, 3],
        default=4,
    )
    return out.sort_values(["usefulness_rank", "avg_r_delta_vs_baseline", "sample_retention_pct"], ascending=[True, False, False])


def _reversal_diagnostics_html(tables: dict[str, pd.DataFrame], trades: pd.DataFrame) -> str:
    rev = trades.loc[_reversal_rejection_mask(trades)].copy() if not trades.empty else pd.DataFrame()
    if rev.empty:
        return "<section><h2>Reversal/Rejection Diagnostics</h2><p class='muted'>No reversal/rejection trades found in this run.</p></section>"
    overall = _reversal_perf_metrics(rev)
    cards = "".join([
        _card("Reversal/rejection trades", overall["trades"], overall["sample_flag"]),
        _card("Profit win rate", f"{_fmt_num(overall['profit_win_rate'])}%", "Closed above 0R"),
        _card("Target hit rate", f"{_fmt_num(overall['target_hit_rate'])}%", "Reached modeled target milestone"),
        _card("Average R", f"{_fmt_num(overall['avg_r'],3)}R", "Baseline replay behavior"),
        _card("Total R", f"{_fmt_num(overall['total_r'],2)}R", "Aggregate contribution"),
        _card("Avg MFE / MAE", f"{_fmt_num(overall['avg_mfe_r'],2)}R / {_fmt_num(overall['avg_mae_r'],2)}R", "Best/worst unrealized"),
    ])
    variants = tables.get("reversal_rejection_rule_variants", pd.DataFrame())
    best = tables.get("reversal_rejection_best_filters", pd.DataFrame())
    breakdowns = tables.get("reversal_rejection_breakdowns", pd.DataFrame())
    scenario = _performance_agg(rev, ["scenario", "side"]) if "scenario" in rev.columns and "side" in rev.columns else pd.DataFrame()
    if not scenario.empty:
        scenario = _edge_table(scenario, ["scenario", "side"])

    details = []
    if breakdowns is not None and not breakdowns.empty:
        for name in [
            "backtest_realism_diagnosis",
            "live_confirmation_bucket",
            "wick_quality_bucket",
            "vpa_confirmation_bucket",
            "momentum_confirmation_bucket",
            "entry_close_zone_location",
            "time_of_day_bucket",
            "exit_reason",
        ]:
            d = breakdowns[(breakdowns["diagnostic_scope"].eq("all_reversal_rejection")) & (breakdowns["breakdown"].eq(name))].copy()
            if not d.empty:
                details.append(f"<details open><summary><strong>{name}</strong></summary>{_table(d, max_rows=20)}</details>")
    detail_html = "".join(details) if details else "<p class='muted'>No breakdown rows available yet.</p>"

    return f"""
    <section id='reversal-diagnostics'>
      <h2>Reversal/Rejection Diagnostics</h2>
      <div class='note'><strong>Purpose:</strong> This audits whether Demand Reversal / Hold — Calls and Supply Rejection — Puts are genuinely weak, or whether the replay is entering/managing them differently than a live trader would. Continuation setups remain excluded because they can stay more permissive.</div>
      <div class='cards'>{cards}</div>
      <h3>Scenario baseline</h3>{_table(scenario)}
      <h3>Live-style rule and management variants</h3>
      <div class='muted'>These are diagnostics, not automatic watchlist changes. Look for average-R improvement with reasonable sample retention.</div>{_table(variants)}
      <h3>Best candidate filters / management changes</h3>{_table(best, max_rows=12)}
      <h3>Breakdown details</h3>{detail_html}
    </section>
    """


def _safe_mean_numeric(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _build_exit_path_first_event(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty or "first_event_after_entry" not in trades.columns:
        return pd.DataFrame()
    rows = []
    scopes = [("all_trades", trades)]
    rev = trades.loc[_reversal_rejection_mask(trades)].copy()
    if not rev.empty:
        scopes.append(("reversal_rejection", rev))
    for scope, df in scopes:
        for event, group in df.groupby("first_event_after_entry", dropna=False):
            rows.append({
                "audit_scope": scope,
                "first_event_after_entry": "unknown" if pd.isna(event) or event == "" else event,
                **_reversal_perf_metrics(group),
                "avg_mfe_until_exit": round(_safe_mean_numeric(group, "mfe_r_until_exit"), 3),
                "avg_mae_until_exit": round(_safe_mean_numeric(group, "mae_r_until_exit"), 3),
                "avg_mfe_full_day": round(_safe_mean_numeric(group, "mfe_r_full_day"), 3),
                "avg_mae_full_day": round(_safe_mean_numeric(group, "mae_r_full_day"), 3),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["audit_scope", "trades", "avg_r"], ascending=[True, False, False])


EXIT_PATH_FLAG_COLUMNS = [
    "target_and_stop_same_candle",
    "same_candle_ambiguity",
    "one_r_and_stop_same_candle",
    "two_r_and_stop_same_candle",
    "three_r_and_stop_same_candle",
    "stop_after_reached_1r",
    "stop_after_reached_2r",
    "stop_after_reached_3r",
    "target_available_but_not_taken",
    "ema_protection_available_but_not_taken",
    "mfe_after_exit_detected",
    "mae_after_exit_detected",
    "unrealistic_r_outlier",
]


def _build_exit_path_flags(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    rows = []
    scopes = [("all_trades", trades)]
    rev = trades.loc[_reversal_rejection_mask(trades)].copy()
    if not rev.empty:
        scopes.append(("reversal_rejection", rev))
    for scope, df in scopes:
        base_n = max(len(df), 1)
        for col in EXIT_PATH_FLAG_COLUMNS:
            if col not in df.columns:
                continue
            mask = _as_bool_col(df, col)
            subset = df.loc[mask].copy()
            metrics = _reversal_perf_metrics(subset)
            rows.append({
                "audit_scope": scope,
                "flag": col,
                **metrics,
                "pct_of_scope": round(len(subset) / base_n * 100.0, 2),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["audit_scope", "trades", "pct_of_scope"], ascending=[True, False, False])


def _build_exit_path_management_variants(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    rev = trades.loc[_reversal_rejection_mask(trades)].copy()
    if rev.empty:
        return pd.DataFrame()
    variants = [
        ("Baseline replay: conservative stop before target", "r_multiple"),
        ("Intrabar target-priority check", "target_priority_3r_exit_r"),
        ("Move stop to breakeven after +1R", "breakeven_after_1r_exit_r"),
        ("9EMA protection after +0.5R", "ema_protect_05r_exit_r"),
        ("Exit on first close back through boundary", "boundary_loss_1_close_exit_r"),
        ("Exit after two closes back through boundary", "boundary_loss_2_closes_exit_r"),
        ("Neutral result: exclude target/stop same-candle ambiguity", "neutral_intrabar_result"),
        ("Optimistic intrabar: target before stop if same candle", "optimistic_intrabar_result"),
    ]
    baseline = _reversal_perf_metrics(rev)
    baseline_avg_r = float(baseline["avg_r"])
    baseline_trades = max(int(baseline["trades"]), 1)
    rows = []
    for name, col in variants:
        if col not in rev.columns:
            continue
        subset = rev.copy()
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset[subset[col].notna()].copy()
        metrics = _reversal_perf_metrics(subset, r_col=col)
        rows.append({
            "variant": name,
            "r_source": col,
            **metrics,
            "sample_retention_pct": round(metrics["trades"] / baseline_trades * 100.0, 2),
            "avg_r_delta_vs_baseline": round(float(metrics["avg_r"]) - baseline_avg_r, 3),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["diagnostic_read"] = np.select(
        [
            out["variant"].str.contains("Neutral", case=False, na=False),
            (out["avg_r_delta_vs_baseline"] > 0) & (out["sample_retention_pct"] >= 70),
            (out["avg_r_delta_vs_baseline"] > 0) & (out["sample_retention_pct"] < 70),
            out["avg_r_delta_vs_baseline"] <= 0,
        ],
        [
            "ambiguity_clean_sample",
            "management_change_candidate",
            "helps_but_changes_sample_or_path",
            "does_not_improve_replay_result",
        ],
        default="review",
    )
    return out.sort_values(["avg_r", "sample_retention_pct"], ascending=[False, False])


def _build_exit_path_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    rows = []
    scopes = [("all_trades", trades)]
    rev = trades.loc[_reversal_rejection_mask(trades)].copy()
    if not rev.empty:
        scopes.append(("reversal_rejection", rev))
    for scope, df in scopes:
        n = max(len(df), 1)
        rows.append({
            "audit_scope": scope,
            "trades": len(df),
            "avg_r": round(float(pd.to_numeric(df.get("r_multiple"), errors="coerce").mean()), 3),
            "stop_after_reached_1r": int(_as_bool_col(df, "stop_after_reached_1r").sum()) if "stop_after_reached_1r" in df.columns else 0,
            "stop_after_reached_2r": int(_as_bool_col(df, "stop_after_reached_2r").sum()) if "stop_after_reached_2r" in df.columns else 0,
            "stop_after_reached_3r": int(_as_bool_col(df, "stop_after_reached_3r").sum()) if "stop_after_reached_3r" in df.columns else 0,
            "target_stop_same_candle": int(_as_bool_col(df, "target_and_stop_same_candle").sum()) if "target_and_stop_same_candle" in df.columns else 0,
            "same_candle_ambiguity": int(_as_bool_col(df, "same_candle_ambiguity").sum()) if "same_candle_ambiguity" in df.columns else 0,
            "target_available_but_not_taken": int(_as_bool_col(df, "target_available_but_not_taken").sum()) if "target_available_but_not_taken" in df.columns else 0,
            "mfe_after_exit_detected": int(_as_bool_col(df, "mfe_after_exit_detected").sum()) if "mfe_after_exit_detected" in df.columns else 0,
            "unrealistic_r_outlier": int(_as_bool_col(df, "unrealistic_r_outlier").sum()) if "unrealistic_r_outlier" in df.columns else 0,
            "pct_stop_after_1r": round((_as_bool_col(df, "stop_after_reached_1r").sum() / n * 100.0), 2) if "stop_after_reached_1r" in df.columns else 0.0,
            "pct_same_candle_ambiguity": round((_as_bool_col(df, "same_candle_ambiguity").sum() / n * 100.0), 2) if "same_candle_ambiguity" in df.columns else 0.0,
            "avg_mfe_until_exit": round(_safe_mean_numeric(df, "mfe_r_until_exit"), 3),
            "avg_mae_until_exit": round(_safe_mean_numeric(df, "mae_r_until_exit"), 3),
            "avg_mfe_full_day": round(_safe_mean_numeric(df, "mfe_r_full_day"), 3),
            "avg_mae_full_day": round(_safe_mean_numeric(df, "mae_r_full_day"), 3),
        })
    return pd.DataFrame(rows)


def _exit_path_audit_html(tables: dict[str, pd.DataFrame], trades: pd.DataFrame) -> str:
    if trades is None or trades.empty:
        return "<section><h2>Exit Path Audit</h2><p class='muted'>No trades found for exit-path audit.</p></section>"
    summary = tables.get("exit_path_summary", pd.DataFrame())
    flags = tables.get("exit_path_flags", pd.DataFrame())
    first = tables.get("exit_path_first_event", pd.DataFrame())
    variants = tables.get("exit_path_management_variants", pd.DataFrame())
    rev = trades.loc[_reversal_rejection_mask(trades)].copy()
    row = summary[summary["audit_scope"].eq("reversal_rejection")].iloc[0].to_dict() if not summary.empty and (summary["audit_scope"] == "reversal_rejection").any() else {}
    cards = "".join([
        _card("Reversal/rejection trades", row.get("trades", len(rev)), "Audit scope"),
        _card("Stop after +1R", row.get("stop_after_reached_1r", 0), f"{_fmt_num(row.get('pct_stop_after_1r', 0))}% of scope"),
        _card("Stop after +3R", row.get("stop_after_reached_3r", 0), "Usually same-candle ambiguity or giveback"),
        _card("Target/stop same candle", row.get("target_stop_same_candle", 0), "5M intrabar order unknown"),
        _card("MFE until/full day", f"{_fmt_num(row.get('avg_mfe_until_exit', 0),2)}R / {_fmt_num(row.get('avg_mfe_full_day', 0),2)}R", "Checks post-exit movement"),
        _card("Unrealistic R outliers", row.get("unrealistic_r_outlier", 0), "Tiny risk / extreme R guard"),
    ])
    rev_first = first[first["audit_scope"].eq("reversal_rejection")].copy() if first is not None and not first.empty else pd.DataFrame()
    rev_flags = flags[flags["audit_scope"].eq("reversal_rejection")].copy() if flags is not None and not flags.empty else pd.DataFrame()
    return f"""
    <section id='exit-path-audit'>
      <h2>Exit Path Audit</h2>
      <div class='note warn'><strong>Purpose:</strong> This section checks whether replay scoring is matching live trade-path management. It does not change the watchlist. It highlights same-candle stop/target ambiguity, trades stopped after reaching +R milestones, MFE/MAE until exit versus full-day movement, and reversal-specific management simulations.</div>
      <div class='cards'>{cards}</div>
      <h3>Exit-path summary</h3>{_table(summary)}
      <h3>Reversal/Rejection first event after entry</h3>{_table(rev_first, max_rows=20)}
      <h3>Reversal/Rejection audit flags</h3>{_table(rev_flags, max_rows=30)}
      <h3>Reversal/Rejection management simulations</h3>
      <div class='muted'>These variants are diagnostic only. They show whether live-style protection would materially change the reversal/rejection read.</div>{_table(variants)}
    </section>
    """

def _json_records(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "[]"
    out = df.copy()
    # Keep only browser-useful columns. Missing columns are simply omitted.
    preferred = [
        "symbol", "trade_date", "entry_time", "exit_time", "scenario", "side", "entry_kind",
        "watchlist_bucket", "setup_quality_grade", "zone_type", "timeframe", "exit_reason",
        "r_multiple", "mfe_r", "mae_r", "reached_1r", "reached_2r", "reached_3r", "reached_atr_1x",
        "entry_time_bucket", "entry_atr_14", "atr_1x_r", "target_1r_result", "target_2r_result", "target_3r_result", "target_atr_1x_result",
        "setup_quality_score", "entry_volume_ratio", "entry_body_ratio",
        "live_confirmation_score", "live_confirmation_bucket", "backtest_realism_diagnosis",
        "wick_quality_bucket", "vpa_confirmation_bucket", "momentum_confirmation_bucket",
        "has_boundary_reclaim_reject", "has_1c_confirmation", "has_2c_confirmation",
        "entry_9ema_relation", "entry_vwap_relation", "entry_close_zone_location",
    ]
    cols = [c for c in preferred if c in out.columns]
    out = out[cols]
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)
    records = out.replace({np.nan: None}).to_dict(orient="records")
    return json.dumps(records)


def _interactive_dashboard_html(trades: pd.DataFrame) -> str:
    records = _json_records(trades)
    # Each select uses exact source values from trades. JavaScript builds the controls from embedded rows.
    return f"""
    <section class='interactive' id='interactiveDashboard'>
      <h2>Interactive Strategy Filters</h2>
      <div class='muted'>Use this section for exploratory optimization. Filters recalculate instantly in the browser; no Python rerun needed. Watch sample size so you do not overfit a tiny subset.</div>
      <div class='toolbar'>
        <button type='button' onclick='selectAllFilters(true)'>Select all</button>
        <button type='button' onclick='selectAllFilters(false)'>Clear all</button>
        <button type='button' onclick='presetExcludeB()'>Exclude B/C grades</button>
        <button type='button' onclick='presetFinalOnly()'>Final only</button>
        <button type='button' onclick='presetContinuationOnly()'>Continuation only</button>
      </div>
      <div class='filtergrid' id='filterGrid'></div>
      <div id='activeFilterNote' class='muted'></div>
      <div class='cards' id='filteredCards'></div>
      <div id='sampleWarning'></div>
      <div class='grid2'>
        <div class='viz'><h3>Filtered average R by scenario</h3><div class='muted'>Shows whether the selected subset is being carried by specific scenario types.</div><div id='scenarioBars'></div></div>
        <div class='viz'><h3>Filtered target progression</h3><div class='muted'>Touched milestone, not necessarily exit point.</div><div id='targetBars'></div></div>
      </div>
      <div class='grid2'>
        <section><h2>Filtered By Scenario</h2><div id='filteredScenarioTable'></div></section>
        <section><h2>Filtered By Watchlist Bucket</h2><div id='filteredBucketTable'></div></section>
      </div>
      <div class='grid2'>
        <section><h2>Filtered By Grade</h2><div id='filteredGradeTable'></div></section>
        <section><h2>Filtered By Exit Reason</h2><div id='filteredExitTable'></div></section>
      </div>
      <details class='note'><summary><strong>Filter variable definitions</strong></summary>
        <table class='tbl'><tbody>
          <tr><td><strong>Watchlist bucket</strong></td><td>Final/Actionable, Developing, or Research. Use this to test whether lower-confidence prep scenarios dilute results.</td></tr>
          <tr><td><strong>Setup grade</strong></td><td>The watchlist quality grade assigned before the intraday entry. Filtering out B/C tests whether lower-quality setups are hurting expectancy.</td></tr>
          <tr><td><strong>Scenario</strong></td><td>The supply/demand trade thesis, such as demand hold calls or supply breakout calls.</td></tr>
          <tr><td><strong>Entry kind</strong></td><td>Simplified entry family: continuation/breakout/breakdown versus reversal/rejection.</td></tr>
          <tr><td><strong>Zone type</strong></td><td>Demand or supply. This tests whether one side is stronger in the sample.</td></tr>
          <tr><td><strong>Exit reason</strong></td><td>How the replay closed the trade. This is mostly diagnostic; do not optimize by exit reason as a pre-entry filter because you would not know it beforehand.</td></tr>
        </tbody></table>
      </details>
    </section>
    <script id='tradeData' type='application/json'>{records}</script>
    <script>
    const RAW_TRADES = JSON.parse(document.getElementById('tradeData').textContent || '[]');
    const FILTERS = [
      ['watchlist_bucket','Watchlist bucket'],
      ['setup_quality_grade','Setup grade'],
      ['scenario','Scenario'],
      ['entry_kind','Entry kind'],
      ['zone_type','Zone type'],
      ['timeframe','Zone timeframe'],
      ['entry_time_bucket','Entry time bucket'],
      ['side','Side'],
      ['symbol','Symbol'],
      ['exit_reason','Exit reason']
    ];
    const PRE_ENTRY_FILTERS = new Set(['watchlist_bucket','setup_quality_grade','scenario','entry_kind','zone_type','timeframe','entry_time_bucket','side','symbol']);
    function clean(v) {{ return (v === null || v === undefined || v === '') ? 'Unknown' : String(v); }}
    function uniq(field) {{ return [...new Set(RAW_TRADES.map(r => clean(r[field])))].sort((a,b)=>a.localeCompare(b)); }}
    function pct(n,d) {{ return d ? (100*n/d) : 0; }}
    function num(v) {{ const x = Number(v); return Number.isFinite(x) ? x : 0; }}
    function avg(arr, f) {{ if(!arr.length) return 0; return arr.reduce((s,r)=>s+num(f(r)),0)/arr.length; }}
    function sum(arr, f) {{ return arr.reduce((s,r)=>s+num(f(r)),0); }}
    function bool(v) {{ return v === true || String(v).toLowerCase() === 'true' || String(v) === '1'; }}
    function fmt(x, d=2) {{ return Number.isFinite(Number(x)) ? Number(x).toFixed(d) : '0.00'; }}
    function renderFilters() {{
      const grid = document.getElementById('filterGrid');
      grid.innerHTML = '';
      FILTERS.forEach(([field,label]) => {{
        const vals = uniq(field);
        if(vals.length <= 1 && vals[0] === 'Unknown') return;
        const box = document.createElement('div');
        box.className='filterbox';
        box.innerHTML = `<div class='filtertitle'>${{label}}</div>` + vals.map(v => `<label><input type='checkbox' data-field='${{field}}' value='${{v.replace(/'/g,"&#39;")}}' checked> ${{v}}</label>`).join('');
        grid.appendChild(box);
      }});
      grid.querySelectorAll('input').forEach(cb => cb.addEventListener('change', updateDashboard));
    }}
    function selectedValues(field) {{ return new Set([...document.querySelectorAll(`input[data-field="${{field}}"]:checked`)].map(x => x.value)); }}
    function filteredTrades() {{
      return RAW_TRADES.filter(r => FILTERS.every(([field]) => {{
        const boxes = [...document.querySelectorAll(`input[data-field="${{field}}"]`)];
        if(!boxes.length) return true;
        const selected = selectedValues(field);
        return selected.has(clean(r[field]));
      }}));
    }}
    function card(label,value,sub='') {{ return `<div class='card'><div class='label'>${{label}}</div><div class='value'>${{value}}</div><div class='subtle'>${{sub}}</div></div>`; }}
    function metrics(rows) {{
      const n = rows.length;
      const wins = rows.filter(r => num(r.r_multiple) > 0).length;
      const rsum = sum(rows, r => r.r_multiple);
      return {{
        trades:n,
        profitWinRate:pct(wins,n),
        targetHitRate:pct(rows.filter(r=>bool(r.reached_3r)).length,n),
        reached1:pct(rows.filter(r=>bool(r.reached_1r)).length,n),
        reached2:pct(rows.filter(r=>bool(r.reached_2r)).length,n),
        avgR:n?rsum/n:0,
        totalR:rsum,
        avgMfe:avg(rows,r=>r.mfe_r),
        avgMae:avg(rows,r=>r.mae_r)
      }};
    }}
    function groupAgg(rows, field) {{
      const groups = {{}};
      rows.forEach(r => {{ const k=clean(r[field]); (groups[k] ||= []).push(r); }});
      return Object.entries(groups).map(([key,rs]) => {{ const m=metrics(rs); return {{bucket:key,...m}}; }}).sort((a,b)=>b.totalR-a.totalR || b.avgR-a.avgR || b.trades-a.trades);
    }}
    function table(rows, firstCol='Bucket') {{
      if(!rows.length) return `<p class='muted'>No rows after filters.</p>`;
      return `<table class='tbl'><thead><tr><th>${{firstCol}}</th><th>Trades</th><th>Profit win rate</th><th>3R target</th><th>Avg R</th><th>Total R</th><th>Reached +1R</th><th>Reached +2R</th><th>Avg MFE</th><th>Avg MAE</th></tr></thead><tbody>` + rows.map(r =>
        `<tr><td>${{r.bucket}}</td><td>${{r.trades}}</td><td>${{fmt(r.profitWinRate)}}%</td><td>${{fmt(r.targetHitRate)}}%</td><td>${{fmt(r.avgR,3)}}R</td><td>${{fmt(r.totalR,2)}}R</td><td>${{fmt(r.reached1)}}%</td><td>${{fmt(r.reached2)}}%</td><td>${{fmt(r.avgMfe,2)}}R</td><td>${{fmt(r.avgMae,2)}}R</td></tr>`
      ).join('') + `</tbody></table>`;
    }}
    function bars(rows, key='avgR', suffix='R', digits=3) {{
      if(!rows.length) return `<p class='muted'>No chart data.</p>`;
      const top = rows.slice(0,10);
      const maxAbs = Math.max(1, ...top.map(r => Math.abs(num(r[key]))));
      return top.map(r => {{
        const val = num(r[key]); const w = Math.max(2, Math.min(100, Math.abs(val)/maxAbs*100));
        const cls = val >= 0 ? 'pos' : 'neg';
        return `<div class='barrow'><div class='barlabel' title='${{r.bucket}}'>${{r.bucket}}</div><div class='bartrack'><div class='barfill ${{cls}}' style='width:${{w}}%'></div></div><div class='barval'>${{fmt(val,digits)}}${{suffix}}</div></div>`;
      }}).join('');
    }}
    function targetRows(rows) {{
      const n = rows.length || 1;
      return [
        {{bucket:'Reached +1R', pct:pct(rows.filter(r=>bool(r.reached_1r)).length,n)}},
        {{bucket:'Reached +2R', pct:pct(rows.filter(r=>bool(r.reached_2r)).length,n)}},
        {{bucket:'Reached +3R / target', pct:pct(rows.filter(r=>bool(r.reached_3r)).length,n)}},
        {{bucket:'Reached 1x ATR', pct:pct(rows.filter(r=>bool(r.reached_atr_1x)).length,n)}},
        {{bucket:'MFE >= +1R', pct:pct(rows.filter(r=>num(r.mfe_r)>=1).length,n)}},
        {{bucket:'MFE >= +2R', pct:pct(rows.filter(r=>num(r.mfe_r)>=2).length,n)}},
        {{bucket:'MFE >= +3R', pct:pct(rows.filter(r=>num(r.mfe_r)>=3).length,n)}}
      ];
    }}
    function activeFilterSummary() {{
      const parts=[];
      FILTERS.forEach(([field,label]) => {{
        const all = uniq(field); const sel = [...selectedValues(field)];
        if(all.length && sel.length !== all.length) parts.push(`${{label}}: ${{sel.join(', ') || 'none'}}`);
      }});
      document.getElementById('activeFilterNote').textContent = parts.length ? 'Active filters — ' + parts.join(' | ') : 'No filters applied.';
    }}
    function updateDashboard() {{
      const rows = filteredTrades();
      const m = metrics(rows);
      document.getElementById('filteredCards').innerHTML = [
        card('Filtered trades', m.trades, `${{RAW_TRADES.length}} total before filters`),
        card('Profit win rate', `${{fmt(m.profitWinRate)}}%`, 'Closed above 0R'),
        card('3R target hit rate', `${{fmt(m.targetHitRate)}}%`, 'Reached full target'),
        card('Average R', `${{fmt(m.avgR,3)}}R`, 'Expectancy per selected trade'),
        card('Total R', `${{fmt(m.totalR,2)}}R`, 'Aggregate selected contribution'),
        card('Reached +1R', `${{fmt(m.reached1)}}%`, 'Touched +1R'),
        card('Reached +2R', `${{fmt(m.reached2)}}%`, 'Touched +2R'),
        card('Avg MFE / MAE', `${{fmt(m.avgMfe,2)}}R / ${{fmt(m.avgMae,2)}}R`, 'Best/worst unrealized')
      ].join('');
      const warn = m.trades < 30 ? 'Very small sample. Treat this as a clue, not a rule.' : (m.trades < 100 ? 'Small sample. Useful for hypotheses, but validate out-of-sample.' : 'Sample size is more useful, but still avoid overfitting.');
      document.getElementById('sampleWarning').innerHTML = `<div class='note ${{m.trades < 100 ? 'warn' : ''}}'><strong>Sample note:</strong> ${{warn}}</div>`;
      const sc = groupAgg(rows,'scenario');
      document.getElementById('scenarioBars').innerHTML = bars(sc,'avgR','R',3);
      document.getElementById('targetBars').innerHTML = bars(targetRows(rows),'pct','%',1);
      document.getElementById('filteredScenarioTable').innerHTML = table(sc,'Scenario');
      document.getElementById('filteredBucketTable').innerHTML = table(groupAgg(rows,'watchlist_bucket'),'Watchlist bucket');
      document.getElementById('filteredGradeTable').innerHTML = table(groupAgg(rows,'setup_quality_grade'),'Setup grade');
      document.getElementById('filteredExitTable').innerHTML = table(groupAgg(rows,'exit_reason'),'Exit reason');
      activeFilterSummary();
    }}
    function selectAllFilters(state) {{ document.querySelectorAll('#filterGrid input').forEach(cb => cb.checked = state); updateDashboard(); }}
    function presetExcludeB() {{
      document.querySelectorAll('input[data-field="setup_quality_grade"]').forEach(cb => {{ cb.checked = !['B','B+','B-','C','C+','C-','D','F'].includes(cb.value); }});
      updateDashboard();
    }}
    function presetFinalOnly() {{
      document.querySelectorAll('input[data-field="watchlist_bucket"]').forEach(cb => {{ cb.checked = cb.value.toLowerCase().includes('final') || cb.value.toLowerCase().includes('action'); }});
      updateDashboard();
    }}
    function presetContinuationOnly() {{
      document.querySelectorAll('input[data-field="scenario"], input[data-field="entry_kind"]').forEach(cb => {{
        const v = cb.value.toLowerCase(); cb.checked = v.includes('continuation') || v.includes('breakout') || v.includes('breakdown');
      }});
      updateDashboard();
    }}
    renderFilters(); updateDashboard();
    </script>
    """


def _write_csvs(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if df is not None and not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=False)


def _write_funnel_outputs(base: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name in ["funnel_summary", *FUNNEL_GROUPS.keys(), "rejection_breakdown", "conversion_rates"]:
        tables.get(name, pd.DataFrame()).to_csv(base / f"{name}.csv", index=False)

    funnel = tables.get("funnel_summary", pd.DataFrame())
    lines = ["# Backtest Report", "", "## Candidate Funnel", ""]
    if funnel.empty:
        lines.append("No candidate lifecycle data available.")
    else:
        lines.append("| Stage | Count | % of Candidates | % From Prior |")
        lines.append("|---|---:|---:|---:|")
        for r in funnel.itertuples(index=False):
            lines.append(f"| {r.stage} | {r.count} | {r.pct_of_candidates}% | {r.pct_from_prior}% |")
    (base / "backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommendations(tables: dict[str, pd.DataFrame], trades: pd.DataFrame) -> list[str]:
    recs: list[str] = []
    if trades.empty:
        return ["No trades were simulated. First inspect the funnel and rejection summary to see whether scenarios are being filtered before entries."]
    by_scenario = tables.get("performance_by_scenario", pd.DataFrame())
    if not by_scenario.empty:
        min_trades = max(3, int(len(trades) * 0.05))
        eligible = by_scenario[by_scenario["trades"] >= min_trades]
        if not eligible.empty:
            best = eligible.sort_values(["avg_r", "profit_win_rate"], ascending=False).iloc[0]
            recs.append(f"Best scenario bucket with at least {min_trades} trades: {best.get('scenario')} / {best.get('side')} — avg R {best.get('avg_r')}, win rate {best.get('profit_win_rate')}%.")
            worst = eligible.sort_values(["avg_r", "profit_win_rate"], ascending=True).iloc[0]
            recs.append(f"Weakest scenario bucket with at least {min_trades} trades: {worst.get('scenario')} / {worst.get('side')} — avg R {worst.get('avg_r')}, win rate {worst.get('profit_win_rate')}%.")
    comp = tables.get("confirmation_components", pd.DataFrame())
    if not comp.empty:
        useful = comp[comp["trades_with"] >= 3].head(3)
        if not useful.empty:
            names = ", ".join(f"{r.confirmation} (+{r.avg_r_lift}R lift)" for r in useful.itertuples())
            recs.append(f"Confirmation components showing the strongest average-R lift: {names}.")
    target = tables.get("target_progress", pd.DataFrame())
    if not target.empty and "mfe_r" in trades.columns:
        reached_2 = target[target["milestone"].eq("MFE reached +2R")]
        reached_3 = target[target["milestone"].eq("MFE reached +3R")]
        if not reached_2.empty and not reached_3.empty:
            recs.append(f"Target progression: {float(reached_2.iloc[0]['pct_trades']):.1f}% reached +2R MFE and {float(reached_3.iloc[0]['pct_trades']):.1f}% reached +3R MFE. Use this to compare 2R partials versus full 3R targeting.")
    return recs or ["Sample is small. Run a wider date range/symbol set before changing strategy rules."]


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate strategy analytics dashboard from daily snapshot backtest outputs.")
    parser.add_argument("--min-trades", type=int, default=3, help="Minimum sample size to emphasize in some tables/recommendations.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base = REPORT_DIR / "backtest"
    trades_path = base / "trades.csv"
    cand_path = base / "entry_candidates.csv"
    summary_path = base / "summary.csv"
    lifecycle_path = base / "candidate_lifecycle.csv"
    if not trades_path.exists():
        raise SystemExit("Run replay_backtest.py first.")
    trades = pd.read_csv(trades_path)
    candidates = pd.read_csv(cand_path) if cand_path.exists() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    lifecycle = pd.read_csv(lifecycle_path) if lifecycle_path.exists() else pd.DataFrame()

    # Normalize common numeric/boolean columns.
    for df in [trades, candidates, lifecycle]:
        if not df.empty:
            for c in ["r_multiple", "mfe_r", "mae_r", "entry_volume_ratio", "entry_body_ratio", "setup_quality_score", "distance_pct", "gap_pct", "entry_atr_14", "atr_1x_r", "target_1r_result", "target_2r_result", "target_3r_result", "target_atr_1x_result"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            for c in ["reached_1r", "reached_2r", "reached_3r", "reached_atr_1x", "entry_eligible"]:
                if c in df.columns:
                    df[c] = _bool_series(df[c])

    if not trades.empty and "entry_time" in trades.columns:
        # Pandas 2.x raises on mixed timezone-aware strings unless utc=True is used.
        # Normalize to UTC, then convert back to New York/RTH time for time-of-day buckets.
        et = pd.to_datetime(trades["entry_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
        trades["entry_time_bucket"] = np.select(
            [et.dt.time < pd.to_datetime("10:30").time(), et.dt.time < pd.to_datetime("12:00").time(), et.dt.time <= pd.to_datetime("13:00").time()],
            ["09:30-10:29", "10:30-11:59", "12:00-13:00"],
            default="After 13:00"
        )

    lifecycle_full = _lifecycle_dataset(lifecycle, candidates, trades)
    tables: dict[str, pd.DataFrame] = {
        "performance_by_scenario": _performance_agg(trades, ["scenario", "side"]),
        "performance_by_symbol": _performance_agg(trades, ["symbol"]),
        "performance_by_watchlist_bucket": _performance_agg(trades, ["watchlist_bucket"]),
        "performance_by_zone_type": _performance_agg(trades, ["zone_type"]),
        "performance_by_quality_grade": _performance_agg(trades, ["setup_quality_grade"]),
        "performance_by_entry_kind": _performance_agg(trades, ["entry_kind"]),
        "performance_by_time_of_day": _time_of_day(trades),
        "performance_by_entry_time_bucket": _performance_agg(trades, ["entry_time_bucket"]),
        "performance_by_snapshot_mode": _performance_agg(trades, ["snapshot_mode"]),
        "performance_by_gap_direction": _performance_agg(trades, ["gap_direction"]),
        "performance_by_gap_zone_context": _performance_agg(trades, ["gap_zone_context"]),
        "performance_by_zone_thesis": _performance_agg(trades, ["zone_thesis"]),
        "performance_by_zone_movement_state": _performance_agg(trades, ["zone_movement_state"]),
        "performance_by_movement_watchlist_bucket": _performance_agg(trades, ["movement_watchlist_bucket"]),
        "performance_by_volume_state": _performance_agg(trades, ["volume_state"]),
        "performance_by_vpa_state": _performance_agg(trades, ["vpa_state"]),
        "performance_by_current_price_session": _performance_agg(trades, ["current_price_session"]),
        "exit_reason_summary": _exit_summary(trades),
        "confirmation_components": _confirmation_components(trades),
        "target_progress": _target_progress(trades),
        "target_model_comparison": _target_model_comparison(trades),
        "entry_funnel": _funnel(candidates, trades),
        "rejection_summary": _rejection_summary(candidates),
        "opportunity_cost_proxy": _opportunity_cost(candidates, trades),
        "funnel_summary": _funnel_summary_lifecycle(lifecycle_full),
        "rejection_breakdown": _rejection_breakdown_lifecycle(lifecycle_full),
        "conversion_rates": _conversion_rates_lifecycle(lifecycle_full),
    }
    tables.update({name: _funnel_by_lifecycle(lifecycle_full, col) for name, col in FUNNEL_GROUPS.items()})

    tables["reversal_rejection_breakdowns"] = _build_reversal_rejection_breakdowns(trades)
    tables["reversal_rejection_rule_variants"] = _build_reversal_rule_variants(trades)
    tables["reversal_rejection_best_filters"] = _identify_reversal_best_filters(tables["reversal_rejection_rule_variants"])
    tables["exit_path_summary"] = _build_exit_path_summary(trades)
    tables["exit_path_first_event"] = _build_exit_path_first_event(trades)
    tables["exit_path_flags"] = _build_exit_path_flags(trades)
    tables["exit_path_management_variants"] = _build_exit_path_management_variants(trades)

    # Add plain-English edge labels to the most important performance tables.
    for _name, _cols in {
        "performance_by_scenario": ["scenario", "side"],
        "performance_by_entry_kind": ["entry_kind"],
        "performance_by_quality_grade": ["setup_quality_grade"],
        "performance_by_watchlist_bucket": ["watchlist_bucket"],
        "performance_by_zone_type": ["zone_type"],
        "performance_by_time_of_day": ["entry_hour"],
        "performance_by_entry_time_bucket": ["entry_time_bucket"],
        "performance_by_symbol": ["symbol"],
        "performance_by_snapshot_mode": ["snapshot_mode"],
        "performance_by_gap_direction": ["gap_direction"],
        "performance_by_gap_zone_context": ["gap_zone_context"],
        "performance_by_zone_thesis": ["zone_thesis"],
        "performance_by_zone_movement_state": ["zone_movement_state"],
        "performance_by_movement_watchlist_bucket": ["movement_watchlist_bucket"],
        "performance_by_volume_state": ["volume_state"],
        "performance_by_vpa_state": ["vpa_state"],
        "performance_by_current_price_session": ["current_price_session"],
    }.items():
        if _name in tables and tables[_name] is not None and not tables[_name].empty:
            tables[_name] = _edge_table(tables[_name], _cols)

    analytics_dir = base / "analytics"
    _write_csvs(analytics_dir, tables)
    _write_funnel_outputs(base, tables)

    # KPI cards
    if not summary.empty:
        row = summary.iloc[0].to_dict()
    else:
        row = {
            "trades": len(trades),
            "profit_win_rate": round((trades["r_multiple"] > 0).mean() * 100, 2) if not trades.empty else 0,
            "target_hit_rate": round(_bool_series(trades.get("reached_3r", pd.Series(False, index=trades.index))).mean() * 100, 2) if not trades.empty else 0,
            "avg_r": round(trades["r_multiple"].mean(), 3) if not trades.empty else 0,
            "total_r": round(trades["r_multiple"].sum(), 3) if not trades.empty else 0,
            "avg_mfe_r": round(trades["mfe_r"].mean(), 3) if "mfe_r" in trades.columns and not trades.empty else 0,
            "avg_mae_r": round(trades["mae_r"].mean(), 3) if "mae_r" in trades.columns and not trades.empty else 0,
        }
    cards = "".join([
        _card("Trades", row.get("trades", len(trades)), _quality_note(trades)),
        _card("Profit win rate", f"{_fmt_num(row.get('profit_win_rate', 0))}%", "Trades that closed above 0R"),
        _card("3R target hit rate", f"{_fmt_num(row.get('target_hit_rate', row.get('reached_3r_rate', 0)))}%", "Full modeled target, not all profitable exits"),
        _card("Average R", _fmt_num(row.get("avg_r", 0), 3), "Expectancy per trade in risk units"),
        _card("Total R", _fmt_num(row.get("total_r", 0), 3), "Aggregate strategy contribution"),
        _card("Reached +1R", f"{_fmt_num(row.get('reached_1r_rate', 0))}%", "Touched +1R at any point"),
        _card("Reached +2R", f"{_fmt_num(row.get('reached_2r_rate', 0))}%", "Touched +2R at any point"),
        _card("Avg MFE / MAE", f"{_fmt_num(row.get('avg_mfe_r', 0), 2)}R / {_fmt_num(row.get('avg_mae_r', 0), 2)}R", "Best/worst unrealized movement"),
    ])

    scenario_chart = _simple_bars(tables.get('performance_by_scenario'), 'scenario', 'avg_r', 'Average R by scenario', 'This is the clearest read on whether results are random or scenario-dependent.', max_rows=8, places=3, suffix='R')
    entry_chart = _simple_bars(tables.get('performance_by_entry_kind'), 'entry_kind', 'avg_r', 'Average R by entry type', 'Breakouts/breakdowns should be evaluated separately from reversals/rejections.', max_rows=8, places=3, suffix='R')
    funnel_chart = _simple_bars(tables.get('entry_funnel'), 'stage', 'count', 'Entry funnel counts', 'Shows where watchlist scenarios turn into simulated entries or get filtered.', max_rows=10, places=0, suffix='')
    candidate_funnel_chart = _simple_bars(tables.get('funnel_summary'), 'stage', 'count', 'Candidate lifecycle funnel', 'Snapshot candidates through zone touch, eligibility, entries, and winners.', max_rows=8, places=0, suffix='')
    target_chart = _simple_bars(tables.get('target_progress'), 'milestone', 'pct_trades', 'Target progression', 'A trade can be useful even if it does not reach the full 3R target.', max_rows=10, places=1, suffix='%')
    strategy_read = _strategy_read_html(tables, trades)
    reversal_diagnostics = _reversal_diagnostics_html(tables, trades)
    exit_path_audit = _exit_path_audit_html(tables, trades)
    glossary = _glossary_html()

    recs = _recommendations(tables, trades)
    rec_html = "".join(f"<li>{r}</li>" for r in recs)

    css = """
    <style>
    :root{--bg:#0f172a;--panel:#111827;--panel2:#1f2937;--line:#374151;--text:#e5e7eb;--muted:#9ca3af;--good:#22c55e;--ok:#84cc16;--warn:#f59e0b;--bad:#ef4444;--blue:#38bdf8}
    body{font-family:Arial,Helvetica,sans-serif;background:var(--bg);color:var(--text);margin:0}.wrap{max-width:1320px;margin:0 auto;padding:28px}h1{margin-bottom:4px}h2{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}h3{margin:0 0 8px}.muted,.subtle{color:var(--muted)}.subtle{font-size:12px;margin-top:4px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}.label{color:var(--muted);font-size:13px}.value{font-size:24px;font-weight:800;margin-top:6px}.tbl{border-collapse:collapse;width:100%;margin:12px 0 24px;background:var(--panel);font-size:13px}.tbl th,.tbl td{border:1px solid var(--line);padding:8px;text-align:left}.tbl th{background:var(--panel2);color:#d1d5db}.note{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.35);border-radius:14px;padding:14px;line-height:1.45;margin:16px 0}.note.warn{background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.35)}.grid2{display:grid;grid-template-columns:1fr;gap:18px}@media(min-width:1000px){.grid2{grid-template-columns:1fr 1fr}}code{background:#020617;border:1px solid var(--line);padding:2px 5px;border-radius:5px}.viz{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;margin:14px 0}.barrow{display:grid;grid-template-columns:220px 1fr 80px;gap:10px;align-items:center;margin:9px 0}.barlabel{font-size:12px;color:#d1d5db;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bartrack{height:12px;border-radius:999px;background:#263244;overflow:hidden}.barfill{height:100%;border-radius:999px}.barfill.pos{background:linear-gradient(90deg,var(--blue),var(--good))}.barfill.neg{background:linear-gradient(90deg,var(--bad),var(--warn))}.barval{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}.good{color:#bbf7d0}.ok{color:#d9f99d}.warn{color:#fde68a}.bad{color:#fecaca}.interactive{background:rgba(15,23,42,.45);border:1px solid var(--line);border-radius:18px;padding:18px;margin:24px 0}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}.toolbar button{background:#1e293b;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:8px 10px;cursor:pointer}.toolbar button:hover{border-color:var(--blue)}.filtergrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:14px 0}.filterbox{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px;max-height:210px;overflow:auto}.filtertitle{font-weight:800;margin-bottom:8px}.filterbox label{display:block;font-size:13px;color:#d1d5db;margin:5px 0}.filterbox input{vertical-align:middle}details summary{cursor:pointer}
    </style>
    """

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Interactive Strategy Analytics Dashboard</title>{css}</head><body><main class='wrap'>
    <h1>Interactive Strategy Analytics Dashboard</h1>
    <div class='muted'>Generated from <code>reports/backtest/trades.csv</code>, <code>reports/backtest/entry_candidates.csv</code>, and <code>reports/backtest/candidate_lifecycle.csv</code>. This dashboard is meant to show which zone scenarios have edge, not just a headline win rate.</div>
    <div class='cards'>{cards}</div>
    {strategy_read}
    <div class='note'><strong>How to read this:</strong> A strategy is not a dice roll if certain scenario buckets consistently show positive average R while others do not. Focus first on <em>average R</em>, <em>reached +1R/+2R</em>, and <em>MFE vs MAE</em>; profit win rate alone can hide whether winners are too small or losses are too large.</div>
    {_interactive_dashboard_html(trades)}
    {reversal_diagnostics}
    {exit_path_audit}
    <h2>Static full-sample dashboard</h2>
    <div class='grid2'>{scenario_chart}{entry_chart}</div>
    <div class='grid2'>{funnel_chart}{target_chart}</div>

    <h2>Variable glossary</h2>{glossary}
    <h2>Candidate Funnel</h2>
    <div class='grid2'>{candidate_funnel_chart}{_table(tables.get('conversion_rates'))}</div>
    {_table(tables.get('funnel_summary'))}
    <div class='grid2'>
      <section><h2>Funnel By Scenario</h2>{_table(tables.get('funnel_by_scenario'))}</section>
      <section><h2>Funnel By Grade</h2>{_table(tables.get('funnel_by_grade'))}</section>
    </div>
    <div class='grid2'>
      <section><h2>Funnel By Distance</h2>{_table(tables.get('funnel_by_distance_bucket'))}</section>
      <section><h2>Lifecycle Rejections</h2>{_table(tables.get('rejection_breakdown'))}</section>
    </div>
    <h2>Opportunity Funnel</h2>{_table(tables['entry_funnel'])}
    <h2>Target Progression</h2>{_table(tables['target_progress'])}
    <h2>Target Model Comparison</h2>{_table(tables['target_model_comparison'])}
    <h2>By Entry Time Bucket</h2>{_table(tables.get('performance_by_entry_time_bucket', pd.DataFrame()))}

    <div class='grid2'>
      <section><h2>By Scenario</h2>{_table(tables['performance_by_scenario'])}</section>
      <section><h2>By Entry Type</h2>{_table(tables['performance_by_entry_kind'])}</section>
    </div>
    <div class='grid2'>
      <section><h2>By Zone Type</h2>{_table(tables['performance_by_zone_type'])}</section>
      <section><h2>By Setup Grade</h2>{_table(tables['performance_by_quality_grade'])}</section>
    </div>
    <div class='grid2'>
      <section><h2>By Watchlist Bucket</h2>{_table(tables['performance_by_watchlist_bucket'])}</section>
      <section><h2>By Time of Day</h2>{_table(tables['performance_by_time_of_day'])}</section>
    </div>
    <h2>By Symbol</h2>{_table(tables['performance_by_symbol'], max_rows=60)}

    <h2>Preopen / Movement Context Analysis</h2>
    <div class='muted'>These tables are most useful when replay was run with <code>--snapshot-mode preopen</code>. They compare 8:00 AM gap/movement context against the later regular-session trade results.</div>
    <div class='grid2'>
      <section><h2>By Snapshot Mode</h2>{_table(tables.get('performance_by_snapshot_mode'))}</section>
      <section><h2>By Current Price Session</h2>{_table(tables.get('performance_by_current_price_session'))}</section>
    </div>
    <div class='grid2'>
      <section><h2>By Gap Direction</h2>{_table(tables.get('performance_by_gap_direction'))}</section>
      <section><h2>By Gap vs Zone Context</h2>{_table(tables.get('performance_by_gap_zone_context'))}</section>
    </div>
    <div class='grid2'>
      <section><h2>By Zone Thesis</h2>{_table(tables.get('performance_by_zone_thesis'))}</section>
      <section><h2>By Zone Movement State</h2>{_table(tables.get('performance_by_zone_movement_state'))}</section>
    </div>
    <div class='grid2'>
      <section><h2>By Movement Watchlist Bucket</h2>{_table(tables.get('performance_by_movement_watchlist_bucket'))}</section>
      <section><h2>By VPA State</h2>{_table(tables.get('performance_by_vpa_state'))}</section>
    </div>
    <h2>Confirmation Component Analysis</h2><div class='muted'>This compares trades that included each confirmation component against trades that did not. Treat components with small samples as hypotheses, not final conclusions.</div>{_table(tables['confirmation_components'])}
    <div class='grid2'>
      <section><h2>Exit Reason Summary</h2>{_table(tables['exit_reason_summary'])}</section>
      <section><h2>Rejected Entry Candidates</h2>{_table(tables['rejection_summary'])}</section>
    </div>
    <h2>Opportunity Cost Proxy</h2><div class='muted'>This summarizes rejected candidate types. A future replay enhancement can add MFE/MAE for rejected candidates to measure what happened after rejection.</div>{_table(tables['opportunity_cost_proxy'])}
    <h2>Raw Summary CSV</h2>{_table(summary)}
    <div class='note'><strong>Files written:</strong> CSV breakdowns are in <code>reports/backtest/analytics/</code>. Main dashboard: <code>reports/backtest/strategy_dashboard.html</code>.</div>
    </main></body></html>"""

    out = base / "strategy_dashboard.html"
    out.write_text(html, encoding="utf-8")
    # Keep old filename for compatibility.
    old = base / "performance_summary.html"
    old.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote analytics CSVs to {analytics_dir}")
    print(f"Wrote lifecycle funnel CSVs and markdown report to {base}")


if __name__ == "__main__":
    main()
