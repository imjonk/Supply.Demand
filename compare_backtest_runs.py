from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import REPORT_DIR


FILES = [
    "summary.csv", "funnel_summary.csv", "conversion_rates.csv", "rejection_breakdown.csv",
    "funnel_by_scenario.csv", "feature_conversion_summary.csv",
    "feature_conversion_best_predictors.csv", "trades.csv", "candidate_lifecycle.csv",
]
DEFAULT_BASELINE = REPORT_DIR / "backtest_baselines" / "v0.38-baseline-feature-analytics"
DEFAULT_CURRENT = REPORT_DIR / "backtest"
METRICS = ["candidate_count", "zone_reached_pct", "zone_exited_pct", "trade_conversion_pct", "win_rate", "avg_r", "total_r", "reached_1r_pct", "reached_2r_pct", "reached_3r_pct"]
FEATURE_METRICS = ["candidate_count", "zone_reached_pct", "zone_exited_pct", "trade_entered_pct", "winner_pct", "reached_1r_pct", "reached_2r_pct", "reached_3r_pct", "avg_r", "total_r", "avg_mfe_r", "avg_mae_r"]


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _num(value, default=0.0) -> float:
    try:
        out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return default if pd.isna(out) else float(out)
    except Exception:
        return default


def _bool_rate(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    s = df[col].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
    return round(s.mean() * 100.0, 2)


def _stage_pct(funnel: pd.DataFrame, stage: str) -> float:
    if funnel.empty or "stage" not in funnel.columns:
        return 0.0
    row = funnel[funnel["stage"].astype(str).str.lower().eq(stage.lower())]
    return _num(row.iloc[0].get("pct_of_candidates"), 0.0) if not row.empty else 0.0


def _stage_count(funnel: pd.DataFrame, stage: str) -> int:
    if funnel.empty or "stage" not in funnel.columns:
        return 0
    row = funnel[funnel["stage"].astype(str).str.lower().eq(stage.lower())]
    return int(_num(row.iloc[0].get("count"), 0.0)) if not row.empty else 0


def _load_run(folder: Path) -> dict[str, pd.DataFrame]:
    return {name: _read(folder / name) for name in FILES}


def _run_metrics(run: dict[str, pd.DataFrame]) -> dict[str, float]:
    summary = run["summary.csv"]
    funnel = run["funnel_summary.csv"]
    trades = run["trades.csv"]
    lifecycle = run["candidate_lifecycle.csv"]
    candidates = _stage_count(funnel, "Snapshot Candidates") or len(lifecycle)
    trade_count = _stage_count(funnel, "Trade Entered") or len(trades)
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    return {
        "candidate_count": float(candidates),
        "zone_reached_pct": _stage_pct(funnel, "Zone Reached"),
        "zone_exited_pct": _stage_pct(funnel, "Zone Exited"),
        "trade_conversion_pct": _stage_pct(funnel, "Trade Entered") or round(trade_count / max(candidates, 1) * 100.0, 2),
        "win_rate": _num(row.get("profit_win_rate"), round((pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce") > 0).mean() * 100.0, 2) if not trades.empty else 0.0),
        "avg_r": _num(row.get("avg_r"), pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce").mean() if not trades.empty else 0.0),
        "total_r": _num(row.get("total_r"), pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce").sum() if not trades.empty else 0.0),
        "reached_1r_pct": _num(row.get("reached_1r_rate"), _bool_rate(trades, "reached_1r")),
        "reached_2r_pct": _num(row.get("reached_2r_rate"), _bool_rate(trades, "reached_2r")),
        "reached_3r_pct": _num(row.get("reached_3r_rate"), _bool_rate(trades, "reached_3r")),
    }


def _label(delta: float, higher_is_better: bool = True) -> str:
    if abs(delta) < 0.01:
        return "neutral"
    return "improvement" if (delta > 0) == higher_is_better else "degradation"


def _sample_label(base_n: float, cur_n: float, trades: float | None = None) -> str:
    n = min(base_n, cur_n)
    if n < 30 or (trades is not None and trades < 10):
        return "sample_size_warning"
    return "ok"


def _comparison_summary(base: dict[str, pd.DataFrame], cur: dict[str, pd.DataFrame]) -> pd.DataFrame:
    b, c = _run_metrics(base), _run_metrics(cur)
    rows = []
    for metric in METRICS:
        delta = c[metric] - b[metric]
        rows.append({"metric": metric, "baseline": round(b[metric], 4), "current": round(c[metric], 4), "delta": round(delta, 4), "pct_delta": round(delta / abs(b[metric]) * 100.0, 2) if b[metric] else 0.0, "label": _label(delta), "sample_size_label": _sample_label(b["candidate_count"], c["candidate_count"], min(_stage_count(base["funnel_summary.csv"], "Trade Entered"), _stage_count(cur["funnel_summary.csv"], "Trade Entered")))})
    return pd.DataFrame(rows)


def _feature_frame(run: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = run["feature_conversion_summary.csv"].copy()
    return df if {"feature", "bucket"}.issubset(df.columns) else pd.DataFrame(columns=["feature", "bucket", *FEATURE_METRICS])


def _scenario_frame(run: dict[str, pd.DataFrame]) -> pd.DataFrame:
    features = _feature_frame(run)
    scenario = features[features["feature"].eq("scenario_family")].rename(columns={"bucket": "scenario_family"})
    if not scenario.empty:
        return scenario.drop(columns=["feature"], errors="ignore")
    f = run["funnel_by_scenario.csv"].copy()
    if f.empty:
        return pd.DataFrame(columns=["scenario_family"])
    f = f.rename(columns={"candidates": "candidate_count", "reached_pct": "zone_reached_pct", "trade_pct": "trade_entered_pct", "win_pct": "winner_pct"})
    exited = f["zone_exited"] if "zone_exited" in f.columns else pd.Series(0, index=f.index)
    f["zone_exited_pct"] = exited.astype(float) / f["candidate_count"].replace(0, pd.NA).astype(float) * 100.0
    return f


def _compare_frames(base: pd.DataFrame, cur: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    b = base.copy(); c = cur.copy()
    for df, suffix in [(b, "_baseline"), (c, "_current")]:
        for m in metrics:
            if m not in df.columns:
                df[m] = 0.0
        df.rename(columns={m: f"{m}{suffix}" for m in metrics}, inplace=True)
    out = b.merge(c, on=keys, how="outer").fillna(0)
    for m in metrics:
        out[f"{m}_delta"] = pd.to_numeric(out[f"{m}_current"], errors="coerce") - pd.to_numeric(out[f"{m}_baseline"], errors="coerce")
    main = "trade_entered_pct" if "trade_entered_pct" in metrics else metrics[0]
    out["label"] = out[f"{main}_delta"].apply(_label)
    out["sample_size_label"] = out.apply(lambda r: _sample_label(_num(r.get("candidate_count_baseline")), _num(r.get("candidate_count_current")), min(_num(r.get("trades_baseline")), _num(r.get("trades_current"))) if "trades" in metrics else None), axis=1)
    return out.sort_values([f"{main}_delta", f"{main}_current"], ascending=[False, False])


def _compare_rejections(base: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    keys = ["rejection_reason"]
    metrics = ["count", "pct_of_candidates", "pct_of_rejections"]
    for df in [base, cur]:
        if "rejection_reason" not in df.columns:
            df["rejection_reason"] = pd.Series(dtype=str)
    out = _compare_frames(base, cur, keys, metrics)
    out["label"] = out["pct_of_candidates_delta"].apply(lambda d: _label(d, higher_is_better=False))
    return out.sort_values(["count_current", "pct_of_candidates_delta"], ascending=[False, True])


def _html_table(df: pd.DataFrame, rows: int = 40) -> str:
    return "<p>No rows.</p>" if df.empty else df.head(rows).to_html(index=False, escape=False)


def _write_html(path: Path, summary: pd.DataFrame, scenario: pd.DataFrame, feature: pd.DataFrame, reject: pd.DataFrame) -> None:
    css = "<style>body{font-family:Arial;margin:24px;max-width:1400px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ccc;padding:6px;text-align:left}th{background:#eee}.note{padding:10px;background:#f6f8fa;border:1px solid #ddd;margin:12px 0}</style>"
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Backtest Run Comparison</title>{css}</head><body><h1>Backtest Run Comparison</h1><div class='note'>Labels are directional diagnostics. Treat rows marked <code>sample_size_warning</code> as hypotheses.</div><h2>Summary</h2>{_html_table(summary)}<h2>By Scenario</h2>{_html_table(scenario)}<h2>By Feature</h2>{_html_table(feature)}<h2>Rejection Breakdown</h2>{_html_table(reject)}</body></html>"
    path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare current backtest analytics against a saved baseline run.")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    args = p.parse_args()
    baseline = _load_run(args.baseline)
    current = _load_run(args.current)
    out_dir = args.current
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _comparison_summary(baseline, current)
    scenario = _compare_frames(_scenario_frame(baseline), _scenario_frame(current), ["scenario_family"], FEATURE_METRICS)
    feature = _compare_frames(_feature_frame(baseline), _feature_frame(current), ["feature", "bucket"], FEATURE_METRICS)
    rejection = _compare_rejections(baseline["rejection_breakdown.csv"].copy(), current["rejection_breakdown.csv"].copy())

    summary.to_csv(out_dir / "comparison_summary.csv", index=False)
    scenario.to_csv(out_dir / "comparison_by_scenario.csv", index=False)
    feature.to_csv(out_dir / "comparison_by_feature.csv", index=False)
    rejection.to_csv(out_dir / "comparison_rejection_breakdown.csv", index=False)
    _write_html(out_dir / "comparison_report.html", summary, scenario, feature, rejection)
    print(f"Wrote comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
