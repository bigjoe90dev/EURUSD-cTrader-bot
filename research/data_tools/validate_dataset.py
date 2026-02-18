from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import dump_json, to_iso_z


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type for {path}. Use parquet or csv.")


def _infer_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"open", "high", "low", "close"}.issubset(cols):
        return "ohlc"
    if {"bid", "ask"}.issubset(cols) or "mid" in cols:
        return "ticks"
    raise ValueError("Could not infer dataset kind. Pass --kind explicitly.")


def _infer_expected_seconds_from_name(path: Path | None) -> int | None:
    if path is None:
        return None
    name = path.name.lower()
    if "_1m" in name:
        return 60
    if "_5m" in name:
        return 300
    if "_1h" in name:
        return 3600
    return None


def _compute_gap_stats(ts: pd.Series, expected_seconds: int | None) -> tuple[int | None, int | None]:
    if len(ts) < 2:
        return 0, 0

    diffs = ts.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs >= 0]
    if diffs.empty:
        return 0, 0

    longest_gap = int(diffs.max())
    if expected_seconds is None:
        return None, longest_gap
    gaps = int((diffs > expected_seconds).sum())
    return gaps, longest_gap


def validate_frame(
    df: pd.DataFrame,
    kind: str,
    expected_seconds: int | None = None,
    path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    summary: dict[str, Any] = {}

    if "ts_utc" not in df.columns:
        return {"ok": False}, ["Missing required column: ts_utc"]

    work = df.copy()
    parsed_ts = pd.to_datetime(work["ts_utc"], utc=True, errors="coerce")
    invalid_ts = int(parsed_ts.isna().sum())
    if invalid_ts > 0:
        errors.append(f"Invalid ts_utc values: {invalid_ts}")

    work["ts_utc"] = parsed_ts
    work = work.dropna(subset=["ts_utc"])

    if kind == "ticks":
        has_mid = "mid" in work.columns
        has_bid_ask = {"bid", "ask"}.issubset(set(work.columns))

        if not has_mid and not has_bid_ask:
            errors.append("Ticks dataset must contain either `mid` or both `bid` and `ask`")

        required = ["ts_utc"]
        if has_mid:
            required.append("mid")
        if has_bid_ask:
            required.extend(["bid", "ask"])

        nulls = int(work[required].isna().sum().sum()) if required else 0
        if nulls > 0:
            errors.append(f"Null values in required tick fields: {nulls}")

        if has_bid_ask:
            bid_gt_ask = int((work["bid"] > work["ask"]).sum())
            summary["bid_greater_than_ask"] = bid_gt_ask
            if bid_gt_ask > 0:
                errors.append(f"Found rows where bid > ask: {bid_gt_ask}")

    elif kind == "ohlc":
        required = ["ts_utc", "open", "high", "low", "close"]
        missing = [c for c in required if c not in work.columns]
        if missing:
            errors.append(f"Missing required OHLC columns: {missing}")
        else:
            nulls = int(work[required].isna().sum().sum())
            if nulls > 0:
                errors.append(f"Null values in required OHLC fields: {nulls}")

            invariant_ok = (
                (work["high"] >= work[["open", "close", "low"]].max(axis=1))
                & (work["low"] <= work[["open", "close", "high"]].min(axis=1))
            )
            invariant_violations = int((~invariant_ok).sum())
            summary["ohlc_invariant_violations"] = invariant_violations
            if invariant_violations > 0:
                errors.append(f"OHLC invariant violations: {invariant_violations}")
    else:
        errors.append(f"Unsupported kind: {kind}")

    ts = work["ts_utc"]
    duplicates = int(ts.duplicated().sum())
    if duplicates > 0:
        errors.append(f"Duplicate timestamps found: {duplicates}")

    strictly_increasing = bool((ts.diff().dropna() > pd.Timedelta(0)).all())
    if not strictly_increasing:
        errors.append("Timestamps are not strictly increasing")

    inferred_expected = _infer_expected_seconds_from_name(path)
    effective_expected = expected_seconds if expected_seconds is not None else inferred_expected
    gaps_found, longest_gap = _compute_gap_stats(ts, effective_expected)

    summary.update(
        {
            "ok": len(errors) == 0,
            "kind": kind,
            "row_count": int(len(work)),
            "start_ts_utc": to_iso_z(ts.iloc[0]) if not work.empty else None,
            "end_ts_utc": to_iso_z(ts.iloc[-1]) if not work.empty else None,
            "duplicates_found": duplicates,
            "strictly_increasing": strictly_increasing,
            "timezone": "UTC",
            "expected_interval_seconds": effective_expected,
            "gap_definition": (
                "gap = delta_seconds > expected_interval_seconds"
                if effective_expected is not None
                else "gap count unavailable because expected interval is undefined"
            ),
            "gaps_found": gaps_found,
            "longest_gap_seconds": longest_gap,
            "errors": errors,
        }
    )

    return summary, errors


def validate_file(
    path: str | Path,
    kind: str = "auto",
    expected_seconds: int | None = None,
) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    df = _load_frame(p)
    use_kind = _infer_kind(df) if kind == "auto" else kind
    summary, errors = validate_frame(df, use_kind, expected_seconds=expected_seconds, path=p)
    summary["file"] = str(p)
    summary["ok"] = len(errors) == 0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate market data datasets (fail-closed)")
    parser.add_argument("--input", nargs="+", required=True, help="Parquet/CSV files to validate")
    parser.add_argument(
        "--kind",
        choices=["auto", "ticks", "ohlc"],
        default="auto",
        help="Dataset kind. Use auto to infer from columns.",
    )
    parser.add_argument("--expected-seconds", type=int, help="Expected spacing in seconds, e.g. 60 for 1m")
    parser.add_argument(
        "--summary-out",
        default="research/artefacts/logs/validation_report.json",
        help="Path to JSON summary report",
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    all_errors: list[str] = []

    for path in args.input:
        summary = validate_file(path, kind=args.kind, expected_seconds=args.expected_seconds)
        summaries.append(summary)
        if not summary["ok"]:
            for err in summary["errors"]:
                all_errors.append(f"{path}: {err}")

    report = {
        "ok": len(all_errors) == 0,
        "dataset_count": len(summaries),
        "datasets": summaries,
        "errors": all_errors,
    }
    print(json.dumps(report, indent=2))
    dump_json(args.summary_out, report)
    print(f"Wrote validation report to {args.summary_out}")

    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
