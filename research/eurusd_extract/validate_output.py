from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format for {path}. Use parquet or csv.")


def _infer_expected_seconds(path: Path, ts: pd.Series) -> int | None:
    name = path.name.lower()
    if "_1m" in name:
        return 60
    if "_5m" in name:
        return 300
    if "_1h" in name:
        return 3600
    if len(ts) < 3:
        return None
    diffs = ts.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    return int(diffs.mode().iloc[0])


def _count_gaps(ts: pd.Series, expected_seconds: int | None) -> dict[str, int | None]:
    if expected_seconds is None or len(ts) < 2:
        return {"gap_intervals": None, "missing_steps": None, "expected_seconds": expected_seconds}
    diffs = ts.diff().dropna().dt.total_seconds()
    gap_intervals = int((diffs > expected_seconds).sum())
    missing_steps = int(np.maximum(np.round(diffs / expected_seconds).astype(int) - 1, 0).sum())
    return {
        "gap_intervals": gap_intervals,
        "missing_steps": missing_steps,
        "expected_seconds": expected_seconds,
    }


def _validate_monotonic_unique_ts(ts: pd.Series) -> dict[str, Any]:
    dup_count = int(ts.duplicated().sum())
    strict_increasing = bool((ts.diff().dropna() > pd.Timedelta(0)).all())
    monotonic = bool(ts.is_monotonic_increasing)
    return {
        "duplicate_timestamps": dup_count,
        "strictly_increasing": strict_increasing,
        "monotonic_increasing": monotonic,
    }


def validate_ticks(path: Path) -> tuple[dict[str, Any], list[str]]:
    df = _load_frame(path)
    errors: list[str] = []

    required_one_of = {"mid"} | {"bid", "ask"}
    has_mid = "mid" in df.columns
    has_bid_ask = {"bid", "ask"}.issubset(df.columns)
    if "ts_utc" not in df.columns:
        errors.append("ticks missing ts_utc column")
    if not has_mid and not has_bid_ask:
        errors.append("ticks must contain either mid or both bid and ask columns")

    if errors:
        return {"file": str(path), "type": "ticks"}, errors

    df = df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    req_cols = ["ts_utc"]
    if has_mid:
        req_cols.append("mid")
    if has_bid_ask:
        req_cols.extend(["bid", "ask"])

    null_required = int(df[req_cols].isna().sum().sum())
    if null_required > 0:
        errors.append(f"ticks required columns contain NaN/null values: {null_required}")

    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    ts = df["ts_utc"]
    mono_info = _validate_monotonic_unique_ts(ts)
    if not mono_info["strictly_increasing"]:
        errors.append("ticks ts_utc is not strictly increasing")
    if mono_info["duplicate_timestamps"] > 0:
        errors.append(f"ticks has duplicate timestamps: {mono_info['duplicate_timestamps']}")

    gap_info = _count_gaps(ts, _infer_expected_seconds(path, ts))

    summary = {
        "file": str(path),
        "type": "ticks",
        "rows": int(len(df)),
        "start_ts": str(ts.iloc[0]) if not df.empty else None,
        "end_ts": str(ts.iloc[-1]) if not df.empty else None,
        **mono_info,
        **gap_info,
    }
    return summary, errors


def validate_ohlc(path: Path) -> tuple[dict[str, Any], list[str]]:
    df = _load_frame(path)
    errors: list[str] = []
    required = ["ts_utc", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"ohlc missing required columns: {missing}")
        return {"file": str(path), "type": "ohlc"}, errors

    df = df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    null_required = int(df[required].isna().sum().sum())
    if null_required > 0:
        errors.append(f"ohlc required columns contain NaN/null values: {null_required}")

    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    ts = df["ts_utc"]
    mono_info = _validate_monotonic_unique_ts(ts)
    if not mono_info["strictly_increasing"]:
        errors.append("ohlc ts_utc is not strictly increasing")
    if mono_info["duplicate_timestamps"] > 0:
        errors.append(f"ohlc has duplicate timestamps: {mono_info['duplicate_timestamps']}")

    invariant_ok = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
    )
    invariant_violations = int((~invariant_ok).sum())
    if invariant_violations > 0:
        errors.append(f"ohlc invariant violations found: {invariant_violations}")

    gap_info = _count_gaps(ts, _infer_expected_seconds(path, ts))

    summary = {
        "file": str(path),
        "type": "ohlc",
        "rows": int(len(df)),
        "bar_count": int(len(df)),
        "start_ts": str(ts.iloc[0]) if not df.empty else None,
        "end_ts": str(ts.iloc[-1]) if not df.empty else None,
        "invariant_violations": invariant_violations,
        **mono_info,
        **gap_info,
    }
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate extracted EURUSD normalized outputs")
    parser.add_argument("--ticks", help="Path to normalized ticks parquet/csv")
    parser.add_argument(
        "--ohlc",
        nargs="+",
        help="One or more normalized OHLC parquet/csv files",
    )
    parser.add_argument(
        "--summary-out",
        default="research/artefacts/logs/validation_summary.json",
        help="Path to write validation summary JSON",
    )
    args = parser.parse_args()

    if not args.ticks and not args.ohlc:
        raise ValueError("Provide --ticks and/or --ohlc")

    summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    if args.ticks:
        summary, errs = validate_ticks(Path(args.ticks))
        summaries.append(summary)
        errors.extend(errs)

    if args.ohlc:
        for p in args.ohlc:
            summary, errs = validate_ohlc(Path(p))
            summaries.append(summary)
            errors.extend(errs)

    result = {"ok": len(errors) == 0, "summaries": summaries, "errors": errors}
    print(json.dumps(result, indent=2))

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote validation summary to {out}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
