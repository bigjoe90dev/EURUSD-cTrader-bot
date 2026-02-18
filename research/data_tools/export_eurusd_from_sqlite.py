from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import dump_json, ensure_dir, load_yaml, parse_utc_ts, to_iso_z
from .validate_dataset import validate_file

TS_CANDIDATES = ("ts_utc", "datetime_utc", "timestamp", "datetime", "time", "ts")


def _q(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def _optional_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "auto"}:
        return None
    return parse_utc_ts(value)


def _pick_ts_col(columns: set[str]) -> str | None:
    for c in TS_CANDIDATES:
        if c in columns:
            return c
    return None


def _load_schema(cur: sqlite3.Cursor) -> dict[str, set[str]]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    out: dict[str, set[str]] = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({_q(t)})")
        out[t] = {r[1] for r in cur.fetchall()}
    return out


def _table_min_max_count(cur: sqlite3.Cursor, table: str, ts_col: str, where_sql: str = "") -> tuple[int, Any, Any]:
    where_clause = f" WHERE {where_sql}" if where_sql else ""
    cur.execute(
        f"SELECT COUNT(*), MIN({_q(ts_col)}), MAX({_q(ts_col)}) FROM {_q(table)}{where_clause}"
    )
    row_count, min_ts, max_ts = cur.fetchone()
    return int(row_count), min_ts, max_ts


def _choose_tick_candidate(cur: sqlite3.Cursor, schema: dict[str, set[str]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for table, cols in schema.items():
        ts_col = _pick_ts_col(cols)
        if not ts_col or not {"bid", "ask"}.issubset(cols):
            continue

        diag: dict[str, Any] = {
            "table": table,
            "kind": "tick_like",
            "ts_col": ts_col,
            "columns": sorted(cols),
            "selection": "full_table",
            "selection_where": "",
            "selection_note": None,
            "row_count": 0,
            "min_ts": None,
            "max_ts": None,
        }

        where_sql = ""
        if "symbol" in cols:
            cur.execute(f"SELECT DISTINCT symbol FROM {_q(table)} ORDER BY 1")
            symbols = [r[0] for r in cur.fetchall()]
            diag["symbol_values"] = symbols
            if "EURUSD" in symbols:
                where_sql = "symbol = 'EURUSD'"
                diag["selection"] = "symbol_filter"
                diag["selection_where"] = where_sql
            else:
                diagnostics.append(diag)
                continue

        elif "symbol_id" in cols:
            cur.execute(
                f"SELECT symbol_id, COUNT(*) c, MIN({_q(ts_col)}), MAX({_q(ts_col)}) "
                f"FROM {_q(table)} GROUP BY 1 ORDER BY c DESC"
            )
            rows = cur.fetchall()
            diag["symbol_id_breakdown"] = [
                {
                    "symbol_id": int(r[0]),
                    "rows": int(r[1]),
                    "min_ts": r[2],
                    "max_ts": r[3],
                }
                for r in rows
            ]
            if rows:
                selected_id = int(rows[0][0])
                where_sql = f"symbol_id = {selected_id}"
                diag["selection"] = "symbol_id_filter"
                diag["selection_where"] = where_sql
                diag["selection_note"] = (
                    "Selected dominant symbol_id by row count. "
                    "Assumed to be EURUSD due project context."
                )

        row_count, min_ts, max_ts = _table_min_max_count(cur, table, ts_col, where_sql)
        diag["row_count"] = row_count
        diag["min_ts"] = min_ts
        diag["max_ts"] = max_ts

        diagnostics.append(diag)

        if row_count <= 0 or min_ts is None or max_ts is None:
            continue

        start = pd.to_datetime(min_ts, utc=True, errors="coerce")
        end = pd.to_datetime(max_ts, utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue

        candidates.append(
            {
                "table": table,
                "ts_col": ts_col,
                "where_sql": where_sql,
                "selection": diag["selection"],
                "selection_note": diag.get("selection_note"),
                "row_count": row_count,
                "min_ts": to_iso_z(start),
                "max_ts": to_iso_z(end),
                "_start": start,
                "_end": end,
            }
        )

    if not candidates:
        return None, diagnostics

    candidates.sort(key=lambda c: (c["_start"], -c["row_count"], c["_end"]))
    best = candidates[0]
    best.pop("_start", None)
    best.pop("_end", None)
    return best, diagnostics


def _choose_ohlc_candidate(cur: sqlite3.Cursor, schema: dict[str, set[str]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for table, cols in schema.items():
        ts_col = _pick_ts_col(cols)
        if not ts_col:
            continue

        ohlc_like = (
            {"open", "high", "low", "close"}.issubset(cols)
            or {"mid_o", "mid_h", "mid_l", "mid_c"}.issubset(cols)
            or {"bid_o", "bid_h", "bid_l", "bid_c"}.issubset(cols)
        )
        if not ohlc_like:
            continue

        row_count, min_ts, max_ts = _table_min_max_count(cur, table, ts_col)
        diag = {
            "table": table,
            "kind": "ohlc_like",
            "ts_col": ts_col,
            "columns": sorted(cols),
            "row_count": row_count,
            "min_ts": min_ts,
            "max_ts": max_ts,
        }
        diagnostics.append(diag)

        if row_count <= 0 or min_ts is None or max_ts is None:
            continue

        start = pd.to_datetime(min_ts, utc=True, errors="coerce")
        end = pd.to_datetime(max_ts, utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue

        priority = 0 if "m1" in table.lower() else 1
        candidates.append(
            {
                "table": table,
                "ts_col": ts_col,
                "row_count": row_count,
                "min_ts": to_iso_z(start),
                "max_ts": to_iso_z(end),
                "priority": priority,
                "_start": start,
                "_end": end,
            }
        )

    if not candidates:
        return None, diagnostics

    candidates.sort(key=lambda c: (c["priority"], c["_start"], -c["row_count"]))
    best = candidates[0]
    best.pop("_start", None)
    best.pop("_end", None)
    return best, diagnostics


def _normalize_ticks(df: pd.DataFrame, ts_col: str, source: str) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    input_rows = int(len(out))
    out["ts_utc"] = _parse_ts_series(out[ts_col])
    dropped_unparsed_ts = int(out["ts_utc"].isna().sum())
    dropped_missing_bid_or_ask = int((out["bid"].isna() | out["ask"].isna()).sum())
    out = out.dropna(subset=["ts_utc", "bid", "ask"])

    dropped_non_positive = int(((out["bid"] <= 0) | (out["ask"] <= 0)).sum())
    dropped_bid_gt_ask = int((out["bid"] > out["ask"]).sum())
    out = out[(out["bid"] > 0) & (out["ask"] > 0) & (out["bid"] <= out["ask"])]

    if "mid" not in out.columns:
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
    else:
        out["mid"] = out["mid"].fillna((out["bid"] + out["ask"]) / 2.0)

    out = out[["ts_utc", "bid", "ask", "mid"]].sort_values("ts_utc")
    out = out.drop_duplicates(subset=["ts_utc"], keep="first")
    out["source"] = source
    normalized = out.reset_index(drop=True)
    cleaning = {
        "input_rows": input_rows,
        "output_rows": int(len(normalized)),
        "dropped_unparsed_ts": dropped_unparsed_ts,
        "dropped_missing_bid_or_ask": dropped_missing_bid_or_ask,
        "dropped_non_positive_bid_or_ask": dropped_non_positive,
        "dropped_bid_gt_ask": dropped_bid_gt_ask,
    }
    return normalized, cleaning


def _ticks_to_1m_ohlc(df_ticks: pd.DataFrame, source: str) -> pd.DataFrame:
    mid = df_ticks.set_index("ts_utc")["mid"].sort_index()
    ohlc = mid.resample("1min", label="left", closed="left").ohlc().dropna().reset_index()
    ohlc["volume"] = np.nan
    ohlc["source"] = source
    return ohlc[["ts_utc", "open", "high", "low", "close", "volume", "source"]]


def _normalize_ohlc(df: pd.DataFrame, ts_col: str, source: str) -> pd.DataFrame:
    out = df.copy()
    out["ts_utc"] = _parse_ts_series(out[ts_col])

    if {"open", "high", "low", "close"}.issubset(out.columns):
        pass
    elif {"mid_o", "mid_h", "mid_l", "mid_c"}.issubset(out.columns):
        out["open"] = out["mid_o"]
        out["high"] = out["mid_h"]
        out["low"] = out["mid_l"]
        out["close"] = out["mid_c"]
    elif {"bid_o", "bid_h", "bid_l", "bid_c"}.issubset(out.columns):
        out["open"] = out["bid_o"]
        out["high"] = out["bid_h"]
        out["low"] = out["bid_l"]
        out["close"] = out["bid_c"]
    else:
        raise ValueError("Unable to map OHLC columns from source table")

    out["volume"] = out["volume"] if "volume" in out.columns else np.nan
    out = out[["ts_utc", "open", "high", "low", "close", "volume"]]
    out = out.dropna(subset=["ts_utc", "open", "high", "low", "close"])
    out = out.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="first")
    out["source"] = source
    return out.reset_index(drop=True)


def _parse_ts_series(values: pd.Series) -> pd.Series:
    # SQLite tables may mix timestamp string formats within the same column
    # (e.g. naive ISO strings and offset-aware strings with microseconds).
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    bad = parsed.isna() & values.notna()
    if bad.any():
        try:
            reparsed = pd.to_datetime(values[bad], utc=True, errors="coerce", format="mixed")
        except TypeError:
            reparsed = values[bad].map(lambda v: pd.to_datetime(v, utc=True, errors="coerce"))
        parsed.loc[bad] = reparsed
    return parsed


def _save(df: pd.DataFrame, path: Path, write_csv: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if write_csv:
        df.to_csv(path.with_suffix(".csv"), index=False)


def export_eurusd(
    config: dict[str, Any],
    db_override: str | None = None,
    no_csv: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    c = config.get("eurusd", {})

    db_path = Path(db_override or c.get("sqlite_db_path", "research/data_cache/eurusd/raw/ict_bot.sqlite3"))
    if not db_path.exists():
        raise FileNotFoundError(f"EURUSD sqlite DB not found: {db_path}")

    raw_dir = Path(c.get("raw_dir", "research/data_cache/eurusd/raw"))
    normalized_dir = Path(c.get("normalized_dir", "research/data_cache/eurusd/normalized"))
    canonical_1m_path = Path(c.get("canonical_1m_path", normalized_dir / "eurusd_ohlc_1m.parquet"))
    ticks_path = Path(c.get("ticks_path", normalized_dir / "eurusd_ticks.parquet"))

    start_ts = _optional_ts(c.get("start_ts_utc"))
    end_ts = _optional_ts(c.get("end_ts_utc"))
    required_start = _optional_ts(c.get("required_start_ts_utc"))

    ensure_dir(raw_dir)
    ensure_dir(normalized_dir)

    db_copy = raw_dir / db_path.name
    if db_path.resolve() != db_copy.resolve():
        shutil.copy2(db_path, db_copy)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    schema = _load_schema(cur)

    tick_choice, tick_diagnostics = _choose_tick_candidate(cur, schema)
    ohlc_choice, ohlc_diagnostics = _choose_ohlc_candidate(cur, schema)

    summary: dict[str, Any] = {
        "db_source": str(db_path),
        "db_copy": str(db_copy),
        "table_diagnostics": {
            "tick_like": tick_diagnostics,
            "ohlc_like": ohlc_diagnostics,
        },
        "selected_tick_source": tick_choice,
        "selected_ohlc_source": ohlc_choice,
        "mode": None,
        "outputs": {},
    }

    if tick_choice:
        table = tick_choice["table"]
        ts_col = tick_choice["ts_col"]
        where_sql = tick_choice.get("where_sql", "")
        cols = schema[table]
        select_cols = [ts_col, "bid", "ask"] + (["mid"] if "mid" in cols else [])

        where_clause = f" WHERE {where_sql}" if where_sql else ""
        sql = (
            "SELECT "
            + ", ".join(_q(cn) for cn in select_cols)
            + f" FROM {_q(table)}{where_clause} ORDER BY {_q(ts_col)}"
        )
        ticks_raw = pd.read_sql_query(sql, conn)
        ticks, ticks_cleaning = _normalize_ticks(ticks_raw, ts_col, source=f"sqlite:{db_path.name}:{table}")

        if start_ts is not None:
            ticks = ticks[ticks["ts_utc"] >= start_ts]
        if end_ts is not None:
            ticks = ticks[ticks["ts_utc"] <= end_ts]
        ticks = ticks.reset_index(drop=True)

        bars_1m = _ticks_to_1m_ohlc(ticks, source=f"sqlite:{db_path.name}:{table}:mid->1m")

        _save(ticks, ticks_path, write_csv=not no_csv)
        _save(bars_1m, canonical_1m_path, write_csv=not no_csv)

        summary["mode"] = "ticks"
        summary["outputs"]["eurusd_ticks"] = {
            "path": str(ticks_path),
            "rows": int(len(ticks)),
            "start_ts_utc": to_iso_z(ticks["ts_utc"].iloc[0]) if not ticks.empty else None,
            "end_ts_utc": to_iso_z(ticks["ts_utc"].iloc[-1]) if not ticks.empty else None,
        }
        summary["outputs"]["eurusd_ohlc_1m"] = {
            "path": str(canonical_1m_path),
            "rows": int(len(bars_1m)),
            "start_ts_utc": to_iso_z(bars_1m["ts_utc"].iloc[0]) if not bars_1m.empty else None,
            "end_ts_utc": to_iso_z(bars_1m["ts_utc"].iloc[-1]) if not bars_1m.empty else None,
        }
        summary["tick_cleaning"] = ticks_cleaning

    elif ohlc_choice:
        table = ohlc_choice["table"]
        ts_col = ohlc_choice["ts_col"]
        cols = schema[table]

        select_pool = [
            ts_col,
            "open",
            "high",
            "low",
            "close",
            "mid_o",
            "mid_h",
            "mid_l",
            "mid_c",
            "bid_o",
            "bid_h",
            "bid_l",
            "bid_c",
            "volume",
        ]
        select_cols = [x for x in select_pool if x in cols]
        sql = (
            "SELECT "
            + ", ".join(_q(cn) for cn in select_cols)
            + f" FROM {_q(table)} ORDER BY {_q(ts_col)}"
        )
        raw = pd.read_sql_query(sql, conn)
        bars_1m = _normalize_ohlc(raw, ts_col, source=f"sqlite:{db_path.name}:{table}")

        if start_ts is not None:
            bars_1m = bars_1m[bars_1m["ts_utc"] >= start_ts]
        if end_ts is not None:
            bars_1m = bars_1m[bars_1m["ts_utc"] <= end_ts]
        bars_1m = bars_1m.reset_index(drop=True)

        _save(bars_1m, canonical_1m_path, write_csv=not no_csv)

        summary["mode"] = "ohlc"
        summary["outputs"]["eurusd_ohlc_1m"] = {
            "path": str(canonical_1m_path),
            "rows": int(len(bars_1m)),
            "start_ts_utc": to_iso_z(bars_1m["ts_utc"].iloc[0]) if not bars_1m.empty else None,
            "end_ts_utc": to_iso_z(bars_1m["ts_utc"].iloc[-1]) if not bars_1m.empty else None,
        }
    else:
        conn.close()
        raise RuntimeError("No tick-like or OHLC-like tables were found in the EURUSD SQLite DB")

    conn.close()

    validations: dict[str, Any] = {}
    if validate:
        if "eurusd_ticks" in summary["outputs"]:
            tick_report = validate_file(summary["outputs"]["eurusd_ticks"]["path"], kind="ticks")
            validations["eurusd_ticks"] = tick_report
            if not tick_report["ok"]:
                raise RuntimeError(f"EURUSD ticks validation failed: {tick_report['errors']}")

        ohlc_report = validate_file(summary["outputs"]["eurusd_ohlc_1m"]["path"], kind="ohlc", expected_seconds=60)
        validations["eurusd_ohlc_1m"] = ohlc_report
        if not ohlc_report["ok"]:
            raise RuntimeError(f"EURUSD 1m OHLC validation failed: {ohlc_report['errors']}")

    summary["validation"] = validations

    actual_start = parse_utc_ts(summary["outputs"]["eurusd_ohlc_1m"]["start_ts_utc"]) if summary["outputs"]["eurusd_ohlc_1m"]["start_ts_utc"] else None
    actual_end = parse_utc_ts(summary["outputs"]["eurusd_ohlc_1m"]["end_ts_utc"]) if summary["outputs"]["eurusd_ohlc_1m"]["end_ts_utc"] else None

    if required_start is not None and actual_start is not None and actual_start > required_start:
        summary["coverage_error"] = (
            f"EURUSD DB only contains data from {to_iso_z(actual_start)} to {to_iso_z(actual_end)}; "
            f"cannot produce requested 7-year coverage from {to_iso_z(required_start)} without additional source data."
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and export EURUSD data from local SQLite DB")
    parser.add_argument("--config", default="research/config_data.yml", help="Config YAML path")
    parser.add_argument("--db", help="Override EURUSD SQLite DB path")
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV outputs")
    parser.add_argument("--no-validate", action="store_true", help="Skip post-export validation")
    parser.add_argument(
        "--summary-out",
        default="research/artefacts/logs/eurusd_export_summary.json",
        help="Summary JSON output path",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    summary = export_eurusd(
        config,
        db_override=args.db,
        no_csv=args.no_csv,
        validate=not args.no_validate,
    )
    print(json.dumps(summary, indent=2))
    dump_json(args.summary_out, summary)
    print(f"Wrote EURUSD export summary to {args.summary_out}")


if __name__ == "__main__":
    main()
