from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from .common import dump_json

TS_CANDIDATES = ("ts_utc", "datetime_utc", "timestamp", "datetime", "time", "ts")


def _q(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def _pick_ts_col(columns: set[str]) -> str | None:
    for col in TS_CANDIDATES:
        if col in columns:
            return col
    return None


def inspect_sqlite(db_path: str | Path) -> dict[str, Any]:
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"DB not found: {p}")

    conn = sqlite3.connect(str(p))
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]

    out_tables: list[dict[str, Any]] = []
    likely_tick_tables: list[str] = []
    likely_ohlc_tables: list[str] = []

    for table in tables:
        cur.execute(f"PRAGMA table_info({_q(table)})")
        col_rows = cur.fetchall()
        col_names = [r[1] for r in col_rows]
        col_set = set(col_names)
        ts_col = _pick_ts_col(col_set)

        cur.execute(f"SELECT COUNT(*) FROM {_q(table)}")
        row_count = int(cur.fetchone()[0])

        min_ts = None
        max_ts = None
        if ts_col:
            cur.execute(
                f"SELECT MIN({_q(ts_col)}), MAX({_q(ts_col)}) FROM {_q(table)}"
            )
            min_ts, max_ts = cur.fetchone()

        tick_like = ts_col is not None and {"bid", "ask"}.issubset(col_set)
        ohlc_like = ts_col is not None and (
            {"open", "high", "low", "close"}.issubset(col_set)
            or {"mid_o", "mid_h", "mid_l", "mid_c"}.issubset(col_set)
            or {"bid_o", "bid_h", "bid_l", "bid_c"}.issubset(col_set)
        )
        if tick_like:
            likely_tick_tables.append(table)
        if ohlc_like:
            likely_ohlc_tables.append(table)

        entry: dict[str, Any] = {
            "table": table,
            "row_count": row_count,
            "columns": [{"name": r[1], "type": r[2]} for r in col_rows],
            "ts_col": ts_col,
            "min_ts": min_ts,
            "max_ts": max_ts,
            "tick_like": tick_like,
            "ohlc_like": ohlc_like,
        }

        if "symbol" in col_set:
            cur.execute(f"SELECT DISTINCT symbol FROM {_q(table)} ORDER BY 1 LIMIT 20")
            entry["symbol_values"] = [r[0] for r in cur.fetchall()]

        if "symbol_id" in col_set:
            cur.execute(
                f"SELECT symbol_id, COUNT(*) c FROM {_q(table)} GROUP BY 1 ORDER BY c DESC LIMIT 20"
            )
            entry["symbol_id_counts"] = [{"symbol_id": r[0], "rows": int(r[1])} for r in cur.fetchall()]

        out_tables.append(entry)

    eurusd_evidence: dict[str, Any] = {}
    if "backtest_runs" in tables:
        cur.execute("SELECT DISTINCT symbol FROM backtest_runs ORDER BY symbol")
        eurusd_evidence["backtest_runs_symbols"] = [r[0] for r in cur.fetchall()]

    if "quotes" in tables:
        cur.execute("SELECT symbol_id, COUNT(*) c FROM quotes GROUP BY 1 ORDER BY c DESC")
        eurusd_evidence["quotes_symbol_id_counts"] = [{"symbol_id": r[0], "rows": int(r[1])} for r in cur.fetchall()]

    conn.close()

    return {
        "db_path": str(p.resolve()),
        "db_size_bytes": p.stat().st_size,
        "table_count": len(tables),
        "likely_tick_tables": likely_tick_tables,
        "likely_ohlc_tables": likely_ohlc_tables,
        "eurusd_evidence": eurusd_evidence,
        "tables": out_tables,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SQLite schema and detect market-data tables")
    parser.add_argument("--db", required=True, help="Path to SQLite DB")
    parser.add_argument(
        "--out",
        default="research/artefacts/logs/sqlite_inspection.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    summary = inspect_sqlite(args.db)
    print(json.dumps(summary, indent=2))
    dump_json(args.out, summary)
    print(f"Wrote inspection report to {args.out}")


if __name__ == "__main__":
    main()
