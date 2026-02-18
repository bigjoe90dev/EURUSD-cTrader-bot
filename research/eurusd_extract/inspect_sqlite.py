from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


TS_CANDIDATES = ("ts_utc", "datetime_utc", "timestamp", "datetime", "time", "ts")


def _quote_ident(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def _table_names(cur: sqlite3.Cursor) -> list[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cur.fetchall()]


def _table_columns(cur: sqlite3.Cursor, table: str) -> list[dict[str, Any]]:
    cur.execute(f"PRAGMA table_info({_quote_ident(table)})")
    rows = cur.fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": bool(row[5]),
        }
        for row in rows
    ]


def _pick_ts_col(column_names: set[str]) -> str | None:
    for candidate in TS_CANDIDATES:
        if candidate in column_names:
            return candidate
    return None


def inspect_db(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    tables = _table_names(cur)
    table_summaries: list[dict[str, Any]] = []
    likely_tick_tables: list[str] = []
    likely_ohlc_tables: list[str] = []

    for table in tables:
        cols = _table_columns(cur, table)
        col_names = {c["name"] for c in cols}
        ts_col = _pick_ts_col(col_names)

        cur.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}")
        row_count = int(cur.fetchone()[0])

        min_ts = None
        max_ts = None
        if ts_col:
            cur.execute(
                f"SELECT MIN({_quote_ident(ts_col)}), MAX({_quote_ident(ts_col)}) "
                f"FROM {_quote_ident(table)}"
            )
            min_ts, max_ts = cur.fetchone()

        has_tick_like = ts_col is not None and {"bid", "ask"}.issubset(col_names)
        has_mid_ohlc = {"mid_o", "mid_h", "mid_l", "mid_c"}.issubset(col_names)
        has_plain_ohlc = {"open", "high", "low", "close"}.issubset(col_names)
        has_bid_ohlc = {"bid_o", "bid_h", "bid_l", "bid_c"}.issubset(col_names)
        has_ask_ohlc = {"ask_o", "ask_h", "ask_l", "ask_c"}.issubset(col_names)
        has_ohlc_like = ts_col is not None and (
            has_mid_ohlc or has_plain_ohlc or has_bid_ohlc or has_ask_ohlc
        )

        if has_tick_like:
            likely_tick_tables.append(table)
        if has_ohlc_like:
            likely_ohlc_tables.append(table)

        table_summary: dict[str, Any] = {
            "table": table,
            "row_count": row_count,
            "columns": cols,
            "ts_col": ts_col,
            "min_ts": min_ts,
            "max_ts": max_ts,
            "tick_like": has_tick_like,
            "ohlc_like": has_ohlc_like,
        }

        if "symbol" in col_names:
            cur.execute(
                f"SELECT DISTINCT {_quote_ident('symbol')} FROM {_quote_ident(table)} "
                "ORDER BY 1 LIMIT 20"
            )
            table_summary["symbol_values"] = [r[0] for r in cur.fetchall()]

        if "symbol_id" in col_names:
            cur.execute(
                f"SELECT {_quote_ident('symbol_id')}, COUNT(*) AS c "
                f"FROM {_quote_ident(table)} GROUP BY 1 ORDER BY c DESC LIMIT 20"
            )
            table_summary["symbol_id_counts"] = [
                {"symbol_id": r[0], "rows": int(r[1])} for r in cur.fetchall()
            ]

        table_summaries.append(table_summary)

    eurusd_evidence: dict[str, Any] = {}
    if "backtest_runs" in tables:
        cur.execute("SELECT DISTINCT symbol FROM backtest_runs ORDER BY symbol")
        eurusd_evidence["backtest_runs_symbols"] = [r[0] for r in cur.fetchall()]

    if "quotes" in tables:
        cur.execute("SELECT symbol_id, COUNT(*) FROM quotes GROUP BY symbol_id ORDER BY 2 DESC")
        eurusd_evidence["quotes_symbol_id_counts"] = [
            {"symbol_id": r[0], "rows": int(r[1])} for r in cur.fetchall()
        ]

    conn.close()

    return {
        "db_path": str(db_path),
        "db_size_bytes": db_path.stat().st_size,
        "table_count": len(tables),
        "likely_tick_tables": likely_tick_tables,
        "likely_ohlc_tables": likely_ohlc_tables,
        "eurusd_evidence": eurusd_evidence,
        "tables": table_summaries,
    }


def _to_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# SQLite Inspection Summary")
    lines.append("")
    lines.append(f"- DB: `{summary['db_path']}`")
    lines.append(f"- Size (bytes): `{summary['db_size_bytes']}`")
    lines.append(f"- Tables: `{summary['table_count']}`")
    lines.append(f"- Likely tick tables: `{', '.join(summary['likely_tick_tables']) or 'none'}`")
    lines.append(f"- Likely OHLC tables: `{', '.join(summary['likely_ohlc_tables']) or 'none'}`")
    lines.append("")
    lines.append("## EURUSD Evidence")
    lines.append("```json")
    lines.append(json.dumps(summary["eurusd_evidence"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Tables")
    for table in summary["tables"]:
        col_short = ", ".join(f"{c['name']}:{c['type']}" for c in table["columns"])
        lines.append(f"- `{table['table']}` rows=`{table['row_count']}`")
        lines.append(f"  ts_col=`{table['ts_col']}` range=`{table['min_ts']}` -> `{table['max_ts']}`")
        lines.append(f"  columns={col_short}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SQLite DB tables for EURUSD data")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--json-out", help="Optional path to write JSON summary")
    parser.add_argument("--md-out", help="Optional path to write markdown summary")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    summary = inspect_db(db_path)

    print(json.dumps(summary, indent=2))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary to {out}")

    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_to_markdown(summary), encoding="utf-8")
        print(f"Wrote markdown summary to {out}")


if __name__ == "__main__":
    main()
