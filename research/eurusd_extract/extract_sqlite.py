from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TS_CANDIDATES = ("ts_utc", "datetime_utc", "timestamp", "datetime", "time", "ts")
BAR_RULES = {
    "1m": "1min",
    "5m": "5min",
    "1h": "1h",
}


def _quote_ident(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def _load_table_columns(cur: sqlite3.Cursor) -> dict[str, set[str]]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    result: dict[str, set[str]] = {}
    for table in tables:
        cur.execute(f"PRAGMA table_info({_quote_ident(table)})")
        cols = {r[1] for r in cur.fetchall()}
        result[table] = cols
    return result


def _pick_ts_col(columns: set[str]) -> str | None:
    for candidate in TS_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _choose_tick_table(schema: dict[str, set[str]]) -> tuple[str, str] | None:
    preferred = ("ticks_clean", "quotes")
    for table in preferred:
        cols = schema.get(table)
        if not cols:
            continue
        ts_col = _pick_ts_col(cols)
        if ts_col and {"bid", "ask"}.issubset(cols):
            return table, ts_col
    for table, cols in schema.items():
        ts_col = _pick_ts_col(cols)
        if ts_col and {"bid", "ask"}.issubset(cols):
            return table, ts_col
    return None


def _choose_ohlc_table(schema: dict[str, set[str]]) -> tuple[str, str] | None:
    preferred = ("candles_m1", "ohlc_m1", "bars_m1")
    for table in preferred:
        cols = schema.get(table)
        if not cols:
            continue
        ts_col = _pick_ts_col(cols)
        if not ts_col:
            continue
        if {"open", "high", "low", "close"}.issubset(cols) or {
            "mid_o",
            "mid_h",
            "mid_l",
            "mid_c",
        }.issubset(cols):
            return table, ts_col
    for table, cols in schema.items():
        ts_col = _pick_ts_col(cols)
        if not ts_col:
            continue
        if {"open", "high", "low", "close"}.issubset(cols) or {
            "mid_o",
            "mid_h",
            "mid_l",
            "mid_c",
        }.issubset(cols):
            return table, ts_col
    return None


def _normalize_ticks(df: pd.DataFrame, ts_col: str, source: str) -> pd.DataFrame:
    out = df.copy()
    out["ts_utc"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out = out.dropna(subset=["ts_utc", "bid", "ask"])

    if "mid" not in out.columns:
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
    else:
        out["mid"] = out["mid"].fillna((out["bid"] + out["ask"]) / 2.0)

    out = out[["ts_utc", "bid", "ask", "mid"]]
    out = out.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="first")
    out["source"] = source
    return out.reset_index(drop=True)


def _resample_ticks_to_ohlc(df_ticks: pd.DataFrame, rule: str, source: str) -> pd.DataFrame:
    series = df_ticks.set_index("ts_utc")["mid"].sort_index()
    ohlc = series.resample(rule, label="left", closed="left").ohlc()
    ohlc = ohlc.dropna()
    out = ohlc.reset_index()
    out["volume"] = np.nan
    out["source"] = source
    return out[["ts_utc", "open", "high", "low", "close", "volume", "source"]]


def _normalize_ohlc_from_table(df: pd.DataFrame, ts_col: str, source: str) -> pd.DataFrame:
    out = df.copy()
    out["ts_utc"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")

    if {"open", "high", "low", "close"}.issubset(out.columns):
        out["open"] = out["open"]
        out["high"] = out["high"]
        out["low"] = out["low"]
        out["close"] = out["close"]
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

    volume_col = "volume" if "volume" in out.columns else None
    if volume_col:
        out["volume"] = out[volume_col]
    else:
        out["volume"] = np.nan

    out = out[["ts_utc", "open", "high", "low", "close", "volume"]]
    out = out.dropna(subset=["ts_utc", "open", "high", "low", "close"])
    out = out.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="first")
    out["source"] = source
    return out.reset_index(drop=True)


def _resample_ohlc(df_ohlc: pd.DataFrame, rule: str, source: str) -> pd.DataFrame:
    x = df_ohlc.set_index("ts_utc").sort_index()
    agg = x.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    out = agg.reset_index()
    out["source"] = source
    return out[["ts_utc", "open", "high", "low", "close", "volume", "source"]]


def _save_dataset(df: pd.DataFrame, parquet_path: Path, write_csv: bool = True) -> dict[str, Any]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    csv_path = parquet_path.with_suffix(".csv")
    if write_csv:
        df.to_csv(csv_path, index=False)

    return {
        "path_parquet": str(parquet_path),
        "path_csv": str(csv_path) if write_csv else None,
        "rows": int(len(df)),
        "start_ts": str(df["ts_utc"].iloc[0]) if not df.empty else None,
        "end_ts": str(df["ts_utc"].iloc[-1]) if not df.empty else None,
    }


def extract(
    db_path: Path,
    raw_dir: Path,
    out_dir: Path,
    write_csv: bool = True,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_db_path = raw_dir / db_path.name
    shutil.copy2(db_path, copied_db_path)

    conn = sqlite3.connect(str(db_path))
    schema = _load_table_columns(conn.cursor())
    cur = conn.cursor()

    tick_choice = _choose_tick_table(schema)
    ohlc_choice = _choose_ohlc_table(schema)

    outputs: dict[str, Any] = {
        "db_source": str(db_path),
        "db_copy": str(copied_db_path),
        "tick_table": tick_choice[0] if tick_choice else None,
        "ohlc_table": ohlc_choice[0] if ohlc_choice else None,
        "datasets": {},
    }

    if tick_choice:
        tick_table, tick_ts_col = tick_choice
        tick_cols = schema[tick_table]
        select_cols = [tick_ts_col, "bid", "ask"]
        if "mid" in tick_cols:
            select_cols.append("mid")
        query = (
            "SELECT "
            + ", ".join(_quote_ident(c) for c in select_cols)
            + f" FROM {_quote_ident(tick_table)} ORDER BY {_quote_ident(tick_ts_col)}"
        )
        df_ticks_raw = pd.read_sql_query(query, conn)
        tick_source = f"{db_path.name}:{tick_table}"
        df_ticks = _normalize_ticks(df_ticks_raw, tick_ts_col, tick_source)
        outputs["datasets"]["eurusd_ticks"] = _save_dataset(
            df_ticks, out_dir / "eurusd_ticks.parquet", write_csv=write_csv
        )

        for label, rule in BAR_RULES.items():
            df_bar = _resample_ticks_to_ohlc(
                df_ticks, rule=rule, source=f"{tick_source}|mid_resample_{label}"
            )
            outputs["datasets"][f"eurusd_ohlc_{label}"] = _save_dataset(
                df_bar, out_dir / f"eurusd_ohlc_{label}.parquet", write_csv=write_csv
            )

    elif ohlc_choice:
        ohlc_table, ohlc_ts_col = ohlc_choice
        cols = schema[ohlc_table]
        select_candidates = [
            ohlc_ts_col,
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
        select_cols = [c for c in select_candidates if c in cols]
        query = (
            "SELECT "
            + ", ".join(_quote_ident(c) for c in select_cols)
            + f" FROM {_quote_ident(ohlc_table)} ORDER BY {_quote_ident(ohlc_ts_col)}"
        )
        df_ohlc_raw = pd.read_sql_query(query, conn)
        ohlc_source = f"{db_path.name}:{ohlc_table}"
        df_ohlc_1m = _normalize_ohlc_from_table(df_ohlc_raw, ohlc_ts_col, ohlc_source)
        outputs["datasets"]["eurusd_ohlc_1m"] = _save_dataset(
            df_ohlc_1m, out_dir / "eurusd_ohlc_1m.parquet", write_csv=write_csv
        )
        for label, rule in BAR_RULES.items():
            if label == "1m":
                continue
            df_bar = _resample_ohlc(
                df_ohlc_1m, rule=rule, source=f"{ohlc_source}|resample_{label}"
            )
            outputs["datasets"][f"eurusd_ohlc_{label}"] = _save_dataset(
                df_bar, out_dir / f"eurusd_ohlc_{label}.parquet", write_csv=write_csv
            )
    else:
        conn.close()
        raise RuntimeError("No tick-like or OHLC-like table was found in SQLite DB")

    conn.close()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EURUSD data from SQLite to normalized files")
    parser.add_argument("--db", required=True, help="Path to SQLite DB file")
    parser.add_argument(
        "--raw-dir",
        default="research/data_cache/eurusd/raw",
        help="Directory to copy original DB and raw artifacts",
    )
    parser.add_argument(
        "--out-dir",
        default="research/data_cache/eurusd/normalized",
        help="Directory for normalized outputs",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write CSV copies next to parquet files",
    )
    parser.add_argument(
        "--summary-out",
        default="research/artefacts/logs/extract_summary.json",
        help="Path to write extraction summary JSON",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    summary = extract(
        db_path=db_path,
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        write_csv=not args.no_csv,
    )

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    main()
