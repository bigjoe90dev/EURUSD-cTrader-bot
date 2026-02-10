"""
Builds cleaned tick data and candle tables for backtesting.
"""
import sqlite3
from typing import Dict

import pandas as pd

from backtest.config import BacktestSettings
from backtest.session import classify_session, parse_session_rules
from shared.db import get_db_path


def load_ticks(db_path: str, symbol_id: int = 1) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "select ts_utc, bid, ask from quotes where symbol_id=? order by ts_utc",
        conn,
        params=(symbol_id,),
    )
    conn.close()
    return df


def clean_ticks(df: pd.DataFrame,
                cfg: BacktestSettings,
                pip_size: float = 0.0001) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], errors="coerce", utc=True)
    out = out.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)

    # Deduplicate timestamps (keep first)
    out = out.drop_duplicates(subset=["ts_utc"], keep="first").reset_index(drop=True)

    # Remove invalid spreads
    spread = out["ask"] - out["bid"]
    out = out[spread > 0].reset_index(drop=True)

    # Recompute mid
    out["mid"] = (out["bid"] + out["ask"]) / 2.0

    # Hard sanity filter for impossible prices (drop corrupted ticks)
    out = out[(out["mid"] >= cfg.price_min) & (out["mid"] <= cfg.price_max)].reset_index(drop=True)

    # Add spread in pips
    out["spread_pips"] = (out["ask"] - out["bid"]) / pip_size

    # Drop only extreme/broken spreads (keep real spikes)
    out = out[(out["spread_pips"] >= cfg.min_spread_pips_abs) &
              (out["spread_pips"] <= cfg.max_spread_pips_abs)].reset_index(drop=True)

    # Session tagging
    rules = parse_session_rules(cfg.session_rules)
    out["session"] = out["ts_utc"].apply(lambda ts: classify_session(ts.to_pydatetime(), rules))
    return out


def build_candles(df_ticks: pd.DataFrame,
                  timeframe: str,
                  pip_size: float = 0.0001) -> pd.DataFrame:
    if df_ticks.empty:
        return df_ticks.copy()

    df = df_ticks.copy()
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    if "spread_pips" not in df.columns:
        df["spread_pips"] = (df["ask"] - df["bid"]) / pip_size
    df = df.set_index("ts_utc")

    agg = {
        "bid": ["first", "max", "min", "last"],
        "ask": ["first", "max", "min", "last"],
        "mid": ["first", "max", "min", "last"],
        "spread_pips": ["first", "last", "mean", "min", "max"],
        "session": "first",
    }
    res = df.resample(timeframe, label="left", closed="left").agg(agg)
    res.columns = [
        "bid_o", "bid_h", "bid_l", "bid_c",
        "ask_o", "ask_h", "ask_l", "ask_c",
        "mid_o", "mid_h", "mid_l", "mid_c",
        "spread_o_pips", "spread_c_pips", "spread_mean_pips", "spread_min_pips", "spread_max_pips",
        "session",
    ]
    res = res.dropna(subset=["bid_o", "ask_o", "bid_c", "ask_c"]).reset_index()
    if res.empty:
        return res

    # Ensure spread stats are sane
    res["spread_o_pips"] = res["spread_o_pips"].fillna((res["ask_o"] - res["bid_o"]) / pip_size)
    res["spread_c_pips"] = res["spread_c_pips"].fillna((res["ask_c"] - res["bid_c"]) / pip_size)
    res["spread_mean_pips"] = res["spread_mean_pips"].fillna(res["spread_c_pips"])
    res["spread_min_pips"] = res["spread_min_pips"].fillna(res["spread_c_pips"])
    res["spread_max_pips"] = res["spread_max_pips"].fillna(res["spread_c_pips"])

    return res


def store_table(conn: sqlite3.Connection, table: str, df: pd.DataFrame):
    if df.empty:
        return
    ensure_columns(conn, table, df)
    df.to_sql(table, conn, if_exists="append", index=False)


def truncate_table(conn: sqlite3.Connection, table: str):
    conn.execute(f"DELETE FROM {table}")


def ensure_columns(conn: sqlite3.Connection, table: str, df: pd.DataFrame):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    row = cur.fetchone()
    if row is None:
        return
    existing = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    existing_set = set(existing)
    for col in df.columns:
        if col in existing_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "REAL"
        else:
            col_type = "TEXT"
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def build_backtest_tables(db_path: str = None,
                          symbol_id: int = 1,
                          cfg: BacktestSettings = None,
                          pip_size: float = 0.0001,
                          force: bool = True) -> Dict[str, int]:
    cfg = cfg or BacktestSettings()
    db_path = db_path or str(get_db_path())
    conn = sqlite3.connect(db_path)

    ticks = load_ticks(db_path, symbol_id=symbol_id)
    ticks_clean = clean_ticks(ticks, cfg, pip_size=pip_size)
    if ticks_clean.empty:
        conn.close()
        raise RuntimeError("No cleaned ticks available; aborting backtest table build.")

    candles_m1 = build_candles(ticks_clean, "1min", pip_size=pip_size)
    candles_m5 = build_candles(ticks_clean, "5min", pip_size=pip_size)
    candles_m15 = build_candles(ticks_clean, "15min", pip_size=pip_size)
    candles_h1 = build_candles(ticks_clean, "1H", pip_size=pip_size)

    if any(df.empty for df in [candles_m1, candles_m5, candles_m15, candles_h1]):
        conn.close()
        raise RuntimeError("One or more candle tables are empty; aborting to avoid truncating existing data.")

    # Persist ticks_clean + sessions
    if force:
        truncate_table(conn, "ticks_clean")
        truncate_table(conn, "sessions")
        truncate_table(conn, "candles_m1")
        truncate_table(conn, "candles_m5")
        truncate_table(conn, "candles_m15")
        truncate_table(conn, "candles_h1")

    store_table(conn, "ticks_clean", ticks_clean)
    store_table(conn, "sessions", ticks_clean[["ts_utc", "session"]])
    store_table(conn, "candles_m1", candles_m1)
    store_table(conn, "candles_m5", candles_m5)
    store_table(conn, "candles_m15", candles_m15)
    store_table(conn, "candles_h1", candles_h1)

    conn.commit()
    conn.close()

    return {
        "ticks_clean": len(ticks_clean),
        "candles_m1": len(candles_m1),
        "candles_m5": len(candles_m5),
        "candles_m15": len(candles_m15),
        "candles_h1": len(candles_h1),
    }
