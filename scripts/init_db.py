"""
Consolidated database schema for ict-bot.
Creates all tables needed by every component.

Run: python -m scripts.init_db
"""

from shared.db import connect

SCHEMA = """
PRAGMA journal_mode=WAL;

-- CORE TABLES

CREATE TABLE IF NOT EXISTS heartbeat (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc  TEXT NOT NULL,
    status  TEXT NOT NULL,
    note    TEXT
);

CREATE TABLE IF NOT EXISTS event_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc  TEXT NOT NULL,
    level   TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quotes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT    NOT NULL,
    env         TEXT    NOT NULL,
    account_id  TEXT    NOT NULL,
    symbol_id   INTEGER NOT NULL,
    bid         REAL,
    ask         REAL
);

CREATE INDEX IF NOT EXISTS idx_quotes_lookup
    ON quotes (env, account_id, symbol_id, ts_utc);

-- CLEANED TICKS + SESSIONS (BACKTEST PIPELINE)

CREATE TABLE IF NOT EXISTS ticks_clean (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT    NOT NULL,
    bid          REAL,
    ask          REAL,
    mid          REAL,
    spread_pips  REAL,
    session      TEXT
);

CREATE INDEX IF NOT EXISTS idx_ticks_clean_ts
    ON ticks_clean (ts_utc);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT    NOT NULL,
    session     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_ts
    ON sessions (ts_utc);

CREATE TABLE IF NOT EXISTS candles_m1 (
    ts_utc          TEXT    NOT NULL,
    bid_o           REAL,
    bid_h           REAL,
    bid_l           REAL,
    bid_c           REAL,
    ask_o           REAL,
    ask_h           REAL,
    ask_l           REAL,
    ask_c           REAL,
    mid_o           REAL,
    mid_h           REAL,
    mid_l           REAL,
    mid_c           REAL,
    spread_o_pips   REAL,
    spread_c_pips   REAL,
    spread_mean_pips REAL,
    spread_min_pips  REAL,
    spread_max_pips  REAL,
    session         TEXT
);

CREATE INDEX IF NOT EXISTS idx_candles_m1_ts
    ON candles_m1 (ts_utc);

CREATE TABLE IF NOT EXISTS candles_m5 (
    ts_utc          TEXT    NOT NULL,
    bid_o           REAL,
    bid_h           REAL,
    bid_l           REAL,
    bid_c           REAL,
    ask_o           REAL,
    ask_h           REAL,
    ask_l           REAL,
    ask_c           REAL,
    mid_o           REAL,
    mid_h           REAL,
    mid_l           REAL,
    mid_c           REAL,
    spread_o_pips   REAL,
    spread_c_pips   REAL,
    spread_mean_pips REAL,
    spread_min_pips  REAL,
    spread_max_pips  REAL,
    session         TEXT
);

CREATE INDEX IF NOT EXISTS idx_candles_m5_ts
    ON candles_m5 (ts_utc);

CREATE TABLE IF NOT EXISTS candles_m15 (
    ts_utc          TEXT    NOT NULL,
    bid_o           REAL,
    bid_h           REAL,
    bid_l           REAL,
    bid_c           REAL,
    ask_o           REAL,
    ask_h           REAL,
    ask_l           REAL,
    ask_c           REAL,
    mid_o           REAL,
    mid_h           REAL,
    mid_l           REAL,
    mid_c           REAL,
    spread_o_pips   REAL,
    spread_c_pips   REAL,
    spread_mean_pips REAL,
    spread_min_pips  REAL,
    spread_max_pips  REAL,
    session         TEXT
);

CREATE INDEX IF NOT EXISTS idx_candles_m15_ts
    ON candles_m15 (ts_utc);

CREATE TABLE IF NOT EXISTS candles_h1 (
    ts_utc          TEXT    NOT NULL,
    bid_o           REAL,
    bid_h           REAL,
    bid_l           REAL,
    bid_c           REAL,
    ask_o           REAL,
    ask_h           REAL,
    ask_l           REAL,
    ask_c           REAL,
    mid_o           REAL,
    mid_h           REAL,
    mid_l           REAL,
    mid_c           REAL,
    spread_o_pips   REAL,
    spread_c_pips   REAL,
    spread_mean_pips REAL,
    spread_min_pips  REAL,
    spread_max_pips  REAL,
    session         TEXT
);

CREATE INDEX IF NOT EXISTS idx_candles_h1_ts
    ON candles_h1 (ts_utc);

-- SIGNAL & PAPER TRADING

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT    NOT NULL,
    env         TEXT    NOT NULL,
    account_id  TEXT    NOT NULL,
    symbol_id   INTEGER NOT NULL,
    signal      TEXT    NOT NULL,
    fast_ma     REAL,
    slow_ma     REAL,
    mid         REAL,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS paper_positions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT    NOT NULL,
    env         TEXT    NOT NULL,
    account_id  TEXT    NOT NULL,
    symbol_id   INTEGER NOT NULL,
    side        TEXT    NOT NULL,
    size_units  REAL    NOT NULL,
    entry_mid   REAL    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'OPEN'
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT    NOT NULL,
    env         TEXT    NOT NULL,
    account_id  TEXT    NOT NULL,
    symbol_id   INTEGER NOT NULL,
    action      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    size_units  REAL    NOT NULL,
    mid         REAL    NOT NULL,
    note        TEXT
);

-- NEWS / ECONOMIC CALENDAR

CREATE TABLE IF NOT EXISTS news_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name      TEXT    NOT NULL,
    currency        TEXT    NOT NULL,
    impact          TEXT    NOT NULL,
    datetime_utc    TEXT    NOT NULL,
    actual          TEXT,
    forecast        TEXT,
    previous        TEXT,
    fetched_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_news_upcoming
    ON news_events (datetime_utc, currency, impact);

-- BACKTEST ENGINE

CREATE TABLE IF NOT EXISTS backtest_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc          TEXT    NOT NULL,
    strategy_name   TEXT    NOT NULL,
    params_json     TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    date_from       TEXT    NOT NULL,
    date_to         TEXT    NOT NULL,
    cost_model_json TEXT,
    note            TEXT
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES backtest_runs(id),
    metric_name     TEXT    NOT NULL,
    metric_value    REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bt_results_run
    ON backtest_results (run_id);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES backtest_runs(id),
    ts_utc          TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    size_units      REAL    NOT NULL,
    price           REAL    NOT NULL,
    spread_cost     REAL    DEFAULT 0,
    slippage_cost   REAL    DEFAULT 0,
    commission_cost REAL    DEFAULT 0,
    swap_cost       REAL    DEFAULT 0,
    pnl             REAL,
    note            TEXT
);

CREATE INDEX IF NOT EXISTS idx_bt_trades_run
    ON backtest_trades (run_id);

CREATE TABLE IF NOT EXISTS backtest_equity (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES backtest_runs(id),
    ts_utc          TEXT    NOT NULL,
    equity          REAL    NOT NULL,
    drawdown        REAL    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_bt_equity_run
    ON backtest_equity (run_id);

-- REGIME DETECTION

CREATE TABLE IF NOT EXISTS regime_labels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    regime      TEXT    NOT NULL,
    adx         REAL,
    atr         REAL
);

CREATE INDEX IF NOT EXISTS idx_regime_lookup
    ON regime_labels (symbol, ts_utc);
"""


def main():
    with connect() as conn:
        conn.executescript(SCHEMA)
    print("OK: All tables created/verified")


if __name__ == "__main__":
    main()
