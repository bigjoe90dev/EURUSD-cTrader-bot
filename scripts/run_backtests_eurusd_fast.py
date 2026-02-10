"""
Fast backtest runner for EURUSD strategies.

Purpose: prove the backtests run and produce results without waiting on
walk-forward/Monte Carlo/overfit/regime analysis or DB writes.
Outputs a CSV to /tmp for quick inspection.
"""
import csv
import os
import sqlite3

import pandas as pd

from backtest.config import load_backtest_settings
from backtest.data_pipeline import build_backtest_tables
from backtest.engine import BacktestEngine
from backtest.costs import CostConfig
from backtest.news_pause import build_news_pause_mask
from shared.config import load_settings
from backtest.eurusd_strategies import (
    LondonOpenBreakoutStrategy,
    AsianRangeFadeStrategy,
    MACrossADXStrategy,
    BollingerBounceStrategy,
    RSIDivergenceStrategy,
)


def load_candles(db_path: str, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"select * from {table} order by ts_utc", conn)
    conn.close()
    return df


def table_for_timeframe(tf: str) -> str:
    tf = tf.lower().strip()
    if tf == "m1":
        return "candles_m1"
    if tf == "m5":
        return "candles_m5"
    if tf == "m15":
        return "candles_m15"
    if tf == "h1":
        return "candles_h1"
    return "candles_m5"


def main():
    settings = load_settings()
    bt_cfg = load_backtest_settings()
    db_path = str(settings.db_path)

    # Only build candles if missing
    print("Checking candles...")
    conn = sqlite3.connect(db_path)
    try:
        c = conn.execute("select count(*) from candles_m5").fetchone()[0]
    except Exception:
        c = 0
    conn.close()

    if c == 0:
        print("No candles found. Building now...")
        build_backtest_tables(db_path=db_path, symbol_id=settings.ctrader_symbol_id or 1, cfg=bt_cfg)

    strategies = [
        LondonOpenBreakoutStrategy(),
        AsianRangeFadeStrategy(),
        MACrossADXStrategy(),
        BollingerBounceStrategy(),
        RSIDivergenceStrategy(),
    ]

    cfg = CostConfig()
    out_path = "/tmp/eurusd_fast_backtests.csv"
    print(f"Writing results to {out_path}")

    with open(out_path, "w", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "timeframe", "cagr", "total_return_pct",
            "sharpe", "profit_factor", "max_drawdown", "trades"
        ])
        f.flush()
        os.fsync(f.fileno())

        for strat in strategies:
            tf_table = table_for_timeframe(strat.timeframe())
            df = load_candles(db_path, tf_table)
            if df.empty:
                print(f"SKIP {strat.name()} - no candles in {tf_table}")
                continue

            print(f"Running {strat.name()} on {tf_table}...")
            engine = BacktestEngine(
                strategy=strat,
                cost_config=cfg,
                initial_balance=10_000.0,
                size_lots=0.1,
                symbol="EURUSD",
                signal_timing=bt_cfg.signal_timing,
                spread_model=bt_cfg.spread_model,
                slippage_model=bt_cfg.slippage_model,
                max_daily_loss_pct=bt_cfg.max_daily_loss_pct,
                max_trades_per_day=bt_cfg.max_trades_per_day,
            )
            pause_mask = build_news_pause_mask(df["ts_utc"])
            result = engine.run(df, pause_mask=pause_mask)
            m = result.metrics
            writer.writerow([
                strat.name(), tf_table,
                f"{m.cagr:.6f}", f"{m.total_return_pct:.6f}",
                f"{m.sharpe_ratio:.3f}", f"{m.profit_factor:.3f}",
                f"{m.max_drawdown_pct:.6f}", m.total_trades
            ])
            f.flush()
            os.fsync(f.fileno())
            print(m.summary())

    print("FAST backtests done.")
    print(f"CSV saved at: {out_path}")


if __name__ == "__main__":
    main()
