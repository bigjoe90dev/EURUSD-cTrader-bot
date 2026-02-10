import json
import sqlite3

import pandas as pd

from backtest.config import load_backtest_settings
from backtest.data_pipeline import build_backtest_tables
from backtest.engine import BacktestEngine, save_backtest_result
from backtest.costs import CostConfig
from backtest.monte_carlo import parameter_sensitivity
from backtest.walk_forward import walk_forward_analysis
from backtest.overfitting import detect_overfitting
from backtest.regime import label_regimes, regime_performance
from backtest.strategy_registry import param_grid, param_sampler, strategy_factory
from scripts import init_db
from shared.config import load_settings
from backtest.news_pause import build_news_pause_mask
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


def existing_runs(db_path: str, note: str) -> set:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "select strategy_name from backtest_runs where note=?",
        (note,),
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


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
    print("Initializing DB schema...")
    init_db.main()
    settings = load_settings()
    bt_cfg = load_backtest_settings()
    db_path = str(settings.db_path)

    # Build cleaned ticks + candle tables
    print("Building cleaned ticks + candle tables...")
    counts = build_backtest_tables(db_path=db_path, symbol_id=settings.ctrader_symbol_id or 1, cfg=bt_cfg)
    print("Counts:", counts)

    note = "eurusd_batch_5_v2"
    done = existing_runs(db_path, note)

    strategies = [
        LondonOpenBreakoutStrategy(),
        AsianRangeFadeStrategy(),
        MACrossADXStrategy(),
        BollingerBounceStrategy(),
        RSIDivergenceStrategy(),
    ]

    cfg = CostConfig()

    for strat in strategies:
        if strat.name() in done:
            print(f"Skipping {strat.name()} (already saved).")
            continue

        tf_table = table_for_timeframe(strat.timeframe())
        df = load_candles(db_path, tf_table)
        if df.empty:
            print(f"No candle data in {tf_table} for {strat.name()}.")
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

        # Robustness / validation layers
        factory = strategy_factory(strat.name())
        grid = param_grid(strat.name())
        sampler = param_sampler(strat.name())
        engine_kwargs = {
            "signal_timing": bt_cfg.signal_timing,
            "spread_model": bt_cfg.spread_model,
            "slippage_model": bt_cfg.slippage_model,
            "max_daily_loss_pct": bt_cfg.max_daily_loss_pct,
            "max_trades_per_day": bt_cfg.max_trades_per_day,
        }

        wf = walk_forward_analysis(
            df,
            param_grid=grid,
            cost_config=cfg,
            initial_balance=10_000.0,
            size_lots=0.1,
            symbol="EURUSD",
            strategy_factory=factory,
            engine_kwargs=engine_kwargs,
        )

        mc = parameter_sensitivity(
            df,
            n_samples=100,
            sharpe_threshold=0.5,
            cost_config=cfg,
            initial_balance=10_000.0,
            size_lots=0.1,
            symbol="EURUSD",
            strategy_factory=factory,
            param_sampler=sampler,
            engine_kwargs=engine_kwargs,
        )

        of = detect_overfitting(
            df,
            strategy_params=strat.params_dict(),
            cost_config=cfg,
            initial_balance=10_000.0,
            size_lots=0.1,
            symbol="EURUSD",
            strategy_factory=factory,
            engine_kwargs=engine_kwargs,
        )

        # Regime performance
        regimes = label_regimes(df)
        trade_regimes = []
        ts_index = pd.to_datetime(df["ts_utc"], utc=True)
        for t in result.trades:
            idx = ts_index.searchsorted(pd.Timestamp(t.ts_open, tz="UTC"), side="right") - 1
            idx = max(0, min(int(idx), len(regimes) - 1))
            trade_regimes.append(str(regimes.iloc[idx]))
        reg_report = regime_performance([t.net_pnl for t in result.trades], trade_regimes)

        extra_metrics = {
            "wf_consistency_pct": wf.consistency_pct,
            "wf_avg_oos_sharpe": wf.avg_oos_sharpe,
            "wf_avg_degradation": wf.avg_degradation,
            "mc_robustness_pct": mc.robustness_pct,
            "mc_median_sharpe": mc.median_sharpe,
            "overfit_score": of.robustness_score,
        }
        for r, stats in reg_report.stats.items():
            extra_metrics[f"regime_{r}_trades"] = stats.trade_count
            extra_metrics[f"regime_{r}_win_rate"] = stats.win_rate
            extra_metrics[f"regime_{r}_total_pnl"] = stats.total_pnl

        cost_model_json = json.dumps({
            "spread_model": bt_cfg.spread_model,
            "slippage_model": bt_cfg.slippage_model,
            "signal_timing": bt_cfg.signal_timing,
        })
        run_id = save_backtest_result(result, note=note, cost_model_json=cost_model_json, extra_metrics=extra_metrics)
        print(result.metrics.summary())
        print(wf.summary())
        print(mc.summary())
        print(of.summary())
        print(reg_report.summary())
        print(f"Saved run_id={run_id}\n")


if __name__ == "__main__":
    main()
