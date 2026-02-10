"""
Backtest configuration loaded from environment.
"""
from dataclasses import dataclass
import os


def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


@dataclass(frozen=True)
class BacktestSettings:
    data_source: str = "sqlite_ticks"          # sqlite_ticks | csv | external
    timezone: str = "UTC"                      # for report only; data stored UTC
    session_rules: str = ""                    # optional rules string
    spread_model: str = "empirical"            # empirical | fixed | hybrid
    slippage_model: str = "session_volatility" # session_volatility | fixed
    signal_timing: str = "open"                # close | open | close_plus_1bar

    # Cleaning / QC thresholds
    spread_outlier_mult: float = 10.0          # drop ticks > median*mult per session
    max_spread_pips_abs: float = 20.0          # absolute spread cap
    min_spread_pips_abs: float = 0.01          # absolute min spread
    price_min: float = 0.5                     # hard sanity bound
    price_max: float = 2.5                     # hard sanity bound


def load_backtest_settings() -> BacktestSettings:
    return BacktestSettings(
        data_source=_get_env("BACKTEST_DATA_SOURCE", "sqlite_ticks"),
        timezone=_get_env("BACKTEST_TIMEZONE", "UTC"),
        session_rules=_get_env("BACKTEST_SESSION_RULES", ""),
        spread_model=_get_env("BACKTEST_SPREAD_MODEL", "empirical"),
        slippage_model=_get_env("BACKTEST_SLIPPAGE_MODEL", "session_volatility"),
        signal_timing=_get_env("BACKTEST_SIGNAL_TIMING", "close_plus_1bar"),
        spread_outlier_mult=float(_get_env("BACKTEST_SPREAD_OUTLIER_MULT", "10.0")),
        max_spread_pips_abs=float(_get_env("BACKTEST_MAX_SPREAD_PIPS", "20.0")),
        min_spread_pips_abs=float(_get_env("BACKTEST_MIN_SPREAD_PIPS", "0.01")),
        price_min=float(_get_env("BACKTEST_PRICE_MIN", "0.5")),
        price_max=float(_get_env("BACKTEST_PRICE_MAX", "2.5")),
    )
