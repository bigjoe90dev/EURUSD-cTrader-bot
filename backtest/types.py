"""
Shared datatypes for backtest engine + strategies.
"""
from dataclasses import dataclass
from datetime import datetime


class Signal:
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    FLAT = "FLAT"


@dataclass
class Bar:
    ts: datetime
    bid_o: float
    bid_h: float
    bid_l: float
    bid_c: float
    ask_o: float
    ask_h: float
    ask_l: float
    ask_c: float
    mid_o: float
    mid_h: float
    mid_l: float
    mid_c: float
    spread_o_pips: float
    spread_c_pips: float
    spread_min_pips: float = 0.0
    spread_mean_pips: float = 0.0
    spread_max_pips: float = 0.0
    session: str = "Off"


@dataclass
class Context:
    atr_pips: float
    session: str
    spread_pips: float
    bar_index: int
