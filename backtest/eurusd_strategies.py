"""
EURUSD strategy pack for backtesting (formalized, no look-ahead).
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional, Tuple
from collections import deque

from dateutil import tz

from backtest.types import Signal, Bar, Context
from backtest.engine import Strategy


# ---------------------------
# Helpers
# ---------------------------

LONDON_TZ = tz.gettz("Europe/London")


def to_london(ts_utc: datetime) -> datetime:
    return ts_utc.astimezone(LONDON_TZ) if LONDON_TZ else ts_utc


def in_window(ts_utc: datetime, start: str, end: str) -> bool:
    local = to_london(ts_utc)
    s_h, s_m = start.split(":")
    e_h, e_m = end.split(":")
    s = time(int(s_h), int(s_m))
    e = time(int(e_h), int(e_m))
    t = local.time()
    if s <= e:
        return s <= t < e
    return t >= s or t < e


def after_time(ts_utc: datetime, t_str: str) -> bool:
    local = to_london(ts_utc)
    h, m = t_str.split(":")
    target = time(int(h), int(m))
    return local.time() >= target


def sma(values: List[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def rsi(values: List[float], n: int = 14) -> Optional[float]:
    if len(values) < n + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-n, 0):
        diff = values[i] - values[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger(values: List[float], n: int = 20, k: float = 2.0) -> Optional[Tuple[float, float, float]]:
    if len(values) < n:
        return None
    window = values[-n:]
    mean = sum(window) / n
    var = sum((x - mean) ** 2 for x in window) / n
    std = var ** 0.5
    return mean, mean + k * std, mean - k * std


def adx(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    trs = []
    pdm = []
    ndm = []
    for i in range(1, n + 1):
        up_move = highs[-i] - highs[-i - 1]
        down_move = lows[-i - 1] - lows[-i]
        pdm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        ndm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr = max(highs[-i] - lows[-i], abs(highs[-i] - closes[-i - 1]), abs(lows[-i] - closes[-i - 1]))
        trs.append(tr)
    tr_sum = sum(trs)
    if tr_sum == 0:
        return 0.0
    pdi = 100 * (sum(pdm) / tr_sum)
    ndi = 100 * (sum(ndm) / tr_sum)
    dx = 100 * abs(pdi - ndi) / max(pdi + ndi, 1e-9)
    return dx


class PivotDetector:
    """Detects pivots without look-ahead (confirms after wing bars)."""

    def __init__(self, wing: int = 2):
        self.wing = wing
        self.window = deque(maxlen=wing * 2 + 1)
        self.index = -1
        self.pivots_high: List[Tuple[int, float]] = []
        self.pivots_low: List[Tuple[int, float]] = []

    def update(self, price_high: float, price_low: float) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, float]]]:
        self.index += 1
        self.window.append((self.index, price_high, price_low))
        if len(self.window) < self.window.maxlen:
            return None, None
        # Center is candidate pivot
        mid_idx, mid_high, mid_low = self.window[self.wing]
        highs = [x[1] for x in self.window]
        lows = [x[2] for x in self.window]
        ph = None
        pl = None
        if mid_high == max(highs):
            self.pivots_high.append((mid_idx, mid_high))
            ph = (mid_idx, mid_high)
        if mid_low == min(lows):
            self.pivots_low.append((mid_idx, mid_low))
            pl = (mid_idx, mid_low)
        return ph, pl


# ---------------------------
# 1) London Open Breakout
# ---------------------------

@dataclass
class LondonOpenBreakoutConfig:
    range_start: str = "07:00"
    range_end: str = "08:00"
    trade_start: str = "08:00"
    trade_end: str = "11:00"
    exit_time: str = "12:00"
    buffer_pips: float = 2.0
    rr: float = 2.0
    min_range_pips: float = 8.0
    max_range_pips: float = 40.0
    max_spread_pips: float = 2.5
    min_atr_pips: float = 0.5
    max_atr_pips: float = 6.0
    min_body_ratio: float = 0.50
    one_trade_per_day: bool = True


class LondonOpenBreakoutStrategy(Strategy):
    def __init__(self, cfg: LondonOpenBreakoutConfig = None):
        self.cfg = cfg or LondonOpenBreakoutConfig()
        self.day = None
        self.range_high = None
        self.range_low = None
        self.range_ready = False
        self.side = None
        self.entry = None
        self.risk_pips = None
        self.traded_today = False

    def name(self) -> str:
        return "london_open_breakout"

    def params_dict(self) -> dict:
        return vars(self.cfg)

    def timeframe(self) -> str:
        return "m5"

    def reset(self):
        self.day = None
        self.range_high = None
        self.range_low = None
        self.range_ready = False
        self.side = None
        self.entry = None
        self.risk_pips = None
        self.traded_today = False

    def on_bar(self, bar: Bar, ctx: Context) -> str:
        local = to_london(bar.ts)
        if self.day != local.date():
            self.day = local.date()
            self.range_high = None
            self.range_low = None
            self.range_ready = False
            self.side = None
            self.entry = None
            self.risk_pips = None
            self.traded_today = False

        # Build range
        if in_window(bar.ts, self.cfg.range_start, self.cfg.range_end):
            self.range_high = bar.mid_h if self.range_high is None else max(self.range_high, bar.mid_h)
            self.range_low = bar.mid_l if self.range_low is None else min(self.range_low, bar.mid_l)

        # Finalize range at trade start
        if not self.range_ready and after_time(bar.ts, self.cfg.trade_start):
            if self.range_high is not None and self.range_low is not None:
                range_pips = (self.range_high - self.range_low) / 0.0001
                if range_pips < self.cfg.min_range_pips or range_pips > self.cfg.max_range_pips:
                    self.range_ready = False
                else:
                    self.range_ready = True
                    self.risk_pips = max(range_pips, 1e-6)

        # Exit logic
        if self.side:
            stop = self.entry - (self.risk_pips * 0.0001) if self.side == "LONG" else self.entry + (self.risk_pips * 0.0001)
            tp = self.entry + (self.cfg.rr * self.risk_pips * 0.0001) if self.side == "LONG" else self.entry - (self.cfg.rr * self.risk_pips * 0.0001)
            # Conservative: if both hit, assume stop first
            if self.side == "LONG":
                if bar.mid_l <= stop:
                    self.side = None
                    return Signal.CLOSE
                if bar.mid_h >= tp:
                    self.side = None
                    return Signal.CLOSE
            if self.side == "SHORT":
                if bar.mid_h >= stop:
                    self.side = None
                    return Signal.CLOSE
                if bar.mid_l <= tp:
                    self.side = None
                    return Signal.CLOSE
            if after_time(bar.ts, self.cfg.exit_time):
                self.side = None
                return Signal.CLOSE

        if not self.range_ready or self.side:
            return Signal.FLAT
        if self.cfg.one_trade_per_day and self.traded_today:
            return Signal.FLAT
        if not in_window(bar.ts, self.cfg.trade_start, self.cfg.trade_end):
            return Signal.FLAT
        if ctx.spread_pips > self.cfg.max_spread_pips:
            return Signal.FLAT
        if not (self.cfg.min_atr_pips <= ctx.atr_pips <= self.cfg.max_atr_pips):
            return Signal.FLAT

        buffer = self.cfg.buffer_pips * 0.0001
        bar_range = max(bar.mid_h - bar.mid_l, 1e-9)
        body = abs(bar.mid_c - bar.mid_o)
        body_ratio = body / bar_range

        if bar.mid_c > (self.range_high + buffer):
            if body_ratio < self.cfg.min_body_ratio:
                return Signal.FLAT
            self.side = "LONG"
            self.entry = bar.mid_c
            self.traded_today = True
            return Signal.BUY
        if bar.mid_c < (self.range_low - buffer):
            if body_ratio < self.cfg.min_body_ratio:
                return Signal.FLAT
            self.side = "SHORT"
            self.entry = bar.mid_c
            self.traded_today = True
            return Signal.SELL
        return Signal.FLAT


# ---------------------------
# 2) Asian Range Fade
# ---------------------------

@dataclass
class AsianRangeFadeConfig:
    session_start: str = "22:00"
    session_end: str = "06:00"
    lookback_hours: int = 4
    buffer_pips: float = 1.0
    max_spread_pips: float = 3.0


class AsianRangeFadeStrategy(Strategy):
    def __init__(self, cfg: AsianRangeFadeConfig = None):
        self.cfg = cfg or AsianRangeFadeConfig()
        self.bars: deque = deque(maxlen=600)
        self.side = None
        self.entry = None
        self.target = None
        self.stop = None

    def name(self) -> str:
        return "asian_range_fade"

    def params_dict(self) -> dict:
        return vars(self.cfg)

    def timeframe(self) -> str:
        return "m5"

    def reset(self):
        self.bars.clear()
        self.side = None
        self.entry = None
        self.target = None
        self.stop = None

    def on_bar(self, bar: Bar, ctx: Context) -> str:
        self.bars.append((bar.ts, bar.mid_h, bar.mid_l, bar.mid_c))

        if not in_window(bar.ts, self.cfg.session_start, self.cfg.session_end):
            if self.side:
                self.side = None
                return Signal.CLOSE
            return Signal.FLAT

        if ctx.spread_pips > self.cfg.max_spread_pips:
            return Signal.FLAT

        # Exit
        if self.side == "LONG":
            if bar.mid_l <= self.stop or bar.mid_h >= self.target:
                self.side = None
                return Signal.CLOSE
        if self.side == "SHORT":
            if bar.mid_h >= self.stop or bar.mid_l <= self.target:
                self.side = None
                return Signal.CLOSE

        if self.side:
            return Signal.FLAT

        needed = int((self.cfg.lookback_hours * 60) / 5)
        if len(self.bars) < needed:
            return Signal.FLAT
        recent = list(self.bars)[-needed:]
        highs = [x[1] for x in recent]
        lows = [x[2] for x in recent]
        range_high = max(highs)
        range_low = min(lows)
        mid_range = (range_high + range_low) / 2
        buffer = self.cfg.buffer_pips * 0.0001

        if bar.mid_c >= range_high - buffer:
            self.side = "SHORT"
            self.entry = bar.mid_c
            self.stop = range_high + buffer
            self.target = mid_range
            return Signal.SELL
        if bar.mid_c <= range_low + buffer:
            self.side = "LONG"
            self.entry = bar.mid_c
            self.stop = range_low - buffer
            self.target = mid_range
            return Signal.BUY

        return Signal.FLAT


# ---------------------------
# 3) MA Cross + ADX filter
# ---------------------------

@dataclass
class MACrossADXConfig:
    fast: int = 20
    slow: int = 50
    adx_period: int = 14
    adx_min: float = 20.0


class MACrossADXStrategy(Strategy):
    def __init__(self, cfg: MACrossADXConfig = None):
        self.cfg = cfg or MACrossADXConfig()
        self.bars: deque = deque(maxlen=500)
        self.fast_prev = None
        self.slow_prev = None
        self.side = None

    def name(self) -> str:
        return "ma_cross_adx"

    def params_dict(self) -> dict:
        return vars(self.cfg)

    def timeframe(self) -> str:
        return "m15"

    def reset(self):
        self.bars.clear()
        self.fast_prev = None
        self.slow_prev = None
        self.side = None

    def on_bar(self, bar: Bar, ctx: Context) -> str:
        self.bars.append((bar.mid_h, bar.mid_l, bar.mid_c))
        closes = [x[2] for x in self.bars]
        highs = [x[0] for x in self.bars]
        lows = [x[1] for x in self.bars]

        fast = sma(closes, self.cfg.fast)
        slow = sma(closes, self.cfg.slow)
        if fast is None or slow is None:
            return Signal.FLAT

        adx_val = adx(highs, lows, closes, self.cfg.adx_period)
        if adx_val is None or adx_val < self.cfg.adx_min:
            if self.side:
                self.side = None
                return Signal.CLOSE
            self.fast_prev = fast
            self.slow_prev = slow
            return Signal.FLAT

        signal = Signal.FLAT
        if self.fast_prev is not None and self.slow_prev is not None:
            crossed_up = self.fast_prev <= self.slow_prev and fast > slow
            crossed_down = self.fast_prev >= self.slow_prev and fast < slow
            if crossed_up:
                self.side = "LONG"
                signal = Signal.BUY
            elif crossed_down:
                self.side = "SHORT"
                signal = Signal.SELL

        self.fast_prev = fast
        self.slow_prev = slow
        return signal


# ---------------------------
# 4) Bollinger Band Bounce (anti-band-walk)
# ---------------------------

@dataclass
class BollingerBounceConfig:
    period: int = 20
    k: float = 2.0
    band_walk_bars: int = 3
    cooldown_bars: int = 5


class BollingerBounceStrategy(Strategy):
    def __init__(self, cfg: BollingerBounceConfig = None):
        self.cfg = cfg or BollingerBounceConfig()
        self.closes: List[float] = []
        self.side = None
        self.outside_count = 0
        self.cooldown = 0
        self.prev_outside = None

    def name(self) -> str:
        return "bollinger_bounce"

    def params_dict(self) -> dict:
        return vars(self.cfg)

    def timeframe(self) -> str:
        return "m15"

    def reset(self):
        self.closes = []
        self.side = None
        self.outside_count = 0
        self.cooldown = 0
        self.prev_outside = None

    def on_bar(self, bar: Bar, ctx: Context) -> str:
        self.closes.append(bar.mid_c)
        bb = bollinger(self.closes, self.cfg.period, self.cfg.k)
        if bb is None:
            return Signal.FLAT
        mid_band, upper, lower = bb

        # Detect band walk
        outside = "upper" if bar.mid_c > upper else ("lower" if bar.mid_c < lower else None)
        if outside:
            self.outside_count += 1
        else:
            self.outside_count = 0
        if self.outside_count >= self.cfg.band_walk_bars:
            self.cooldown = self.cfg.cooldown_bars

        if self.cooldown > 0:
            self.cooldown -= 1
            self.prev_outside = outside
            return Signal.FLAT

        # Exit at mid band
        if self.side == "LONG" and bar.mid_c >= mid_band:
            self.side = None
            return Signal.CLOSE
        if self.side == "SHORT" and bar.mid_c <= mid_band:
            self.side = None
            return Signal.CLOSE

        if self.side:
            self.prev_outside = outside
            return Signal.FLAT

        # Entry only on reclaim (prev outside, current inside)
        if self.prev_outside == "lower" and bar.mid_c > lower:
            self.side = "LONG"
            return Signal.BUY
        if self.prev_outside == "upper" and bar.mid_c < upper:
            self.side = "SHORT"
            return Signal.SELL

        self.prev_outside = outside
        return Signal.FLAT


# ---------------------------
# 5) RSI Divergence (pivot-confirmed)
# ---------------------------

@dataclass
class RSIDivergenceConfig:
    rsi_period: int = 14
    lookback: int = 80
    pivot_wing: int = 2
    min_rsi_diff: float = 6.0
    min_price_pips: float = 5.0
    max_hold_bars: int = 6


class RSIDivergenceStrategy(Strategy):
    def __init__(self, cfg: RSIDivergenceConfig = None):
        self.cfg = cfg or RSIDivergenceConfig()
        self.bars: List[Tuple[float, float, float]] = []  # (high, low, close)
        self.side = None
        self.hold_bars = 0
        self.pivots = PivotDetector(wing=self.cfg.pivot_wing)

    def name(self) -> str:
        return "rsi_divergence"

    def params_dict(self) -> dict:
        return vars(self.cfg)

    def timeframe(self) -> str:
        return "m15"

    def reset(self):
        self.bars = []
        self.side = None
        self.hold_bars = 0
        self.pivots = PivotDetector(wing=self.cfg.pivot_wing)

    def on_bar(self, bar: Bar, ctx: Context) -> str:
        self.bars.append((bar.mid_h, bar.mid_l, bar.mid_c))
        if len(self.bars) < self.cfg.lookback:
            return Signal.FLAT

        highs = [x[0] for x in self.bars]
        lows = [x[1] for x in self.bars]
        closes = [x[2] for x in self.bars]

        ph, pl = self.pivots.update(bar.mid_h, bar.mid_l)

        # Exit after N bars or RSI mean reversion
        if self.side:
            self.hold_bars += 1
            r = rsi(closes, self.cfg.rsi_period)
            if self.hold_bars >= self.cfg.max_hold_bars:
                self.side = None
                self.hold_bars = 0
                return Signal.CLOSE
            if self.side == "LONG" and r is not None and r > 55:
                self.side = None
                self.hold_bars = 0
                return Signal.CLOSE
            if self.side == "SHORT" and r is not None and r < 45:
                self.side = None
                self.hold_bars = 0
                return Signal.CLOSE
            return Signal.FLAT

        # Check divergence on confirmed pivots
        if ph:
            # compare last two pivot highs
            if len(self.pivots.pivots_high) >= 2:
                i1, p1 = self.pivots.pivots_high[-2]
                i2, p2 = self.pivots.pivots_high[-1]
                if (p2 - p1) / 0.0001 >= self.cfg.min_price_pips:
                    r1 = rsi(closes[:i1 + 1], self.cfg.rsi_period)
                    r2 = rsi(closes[:i2 + 1], self.cfg.rsi_period)
                    if r1 is not None and r2 is not None and (r1 - r2) >= self.cfg.min_rsi_diff:
                        self.side = "SHORT"
                        self.hold_bars = 0
                        return Signal.SELL

        if pl:
            if len(self.pivots.pivots_low) >= 2:
                i1, p1 = self.pivots.pivots_low[-2]
                i2, p2 = self.pivots.pivots_low[-1]
                if (p1 - p2) / 0.0001 >= self.cfg.min_price_pips:
                    r1 = rsi(closes[:i1 + 1], self.cfg.rsi_period)
                    r2 = rsi(closes[:i2 + 1], self.cfg.rsi_period)
                    if r1 is not None and r2 is not None and (r2 - r1) >= self.cfg.min_rsi_diff:
                        self.side = "LONG"
                        self.hold_bars = 0
                        return Signal.BUY

        return Signal.FLAT
