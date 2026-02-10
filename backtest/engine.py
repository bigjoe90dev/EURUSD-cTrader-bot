"""
Core backtest engine (bar-based).

Replays historical candle data, applies strategy signals, simulates fills
with session-aware spreads/slippage, and computes performance metrics.
"""

import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from backtest.costs import (
    CostConfig,
    commission_cost_usd,
    swap_cost_usd,
    session_spread_multiplier,
    session_slip_multiplier,
    clamp_spread,
)
from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.types import Bar, Context, Signal
from shared.db import connect


# ──────────────────────────────────────────────
# Strategy interface
# ──────────────────────────────────────────────

class Strategy(ABC):
    """Abstract base for all trading strategies."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def params_dict(self) -> dict:
        ...

    @abstractmethod
    def on_bar(self, bar: Bar, ctx: Context) -> str:
        """Process a bar and return a Signal (BUY, SELL, CLOSE, or FLAT)."""
        ...

    def reset(self):
        """Reset internal state for a fresh run."""
        pass

    def timeframe(self) -> str:
        """Return timeframe string for this strategy (e.g., 'm5', 'm15')."""
        return "m5"


class MACrossStrategy(Strategy):
    """Simple moving average crossover — baseline strategy."""

    def __init__(self, fast_n: int = 10, slow_n: int = 30):
        self.fast_n = fast_n
        self.slow_n = slow_n
        self._mids: deque = deque(maxlen=slow_n + 5)

    def name(self) -> str:
        return "ma_cross"

    def params_dict(self) -> dict:
        return {"fast_n": self.fast_n, "slow_n": self.slow_n}

    def reset(self):
        self._mids.clear()

    def on_bar(self, bar: Bar, ctx: Context) -> str:
        self._mids.append(bar.mid_c)
        if len(self._mids) < self.slow_n:
            return Signal.FLAT
        vals = list(self._mids)
        fast = sum(vals[-self.fast_n:]) / self.fast_n
        slow = sum(vals[-self.slow_n:]) / self.slow_n
        if fast > slow:
            return Signal.BUY
        if fast < slow:
            return Signal.SELL
        return Signal.FLAT


# ──────────────────────────────────────────────
# Position tracking
# ──────────────────────────────────────────────

@dataclass
class Position:
    side: str           # "LONG" or "SHORT"
    entry_price: float
    size_lots: float
    entry_ts: datetime
    entry_weekday: int  # 0=Mon..6=Sun
    entry_spread_pips: float
    entry_slip_pips: float


@dataclass
class TradeRecord:
    ts_open: datetime
    ts_close: datetime
    side: str
    size_lots: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    spread_cost: float
    slippage_cost: float
    commission_cost: float
    swap_cost: float
    net_pnl: float


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────

@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: List[TradeRecord]
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    strategy_name: str = ""
    strategy_params: dict = field(default_factory=dict)
    symbol: str = ""
    date_from: str = ""
    date_to: str = ""


class BacktestEngine:
    """
    Offline backtest engine.

    Replays candle data, applies strategy signals, manages one position
    at a time, tracks equity, and computes metrics.
    """

    def __init__(self,
                 strategy: Strategy,
                 cost_config: CostConfig = None,
                 initial_balance: float = 10_000.0,
                 size_lots: float = 0.1,
                 symbol: str = "EURUSD",
                 atr_lookback: int = 14,
                 signal_timing: str = "open",
                 spread_model: str = "empirical",
                 slippage_model: str = "session_volatility",
                 max_daily_loss_pct: float = 0.0,
                 max_trades_per_day: int = 0):
        self.strategy = strategy
        self.cfg = cost_config or CostConfig()
        self.initial_balance = initial_balance
        self.size_lots = size_lots
        self.symbol = symbol
        self.atr_lookback = atr_lookback
        self.signal_timing = signal_timing
        self.spread_model = spread_model
        self.slippage_model = slippage_model
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_trades_per_day = max_trades_per_day

    def run(self, df: pd.DataFrame, pause_mask=None) -> BacktestResult:
        """
        Run a backtest on a DataFrame of candles.

        Required columns:
            ts_utc, bid_o, bid_h, bid_l, bid_c,
            ask_o, ask_h, ask_l, ask_c,
            mid_o, mid_h, mid_l, mid_c,
            spread_o_pips, spread_c_pips, session
        """
        self.strategy.reset()

        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["ts_utc"]):
            data["ts_utc"] = pd.to_datetime(data["ts_utc"], errors="coerce", utc=True)
            data = data.dropna(subset=["ts_utc"])
        data = data.sort_values("ts_utc").reset_index(drop=True)

        n = len(data)
        equity = np.full(n, self.initial_balance, dtype=float)
        balance = self.initial_balance
        position: Optional[Position] = None
        trades: List[TradeRecord] = []
        recent_mids: deque = deque(maxlen=self.atr_lookback + 1)

        # Cost accumulators
        total_costs = {"spread": 0.0, "slippage": 0.0, "commission": 0.0, "swap": 0.0}

        pending_signal = None
        prev_bar = None
        prev_ctx = None
        trades_today = 0
        day_start_balance = self.initial_balance
        day_halt = False
        current_day = None

        if pause_mask is not None:
            pause_mask = pd.Series(pause_mask).reset_index(drop=True)

        for i in range(n):
            row = data.iloc[i]
            bar = self._row_to_bar(row)
            bar_day = bar.ts.date()
            if current_day is None or bar_day != current_day:
                current_day = bar_day
                day_start_balance = balance
                trades_today = 0
                day_halt = False
            atr_pips = self._estimate_atr_pips(recent_mids)
            spread_for_ctx = bar.spread_max_pips if bar.spread_max_pips > 0 else bar.spread_c_pips
            ctx = Context(atr_pips=atr_pips, session=bar.session,
                          spread_pips=spread_for_ctx, bar_index=i)

            # Daily loss guard (equity-based)
            if self.max_daily_loss_pct > 0.0:
                if position is not None:
                    mtm = self._mark_to_market(position, bar, ctx)
                    equity_now = balance + mtm
                else:
                    equity_now = balance
                if not day_halt:
                    loss_pct = max(0.0, day_start_balance - equity_now) / max(day_start_balance, 1e-9)
                    if loss_pct >= self.max_daily_loss_pct:
                        day_halt = True
                        if position is not None:
                            position, balance = self._apply_signal(
                                Signal.CLOSE, position, balance, bar, ctx, price_point="close", trades=trades, totals=total_costs
                            )

            # Execute any pending signal at open (close_plus_1bar)
            if self.signal_timing == "close_plus_1bar" and pending_signal:
                sig = pending_signal
                if pause_mask is not None and bool(pause_mask.iloc[i]) and sig in (Signal.BUY, Signal.SELL):
                    sig = Signal.FLAT
                if day_halt and sig in (Signal.BUY, Signal.SELL):
                    sig = Signal.FLAT
                if self.max_trades_per_day and trades_today >= self.max_trades_per_day and sig in (Signal.BUY, Signal.SELL):
                    sig = Signal.CLOSE if position is not None else Signal.FLAT
                pre_pos = position
                position, balance = self._apply_signal(
                    sig, position, balance, bar, ctx, price_point="open", trades=trades, totals=total_costs
                )
                if sig in (Signal.BUY, Signal.SELL):
                    if pre_pos is None and position is not None:
                        trades_today += 1
                    elif pre_pos is not None and position is not None and pre_pos.entry_ts != position.entry_ts:
                        trades_today += 1
                pending_signal = None

            # "open" timing: evaluate previous bar, execute at current open
            if self.signal_timing == "open" and prev_bar is not None and prev_ctx is not None:
                signal = self.strategy.on_bar(prev_bar, prev_ctx)
                if pause_mask is not None and bool(pause_mask.iloc[i]) and signal in (Signal.BUY, Signal.SELL):
                    signal = Signal.FLAT
                if day_halt and signal in (Signal.BUY, Signal.SELL):
                    signal = Signal.FLAT
                if self.max_trades_per_day and trades_today >= self.max_trades_per_day and signal in (Signal.BUY, Signal.SELL):
                    signal = Signal.CLOSE if position is not None else Signal.FLAT
                pre_pos = position
                position, balance = self._apply_signal(
                    signal, position, balance, bar, ctx, price_point="open", trades=trades, totals=total_costs
                )
                if signal in (Signal.BUY, Signal.SELL):
                    if pre_pos is None and position is not None:
                        trades_today += 1
                    elif pre_pos is not None and position is not None and pre_pos.entry_ts != position.entry_ts:
                        trades_today += 1

            # "close" and "close_plus_1bar" timing: evaluate current bar
            if self.signal_timing in ("close", "close_plus_1bar"):
                signal = self.strategy.on_bar(bar, ctx)
                if pause_mask is not None and bool(pause_mask.iloc[i]) and signal in (Signal.BUY, Signal.SELL):
                    signal = Signal.FLAT
                if day_halt and signal in (Signal.BUY, Signal.SELL):
                    signal = Signal.FLAT
                if self.max_trades_per_day and trades_today >= self.max_trades_per_day and signal in (Signal.BUY, Signal.SELL):
                    signal = Signal.CLOSE if position is not None else Signal.FLAT
                if self.signal_timing == "close":
                    pre_pos = position
                    position, balance = self._apply_signal(
                        signal, position, balance, bar, ctx, price_point="close", trades=trades, totals=total_costs
                    )
                    if signal in (Signal.BUY, Signal.SELL):
                        if pre_pos is None and position is not None:
                            trades_today += 1
                        elif pre_pos is not None and position is not None and pre_pos.entry_ts != position.entry_ts:
                            trades_today += 1
                else:
                    pending_signal = signal

            # Mark-to-market at close
            if position is not None:
                mtm = self._mark_to_market(position, bar, ctx)
                equity[i] = balance + mtm
            else:
                equity[i] = balance

            recent_mids.append(bar.mid_c)

            prev_bar = bar
            prev_ctx = ctx

        # Close any open position at end
        if position is not None and n > 0:
            row = data.iloc[-1]
            bar = self._row_to_bar(row)
            atr_pips = self._estimate_atr_pips(recent_mids)
            spread_for_ctx = bar.spread_max_pips if bar.spread_max_pips > 0 else bar.spread_c_pips
            ctx = Context(atr_pips=atr_pips, session=bar.session,
                          spread_pips=spread_for_ctx, bar_index=n - 1)
            position, balance = self._apply_signal(
                Signal.CLOSE, position, balance, bar, ctx, price_point="close", trades=trades, totals=total_costs
            )
            equity[-1] = balance

        # Drawdown curve
        running_max = np.maximum.accumulate(equity)
        dd_curve = (equity - running_max) / np.where(running_max > 0, running_max, 1)

        # Compute metrics
        trade_pnls = [t.net_pnl for t in trades]

        if n > 1:
            total_seconds = (data["ts_utc"].iloc[-1] - data["ts_utc"].iloc[0]).total_seconds()
            secs_per_bar = total_seconds / (n - 1) if n > 1 else 1
            bars_per_year = (365.25 * 86400) / max(secs_per_bar, 0.001)
        else:
            bars_per_year = 252 * 86400

        metrics = compute_metrics(equity, trade_pnls, total_costs, bars_per_year=bars_per_year)

        return BacktestResult(
            metrics=metrics,
            trades=trades,
            equity_curve=equity,
            drawdown_curve=dd_curve,
            strategy_name=self.strategy.name(),
            strategy_params=self.strategy.params_dict(),
            symbol=self.symbol,
            date_from=str(data["ts_utc"].iloc[0]) if n > 0 else "",
            date_to=str(data["ts_utc"].iloc[-1]) if n > 0 else "",
        )

    # ----------------------- internals -----------------------

    def _row_to_bar(self, row) -> Bar:
        return Bar(
            ts=row["ts_utc"].to_pydatetime(),
            bid_o=float(row["bid_o"]),
            bid_h=float(row["bid_h"]),
            bid_l=float(row["bid_l"]),
            bid_c=float(row["bid_c"]),
            ask_o=float(row["ask_o"]),
            ask_h=float(row["ask_h"]),
            ask_l=float(row["ask_l"]),
            ask_c=float(row["ask_c"]),
            mid_o=float(row["mid_o"]),
            mid_h=float(row["mid_h"]),
            mid_l=float(row["mid_l"]),
            mid_c=float(row["mid_c"]),
            spread_o_pips=float(row["spread_o_pips"]),
            spread_c_pips=float(row["spread_c_pips"]),
            spread_min_pips=float(row.get("spread_min_pips", 0.0)),
            spread_mean_pips=float(row.get("spread_mean_pips", 0.0)),
            spread_max_pips=float(row.get("spread_max_pips", 0.0)),
            session=str(row.get("session", "Off")),
        )

    def _select_spread_pips(self, bar: Bar, price_point: str) -> float:
        if bar.spread_max_pips > 0:
            return bar.spread_max_pips
        if price_point == "open":
            return bar.spread_o_pips
        return bar.spread_c_pips

    def _effective_bid_ask(self, mid: float, spread_pips: float, session: str):
        if self.spread_model == "fixed":
            base = self.cfg.spread_typical_pips
        elif self.spread_model == "hybrid":
            base = clamp_spread(spread_pips, self.cfg)
        else:  # empirical
            base = spread_pips

        mult = session_spread_multiplier(session, self.cfg)
        adj = clamp_spread(base * mult, self.cfg)
        half = (adj * self.cfg.pip_size) / 2.0
        bid = mid - half
        ask = mid + half
        return bid, ask, adj

    def _slippage_pips(self, atr_pips: float, session: str) -> float:
        if self.slippage_model == "fixed":
            return self.cfg.slippage_base_pips
        base = self.cfg.slippage_base_pips + (self.cfg.slippage_vol_factor * atr_pips)
        return base * session_slip_multiplier(session, self.cfg)

    def _apply_signal(self, signal: str, position: Optional[Position],
                      balance: float, bar: Bar, ctx: Context,
                      price_point: str, trades: List[TradeRecord],
                      totals: dict):
        if signal == Signal.FLAT:
            return position, balance

        # Determine effective bid/ask at this price point
        if price_point == "open":
            mid = bar.mid_o
        else:
            mid = bar.mid_c
        spread_pips = self._select_spread_pips(bar, price_point)

        eff_bid, eff_ask, eff_spread = self._effective_bid_ask(mid, spread_pips, ctx.session)
        slip_pips = self._slippage_pips(ctx.atr_pips, ctx.session)
        slip_price = slip_pips * self.cfg.pip_size

        def fill_price(side: str) -> float:
            if side in ("BUY", "LONG"):
                return eff_ask + slip_price
            return eff_bid - slip_price

        if signal == Signal.CLOSE:
            if position is not None:
                trade = self._close_position(position, bar.ts, eff_bid, eff_ask, eff_spread, slip_pips)
                trades.append(trade)
                balance += trade.net_pnl
                self._accum_costs(totals, trade)
                position = None
            return position, balance

        if signal == Signal.BUY:
            if position is not None and position.side == "SHORT":
                trade = self._close_position(position, bar.ts, eff_bid, eff_ask, eff_spread, slip_pips)
                trades.append(trade)
                balance += trade.net_pnl
                self._accum_costs(totals, trade)
                position = None

            if position is None:
                entry_price = fill_price("LONG")
                position = Position(
                    side="LONG",
                    entry_price=entry_price,
                    size_lots=self.size_lots,
                    entry_ts=bar.ts,
                    entry_weekday=bar.ts.weekday(),
                    entry_spread_pips=eff_spread,
                    entry_slip_pips=slip_pips,
                )

        if signal == Signal.SELL:
            if position is not None and position.side == "LONG":
                trade = self._close_position(position, bar.ts, eff_bid, eff_ask, eff_spread, slip_pips)
                trades.append(trade)
                balance += trade.net_pnl
                self._accum_costs(totals, trade)
                position = None

            if position is None:
                entry_price = fill_price("SHORT")
                position = Position(
                    side="SHORT",
                    entry_price=entry_price,
                    size_lots=self.size_lots,
                    entry_ts=bar.ts,
                    entry_weekday=bar.ts.weekday(),
                    entry_spread_pips=eff_spread,
                    entry_slip_pips=slip_pips,
                )

        return position, balance

    def _close_position(self, pos: Position, ts: datetime,
                        eff_bid: float, eff_ask: float,
                        eff_spread_pips: float,
                        slip_pips: float) -> TradeRecord:
        exit_side = "SELL" if pos.side == "LONG" else "BUY"
        slip_price = slip_pips * self.cfg.pip_size

        if exit_side == "SELL":
            exit_price = eff_bid - slip_price
        else:
            exit_price = eff_ask + slip_price

        if pos.side == "LONG":
            price_diff = exit_price - pos.entry_price
        else:
            price_diff = pos.entry_price - exit_price

        pips = price_diff / self.cfg.pip_size
        gross_pnl = pips * self.cfg.pip_value_per_lot * pos.size_lots

        # Reporting-only costs (already baked into fills)
        spread_cost = ((pos.entry_spread_pips + eff_spread_pips) / 2.0) * self.cfg.pip_value_per_lot * pos.size_lots
        slippage_cost = (pos.entry_slip_pips + slip_pips) * self.cfg.pip_value_per_lot * pos.size_lots
        commission = commission_cost_usd(pos.size_lots, self.cfg)

        days_held = (ts - pos.entry_ts).total_seconds() / 86400 if hasattr(ts, "timestamp") else 0
        swap = swap_cost_usd(pos.side, pos.size_lots, days_held, pos.entry_weekday, self.cfg)

        net_pnl = gross_pnl - commission - swap

        return TradeRecord(
            ts_open=pos.entry_ts, ts_close=ts,
            side=pos.side, size_lots=pos.size_lots,
            entry_price=pos.entry_price, exit_price=exit_price,
            gross_pnl=gross_pnl,
            spread_cost=spread_cost, slippage_cost=slippage_cost,
            commission_cost=commission, swap_cost=swap,
            net_pnl=net_pnl,
        )

    def _mark_to_market(self, pos: Position, bar: Bar, ctx: Context) -> float:
        spread_pips = bar.spread_mean_pips if bar.spread_mean_pips > 0 else bar.spread_c_pips
        eff_bid, eff_ask, _ = self._effective_bid_ask(bar.mid_c, spread_pips, ctx.session)
        if pos.side == "LONG":
            price_diff = eff_bid - pos.entry_price
        else:
            price_diff = pos.entry_price - eff_ask
        pips = price_diff / self.cfg.pip_size
        return pips * self.cfg.pip_value_per_lot * pos.size_lots

    def _estimate_atr_pips(self, mids: deque) -> float:
        if len(mids) < 2:
            return 1.0
        vals = list(mids)
        changes = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
        atr = sum(changes) / len(changes)
        return atr / self.cfg.pip_size

    @staticmethod
    def _accum_costs(totals: dict, trade: TradeRecord):
        totals["spread"] += trade.spread_cost
        totals["slippage"] += trade.slippage_cost
        totals["commission"] += trade.commission_cost
        totals["swap"] += trade.swap_cost


# ──────────────────────────────────────────────
# DB persistence
# ──────────────────────────────────────────────

def save_backtest_result(result: BacktestResult, note: str = "", cost_model_json: str = "{}", extra_metrics: dict = None) -> int:
    """Save a BacktestResult to the database. Returns the run_id."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db = connect()

    cur = db.execute(
        """INSERT INTO backtest_runs
           (ts_utc, strategy_name, params_json, symbol, date_from, date_to, cost_model_json, note)
           VALUES (?,?,?,?,?,?,?,?)""",
        (ts, result.strategy_name, json.dumps(result.strategy_params),
         result.symbol, result.date_from, result.date_to,
         cost_model_json, note),
    )
    run_id = cur.lastrowid

    extra_metrics = extra_metrics or {}

    for k, v in result.metrics.as_dict().items():
        if isinstance(v, (int, float)):
            db.execute(
                "INSERT INTO backtest_results (run_id, metric_name, metric_value) VALUES (?,?,?)",
                (run_id, k, float(v)),
            )
    for k, v in extra_metrics.items():
        if isinstance(v, (int, float)):
            db.execute(
                "INSERT INTO backtest_results (run_id, metric_name, metric_value) VALUES (?,?,?)",
                (run_id, k, float(v)),
            )

    for t in result.trades:
        db.execute(
            """INSERT INTO backtest_trades
               (run_id, ts_utc, action, side, size_units, price,
                spread_cost, slippage_cost, commission_cost, swap_cost, pnl, note)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, str(t.ts_close), "CLOSE", t.side, t.size_lots,
             t.exit_price, t.spread_cost, t.slippage_cost,
             t.commission_cost, t.swap_cost, t.net_pnl, ""),
        )

    eq = result.equity_curve
    dd = result.drawdown_curve
    step = max(1, len(eq) // 500)
    for i in range(0, len(eq), step):
        db.execute(
            "INSERT INTO backtest_equity (run_id, ts_utc, equity, drawdown) VALUES (?,?,?,?)",
            (run_id, str(i), float(eq[i]), float(dd[i])),
        )

    db.commit()
    db.close()
    return run_id
