# Read‑Only Audit Prompt (EURUSD‑cTrader‑bot)

You are a **Senior Quantitative Systems Auditor**. Your job is to review this repo **read‑only** and produce a ruthless, practical audit. This is **not HFT** and must run on **free/low‑cost infrastructure** (no latency assumptions). Focus on **realism, robustness, and survival in live trading**.

## Context
- Instrument: **EURUSD**
- Broker/API: **cTrader Open API**
- Data: **SQLite ticks → candles (m1/m5/m15/h1)**
- Backtests: **bar‑based replay with bid/ask, session‑aware spreads & slippage**
- Goal: production‑grade backtesting and a strategy that can survive live execution.

## What You Must Audit
### 1) Data Integrity & Pipeline
- Tick cleaning: duplicates, gaps, time ordering, price sanity, spread handling.
- Candle construction: confirm **mid OHLC is built from mid ticks** (not bid/ask max/min mismatches).
- Spread stats: verify **min/mean/max** spreads per bar are captured and used.
- Session tagging: DST and timezone correctness.

### 2) Backtest Engine Realism
- Lookahead bias: confirm signals do **not** use future information.
- Execution logic: verify **bid/ask fills**, slippage, and spreads are applied realistically.
- Signal timing: open/close/close+1bar handling and implications.
- News pause: ensure backtests **block entries** during high‑impact events.
- Risk controls: daily loss caps, max trades, position sizing, margin safety.

### 3) Strategy Logic (EURUSD)
Audit these 5 strategies in `backtest/eurusd_strategies.py`:
1. London Open Breakout
2. Asian Range Fade
3. MA Cross + ADX
4. Bollinger Bounce
5. RSI Divergence

For each:
- **Logic traps** (where it cheats or breaks in live)
- **Data traps** (spread spikes, session quirks, gaps)
- **Execution traps** (slippage, order fill behavior)
- **Fixes** that are realistic for free infra

### 4) Robustness & Validation
- Walk‑forward validity
- Parameter stability (avoid hill‑climbing)
- Regime dependence
- Monte Carlo + perturbations
- Sensitivity to costs and latency

### 5) Production Gaps
- Crash recovery, state reconciliation, reconnects
- Logging/monitoring
- Misconfiguration risk (envs, schema drift)

## What to Deliver (Required Format)
1. **Executive Summary** (plain English)
2. **Critical Flaws** (top 5, ordered by severity)
3. **Strategy‑Specific Forensics** (for all 5 strategies)
4. **Backtest Realism Score** (0–10) + justification
5. **Fix Plan (prioritized)**
6. **Quick Wins (1–2 days)**
7. **Long‑Term Fixes (1–4 weeks)**

## Constraints
- Do **not** suggest HFT or paid infra.
- Do **not** assume we can get tick‑perfect fills.
- Do **not** require proprietary tools.

## Useful Entry Points
- `backtest/engine.py`
- `backtest/data_pipeline.py`
- `backtest/eurusd_strategies.py`
- `backtest/session.py`
- `backtest/costs.py`
- `scripts/build_backtest_data.py`
- `scripts/run_backtests_eurusd_fast.py`

## Important Note
Please **validate every claim against code**. If something is unclear, flag it as an assumption. This audit will directly drive production changes.
