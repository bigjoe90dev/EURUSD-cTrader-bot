# Codex 5.3 Handoff — EURUSD cTrader Bot

**Date:** 2026-02-10  
**Repo:** `/Users/joe/2026 projects/EURUSD-cTrader-bot`  
**Goal:** Build a production‑grade EURUSD bot with realistic backtesting, then tune a viable strategy (starting with London Open Breakout).

---

## 1) End‑to‑End Summary (Plain English)

We collect EURUSD ticks from cTrader into SQLite, build cleaned candle tables, and run backtests against those candles.  
We have now **tightened realism** (worse spreads, no lookahead) which made results tougher.  
Right now all strategies are losing and **London Open Breakout is closest to breakeven**.  
Next step is to **tune London Open Breakout** and then run the **full robust backtest suite** (walk‑forward, Monte Carlo, regimes).

**Important constraint:** We are **not doing HFT**. This system is built for **free infrastructure**, so all logic must survive higher spreads, slippage, and slower execution.

---

## 2) Codebase Overview (What Each Part Does)

### Data + DB
- `data/ict_bot.sqlite3` — main SQLite DB (local)
- `shared/db.py` — DB path + connector
- `scripts/init_db.py` — creates all tables

### Data Pipeline (Backtests)
- `backtest/data_pipeline.py`
  - Cleans ticks
  - Tags sessions (Asia/Frankfurt/London/NY)
  - Builds candle tables: `candles_m1`, `candles_m5`, `candles_m15`, `candles_h1`
- `backtest/session.py` — DST‑aware session classification
- `backtest/news_pause.py` — builds a news pause mask for backtests

### Backtest Engine
- `backtest/engine.py`
  - Candle‑based execution
  - Bid/ask fills
  - Session‑based spread/slippage
  - Signal timing modes: `close`, `open`, `close_plus_1bar`
  - News pause support (blocks new entries during events)

### Strategies (EURUSD)
File: `backtest/eurusd_strategies.py`
- London Open Breakout
- Asian Range Fade
- MA Cross + ADX
- Bollinger Bounce
- RSI Divergence

### Robustness Tooling
Files:
- `backtest/walk_forward.py`
- `backtest/monte_carlo.py`
- `backtest/overfitting.py`
- `backtest/regime.py`

### Runners
- `scripts/build_backtest_data.py` — builds candles
- `scripts/run_backtests_eurusd_fast.py` — fast backtest → `/tmp/eurusd_fast_backtests.csv`
- `scripts/run_backtests_eurusd.py` — full robust backtests + DB writes

---

## 3) What We Have Done (Chronological)

1. **Renamed repo** to `EURUSD-cTrader-bot`
2. **Built backtest pipeline**:
   - Clean ticks
   - Session tagging
   - Candle tables
3. **Upgraded backtest engine**:
   - Bid/ask execution
   - Session‑based spread/slippage
   - Signal timing options
4. **Improved London Open Breakout**:
   - Max spread filter
   - ATR range filter
   - Candle body confirmation
   - One trade per day
5. **Added news pause support in backtests**
6. **Adjusted data cleaning**:
   - Drop impossible price ticks
   - Keep real spread spikes (only drop extreme/broken spreads)
7. **Improved candle construction**:
   - Mid OHLC now built from mid ticks (no fake highs/lows)
   - Spread stats stored per bar (min/mean/max) for more realistic fills
8. **ATR now computed without current bar contamination** (reduces lookahead bias)
9. **Backtest signal timing default set to `open`**
10. **Verified data exists**:
   - `quotes` ~375k rows
   - Candle tables now built successfully
9. **Ran fast backtests** and wrote results to:
   - `/tmp/eurusd_fast_backtests.csv`

---

## 4) Current State (Right Now)

### Data
- SQLite DB exists locally:
  - `data/ict_bot.sqlite3`
- Candle tables built:
  - `candles_m1=363,339`
  - `candles_m5=73,283`
  - `candles_m15=24,463`
  - `candles_h1=6,119`

### Fast Backtest Results (Latest)
File: `/tmp/eurusd_fast_backtests.csv`

These worsened after realism upgrades (spread max + no lookahead):
- London Open Breakout: `cagr=-0.000254`, `profit_factor=0.995`, `max_drawdown=1.96%`, trades 159  
- Asian Range Fade: `cagr=-0.186790`, `profit_factor=0.240`, `max_drawdown=32.87%`, trades 1549  
- MA Cross + ADX: `cagr=-0.042967`, `profit_factor=0.607`, `max_drawdown=8.18%`, trades 405  
- Bollinger Bounce: `cagr=-0.071506`, `profit_factor=0.646`, `max_drawdown=13.64%`, trades 995  
- RSI Divergence: `cagr=-0.031494`, `profit_factor=0.717`, `max_drawdown=7.03%`, trades 457

---

## 5) What’s Blocking / Confusing

**Main confusion:** backtests were “not working” because candle tables were empty.  
This is now fixed. Backtests run and produce output.

**LLM review status:** GPT, Gemini, Grok, and DeepSeek reviews are now populated in `LLM_reviews/v1/*/review 1`.  
DeepSeek includes some generic snippets; verify any claims against repo before changes.

---

## 6) Where We’re Headed (Next Steps)

1. **Tune London Open Breakout** against realistic costs
   - Re‑optimize buffer, ATR bounds, body ratio, and exit timing
2. **Run full robustness backtests**
   - Walk‑forward, Monte Carlo, regime analysis
3. **Add stronger risk controls** in backtests
   - Daily loss cap, max trades/day, and lot sizing via `backtest/position_sizer.py`
4. Decide:
   - If London Open Breakout survives robust tests → proceed to paper trading
   - If it fails → redesign strategy

---

## 7) How to Run (Cheat Sheet)

### Build candles
```
PYTHONPATH=. .venv/bin/python scripts/build_backtest_data.py
```

### Fast backtest (CSV)
```
PYTHONPATH=. .venv/bin/python scripts/run_backtests_eurusd_fast.py
```

### Full backtest
```
PYTHONPATH=. .venv/bin/python scripts/run_backtests_eurusd.py
```

---

## 8) Sharing Safety

`.env` and all secrets are ignored by git.  
Use GitHub zip or `git archive` to share safely.

---

## 9) LLM Review Drop‑Folders

Created:
```
LLM_reviews/v1/{grok,gemini,gpt,deepseek,kimi}
LLM_reviews/v2/{grok,gemini,gpt,deepseek,kimi}
LLM_reviews/v3/{grok,gemini,gpt,deepseek,kimi}
```

Paste reviews into those folders so we can compare and iterate.

---

## 10) Nautilus Trader (Unzipped for Review)

Location:
```
RESEARCH ME/nautilus_trader-develop/
```

We can borrow ideas such as:
- event‑driven engine structure
- order types (OCO, limit, stop)
- execution simulation
- portfolio accounting and risk engine

Full integration is heavy; we’ll copy design patterns instead.
