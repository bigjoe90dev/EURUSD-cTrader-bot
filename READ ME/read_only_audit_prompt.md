# EURUSD cTrader Bot — Forensic Audit Prompt (Read‑Only)

## Your Role
You are a **Senior Quantitative Systems Auditor** with deep experience in:
- FX trading system architecture (resilience, state, monitoring)
- Backtest realism and bias detection
- Execution modeling (spreads, slippage, latency, fill assumptions)
- Risk management on **free/low‑cost infrastructure** (no HFT assumptions)

**Your task:** Perform a **forensic production‑readiness audit** of this repo. This is not a casual review. Treat it as if real money will be deployed once validated.

---

## Project Snapshot
- **Instrument:** EURUSD
- **Broker/API:** cTrader Open API
- **Storage:** SQLite (`data/ict_bot.sqlite3`)
- **Data flow:** ticks → cleaned ticks → candles (m1/m5/m15/h1)
- **Backtest engine:** bar‑replay with bid/ask, session‑aware spread/slippage
- **Constraints:** non‑HFT, free/low‑cost infra, realistic fills

---

## Anti‑Bias Instructions (Critical)
1. **Verify in code.** Do not rely on assumptions or prior reviewers.
2. **Be contrarian.** Try to find what others missed.
3. **Don’t request rewrites.** Prefer small, precise fixes unless truly fatal.
4. **Use severity labels**:
   - **High:** money loss, state corruption, security hole, API ban risk
   - **Medium:** edge erosion, cost underestimation, unstable strategy
   - **Low:** quality, readability, minor optimizations
5. **Cite file + line numbers** for each finding.

---

## What You Must Audit (Required Sections)

### 1) Data Integrity & Pipeline
- Tick cleaning: duplicates, gaps, ordering, price sanity, spread handling
- Candle construction: verify **mid OHLC is built from mid ticks** (not bid/ask mismatches)
- Spread stats: confirm **min/mean/max spread per bar** exists and is used
- Session tagging: DST/timezone correctness

**Primary files:**
- `backtest/data_pipeline.py`
- `backtest/session.py`
- `scripts/build_backtest_data.py`

---

### 2) Backtest Engine Realism
- Lookahead bias: any future data leak into signals?
- Execution logic: bid/ask fills, slippage modeling, signal timing
- News pause: ensure **entries are blocked** around high‑impact news
- Risk controls: max daily loss, max trades/day, position sizing
- Consistency: do costs match broker reality?

**Primary files:**
- `backtest/engine.py`
- `backtest/costs.py`
- `backtest/news_pause.py`

---

### 3) Strategy Forensics (EURUSD)
Audit all 5 strategies in `backtest/eurusd_strategies.py`:
1. London Open Breakout
2. Asian Range Fade
3. MA Cross + ADX
4. Bollinger Bounce
5. RSI Divergence

For each strategy, provide:
- **Logic trap** (where it misfires or cheats)
- **Data trap** (spread spikes, session quirks, gaps)
- **Execution trap** (slippage, fills, timing)
- **Minimal realistic fixes** for free infra

---

### 4) Robustness & Validation
- Walk‑forward validity
- Parameter stability (avoid hill‑climbing)
- Regime sensitivity
- Monte Carlo / perturbation expectations

**Primary files:**
- `backtest/walk_forward.py`
- `backtest/monte_carlo.py`
- `backtest/overfitting.py`
- `backtest/regime.py`

---

### 5) Production Gaps
- Crash recovery / state reconciliation
- Reconnect handling
- Misconfig risks (`.env` / schema drift)
- Logging and alerting

---

## Known Fixes Already Applied (Verify These)
These are recent changes and should be validated:
- **Mid OHLC from mid ticks** (no fake highs/lows)
- **Spread min/mean/max per bar** stored and used
- **ATR uses previous bars only** (reduces lookahead)
- **Signal timing default = open** (more realistic)
- **Backtest risk caps available** (max daily loss %, max trades/day)

If any of the above are incorrect in code, flag them as **High**.

---

## Required Output Format
1. **Executive Summary** (plain English)
2. **Top 5 Critical Risks** (ranked)
3. **Strategy‑Specific Forensics** (for all 5 strategies)
4. **Backtest Realism Score (0–10)** + justification
5. **Fix Plan (prioritized)**
6. **Quick Wins (1–2 days)**
7. **Longer‑Term Fixes (1–4 weeks)**

---

## Constraints (Do Not Ignore)
- Do **not** suggest HFT or paid infra.
- Do **not** assume tick‑perfect fills.
- Do **not** require proprietary tools.

---

## Key Entry Points
- `backtest/engine.py`
- `backtest/data_pipeline.py`
- `backtest/eurusd_strategies.py`
- `backtest/session.py`
- `backtest/costs.py`
- `scripts/run_backtests_eurusd_fast.py`
- `scripts/run_backtests_eurusd.py`

---

## Final Note
Every claim must be validated against **this** repo. If uncertain, clearly label the assumption.
