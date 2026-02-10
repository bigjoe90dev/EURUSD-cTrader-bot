# Polymarket Copy Trading Bot (plus Arb Scanner) — Forensic Technical Audit Request (Repo ZIP)

## Your Role

You are an expert trading systems auditor with deep knowledge of:

* Python trading bot architecture (reliability, concurrency, state, observability)
* Polymarket ecosystem: **CLOB (order book)** + **public Data API** (leaderboard/activity/trades)
* Risk management for small-to-mid sized accounts (exposure limits, kill-switches, failure modes)
* Statistical edge analysis under **fees, slippage, and stale signals**
* Paper trading realism and simulation calibration

**Your task:** Perform a **forensic, production-readiness audit** of the attached repository ZIP. This is **not** a casual review. Treat it like a system that will trade real money (even if currently PAPER mode).

---

## Version Info

**CONFIG_VERSION**: 11 (as of February 10, 2026)
**Status**: PAPER MODE — Not yet deployed live
**Starting Balance**: $50 (paper trading)
**Review Round**: 2 (previous round: 4 LLM reviews identified 5 critical issues — **ALL NOW FIXED IN v11**)

**Critical Fixes in v11** (implemented February 10, 2026):
- ✅ **Fee calculation FIXED**: Now uses curved Polymarket formula `fee = (bps/10000) * price * (1-price)` instead of flat fee (was overcharging by 4× at mid-prices)
- ✅ **Staleness measurement FIXED**: Now uses whale's trade timestamp from API, not detection time (was undercounting by 5+ minutes with 700 wallet polling)
- ✅ **Exposure persistence FIXED**: RiskGuard now saves state to `data/risk_state.json` (exposure + daily loss survives restarts)
- ✅ **Arb scanner DISABLED**: `ENABLE_ARB_SCANNER = False` by default (negative EV: HFT competition + two-leg execution risk)

**Changes in v10**:
- Dynamic TP/SL: Fast markets 20% TP / 12% SL, Slow markets 30% TP / 15% SL
- Category-specific wallet scoring: Separates whale performance by market type
- Telegram notification system (code complete, not yet configured)
- Removed crypto-only filter: Now trades ALL markets (crypto, sports, politics, etc.)
- Winner's Curse protection: 8% max price deviation cap
- Time Intelligence: 3-minute expiry block, exponential decay near expiry
- Faster polling: 0.5s main loop, 1s whale poll (was 1s / 2s)

> **Important context:** This bot is **not HFT** and does **not** require low-latency or paid infrastructure. It is intended to run **24/7 for free on a local MacBook**. Your audit must reflect that reality (e.g., sleep/wake, Wi-Fi drops, local-only dashboard security, no "VPS latency" assumptions).

---

## Anti-Bias Instructions (CRITICAL — READ FIRST)

**This codebase may have been reviewed by other LLMs before you.** To prevent groupthink and ensure independent analysis, you **MUST** follow these anti-bias protocols:

### 1. **Do Not Agree by Default**
- If you find yourself wanting to say "I agree with the previous review," **STOP**.
- Re-examine the code yourself. Previous reviewers may be wrong.
- Only agree if you independently verified the issue by reading the actual code.

### 2. **Active Contrarian Thinking**
Your job is to find what **others missed**. Prioritize:
- **Bugs they didn't catch** (look for off-by-one errors, race conditions, math bugs)
- **Edge cases they ignored** (what happens when whale_price = 0? When API returns empty list?)
- **Architectural blind spots** (what if the user restarts the bot mid-trade? What if JSON is corrupted?)
- **Overly pessimistic assumptions** (maybe the stress sim is TOO conservative and paper results underestimate real performance?)

### 3. **Avoid Recursion Churn**
- Do **NOT** suggest changes that will require another round of reviews.
- Focus on **FATAL** issues only (money loss, state corruption, security holes).
- If a feature is "nice to have" but not critical, mark it as **Low** severity and move on.
- Prefer fixes that are <20 lines of code. Avoid suggesting full rewrites.

### 4. **Check Your Own Math**
- When calculating breakeven win rates, show your work step-by-step.
- Use the **actual constants from the code** (e.g., `SLIPPAGE_BASELINE = 0.003` from stress_sim.py).
- If you estimate "60% win rate needed," prove it with arithmetic, not intuition.

### 5. **You Don't Have Full Context**
- You are seeing a **snapshot** of the codebase.
- You do NOT know what changes were made since the last review.
- You do NOT know what issues were already fixed.
- If you suspect something is wrong, cite the **exact file and line number** so the developer can verify.

### 6. **Severity Calibration**
Use this exact rubric:
- **High**: Money loss, state corruption, security hole, API ban risk
- **Medium**: Performance degradation, suboptimal sizing, edge erosion
- **Low**: Code quality, readability, minor optimizations

If you're tempted to mark everything as "High," you're being too pessimistic. Most issues are "Medium."

### 7. **Concrete > Abstract**
- BAD: "The bot needs better error handling."
- GOOD: "In `src/paper_engine.py:425`, if `signal.get('timestamp')` returns None, the code will crash with TypeError. Add `or time.time()` as a fallback."

### 8. **Distrust Your Priors**
- You may have been trained on other trading bot codebases that use different architectures.
- **This bot uses atomic writes with os.replace** — don't suggest adding a database unless you can prove it's necessary.
- **This bot uses paper trading for testing** — don't suggest running it live without validation unless you can prove paper is accurate.

### 9. **Cross-Check with Other Reviewers (If Provided)**
- If you are given previous reviews, read them **after** you've written your initial findings.
- Then add a section: "What I Found That Others Missed" and "What Others Found That I Disagree With."
- Provide technical justification for disagreements (cite code, not vibes).

### 10. **The Ultimate Test**
Before submitting your review, ask yourself:
- **If this were my own $5,000, would I trust my review enough to act on it?**
- **Did I actually read the code, or am I just pattern-matching against "typical trading bot issues"?**
- **If the developer implements only my "High" severity fixes, will the bot be safer?**

**Your goal is not to be the harshest reviewer. Your goal is to be the most *correct* reviewer.**

---

## Known Critical Issues (From Previous Reviews)

**Status: FIXED IN v11** — These issues were identified by 4 independent LLM reviews (DeepSeek, Gemini, Grok, GPT-4o) and have been fixed in CONFIG_VERSION 11. **Your job is to verify the fixes are correct and look for NEW issues.**

### 1. Fee Calculation Bug (FATAL) — ✅ FIXED
- **Location**: `src/paper_engine.py` lines 458, 595, 712
- **Issue**: Used flat fee formula `fee = cost * bps/10000`, but Polymarket uses curved formula `fee = (bps/10000) * price * (1-price)`. Was overcharging fees by 4× at mid-prices.
- **Impact**: Paper results were MORE pessimistic than reality (good for safety, but misleading).
- **Fix**: Now calls `calculate_trading_fee(price, size, fee_bps)` which uses the correct curved formula.
- **Verification**: Check lines 458, 595, 712 in paper_engine.py — should all use `calculate_trading_fee()`.

### 2. Staleness Measurement Bug (FATAL) — ✅ FIXED
- **Location**: `src/paper_engine.py:execute_copy_trade()` line ~427
- **Issue**: Measured staleness as `time.time() - signal.get("detected_at")`, which is when WE detected the signal, not when the whale actually traded. With 700 wallets polled every 1-2 seconds, whale trades could be 5+ minutes stale.
- **Impact**: Winner's curse protection and stress simulation were using wrong timestamps, underestimating staleness penalties.
- **Fix**: Now uses `whale_trade_time = signal.get("timestamp", signal.get("detected_at"))` — fallback to detected_at only if API doesn't provide timestamp.
- **Verification**: Check line ~427 — should use `signal.get("timestamp")` first.

### 3. Exposure Accounting Not Persisted (HIGH) — ✅ FIXED
- **Location**: `src/risk.py`
- **Issue**: Current exposure was tracked in memory only. On bot restart, exposure reset to 0 even if positions were open, allowing risk limits to be bypassed.
- **Impact**: Risk limits could be exceeded after restart. Daily loss limits also reset.
- **Fix**: RiskGuard now persists state to `data/risk_state.json` using atomic writes. Loads on __init__, saves after add_exposure(), remove_exposure(), record_loss().
- **Verification**: Check risk.py has `_save_state()`, `_load_state()`, and calls to `_save_state()` in add/remove/record methods.

### 4. Arb Scanner Negative EV (STRATEGIC) — ✅ DISABLED
- **Location**: `src/bot.py` line ~148
- **Issue**: All 4 reviewers agreed arb scanner has negative EV (HFT competition + two-leg execution risk).
- **Impact**: Arb strategy loses money.
- **Fix**: Added `ENABLE_ARB_SCANNER = False` config flag. Arb scanning code now wrapped in `if config.get("ENABLE_ARB_SCANNER", False):`.
- **Verification**: Check bot.py line ~148 — arb scanning should be wrapped in if statement checking ENABLE_ARB_SCANNER.

### 5. State Corruption Risk (MEDIUM) — ⚠️ PARTIALLY MITIGATED
- **Location**: All `_save_state()` methods
- **Issue**: No CRC checksums, schema versioning, or consistency checks on load.
- **Status**: Atomic writes (`os.replace`) prevent partial writes, but no checksums yet.
- **Remaining Work**: Add schema version field to all state files, add CRC or hash validation on load.

**Your focus should be:**
1. Verify the 4 FIXED issues are actually fixed correctly (check the code, not just trust the changelog)
2. Look for NEW bugs introduced by the fixes (e.g., did calculate_trading_fee() break something else?)
3. Find issues the previous reviewers MISSED

---

## What This Repo Actually Implements (Based on Code Structure)

### A) Copy Trading Subsystem (Directional)

* **Whale discovery & polling:** `src/whale_tracker.py`

  * Uses **public Polymarket Data API** endpoints:

    * `/v1/leaderboard` (MONTH PnL sorted) to discover traders
    * `/activity?user=<wallet>` to detect trades
    * `/trades?market=<conditionId>` for market-level context
  * Maintains state in `data/whale_state.json` with atomic write pattern.
  * Has heuristics and filters (PnL/volume “farmer test”, inactivity, wash ratio, hold-time checks, etc.).

* **Wallet scoring / category scoring / anti-hedge / flow clustering:** `src/wallet_scorer.py`

  * Bayesian Beta prior **Beta(2,2)** for win-rate smoothing (plus ROI and confidence weighting)
  * Category-aware scoring via market title regex classification:

    * `crypto_fast`, `sports_fast`, `slow`, `unknown`
  * Anti-hedge: prevents holding both YES and NO copy positions in same condition.
  * “Flow strength” (cluster activity) as a feature.

* **Copy execution in PAPER (and exit-copy):** `src/paper_engine.py`

  * `execute_copy_trade()` opens positions sized via:

    * **Half-Kelly** if wallet has ≥5 settled results (empirical win rate)
    * Otherwise a score-based sizing fallback with min/max clamps
  * Winner’s curse protection via max price deviation
  * Time intelligence: blocks near expiry and models expiry decay
  * Auto TP/SL management for unsettled copy positions using dynamic thresholds
  * State persistence: `data/paper_state.json` (atomic writes)
  * Integrates the stress simulator below.

* **Stress simulator (paper pessimism):** `src/stress_sim.py`

  * Models friction layers: rejection, partial fills, slippage stack, staleness, crowd, depletion, rate limiting, API failures, gas, spread widening, off-hours, expiry decay.

### B) Separate Arb Scanner Strategy (Locked Profit)

This repo also contains a second strategy that runs alongside copy trading in the main loop:

* **Arbitrage detection:** `src/strategy.py`

  * Detects `ask_yes + ask_no + buffer < 1.00` opportunities
  * Intended “free-infra edge” = breadth scanning of long-tail markets

* **Execution routing:** `src/execution.py`

  * PAPER routes to paper engine
  * LIVE uses `py-clob-client`, with two-leg logic, hedge/cancel policies

### C) Main Orchestration / Ops

* **Main loop:** `src/bot.py`

  * Every cycle:

    1. Updates dynamic risk limits (based on paper portfolio balance)
    2. Polls whales and executes copy trades/exits (paper)
    3. Scans market order books for arb opportunities
    4. Settles paper positions / records snapshots
  * Heartbeat watchdog thread for hang detection + emergency state save
* **Risk management:** `src/risk.py`
* **Market data service:** `src/market.py`
* **Local dashboard server:** `src/web_server.py` + `static/`
* **Notifications:** `src/notifier.py` (Telegram)
* **Data collection/backtesting:** `src/data_collector.py`, `src/backtester.py`, `backtest.py`

---

## Core Audit Questions (What You Must Evaluate)

### 1) **Is There Real Edge After Friction? (CRITICAL)**

You must quantify whether the **copy trading component** has a plausible edge under its own modeled friction.

**For copy trading**, compute (at minimum) breakeven requirements given:

* Fee model used in the code (see `src/paper_fees.py` + `PaperTradingEngine._get_fee_rate()` in `src/paper_engine.py`)
* Stress simulator’s baseline slippage + staleness + crowd penalties (`src/stress_sim.py`)
* Winner’s curse skip logic (`MAX_PRICE_DEVIATION` and config overrides)
* Half-Kelly sizing and min/max trade clamps (copy sizing in `execute_copy_trade()`)

**Deliverables:**

* Minimum *empirical* whale win rate needed for profitability at typical entry prices (e.g., price=0.45, 0.55, 0.70) under:

  * “Optimistic” friction (low slippage, minimal staleness)
  * “As-coded stress” friction (use simulator constants)
* A sanity check: do the simulator constants make paper results **too pessimistic** or **not pessimistic enough**?

**For the arb scanner**, assess:

* How often the “locked profit” condition is *actually* fillable (book depth realism)
* Execution failure modes (two-leg partial fills, hedge slippage, cancellation behavior)
* Whether the strategy survives fees/spread and produces positive EV at realistic fill rates

---

### 2) **Execution & Timing — Non-HFT Reality Check**

This system is **not** competing in microseconds. However, it does make frequent HTTP calls and runs 24/7 on a MacBook.

Audit:

* Polling frequency and CPU/network usage:

  * `CYCLE_SLEEP` and market batch behavior in `src/bot.py`
  * Whale polling interval constants in `src/whale_tracker.py`
* Rate limit / ban risk:

  * Data API calls (leaderboard pagination, activity polling)
  * CLOB calls (order books, market data)
* Whether the system is **over-polling** (wasted resources) or **under-polling** (stale signals) given prediction market dynamics

**MacBook-specific operational risks you must address:**

* Sleep / lid close / power saving killing the process
* Wi-Fi drops / IP changes / captive portal events
* Local disk full, log bloat, JSON state file growth over weeks
* Time sync drift (affects expiry parsing and time-based rules)

---

### 3) **Risk Management Robustness**

You must treat risk as adversarial.

Audit:

* Dynamic risk limits in `src/config.py` + updates in `src/bot.py` + enforcement in `src/risk.py`
* Concentration logic in `execute_copy_trade()` (per-market cap, exposure cap)
* Daily loss tracking / kill switch behavior
* Exposure accounting correctness:

  * Add exposure on entry; remove exposure on exit and settlement
  * Ensure copy exits and TP/SL exits always reconcile exposure correctly

**Catastrophic loss scenarios to test conceptually:**

* Many whales pile into the same market and are wrong (correlation blow-up)
* API failures during exits (stuck positions)
* TP/SL triggers based on stale prices (wrong current_price)
* Whale “exit-copy” signal arrives late and the book moved heavily
* State corruption: crash between cash deduction and position persistence

Deliver a list of “ways this can go to zero” and whether safeguards actually prevent them.

---

### 4) **Data Integrity & Whale Scoring Correctness**

Audit the entire whale pipeline:

* **Discovery bias:** leaderboard MONTH PnL selection inherently has survivorship / selection bias
* “Farmer test” and filters:

  * PnL/volume thresholds, inactivity, hold time, wash ratio logic in `src/whale_tracker.py`
* **Bayesian scoring implementation correctness:**

  * Priors, posterior mean logic, and update rules in `src/wallet_scorer.py`
  * Category scoring: does it actually prevent cross-domain overconfidence?
* **Minimum sample sizes:** are 3 results (score) and 5 results (Kelly) defensible?
* **Market classification via regex:** false positives/negatives

  * How classification errors propagate into TP/SL bands and sizing

You must identify any math bugs, leakage between categories, or score inflation/deflation issues.

---

### 5) **Paper Trading Realism vs. Self-Deception**

This repo relies heavily on PAPER mode as a “truth machine.”

Audit:

* Whether paper fills, slippage, fees, and gas models match Polymarket reality
* Whether paper assumes execution that would not be possible live (especially for arb two-leg fills)
* Whether stress simulation is internally consistent:

  * Are penalties double-counted?
  * Are failure probabilities plausible?
  * Do they scale sensibly with signal age, market fatigue, and trading cadence?

Deliver: a “paper-to-live gap” assessment and what must change before trusting paper PnL.

---

### 6) **Architecture, Reliability, and Production Readiness**

This runs 24/7. That means operational excellence matters more than clever logic.

Audit:

* Concurrency and thread safety:

  * Heartbeat watchdog interactions with state saving (`src/bot.py`)
  * Locks in `src/paper_engine.py`
  * Any shared dict mutation without locks (`whale_tracker`, `wallet_scorer`)
* Error handling:

  * Network timeouts, JSON parse failures, partial API responses
  * Retries/backoff and whether failures spiral into bans or silent no-trade
* State persistence:

  * Atomic write patterns used (tmp + `os.replace`)
  * Risk of partial writes and schema drift across versions
  * Growth of state files (`seen_tx_hashes`, trade history, positions)
* Observability:

  * Can you detect “bot is alive but doing nothing”?
  * Are there meaningful logs/metrics/alerts (Telegram integration, status reports)?

---

### 7) **Security Review (Local-Only, But Still)**

Even “localhost-only” services can be risky.

Audit:

* Dashboard access controls:

  * Token auth design in `src/web_server.py`
  * Binding behavior (`DASHBOARD_BIND` in config defaults)
  * Any unsafe routes, command execution, file writes, or info leaks
* Secrets handling:

  * Config file storage of API keys/private keys
  * Risk of accidentally logging secrets
* Supply chain / dependency risk:

  * `requirements.txt`
  * Unsafe imports or code paths that download/execute content

---

### 8) **Scaling & Capital Deployment Limits**

Even if this starts at $50, the user intends to scale.

Audit:

* What breaks at $500 / $2,000 / $5,000+?

  * Copy sizing caps and exposure percentages: do they scale safely?
  * Market liquidity constraints (thin books → awful fills)
  * Correlation: bigger account = bigger drawdowns from clustered whale behavior
* For arb scanning:

  * Can this strategy deploy meaningful size without moving the market?
  * Two-leg risk increases with size

Deliver a recommended scaling path and when to stop scaling.

---

### 9) **Missing Features / Blind Spots (Only If High-Impact)**

Do not suggest “extra complexity” unless it clearly improves survivability or EV.

Priority-rank additions that would materially reduce risk or improve edge, such as:

* Robust backoff + circuit breakers for API bans/rate limits
* Better data validation and schema versioning for state files
* “Shadow” mode parity tests (watch-only) before LIVE
* Maker/limit order logic (if fees/slippage dominate)
* Automated “stuck position” recovery logic for LIVE execution

---

## Audit Instructions (Process You Must Follow)

1. **Read the entire repository**:

   * All files in `src/`, plus `run.py`, `backtest.py`, `SPEC.md`, `AUDIT_PROMPT.md`, `llm_review_pack.txt`, `config/`
2. **Trace execution end-to-end**:

   * Startup → config → bot loop → whale discovery/polling → signal → copy sizing → stress sim → position state → TP/SL → exit-copy → settlement
   * Also trace arb scanning loop → opportunity → execution → hedge logic → settlement
3. **Validate every money-moving line**:

   * Fee math, slippage application, PnL calculations, exposure accounting, cash balance updates
4. **Think adversarially**:

   * Assume APIs degrade, whales behave strategically, markets move against you, and your machine crashes
5. **Judge against “24/7 local production” standards**:

   * Not research quality; not hobby code; real-money robustness

---

## Required Output Format

### A) Executive Summary (3–6 sentences)

* Overall verdict: **Production-ready / Needs work / Not viable**
* Biggest strength
* Biggest risk
* What you would do next (concrete)

### B) Critical Issues (Must Fix Before LIVE)

For each issue include:

* Severity: **High / Medium / Low**
* Location: `file.py:function` (and line numbers if you can)
* Why it matters
* Exact suggested fix (implementation-level)

### C) Edge Analysis

* Copy trading breakeven math (include assumptions and friction scenarios)
* Arb strategy fillability and net EV assessment
* “Probability of profitability over 6 months” with reasoning (not vibes)

### D) Risk & Failure Mode Review

* Top 10 ways the system can fail
* Which are mitigated, which are not
* Suggested kill-switch / circuit breaker improvements

### E) Paper-to-Live Gap

* Where paper is faithful
* Where paper is optimistic/pessimistic
* What validation steps are required before trusting it

### F) Scaling Assessment

* What changes at $500 / $2k / $5k+
* Liquidity + correlation + operational constraints
* Recommended capital deployment strategy (and when to stop)

### G) Architecture & Ops Grade

* Code quality: A/B/C/D/F
* Production readiness score: 0–100
* Operational checklist for “run 24/7 on MacBook” (sleep prevention, restart policy, log rotation, backups, alerts)

### H) Brutal Honesty Take

* Is this likely to make money, or mostly pay fees/slippage?
* If this were your own $5,000, would you run it? Under what strict conditions?
* The single biggest flaw that could sink it

---

## Notes You Must Respect

* This is **not HFT**; do not recommend paid infra or ultra-low latency as a default “fix.”
* Assume **local-only, free, always-on** operation is a hard constraint.
* Be specific. Vague advice (“add monitoring”) is not acceptable without concrete implementation guidance tied to repo files.

**Be harsh. Be technical. Treat every dollar like it’s real.**
