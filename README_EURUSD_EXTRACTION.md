# EURUSD SQLite Extraction Evidence

## Scope
- Working copy path: `/Users/joe/2026 projects/EURUSD-cTrader-bot copy for CODEX 5.3 copy`
- Extraction date: `2026-02-17`
- Source DB discovered: `data/ict_bot.sqlite3` (`228,917,248` bytes, about `218 MB`)
- Source DB copied to: `research/data_cache/eurusd/raw/ict_bot.sqlite3`

## SQLite Discovery + Inspection
- Candidate SQLite files found:
  - `data/ict_bot.sqlite3` (`228,917,248` bytes)
- DB table count: `19`
- EURUSD evidence:
  - `backtest_runs.symbol` distinct values: `EURUSD`
  - `quotes.symbol_id` counts: `1 -> 371,395`, `128 -> 3,481`, `41 -> 729`

## Relevant Market Data Tables
- `ticks_clean`
  - Rows: `365,220`
  - Timestamp column: `ts_utc` (`TEXT`)
  - Time range: `2024-01-01 22:00:00+00:00` to `2025-12-01 23:59:00+00:00`
  - Price columns: `bid`, `ask`, `mid`
  - Volume column: not present
- `candles_m1`
  - Rows: `365,220`
  - Timestamp column: `ts_utc` (`TEXT`)
  - Time range: `2024-01-01 22:00:00+00:00` to `2025-12-01 23:59:00+00:00`
  - OHLC columns: `mid_o`, `mid_h`, `mid_l`, `mid_c` (+ bid/ask OHLC variants)
- `candles_m5`
  - Rows: `73,421`
  - Time range: `2024-01-01 22:00:00+00:00` to `2025-12-01 23:55:00+00:00`
- `candles_h1`
  - Rows: `6,119`
  - Time range: `2024-01-01 22:00:00+00:00` to `2025-12-01 23:00:00+00:00`

## Data Type Decision
- Primary extraction source used: `ticks_clean` (tick-like bid/ask/mid stream).
- Note: `ticks_clean` is sampled at 1-minute cadence in this DB.
- Normalized bars were generated from `mid` price via resampling.
- Assumption recorded: EURUSD identity is inferred from `backtest_runs.symbol = EURUSD` and project context; `ticks_clean` has no explicit symbol column.

## Normalized Outputs
Saved under `research/data_cache/eurusd/normalized/`:

- `eurusd_ticks.parquet` (+ `eurusd_ticks.csv`)
  - Rows: `365,220`
  - Range: `2024-01-01 22:00:00+00:00` to `2025-12-01 23:59:00+00:00`
  - Schema: `ts_utc, bid, ask, mid, source`
- `eurusd_ohlc_1m.parquet` (+ `eurusd_ohlc_1m.csv`)
  - Rows: `365,220`
  - Range: `2024-01-01 22:00:00+00:00` to `2025-12-01 23:59:00+00:00`
- `eurusd_ohlc_5m.parquet` (+ `eurusd_ohlc_5m.csv`)
  - Rows: `73,421`
  - Range: `2024-01-01 22:00:00+00:00` to `2025-12-01 23:55:00+00:00`
- `eurusd_ohlc_1h.parquet` (+ `eurusd_ohlc_1h.csv`)
  - Rows: `6,119`
  - Range: `2024-01-01 22:00:00+00:00` to `2025-12-01 23:00:00+00:00`

Bar schema used:
- `ts_utc, open, high, low, close, volume, source`
- `volume` is `null` (not available in source ticks table)

## Validation Results
Validator script: `research/eurusd_extract/validate_output.py`

Checks performed:
- timestamps strictly increasing
- duplicate timestamp detection
- no NaN in required fields
- OHLC invariants (`high >= open/close/low`, `low <= open/close/high`)
- gap detection + count reporting

Validation status: `PASS`

Gap summary from run:
- `eurusd_ticks.parquet`: `gap_intervals=1541`, `missing_steps=642900`
- `eurusd_ohlc_1m.parquet`: `gap_intervals=1541`, `missing_steps=642900`
- `eurusd_ohlc_5m.parquet`: `gap_intervals=59`, `missing_steps=128203`
- `eurusd_ohlc_1h.parquet`: `gap_intervals=52`, `missing_steps=10683`

Interpretation:
- Gap counts are reported for awareness and are not treated as fatal validation errors.

## Scripts + Logs
Scripts created:
- `research/eurusd_extract/inspect_sqlite.py`
- `research/eurusd_extract/extract_sqlite.py`
- `research/eurusd_extract/validate_output.py`

Log artifacts:
- `research/artefacts/logs/inspect_summary.json`
- `research/artefacts/logs/inspect_summary.md`
- `research/artefacts/logs/extract_summary.json`
- `research/artefacts/logs/validation_summary.json`

## Repro Commands
```bash
python -m research.eurusd_extract.inspect_sqlite --db data/ict_bot.sqlite3
python -m research.eurusd_extract.extract_sqlite --db data/ict_bot.sqlite3
python -m research.eurusd_extract.validate_output \
  --ticks research/data_cache/eurusd/normalized/eurusd_ticks.parquet \
  --ohlc research/data_cache/eurusd/normalized/eurusd_ohlc_1m.parquet \
         research/data_cache/eurusd/normalized/eurusd_ohlc_5m.parquet \
         research/data_cache/eurusd/normalized/eurusd_ohlc_1h.parquet
```
