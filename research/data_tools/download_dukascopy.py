from __future__ import annotations

import argparse
import datetime as dt
import json
import lzma
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .common import day_range, dump_json, ensure_dir, load_yaml, parse_utc_ts, to_iso_z
from .validate_dataset import validate_file

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
SIDES = ("BID", "ASK")


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"days": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("days"), dict):
            return data
    except Exception:
        pass
    return {"days": {}}


def _save_index(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _day_key(day_ts: pd.Timestamp) -> str:
    return day_ts.strftime("%Y-%m-%d")


def _daily_url(symbol: str, day_ts: pd.Timestamp, side: str) -> str:
    year = day_ts.year
    month0 = day_ts.month - 1
    day = day_ts.day
    return f"{BASE_URL}/{symbol}/{year}/{month0:02d}/{day:02d}/{side}_candles_min_1.bi5"


def _daily_path(raw_root: Path, symbol: str, day_ts: pd.Timestamp, side: str) -> Path:
    return raw_root / symbol / f"{day_ts.year:04d}" / f"{day_ts.month:02d}" / f"{day_ts.day:02d}" / f"{side}_candles_min_1.bi5"


def _download_file(url: str, path: Path, timeout_sec: int = 45) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    retryable_statuses = {429, 500, 502, 503, 504}
    max_retries = 6
    backoff_sec = 1.0

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout_sec)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Dukascopy request error after retries for {url}: {exc}") from exc
            time.sleep(backoff_sec * (2**attempt))
            continue

        if r.status_code == 200:
            path.write_bytes(r.content)
            return 200
        if r.status_code == 404:
            return 404
        if r.status_code in retryable_statuses and attempt < max_retries - 1:
            time.sleep(backoff_sec * (2**attempt))
            continue

        raise RuntimeError(f"Dukascopy download failed {r.status_code} for {url}")

    raise RuntimeError(f"Dukascopy download failed after retries for {url}")


def _decode_daily_candles(raw_bytes: bytes, day_ts: pd.Timestamp, price_scale: float) -> pd.DataFrame:
    if not raw_bytes:
        return pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"])

    decoded = lzma.decompress(raw_bytes)
    if len(decoded) % 24 != 0:
        raise ValueError(f"Unexpected decoded candle payload size={len(decoded)} (not divisible by 24)")

    arr = np.frombuffer(
        decoded,
        dtype=[
            ("offset_sec", ">u4"),
            ("open_i", ">u4"),
            ("close_i", ">u4"),
            ("low_i", ">u4"),
            ("high_i", ">u4"),
            ("volume", ">f4"),
        ],
    )

    ts = day_ts + pd.to_timedelta(arr["offset_sec"].astype(np.int64), unit="s")
    out = pd.DataFrame(
        {
            "ts_utc": ts,
            "open": arr["open_i"].astype(np.float64) / float(price_scale),
            "high": arr["high_i"].astype(np.float64) / float(price_scale),
            "low": arr["low_i"].astype(np.float64) / float(price_scale),
            "close": arr["close_i"].astype(np.float64) / float(price_scale),
            "volume": arr["volume"].astype(np.float64),
        }
    )
    return out


def _update_side_index(index: dict[str, Any], day: pd.Timestamp, side: str, status: str, path: Path, url: str) -> None:
    dkey = _day_key(day)
    drec = index.setdefault("days", {}).setdefault(dkey, {})
    srec = drec.setdefault(side.lower(), {})
    srec.update(
        {
            "status": status,
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "url": url,
            "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def _merge_bid_ask_to_mid(
    bid_df: pd.DataFrame,
    ask_df: pd.DataFrame,
    single_side_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bid_df.empty and ask_df.empty:
        return (
            pd.DataFrame(columns=["ts_utc", "bid", "ask", "mid", "source"]),
            pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume", "source"]),
        )

    if bid_df.empty and not ask_df.empty:
        if single_side_mode == "fail":
            raise RuntimeError("ASK candles exist but BID candles are missing")
        bid_df = ask_df.copy()
    elif ask_df.empty and not bid_df.empty:
        if single_side_mode == "fail":
            raise RuntimeError("BID candles exist but ASK candles are missing")
        ask_df = bid_df.copy()

    merged = bid_df.merge(
        ask_df,
        on="ts_utc",
        suffixes=("_bid", "_ask"),
        how="inner",
    )
    if merged.empty:
        return (
            pd.DataFrame(columns=["ts_utc", "bid", "ask", "mid", "source"]),
            pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume", "source"]),
        )

    quotes = pd.DataFrame(
        {
            "ts_utc": merged["ts_utc"],
            "bid": merged["close_bid"],
            "ask": merged["close_ask"],
        }
    )
    quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2.0
    quotes["source"] = "dukascopy:datafeed.dukascopy.com:daily_1m_candles"

    mid_open = (merged["open_bid"] + merged["open_ask"]) / 2.0
    mid_close = (merged["close_bid"] + merged["close_ask"]) / 2.0
    mid_high_raw = (merged["high_bid"] + merged["high_ask"]) / 2.0
    mid_low_raw = (merged["low_bid"] + merged["low_ask"]) / 2.0

    ohlc = pd.DataFrame(
        {
            "ts_utc": merged["ts_utc"],
            "open": mid_open,
            "close": mid_close,
            "high": np.maximum.reduce([mid_open.values, mid_close.values, mid_high_raw.values]),
            "low": np.minimum.reduce([mid_open.values, mid_close.values, mid_low_raw.values]),
            "volume": (merged["volume_bid"] + merged["volume_ask"]) / 2.0,
            "source": "dukascopy:datafeed.dukascopy.com:daily_1m_candles_mid",
        }
    )

    return quotes, ohlc


def download_dukascopy(config: dict[str, Any], force: bool = False, no_csv: bool = False) -> dict[str, Any]:
    c = config.get("xauusd", {})
    symbol = c.get("symbol", "XAUUSD")
    price_scale = float(c.get("price_scale", 1000))
    start_ts = parse_utc_ts(c.get("start_ts_utc", "2019-01-01T00:00:00Z"))
    end_ts = parse_utc_ts(c.get("end_ts_utc", "2025-12-31T23:59:59Z"))

    raw_root = Path(c.get("raw_dir", "research/data_cache/xauusd/dukascopy/raw"))
    normalized_dir = Path(c.get("normalized_dir", "research/data_cache/xauusd/dukascopy/normalized"))
    quotes_path = Path(c.get("quotes_path", normalized_dir / "xauusd_quotes_1m.parquet"))
    canonical_1m_path = Path(c.get("canonical_1m_path", normalized_dir / "xauusd_1m.parquet"))
    index_path = Path(c.get("download_index_path", raw_root / "download_index.json"))
    timeout_sec = int(c.get("request_timeout_sec", 45))
    max_workers = int(c.get("max_workers", 24))
    single_side_mode = str(c.get("single_side_mode", "fail")).lower()

    if single_side_mode not in {"fail", "allow"}:
        raise ValueError("xauusd.single_side_mode must be 'fail' or 'allow'")

    ensure_dir(raw_root)
    ensure_dir(normalized_dir)

    index = _load_index(index_path)
    days = [pd.Timestamp(d).tz_convert("UTC").floor("d") for d in day_range(str(start_ts), str(end_ts))]
    if not days:
        raise RuntimeError("No days to download for configured XAUUSD date range")

    downloaded_files: list[str] = []
    reused_files: list[str] = []

    # Stage 1: concurrent download/reuse resolution for all day/side files.
    status_by_day_side: dict[str, dict[str, str]] = {}
    path_by_day_side: dict[str, dict[str, Path]] = {}
    download_tasks: list[tuple[pd.Timestamp, str, Path, str]] = []

    for day_ts in days:
        day_key = _day_key(day_ts)
        status_by_day_side.setdefault(day_key, {})
        path_by_day_side.setdefault(day_key, {})

        for side in SIDES:
            path = _daily_path(raw_root, symbol, day_ts, side)
            url = _daily_url(symbol, day_ts, side)
            path_by_day_side[day_key][side] = path

            existing = path.exists() and path.stat().st_size > 0
            if existing and not force:
                status_by_day_side[day_key][side] = "reused"
                reused_files.append(str(path))
                _update_side_index(index, day_ts, side, "reused", path, url)
            else:
                download_tasks.append((day_ts, side, path, url))

    if download_tasks:
        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_to_task = {
                    ex.submit(_download_file, url, path, timeout_sec): (day_ts, side, path, url)
                    for day_ts, side, path, url in download_tasks
                }
                for fut in as_completed(fut_to_task):
                    day_ts, side, path, url = fut_to_task[fut]
                    day_key = _day_key(day_ts)
                    status = fut.result()

                    if status == 404:
                        status_by_day_side[day_key][side] = "missing_404"
                        _update_side_index(index, day_ts, side, "missing_404", path, url)
                    else:
                        status_by_day_side[day_key][side] = "downloaded"
                        downloaded_files.append(str(path))
                        _update_side_index(index, day_ts, side, "downloaded", path, url)

                    completed += 1
                    if completed % 500 == 0:
                        _save_index(index_path, index)
        finally:
            _save_index(index_path, index)
    else:
        _save_index(index_path, index)

    # Stage 2: deterministic sequential decode + parquet write.
    quotes_writer: pq.ParquetWriter | None = None
    ohlc_writer: pq.ParquetWriter | None = None

    no_data_days: list[str] = []
    single_side_days: list[str] = []

    row_count_quotes = 0
    row_count_ohlc = 0
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None

    try:
        for day_ts in days:
            day_key = _day_key(day_ts)
            bid_status = status_by_day_side.get(day_key, {}).get("BID", "missing_404")
            ask_status = status_by_day_side.get(day_key, {}).get("ASK", "missing_404")
            bid_path = path_by_day_side[day_key]["BID"]
            ask_path = path_by_day_side[day_key]["ASK"]

            bid_missing = bid_status == "missing_404" or not (bid_path.exists() and bid_path.stat().st_size > 0)
            ask_missing = ask_status == "missing_404" or not (ask_path.exists() and ask_path.stat().st_size > 0)

            if bid_missing and ask_missing:
                no_data_days.append(day_key)
                continue

            if bid_missing != ask_missing:
                single_side_days.append(day_key)

            bid_df = _decode_daily_candles(bid_path.read_bytes() if bid_path.exists() else b"", day_ts, price_scale)
            ask_df = _decode_daily_candles(ask_path.read_bytes() if ask_path.exists() else b"", day_ts, price_scale)

            quotes_day, ohlc_day = _merge_bid_ask_to_mid(bid_df, ask_df, single_side_mode=single_side_mode)
            if quotes_day.empty or ohlc_day.empty:
                no_data_days.append(day_key)
                continue

            quotes_day = quotes_day[(quotes_day["ts_utc"] >= start_ts) & (quotes_day["ts_utc"] <= end_ts)]
            ohlc_day = ohlc_day[(ohlc_day["ts_utc"] >= start_ts) & (ohlc_day["ts_utc"] <= end_ts)]
            if quotes_day.empty or ohlc_day.empty:
                continue

            quotes_day = quotes_day.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="first")
            ohlc_day = ohlc_day.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="first")

            if last_ts is not None:
                quotes_day = quotes_day[quotes_day["ts_utc"] > last_ts]
                ohlc_day = ohlc_day[ohlc_day["ts_utc"] > last_ts]
            if quotes_day.empty or ohlc_day.empty:
                continue

            bid_gt_ask = int((quotes_day["bid"] > quotes_day["ask"]).sum())
            if bid_gt_ask > 0:
                raise RuntimeError(f"Found bid > ask on {day_key} in {bid_gt_ask} rows")

            q_table = pa.Table.from_pandas(quotes_day, preserve_index=False)
            o_table = pa.Table.from_pandas(ohlc_day, preserve_index=False)

            if quotes_writer is None:
                quotes_writer = pq.ParquetWriter(str(quotes_path), q_table.schema, compression="snappy")
            if ohlc_writer is None:
                ohlc_writer = pq.ParquetWriter(str(canonical_1m_path), o_table.schema, compression="snappy")

            quotes_writer.write_table(q_table)
            ohlc_writer.write_table(o_table)

            row_count_quotes += int(len(quotes_day))
            row_count_ohlc += int(len(ohlc_day))

            if first_ts is None:
                first_ts = pd.to_datetime(ohlc_day["ts_utc"].iloc[0], utc=True)
            last_ts = pd.to_datetime(ohlc_day["ts_utc"].iloc[-1], utc=True)

        if quotes_writer is None or ohlc_writer is None or row_count_ohlc == 0:
            raise RuntimeError(
                "Dukascopy 1m candle build produced zero rows. "
                "If endpoints are inaccessible, provide an alternative source export."
            )
    finally:
        if quotes_writer is not None:
            quotes_writer.close()
        if ohlc_writer is not None:
            ohlc_writer.close()

    if not no_csv:
        pd.read_parquet(quotes_path).to_csv(quotes_path.with_suffix(".csv"), index=False)
        pd.read_parquet(canonical_1m_path).to_csv(canonical_1m_path.with_suffix(".csv"), index=False)

    stale_tick_files = [
        normalized_dir / "xauusd_ticks.parquet",
        normalized_dir / "xauusd_ticks.csv",
    ]
    for fp in stale_tick_files:
        if fp.exists():
            fp.unlink()

    quotes_val = validate_file(quotes_path, kind="ticks", expected_seconds=60)
    ohlc_val = validate_file(canonical_1m_path, kind="ohlc", expected_seconds=60)

    if not quotes_val["ok"]:
        raise RuntimeError(f"XAUUSD 1m quote validation failed: {quotes_val['errors']}")
    if not ohlc_val["ok"]:
        raise RuntimeError(f"XAUUSD 1m OHLC validation failed: {ohlc_val['errors']}")

    summary: dict[str, Any] = {
        "source": "dukascopy",
        "symbol": symbol,
        "timeframe": "1m",
        "requested_start_ts_utc": to_iso_z(start_ts),
        "requested_end_ts_utc": to_iso_z(end_ts),
        "download_index": str(index_path),
        "days_requested": len(days),
        "days_no_data": len(set(no_data_days)),
        "single_side_days": sorted(set(single_side_days)),
        "downloaded_files": sorted(set(downloaded_files)),
        "reused_files": sorted(set(reused_files)),
        "quotes_1m_path": str(quotes_path),
        "canonical_1m_path": str(canonical_1m_path),
        "row_count_quotes_1m": row_count_quotes,
        "row_count_ohlc_1m": row_count_ohlc,
        "start_ts_utc": to_iso_z(first_ts),
        "end_ts_utc": to_iso_z(last_ts),
        "validation": {
            "quotes_1m": quotes_val,
            "ohlc_1m": ohlc_val,
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Dukascopy XAUUSD 1m BID/ASK daily candles and normalize")
    parser.add_argument("--config", default="research/config_data.yml", help="Config YAML path")
    parser.add_argument("--force", action="store_true", help="Redownload raw files even when present")
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV companions")
    parser.add_argument(
        "--summary-out",
        default="research/artefacts/logs/xauusd_dukascopy_summary.json",
        help="Summary JSON output path",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    summary = download_dukascopy(config, force=args.force, no_csv=args.no_csv)
    print(json.dumps(summary, indent=2))
    dump_json(args.summary_out, summary)
    print(f"Wrote Dukascopy summary to {args.summary_out}")


if __name__ == "__main__":
    main()
