from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .common import dump_json, ensure_dir, load_yaml, month_range, parse_utc_ts, to_iso_z
from .validate_dataset import validate_file

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def _latest_full_month() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    if now.month == 1:
        return f"{now.year - 1}-12"
    return f"{now.year:04d}-{now.month - 1:02d}"


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"months": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("months"), dict):
            return data
    except Exception:
        pass
    return {"months": {}}


def _save_index(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _download_zip(url: str, out_path: Path, timeout_sec: int = 120) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout_sec)
    if r.status_code == 404:
        return 404
    if r.status_code != 200:
        raise RuntimeError(f"Download failed {r.status_code} for {url}")
    out_path.write_bytes(r.content)
    return 200


def _load_zip_csv(zip_path: Path) -> pd.DataFrame:
    df = pd.read_csv(zip_path, compression="zip", header=None, names=KLINE_COLUMNS)
    open_time = pd.to_numeric(df["open_time"], errors="coerce")

    if open_time.isna().all():
        raise RuntimeError(f"open_time is non-numeric in {zip_path}")

    max_val = int(open_time.dropna().max())
    unit = "us" if max_val > 10**14 else "ms"
    df["ts_utc"] = pd.to_datetime(open_time, unit=unit, utc=True, errors="coerce")

    out = df[["ts_utc", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _month_zip_path(raw_zip_dir: Path, symbol: str, timeframe: str, month: str) -> Path:
    return raw_zip_dir / f"{symbol}-{timeframe}-{month}.zip"


def _update_month_index(index: dict[str, Any], month: str, status: str, path: Path, url: str) -> None:
    m = index.setdefault("months", {}).setdefault(month, {})
    m.update(
        {
            "status": status,
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "url": url,
            "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def download_binance(
    config: dict[str, Any],
    start_month: str | None = None,
    end_month: str | None = None,
    force: bool = False,
    no_csv: bool = False,
) -> dict[str, Any]:
    c = config.get("btcusdt", {})
    symbol = c.get("symbol", "BTCUSDT")
    timeframe = c.get("timeframe", "1m")
    raw_zip_dir = Path(c.get("raw_zip_dir", "research/data_cache/btcusdt/binance/raw_zips"))
    normalized_path = Path(c.get("canonical_1m_path", "research/data_cache/btcusdt/binance/normalized/btcusdt_1m.parquet"))
    index_path = Path(c.get("download_index_path", raw_zip_dir / "download_index.json"))

    start_m = start_month or c.get("start_month", "2019-01")
    conf_end = end_month or c.get("end_month", "auto_latest_full_month")
    end_m = _latest_full_month() if str(conf_end).lower().startswith("auto") else str(conf_end)

    start_ts = parse_utc_ts(c.get("start_ts_utc", f"{start_m}-01T00:00:00Z"))
    end_ts_raw = c.get("end_ts_utc")
    if end_ts_raw:
        end_ts = parse_utc_ts(end_ts_raw)
    else:
        # End at final minute of selected end month
        end_month_dt = dt.datetime.strptime(end_m, "%Y-%m")
        if end_month_dt.month == 12:
            next_month = end_month_dt.replace(year=end_month_dt.year + 1, month=1)
        else:
            next_month = end_month_dt.replace(month=end_month_dt.month + 1)
        end_ts = pd.Timestamp(next_month, tz="UTC") - pd.Timedelta(minutes=1)

    ensure_dir(raw_zip_dir)
    ensure_dir(normalized_path.parent)

    months = month_range(start_m, end_m)
    index = _load_index(index_path)

    downloaded: list[str] = []
    reused: list[str] = []
    leading_unavailable: list[str] = []
    available_months: list[str] = []
    first_available_seen = False

    for month in months:
        url = f"{BASE_URL}/{symbol}/{timeframe}/{symbol}-{timeframe}-{month}.zip"
        zip_path = _month_zip_path(raw_zip_dir, symbol, timeframe, month)
        entry = index.get("months", {}).get(month, {})
        local_ok = zip_path.exists() and zip_path.stat().st_size > 0

        if local_ok and not force:
            # Accept existing files; if index recorded a different size, refresh by redownload.
            recorded_size = entry.get("size_bytes")
            if recorded_size is not None and int(recorded_size) != int(zip_path.stat().st_size):
                status = _download_zip(url, zip_path)
                if status == 404:
                    if not first_available_seen:
                        leading_unavailable.append(month)
                        _update_month_index(index, month, "missing_404", zip_path, url)
                        continue
                    raise RuntimeError(f"BTCUSDT missing month after coverage started: {month} ({url})")
                downloaded.append(str(zip_path))
                _update_month_index(index, month, "downloaded", zip_path, url)
            else:
                reused.append(str(zip_path))
                _update_month_index(index, month, "reused", zip_path, url)

            available_months.append(month)
            first_available_seen = True
            continue

        status = _download_zip(url, zip_path)
        if status == 404:
            if not first_available_seen:
                leading_unavailable.append(month)
                _update_month_index(index, month, "missing_404", zip_path, url)
                continue
            raise RuntimeError(
                f"BTCUSDT month missing inside requested coverage: {month}. "
                f"Download URL returned 404: {url}"
            )

        downloaded.append(str(zip_path))
        _update_month_index(index, month, "downloaded", zip_path, url)
        available_months.append(month)
        first_available_seen = True

    _save_index(index_path, index)

    if not available_months:
        raise RuntimeError(
            f"No BTCUSDT files available in requested range {start_m}..{end_m}. "
            f"Leading missing months: {leading_unavailable[:10]}"
        )

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None

    try:
        for month in available_months:
            zip_path = _month_zip_path(raw_zip_dir, symbol, timeframe, month)
            df = _load_zip_csv(zip_path)
            df = df.dropna(subset=["ts_utc", "open", "high", "low", "close"])
            df = df.sort_values("ts_utc")
            df = df[(df["ts_utc"] >= start_ts) & (df["ts_utc"] <= end_ts)]
            if df.empty:
                continue

            if last_ts is not None:
                df = df[df["ts_utc"] > last_ts]
            if df.empty:
                continue

            df = df[["ts_utc", "open", "high", "low", "close", "volume"]].copy()
            df["source"] = "binance:data.binance.vision:spot_monthly_klines"

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(normalized_path), table.schema, compression="snappy")

            writer.write_table(table)

            total_rows += int(len(df))
            if first_ts is None:
                first_ts = pd.to_datetime(df["ts_utc"].iloc[0], utc=True)
            last_ts = pd.to_datetime(df["ts_utc"].iloc[-1], utc=True)

        if writer is None or total_rows == 0:
            raise RuntimeError("BTCUSDT normalization produced zero rows after filtering")
    finally:
        if writer is not None:
            writer.close()

    if not no_csv:
        pd.read_parquet(normalized_path).to_csv(normalized_path.with_suffix(".csv"), index=False)

    validation = validate_file(normalized_path, kind="ohlc", expected_seconds=60)
    if not validation["ok"]:
        raise RuntimeError(f"BTCUSDT validation failed: {validation['errors']}")

    summary: dict[str, Any] = {
        "source": "binance",
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_start_month": start_m,
        "requested_end_month": end_m,
        "earliest_available_month_in_request": available_months[0],
        "latest_available_month_in_request": available_months[-1],
        "leading_unavailable_months": leading_unavailable,
        "months_covered": available_months,
        "downloaded_files": downloaded,
        "reused_files": reused,
        "download_index": str(index_path),
        "normalized_path": str(normalized_path),
        "row_count": total_rows,
        "start_ts_utc": to_iso_z(first_ts),
        "end_ts_utc": to_iso_z(last_ts),
        "validation": validation,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize BTCUSDT 1m spot klines from data.binance.vision")
    parser.add_argument("--config", default="research/config_data.yml", help="Config YAML path")
    parser.add_argument("--start-month", help="YYYY-MM")
    parser.add_argument("--end-month", help="YYYY-MM or auto_latest_full_month")
    parser.add_argument("--force", action="store_true", help="Redownload even if local file exists")
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV companion file")
    parser.add_argument(
        "--summary-out",
        default="research/artefacts/logs/btcusdt_binance_summary.json",
        help="Summary JSON output path",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    summary = download_binance(
        config,
        start_month=args.start_month,
        end_month=args.end_month,
        force=args.force,
        no_csv=args.no_csv,
    )
    print(json.dumps(summary, indent=2))
    dump_json(args.summary_out, summary)
    print(f"Wrote Binance summary to {args.summary_out}")


if __name__ == "__main__":
    main()
