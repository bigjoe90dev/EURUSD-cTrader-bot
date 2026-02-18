from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file is not a mapping: {p}")
    return data


def dump_yaml(path: str | Path, data: dict[str, Any]) -> None:
    p = ensure_parent(Path(path))
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def dump_json(path: str | Path, data: Any) -> None:
    p = ensure_parent(Path(path))
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_utc_ts(value: str | dt.datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Unable to parse timestamp: {value}")
    return ts


def to_iso_z(ts: pd.Timestamp | dt.datetime | str | None) -> str | None:
    if ts is None:
        return None
    x = pd.to_datetime(ts, utc=True)
    if pd.isna(x):
        return None
    return x.strftime("%Y-%m-%dT%H:%M:%SZ")


def month_range(start_month: str, end_month: str) -> list[str]:
    start = dt.datetime.strptime(start_month, "%Y-%m")
    end = dt.datetime.strptime(end_month, "%Y-%m")
    if end < start:
        raise ValueError(f"end_month {end_month} is before start_month {start_month}")
    out: list[str] = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return out


def hour_range(start_ts_utc: str, end_ts_utc: str) -> Iterable[pd.Timestamp]:
    start = parse_utc_ts(start_ts_utc).floor("h")
    end = parse_utc_ts(end_ts_utc).floor("h")
    if end < start:
        raise ValueError(f"end_ts_utc {end_ts_utc} is before start_ts_utc {start_ts_utc}")
    cur = start
    one_hour = pd.Timedelta(hours=1)
    while cur <= end:
        yield cur
        cur += one_hour


def day_range(start_ts_utc: str, end_ts_utc: str) -> Iterable[pd.Timestamp]:
    start = parse_utc_ts(start_ts_utc).floor("d")
    end = parse_utc_ts(end_ts_utc).floor("d")
    if end < start:
        raise ValueError(f"end_ts_utc {end_ts_utc} is before start_ts_utc {start_ts_utc}")
    cur = start
    one_day = pd.Timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one_day


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def list_files(root: str | Path) -> list[str]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.rglob("*") if x.is_file())
