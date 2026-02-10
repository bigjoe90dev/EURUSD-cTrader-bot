"""
Build a pause mask for backtests based on news_events table.
"""
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from shared.db import get_db_path


def _parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def build_news_pause_mask(ts_utc: Iterable,
                          db_path: str = None) -> pd.Series:
    """
    Return boolean Series aligned to ts_utc (True = pause trading).
    """
    db_path = db_path or str(get_db_path())
    ts = pd.to_datetime(pd.Series(ts_utc), utc=True, errors="coerce")
    mask = pd.Series(False, index=ts.index)

    # Load config
    pause_before = int(os.getenv("NEWS_PAUSE_BEFORE_MIN", "15"))
    pause_after = int(os.getenv("NEWS_PAUSE_AFTER_MIN", "30"))
    currencies = [c.strip().upper() for c in os.getenv("NEWS_CURRENCIES", "EUR,USD").split(",") if c.strip()]
    min_impact = os.getenv("NEWS_MIN_IMPACT", "high").strip().lower()

    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT event_name, currency, impact, datetime_utc FROM news_events"
        ).fetchall()
        conn.close()
    except Exception:
        return mask  # if no table or error, no pause

    # Filter events
    events = []
    for _, currency, impact, dt in rows:
        if currency and currency.upper() not in currencies:
            continue
        if impact and impact.lower() != min_impact:
            continue
        try:
            ev_time = _parse_time(dt)
            events.append(ev_time)
        except Exception:
            continue

    if not events:
        return mask

    # Build mask: for each event, pause in [t - after, t + before]
    for ev in events:
        start = ev - timedelta(minutes=pause_after)
        end = ev + timedelta(minutes=pause_before)
        mask |= (ts >= start) & (ts <= end)

    return mask.fillna(False)
