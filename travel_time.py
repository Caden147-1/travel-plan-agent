from __future__ import annotations

import os
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Optional

import requests


DEFAULT_TIME_API_URL = "https://worldtimeapi.org/api/timezone/Asia/Shanghai"


def _parse_date_from_payload(payload: Any) -> Optional[date]:
    if not isinstance(payload, dict):
        return None

    for key in ("datetime", "utc_datetime"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) < 10:
            continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except Exception:
            try:
                return datetime.strptime(value[:10], "%Y-%m-%d").date()
            except Exception:
                pass

    value = payload.get("date")
    if isinstance(value, str):
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except Exception:
            return None

    return None


@lru_cache(maxsize=1)
def get_today_date() -> date:
    """
    Get today's date from an external time API.

    Falls back to the local system date only if the external API is unavailable.
    Override the endpoint with TRAVEL_CURRENT_DATE_API_URL if needed.
    """
    url = os.getenv("TRAVEL_CURRENT_DATE_API_URL", DEFAULT_TIME_API_URL)
    try:
        response = requests.get(url, timeout=6)
        response.raise_for_status()
        parsed = _parse_date_from_payload(response.json())
        if parsed:
            return parsed
    except Exception:
        pass

    return datetime.today().date()
