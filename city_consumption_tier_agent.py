from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=False)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
TRAVEL_LLM_MODEL = os.getenv("TRAVEL_LLM_MODEL", "qwen-plus")
DASHSCOPE_CHAT_BASE_URL = os.getenv(
    "DASHSCOPE_CHAT_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

CITY_CONSUMPTION_TIERS = {"low", "medium", "high"}
DEFAULT_CITY_CONSUMPTION_TIER = "medium"
MID_DISTANCE_CAR_RENTAL_COST_CNY = {
    "low": 300,
    "medium": 400,
    "high": 500,
}


def get_city_consumption_tier_backend_status() -> Dict[str, Any]:
    return {
        "provider": "dashscope-compatible",
        "configured": bool(DASHSCOPE_API_KEY),
        "model": TRAVEL_LLM_MODEL,
        "base_url": DASHSCOPE_CHAT_BASE_URL,
        "api_key_present": bool(DASHSCOPE_API_KEY),
    }


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_city_key(city: Any) -> str:
    text = _normalize_text(city).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _safe_int(value: Any, default: int = 1) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(float(str(value).strip().replace(",", "")))
    except Exception:
        return default


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = _normalize_text(text)
    if not text:
        raise RuntimeError("Empty LLM response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError("No JSON object found in LLM response.")

    return json.loads(match.group(0))


def _normalize_city_days(city_days: Any) -> List[Dict[str, Any]]:
    if city_days is None:
        return []
    if isinstance(city_days, dict):
        city_days = [city_days]
    if isinstance(city_days, str):
        city_days = [{"city": city_days, "days": 1}]
    if not isinstance(city_days, list):
        return []

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in city_days:
        if isinstance(item, str):
            city = _normalize_text(item)
            days = 1
        elif isinstance(item, dict):
            city = _normalize_text(
                item.get("city")
                or item.get("destination")
                or item.get("name")
            )
            days = _safe_int(item.get("days"), 1)
        else:
            continue

        if not city:
            continue

        key = _normalize_city_key(city)
        if key in seen:
            for existing in normalized:
                if _normalize_city_key(existing.get("city")) == key:
                    existing["days"] = max(1, _safe_int(existing.get("days"), 1) + max(1, days))
                    break
            continue

        normalized.append({
            "city": city,
            "days": max(1, days),
        })
        seen.add(key)

    return normalized


def _fallback_city_tiers(city_days: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
    return [
        {
            "city": item["city"],
            "tier": DEFAULT_CITY_CONSUMPTION_TIER,
            "confidence": 0.0,
            "reason": reason,
            "source": "fallback",
        }
        for item in city_days
    ]


def _build_city_tier_prompt(city_days: List[Dict[str, Any]], country: Optional[str]) -> str:
    city_payload = [
        {"city": item["city"], "days": item["days"]}
        for item in city_days
    ]
    country_text = country or "unknown / mixed"
    return (
        "You are a global city travel-cost tier classifier. Return JSON only, no markdown.\n"
        "Task: classify each destination city's overall travel consumption tier for rental-car cost estimation "
        "in mid-distance / district-change / short intercity movement.\n"
        "Allowed tiers only: low, medium, high.\n"
        "Scope: global cities, not limited to China.\n"
        "Judging criteria: local tourism cost, city prosperity, rental-car prices, taxi prices, fuel, parking, "
        "congestion cost, and common tourist spending level.\n"
        "low: relatively low travel transport and daily spending.\n"
        "medium: mainstream tourist city with moderate or moderately high cost.\n"
        "high: global top-tier, high-income, or clearly expensive rental/parking/tourist-spending city.\n"
        "Fixed cost mapping: low=300 CNY/day, medium=400 CNY/day, high=500 CNY/day.\n"
        "Classify every destination separately. Do not return one overall tier. "
        "The city field must exactly use the input city name.\n"
        f"Country or region reference: {country_text}\n"
        f"Destinations and stay days: {json.dumps(city_payload, ensure_ascii=False)}\n"
        "Output format: "
        '{"cities":[{"city":"input city name","tier":"low|medium|high","confidence":0.0,"reason":"one short reason"}]}'
    )


def _call_llm_city_tiers(city_days: List[Dict[str, Any]], country: Optional[str] = None) -> List[Dict[str, Any]]:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or QWEN_API_KEY.")

    url = f"{DASHSCOPE_CHAT_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
    }
    payload = {
        "model": TRAVEL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": _build_city_tier_prompt(city_days, country)},
        ],
        "temperature": 0.1,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"DashScope city tier call failed: HTTP {resp.status_code}")

    data = resp.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    parsed = _extract_json_from_text(content)
    cities = parsed.get("cities")
    if not isinstance(cities, list):
        raise RuntimeError("LLM response missing cities list.")

    by_key: Dict[str, Dict[str, Any]] = {}
    for item in cities:
        if not isinstance(item, dict):
            continue
        city = _normalize_text(item.get("city"))
        tier = _normalize_text(item.get("tier")).lower()
        if not city or tier not in CITY_CONSUMPTION_TIERS:
            continue
        by_key[_normalize_city_key(city)] = {
            "city": city,
            "tier": tier,
            "confidence": item.get("confidence", None),
            "reason": _normalize_text(item.get("reason")) or "classified by LLM",
            "source": "llm",
        }

    result: List[Dict[str, Any]] = []
    for item in city_days:
        city = item["city"]
        matched = by_key.get(_normalize_city_key(city))
        if matched:
            matched["city"] = city
            result.append(matched)
        else:
            result.append({
                "city": city,
                "tier": DEFAULT_CITY_CONSUMPTION_TIER,
                "confidence": 0.0,
                "reason": "LLM did not return this city; neutral fallback tier used",
                "source": "fallback_after_llm",
            })

    return result


def classify_city_consumption_tiers(
    city_days: Any,
    country: Optional[str] = None,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    normalized_city_days = _normalize_city_days(city_days)
    if not normalized_city_days:
        return []

    if not use_llm:
        return _fallback_city_tiers(normalized_city_days, "LLM disabled; fallback tier used")

    try:
        return _call_llm_city_tiers(normalized_city_days, country=country)
    except Exception as exc:
        return _fallback_city_tiers(
            normalized_city_days,
            f"LLM unavailable; fallback tier used ({exc})",
        )


def estimate_mid_distance_car_rental(
    city_days: Any,
    country: Optional[str] = None,
    currency: str = "CNY",
    use_llm: bool = True,
) -> Dict[str, Any]:
    normalized_city_days = _normalize_city_days(city_days)
    tiers = classify_city_consumption_tiers(
        normalized_city_days,
        country=country,
        use_llm=use_llm,
    )
    tier_by_city = {
        _normalize_city_key(item.get("city")): item
        for item in tiers
        if isinstance(item, dict)
    }

    city_estimates: List[Dict[str, Any]] = []
    total = 0
    for item in normalized_city_days:
        city = item["city"]
        days = max(1, _safe_int(item.get("days"), 1))
        tier_info = tier_by_city.get(_normalize_city_key(city), {})
        tier = _normalize_text(tier_info.get("tier")).lower()
        if tier not in CITY_CONSUMPTION_TIERS:
            tier = DEFAULT_CITY_CONSUMPTION_TIER

        daily_cost = MID_DISTANCE_CAR_RENTAL_COST_CNY[tier]
        city_total = daily_cost * days
        total += city_total
        city_estimates.append({
            "city": city,
            "days": days,
            "consumption_tier": tier,
            "daily_car_rental_cost": daily_cost,
            "total_car_rental_cost": city_total,
            "currency": currency,
            "formula": f"{daily_cost} * {days}",
            "tier_source": tier_info.get("source") or "fallback",
            "tier_confidence": tier_info.get("confidence"),
            "tier_reason": tier_info.get("reason") or "fallback tier used",
        })

    return {
        "transport_mode": "rental_car",
        "scenario": "mid_distance_city_or_district_transfer",
        "currency": currency,
        "cost_rule": {
            "low": MID_DISTANCE_CAR_RENTAL_COST_CNY["low"],
            "medium": MID_DISTANCE_CAR_RENTAL_COST_CNY["medium"],
            "high": MID_DISTANCE_CAR_RENTAL_COST_CNY["high"],
            "unit": "CNY/day",
        },
        "city_estimates": city_estimates,
        "total_car_rental_cost": total,
    }


if __name__ == "__main__":
    sample = [
        {"city": "Tokyo", "days": 3},
        {"city": "Kyoto", "days": 2},
    ]
    print(json.dumps(estimate_mid_distance_car_rental(sample), ensure_ascii=False, indent=2))
