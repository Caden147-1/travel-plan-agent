from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

print(">>> NEW travel_budget_agent.py loaded <<<")


# =========================================================
# Backend
# =========================================================

def get_budget_agent_backend_status() -> dict:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("TRAVEL_LLM_MODEL", "qwen-plus")
    base_url = os.getenv(
        "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    return {
        "provider": "dashscope-compatible",
        "configured": bool(api_key),
        "model": model,
        "base_url": base_url,
        "api_key_present": bool(api_key),
    }


# =========================================================
# Result template
# =========================================================


def _default_result() -> Dict[str, Any]:
    return {
        "origin": None,
        "destination": [],
        "country": None,
        "num_people": None,
        "travel_dates": {
            "start": None,
            "end": None,
            "text": None
        },
        "duration_days": None,
        "budget": {
            "total": None,
            "currency": "CNY"
        },
        "preferences": {
            "flight": None,
            "hotel": None,
            "food": None,
            "attraction_style": None,
            "hotel_level": None,
            "hotel_features": [],
            "food_preferences": [],
            "attraction_styles": []
        },
        "planning_signals": {
            "flight_priority": None,
            "hotel_level": None,
            "food_priority": None,
            "attraction_intensity": None,
            "hotel_priority": None
        },

        "allocation_basis": {
            "model": "rule_based",
            "destination_profile": None,
            "destination_profile_tier": None,
            "effective_cities": [],
            "city_plan_mode": None,
            "llm_adjustments": [],
            "preference_impact": [],
            "llm_ratio": None,
            "baseline_ratio": None,
            "final_ratio": None,
            "season_factor": None,
            "cost_level_sources": {
                "llm_cost_level": None,
                "profile_cost_level": None,
                "final_cost_level": None
            }
        },

        "estimated_cost": {},
        "optional_upgrades": {},
        "budget_plan": {},
        "city_budget_plan": [],
        "budget_strategy": None,
        "cost_level": None,
        "budget_status": None,
        "remaining_budget": 0,
        "unallocated_budget": 0,
        "normalized_preferences": {},

        "meta": {
            "error": None
        }
    }

# =========================================================
# Utilities
# =========================================================

def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(float(str(value).strip().replace(",", "")))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except Exception:
        return default


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError("No JSON object found in LLM response.")

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON from LLM response: {exc}") from exc


def _normalize_assumptions(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = _normalize_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_adjustments(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = _normalize_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_preference_impact(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []

    cleaned: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            "preference": _normalize_text(item.get("preference")) or "",
            "affected_category": _normalize_text(item.get("affected_category")) or "",
            "direction": _normalize_text(item.get("direction")) or "",
            "magnitude": _normalize_text(item.get("magnitude")) or "",
            "reason": _normalize_text(item.get("reason")) or "",
        })
    return cleaned


def _ensure_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _round_to_100(value: float) -> int:
    return int(round(value / 100.0) * 100)


def _clamp_ratio(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_ratio_map(ratio_map: Dict[str, float]) -> Dict[str, float]:
    normalized = dict(ratio_map)
    total = sum(max(0.0, v) for v in normalized.values())
    if total <= 0:
        return {
            "transport": 0.30,
            "hotel": 0.35,
            "food": 0.15,
            "attractions": 0.12,
            "buffer": 0.08
        }
    return {k: round(max(0.0, v) / total, 4) for k, v in normalized.items()}


def _normalize_plan_to_total(plan: Dict[str, int], total_budget: int) -> Dict[str, int]:
    normalized = {k: max(0, int(v)) for k, v in plan.items()}

    for key in ["transport", "hotel", "food", "attractions", "buffer"]:
        normalized.setdefault(key, 0)

    current_total = sum(normalized.values())

    if current_total <= total_budget:
        normalized["buffer"] += total_budget - current_total
        return normalized

    shortage = current_total - total_budget

    for key in ["buffer", "attractions", "food", "hotel", "transport"]:
        if shortage <= 0:
            break
        reducible = min(normalized.get(key, 0), shortage)
        normalized[key] -= reducible
        shortage -= reducible

    final_total = sum(normalized.values())
    if final_total > total_budget:
        overflow = final_total - total_budget
        normalized["transport"] = max(0, normalized.get("transport", 0) - overflow)

    return normalized


# =========================================================
# City plan reader (Agent2 reads Agent1 output only)
# =========================================================

def _get_duration_days(parsed_request: Dict[str, Any]) -> int:
    return max(1, _safe_int(parsed_request.get("duration_days"), 1))


def _get_num_people(parsed_request: Dict[str, Any]) -> int:
    return max(1, _safe_int(parsed_request.get("num_people"), 1))


def _get_total_budget(parsed_request: Dict[str, Any]) -> int:
    return max(0, _safe_int(parsed_request.get("budget", {}).get("total"), 0))


def _is_budget_cap_mode(parsed_request: Dict[str, Any]) -> bool:
    signals = parsed_request.get("planning_signals", {}) or {}
    return bool(parsed_request.get("budget_cap_mode")) or signals.get("budget_discipline") == "cap"


def _get_effective_cities(parsed_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Agent2 must not generate or recommend cities by itself.
    It only reads the city_plan produced by Agent1 / destination.py.
    """
    city_plan = parsed_request.get("city_plan", {}) or {}
    planned_cities = city_plan.get("cities", [])

    if not isinstance(planned_cities, list):
        return []

    result: List[Dict[str, Any]] = []
    for item in planned_cities:
        if not isinstance(item, dict):
            continue

        city_name = _normalize_text(item.get("city"))
        if not city_name:
            continue

        days = max(1, _safe_int(item.get("days"), 0))
        source = _normalize_text(item.get("source")) or "agent1"

        result.append({
            "city": city_name,
            "days": days,
            "source": source
        })

    return result


def _get_effective_primary_destination(parsed_request: Dict[str, Any]) -> str:
    cities = _get_effective_cities(parsed_request)
    if cities:
        return (cities[0].get("city") or "").strip().lower()

    country = _normalize_text(parsed_request.get("country"))
    return (country or "").strip().lower()


def _get_city_plan_mode(parsed_request: Dict[str, Any]) -> str:
    city_plan = parsed_request.get("city_plan", {}) or {}
    mode = _normalize_text(city_plan.get("mode"))
    return mode or "unknown"


# =========================================================
# Profiles (Three-layer structure)
# =========================================================

DEFAULT_COST_PROFILE = {
    "cost_level": "medium",
    "hotel_budget_per_night": 320,
    "hotel_standard_per_night": 580,
    "hotel_premium_per_night": 1050,
    "food_budget_per_day": 75,
    "food_standard_per_day": 140,
    "food_premium_per_day": 250,
    "attraction_light_per_day": 55,
    "attraction_medium_per_day": 130,
    "attraction_heavy_per_day": 230,
    "transport_base_per_person": 2000,
    "weights": {
        "hotel": 1.0,
        "food": 1.0,
        "attractions": 1.0
    }
}

COUNTRY_COST_PROFILE = {
    "japan": {
        "cost_level": "high",
        "hotel_budget_per_night": 430,
        "hotel_standard_per_night": 720,
        "hotel_premium_per_night": 1250,
        "food_budget_per_day": 95,
        "food_standard_per_day": 175,
        "food_premium_per_day": 310,
        "attraction_light_per_day": 75,
        "attraction_medium_per_day": 165,
        "attraction_heavy_per_day": 285,
        "transport_base_per_person": 2400,
        "weights": {"hotel": 1.15, "food": 1.10, "attractions": 1.05}
    },
    "south korea": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 340,
        "hotel_standard_per_night": 590,
        "hotel_premium_per_night": 980,
        "food_budget_per_day": 80,
        "food_standard_per_day": 145,
        "food_premium_per_day": 250,
        "attraction_light_per_day": 60,
        "attraction_medium_per_day": 135,
        "attraction_heavy_per_day": 230,
        "transport_base_per_person": 1900,
        "weights": {"hotel": 1.05, "food": 1.05, "attractions": 1.0}
    },
    "korea": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 340,
        "hotel_standard_per_night": 590,
        "hotel_premium_per_night": 980,
        "food_budget_per_day": 80,
        "food_standard_per_day": 145,
        "food_premium_per_day": 250,
        "attraction_light_per_day": 60,
        "attraction_medium_per_day": 135,
        "attraction_heavy_per_day": 230,
        "transport_base_per_person": 1900,
        "weights": {"hotel": 1.05, "food": 1.05, "attractions": 1.0}
    },
    "thailand": {
        "cost_level": "low",
        "hotel_budget_per_night": 190,
        "hotel_standard_per_night": 360,
        "hotel_premium_per_night": 860,
        "food_budget_per_day": 55,
        "food_standard_per_day": 105,
        "food_premium_per_day": 195,
        "attraction_light_per_day": 45,
        "attraction_medium_per_day": 95,
        "attraction_heavy_per_day": 165,
        "transport_base_per_person": 1500,
        "weights": {"hotel": 0.95, "food": 0.95, "attractions": 1.0}
    },
    "malaysia": {
        "cost_level": "low",
        "hotel_budget_per_night": 170,
        "hotel_standard_per_night": 330,
        "hotel_premium_per_night": 720,
        "food_budget_per_day": 50,
        "food_standard_per_day": 95,
        "food_premium_per_day": 175,
        "attraction_light_per_day": 40,
        "attraction_medium_per_day": 85,
        "attraction_heavy_per_day": 150,
        "transport_base_per_person": 1300,
        "weights": {"hotel": 0.9, "food": 0.9, "attractions": 0.95}
    },
    "singapore": {
        "cost_level": "high",
        "hotel_budget_per_night": 540,
        "hotel_standard_per_night": 900,
        "hotel_premium_per_night": 1550,
        "food_budget_per_day": 120,
        "food_standard_per_day": 220,
        "food_premium_per_day": 360,
        "attraction_light_per_day": 90,
        "attraction_medium_per_day": 180,
        "attraction_heavy_per_day": 300,
        "transport_base_per_person": 1200,
        "weights": {"hotel": 1.25, "food": 1.20, "attractions": 1.10}
    },
    "france": {
        "cost_level": "high",
        "hotel_budget_per_night": 540,
        "hotel_standard_per_night": 880,
        "hotel_premium_per_night": 1650,
        "food_budget_per_day": 120,
        "food_standard_per_day": 225,
        "food_premium_per_day": 380,
        "attraction_light_per_day": 95,
        "attraction_medium_per_day": 210,
        "attraction_heavy_per_day": 345,
        "transport_base_per_person": 4300,
        "weights": {"hotel": 1.20, "food": 1.15, "attractions": 1.10}
    },
    "italy": {
        "cost_level": "high",
        "hotel_budget_per_night": 430,
        "hotel_standard_per_night": 760,
        "hotel_premium_per_night": 1350,
        "food_budget_per_day": 105,
        "food_standard_per_day": 190,
        "food_premium_per_day": 325,
        "attraction_light_per_day": 85,
        "attraction_medium_per_day": 180,
        "attraction_heavy_per_day": 305,
        "transport_base_per_person": 4100,
        "weights": {"hotel": 1.10, "food": 1.05, "attractions": 1.10}
    },
    "united kingdom": {
        "cost_level": "high",
        "hotel_budget_per_night": 590,
        "hotel_standard_per_night": 980,
        "hotel_premium_per_night": 1780,
        "food_budget_per_day": 130,
        "food_standard_per_day": 235,
        "food_premium_per_day": 390,
        "attraction_light_per_day": 100,
        "attraction_medium_per_day": 220,
        "attraction_heavy_per_day": 360,
        "transport_base_per_person": 4500,
        "weights": {"hotel": 1.25, "food": 1.15, "attractions": 1.10}
    },
    "uk": {
        "cost_level": "high",
        "hotel_budget_per_night": 590,
        "hotel_standard_per_night": 980,
        "hotel_premium_per_night": 1780,
        "food_budget_per_day": 130,
        "food_standard_per_day": 235,
        "food_premium_per_day": 390,
        "attraction_light_per_day": 100,
        "attraction_medium_per_day": 220,
        "attraction_heavy_per_day": 360,
        "transport_base_per_person": 4500,
        "weights": {"hotel": 1.25, "food": 1.15, "attractions": 1.10}
    },
    "united states": {
        "cost_level": "high",
        "hotel_budget_per_night": 700,
        "hotel_standard_per_night": 1180,
        "hotel_premium_per_night": 2100,
        "food_budget_per_day": 145,
        "food_standard_per_day": 265,
        "food_premium_per_day": 430,
        "attraction_light_per_day": 115,
        "attraction_medium_per_day": 250,
        "attraction_heavy_per_day": 400,
        "transport_base_per_person": 5000,
        "weights": {"hotel": 1.35, "food": 1.20, "attractions": 1.10}
    },
    "usa": {
        "cost_level": "high",
        "hotel_budget_per_night": 700,
        "hotel_standard_per_night": 1180,
        "hotel_premium_per_night": 2100,
        "food_budget_per_day": 145,
        "food_standard_per_day": 265,
        "food_premium_per_day": 430,
        "attraction_light_per_day": 115,
        "attraction_medium_per_day": 250,
        "attraction_heavy_per_day": 400,
        "transport_base_per_person": 5000,
        "weights": {"hotel": 1.35, "food": 1.20, "attractions": 1.10}
    },
    "australia": {
        "cost_level": "high",
        "hotel_budget_per_night": 460,
        "hotel_standard_per_night": 790,
        "hotel_premium_per_night": 1420,
        "food_budget_per_day": 110,
        "food_standard_per_day": 205,
        "food_premium_per_day": 340,
        "attraction_light_per_day": 85,
        "attraction_medium_per_day": 180,
        "attraction_heavy_per_day": 305,
        "transport_base_per_person": 3400,
        "weights": {"hotel": 1.10, "food": 1.10, "attractions": 1.05}
    },
}

CITY_COST_PROFILE = {
    # Japan
    "tokyo": {
        "cost_level": "high",
        "hotel_budget_per_night": 500,
        "hotel_standard_per_night": 800,
        "hotel_premium_per_night": 1400,
        "food_budget_per_day": 100,
        "food_standard_per_day": 180,
        "food_premium_per_day": 320,
        "attraction_light_per_day": 80,
        "attraction_medium_per_day": 180,
        "attraction_heavy_per_day": 300,
        "transport_base_per_person": 2600,
        "weights": {"hotel": 1.25, "food": 1.15, "attractions": 1.05}
    },
    "osaka": {
        "cost_level": "high",
        "hotel_budget_per_night": 450,
        "hotel_standard_per_night": 700,
        "hotel_premium_per_night": 1200,
        "food_budget_per_day": 90,
        "food_standard_per_day": 170,
        "food_premium_per_day": 300,
        "attraction_light_per_day": 70,
        "attraction_medium_per_day": 160,
        "attraction_heavy_per_day": 280,
        "transport_base_per_person": 2400,
        "weights": {"hotel": 1.05, "food": 1.05, "attractions": 1.0}
    },
    "kyoto": {
        "cost_level": "high",
        "hotel_budget_per_night": 460,
        "hotel_standard_per_night": 760,
        "hotel_premium_per_night": 1280,
        "food_budget_per_day": 92,
        "food_standard_per_day": 172,
        "food_premium_per_day": 305,
        "attraction_light_per_day": 85,
        "attraction_medium_per_day": 175,
        "attraction_heavy_per_day": 310,
        "transport_base_per_person": 2400,
        "weights": {"hotel": 1.10, "food": 1.00, "attractions": 1.15}
    },
    "sapporo": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 380,
        "hotel_standard_per_night": 650,
        "hotel_premium_per_night": 1100,
        "food_budget_per_day": 85,
        "food_standard_per_day": 160,
        "food_premium_per_day": 280,
        "attraction_light_per_day": 65,
        "attraction_medium_per_day": 145,
        "attraction_heavy_per_day": 250,
        "transport_base_per_person": 2350,
        "weights": {"hotel": 0.95, "food": 1.00, "attractions": 0.95}
    },
    "fukuoka": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 350,
        "hotel_standard_per_night": 620,
        "hotel_premium_per_night": 1080,
        "food_budget_per_day": 82,
        "food_standard_per_day": 155,
        "food_premium_per_day": 275,
        "attraction_light_per_day": 60,
        "attraction_medium_per_day": 140,
        "attraction_heavy_per_day": 245,
        "transport_base_per_person": 2250,
        "weights": {"hotel": 0.92, "food": 0.98, "attractions": 0.92}
    },
    "nagoya": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 340,
        "hotel_standard_per_night": 600,
        "hotel_premium_per_night": 1020,
        "food_budget_per_day": 80,
        "food_standard_per_day": 150,
        "food_premium_per_day": 265,
        "attraction_light_per_day": 58,
        "attraction_medium_per_day": 135,
        "attraction_heavy_per_day": 235,
        "transport_base_per_person": 2200,
        "weights": {"hotel": 0.90, "food": 0.95, "attractions": 0.90}
    },

    # Korea
    "seoul": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 380,
        "hotel_standard_per_night": 650,
        "hotel_premium_per_night": 1100,
        "food_budget_per_day": 85,
        "food_standard_per_day": 150,
        "food_premium_per_day": 260,
        "attraction_light_per_day": 60,
        "attraction_medium_per_day": 140,
        "attraction_heavy_per_day": 240,
        "transport_base_per_person": 2000,
        "weights": {"hotel": 1.08, "food": 1.08, "attractions": 1.00}
    },
    "busan": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 320,
        "hotel_standard_per_night": 520,
        "hotel_premium_per_night": 900,
        "food_budget_per_day": 75,
        "food_standard_per_day": 135,
        "food_premium_per_day": 230,
        "attraction_light_per_day": 55,
        "attraction_medium_per_day": 120,
        "attraction_heavy_per_day": 210,
        "transport_base_per_person": 1800,
        "weights": {"hotel": 0.92, "food": 0.95, "attractions": 0.95}
    },
    "jeju": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 330,
        "hotel_standard_per_night": 560,
        "hotel_premium_per_night": 980,
        "food_budget_per_day": 78,
        "food_standard_per_day": 138,
        "food_premium_per_day": 240,
        "attraction_light_per_day": 62,
        "attraction_medium_per_day": 135,
        "attraction_heavy_per_day": 225,
        "transport_base_per_person": 1850,
        "weights": {"hotel": 0.95, "food": 0.95, "attractions": 1.08}
    },

    # Southeast Asia
    "bangkok": {
        "cost_level": "medium",
        "hotel_budget_per_night": 220,
        "hotel_standard_per_night": 420,
        "hotel_premium_per_night": 1150,
        "food_budget_per_day": 60,
        "food_standard_per_day": 120,
        "food_premium_per_day": 240,
        "attraction_light_per_day": 50,
        "attraction_medium_per_day": 110,
        "attraction_heavy_per_day": 180,
        "transport_base_per_person": 1700,
        "weights": {"hotel": 1.05, "food": 1.03, "attractions": 1.00}
    },
    "chiang mai": {
        "cost_level": "low",
        "hotel_budget_per_night": 160,
        "hotel_standard_per_night": 300,
        "hotel_premium_per_night": 850,
        "food_budget_per_day": 45,
        "food_standard_per_day": 95,
        "food_premium_per_day": 180,
        "attraction_light_per_day": 40,
        "attraction_medium_per_day": 90,
        "attraction_heavy_per_day": 150,
        "transport_base_per_person": 300,
        "weights": {"hotel": 0.82, "food": 0.88, "attractions": 0.95}
    },
    "phuket": {
        "cost_level": "medium",
        "hotel_budget_per_night": 240,
        "hotel_standard_per_night": 450,
        "hotel_premium_per_night": 1250,
        "food_budget_per_day": 65,
        "food_standard_per_day": 125,
        "food_premium_per_day": 245,
        "attraction_light_per_day": 55,
        "attraction_medium_per_day": 120,
        "attraction_heavy_per_day": 195,
        "transport_base_per_person": 450,
        "weights": {"hotel": 1.10, "food": 1.00, "attractions": 1.12}
    },
    "pattaya": {
        "cost_level": "low",
        "hotel_budget_per_night": 170,
        "hotel_standard_per_night": 320,
        "hotel_premium_per_night": 900,
        "food_budget_per_day": 48,
        "food_standard_per_day": 98,
        "food_premium_per_day": 185,
        "attraction_light_per_day": 45,
        "attraction_medium_per_day": 95,
        "attraction_heavy_per_day": 155,
        "transport_base_per_person": 350,
        "weights": {"hotel": 0.88, "food": 0.90, "attractions": 0.90}
    },
    "kuala lumpur": {
        "cost_level": "low",
        "hotel_budget_per_night": 160,
        "hotel_standard_per_night": 320,
        "hotel_premium_per_night": 650,
        "food_budget_per_day": 45,
        "food_standard_per_day": 90,
        "food_premium_per_day": 160,
        "attraction_light_per_day": 35,
        "attraction_medium_per_day": 80,
        "attraction_heavy_per_day": 150,
        "transport_base_per_person": 1300,
        "weights": {"hotel": 0.90, "food": 0.90, "attractions": 0.95}
    },
    "penang": {
        "cost_level": "low",
        "hotel_budget_per_night": 150,
        "hotel_standard_per_night": 290,
        "hotel_premium_per_night": 620,
        "food_budget_per_day": 42,
        "food_standard_per_day": 85,
        "food_premium_per_day": 150,
        "attraction_light_per_day": 34,
        "attraction_medium_per_day": 78,
        "attraction_heavy_per_day": 142,
        "transport_base_per_person": 500,
        "weights": {"hotel": 0.85, "food": 0.92, "attractions": 0.92}
    },
    "singapore": {
        "cost_level": "high",
        "hotel_budget_per_night": 550,
        "hotel_standard_per_night": 900,
        "hotel_premium_per_night": 1600,
        "food_budget_per_day": 120,
        "food_standard_per_day": 220,
        "food_premium_per_day": 360,
        "attraction_light_per_day": 90,
        "attraction_medium_per_day": 180,
        "attraction_heavy_per_day": 300,
        "transport_base_per_person": 1200,
        "weights": {"hotel": 1.25, "food": 1.20, "attractions": 1.10}
    },
    "bali": {
        "cost_level": "medium",
        "hotel_budget_per_night": 220,
        "hotel_standard_per_night": 400,
        "hotel_premium_per_night": 950,
        "food_budget_per_day": 58,
        "food_standard_per_day": 110,
        "food_premium_per_day": 210,
        "attraction_light_per_day": 50,
        "attraction_medium_per_day": 105,
        "attraction_heavy_per_day": 175,
        "transport_base_per_person": 1500,
        "weights": {"hotel": 1.00, "food": 0.95, "attractions": 1.08}
    },
    "jakarta": {
        "cost_level": "low",
        "hotel_budget_per_night": 170,
        "hotel_standard_per_night": 320,
        "hotel_premium_per_night": 700,
        "food_budget_per_day": 45,
        "food_standard_per_day": 90,
        "food_premium_per_day": 165,
        "attraction_light_per_day": 38,
        "attraction_medium_per_day": 82,
        "attraction_heavy_per_day": 145,
        "transport_base_per_person": 1400,
        "weights": {"hotel": 0.90, "food": 0.90, "attractions": 0.88}
    },
    "hanoi": {
        "cost_level": "low",
        "hotel_budget_per_night": 140,
        "hotel_standard_per_night": 270,
        "hotel_premium_per_night": 620,
        "food_budget_per_day": 38,
        "food_standard_per_day": 78,
        "food_premium_per_day": 145,
        "attraction_light_per_day": 32,
        "attraction_medium_per_day": 74,
        "attraction_heavy_per_day": 135,
        "transport_base_per_person": 1200,
        "weights": {"hotel": 0.82, "food": 0.85, "attractions": 0.90}
    },
    "ho chi minh city": {
        "cost_level": "low",
        "hotel_budget_per_night": 150,
        "hotel_standard_per_night": 290,
        "hotel_premium_per_night": 660,
        "food_budget_per_day": 40,
        "food_standard_per_day": 82,
        "food_premium_per_day": 150,
        "attraction_light_per_day": 34,
        "attraction_medium_per_day": 76,
        "attraction_heavy_per_day": 138,
        "transport_base_per_person": 1250,
        "weights": {"hotel": 0.85, "food": 0.88, "attractions": 0.90}
    },
    "da nang": {
        "cost_level": "low",
        "hotel_budget_per_night": 145,
        "hotel_standard_per_night": 275,
        "hotel_premium_per_night": 630,
        "food_budget_per_day": 38,
        "food_standard_per_day": 78,
        "food_premium_per_day": 145,
        "attraction_light_per_day": 34,
        "attraction_medium_per_day": 80,
        "attraction_heavy_per_day": 145,
        "transport_base_per_person": 500,
        "weights": {"hotel": 0.84, "food": 0.85, "attractions": 0.95}
    },

    # Europe
    "paris": {
        "cost_level": "high",
        "hotel_budget_per_night": 600,
        "hotel_standard_per_night": 950,
        "hotel_premium_per_night": 1800,
        "food_budget_per_day": 130,
        "food_standard_per_day": 240,
        "food_premium_per_day": 400,
        "attraction_light_per_day": 100,
        "attraction_medium_per_day": 220,
        "attraction_heavy_per_day": 360,
        "transport_base_per_person": 4500,
        "weights": {"hotel": 1.25, "food": 1.18, "attractions": 1.10}
    },
    "london": {
        "cost_level": "high",
        "hotel_budget_per_night": 650,
        "hotel_standard_per_night": 1050,
        "hotel_premium_per_night": 1900,
        "food_budget_per_day": 135,
        "food_standard_per_day": 245,
        "food_premium_per_day": 405,
        "attraction_light_per_day": 102,
        "attraction_medium_per_day": 225,
        "attraction_heavy_per_day": 365,
        "transport_base_per_person": 4700,
        "weights": {"hotel": 1.28, "food": 1.18, "attractions": 1.08}
    },
    "rome": {
        "cost_level": "high",
        "hotel_budget_per_night": 450,
        "hotel_standard_per_night": 780,
        "hotel_premium_per_night": 1400,
        "food_budget_per_day": 110,
        "food_standard_per_day": 200,
        "food_premium_per_day": 340,
        "attraction_light_per_day": 90,
        "attraction_medium_per_day": 190,
        "attraction_heavy_per_day": 320,
        "transport_base_per_person": 4300,
        "weights": {"hotel": 1.05, "food": 1.02, "attractions": 1.10}
    },
    "milan": {
        "cost_level": "high",
        "hotel_budget_per_night": 470,
        "hotel_standard_per_night": 800,
        "hotel_premium_per_night": 1450,
        "food_budget_per_day": 108,
        "food_standard_per_day": 195,
        "food_premium_per_day": 330,
        "attraction_light_per_day": 82,
        "attraction_medium_per_day": 175,
        "attraction_heavy_per_day": 295,
        "transport_base_per_person": 4250,
        "weights": {"hotel": 1.08, "food": 1.02, "attractions": 0.95}
    },
    "venice": {
        "cost_level": "high",
        "hotel_budget_per_night": 500,
        "hotel_standard_per_night": 840,
        "hotel_premium_per_night": 1520,
        "food_budget_per_day": 112,
        "food_standard_per_day": 200,
        "food_premium_per_day": 335,
        "attraction_light_per_day": 95,
        "attraction_medium_per_day": 200,
        "attraction_heavy_per_day": 330,
        "transport_base_per_person": 4200,
        "weights": {"hotel": 1.12, "food": 1.00, "attractions": 1.12}
    },
    "florence": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 410,
        "hotel_standard_per_night": 700,
        "hotel_premium_per_night": 1280,
        "food_budget_per_day": 100,
        "food_standard_per_day": 185,
        "food_premium_per_day": 315,
        "attraction_light_per_day": 90,
        "attraction_medium_per_day": 190,
        "attraction_heavy_per_day": 315,
        "transport_base_per_person": 4150,
        "weights": {"hotel": 1.00, "food": 1.00, "attractions": 1.08}
    },
    "barcelona": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 420,
        "hotel_standard_per_night": 720,
        "hotel_premium_per_night": 1320,
        "food_budget_per_day": 102,
        "food_standard_per_day": 190,
        "food_premium_per_day": 320,
        "attraction_light_per_day": 88,
        "attraction_medium_per_day": 185,
        "attraction_heavy_per_day": 310,
        "transport_base_per_person": 4050,
        "weights": {"hotel": 1.02, "food": 1.02, "attractions": 1.05}
    },
    "madrid": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 380,
        "hotel_standard_per_night": 660,
        "hotel_premium_per_night": 1220,
        "food_budget_per_day": 95,
        "food_standard_per_day": 180,
        "food_premium_per_day": 305,
        "attraction_light_per_day": 82,
        "attraction_medium_per_day": 175,
        "attraction_heavy_per_day": 295,
        "transport_base_per_person": 3950,
        "weights": {"hotel": 0.95, "food": 0.98, "attractions": 1.00}
    },
    "amsterdam": {
        "cost_level": "high",
        "hotel_budget_per_night": 500,
        "hotel_standard_per_night": 820,
        "hotel_premium_per_night": 1500,
        "food_budget_per_day": 115,
        "food_standard_per_day": 210,
        "food_premium_per_day": 350,
        "attraction_light_per_day": 92,
        "attraction_medium_per_day": 195,
        "attraction_heavy_per_day": 325,
        "transport_base_per_person": 4300,
        "weights": {"hotel": 1.12, "food": 1.08, "attractions": 1.03}
    },
    "zurich": {
        "cost_level": "high",
        "hotel_budget_per_night": 700,
        "hotel_standard_per_night": 1150,
        "hotel_premium_per_night": 2100,
        "food_budget_per_day": 145,
        "food_standard_per_day": 270,
        "food_premium_per_day": 440,
        "attraction_light_per_day": 108,
        "attraction_medium_per_day": 235,
        "attraction_heavy_per_day": 380,
        "transport_base_per_person": 4700,
        "weights": {"hotel": 1.35, "food": 1.20, "attractions": 1.05}
    },
    "geneva": {
        "cost_level": "high",
        "hotel_budget_per_night": 690,
        "hotel_standard_per_night": 1120,
        "hotel_premium_per_night": 2050,
        "food_budget_per_day": 142,
        "food_standard_per_day": 265,
        "food_premium_per_day": 430,
        "attraction_light_per_day": 105,
        "attraction_medium_per_day": 228,
        "attraction_heavy_per_day": 370,
        "transport_base_per_person": 4680,
        "weights": {"hotel": 1.32, "food": 1.18, "attractions": 1.03}
    },
    "vienna": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 390,
        "hotel_standard_per_night": 680,
        "hotel_premium_per_night": 1250,
        "food_budget_per_day": 98,
        "food_standard_per_day": 182,
        "food_premium_per_day": 310,
        "attraction_light_per_day": 86,
        "attraction_medium_per_day": 180,
        "attraction_heavy_per_day": 300,
        "transport_base_per_person": 4050,
        "weights": {"hotel": 0.98, "food": 1.00, "attractions": 1.05}
    },
    "prague": {
        "cost_level": "medium",
        "hotel_budget_per_night": 300,
        "hotel_standard_per_night": 530,
        "hotel_premium_per_night": 980,
        "food_budget_per_day": 75,
        "food_standard_per_day": 145,
        "food_premium_per_day": 255,
        "attraction_light_per_day": 72,
        "attraction_medium_per_day": 155,
        "attraction_heavy_per_day": 260,
        "transport_base_per_person": 3800,
        "weights": {"hotel": 0.85, "food": 0.90, "attractions": 1.00}
    },
    "berlin": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 360,
        "hotel_standard_per_night": 620,
        "hotel_premium_per_night": 1120,
        "food_budget_per_day": 90,
        "food_standard_per_day": 168,
        "food_premium_per_day": 290,
        "attraction_light_per_day": 78,
        "attraction_medium_per_day": 168,
        "attraction_heavy_per_day": 282,
        "transport_base_per_person": 3980,
        "weights": {"hotel": 0.92, "food": 0.95, "attractions": 0.98}
    },
    "munich": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 410,
        "hotel_standard_per_night": 700,
        "hotel_premium_per_night": 1280,
        "food_budget_per_day": 100,
        "food_standard_per_day": 185,
        "food_premium_per_day": 315,
        "attraction_light_per_day": 82,
        "attraction_medium_per_day": 175,
        "attraction_heavy_per_day": 295,
        "transport_base_per_person": 4100,
        "weights": {"hotel": 1.00, "food": 1.00, "attractions": 0.98}
    },
    "athens": {
        "cost_level": "medium",
        "hotel_budget_per_night": 290,
        "hotel_standard_per_night": 520,
        "hotel_premium_per_night": 980,
        "food_budget_per_day": 72,
        "food_standard_per_day": 142,
        "food_premium_per_day": 250,
        "attraction_light_per_day": 75,
        "attraction_medium_per_day": 160,
        "attraction_heavy_per_day": 275,
        "transport_base_per_person": 3900,
        "weights": {"hotel": 0.84, "food": 0.88, "attractions": 1.05}
    },
    "lisbon": {
        "cost_level": "medium",
        "hotel_budget_per_night": 300,
        "hotel_standard_per_night": 540,
        "hotel_premium_per_night": 1020,
        "food_budget_per_day": 75,
        "food_standard_per_day": 145,
        "food_premium_per_day": 255,
        "attraction_light_per_day": 72,
        "attraction_medium_per_day": 155,
        "attraction_heavy_per_day": 268,
        "transport_base_per_person": 3920,
        "weights": {"hotel": 0.86, "food": 0.90, "attractions": 1.00}
    },

    # North America
    "new york": {
        "cost_level": "high",
        "hotel_budget_per_night": 800,
        "hotel_standard_per_night": 1300,
        "hotel_premium_per_night": 2400,
        "food_budget_per_day": 150,
        "food_standard_per_day": 280,
        "food_premium_per_day": 450,
        "attraction_light_per_day": 120,
        "attraction_medium_per_day": 260,
        "attraction_heavy_per_day": 420,
        "transport_base_per_person": 5000,
        "weights": {"hotel": 1.40, "food": 1.20, "attractions": 1.10}
    },
    "los angeles": {
        "cost_level": "high",
        "hotel_budget_per_night": 620,
        "hotel_standard_per_night": 1050,
        "hotel_premium_per_night": 1900,
        "food_budget_per_day": 135,
        "food_standard_per_day": 245,
        "food_premium_per_day": 395,
        "attraction_light_per_day": 108,
        "attraction_medium_per_day": 235,
        "attraction_heavy_per_day": 380,
        "transport_base_per_person": 4850,
        "weights": {"hotel": 1.20, "food": 1.10, "attractions": 1.05}
    },
    "san francisco": {
        "cost_level": "high",
        "hotel_budget_per_night": 680,
        "hotel_standard_per_night": 1120,
        "hotel_premium_per_night": 2050,
        "food_budget_per_day": 140,
        "food_standard_per_day": 255,
        "food_premium_per_day": 410,
        "attraction_light_per_day": 110,
        "attraction_medium_per_day": 240,
        "attraction_heavy_per_day": 390,
        "transport_base_per_person": 4900,
        "weights": {"hotel": 1.28, "food": 1.15, "attractions": 1.05}
    },
    "las vegas": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 430,
        "hotel_standard_per_night": 760,
        "hotel_premium_per_night": 1420,
        "food_budget_per_day": 105,
        "food_standard_per_day": 195,
        "food_premium_per_day": 330,
        "attraction_light_per_day": 95,
        "attraction_medium_per_day": 210,
        "attraction_heavy_per_day": 350,
        "transport_base_per_person": 4700,
        "weights": {"hotel": 1.02, "food": 1.00, "attractions": 1.15}
    },
    "chicago": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 520,
        "hotel_standard_per_night": 870,
        "hotel_premium_per_night": 1600,
        "food_budget_per_day": 120,
        "food_standard_per_day": 220,
        "food_premium_per_day": 360,
        "attraction_light_per_day": 100,
        "attraction_medium_per_day": 220,
        "attraction_heavy_per_day": 360,
        "transport_base_per_person": 4800,
        "weights": {"hotel": 1.10, "food": 1.05, "attractions": 1.00}
    },
    "seattle": {
        "cost_level": "high",
        "hotel_budget_per_night": 560,
        "hotel_standard_per_night": 930,
        "hotel_premium_per_night": 1700,
        "food_budget_per_day": 125,
        "food_standard_per_day": 230,
        "food_premium_per_day": 375,
        "attraction_light_per_day": 102,
        "attraction_medium_per_day": 225,
        "attraction_heavy_per_day": 365,
        "transport_base_per_person": 4820,
        "weights": {"hotel": 1.15, "food": 1.08, "attractions": 1.00}
    },
    "toronto": {
        "cost_level": "high",
        "hotel_budget_per_night": 520,
        "hotel_standard_per_night": 880,
        "hotel_premium_per_night": 1620,
        "food_budget_per_day": 118,
        "food_standard_per_day": 218,
        "food_premium_per_day": 355,
        "attraction_light_per_day": 95,
        "attraction_medium_per_day": 210,
        "attraction_heavy_per_day": 345,
        "transport_base_per_person": 4600,
        "weights": {"hotel": 1.08, "food": 1.05, "attractions": 1.00}
    },
    "vancouver": {
        "cost_level": "high",
        "hotel_budget_per_night": 510,
        "hotel_standard_per_night": 860,
        "hotel_premium_per_night": 1580,
        "food_budget_per_day": 115,
        "food_standard_per_day": 210,
        "food_premium_per_day": 345,
        "attraction_light_per_day": 92,
        "attraction_medium_per_day": 205,
        "attraction_heavy_per_day": 340,
        "transport_base_per_person": 4550,
        "weights": {"hotel": 1.05, "food": 1.02, "attractions": 1.02}
    },

    # Oceania
    "sydney": {
        "cost_level": "high",
        "hotel_budget_per_night": 480,
        "hotel_standard_per_night": 820,
        "hotel_premium_per_night": 1480,
        "food_budget_per_day": 112,
        "food_standard_per_day": 205,
        "food_premium_per_day": 340,
        "attraction_light_per_day": 86,
        "attraction_medium_per_day": 182,
        "attraction_heavy_per_day": 308,
        "transport_base_per_person": 3400,
        "weights": {"hotel": 1.08, "food": 1.08, "attractions": 1.05}
    },
    "melbourne": {
        "cost_level": "high",
        "hotel_budget_per_night": 450,
        "hotel_standard_per_night": 780,
        "hotel_premium_per_night": 1420,
        "food_budget_per_day": 108,
        "food_standard_per_day": 200,
        "food_premium_per_day": 335,
        "attraction_light_per_day": 84,
        "attraction_medium_per_day": 178,
        "attraction_heavy_per_day": 300,
        "transport_base_per_person": 3350,
        "weights": {"hotel": 1.02, "food": 1.08, "attractions": 1.00}
    },
    "brisbane": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 390,
        "hotel_standard_per_night": 700,
        "hotel_premium_per_night": 1280,
        "food_budget_per_day": 95,
        "food_standard_per_day": 182,
        "food_premium_per_day": 310,
        "attraction_light_per_day": 76,
        "attraction_medium_per_day": 165,
        "attraction_heavy_per_day": 280,
        "transport_base_per_person": 3250,
        "weights": {"hotel": 0.95, "food": 0.98, "attractions": 0.98}
    },
    "auckland": {
        "cost_level": "medium_high",
        "hotel_budget_per_night": 380,
        "hotel_standard_per_night": 680,
        "hotel_premium_per_night": 1240,
        "food_budget_per_day": 92,
        "food_standard_per_day": 178,
        "food_premium_per_day": 305,
        "attraction_light_per_day": 74,
        "attraction_medium_per_day": 160,
        "attraction_heavy_per_day": 272,
        "transport_base_per_person": 3200,
        "weights": {"hotel": 0.92, "food": 0.95, "attractions": 0.98}
    },
}


# =========================================================
# Profile lookup helpers
# =========================================================

def _country_aliases(country: Optional[str]) -> List[str]:
    raw = (country or "").strip().lower()
    if not raw:
        return []

    aliases = [raw]
    mapping = {
        "korea": ["korea", "south korea"],
        "south korea": ["south korea", "korea"],
        "usa": ["usa", "united states"],
        "united states": ["united states", "usa"],
        "uk": ["uk", "united kingdom"],
        "united kingdom": ["united kingdom", "uk"],
    }
    return mapping.get(raw, aliases)


def _get_profile_for_city_or_country(city: Optional[str], country: Optional[str]) -> Tuple[Dict[str, Any], str, str]:
    city_key = (city or "").strip().lower()
    if city_key and city_key in CITY_COST_PROFILE:
        return CITY_COST_PROFILE[city_key], city_key, "city"

    for c in _country_aliases(country):
        if c in COUNTRY_COST_PROFILE:
            return COUNTRY_COST_PROFILE[c], c, "country"

    return DEFAULT_COST_PROFILE, "default", "default"


def _get_primary_profile(parsed_request: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    effective_cities = _get_effective_cities(parsed_request)
    country = _normalize_text(parsed_request.get("country"))

    primary_city = None
    if effective_cities:
        primary_city = effective_cities[0].get("city")

    return _get_profile_for_city_or_country(primary_city, country)


def _get_effective_city_profiles(parsed_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    cities = _get_effective_cities(parsed_request)
    country = _normalize_text(parsed_request.get("country"))

    if not cities:
        profile, matched_key, tier = _get_profile_for_city_or_country(None, country)
        return [{
            "city": matched_key,
            "days": _get_duration_days(parsed_request),
            "profile": profile,
            "tier": tier,
            "matched_key": matched_key
        }]

    result = []
    for item in cities:
        city_name = _normalize_text(item.get("city")) or ""
        days = max(1, _safe_int(item.get("days"), 1))
        profile, matched_key, tier = _get_profile_for_city_or_country(city_name, country)
        result.append({
            "city": city_name,
            "days": days,
            "profile": profile,
            "tier": tier,
            "matched_key": matched_key
        })
    return result


def _get_weighted_cost_profile(parsed_request: Dict[str, Any]) -> Dict[str, Any]:
    city_profiles = _get_effective_city_profiles(parsed_request)
    if not city_profiles:
        return DEFAULT_COST_PROFILE

    total_days = sum(item["days"] for item in city_profiles) or 1

    numeric_keys = [
        "hotel_budget_per_night",
        "hotel_standard_per_night",
        "hotel_premium_per_night",
        "food_budget_per_day",
        "food_standard_per_day",
        "food_premium_per_day",
        "attraction_light_per_day",
        "attraction_medium_per_day",
        "attraction_heavy_per_day",
        "transport_base_per_person"
    ]

    weighted: Dict[str, Any] = {}
    for key in numeric_keys:
        weighted[key] = sum(
            _safe_float(item["profile"].get(key), 0.0) * item["days"]
            for item in city_profiles
        ) / total_days

    level_score_map = {"low": 0, "medium": 1, "medium_high": 2, "high": 3}
    avg_level_score = sum(
        level_score_map.get(item["profile"].get("cost_level", "medium"), 1) * item["days"]
        for item in city_profiles
    ) / total_days

    if avg_level_score < 0.5:
        cost_level = "low"
    elif avg_level_score < 1.5:
        cost_level = "medium"
    elif avg_level_score < 2.5:
        cost_level = "medium_high"
    else:
        cost_level = "high"

    weighted["cost_level"] = cost_level
    weighted["weights"] = {"hotel": 1.0, "food": 1.0, "attractions": 1.0}
    return weighted


# =========================================================
# LLM
# =========================================================

def _call_llm(prompt: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("TRAVEL_LLM_MODEL", "qwen-plus")
    base_url = os.getenv(
        "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY in .env")

    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "temperature": 0.05,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a travel budget interpretation assistant. "
                    "Return only one valid JSON object. "
                    "Do not use markdown. "
                    "Interpret user preferences conservatively but realistically. "
                    "You may suggest budget ratios, but the program will still validate and adjust them."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM API request failed: {response.status_code} - {response.text}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected LLM response format: {json.dumps(data, ensure_ascii=False)}"
        )


def _build_prompt(user_request: str) -> str:
    """
    Build a strict prompt for travel requirement extraction.
    Agent1 should return controlled vocabulary directly for downstream agents.
    """
    return f"""
You are a travel requirement extraction agent.

Your task is to extract structured travel requirements from the user's request
and return ONLY one valid JSON object.

Strict rules:
1. Return JSON only.
2. Do not use markdown.
3. Do not wrap the output in triple backticks.
4. If a field is missing, use null.
5. "destination" must always be a JSON array.
6. If only one destination is found, still return it as an array.
7. If the user mentions a country but not a city, put the country in "country" and keep "destination" as [].
8. "budget.total" must be a number if possible.
9. "num_people" must be an integer if possible.
10. "duration_days" must be an integer if possible.
11. Only fill "travel_dates.start" and "travel_dates.end" when the date is explicit enough.
12. If the user gives month/day expressions without a year, you may resolve them to the nearest reasonable future dates relative to today.
13. If a missing year must be resolved, prefer future travel dates rather than past dates.
14. If exact dates still cannot be safely determined, set "start" and "end" to null and keep the original phrase in "travel_dates.text".
15. Do not invent any information not explicitly mentioned by the user, except for resolving a missing year in travel dates to a reasonable future year.
16. Preference values must use controlled vocabulary only. Do not output free-form preference text.

Allowed values:

preferences.flight:
- "direct"
- "cheap"
- "comfort"
- null

preferences.hotel:
- "budget"
- "standard"
- "luxury"
- "central"
- "clean"
- null

preferences.food:
- "local"
- "fine_dining"
- "street_food"
- "mixed"
- null

preferences.attraction_style:
- "historical"
- "nature"
- "shopping"
- "entertainment"
- "art"
- "mixed"
- null

planning_signals.flight_priority:
- "low"
- "medium"
- "high"
- null

planning_signals.hotel_level:
- "low"
- "medium"
- "high"
- null

planning_signals.food_priority:
- "low"
- "medium"
- "high"
- null

planning_signals.attraction_intensity:
- "low"
- "medium"
- "high"
- null

Interpretation examples:
- "prefer direct flights" -> preferences.flight = "direct", planning_signals.flight_priority = "high"
- "cheap flights" -> preferences.flight = "cheap", planning_signals.flight_priority = "low"
- "want a good hotel" / "four star hotel" / "4 star hotel" -> preferences.hotel = "luxury", planning_signals.hotel_level = "high"
- "clean and centrally located hotel" -> preferences.hotel = "central" or "clean", planning_signals.hotel_level = "medium"
- "budget hotel" -> preferences.hotel = "budget", planning_signals.hotel_level = "low"
- "love local food" -> preferences.food = "local", planning_signals.food_priority = "high"
- "street food" -> preferences.food = "street_food", planning_signals.food_priority = "medium"
- "like history" / "museums" / "old towns" -> preferences.attraction_style = "historical"
- "nature" / "mountains" / "parks" -> preferences.attraction_style = "nature"
- "shopping" -> preferences.attraction_style = "shopping"
- "many attractions" / "packed itinerary" -> planning_signals.attraction_intensity = "high"
- "relaxed trip" / "not too many attractions" -> planning_signals.attraction_intensity = "low"
- if the user wants attractions but does not specify a type, use preferences.attraction_style = "mixed"

Output schema:
{{
  "origin": null,
  "destination": [],
  "country": null,
  "num_people": null,
  "travel_dates": {{
    "start": null,
    "end": null,
    "text": null
  }},
  "duration_days": null,
  "budget": {{
    "total": null,
    "currency": "CNY"
  }},
  "preferences": {{
    "flight": null,
    "hotel": null,
    "food": null,
    "attraction_style": null
  }},
  "planning_signals": {{
    "flight_priority": null,
    "hotel_level": null,
    "food_priority": null,
    "attraction_intensity": null
  }}
}}

User request:
{user_request}
""".strip()


# =========================================================
# Preference / Context Recognition
# =========================================================


def _get_preferences(parsed_request: Dict[str, Any]) -> Dict[str, Any]:
    preferences = parsed_request.get("preferences", {}) or {}

    hotel_features = preferences.get("hotel_features") or []
    if isinstance(hotel_features, str):
        hotel_features = [hotel_features]
    hotel_features = [
        (_normalize_text(x) or "").lower()
        for x in hotel_features
        if _normalize_text(x)
    ]

    food_preferences = preferences.get("food_preferences") or []
    if isinstance(food_preferences, str):
        food_preferences = [food_preferences]
    food_preferences = [
        (_normalize_text(x) or "").lower()
        for x in food_preferences
        if _normalize_text(x)
    ]

    attraction_styles = preferences.get("attraction_styles") or []
    if isinstance(attraction_styles, str):
        attraction_styles = [attraction_styles]
    attraction_styles = [
        (_normalize_text(x) or "").lower()
        for x in attraction_styles
        if _normalize_text(x)
    ]

    return {
        "flight": (_normalize_text(preferences.get("flight")) or "").lower(),
        "hotel": (_normalize_text(preferences.get("hotel")) or "").lower(),
        "food": (_normalize_text(preferences.get("food")) or "").lower(),
        "attraction_style": (_normalize_text(preferences.get("attraction_style")) or "").lower(),
        "hotel_level": (_normalize_text(preferences.get("hotel_level")) or "").lower(),
        "hotel_features": hotel_features,
        "food_preferences": food_preferences,
        "attraction_styles": attraction_styles,
    }


def _get_planning_signals(parsed_request: Dict[str, Any]) -> Dict[str, str]:
    planning_signals = parsed_request.get("planning_signals", {}) or {}
    hotel_signal = (_normalize_text(planning_signals.get("hotel_level")) or "").lower()
    hotel_priority = (_normalize_text(planning_signals.get("hotel_priority")) or "").lower()

    if not hotel_signal:
        hotel_signal = hotel_priority
    if not hotel_priority:
        hotel_priority = hotel_signal

    return {
        "flight_priority": (_normalize_text(planning_signals.get("flight_priority")) or "").lower(),
        "hotel_level": hotel_signal,
        "food_priority": (_normalize_text(planning_signals.get("food_priority")) or "").lower(),
        "attraction_intensity": (_normalize_text(planning_signals.get("attraction_intensity")) or "").lower(),
        "hotel_priority": hotel_priority,
    }


def _primary_from_list(values: List[str], priority: List[str], fallback: str = "") -> str:
    value_set = set(v for v in values if v)
    for item in priority:
        if item in value_set:
            return item
    return values[0] if values else fallback


def _has_feature(values: List[str], target: str) -> bool:
    return target in set(values or [])


def _has_strong_hotel_preference(hotel_pref: str) -> bool:
    keywords = [
        "five star", "5 star", "five-star", "luxury", "high-end",
        "premium", "resort", "upscale", "deluxe"
    ]
    return any(k in hotel_pref for k in keywords)


def _has_budget_hotel_preference(hotel_pref: str) -> bool:
    keywords = [
        "budget", "cheap", "economy", "hostel", "affordable", "low-cost",
        "not very good", "basic"
    ]
    return any(k in hotel_pref for k in keywords)


def _has_direct_flight_preference(flight_pref: str) -> bool:
    return "direct" in flight_pref or "non-stop" in flight_pref or "nonstop" in flight_pref


def _has_better_food_preference(food_pref: str) -> bool:
    keywords = ["better", "good", "fine", "fine dining", "premium", "high-end", "gourmet"]
    return any(k in food_pref for k in keywords)


def _has_local_food_preference(food_pref: str) -> bool:
    return "local" in food_pref or "street food" in food_pref


def _has_shopping_preference(attraction_pref: str) -> bool:
    return "shopping" in attraction_pref


def _has_sightseeing_preference(attraction_pref: str) -> bool:
    keywords = [
        "sightseeing", "landmark", "museum", "culture", "scenic",
        "lots of attractions", "many attractions"
    ]
    return any(k in attraction_pref for k in keywords)


def _contains_long_haul_destination(parsed_request: Dict[str, Any]) -> bool:
    origin = (_normalize_text(parsed_request.get("origin")) or "").lower()
    cities = [
        (item.get("city") or "").lower()
        for item in _get_effective_cities(parsed_request)
    ]
    country = (_normalize_text(parsed_request.get("country")) or "").lower()

    all_tokens = set([c for c in cities if c])
    if country:
        all_tokens.add(country)

    long_haul_markers = {
        "new york", "usa", "united states", "london", "paris",
        "rome", "milan", "zurich", "geneva", "toronto", "vancouver",
        "san francisco", "los angeles", "sydney", "melbourne"
    }

    if origin in {"beijing", "shanghai", "shenzhen", "guangzhou", "singapore", "zhuhai"}:
        return any(token in long_haul_markers for token in all_tokens)
    return False


def _is_multi_city(parsed_request: Dict[str, Any]) -> bool:
    cities = _get_effective_cities(parsed_request)
    return len(cities) >= 2


def _extract_month_from_request(parsed_request: Dict[str, Any]) -> Optional[int]:
    travel_dates = parsed_request.get("travel_dates", {}) or {}

    for key in ["start", "end"]:
        value = _normalize_text(travel_dates.get(key))
        if value:
            match = re.match(r"^\d{4}-(\d{2})-\d{2}$", value)
            if match:
                return int(match.group(1))

    text = (_normalize_text(travel_dates.get("text")) or "").lower()

    month_map = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }

    for token, month in month_map.items():
        if token in text:
            return month

    return None


# =========================================================
# Heuristics
# =========================================================

def _heuristic_cost_level(parsed_request: Dict[str, Any]) -> str:
    cities = _get_effective_cities(parsed_request)
    country = _normalize_text(parsed_request.get("country"))

    tokens = {
        (item.get("city") or "").lower().strip()
        for item in cities
        if isinstance(item, dict) and (item.get("city") or "").strip()
    }

    if country:
        tokens.add(country.lower().strip())

    high_cost = {
        "japan", "tokyo", "osaka", "kyoto", "sapporo", "france", "paris",
        "uk", "united kingdom", "london", "switzerland", "zurich", "geneva",
        "italy", "rome", "milan", "venice", "singapore",
        "usa", "united states", "new york", "san francisco", "los angeles",
        "sydney", "melbourne"
    }

    medium_high = {
        "korea", "south korea", "seoul", "busan", "jeju",
        "spain", "madrid", "barcelona", "australia", "brisbane",
        "amsterdam", "vienna", "munich", "berlin"
    }

    low_cost = {
        "thailand", "bangkok", "chiang mai", "phuket",
        "malaysia", "kuala lumpur", "penang",
        "vietnam", "hanoi", "ho chi minh city", "da nang",
        "indonesia", "jakarta", "bali"
    }

    if any(token in high_cost for token in tokens):
        return "high"
    if any(token in medium_high for token in tokens):
        return "medium_high"
    if any(token in low_cost for token in tokens):
        return "low"
    return "medium"


def _heuristic_strategy(parsed_request: Dict[str, Any], cost_level: str) -> str:
    if _is_budget_cap_mode(parsed_request):
        return "tight"

    total_budget = _get_total_budget(parsed_request)
    num_people = _get_num_people(parsed_request)
    duration_days = _get_duration_days(parsed_request)

    per_person_per_day = total_budget / max(1, num_people * duration_days)

    if cost_level == "high":
        if per_person_per_day < 900:
            return "tight"
        if per_person_per_day < 1800:
            return "balanced"
        return "comfortable"

    if cost_level == "medium_high":
        if per_person_per_day < 700:
            return "tight"
        if per_person_per_day < 1500:
            return "balanced"
        return "comfortable"

    if cost_level == "low":
        if per_person_per_day < 400:
            return "tight"
        if per_person_per_day < 1000:
            return "balanced"
        return "comfortable"

    if per_person_per_day < 550:
        return "tight"
    if per_person_per_day < 1200:
        return "balanced"
    return "comfortable"


# =========================================================
# Preference Normalization
# =========================================================


def _normalize_preference_levels(parsed_request: Dict[str, Any]) -> Dict[str, str]:
    prefs = _get_preferences(parsed_request)
    signals = _get_planning_signals(parsed_request)

    flight_pref = prefs["flight"]
    hotel_pref = prefs["hotel"]
    food_pref = prefs["food"]
    attraction_pref = prefs["attraction_style"]

    hotel_level_pref = prefs.get("hotel_level", "")
    hotel_features = prefs.get("hotel_features", []) or []
    food_preferences = prefs.get("food_preferences", []) or []
    attraction_styles = prefs.get("attraction_styles", []) or []

    flight_priority = signals["flight_priority"]
    hotel_level_signal = signals["hotel_level"] or signals.get("hotel_priority", "")
    food_priority = signals["food_priority"]
    attraction_intensity_signal = signals["attraction_intensity"]

    hotel_signal_map = {
        "high": "premium",
        "medium": "standard",
        "low": "budget"
    }
    food_signal_map = {
        "high": "premium",
        "medium": "standard",
        "low": "standard"
    }
    attraction_signal_map = {
        "high": "heavy",
        "medium": "medium",
        "low": "light"
    }
    flight_signal_map = {
        "high": "direct" if flight_pref == "direct" else ("comfort" if flight_pref == "comfort" else "standard"),
        "medium": "standard",
        "low": "budget" if flight_pref == "cheap" else "standard"
    }

    # richer hotel interpretation
    hotel_level_detail = "standard"
    if hotel_level_pref in {"budget", "standard", "premium", "luxury"}:
        hotel_level_detail = hotel_level_pref
    elif hotel_level_signal in hotel_signal_map:
        signal_level = hotel_signal_map[hotel_level_signal]
        hotel_level_detail = "premium" if signal_level == "premium" else signal_level
    else:
        if hotel_pref == "budget" or _has_budget_hotel_preference(hotel_pref):
            hotel_level_detail = "budget"
        elif hotel_pref == "luxury" or _has_strong_hotel_preference(hotel_pref):
            hotel_level_detail = "premium"
        else:
            hotel_level_detail = "standard"

    hotel_level = "premium" if hotel_level_detail in {"premium", "luxury"} else hotel_level_detail
    hotel_central = "yes" if _has_feature(hotel_features, "central") or hotel_pref == "central" else "no"
    hotel_clean = "yes" if _has_feature(hotel_features, "clean") or hotel_pref == "clean" else "no"

    # richer food interpretation
    primary_food = _primary_from_list(
        food_preferences,
        ["local", "fine_dining", "street_food", "mixed"],
        fallback=food_pref
    )
    wants_local_food = "yes" if ("local" in food_preferences or food_pref == "local") else "no"
    wants_fine_dining = "yes" if ("fine_dining" in food_preferences or food_pref == "fine_dining") else "no"

    if food_priority in food_signal_map:
        food_level = food_signal_map[food_priority]
        if wants_fine_dining == "yes" and food_level == "standard":
            food_level = "premium"
    else:
        if wants_fine_dining == "yes" or _has_better_food_preference(food_pref):
            food_level = "premium"
        else:
            food_level = "standard"

    # richer attraction interpretation
    primary_attraction_style = _primary_from_list(
        attraction_styles,
        ["historical", "art", "nature", "shopping", "entertainment", "mixed"],
        fallback=attraction_pref
    )

    if attraction_intensity_signal in attraction_signal_map:
        attraction_intensity = attraction_signal_map[attraction_intensity_signal]
    else:
        joined = " ".join([attraction_pref] + attraction_styles)
        if "lots of attractions" in joined or "many attractions" in joined or "packed itinerary" in joined:
            attraction_intensity = "heavy"
        elif joined:
            attraction_intensity = "medium"
        else:
            attraction_intensity = "light"

    if flight_priority in flight_signal_map:
        flight_type = flight_signal_map[flight_priority]
    else:
        if flight_pref == "direct" or _has_direct_flight_preference(flight_pref):
            flight_type = "direct"
        elif flight_pref == "comfort":
            flight_type = "comfort"
        elif flight_pref == "cheap":
            flight_type = "budget"
        else:
            flight_type = "standard"

    return {
        "hotel_level": hotel_level,
        "hotel_level_detail": hotel_level_detail,
        "hotel_central": hotel_central,
        "hotel_clean": hotel_clean,
        "food_level": food_level,
        "primary_food_preference": primary_food or "mixed",
        "wants_local_food": wants_local_food,
        "wants_fine_dining": wants_fine_dining,
        "attraction_intensity": attraction_intensity,
        "attraction_focus": primary_attraction_style or "mixed",
        "flight_type": flight_type
    }


def _apply_budget_feasibility_guard(
    parsed_request: Dict[str, Any],
    weighted_cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str]
) -> Tuple[Dict[str, str], List[str]]:
    guarded = dict(normalized_prefs)
    adjustments: List[str] = []

    total_budget = _get_total_budget(parsed_request)
    duration_days = _get_duration_days(parsed_request)
    num_people = _get_num_people(parsed_request)
    city_count = max(1, len(_get_effective_cities(parsed_request)))

    transport_min = int(weighted_cost_profile.get("transport_base_per_person", 0) * max(1, num_people))
    if guarded.get("flight_type") == "first_class":
        transport_min = int(round(transport_min * 1.25))
    elif guarded.get("flight_type") == "direct":
        transport_min = int(round(transport_min * 1.08))
    elif guarded.get("flight_type") == "comfort":
        transport_min = int(round(transport_min * 1.12))
    elif guarded.get("flight_type") == "budget":
        transport_min = int(round(transport_min * 0.92))

    hotel_nights = max(1, duration_days - city_count + 1)
    if guarded.get("hotel_level_detail") == "luxury":
        hotel_per_night = weighted_cost_profile.get("hotel_premium_per_night", 0)
    elif guarded.get("hotel_level") == "premium":
        hotel_per_night = weighted_cost_profile.get("hotel_premium_per_night", 0)
    elif guarded.get("hotel_level") == "budget":
        hotel_per_night = weighted_cost_profile.get("hotel_budget_per_night", 0)
    else:
        hotel_per_night = weighted_cost_profile.get("hotel_standard_per_night", 0)

    hotel_min = int(hotel_per_night * hotel_nights)
    essentials_min = transport_min + hotel_min

    if guarded.get("hotel_level_detail") == "luxury" and total_budget > 0 and essentials_min > total_budget:
        guarded["hotel_level_detail"] = "standard"
        guarded["hotel_level"] = "standard"
        adjustments.append(
            "Luxury hotel preference was kept as a user preference signal, but budget execution was downgraded to standard because direct transport plus luxury hotel minimum cost exceeded the total budget."
        )

    return guarded, adjustments



# =========================================================
# Cost-Level Fusion
# =========================================================

COST_LEVEL_ORDER = ["low", "medium", "medium_high", "high", "ultra_high"]


def _cost_level_to_index(level: Optional[str]) -> int:
    if level in COST_LEVEL_ORDER:
        return COST_LEVEL_ORDER.index(level)
    return 1


def _index_to_cost_level(index: int) -> str:
    index = max(0, min(index, len(COST_LEVEL_ORDER) - 1))
    return COST_LEVEL_ORDER[index]


def _merge_cost_level(
    llm_cost_level: Optional[str],
    profile_cost_level: Optional[str],
    parsed_request: Dict[str, Any],
    normalized_prefs: Dict[str, str]
) -> str:
    base_level = profile_cost_level or llm_cost_level or _heuristic_cost_level(parsed_request)
    base_index = _cost_level_to_index(base_level)

    delta = 0

    if normalized_prefs["hotel_level"] == "premium":
        delta += 1
    elif normalized_prefs["hotel_level"] == "budget":
        delta -= 1

    if normalized_prefs.get("hotel_level_detail") == "luxury":
        delta += 1

    if normalized_prefs.get("hotel_central") == "yes":
        delta += 1
    if normalized_prefs["food_level"] == "premium":
        delta += 1
    if normalized_prefs.get("wants_fine_dining") == "yes":
        delta += 1

    if normalized_prefs["flight_type"] == "first_class":
        delta += 2
    elif normalized_prefs["flight_type"] == "direct":
        delta += 1
    elif normalized_prefs["flight_type"] == "comfort":
        delta += 1
    elif normalized_prefs["flight_type"] == "budget":
        delta -= 1

    if normalized_prefs["attraction_intensity"] == "heavy":
        delta += 1

    if _is_multi_city(parsed_request):
        delta += 1

    total_budget = _get_total_budget(parsed_request)
    num_people = _get_num_people(parsed_request)
    duration_days = _get_duration_days(parsed_request)
    per_person_per_day = total_budget / max(1, num_people * duration_days)

    if per_person_per_day < 500:
        delta -= 1
    elif per_person_per_day > 3500:
        delta += 2
    elif per_person_per_day > 1800:
        delta += 1

    if _is_budget_cap_mode(parsed_request):
        delta -= 2

    if llm_cost_level in COST_LEVEL_ORDER:
        llm_index = _cost_level_to_index(llm_cost_level)
        if llm_index > base_index:
            delta += 1
        elif llm_index < base_index:
            delta -= 1

    final_index = base_index
    if delta >= 2:
        final_index += 1
    elif delta <= -2:
        final_index -= 1

    return _index_to_cost_level(final_index)


# =========================================================
# Season / Dynamic Factors
# =========================================================

def _get_season_factor(parsed_request: Dict[str, Any], destination: str) -> float:
    month = _extract_month_from_request(parsed_request)
    if month is None:
        return 1.00

    japan_markers = {"tokyo", "osaka", "kyoto", "sapporo", "fukuoka", "japan"}
    europe_markers = {
        "paris", "rome", "london", "milan", "venice", "florence",
        "barcelona", "madrid", "amsterdam", "zurich", "geneva",
        "vienna", "prague", "berlin", "munich", "athens", "lisbon",
        "france", "italy", "uk", "united kingdom"
    }
    korea_markers = {"seoul", "busan", "jeju", "korea", "south korea"}
    tropical_markers = {
        "bangkok", "chiang mai", "phuket", "kuala lumpur", "penang",
        "singapore", "thailand", "malaysia", "bali", "jakarta",
        "hanoi", "ho chi minh city", "da nang"
    }

    if destination in japan_markers:
        if month in {3, 4, 11}:
            return 1.18
        if month in {1, 2, 6}:
            return 0.96

    if destination in europe_markers:
        if month in {6, 7, 8, 12}:
            return 1.15
        if month in {1, 2, 11}:
            return 0.95

    if destination in korea_markers:
        if month in {4, 5, 10, 11}:
            return 1.08
        if month in {1, 2, 7, 8}:
            return 0.97

    if destination in tropical_markers:
        if month in {12, 1, 2, 7, 8}:
            return 1.08
        if month in {4, 5, 9, 10}:
            return 0.97

    if month in {7, 8, 12}:
        return 1.08

    return 1.00


# =========================================================
# Ratio Logic
# =========================================================

def _build_baseline_ratio(
    normalized_prefs: Dict[str, str],
    budget_strategy: str,
    parsed_request: Dict[str, Any]
) -> Dict[str, float]:
    ratio = {
        "transport": 0.30,
        "hotel": 0.35,
        "food": 0.15,
        "attractions": 0.12,
        "buffer": 0.08
    }

    if normalized_prefs["flight_type"] == "first_class":
        ratio["transport"] += 0.08
        ratio["buffer"] -= 0.02
        ratio["food"] -= 0.02
        ratio["attractions"] -= 0.02
        ratio["hotel"] -= 0.02
    elif normalized_prefs["flight_type"] == "direct":
        ratio["transport"] += 0.04
        ratio["buffer"] -= 0.01
        ratio["food"] -= 0.01
        ratio["attractions"] -= 0.02
    elif normalized_prefs["flight_type"] == "comfort":
        ratio["transport"] += 0.03
        ratio["buffer"] -= 0.01
        ratio["food"] -= 0.01
        ratio["attractions"] -= 0.01
    elif normalized_prefs["flight_type"] == "budget":
        ratio["transport"] -= 0.04
        ratio["buffer"] += 0.02
        ratio["hotel"] += 0.01
        ratio["food"] += 0.01

    if normalized_prefs["hotel_level"] == "premium":
        ratio["hotel"] += 0.06
        ratio["food"] -= 0.01
        ratio["attractions"] -= 0.02
        ratio["buffer"] -= 0.03
    elif normalized_prefs["hotel_level"] == "budget":
        ratio["hotel"] -= 0.07
        ratio["buffer"] += 0.03
        ratio["food"] += 0.01
        ratio["attractions"] += 0.03

    if normalized_prefs.get("hotel_level_detail") == "luxury":
        ratio["hotel"] += 0.03
        ratio["buffer"] -= 0.01
        ratio["food"] -= 0.01
        ratio["attractions"] -= 0.01

    if normalized_prefs.get("hotel_central") == "yes":
        ratio["hotel"] += 0.02
        ratio["buffer"] -= 0.01
    if normalized_prefs.get("hotel_clean") == "yes":
        ratio["hotel"] += 0.01
        ratio["buffer"] -= 0.01

    if normalized_prefs["food_level"] == "premium":
        ratio["food"] += 0.05
        ratio["buffer"] -= 0.02
        ratio["attractions"] -= 0.01
        ratio["transport"] -= 0.02

    if normalized_prefs.get("wants_fine_dining") == "yes":
        ratio["food"] += 0.03
        ratio["buffer"] -= 0.01
        ratio["transport"] -= 0.01
    elif normalized_prefs.get("wants_local_food") == "yes":
        ratio["food"] += 0.01
        ratio["buffer"] -= 0.01

    if normalized_prefs["attraction_intensity"] == "heavy":
        ratio["attractions"] += 0.06
        ratio["buffer"] -= 0.02
        ratio["food"] -= 0.01
        ratio["hotel"] -= 0.03
    elif normalized_prefs["attraction_intensity"] == "light":
        ratio["attractions"] -= 0.03
        ratio["buffer"] += 0.02
        ratio["hotel"] += 0.01

    if _is_multi_city(parsed_request):
        ratio["transport"] += 0.03
        ratio["hotel"] -= 0.01
        ratio["buffer"] -= 0.01
        ratio["food"] -= 0.01

    if budget_strategy == "tight":
        ratio["buffer"] = 0.06
    elif budget_strategy == "comfortable":
        ratio["buffer"] = 0.10

    ratio["buffer"] = _clamp_ratio(ratio["buffer"], 0.05, 0.12)

    return _round_ratio_map(ratio)

def _normalize_recommended_ratio(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None

    keys = ["transport", "hotel", "food", "attractions", "buffer"]
    ratio: Dict[str, float] = {}
    for key in keys:
        ratio[key] = max(0.0, _safe_float(value.get(key), 0.0))

    total = sum(ratio.values())
    if total <= 0:
        return None

    normalized = {k: v / total for k, v in ratio.items()}
    normalized["buffer"] = _clamp_ratio(normalized["buffer"], 0.05, 0.12)
    return _round_ratio_map(normalized)


def _merge_ratios(
    baseline_ratio: Dict[str, float],
    llm_ratio: Optional[Dict[str, float]]
) -> Dict[str, float]:
    if not llm_ratio:
        return baseline_ratio

    merged = {}
    for key in baseline_ratio:
        merged[key] = baseline_ratio[key] * 0.65 + llm_ratio.get(key, 0.0) * 0.35

    merged["buffer"] = _clamp_ratio(merged["buffer"], 0.05, 0.12)
    return _round_ratio_map(merged)


# =========================================================
# Cost Estimation
# =========================================================

def _estimate_transport_cost(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str],
    season_factor: float
) -> int:
    num_people = _get_num_people(parsed_request)
    duration_days = _get_duration_days(parsed_request)

    base = _safe_float(cost_profile["transport_base_per_person"]) * num_people

    if normalized_prefs["flight_type"] == "first_class":
        base *= 1.35
    elif normalized_prefs["flight_type"] == "direct":
        base *= 1.15
    elif normalized_prefs["flight_type"] == "comfort":
        base *= 1.10
    elif normalized_prefs["flight_type"] == "budget":
        base *= 0.92

    if _is_multi_city(parsed_request):
        base *= 1.20

    if _contains_long_haul_destination(parsed_request):
        base *= 1.08

    if duration_days >= 8:
        base *= 1.05

    base *= (1.0 + (season_factor - 1.0) * 0.70)
    return _round_to_100(base)

def _estimate_hotel_cost(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str],
    season_factor: float
) -> int:
    num_people = _get_num_people(parsed_request)
    duration_days = _get_duration_days(parsed_request)
    nights = max(1, duration_days - 1)

    hotel_level = normalized_prefs["hotel_level"]
    price_key = f"hotel_{hotel_level}_per_night"
    nightly_price = _safe_float(cost_profile[price_key])

    if normalized_prefs.get("hotel_level_detail") == "luxury":
        nightly_price *= 1.18
    if normalized_prefs.get("hotel_central") == "yes":
        nightly_price *= 1.10
    if normalized_prefs.get("hotel_clean") == "yes":
        nightly_price *= 1.04

    rooms = max(1, (num_people + 1) // 2)
    total = nightly_price * nights * rooms * season_factor

    return _round_to_100(total)

def _estimate_food_cost(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str],
    season_factor: float
) -> int:
    num_people = _get_num_people(parsed_request)
    duration_days = _get_duration_days(parsed_request)

    food_level = normalized_prefs["food_level"]
    price_key = f"food_{food_level}_per_day"
    per_day = _safe_float(cost_profile[price_key])

    if normalized_prefs.get("wants_fine_dining") == "yes":
        per_day *= 1.12
    if normalized_prefs.get("wants_local_food") == "yes" and normalized_prefs.get("wants_fine_dining") != "yes":
        per_day *= 0.96

    total = per_day * duration_days * num_people
    total *= (1.0 + (season_factor - 1.0) * 0.35)
    return _round_to_100(total)

def _estimate_attraction_cost(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str],
    season_factor: float
) -> int:
    num_people = _get_num_people(parsed_request)
    duration_days = _get_duration_days(parsed_request)

    intensity = normalized_prefs["attraction_intensity"]
    price_key = f"attraction_{intensity}_per_day"
    per_day = _safe_float(cost_profile[price_key])

    total = per_day * duration_days * num_people
    total *= (1.0 + (season_factor - 1.0) * 0.45)
    return _round_to_100(total)


def _estimate_buffer_cost(subtotal: int, strategy: str) -> int:
    if strategy == "tight":
        ratio = 0.06
    elif strategy == "comfortable":
        ratio = 0.10
    else:
        ratio = 0.08
    return _round_to_100(subtotal * ratio)


def _estimated_total_with_buffer(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str],
    season_factor: float,
    budget_strategy: str,
) -> int:
    transport = _estimate_transport_cost(parsed_request, cost_profile, normalized_prefs, season_factor)
    hotel = _estimate_hotel_cost(parsed_request, cost_profile, normalized_prefs, season_factor)
    food = _estimate_food_cost(parsed_request, cost_profile, normalized_prefs, season_factor)
    attractions = _estimate_attraction_cost(parsed_request, cost_profile, normalized_prefs, season_factor)
    return transport + hotel + food + attractions + _estimate_buffer_cost(
        transport + hotel + food + attractions,
        budget_strategy,
    )


def _apply_budget_pressure_downgrades(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str],
    season_factor: float,
    budget_strategy: str,
) -> Tuple[Dict[str, str], List[str]]:
    total_budget = _get_total_budget(parsed_request)
    if total_budget <= 0:
        return normalized_prefs, []

    guarded = dict(normalized_prefs)
    adjustments: List[str] = []

    def estimated_total() -> int:
        return _estimated_total_with_buffer(
            parsed_request=parsed_request,
            cost_profile=cost_profile,
            normalized_prefs=guarded,
            season_factor=season_factor,
            budget_strategy=budget_strategy,
        )

    if estimated_total() <= total_budget:
        return guarded, adjustments

    downgrade_steps = [
        ("flight_type", {"first_class": "comfort", "direct": "standard", "comfort": "standard", "standard": "budget"}),
        ("hotel_level_detail", {"luxury": "standard", "premium": "standard"}),
        ("hotel_level", {"premium": "standard", "standard": "budget"}),
        ("food_level", {"premium": "standard", "standard": "budget"}),
        ("attraction_intensity", {"heavy": "medium", "medium": "light"}),
        ("hotel_central", {"yes": "no"}),
        ("hotel_clean", {"yes": "no"}),
        ("flight_type", {"standard": "budget"}),
    ]

    labels = {
        "flight_type": "Downgraded flight execution toward cheaper/later/connecting options before asking for more budget.",
        "hotel_level_detail": "Downgraded luxury hotel execution before asking for more budget.",
        "hotel_level": "Downgraded hotel execution toward budget hotels before asking for more budget.",
        "food_level": "Downgraded food execution toward budget/local choices before asking for more budget.",
        "attraction_intensity": "Reduced paid activity intensity before asking for more budget.",
        "hotel_central": "Relaxed central-location hotel requirement before asking for more budget.",
        "hotel_clean": "Relaxed clean/high-standard hotel preference before asking for more budget.",
    }

    for key, mapping in downgrade_steps:
        current = guarded.get(key)
        replacement = mapping.get(current)
        if not replacement or replacement == current:
            continue

        guarded[key] = replacement
        if key == "hotel_level_detail" and replacement == "standard" and guarded.get("hotel_level") == "premium":
            guarded["hotel_level"] = "standard"
        adjustments.append(labels[key])

        if estimated_total() <= total_budget:
            break

    if adjustments:
        guarded["budget_pressure_mode"] = "yes"

    return guarded, adjustments


# =========================================================
# Weighted Budget Compression
# =========================================================

def _build_compression_priority(normalized_prefs: Dict[str, str]) -> Dict[str, float]:
    priority = {
        "buffer": 1.00,
        "attractions": 0.70,
        "food": 0.55,
        "hotel": 0.50,
        "transport": 0.30
    }

    if normalized_prefs["attraction_intensity"] == "heavy":
        priority["attractions"] = 0.25
    elif normalized_prefs["attraction_intensity"] == "light":
        priority["attractions"] = 0.85

    if normalized_prefs["hotel_level"] == "premium":
        priority["hotel"] = 0.20
    elif normalized_prefs["hotel_level"] == "budget":
        priority["hotel"] = 0.80

    if normalized_prefs.get("hotel_level_detail") == "luxury":
        priority["hotel"] = 0.12
    if normalized_prefs.get("hotel_central") == "yes":
        priority["hotel"] = min(priority["hotel"], 0.25)

    if normalized_prefs["food_level"] == "premium":
        priority["food"] = 0.30
    if normalized_prefs.get("wants_fine_dining") == "yes":
        priority["food"] = min(priority["food"], 0.28)

    if normalized_prefs["flight_type"] == "first_class":
        priority["transport"] = 0.12
    elif normalized_prefs["flight_type"] == "direct":
        priority["transport"] = 0.18
    elif normalized_prefs["flight_type"] == "comfort":
        priority["transport"] = 0.22
    elif normalized_prefs["flight_type"] == "budget":
        priority["transport"] = 0.45

    return priority

def _fit_estimated_cost_to_budget(
    estimated: Dict[str, int],
    total_budget: int,
    normalized_prefs: Dict[str, str]
) -> Tuple[Dict[str, int], str, List[str]]:
    adjusted = dict(estimated)
    adjustments: List[str] = []

    current_total = sum(adjusted.values())
    if current_total < total_budget:
        return adjusted, "under_budget", adjustments
    if current_total == total_budget:
        return adjusted, "within_budget", adjustments

    overflow = current_total - total_budget
    priority = _build_compression_priority(normalized_prefs)

    hotel_keep = 0.80 if normalized_prefs["hotel_level"] == "premium" else 0.60
    if normalized_prefs.get("hotel_level_detail") == "luxury":
        hotel_keep = 0.86
    food_keep = 0.75 if normalized_prefs["food_level"] == "premium" else 0.60
    if normalized_prefs.get("wants_fine_dining") == "yes":
        food_keep = max(food_keep, 0.78)

    min_keep_ratio = {
        "transport": 0.82 if normalized_prefs["flight_type"] == "first_class" else (0.75 if normalized_prefs["flight_type"] in {"direct", "comfort"} else 0.65),
        "hotel": hotel_keep,
        "food": food_keep,
        "attractions": 0.80 if normalized_prefs["attraction_intensity"] == "heavy" else 0.50,
        "buffer": 0.00
    }

    original = dict(adjusted)
    categories = sorted(priority.keys(), key=lambda k: priority[k], reverse=True)

    for key in categories:
        if overflow <= 0:
            break

        current_value = adjusted.get(key, 0)
        min_allowed = int(original.get(key, 0) * min_keep_ratio.get(key, 0.0))
        reducible = max(0, current_value - min_allowed)

        if reducible <= 0:
            continue

        reduction = min(reducible, overflow)
        reduction = _round_to_100(reduction)
        reduction = min(reduction, adjusted[key] - min_allowed)

        if reduction <= 0 and overflow > 0:
            reduction = min(adjusted[key] - min_allowed, overflow)

        if reduction > 0:
            adjusted[key] -= reduction
            overflow -= reduction
            adjustments.append(f"Reduced {key} allocation to better fit the budget.")

    current_total = sum(adjusted.values())
    status = "over_budget" if current_total > total_budget else "adjusted_to_budget"
    return adjusted, status, adjustments


def _estimate_core_trio_total(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str],
    season_factor: float,
) -> int:
    transport = _estimate_transport_cost(parsed_request, cost_profile, normalized_prefs, season_factor)
    hotel = _estimate_hotel_cost(parsed_request, cost_profile, normalized_prefs, season_factor)
    food = _estimate_food_cost(parsed_request, cost_profile, normalized_prefs, season_factor)
    return transport + hotel + food


def _next_core_upgrade_candidates(normalized_prefs: Dict[str, str]) -> List[Tuple[str, Dict[str, str], str]]:
    current = dict(normalized_prefs)
    candidates: List[Tuple[str, Dict[str, str], str]] = []

    flight_type = current.get("flight_type", "standard")
    next_flight = {
        "budget": "standard",
        "standard": "comfort",
        "comfort": "first_class",
        "direct": "first_class",
    }.get(flight_type)
    if next_flight:
        upgraded = dict(current)
        upgraded["flight_type"] = next_flight
        candidates.append(("flight", upgraded, f"Upgraded flight execution from {flight_type} to {next_flight}"))

    hotel_detail = current.get("hotel_level_detail", current.get("hotel_level", "standard"))
    if hotel_detail == "budget":
        upgraded = dict(current)
        upgraded["hotel_level"] = "standard"
        upgraded["hotel_level_detail"] = "standard"
        candidates.append(("hotel", upgraded, "Upgraded hotel execution from budget to standard"))
    elif hotel_detail == "standard":
        upgraded = dict(current)
        upgraded["hotel_level"] = "premium"
        upgraded["hotel_level_detail"] = "premium"
        candidates.append(("hotel", upgraded, "Upgraded hotel execution from standard to premium"))
    elif hotel_detail == "premium":
        upgraded = dict(current)
        upgraded["hotel_level"] = "premium"
        upgraded["hotel_level_detail"] = "luxury"
        candidates.append(("hotel", upgraded, "Upgraded hotel execution from premium to luxury"))

    food_level = current.get("food_level", "standard")
    wants_fine_dining = current.get("wants_fine_dining", "no")
    if food_level == "budget":
        upgraded = dict(current)
        upgraded["food_level"] = "standard"
        candidates.append(("food", upgraded, "Upgraded dining execution from budget to standard"))
    elif food_level == "standard":
        upgraded = dict(current)
        upgraded["food_level"] = "premium"
        candidates.append(("food", upgraded, "Upgraded dining execution from standard to premium"))
    elif food_level == "premium" and wants_fine_dining != "yes":
        upgraded = dict(current)
        upgraded["wants_fine_dining"] = "yes"
        candidates.append(("food", upgraded, "Enabled fine-dining upgrades because the budget remains very loose"))

    return candidates


def _upgrade_core_preferences_toward_target(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str],
    season_factor: float,
    total_budget: int,
    target_core_ratio: float = 0.60,
) -> Tuple[Dict[str, str], int, List[str]]:
    upgraded = dict(normalized_prefs)
    adjustments: List[str] = []
    target_total = int(total_budget * target_core_ratio)
    current_total = _estimate_core_trio_total(parsed_request, cost_profile, upgraded, season_factor)

    if total_budget <= 0 or current_total >= target_total:
        return upgraded, current_total, adjustments

    while current_total < target_total:
        best_candidate: Optional[Tuple[Dict[str, str], int, str]] = None
        smallest_overshoot: Optional[Tuple[Dict[str, str], int, str]] = None

        for _, candidate_prefs, candidate_reason in _next_core_upgrade_candidates(upgraded):
            candidate_total = _estimate_core_trio_total(parsed_request, cost_profile, candidate_prefs, season_factor)
            if candidate_total <= current_total:
                continue
            if candidate_total <= target_total:
                if best_candidate is None or candidate_total > best_candidate[1]:
                    best_candidate = (candidate_prefs, candidate_total, candidate_reason)
            else:
                if smallest_overshoot is None or candidate_total < smallest_overshoot[1]:
                    smallest_overshoot = (candidate_prefs, candidate_total, candidate_reason)

        chosen = best_candidate or smallest_overshoot
        if chosen is None:
            break

        upgraded, current_total, reason = chosen
        adjustments.append(reason)

        if best_candidate is None:
            break

    if adjustments:
        adjustments.append(
            f"Core trio upgrade rule applied: flight + hotel + food was gradually upgraded until it reached about {int(target_core_ratio * 100)}% of the total budget or hit the highest supported tiers."
        )

    return upgraded, current_total, adjustments

# =========================================================
# Recommended Spend + Optional Upgrades
# =========================================================

def _allocate_by_ratio(
    total_budget: int,
    ratio_map: Dict[str, float]
) -> Dict[str, int]:
    plan = {k: _round_to_100(total_budget * ratio_map.get(k, 0.0)) for k in ratio_map}
    return _normalize_plan_to_total(plan, total_budget)


def _split_under_budget_plan(
    parsed_request: Dict[str, Any],
    cost_profile: Dict[str, Any],
    adjusted_plan: Dict[str, int],
    total_budget: int,
    normalized_prefs: Dict[str, str],
    season_factor: float,
    final_ratio: Optional[Dict[str, float]] = None,
    allow_optional_upgrades: bool = True
) -> Tuple[Dict[str, int], Dict[str, int], int, List[str], Dict[str, str]]:
    adjustments: List[str] = []

    adjusted_total = sum(adjusted_plan.values())

    if adjusted_total >= total_budget:
        recommended = _normalize_plan_to_total(adjusted_plan, total_budget)
        optional = {
            "transport": 0,
            "hotel": 0,
            "food": 0,
            "attractions": 0,
            "buffer": 0
        }
        adjustments.append(
            "Estimated base spending already reached or exceeded the total budget, so the final plan was normalized directly to the full budget."
        )
        return recommended, optional, 0, adjustments, dict(normalized_prefs)

    recommended = {
        "transport": max(0, int(adjusted_plan.get("transport", 0))),
        "hotel": max(0, int(adjusted_plan.get("hotel", 0))),
        "food": max(0, int(adjusted_plan.get("food", 0))),
        "attractions": max(0, int(adjusted_plan.get("attractions", 0))),
        "buffer": max(0, int(adjusted_plan.get("buffer", 0))),
    }

    effective_prefs = dict(normalized_prefs)
    effective_prefs, _, core_upgrade_adjustments = _upgrade_core_preferences_toward_target(
        parsed_request=parsed_request,
        cost_profile=cost_profile,
        normalized_prefs=effective_prefs,
        season_factor=season_factor,
        total_budget=total_budget,
    )
    adjustments.extend(core_upgrade_adjustments)

    if core_upgrade_adjustments:
        recommended["transport"] = _estimate_transport_cost(parsed_request, cost_profile, effective_prefs, season_factor)
        recommended["hotel"] = _estimate_hotel_cost(parsed_request, cost_profile, effective_prefs, season_factor)
        recommended["food"] = _estimate_food_cost(parsed_request, cost_profile, effective_prefs, season_factor)
        adjusted_total = sum(recommended.values())
        if adjusted_total >= total_budget:
            recommended = _normalize_plan_to_total(recommended, total_budget)
            optional = {
                "transport": 0,
                "hotel": 0,
                "food": 0,
                "attractions": 0,
                "buffer": 0
            }
            adjustments.append(
                "Core trio upgrades used up the available budget, so the plan was normalized directly to the total budget after raising flight, hotel, and food tiers."
            )
            return recommended, optional, 0, adjustments, effective_prefs

    remaining = max(0, total_budget - adjusted_total)

    if remaining == 0 or not allow_optional_upgrades:
        optional = {
            "transport": 0,
            "hotel": 0,
            "food": 0,
            "attractions": 0,
            "buffer": 0
        }
        if remaining > 0 and not allow_optional_upgrades:
            recommended["buffer"] += remaining
            adjustments.append(
                "Budget cap mode is active, so remaining money was kept as buffer instead of being spent on optional upgrades."
            )
        return recommended, optional, 0, adjustments, effective_prefs

    upgrade_ratio = {
        "transport": 0.14,
        "hotel": 0.38,
        "food": 0.24,
        "attractions": 0.21,
        "buffer": 0.03
    }

    if effective_prefs["hotel_level"] == "premium":
        upgrade_ratio["hotel"] += 0.10
        upgrade_ratio["buffer"] -= 0.05
        upgrade_ratio["transport"] -= 0.02
        upgrade_ratio["food"] -= 0.03
    elif effective_prefs["hotel_level"] == "budget":
        upgrade_ratio["hotel"] -= 0.12
        upgrade_ratio["buffer"] += 0.08
        upgrade_ratio["attractions"] += 0.04

    if effective_prefs.get("hotel_level_detail") == "luxury":
        upgrade_ratio["hotel"] += 0.06
        upgrade_ratio["buffer"] -= 0.03
        upgrade_ratio["food"] -= 0.02
        upgrade_ratio["transport"] -= 0.01

    if effective_prefs["food_level"] == "premium":
        upgrade_ratio["food"] += 0.08
        upgrade_ratio["buffer"] -= 0.04
        upgrade_ratio["transport"] -= 0.02
        upgrade_ratio["attractions"] -= 0.02
    elif effective_prefs["food_level"] == "budget":
        upgrade_ratio["food"] -= 0.08
        upgrade_ratio["buffer"] += 0.05
        upgrade_ratio["attractions"] += 0.03

    if effective_prefs.get("wants_fine_dining") == "yes":
        upgrade_ratio["food"] += 0.05
        upgrade_ratio["buffer"] -= 0.03
        upgrade_ratio["transport"] -= 0.01

    if effective_prefs["attraction_intensity"] == "heavy":
        upgrade_ratio["attractions"] += 0.10
        upgrade_ratio["buffer"] -= 0.05
        upgrade_ratio["hotel"] -= 0.03
        upgrade_ratio["transport"] -= 0.02
    elif effective_prefs["attraction_intensity"] == "light":
        upgrade_ratio["attractions"] -= 0.05
        upgrade_ratio["buffer"] += 0.03
        upgrade_ratio["hotel"] += 0.01
        upgrade_ratio["food"] += 0.01

    if effective_prefs["flight_type"] == "first_class":
        upgrade_ratio["transport"] += 0.10
        upgrade_ratio["buffer"] -= 0.04
        upgrade_ratio["hotel"] -= 0.03
        upgrade_ratio["food"] -= 0.02
    elif effective_prefs["flight_type"] == "direct":
        upgrade_ratio["transport"] += 0.06
        upgrade_ratio["buffer"] -= 0.03
        upgrade_ratio["hotel"] -= 0.02
        upgrade_ratio["food"] -= 0.01
    elif effective_prefs["flight_type"] == "comfort":
        upgrade_ratio["transport"] += 0.04
        upgrade_ratio["buffer"] -= 0.02
        upgrade_ratio["hotel"] -= 0.01
    elif effective_prefs["flight_type"] == "budget":
        upgrade_ratio["transport"] -= 0.03
        upgrade_ratio["buffer"] += 0.02

    if final_ratio:
        normalized_upgrade_ratio = _round_ratio_map(upgrade_ratio)
        normalized_final_ratio = _round_ratio_map(final_ratio)

        blended_ratio: Dict[str, float] = {}
        for key in ["transport", "hotel", "food", "attractions", "buffer"]:
            blended_ratio[key] = (
                0.60 * normalized_upgrade_ratio.get(key, 0.0) +
                0.40 * normalized_final_ratio.get(key, 0.0)
            )
        upgrade_ratio = _round_ratio_map(blended_ratio)
    else:
        upgrade_ratio = _round_ratio_map(upgrade_ratio)

    optional = _allocate_by_ratio(remaining, upgrade_ratio)

    combined_plan = {
        key: recommended.get(key, 0) + optional.get(key, 0)
        for key in ["transport", "hotel", "food", "attractions", "buffer"]
    }
    combined_plan = _normalize_plan_to_total(combined_plan, total_budget)

    optional = {
        key: max(0, combined_plan.get(key, 0) - recommended.get(key, 0))
        for key in ["transport", "hotel", "food", "attractions", "buffer"]
    }

    combined_total = sum(recommended.values()) + sum(optional.values())
    diff = total_budget - combined_total
    if diff != 0:
        optional["buffer"] += diff

    adjustments.append(
        "Estimated trip cost was below the total budget, so the remaining budget was fully distributed into optional upgrades instead of being left unallocated."
    )
    adjustments.append(
        "The forced full-budget upgrade allocation was blended with the final merged category ratio to stay consistent with Agent2's planning logic."
    )
    return recommended, optional, 0, adjustments, effective_prefs

# =========================================================
# City-level intelligent budget allocation
# =========================================================

def _get_city_preference_multiplier(
    category: str,
    city_profile: Dict[str, Any],
    normalized_prefs: Dict[str, str]
) -> float:
    multiplier = 1.0
    weights = city_profile.get("weights", {}) or {}

    if category == "hotel":
        multiplier *= _safe_float(weights.get("hotel"), 1.0)
        if normalized_prefs.get("hotel_level") == "premium":
            if _safe_float(weights.get("hotel"), 1.0) >= 1.05:
                multiplier *= 1.08
            elif _safe_float(weights.get("hotel"), 1.0) <= 0.90:
                multiplier *= 0.96
        elif normalized_prefs.get("hotel_level") == "budget":
            if _safe_float(weights.get("hotel"), 1.0) >= 1.10:
                multiplier *= 0.94

        if normalized_prefs.get("hotel_level_detail") == "luxury":
            multiplier *= 1.06
        if normalized_prefs.get("hotel_central") == "yes":
            multiplier *= 1.05
        if normalized_prefs.get("hotel_clean") == "yes":
            multiplier *= 1.02

    elif category == "food":
        multiplier *= _safe_float(weights.get("food"), 1.0)
        if normalized_prefs.get("food_level") == "premium":
            if _safe_float(weights.get("food"), 1.0) >= 1.05:
                multiplier *= 1.08
        if normalized_prefs.get("wants_local_food") == "yes":
            multiplier *= 0.98
        if normalized_prefs.get("wants_fine_dining") == "yes":
            multiplier *= 1.06

    elif category == "attractions":
        multiplier *= _safe_float(weights.get("attractions"), 1.0)
        if normalized_prefs.get("attraction_intensity") == "heavy":
            if _safe_float(weights.get("attractions"), 1.0) >= 1.05:
                multiplier *= 1.12
            elif _safe_float(weights.get("attractions"), 1.0) <= 0.90:
                multiplier *= 0.96

        attraction_focus = normalized_prefs.get("attraction_focus")
        if attraction_focus in {"historical", "art"}:
            multiplier *= 1.03

    return multiplier

def _allocate_city_category_budgets(
    effective_city_profiles: List[Dict[str, Any]],
    total_hotel_budget: int,
    total_food_budget: int,
    total_attractions_budget: int,
    normalized_prefs: Dict[str, str],
) -> List[Dict[str, Any]]:
    if not effective_city_profiles:
        return []

    categories = {
        "hotel": total_hotel_budget,
        "food": total_food_budget,
        "attractions": total_attractions_budget
    }

    city_budget_map: List[Dict[str, Any]] = []
    for item in effective_city_profiles:
        city_budget_map.append({
            "city": item["city"],
            "days": item["days"],
            "profile_tier": item["tier"],
            "matched_profile": item["matched_key"],
            "cost_weights": item["profile"].get("weights", {}),
            "hotel_budget": 0,
            "food_budget": 0,
            "attractions_budget": 0
        })

    for category, total_budget in categories.items():
        if total_budget <= 0:
            continue

        raw_weights = []
        for idx, item in enumerate(effective_city_profiles):
            days = max(1, item["days"])
            pref_multiplier = _get_city_preference_multiplier(
                category=category,
                city_profile=item["profile"],
                normalized_prefs=normalized_prefs
            )
            weight = days * pref_multiplier
            raw_weights.append((idx, weight))

        weight_sum = sum(w for _, w in raw_weights)
        if weight_sum <= 0:
            equal_share = total_budget // max(1, len(city_budget_map))
            for city_item in city_budget_map:
                city_item[f"{category}_budget"] = equal_share
            continue

        provisional = []
        for idx, w in raw_weights:
            provisional_amount = _round_to_100(total_budget * (w / weight_sum))
            provisional.append((idx, provisional_amount))

        current_total = sum(v for _, v in provisional)
        diff = total_budget - current_total

        if provisional:
            last_idx = provisional[-1][0]
            provisional[-1] = (last_idx, provisional[-1][1] + diff)

        for idx, amount in provisional:
            city_budget_map[idx][f"{category}_budget"] = max(0, amount)

    return city_budget_map


# =========================================================
# Main
# =========================================================

def travel_budget_agent(parsed_request: Dict[str, Any]) -> Dict[str, Any]:
    result = _default_result()
    result["meta"]["llm_called"] = True

    try:
        total_budget = _get_total_budget(parsed_request)
        if total_budget <= 0:
            raise ValueError("Missing or invalid budget.total")

        prompt = _build_prompt(parsed_request)
        raw_text = _call_llm(prompt)
        parsed = _extract_json_from_text(raw_text)

        llm_adjustments = _normalize_adjustments(parsed.get("llm_adjustments"))
        preference_impact = _normalize_preference_impact(parsed.get("preference_impact"))
        assumptions = _normalize_assumptions(parsed.get("assumptions"))
        llm_ratio = _normalize_recommended_ratio(parsed.get("recommended_ratio"))

        llm_cost_level = _normalize_text(parsed.get("cost_level"))
        if llm_cost_level not in {"low", "medium", "medium_high", "high"}:
            llm_cost_level = _heuristic_cost_level(parsed_request)

        budget_strategy = _normalize_text(parsed.get("budget_strategy"))
        if budget_strategy not in {"tight", "balanced", "comfortable"}:
            budget_strategy = _heuristic_strategy(parsed_request, llm_cost_level)
        if _is_budget_cap_mode(parsed_request):
            budget_strategy = "tight"

        normalized_prefs = _normalize_preference_levels(parsed_request)
        weighted_cost_profile = _get_weighted_cost_profile(parsed_request)
        normalized_prefs, feasibility_adjustments = _apply_budget_feasibility_guard(
            parsed_request=parsed_request,
            weighted_cost_profile=weighted_cost_profile,
            normalized_prefs=normalized_prefs
        )
        effective_city_profiles = _get_effective_city_profiles(parsed_request)
        effective_cities = _get_effective_cities(parsed_request)
        city_plan_mode = _get_city_plan_mode(parsed_request)

        primary_profile, destination_profile, destination_profile_tier = _get_primary_profile(parsed_request)

        profile_cost_level = weighted_cost_profile.get("cost_level") or primary_profile.get("cost_level")
        final_cost_level = _merge_cost_level(
            llm_cost_level=llm_cost_level,
            profile_cost_level=profile_cost_level,
            parsed_request=parsed_request,
            normalized_prefs=normalized_prefs
        )

        season_factor = _get_season_factor(parsed_request, destination_profile)

        normalized_prefs, budget_pressure_adjustments = _apply_budget_pressure_downgrades(
            parsed_request=parsed_request,
            cost_profile=weighted_cost_profile,
            normalized_prefs=normalized_prefs,
            season_factor=season_factor,
            budget_strategy=budget_strategy,
        )
        if normalized_prefs.get("budget_pressure_mode") == "yes":
            budget_strategy = "tight"
            final_cost_level = _merge_cost_level(
                llm_cost_level=llm_cost_level,
                profile_cost_level=profile_cost_level,
                parsed_request=parsed_request,
                normalized_prefs=normalized_prefs,
            )

        transport_cost = _estimate_transport_cost(
            parsed_request, weighted_cost_profile, normalized_prefs, season_factor
        )
        hotel_cost = _estimate_hotel_cost(
            parsed_request, weighted_cost_profile, normalized_prefs, season_factor
        )
        food_cost = _estimate_food_cost(
            parsed_request, weighted_cost_profile, normalized_prefs, season_factor
        )
        attraction_cost = _estimate_attraction_cost(
            parsed_request, weighted_cost_profile, normalized_prefs, season_factor
        )

        subtotal = transport_cost + hotel_cost + food_cost + attraction_cost
        buffer_cost = _estimate_buffer_cost(subtotal, budget_strategy)

        estimated_cost = {
            "transport": transport_cost,
            "hotel": hotel_cost,
            "food": food_cost,
            "attractions": attraction_cost,
            "buffer": buffer_cost
        }

        baseline_ratio = _build_baseline_ratio(
            normalized_prefs=normalized_prefs,
            budget_strategy=budget_strategy,
            parsed_request=parsed_request
        )
        final_ratio = _merge_ratios(baseline_ratio, llm_ratio)

        adjusted_plan, budget_status, optimizer_adjustments = _fit_estimated_cost_to_budget(
            estimated=estimated_cost,
            total_budget=total_budget,
            normalized_prefs=normalized_prefs
        )

        estimated_cost_final, optional_upgrades, unallocated_budget, expansion_adjustments, normalized_prefs = _split_under_budget_plan(
            parsed_request=parsed_request,
            cost_profile=weighted_cost_profile,
            adjusted_plan=adjusted_plan,
            total_budget=total_budget,
            normalized_prefs=normalized_prefs,
            season_factor=season_factor,
            final_ratio=final_ratio,
            allow_optional_upgrades=not _is_budget_cap_mode(parsed_request)
        )

        budget_plan = {
            k: estimated_cost_final.get(k, 0) + optional_upgrades.get(k, 0)
            for k in ["transport", "hotel", "food", "attractions", "buffer"]
        }

        budget_plan = _normalize_plan_to_total(budget_plan, total_budget)
        budget_total_after_normalize = sum(budget_plan.values())
        if budget_total_after_normalize > total_budget:
            raise RuntimeError(
                f"Budget normalization failed: plan total {budget_total_after_normalize} exceeds total budget {total_budget}"
            )
        unallocated_budget = 0
        remaining_budget = total_budget - budget_total_after_normalize

        optional_upgrades = {
            key: max(0, budget_plan.get(key, 0) - estimated_cost_final.get(key, 0))
            for key in ["transport", "hotel", "food", "attractions", "buffer"]
        }

        split_total = sum(estimated_cost_final.values()) + sum(optional_upgrades.values())
        split_diff = total_budget - split_total
        if split_diff != 0:
            optional_upgrades["buffer"] += split_diff

        if budget_status == "under_budget":
            budget_status = "within_budget"

        city_budget_plan = _allocate_city_category_budgets(
            effective_city_profiles=effective_city_profiles,
            total_hotel_budget=budget_plan.get("hotel", 0),
            total_food_budget=budget_plan.get("food", 0),
            total_attractions_budget=budget_plan.get("attractions", 0),
            normalized_prefs=normalized_prefs
        )

        result["origin"] = parsed_request.get("origin")
        result["destination"] = parsed_request.get("destination", [])
        result["country"] = parsed_request.get("country")
        result["num_people"] = parsed_request.get("num_people")
        result["travel_dates"] = parsed_request.get("travel_dates", {})
        result["duration_days"] = parsed_request.get("duration_days")
        result["budget"] = parsed_request.get("budget", {})
        result["preferences"] = parsed_request.get("preferences", {})
        result["planning_signals"] = parsed_request.get("planning_signals", {})

        result["estimated_cost"] = estimated_cost_final
        result["optional_upgrades"] = optional_upgrades
        result["budget_plan"] = budget_plan
        result["city_budget_plan"] = city_budget_plan
        result["budget_strategy"] = budget_strategy
        result["cost_level"] = final_cost_level
        result["budget_status"] = budget_status
        result["remaining_budget"] = remaining_budget
        result["unallocated_budget"] = unallocated_budget
        result["normalized_preferences"] = normalized_prefs

        city_names = [item["city"] for item in effective_cities] if effective_cities else []


        richer_preference_impact = []
        if normalized_prefs.get("hotel_level_detail") == "luxury":
            richer_preference_impact.append({
                "preference": "preferences.hotel_level",
                "affected_category": "hotel",
                "direction": "increase",
                "magnitude": "high",
                "reason": "Luxury hotel level raises the expected nightly hotel spend."
            })
        if normalized_prefs.get("hotel_central") == "yes":
            richer_preference_impact.append({
                "preference": "preferences.hotel_features.central",
                "affected_category": "hotel",
                "direction": "increase",
                "magnitude": "medium",
                "reason": "Central location usually increases hotel rates."
            })
        if normalized_prefs.get("hotel_clean") == "yes":
            richer_preference_impact.append({
                "preference": "preferences.hotel_features.clean",
                "affected_category": "hotel",
                "direction": "increase",
                "magnitude": "low",
                "reason": "A clean/high-standard hotel slightly raises the hotel budget."
            })
        if normalized_prefs.get("wants_fine_dining") == "yes":
            richer_preference_impact.append({
                "preference": "preferences.food_preferences.fine_dining",
                "affected_category": "food",
                "direction": "increase",
                "magnitude": "medium",
                "reason": "Fine dining preference raises the food allocation."
            })
        if normalized_prefs.get("wants_local_food") == "yes":
            richer_preference_impact.append({
                "preference": "preferences.food_preferences.local",
                "affected_category": "food",
                "direction": "rebalance",
                "magnitude": "low",
                "reason": "Local food tends to keep food allocation realistic rather than aggressively upscale."
            })
        llm_adjustments = llm_adjustments + feasibility_adjustments + budget_pressure_adjustments
        result["allocation_basis"]["destination_profile"] = destination_profile
        result["allocation_basis"]["destination_profile_tier"] = destination_profile_tier
        result["allocation_basis"]["effective_cities"] = effective_cities
        result["allocation_basis"]["city_plan_mode"] = city_plan_mode
        result["allocation_basis"]["llm_adjustments"] = (
            llm_adjustments + optimizer_adjustments + expansion_adjustments
        )
        result["allocation_basis"]["preference_impact"] = preference_impact + richer_preference_impact
        result["allocation_basis"]["llm_ratio"] = llm_ratio
        result["allocation_basis"]["baseline_ratio"] = baseline_ratio
        result["allocation_basis"]["final_ratio"] = final_ratio
        result["allocation_basis"]["season_factor"] = round(season_factor, 3)
        result["allocation_basis"]["cost_level_sources"] = {
            "llm_cost_level": llm_cost_level,
            "profile_cost_level": profile_cost_level,
            "final_cost_level": final_cost_level
        }

        assumptions.append(
            f"Primary destination profile '{destination_profile}' was used as the leading reference."
        )
        assumptions.append(
            f"Profile tier used for the primary destination: {destination_profile_tier}."
        )
        assumptions.append(
            f"Effective planning cities used by Agent2: {city_names if city_names else ['(none)']}."
        )
        assumptions.append(f"City planning mode: {city_plan_mode}.")
        assumptions.append(
            "This version uses a three-layer destination profile system: city profile first, country profile second, and global default as fallback."
        )
        assumptions.append(
            "Multi-city budget allocation uses category-level totals first, then distributes city budgets with day-based and city-cost-weighted allocation."
        )

        if llm_ratio:
            assumptions.append(
                "The LLM suggested a high-level allocation ratio, but the program still validated and merged it with baseline budget rules."
            )

        if sum(estimated_cost.values()) < total_budget:
            assumptions.append(
                "Because the estimated trip cost was below the total budget, the remaining money was fully distributed into optional upgrades for the final executable plan."
            )


        if normalized_prefs.get("hotel_level_detail") == "luxury":
            assumptions.append("Agent2 detected a luxury-tier hotel requirement and protected hotel budget accordingly.")
        if normalized_prefs.get("hotel_central") == "yes" or normalized_prefs.get("hotel_clean") == "yes":
            assumptions.append("Agent2 incorporated hotel feature preferences such as central location / cleanliness into hotel budget interpretation.")
        if normalized_prefs.get("wants_fine_dining") == "yes":
            assumptions.append("Agent2 incorporated fine-dining preference into food budget interpretation.")
        result["assumptions"] = assumptions
        result["meta"]["llm_call_success"] = True
        result["meta"]["raw_response_preview"] = raw_text[:500]
        result["meta"]["error"] = None

        return result

    except Exception as exc:
        result["meta"]["error"] = str(exc)
        return result


# =========================================================
# Debug
# =========================================================

if __name__ == "__main__":
    sample_parsed_request = {
        "origin": "Beijing",
        "destination": [],
        "country": "Thailand",
        "num_people": 2,
        "travel_dates": {
            "start": None,
            "end": None,
            "text": "early May"
        },
        "duration_days": 5,
        "budget": {
            "total": 30000,
            "currency": "CNY"
        },
        "preferences": {
            "flight": "direct flights",
            "hotel": "five star hotels",
            "food": "better food options",
            "attraction_style": None
        },
        "city_plan": {
            "mode": "country_recommendation",
            "cities": [
                {"city": "Bangkok", "days": 3, "source": "system"},
                {"city": "Chiang Mai", "days": 2, "source": "system"}
            ]
        },
        "missing_fields": {
            "critical": [],
            "optional": []
        },
        "is_ready_for_planning": True,
        "parse_status": "success",
        "meta": {
            "llm_called": True,
            "llm_call_success": True,
            "raw_response_preview": None,
            "error": None
        }
    }

    print("=== Budget Agent Backend Status ===")
    print(json.dumps(get_budget_agent_backend_status(), ensure_ascii=False, indent=2))

    print("\n=== Agent 2 Result ===")
    print(json.dumps(travel_budget_agent(sample_parsed_request), ensure_ascii=False, indent=2))
