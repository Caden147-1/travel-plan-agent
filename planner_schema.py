from __future__ import annotations

from typing import Any, Dict, List

DAY_AVAILABLE_HOURS = 13.0
POI_VISIT_HOURS = 1.5
MEAL_HOURS = 1.0
FLIGHT_BUFFER_HOURS = 2.0
REDEYE_TIME_FACTOR = 0.5
LOCAL_TRANSPORT_PER_DAY = 200
MID_DISTANCE_CAR_COST = {"low": 300, "medium": 400, "high": 500}
MAX_DRIVE_HOURS_PER_LEG = 5.0
HOTEL_CLUSTER_RADIUS_KM = 5.0
HOTEL_CLUSTER_MAX_ROUTE_KM = 10.0


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
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


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_router_payload(router_output: Dict[str, Any]) -> Dict[str, Any]:
    payload = router_output if isinstance(router_output, dict) else {}
    trip_summary = payload.get("trip_summary", {}) or {}
    preferences = trip_summary.get("preferences", {}) or {}
    return {
        "trip_summary": trip_summary,
        "city_date_ranges": payload.get("city_date_ranges", {}) or {},
        "city_budget_plan": payload.get("city_budget_plan", []) or [],
        "selected_hotels": payload.get("selected_hotels", []) or [],
        "selected_restaurants": payload.get("selected_restaurants", []) or [],
        "selected_pois": payload.get("selected_pois", []) or [],
        "selected_all_flight_legs": payload.get("selected_all_flight_legs", []) or [],
        "must_visit_attractions": preferences.get("must_visit_attractions", []) or [],
    }


def empty_agent4_output(error: str) -> Dict[str, Any]:
    return {
        "success": False,
        "stage": "agent4_itinerary_planning",
        "error": normalize_text(error),
        "daily_plan": [],
        "warnings": [normalize_text(error)] if normalize_text(error) else [],
    }
