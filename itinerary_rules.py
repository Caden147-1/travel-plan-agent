from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from planner_schema import (
    DAY_AVAILABLE_HOURS,
    FLIGHT_BUFFER_HOURS,
    LOCAL_TRANSPORT_PER_DAY,
    MAX_DRIVE_HOURS_PER_LEG,
    MEAL_HOURS,
    MID_DISTANCE_CAR_COST,
    POI_VISIT_HOURS,
    REDEYE_TIME_FACTOR,
    normalize_text,
    safe_float,
)


def parse_hhmm(value: str) -> Optional[datetime]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return datetime.strptime(text, "%H:%M")
    except Exception:
        return None


def extract_hour(value: Any) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    for fmt in ["%H:%M", "%Y-%m-%d %H:%M", "%I:%M %p"]:
        try:
            return datetime.strptime(text, fmt).hour
        except Exception:
            pass
    if len(text) >= 2 and text[:2].isdigit():
        return int(text[:2])
    return None


def is_redeye_flight(flight: Dict[str, Any]) -> bool:
    dep_h = extract_hour(flight.get("departure_time"))
    arr_h = extract_hour(flight.get("arrival_time"))
    night_dep = dep_h is not None and (dep_h >= 22 or dep_h <= 5)
    night_arr = arr_h is not None and (0 <= arr_h <= 7)
    return bool(night_dep or night_arr)


def estimate_flight_time_cost_hours(flight: Dict[str, Any]) -> float:
    duration_min = safe_float(flight.get("duration_minutes"), 0.0)
    hours = duration_min / 60.0 if duration_min > 0 else 2.0
    total = hours + FLIGHT_BUFFER_HOURS
    if is_redeye_flight(flight):
        total *= REDEYE_TIME_FACTOR
    return round(max(0.5, total), 2)


def estimate_mid_distance_transport_cost(city_tier: str) -> int:
    tier = normalize_text(city_tier).lower()
    if tier not in MID_DISTANCE_CAR_COST:
        tier = "medium"
    return int(MID_DISTANCE_CAR_COST[tier])


def estimate_local_transport_cost_per_day() -> int:
    return int(LOCAL_TRANSPORT_PER_DAY)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    import math

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def estimate_drive_time_hours(from_place: Dict[str, Any], to_place: Dict[str, Any], intra_city_speed_kmh: float = 25.0, intercity_speed_kmh: float = 60.0, is_intercity: bool = False) -> float:
    f_lat = safe_float(from_place.get("lat"), None)
    f_lng = safe_float(from_place.get("lng"), None)
    t_lat = safe_float(to_place.get("lat"), None)
    t_lng = safe_float(to_place.get("lng"), None)

    speed = intercity_speed_kmh if is_intercity else intra_city_speed_kmh

    if f_lat is not None and f_lng is not None and t_lat is not None and t_lng is not None:
        km = haversine_km(f_lat, f_lng, t_lat, t_lng)
        return round(max(0.25, km / max(1.0, speed)), 2)

    route_km = safe_float(to_place.get("route_km"), None)
    if route_km is not None and route_km > 0:
        return round(max(0.25, route_km / max(1.0, speed)), 2)

    return 1.0 if not is_intercity else 3.0


def build_day_template(slot: Dict[str, Any], hotel: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "date": normalize_text(slot.get("date")),
        "city": normalize_text(slot.get("city")),
        "hotel": {
            "name": normalize_text((hotel or {}).get("name")),
            "area": normalize_text((hotel or {}).get("area")),
            "estimated_total_cost": safe_float((hotel or {}).get("estimated_total_cost"), 0.0),
        } if hotel else None,
        "time_budget_hours": float(DAY_AVAILABLE_HOURS),
        "used_hours": 0.0,
        "transport_cost_estimated": 0.0,
        "events": [],
        "constraints": {
            "within_day_hours": True,
            "max_drive_leg_le_5h": True,
        },
        "warnings": [],
    }


def append_event(day_plan: Dict[str, Any], event_type: str, name: str, duration_h: float, cost_estimated: float = 0.0, meta: Optional[Dict[str, Any]] = None) -> None:
    events = day_plan.setdefault("events", [])

    if events:
        start = parse_hhmm(normalize_text(events[-1].get("end"))) or parse_hhmm("08:00")
    else:
        start = parse_hhmm("08:00")

    dur = max(0.0, safe_float(duration_h, 0.0))
    end = start + timedelta(minutes=int(round(dur * 60)))

    events.append({
        "type": normalize_text(event_type),
        "name": normalize_text(name),
        "start": start.strftime("%H:%M"),
        "end": end.strftime("%H:%M"),
        "duration_h": round(dur, 2),
        "cost_estimated": round(max(0.0, safe_float(cost_estimated, 0.0)), 2),
        "meta": meta or {},
    })

    day_plan["used_hours"] = round(safe_float(day_plan.get("used_hours"), 0.0) + dur, 2)


def trim_day_events_to_available_hours(day_plan: Dict[str, Any]) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = day_plan.get("events", []) or []

    while safe_float(day_plan.get("used_hours"), 0.0) > DAY_AVAILABLE_HOURS and events:
        idx = None
        for i in range(len(events) - 1, -1, -1):
            if normalize_text(events[i].get("type")) in {"poi", "meal"}:
                idx = i
                break
        if idx is None:
            idx = len(events) - 1
        removed = events.pop(idx)
        day_plan["used_hours"] = round(max(0.0, safe_float(day_plan.get("used_hours"), 0.0) - safe_float(removed.get("duration_h"), 0.0)), 2)

    if safe_float(day_plan.get("used_hours"), 0.0) > DAY_AVAILABLE_HOURS:
        day_plan.setdefault("warnings", []).append(
            f"Day time overrun: unable to trim below {DAY_AVAILABLE_HOURS}h."
        )

    return day_plan


def validate_day_time_and_drive(day_plan: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    within_day_hours = safe_float(day_plan.get("used_hours"), 0.0) <= DAY_AVAILABLE_HOURS
    if not within_day_hours:
        warnings.append(f"Time constraint violated: used {day_plan.get('used_hours')}h > {DAY_AVAILABLE_HOURS}h.")

    drive_ok = True
    for ev in day_plan.get("events", []) or []:
        if normalize_text(ev.get("type")) == "drive":
            d = safe_float(ev.get("duration_h"), 0.0)
            if d > MAX_DRIVE_HOURS_PER_LEG:
                drive_ok = False
                warnings.append(f"Drive constraint violated: {d}h > {MAX_DRIVE_HOURS_PER_LEG}h.")

    return {
        "within_day_hours": within_day_hours,
        "max_drive_leg_le_5h": drive_ok,
        "warnings": warnings,
    }


def poi_visit_hours() -> float:
    return float(POI_VISIT_HOURS)


def meal_hours() -> float:
    return float(MEAL_HOURS)
