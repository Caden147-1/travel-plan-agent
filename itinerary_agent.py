from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

DAY_AVAILABLE_HOURS = 13.0
POI_VISIT_HOURS = 1.5
MEAL_HOURS = 1.0
FLIGHT_BUFFER_HOURS = 2.0
REDEYE_TIME_FACTOR = 0.5
DEFAULT_DAY_START = "09:00"
DEFAULT_DAY_END = "20:30"
NORMAL_BREAKFAST_START = "09:00"
NORMAL_BREAKFAST_END = "10:30"
NORMAL_LUNCH_START = "12:00"
NORMAL_LUNCH_END = "14:00"
NORMAL_DINNER_START = "18:00"
NORMAL_DINNER_END = "20:00"
LOCAL_TRANSPORT_RATE_BY_TIER = {"low": 1.0, "medium": 2.0, "high": 8.0, "ultra": 30.0}
MID_DISTANCE_CAR_COST = {"low": 300, "medium": 400, "high": 500, "ultra": 800}
MAX_DRIVE_HOURS_PER_LEG = 5.0
HOTEL_CLUSTER_RADIUS_KM = 5.0
HOTEL_CLUSTER_MAX_ROUTE_KM = 10.0
HOTEL_ACTIVITY_RADIUS_KM = 10.0
HOTEL_MEAL_FALLBACK_RADIUS_KM = 15.0
HEIGIT_API_KEY = os.getenv("HEIGIT_API_KEY", "")
HEIGIT_BASE_URL = os.getenv("HEIGIT_BASE_URL", "https://api.openrouteservice.org")
_CITY_COORD_CACHE: Dict[str, Dict[str, float]] = {}
_ROUTE_METRICS_CACHE: Dict[str, Dict[str, float]] = {}
_CITY_COORD_CACHE_LOCK = threading.Lock()
_ROUTE_METRICS_CACHE_LOCK = threading.Lock()

PLACE_EVENT_TYPES = {"poi", "meal", "checkin", "checkout"}
FIXED_TIME_EVENT_TYPES = {"flight", "drive"}
DAILY_REQUIRED_MEALS = 3
DAILY_MIN_POIS = 2
CHECKIN_TRANSFER_MINUTES = 20
LOW_QUALITY_MEAL_COST_FLOOR = 30.0
LOW_QUALITY_POI_COST_FLOOR = 20.0
HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES = 15
DEFAULT_INTER_EVENT_TRANSFER_MINUTES = 15
INTERNATIONAL_LOCAL_TRANSPORT_RATE_PER_MINUTE = 7.0
CHINA_COUNTRY_ALIASES = {"china", "cn", "中国", "中华人民共和国", "people's republic of china", "prc"}


def _build_flight_day_constraints(selected_all_flight_legs: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    constraints: Dict[str, Dict[str, str]] = {}
    if not isinstance(selected_all_flight_legs, list):
        return constraints

    for leg in selected_all_flight_legs:
        if not isinstance(leg, dict):
            continue
        dep_date = _normalize_text(leg.get("departure_date"))
        if not dep_date:
            continue
        flights = leg.get("selected_flights") or []
        leg_type = _normalize_text(leg.get("leg_type"))
        if not flights and leg_type == "outbound" and not _normalize_text(leg.get("transport_mode")):
            day_constraints = constraints.setdefault(dep_date, {})
            day_constraints["skip_city_schedule"] = "true"
            day_constraints["skip_city_local_transport"] = "true"
            day_constraints["missing_required_outbound_transport"] = "true"
            continue
        if not flights and _normalize_text(leg.get("transport_mode")) in {"direct_drive", "rental_car", "drive"}:
            day_constraints = constraints.setdefault(dep_date, {})
            drive_h = _safe_float(leg.get("direct_drive_hours"), 0.0)
            if drive_h <= 0:
                drive_h = _safe_float(
                    leg.get("rental_car", {}).get("estimated_drive_hours")
                    if isinstance(leg.get("rental_car"), dict)
                    else None,
                    0.0,
                )
            if drive_h > 0 and leg_type == "outbound":
                arrival = _add_minutes_to_time("08:00", int(round(drive_h * 60)))
                if arrival:
                    day_constraints["schedule_start"] = arrival
                    if _event_minutes_from_hhmm(arrival) and _event_minutes_from_hhmm(arrival) >= 18 * 60:
                        day_constraints["skip_city_schedule"] = "true"
            continue
        if not flights or not isinstance(flights[0], dict):
            continue
        flight = flights[0]
        day_constraints = constraints.setdefault(dep_date, {})
        if leg_type == "outbound":
            dep_minutes = _event_minutes_from_hhmm(flight.get("departure_time"))
            arr_minutes = _event_minutes_from_hhmm(flight.get("arrival_time"))
            arrival = _format_time_value(flight.get("arrival_time"))
            if arrival:
                arrival_date = dep_date
                if dep_minutes is not None and arr_minutes is not None and arr_minutes <= dep_minutes:
                    dep_dt = _parse_date(dep_date)
                    if dep_dt:
                        arrival_date = (dep_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                        day_constraints["skip_city_schedule"] = "true"
                        day_constraints["skip_city_local_transport"] = "true"
                arrival_constraints = constraints.setdefault(arrival_date, {})
                recovery_start = _add_minutes_to_time(arrival, 30) or arrival
                arrival_constraints["schedule_start"] = _max_time_value(
                    recovery_start,
                    DEFAULT_DAY_START,
                ) or recovery_start
        elif leg_type == "inbound":
            departure = _format_time_value(flight.get("departure_time"))
            if departure:
                day_constraints["schedule_end"] = departure
    return constraints


def run_itinerary_agent(router_output: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_agent3_payload(router_output)
    local_transport_tier = _local_transport_tier_from_cost_level(normalized.get("cost_level"))
    local_transport_rate = _local_transport_rate_per_minute(
        local_transport_tier,
        is_international=_is_international_trip(normalized),
    )
    city_budget_map = _build_city_budget_map(normalized.get("city_budget_plan", []))
    daily_slots = _extract_daily_city_slots(normalized.get("city_date_ranges", {}))
    flight_day_constraints = _build_flight_day_constraints(normalized.get("selected_all_flight_legs", []))

    if not daily_slots:
        return {
            "success": False,
            "stage": "agent4_itinerary_planning",
            "error": "No daily slots available from city_date_ranges.",
            "daily_plan": [],
            "warnings": ["Missing city_date_ranges or unable to expand to daily slots."]
        }

    hotels = normalized.get("selected_hotels", [])
    pois = normalized.get("selected_pois", [])
    restaurants = normalized.get("selected_restaurants", [])

    city_hotel_map: Dict[str, Optional[Dict[str, Any]]] = {}
    city_hotel_clusters_map: Dict[str, List[Dict[str, Any]]] = {}
    city_candidates_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    must_visit_names = normalized.get("must_visit_attractions", [])
    must_visit_remaining_by_city: Dict[str, List[Dict[str, Any]]] = {}

    unique_cities = []
    for slot in daily_slots:
        city = _normalize_text(slot.get("city"))
        if city and city not in unique_cities:
            unique_cities.append(city)

    for city in unique_cities:
        city_hotel_clusters = _cluster_points_by_hotel_radius(
            hotels=hotels,
            pois=pois,
            restaurants=restaurants,
            city=city,
            radius_km=HOTEL_CLUSTER_RADIUS_KM,
            max_route_km=HOTEL_CLUSTER_MAX_ROUTE_KM,
        )
        city_hotel_clusters_map[city] = city_hotel_clusters
        primary_hotel = _select_primary_hotel_for_city(city=city, hotel_clusters=city_hotel_clusters)
        city_hotel_map[city] = primary_hotel

        city_candidates = _collect_city_candidates(
            city=city,
            selected_pois=pois,
            selected_restaurants=restaurants,
        )
        city_candidates = _filter_city_candidates_by_hotel_radius(
            city_candidates=city_candidates,
            hotel=primary_hotel,
        )
        city_candidates_map[city] = city_candidates

        must_city, _ = _split_must_visit_and_regular_pois(
            pois=city_candidates.get("pois", []),
            must_visit_names=must_visit_names,
        )
        must_visit_remaining_by_city[city] = list(must_city)

    daily_plans: List[Dict[str, Any]] = []
    restaurant_last_used_day_index: Dict[str, int] = {}
    restaurant_cooldown_days = 2
    used_poi_keys: set = set()
    used_restaurant_keys: set = set()
    previous_hotel_by_city: Dict[str, str] = {}

    for day_index, slot in enumerate(daily_slots, start=1):
        city = _normalize_text(slot.get("city"))
        selected_cluster = _select_daily_hotel_cluster(
            city=city,
            hotel_clusters=city_hotel_clusters_map.get(city, []),
            used_poi_keys=used_poi_keys,
            used_restaurant_keys=used_restaurant_keys,
            previous_hotel_name=previous_hotel_by_city.get(city, ""),
        )
        hotel = (
            selected_cluster.get("hotel")
            if isinstance(selected_cluster, dict) and isinstance(selected_cluster.get("hotel"), dict)
            else city_hotel_map.get(city)
        )
        day_plan = _init_day_plan(slot=slot, hotel=hotel)
        day_plan["local_transport_tier"] = local_transport_tier
        day_plan["local_transport_rate_per_minute"] = local_transport_rate
        day_plan["local_transport_is_international"] = _is_international_trip(normalized)
        day_plan.update(flight_day_constraints.get(_normalize_text(slot.get("date")), {}))
        if isinstance(hotel, dict):
            previous_hotel_by_city[city] = _normalize_text(hotel.get("name"))

        day_budget = city_budget_map.get(city, {})
        if isinstance(selected_cluster, dict):
            city_candidates = {
                "pois": selected_cluster.get("pois", []) or [],
                "restaurants": selected_cluster.get("restaurants", []) or [],
                "radius_meta": {
                    "poi_limit_km": HOTEL_ACTIVITY_RADIUS_KM,
                    "meal_limit_km": HOTEL_ACTIVITY_RADIUS_KM,
                    "meal_fallback_used": False,
                    "poi_in_radius_count": len(selected_cluster.get("pois", []) or []),
                    "meal_in_10km_count": len(selected_cluster.get("restaurants", []) or []),
                    "meal_in_15km_count": len(selected_cluster.get("restaurants", []) or []),
                    "hotel_missing_for_radius_filter": False,
                    "hotel_switched_for_variety": _normalize_text(previous_hotel_by_city.get(city)) != _normalize_text((hotel or {}).get("name")),
                },
            }
        else:
            city_candidates = city_candidates_map.get(city, {"pois": [], "restaurants": [], "radius_meta": {}})
        radius_meta = city_candidates.get("radius_meta", {}) if isinstance(city_candidates, dict) else {}
        if isinstance(radius_meta, dict):
            if bool(radius_meta.get("hotel_missing_for_radius_filter")):
                day_plan.setdefault("warnings", []).append(
                    "Hotel coordinate missing; cannot strictly enforce 10km activity radius."
                )
            elif bool(radius_meta.get("hotel_switched_for_variety")):
                day_plan.setdefault("warnings", []).append(
                    "Switched hotel area to unlock new nearby attractions and restaurants."
                )
            elif bool(radius_meta.get("meal_fallback_used")):
                day_plan.setdefault("warnings", []).append(
                    f"No restaurants found within {int(HOTEL_ACTIVITY_RADIUS_KM)}km of hotel; meal search relaxed to {int(HOTEL_MEAL_FALLBACK_RADIUS_KM)}km."
                )
            if int(_safe_float(radius_meta.get("poi_in_radius_count"), 0)) <= 0:
                day_plan.setdefault("warnings", []).append(
                    f"No POIs found within {int(HOTEL_ACTIVITY_RADIUS_KM)}km of hotel; POI options may be limited."
                )
        must_visit_left = must_visit_remaining_by_city.get(city, [])

        schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start"))
        if _normalize_text(day_plan.get("skip_city_schedule")).lower() == "true":
            if _normalize_text(day_plan.get("missing_required_outbound_transport")).lower() == "true":
                day_plan.setdefault("warnings", []).append(
                    "City activities skipped because the required outbound ticket was not selected."
                )
            else:
                day_plan.setdefault("warnings", []).append(
                    "City activities skipped because the selected outbound flight arrives the next day."
                )
        elif schedule_start is None or schedule_start < 18 * 60:
            day_plan = _schedule_city_day(
                day_plan=day_plan,
                city_candidates=city_candidates,
                city_budget=day_budget,
                must_visit_left=must_visit_left,
                current_day_index=day_index,
                restaurant_last_used_day_index=restaurant_last_used_day_index,
                restaurant_cooldown_days=restaurant_cooldown_days,
                used_poi_keys=used_poi_keys,
                used_restaurant_keys=used_restaurant_keys,
            )
        else:
            day_plan.setdefault("warnings", []).append(
                "City activities skipped because the selected outbound flight arrives too late for daytime sightseeing."
            )

        day_plan = _trim_day_to_available_hours(day_plan)
        daily_plans.append(day_plan)

    daily_plans = _insert_checkin_checkout_events(daily_plans=daily_plans)
    daily_plans = _insert_flight_and_transfer_events(
        daily_plans=daily_plans,
        selected_all_flight_legs=normalized.get("selected_all_flight_legs", []),
    )
    daily_plans = _enforce_unique_poi_visits(daily_plans)
    daily_plans = _refill_missing_pois_after_cleanup(
        daily_plans=daily_plans,
        city_candidates_map=city_candidates_map,
    )
    for day in daily_plans:
        _ensure_checkin_before_first_activity(day)
        _shift_flexible_events_after_hotel_windows(day)
        _ensure_hotel_day_boundary_transfers(day)
        _enforce_meal_time_windows(day)
        _shift_flexible_events_after_hotel_windows(day)
        _sanitize_transfer_timeline(day)
        _ensure_adjacent_event_transfers(day)
        _resolve_timeline_overlaps(day)
        _compress_flexible_timeline(day)
        _ensure_hotel_day_boundary_transfers(day)
        _sanitize_transfer_timeline(day)
        _ensure_adjacent_event_transfers(day)
        _resolve_timeline_overlaps(day)
        _sort_and_recompute_day(day)

    total_budget = float(_safe_float(normalized.get("trip_summary", {}).get("budget", {}).get("total"), 0.0))
    budget_trim_warnings = _enforce_total_budget_hard_cap(daily_plans, total_budget)
    budget_downgrade_warnings = _downgrade_lifestyle_costs_for_budget(daily_plans, total_budget)
    for day in daily_plans:
        _ensure_checkin_before_first_activity(day)
        _shift_flexible_events_after_hotel_windows(day)
        _ensure_hotel_day_boundary_transfers(day)
        _enforce_meal_time_windows(day)
        _shift_flexible_events_after_hotel_windows(day)
        _sanitize_transfer_timeline(day)
        _ensure_adjacent_event_transfers(day)
        _resolve_timeline_overlaps(day)
        _compress_flexible_timeline(day)
        _ensure_hotel_day_boundary_transfers(day)
        _sanitize_transfer_timeline(day)
        _ensure_adjacent_event_transfers(day)
        _resolve_timeline_overlaps(day)
        _sort_and_recompute_day(day)

    validated_plans: List[Dict[str, Any]] = []
    for day in daily_plans:
        checks = _validate_day_constraints(day)
        day["constraints"] = {
            "within_day_hours": checks.get("within_day_hours", True),
            "max_drive_leg_le_5h": checks.get("max_drive_leg_le_5h", True),
        }
        for w in checks.get("warnings", []):
            day.setdefault("warnings", []).append(w)
        validated_plans.append(day)

    budget_check = _validate_budget_constraints(
        daily_plans=validated_plans,
        total_budget=total_budget,
    )

    must_visit_total = len(must_visit_names)
    must_visit_done = 0
    covered = set()
    for day in validated_plans:
        for event in day.get("events", []):
            meta = event.get("meta", {}) or {}
            matched = meta.get("must_visit_name")
            if matched:
                covered.add(_normalize_text(matched).lower())
    for name in must_visit_names:
        if _normalize_text(name).lower() in covered:
            must_visit_done += 1

    must_uncovered = []
    for name in must_visit_names:
        if _normalize_text(name).lower() not in covered:
            must_uncovered.append(name)

    warnings = _build_itinerary_warnings(
        daily_plans=validated_plans,
        must_visit_uncovered=must_uncovered,
    )
    warnings.extend(budget_downgrade_warnings)
    warnings.extend(budget_trim_warnings)

    cost_rollup = _build_cost_rollup(validated_plans)
    itinerary_summary = _build_itinerary_summary(
        daily_plans=validated_plans,
        budget_check=budget_check,
        must_visit_total=must_visit_total,
        must_visit_done=must_visit_done,
    )

    return _format_agent4_output(
        daily_plans=validated_plans,
        itinerary_summary=itinerary_summary,
        cost_rollup=cost_rollup,
        warnings=warnings,
    )


def _normalize_agent3_payload(router_output: Dict[str, Any]) -> Dict[str, Any]:
    payload = router_output if isinstance(router_output, dict) else {}
    trip_summary = payload.get("trip_summary", {}) or {}
    preferences = trip_summary.get("preferences", {}) or {}
    agent2_summary = payload.get("agent2_summary", {}) or {}

    return {
        "trip_summary": trip_summary,
        "country": trip_summary.get("country"),
        "cost_level": payload.get("cost_level") or agent2_summary.get("cost_level"),
        "city_date_ranges": payload.get("city_date_ranges", {}) or {},
        "city_budget_plan": payload.get("city_budget_plan", []) or [],
        "selected_hotels": payload.get("selected_hotels", []) or [],
        "selected_restaurants": payload.get("selected_restaurants", []) or [],
        "selected_pois": payload.get("selected_pois", []) or [],
        "selected_all_flight_legs": payload.get("selected_all_flight_legs", []) or [],
        "must_visit_attractions": preferences.get("must_visit_attractions", []) or [],
    }


def _extract_daily_city_slots(city_date_ranges: Dict[str, Any]) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []

    if not isinstance(city_date_ranges, dict):
        return slots

    for city, info in city_date_ranges.items():
        if not isinstance(info, dict):
            continue
        check_in = _normalize_text(info.get("check_in"))
        check_out = _normalize_text(info.get("check_out"))
        if not check_in or not check_out:
            days = max(1, int(_safe_float(info.get("days"), 1)))
            for i in range(days):
                slots.append({
                    "date": f"day_{i+1}",
                    "city": _normalize_text(city),
                    "day_index_in_city": i + 1,
                    "days_in_city": days,
                    "check_in": check_in or None,
                    "check_out": check_out or None,
                })
            continue

        start_dt = _parse_date(check_in)
        end_dt = _parse_date(check_out)
        if not start_dt or not end_dt:
            continue

        delta = (end_dt - start_dt).days
        if delta <= 0:
            delta = 1
        explicit_days = int(_safe_float(info.get("days"), 0))
        if explicit_days > delta:
            delta = explicit_days

        for i in range(delta):
            day = start_dt + timedelta(days=i)
            slots.append({
                "date": day.strftime("%Y-%m-%d"),
                "city": _normalize_text(city),
                "day_index_in_city": i + 1,
                "days_in_city": delta,
                "check_in": check_in,
                "check_out": check_out,
            })

    def _slot_key(s: Dict[str, Any]) -> Tuple[str, str]:
        return (_normalize_text(s.get("date")), _normalize_text(s.get("city")))

    slots.sort(key=_slot_key)
    return slots


def _build_city_budget_map(city_budget_plan: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    if not isinstance(city_budget_plan, list):
        return result

    for item in city_budget_plan:
        if not isinstance(item, dict):
            continue
        city = _normalize_text(item.get("city"))
        if not city:
            continue
        result[city] = {
            "hotel_budget": float(_safe_float(item.get("hotel_budget"), 0.0)),
            "food_budget": float(_safe_float(item.get("food_budget"), 0.0)),
            "attractions_budget": float(_safe_float(item.get("attractions_budget"), 0.0)),
            "days": float(_safe_float(item.get("days"), 1.0)),
        }
    return result


def _cluster_points_by_hotel_radius(
    hotels: List[Dict[str, Any]],
    pois: List[Dict[str, Any]],
    restaurants: List[Dict[str, Any]],
    city: str,
    radius_km: float = 5.0,
    max_route_km: float = 10.0,
) -> List[Dict[str, Any]]:
    city_hotels = [h for h in hotels if _normalize_text(h.get("city")).lower() == city.lower()]
    city_pois = [p for p in pois if _normalize_text(p.get("city")).lower() == city.lower()]
    city_restaurants = [r for r in restaurants if _normalize_text(r.get("city")).lower() == city.lower()]

    clusters: List[Dict[str, Any]] = []

    if not city_hotels:
        return clusters

    for hotel in city_hotels:
        h_lat = _safe_float(hotel.get("lat"), None)
        h_lng = _safe_float(hotel.get("lng"), None)

        poi_in = []
        for p in city_pois:
            route_km = _route_distance_km_between_places(hotel, p)
            if route_km is None:
                poi_in.append(p)
                continue
            if route_km <= max_route_km and (
                route_km <= radius_km
                or _straight_line_distance_km_between_places(hotel, p) <= radius_km
            ):
                poi_in.append(p)

        rest_in = []
        for r in city_restaurants:
            route_km = _route_distance_km_between_places(hotel, r)
            if route_km is None:
                rest_in.append(r)
                continue
            if route_km <= max_route_km and (
                route_km <= radius_km
                or _straight_line_distance_km_between_places(hotel, r) <= radius_km
            ):
                rest_in.append(r)

        clusters.append({
            "city": city,
            "hotel": hotel,
            "pois": poi_in,
            "restaurants": rest_in,
            "coverage_count": len(poi_in) + len(rest_in),
        })

    clusters.sort(
        key=lambda x: (
            int(x.get("coverage_count", 0)),
            _safe_float((x.get("hotel") or {}).get("rating"), 0.0),
            _safe_float((x.get("hotel") or {}).get("stars"), 0.0),
        ),
        reverse=True,
    )
    return clusters


def _select_primary_hotel_for_city(
    city: str,
    hotel_clusters: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not hotel_clusters:
        return None
    top = hotel_clusters[0]
    hotel = top.get("hotel")
    if isinstance(hotel, dict):
        return hotel
    return None


def _select_daily_hotel_cluster(
    city: str,
    hotel_clusters: List[Dict[str, Any]],
    used_poi_keys: set,
    used_restaurant_keys: set,
    previous_hotel_name: str = "",
) -> Optional[Dict[str, Any]]:
    if not hotel_clusters:
        return None

    ranked: List[Tuple[Tuple[int, int, int, float, float], Dict[str, Any]]] = []
    city_key = _normalize_text(city)
    for cluster in hotel_clusters:
        hotel = cluster.get("hotel") or {}
        hotel_name = _normalize_text(hotel.get("name"))
        pois = cluster.get("pois", []) or []
        restaurants = cluster.get("restaurants", []) or []

        remaining_pois = 0
        for poi in pois:
            key = _poi_dedupe_key(poi, city=city_key)
            if not key or key not in used_poi_keys:
                remaining_pois += 1

        remaining_restaurants = 0
        for restaurant in restaurants:
            key = _restaurant_dedupe_key(restaurant, city=city_key)
            if not key or key not in used_restaurant_keys:
                remaining_restaurants += 1

        same_as_previous_penalty = 1 if previous_hotel_name and hotel_name == previous_hotel_name else 0
        score = (
            remaining_pois + remaining_restaurants,
            remaining_pois,
            remaining_restaurants,
            _safe_float(hotel.get("rating"), 0.0),
            _safe_float(hotel.get("stars"), 0.0),
        )
        ranked.append(((-same_as_previous_penalty, *score), cluster))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else hotel_clusters[0]


def _is_redeye_flight(flight: Dict[str, Any]) -> bool:
    dep = _normalize_text(flight.get("departure_time"))
    arr = _normalize_text(flight.get("arrival_time"))
    dep_h = _extract_hour(dep)
    arr_h = _extract_hour(arr)
    if dep_h is None and arr_h is None:
        return False
    night_dep = dep_h is not None and (dep_h >= 22 or dep_h <= 5)
    night_arr = arr_h is not None and (arr_h >= 0 and arr_h <= 7)
    return bool(night_dep or night_arr)


def _estimate_flight_time_cost_hours(flight: Dict[str, Any]) -> float:
    if _is_redeye_flight(flight):
        return 0.0
    duration_min = _safe_float(flight.get("duration_minutes"), 0.0)
    hours = duration_min / 60.0 if duration_min > 0 else 2.0
    total = hours + FLIGHT_BUFFER_HOURS
    return round(max(0.5, total), 2)


def _needs_transport_placeholder(leg: Dict[str, Any]) -> bool:
    if not isinstance(leg, dict):
        return False
    if leg.get("selected_flights"):
        return False
    if _normalize_text(leg.get("transport_mode")) in {"direct_drive", "rental_car", "drive"}:
        return False
    return _normalize_text(leg.get("leg_type")) in {"outbound", "inbound", "intercity"}


def _estimate_drive_time_hours(
    from_place: Dict[str, Any],
    to_place: Dict[str, Any],
    city: str,
) -> float:
    f_lat = _safe_float(from_place.get("lat"), None)
    f_lng = _safe_float(from_place.get("lng"), None)
    t_lat = _safe_float(to_place.get("lat"), None)
    t_lng = _safe_float(to_place.get("lng"), None)

    if f_lat is not None and f_lng is not None and t_lat is not None and t_lng is not None:
        km = _haversine_km(f_lat, f_lng, t_lat, t_lng)
        speed = 25.0
        return round(max(0.25, km / speed), 2)

    route_km = _safe_float(to_place.get("route_km"), None)
    if route_km is not None and route_km > 0:
        return round(max(0.25, route_km / 25.0), 2)

    return 1.0


def _place_coord(place: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not isinstance(place, dict):
        return None
    lat = _safe_float(place.get("lat"), None)
    lng = _safe_float(place.get("lng"), None)
    if lat is None or lng is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return {"lat": lat, "lng": lng}


def _route_metrics_heigit_coords(origin: Dict[str, float], destination: Dict[str, float]) -> Optional[Dict[str, float]]:
    if not HEIGIT_API_KEY:
        return None

    key = (
        f"{origin.get('lat'):.6f},{origin.get('lng'):.6f}->"
        f"{destination.get('lat'):.6f},{destination.get('lng'):.6f}"
    )
    with _ROUTE_METRICS_CACHE_LOCK:
        cached = _ROUTE_METRICS_CACHE.get(key)
    if cached is not None:
        return dict(cached) if cached else None

    url = f"{HEIGIT_BASE_URL.rstrip('/')}/v2/directions/driving-car"
    headers = {"Authorization": HEIGIT_API_KEY, "Content-Type": "application/json"}
    payload = {
        "coordinates": [
            [origin["lng"], origin["lat"]],
            [destination["lng"], destination["lat"]],
        ]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            with _ROUTE_METRICS_CACHE_LOCK:
                _ROUTE_METRICS_CACHE[key] = {}
            return None
        data = resp.json() if resp.text else {}
        routes = data.get("routes", []) if isinstance(data, dict) else []
        if not routes:
            with _ROUTE_METRICS_CACHE_LOCK:
                _ROUTE_METRICS_CACHE[key] = {}
            return None
        summary = routes[0].get("summary", {}) if isinstance(routes[0], dict) else {}
        duration_sec = _safe_float(summary.get("duration"), 0.0)
        distance_m = _safe_float(summary.get("distance"), 0.0)
        if duration_sec <= 0 or distance_m <= 0:
            with _ROUTE_METRICS_CACHE_LOCK:
                _ROUTE_METRICS_CACHE[key] = {}
            return None
        result = {
            "duration_h": round(duration_sec / 3600.0, 2),
            "distance_km": round(distance_m / 1000.0, 2),
        }
        with _ROUTE_METRICS_CACHE_LOCK:
            _ROUTE_METRICS_CACHE[key] = result
        return dict(result)
    except Exception:
        with _ROUTE_METRICS_CACHE_LOCK:
            _ROUTE_METRICS_CACHE[key] = {}
        return None


def _straight_line_distance_km_between_places(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    a_coord = _place_coord(a)
    b_coord = _place_coord(b)
    if not a_coord or not b_coord:
        return None
    return round(_haversine_km(a_coord["lat"], a_coord["lng"], b_coord["lat"], b_coord["lng"]), 2)


def _route_distance_km_between_places(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    a_coord = _place_coord(a)
    b_coord = _place_coord(b)
    if not a_coord or not b_coord:
        return None
    metrics = _route_metrics_heigit_coords(a_coord, b_coord)
    if metrics:
        return _safe_float(metrics.get("distance_km"), None)
    straight = _straight_line_distance_km_between_places(a, b)
    if straight is None:
        return None
    return round(straight * 1.35, 2)


def _geocode_city_heigit(city: str) -> Optional[Dict[str, float]]:
    city_key = _normalize_text(city).lower()
    if not city_key:
        return None
    with _CITY_COORD_CACHE_LOCK:
        cached = _CITY_COORD_CACHE.get(city_key)
    if cached is not None:
        if cached:
            return cached
        return None
    if not HEIGIT_API_KEY:
        with _CITY_COORD_CACHE_LOCK:
            _CITY_COORD_CACHE[city_key] = {}
        return None

    url = f"{HEIGIT_BASE_URL.rstrip('/')}/geocode/search"
    params = {
        "api_key": HEIGIT_API_KEY,
        "text": city,
        "size": 1,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            with _CITY_COORD_CACHE_LOCK:
                _CITY_COORD_CACHE[city_key] = {}
            return None
        data = resp.json() if resp.text else {}
        features = data.get("features", []) if isinstance(data, dict) else []
        if not features:
            with _CITY_COORD_CACHE_LOCK:
                _CITY_COORD_CACHE[city_key] = {}
            return None
        geometry = features[0].get("geometry", {}) if isinstance(features[0], dict) else {}
        coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
        if not isinstance(coords, list) or len(coords) < 2:
            with _CITY_COORD_CACHE_LOCK:
                _CITY_COORD_CACHE[city_key] = {}
            return None
        lng = _safe_float(coords[0], 0.0)
        lat = _safe_float(coords[1], 0.0)
        if lat == 0.0 and lng == 0.0:
            with _CITY_COORD_CACHE_LOCK:
                _CITY_COORD_CACHE[city_key] = {}
            return None
        result = {"lat": lat, "lng": lng}
        with _CITY_COORD_CACHE_LOCK:
            _CITY_COORD_CACHE[city_key] = result
        return result
    except Exception:
        with _CITY_COORD_CACHE_LOCK:
            _CITY_COORD_CACHE[city_key] = {}
        return None


def _compute_drive_duration_hours_heigit(from_city: str, to_city: str) -> Optional[float]:
    if not HEIGIT_API_KEY:
        return None

    origin = _geocode_city_heigit(from_city)
    destination = _geocode_city_heigit(to_city)
    if not origin or not destination:
        return None

    metrics = _route_metrics_heigit_coords(origin, destination)
    if not metrics:
        return None
    return _safe_float(metrics.get("duration_h"), None)


def _estimate_mid_distance_transport_cost(city_tier: str) -> int:
    tier = _normalize_text(city_tier).lower()
    if tier not in MID_DISTANCE_CAR_COST:
        tier = "medium"
    return int(MID_DISTANCE_CAR_COST[tier])


def _local_transport_tier_from_cost_level(cost_level: Any) -> str:
    level = _normalize_text(cost_level).lower()
    if level in {"ultra_high", "ultra_luxury", "top", "vip"}:
        return "ultra"
    if level in {"high", "luxury"}:
        return "high"
    if level in {"medium", "medium_high"}:
        return "medium"
    return "low"


def _is_international_trip(normalized: Dict[str, Any]) -> bool:
    country = _normalize_text(normalized.get("country")).lower()
    if not country:
        return False
    return country not in CHINA_COUNTRY_ALIASES


def _local_transport_rate_per_minute(tier: str, is_international: bool = False) -> float:
    key = _normalize_text(tier).lower()
    if key == "ultra":
        return float(LOCAL_TRANSPORT_RATE_BY_TIER["ultra"])
    if is_international:
        return INTERNATIONAL_LOCAL_TRANSPORT_RATE_PER_MINUTE
    return float(LOCAL_TRANSPORT_RATE_BY_TIER.get(key, LOCAL_TRANSPORT_RATE_BY_TIER["medium"]))


def _estimate_local_transfer_cost(duration_h: float, tier: Any, is_international: bool = False) -> float:
    minutes = max(0, int(round(_safe_float(duration_h, 0.0) * 60)))
    if minutes <= 0:
        return 0.0
    return round(minutes * _local_transport_rate_per_minute(_normalize_text(tier), is_international=is_international), 2)


def _collect_city_candidates(
    city: str,
    selected_pois: List[Dict[str, Any]],
    selected_restaurants: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    city_lower = city.lower()
    pois = [x for x in selected_pois if _normalize_text(x.get("city")).lower() == city_lower]
    rests = [x for x in selected_restaurants if _normalize_text(x.get("city")).lower() == city_lower]

    pois_sorted = sorted(
        pois,
        key=lambda x: (
            _safe_float(x.get("rating"), 0.0),
            _safe_float(x.get("estimated_total_cost"), 0.0) * -1,
        ),
        reverse=True,
    )
    rests_sorted = sorted(
        rests,
        key=lambda x: (
            _safe_float(x.get("rating"), 0.0),
            _safe_float(x.get("estimated_total_cost"), 0.0) * -1,
        ),
        reverse=True,
    )

    return {
        "pois": pois_sorted,
        "restaurants": rests_sorted,
    }


def _distance_km_between_places(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    route_km = _route_distance_km_between_places(a, b)
    if route_km is not None:
        return route_km
    return _straight_line_distance_km_between_places(a, b)


def _is_within_hotel_radius(hotel: Dict[str, Any], place: Dict[str, Any], radius_km: float) -> bool:
    if not isinstance(hotel, dict) or not isinstance(place, dict):
        return False
    distance_km = _distance_km_between_places(hotel, place)
    if distance_km is None:
        return False
    return distance_km <= max(0.0, _safe_float(radius_km, 0.0))


def _filter_city_candidates_by_hotel_radius(
    city_candidates: Dict[str, List[Dict[str, Any]]],
    hotel: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    pois = [x for x in (city_candidates.get("pois", []) or []) if isinstance(x, dict)]
    restaurants = [x for x in (city_candidates.get("restaurants", []) or []) if isinstance(x, dict)]

    if not isinstance(hotel, dict):
        return {
            "pois": pois,
            "restaurants": restaurants,
            "radius_meta": {
                "poi_limit_km": HOTEL_ACTIVITY_RADIUS_KM,
                "meal_limit_km": HOTEL_ACTIVITY_RADIUS_KM,
                "meal_fallback_used": False,
                "poi_in_radius_count": len(pois),
                "meal_in_10km_count": len(restaurants),
                "meal_in_15km_count": len(restaurants),
                "hotel_missing_for_radius_filter": True,
            },
        }

    pois_in_radius = [p for p in pois if _is_within_hotel_radius(hotel, p, HOTEL_ACTIVITY_RADIUS_KM)]
    restaurants_in_10 = [r for r in restaurants if _is_within_hotel_radius(hotel, r, HOTEL_ACTIVITY_RADIUS_KM)]
    restaurants_in_15 = [r for r in restaurants if _is_within_hotel_radius(hotel, r, HOTEL_MEAL_FALLBACK_RADIUS_KM)]

    use_meal_fallback = not restaurants_in_10 and bool(restaurants_in_15)
    filtered_restaurants = restaurants_in_10 if restaurants_in_10 else restaurants_in_15

    return {
        "pois": pois_in_radius,
        "restaurants": filtered_restaurants,
        "radius_meta": {
            "poi_limit_km": HOTEL_ACTIVITY_RADIUS_KM,
            "meal_limit_km": HOTEL_MEAL_FALLBACK_RADIUS_KM if use_meal_fallback else HOTEL_ACTIVITY_RADIUS_KM,
            "meal_fallback_used": use_meal_fallback,
            "poi_in_radius_count": len(pois_in_radius),
            "meal_in_10km_count": len(restaurants_in_10),
            "meal_in_15km_count": len(restaurants_in_15),
            "hotel_missing_for_radius_filter": False,
        },
    }


def _split_must_visit_and_regular_pois(
    pois: List[Dict[str, Any]],
    must_visit_names: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    targets = [_normalize_text(x).lower() for x in must_visit_names if _normalize_text(x)]
    must: List[Dict[str, Any]] = []
    regular: List[Dict[str, Any]] = []

    for poi in pois:
        name = _normalize_text(poi.get("name")).lower()
        addr = _normalize_text(poi.get("address")).lower()
        text = f"{name} {addr}".strip()
        matched_name = None
        for t in targets:
            if t and t in text:
                matched_name = t
                break
        if matched_name is not None:
            item = dict(poi)
            item["_must_visit_name_norm"] = matched_name
            must.append(item)
        else:
            regular.append(poi)

    return must, regular


def _init_day_plan(slot: Dict[str, Any], hotel: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "date": _normalize_text(slot.get("date")),
        "city": _normalize_text(slot.get("city")),
        "day_index_in_city": int(_safe_float(slot.get("day_index_in_city"), 0)),
        "days_in_city": int(_safe_float(slot.get("days_in_city"), 0)),
        "check_in": _normalize_text(slot.get("check_in")),
        "check_out": _normalize_text(slot.get("check_out")),
        "schedule_start": DEFAULT_DAY_START,
        "schedule_end": DEFAULT_DAY_END,
        "hotel": {
            "name": _normalize_text((hotel or {}).get("name")),
            "area": _normalize_text((hotel or {}).get("area")),
            "estimated_total_cost": _safe_float((hotel or {}).get("estimated_total_cost"), 0.0),
            "lat": _safe_float((hotel or {}).get("lat"), None),
            "lng": _safe_float((hotel or {}).get("lng"), None),
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


def _append_event(
    day_plan: Dict[str, Any],
    event_type: str,
    name: str,
    duration_h: float,
    cost_estimated: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    events = day_plan.setdefault("events", [])
    used = _safe_float(day_plan.get("used_hours"), 0.0)
    meta = meta or {}

    if meta.get("all_day_cost") and max(0.0, duration_h) == 0:
        start_dt = None
        end_dt = None
    else:
        last_timed_event = None
        for existing in reversed(events):
            existing_meta = existing.get("meta") or {}
            if existing_meta.get("all_day_cost"):
                continue
            if _normalize_text(existing.get("end")):
                last_timed_event = existing
                break

        if last_timed_event is not None:
            start_time = _normalize_text(last_timed_event.get("end"))
            start_dt = _parse_hhmm(start_time) or _parse_hhmm("08:00")
        else:
            start_dt = _parse_hhmm(_normalize_text(day_plan.get("schedule_start"))) or _parse_hhmm("08:00")

        dur_minutes = int(round(max(0.0, duration_h) * 60))
        end_dt = start_dt + timedelta(minutes=dur_minutes)

    if start_dt is not None and end_dt is not None:
        start_label = start_dt.strftime("%H:%M")
        end_label = end_dt.strftime("%H:%M")
    else:
        start_label = ""
        end_label = ""

    event = {
        "type": _normalize_text(event_type),
        "name": _normalize_text(name),
        "start": start_label,
        "end": end_label,
        "duration_h": round(max(0.0, duration_h), 2),
        "cost_estimated": round(max(0.0, _safe_float(cost_estimated, 0.0)), 2),
        "meta": meta,
    }

    events.append(event)
    day_plan["used_hours"] = round(used + max(0.0, duration_h), 2)


def _append_fixed_time_event(
    day_plan: Dict[str, Any],
    event_type: str,
    name: str,
    start: str,
    end: str,
    duration_h: float,
    cost_estimated: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    event = {
        "type": _normalize_text(event_type),
        "name": _normalize_text(name),
        "start": _normalize_text(start),
        "end": _normalize_text(end),
        "duration_h": round(max(0.0, duration_h), 2),
        "cost_estimated": round(max(0.0, _safe_float(cost_estimated, 0.0)), 2),
        "meta": meta or {},
    }
    day_plan.setdefault("events", []).append(event)
    day_plan["used_hours"] = round(
        _safe_float(day_plan.get("used_hours"), 0.0) + max(0.0, duration_h),
        2,
    )


def _count_events_of_type(day_plan: Dict[str, Any], event_type: str) -> int:
    total = 0
    target = _normalize_text(event_type)
    for event in day_plan.get("events", []) or []:
        if _normalize_text(event.get("type")) == target:
            total += 1
    return total


def _count_poi_events(day_plan: Dict[str, Any]) -> int:
    return _count_events_of_type(day_plan, "poi")


def _fallback_meal_cost(city_budget: Dict[str, float]) -> float:
    days = max(1.0, _safe_float(city_budget.get("days"), 1.0))
    day_food_budget = _safe_float(city_budget.get("food_budget"), 0.0) / days
    if day_food_budget > 0:
        return round(max(LOW_QUALITY_MEAL_COST_FLOOR, day_food_budget / float(DAILY_REQUIRED_MEALS)), 2)
    return round(LOW_QUALITY_MEAL_COST_FLOOR, 2)


def _meal_display_name(candidate: Dict[str, Any], city: str, index: int) -> str:
    base = _normalize_text(candidate.get("name"))
    if base:
        return base
    city_name = city or "City"
    return f"Local Meal Option {index + 1} ({city_name})"


def _fallback_meal_name(day_plan: Dict[str, Any], city: str, index: int) -> str:
    city_name = city or "City"
    day_marker = _normalize_text(day_plan.get("date")) or f"day-{max(1, int(_safe_float(day_plan.get('day_index_in_city'), 1)))}"
    return f"Local Meal Option {index + 1} ({city_name}, {day_marker})"


def _meal_event_cost(candidate: Dict[str, Any], fallback_meal_cost: float) -> float:
    estimated_total = _safe_float(candidate.get("estimated_total_cost"), 0.0)
    recommended_visits = max(1.0, _safe_float(candidate.get("recommended_visits"), 1.0))
    if estimated_total > 0:
        return round(max(LOW_QUALITY_MEAL_COST_FLOOR, estimated_total / recommended_visits), 2)

    estimated_meal_per_person = _safe_float(candidate.get("estimated_meal_per_person"), 0.0)
    if estimated_meal_per_person > 0:
        return round(max(LOW_QUALITY_MEAL_COST_FLOOR, estimated_meal_per_person), 2)

    return round(max(LOW_QUALITY_MEAL_COST_FLOOR, fallback_meal_cost), 2)


def _event_start_minutes(event: Dict[str, Any]) -> Optional[int]:
    dt = _parse_time_value(event.get("start"))
    if not dt:
        return None
    return dt.hour * 60 + dt.minute


def _event_end_minutes(event: Dict[str, Any]) -> Optional[int]:
    dt = _parse_time_value(event.get("end"))
    if not dt:
        return None
    return dt.hour * 60 + dt.minute


def _event_minutes_from_hhmm(value: Any) -> Optional[int]:
    dt = _parse_time_value(value)
    if not dt:
        return None
    return dt.hour * 60 + dt.minute


def _add_minutes_to_time(value: Any, minutes: int) -> str:
    dt = _parse_time_value(value)
    if not dt:
        return ""
    return (dt + timedelta(minutes=minutes)).strftime("%H:%M")


def _max_time_value(*values: Any) -> str:
    parsed = [_parse_time_value(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return ""
    return max(parsed).strftime("%H:%M")


def _hotel_checkin_window(arrival_time: Any = None) -> Tuple[str, str]:
    if _format_time_value(arrival_time):
        start = _add_minutes_to_time(arrival_time, 30) or "14:00"
    else:
        start = "14:00"
    end = _add_minutes_to_time(start, 30) or "15:30"
    return start, end


def _hotel_checkout_window(departure_time: Any = None) -> Tuple[str, str]:
    departure = _parse_time_value(departure_time)
    noon = _parse_time_value("12:00")
    if departure:
        departure_minutes = departure.hour * 60 + departure.minute
        # For very early departures after midnight, keep checkout shortly before leaving
        # instead of wrapping it to the previous evening on the same report day.
        if departure_minutes <= 3 * 60:
            end_minutes = max(0, departure_minutes - 30)
            end = f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
        elif noon and departure < noon:
            end_minutes = max(6 * 60, departure_minutes - 120)
            end = f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
        else:
            end = "12:00"
    else:
        end = "12:00"
    start = _add_minutes_to_time(end, -30) or "11:30"
    return start, end


def _sort_and_recompute_day(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        day_plan["events"] = []
        day_plan["used_hours"] = 0.0
        return

    def sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        idx, event = item
        event_type = _normalize_text(event.get("type"))
        if event_type == "local_transport" and (event.get("meta") or {}).get("all_day_cost"):
            return -1, idx
        start = _event_start_minutes(event)
        return (start if start is not None else 24 * 60 + idx, idx)

    day_plan["events"] = [event for _, event in sorted(enumerate(events), key=sort_key)]
    day_plan["used_hours"] = round(
        sum(_safe_float(event.get("duration_h"), 0.0) for event in day_plan["events"]),
        2,
    )


def _remove_impossible_city_events(day_plan: Dict[str, Any], leg: Dict[str, Any], flight: Dict[str, Any]) -> None:
    leg_type = _normalize_text(leg.get("leg_type"))
    if leg_type not in {"outbound", "inbound"}:
        return

    dep_dt = _parse_time_value(flight.get("departure_time"))
    arr_dt = _parse_time_value(flight.get("arrival_time"))
    if leg_type == "outbound" and not arr_dt:
        return
    if leg_type == "inbound" and not dep_dt:
        return

    cutoff = (arr_dt.hour * 60 + arr_dt.minute) if leg_type == "outbound" and arr_dt else (dep_dt.hour * 60 + dep_dt.minute)
    kept: List[Dict[str, Any]] = []
    removed_names: List[str] = []
    for event in day_plan.get("events", []) or []:
        event_type = _normalize_text(event.get("type"))
        if event_type in {"flight", "drive", "local_transport"}:
            kept.append(event)
            continue
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if leg_type == "outbound" and event_type == "checkin" and start is not None and start < cutoff:
            dur_minutes = int(round(_safe_float(event.get("duration_h"), 0.5) * 60))
            new_start = _parse_hhmm(f"{cutoff // 60:02d}:{cutoff % 60:02d}") or _parse_hhmm("15:00")
            new_end = new_start + timedelta(minutes=max(0, dur_minutes))
            event["start"] = new_start.strftime("%H:%M")
            event["end"] = new_end.strftime("%H:%M")
            kept.append(event)
            continue
        if leg_type == "inbound" and event_type == "checkout" and end is not None and end > cutoff:
            dur_minutes = int(round(_safe_float(event.get("duration_h"), 0.5) * 60))
            new_end = _parse_hhmm(f"{cutoff // 60:02d}:{cutoff % 60:02d}") or _parse_hhmm("12:00")
            new_start = new_end - timedelta(minutes=max(0, dur_minutes))
            event["start"] = new_start.strftime("%H:%M")
            event["end"] = new_end.strftime("%H:%M")
            kept.append(event)
            continue
        impossible = False
        if leg_type == "outbound":
            impossible = start is not None and start < cutoff
        elif leg_type == "inbound":
            impossible = end is not None and end > cutoff
        if impossible:
            removed_names.append(_normalize_text(event.get("name")))
            continue
        kept.append(event)

    if removed_names:
        day_plan["events"] = kept
        day_plan.setdefault("warnings", []).append(
            "Removed impossible scheduled events around flight time: "
            + ", ".join(name for name in removed_names if name)
        )
        _sort_and_recompute_day(day_plan)


def _flight_arrival_date(dep_date: str, flight: Dict[str, Any]) -> str:
    dep_date = _normalize_text(dep_date)
    dep_minutes = _event_minutes_from_hhmm(flight.get("departure_time"))
    arr_minutes = _event_minutes_from_hhmm(flight.get("arrival_time"))
    if dep_minutes is None or arr_minutes is None or arr_minutes > dep_minutes:
        return dep_date
    dep_dt = _parse_date(dep_date)
    if not dep_dt:
        return dep_date
    return (dep_dt + timedelta(days=1)).strftime("%Y-%m-%d")


def _remove_event_type(day_plan: Dict[str, Any], event_type: str) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return
    target = _normalize_text(event_type)
    day_plan["events"] = [event for event in events if _normalize_text(event.get("type")) != target]
    _sort_and_recompute_day(day_plan)


def _event_place_meta(event: Dict[str, Any]) -> Dict[str, Any]:
    meta = event.get("meta") or {}
    if not isinstance(meta, dict):
        return {}
    return meta


def _last_place_event(day_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for event in reversed(day_plan.get("events", []) or []):
        if _normalize_text(event.get("type")) in PLACE_EVENT_TYPES:
            return event
    return None


def _estimate_commute_hours_between_meta(from_meta: Dict[str, Any], to_meta: Dict[str, Any]) -> Tuple[float, str, Optional[float]]:
    from_coord = _place_coord(from_meta)
    to_coord = _place_coord(to_meta)
    if from_coord and to_coord:
        metrics = _route_metrics_heigit_coords(from_coord, to_coord)
        if metrics:
            return max(0.08, _safe_float(metrics.get("duration_h"), 0.0)), "heigit_routes_api", _safe_float(metrics.get("distance_km"), None)
        distance = _haversine_km(from_coord["lat"], from_coord["lng"], to_coord["lat"], to_coord["lng"])
        return max(0.15, round((distance * 1.35) / 25.0, 2)), "fallback_haversine_road_factor", round(distance * 1.35, 2)

    return 0.25, "fallback_missing_coordinates", None


def _append_commute_before_place(
    day_plan: Dict[str, Any],
    to_name: str,
    to_meta: Dict[str, Any],
) -> float:
    previous = _last_place_event(day_plan)
    if previous is None:
        return 0.0

    previous_meta = _event_place_meta(previous)
    commute_h, source, distance_km = _estimate_commute_hours_between_meta(previous_meta, to_meta)
    if commute_h <= 0:
        return 0.0

    from_name = _normalize_text(previous.get("name")) or "previous stop"
    city = _normalize_text(day_plan.get("city"))
    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    is_international = bool(day_plan.get("local_transport_is_international"))
    rate = _local_transport_rate_per_minute(local_tier, is_international=is_international)
    cost = _estimate_local_transfer_cost(commute_h, local_tier, is_international=is_international)
    _append_event(
        day_plan=day_plan,
        event_type="transfer",
        name=f"Transfer {from_name} -> {to_name}",
        duration_h=commute_h,
        cost_estimated=cost,
        meta={
            "city": city,
            "from": from_name,
            "to": _normalize_text(to_name),
            "duration_source": source,
            "distance_km": distance_km,
            "pricing_rule": f"{'international_default' if is_international else local_tier + '_tier'}_{rate:g}_cny_per_minute",
            "duration_minutes": int(round(commute_h * 60)),
        },
    )
    day_plan["transport_cost_estimated"] = round(
        _safe_float(day_plan.get("transport_cost_estimated"), 0.0) + cost,
        2,
    )
    if source != "heigit_routes_api":
        day_plan.setdefault("warnings", []).append(
            f"Transfer time fallback used for {from_name}->{_normalize_text(to_name)} ({source})."
        )
    return commute_h


def _relocate_hotel_events_for_flights(
    daily_plans: List[Dict[str, Any]],
    selected_all_flight_legs: List[Dict[str, Any]],
) -> None:
    by_date = {_normalize_text(day.get("date")): day for day in daily_plans if isinstance(day, dict)}
    if not by_date:
        return

    for leg in selected_all_flight_legs:
        if not isinstance(leg, dict):
            continue
        flights = leg.get("selected_flights") or []
        if not flights or not isinstance(flights[0], dict):
            continue
        flight = flights[0]
        leg_type = _normalize_text(leg.get("leg_type"))
        dep_date = _normalize_text(leg.get("departure_date"))

        if leg_type == "outbound":
            arrival_date = _flight_arrival_date(dep_date, flight)
            day = by_date.get(arrival_date)
            if not day or not day.get("hotel"):
                continue
            for candidate in daily_plans:
                _remove_event_type(candidate, "checkin")
            start, end = _hotel_checkin_window(flight.get("arrival_time"))
            _append_fixed_time_event(
                day_plan=day,
                event_type="checkin",
                name=f"Check-in { _normalize_text((day.get('hotel') or {}).get('name')) or 'Hotel' }",
                start=start,
                end=end,
                duration_h=0.5,
                cost_estimated=0.0,
                meta={
                    "city": _normalize_text(day.get("city")),
                    "lat": _safe_float((day.get("hotel") or {}).get("lat"), None),
                    "lng": _safe_float((day.get("hotel") or {}).get("lng"), None),
                },
            )
            _sort_and_recompute_day(day)

        elif leg_type == "inbound":
            day = by_date.get(dep_date)
            if not day or not day.get("hotel"):
                continue
            for candidate in daily_plans:
                _remove_event_type(candidate, "checkout")
            start, end = _hotel_checkout_window(flight.get("departure_time"))
            _append_fixed_time_event(
                day_plan=day,
                event_type="checkout",
                name=f"Check-out { _normalize_text((day.get('hotel') or {}).get('name')) or 'Hotel' }",
                start=start,
                end=end,
                duration_h=0.5,
                cost_estimated=0.0,
                meta={
                    "city": _normalize_text(day.get("city")),
                    "lat": _safe_float((day.get("hotel") or {}).get("lat"), None),
                    "lng": _safe_float((day.get("hotel") or {}).get("lng"), None),
                },
            )
            _sort_and_recompute_day(day)


def _target_poi_quota_for_day(day_plan: Dict[str, Any]) -> int:
    days_in_city = max(1, int(_safe_float(day_plan.get("days_in_city"), 1)))
    day_index_in_city = max(1, int(_safe_float(day_plan.get("day_index_in_city"), 1)))
    schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start"))
    schedule_end = _event_minutes_from_hhmm(day_plan.get("schedule_end"))

    if schedule_start is not None and schedule_start >= 15 * 60:
        base = 1
    elif schedule_start is not None and schedule_start >= 12 * 60:
        base = 2
    else:
        base = 3 if days_in_city <= 1 else 2

    if schedule_end is not None and schedule_end <= 14 * 60:
        base = max(1, base - 1)

    if days_in_city >= 2 and day_index_in_city == 1:
        base = max(1, base - 1)

    return max(DAILY_MIN_POIS, min(3, base))


def _schedule_city_day(
    day_plan: Dict[str, Any],
    city_candidates: Dict[str, List[Dict[str, Any]]],
    city_budget: Dict[str, float],
    must_visit_left: List[Dict[str, Any]],
    current_day_index: int,
    restaurant_last_used_day_index: Dict[str, int],
    restaurant_cooldown_days: int = 2,
    used_poi_keys: Optional[set] = None,
    used_restaurant_keys: Optional[set] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    pois = city_candidates.get("pois", []) or []
    restaurants = city_candidates.get("restaurants", []) or []

    must_now = list(must_visit_left)
    must_ids = {_normalize_text(x.get("name")).lower() for x in must_now}
    regular = [p for p in pois if _normalize_text(p.get("name")).lower() not in must_ids]

    poi_quota = _target_poi_quota_for_day(day_plan)

    selected_pois: List[Dict[str, Any]] = []
    for p in must_now:
        if len(selected_pois) >= poi_quota:
            break
        poi_key = _poi_dedupe_key(p, city=_normalize_text(day_plan.get("city")))
        if used_poi_keys is not None and poi_key and poi_key in used_poi_keys:
            continue
        selected_pois.append(p)

    if len(selected_pois) < poi_quota:
        for p in regular:
            if len(selected_pois) >= poi_quota:
                break
            poi_key = _poi_dedupe_key(p, city=_normalize_text(day_plan.get("city")))
            if used_poi_keys is not None and poi_key and poi_key in used_poi_keys:
                continue
            selected_pois.append(p)

    while len(selected_pois) < poi_quota:
        hotel = day_plan.get("hotel") or {}
        fallback_index = len(selected_pois) + 1
        day_marker = _normalize_text(day_plan.get("date")) or f"day-{max(1, int(_safe_float(day_plan.get('day_index_in_city'), 1)))}"
        fallback_names = [
            f"City walk near hotel in {_normalize_text(day_plan.get('city')) or 'city center'} ({day_marker})",
            f"Local park stroll in {_normalize_text(day_plan.get('city')) or 'city center'} ({day_marker})",
            f"Neighborhood sightseeing in {_normalize_text(day_plan.get('city')) or 'city center'} ({day_marker})",
        ]
        selected_pois.append({
            "name": fallback_names[min(fallback_index - 1, len(fallback_names) - 1)],
            "estimated_total_cost": 0.0,
            "lat": _safe_float(hotel.get("lat"), None) if isinstance(hotel, dict) else None,
            "lng": _safe_float(hotel.get("lng"), None) if isinstance(hotel, dict) else None,
        })

    for poi in selected_pois:
        mv_name = poi.get("_must_visit_name_norm")
        meta = {
            "city": _normalize_text(day_plan.get("city")),
            "poi_name": _normalize_text(poi.get("name")),
            "lat": _safe_float(poi.get("lat"), None),
            "lng": _safe_float(poi.get("lng"), None),
        }
        if mv_name:
            meta["must_visit_name"] = mv_name
        commute_h = 0.0
        previous = _last_place_event(day_plan)
        if previous is not None:
            commute_h, _, _ = _estimate_commute_hours_between_meta(_event_place_meta(previous), meta)
        if _safe_float(day_plan.get("used_hours"), 0.0) + commute_h + POI_VISIT_HOURS > DAY_AVAILABLE_HOURS:
            break
        _append_commute_before_place(day_plan, _normalize_text(poi.get("name")) or "POI", meta)
        _append_event(
            day_plan=day_plan,
            event_type="poi",
            name=_normalize_text(poi.get("name")) or "POI",
            duration_h=POI_VISIT_HOURS,
            cost_estimated=_safe_float(poi.get("estimated_total_cost"), 0.0),
            meta=meta,
        )
        poi_key = _poi_dedupe_key(poi, city=_normalize_text(day_plan.get("city")))
        if used_poi_keys is not None and poi_key:
            used_poi_keys.add(poi_key)

    meal_quota = DAILY_REQUIRED_MEALS
    meal_added = 0
    used_today_restaurant_keys = set()
    city_name = _normalize_text(day_plan.get("city"))
    fallback_meal_cost = _fallback_meal_cost(city_budget)
    fallback_restaurants: List[Dict[str, Any]] = [r for r in restaurants if isinstance(r, dict)]
    fallback_cursor = 0
    for r in restaurants:
        if meal_added >= meal_quota:
            break
        if _safe_float(day_plan.get("used_hours"), 0.0) + MEAL_HOURS > DAY_AVAILABLE_HOURS:
            break

        restaurant_key = _restaurant_dedupe_key(r, city=_normalize_text(day_plan.get("city")))
        if not restaurant_key:
            continue

        if restaurant_key in used_today_restaurant_keys:
            continue
        last_used_day = restaurant_last_used_day_index.get(restaurant_key)
        if (
            last_used_day is not None
            and (current_day_index - last_used_day) < max(1, restaurant_cooldown_days)
        ):
            continue

        meta = {
            "city": _normalize_text(day_plan.get("city")),
            "restaurant_key": restaurant_key,
            "restaurant_name": _normalize_text(r.get("name")),
            "lat": _safe_float(r.get("lat"), None),
            "lng": _safe_float(r.get("lng"), None),
        }
        commute_h = 0.0
        previous = _last_place_event(day_plan)
        if previous is not None:
            commute_h, _, _ = _estimate_commute_hours_between_meta(_event_place_meta(previous), meta)
        if _safe_float(day_plan.get("used_hours"), 0.0) + commute_h + MEAL_HOURS > DAY_AVAILABLE_HOURS:
            break
        meal_name = _meal_display_name(r, city_name, meal_added)
        meal_cost = _meal_event_cost(r, fallback_meal_cost)
        _append_commute_before_place(day_plan, meal_name, meta)
        _append_event(
            day_plan=day_plan,
            event_type="meal",
            name=meal_name,
            duration_h=MEAL_HOURS,
            cost_estimated=meal_cost,
            meta=meta,
        )
        restaurant_last_used_day_index[restaurant_key] = current_day_index
        used_today_restaurant_keys.add(restaurant_key)
        meal_added += 1

    while meal_added < DAILY_REQUIRED_MEALS and _safe_float(day_plan.get("used_hours"), 0.0) + MEAL_HOURS <= DAY_AVAILABLE_HOURS:
        if fallback_restaurants:
            fallback = None
            fallback_key = ""
            for _ in range(len(fallback_restaurants)):
                candidate = fallback_restaurants[fallback_cursor % len(fallback_restaurants)]
                fallback_cursor += 1
                candidate_key = (
                    _restaurant_dedupe_key(candidate, city=city_name)
                    or f"{city_name.lower()}|fallback:{fallback_cursor}"
                )
                if candidate_key not in used_today_restaurant_keys:
                    last_used_day = restaurant_last_used_day_index.get(candidate_key)
                    if (
                        last_used_day is not None
                        and (current_day_index - last_used_day) < max(1, restaurant_cooldown_days)
                    ):
                        continue
                    fallback = candidate
                    fallback_key = candidate_key
                    break
            if fallback is not None:
                meal_name = _meal_display_name(fallback, city_name, meal_added)
                meal_cost = _meal_event_cost(fallback, fallback_meal_cost)
                meta = {
                    "city": city_name,
                    "restaurant_key": fallback_key,
                    "restaurant_name": _normalize_text(fallback.get("name")),
                    "lat": _safe_float(fallback.get("lat"), None),
                    "lng": _safe_float(fallback.get("lng"), None),
                }
            else:
                meal_name = _fallback_meal_name(day_plan, city_name, meal_added)
                meal_cost = fallback_meal_cost
                hotel = day_plan.get("hotel") or {}
                meta = {
                    "city": city_name,
                    "restaurant_key": f"{city_name.lower()}|virtual:{meal_added+1}",
                    "restaurant_name": meal_name,
                    "lat": _safe_float(hotel.get("lat"), None) if isinstance(hotel, dict) else None,
                    "lng": _safe_float(hotel.get("lng"), None) if isinstance(hotel, dict) else None,
                }
        else:
            meal_name = _fallback_meal_name(day_plan, city_name, meal_added)
            meal_cost = fallback_meal_cost
            hotel = day_plan.get("hotel") or {}
            meta = {
                "city": city_name,
                "restaurant_key": f"{city_name.lower()}|virtual:{meal_added+1}",
                "restaurant_name": meal_name,
                "lat": _safe_float(hotel.get("lat"), None) if isinstance(hotel, dict) else None,
                "lng": _safe_float(hotel.get("lng"), None) if isinstance(hotel, dict) else None,
            }
        commute_h = 0.0
        previous = _last_place_event(day_plan)
        if previous is not None:
            commute_h, _, _ = _estimate_commute_hours_between_meta(_event_place_meta(previous), meta)
        if _safe_float(day_plan.get("used_hours"), 0.0) + commute_h + MEAL_HOURS > DAY_AVAILABLE_HOURS:
            break
        _append_commute_before_place(day_plan, meal_name, meta)
        _append_event(
            day_plan=day_plan,
            event_type="meal",
            name=meal_name,
            duration_h=MEAL_HOURS,
            cost_estimated=meal_cost,
            meta=meta,
        )
        restaurant_key = _normalize_text(meta.get("restaurant_key"))
        if restaurant_key:
            used_today_restaurant_keys.add(restaurant_key)
            restaurant_last_used_day_index[restaurant_key] = current_day_index
        meal_added += 1

    food_budget = _safe_float(city_budget.get("food_budget"), 0.0)
    attr_budget = _safe_float(city_budget.get("attractions_budget"), 0.0)
    days = max(1.0, _safe_float(city_budget.get("days"), 1.0))

    day_food_budget = food_budget / days
    day_attr_budget = attr_budget / days

    food_spend = 0.0
    attr_spend = 0.0
    for ev in day_plan.get("events", []):
        if ev.get("type") == "meal":
            food_spend += _safe_float(ev.get("cost_estimated"), 0.0)
        if ev.get("type") == "poi":
            attr_spend += _safe_float(ev.get("cost_estimated"), 0.0)

    if day_food_budget > 0 and food_spend > day_food_budget * 1.2:
        day_plan.setdefault("warnings", []).append(
            f"Food daily spend warning: {round(food_spend,2)} exceeds daily food budget reference {round(day_food_budget,2)}."
        )
    if day_attr_budget > 0 and attr_spend > day_attr_budget * 1.2:
        day_plan.setdefault("warnings", []).append(
            f"Attractions daily spend warning: {round(attr_spend,2)} exceeds daily attractions budget reference {round(day_attr_budget,2)}."
        )

    return day_plan


def _trim_day_to_available_hours(day_plan: Dict[str, Any]) -> Dict[str, Any]:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        day_plan["events"] = []
        day_plan["used_hours"] = 0.0
        return day_plan

    while _safe_float(day_plan.get("used_hours"), 0.0) > DAY_AVAILABLE_HOURS and events:
        idx_to_remove = None
        poi_count = _count_poi_events(day_plan)
        for i in range(len(events) - 1, -1, -1):
            if _normalize_text(events[i].get("type")) == "poi" and poi_count > DAILY_MIN_POIS:
                idx_to_remove = i
                break
        if idx_to_remove is None:
            for i in range(len(events) - 1, -1, -1):
                meta = events[i].get("meta") or {}
                reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
                if (
                    _normalize_text(events[i].get("type")) == "transfer"
                    and reason not in {"hotel_checkin_navigation", "hotel_day_start", "hotel_day_end"}
                ):
                    idx_to_remove = i
                    break
        if idx_to_remove is None:
            meal_count = _count_events_of_type(day_plan, "meal")
            for i in range(len(events) - 1, -1, -1):
                if _normalize_text(events[i].get("type")) == "meal" and meal_count > DAILY_REQUIRED_MEALS:
                    idx_to_remove = i
                    break
        if idx_to_remove is None:
            day_plan.setdefault("warnings", []).append(
                f"Day time overrun kept because minimum constraints are locked (>= {DAILY_MIN_POIS} POIs and >= {DAILY_REQUIRED_MEALS} meals)."
            )
            break
        removed = events.pop(idx_to_remove)
        dur = _safe_float(removed.get("duration_h"), 0.0)
        day_plan["used_hours"] = round(max(0.0, _safe_float(day_plan.get("used_hours"), 0.0) - dur), 2)
        for neighbor_idx in sorted([idx_to_remove, idx_to_remove - 1], reverse=True):
            if 0 <= neighbor_idx < len(events) and _normalize_text(events[neighbor_idx].get("type")) == "transfer":
                meta = events[neighbor_idx].get("meta") or {}
                reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
                if reason in {"hotel_checkin_navigation", "hotel_day_start", "hotel_day_end"}:
                    continue
                transfer = events.pop(neighbor_idx)
                day_plan["used_hours"] = round(
                    max(0.0, _safe_float(day_plan.get("used_hours"), 0.0) - _safe_float(transfer.get("duration_h"), 0.0)),
                    2,
                )

    if _safe_float(day_plan.get("used_hours"), 0.0) > DAY_AVAILABLE_HOURS:
        day_plan.setdefault("warnings", []).append(
            f"Day time overrun: unable to trim below {DAY_AVAILABLE_HOURS}h."
        )

    return day_plan


def _insert_flight_and_transfer_events(
    daily_plans: List[Dict[str, Any]],
    selected_all_flight_legs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(selected_all_flight_legs, list) or not daily_plans:
        return daily_plans

    by_date: Dict[str, Dict[str, Any]] = {}
    for day in daily_plans:
        by_date[_normalize_text(day.get("date"))] = day

    sorted_day_dates = sorted(
        [d for d in by_date.keys() if _parse_date(d) is not None],
        key=lambda x: _parse_date(x) or datetime.max
    )

    def _fallback_day_for_leg(leg: Dict[str, Any], dep_date: str) -> Optional[Dict[str, Any]]:
        leg_type = _normalize_text(leg.get("leg_type"))
        if not daily_plans:
            return None

        if leg_type == "inbound":
            return daily_plans[-1]
        if leg_type == "outbound":
            return daily_plans[0]

        target_dt = _parse_date(dep_date)
        if target_dt and sorted_day_dates:
            best_key = sorted_day_dates[-1]
            best_gap = abs((_parse_date(best_key) - target_dt).days)
            for date_key in sorted_day_dates:
                dt = _parse_date(date_key)
                if not dt:
                    continue
                gap = abs((dt - target_dt).days)
                if gap < best_gap:
                    best_gap = gap
                    best_key = date_key
            return by_date.get(best_key)

        from_city = _normalize_text(leg.get("from_city")).lower()
        if from_city:
            city_days = [x for x in daily_plans if _normalize_text(x.get("city")).lower() == from_city]
            if city_days:
                return city_days[-1]

        return daily_plans[-1]

    for leg in selected_all_flight_legs:
        if not isinstance(leg, dict):
            continue
        dep_date = _normalize_text(leg.get("departure_date"))
        day = by_date.get(dep_date)
        if day is None:
            day = _fallback_day_for_leg(leg, dep_date)
            if day is None:
                continue
            day.setdefault("warnings", []).append(
                f"Flight leg date fallback applied for { _normalize_text(leg.get('leg_type')) } leg: "
                f"departure_date={dep_date or '(missing)'} was not found in daily slots."
            )
        flights = leg.get("selected_flights", []) or []
        if flights:
            f = flights[0]
            dur = _estimate_flight_time_cost_hours(f)
            start = _format_time_value(f.get("departure_time"))
            end = _format_time_value(f.get("arrival_time"))
            meta = {
                "leg_type": _normalize_text(leg.get("leg_type")),
                "from_city": _normalize_text(leg.get("from_city")),
                "to_city": _normalize_text(leg.get("to_city")),
                "departure_time": _normalize_text(f.get("departure_time")),
                "arrival_time": _normalize_text(f.get("arrival_time")),
                "travel_class": _normalize_text(f.get("travel_class")).lower(),
                "time_source": _normalize_text(f.get("provider")) or "selected_flight",
                "redeye_or_night_flight": _is_redeye_flight(f),
                "time_counting_rule": "night_flight_not_counted_in_day_hours" if _is_redeye_flight(f) else "flight_duration_plus_2h_buffer",
                "is_estimated_fallback": bool(f.get("is_estimated_fallback")),
            }
            if start and end:
                _append_fixed_time_event(
                    day_plan=day,
                    event_type="flight",
                    name=f"Flight { _normalize_text(leg.get('from_city')) } -> { _normalize_text(leg.get('to_city')) }",
                    start=start,
                    end=end,
                    duration_h=dur,
                    cost_estimated=_safe_float(f.get("price_total"), 0.0),
                    meta=meta,
                )
            else:
                _append_event(
                    day_plan=day,
                    event_type="flight",
                    name=f"Flight { _normalize_text(leg.get('from_city')) } -> { _normalize_text(leg.get('to_city')) }",
                    duration_h=dur,
                    cost_estimated=_safe_float(f.get("price_total"), 0.0),
                    meta=meta,
                )
            _remove_impossible_city_events(day, leg, f)
            day["transport_cost_estimated"] = round(
                _safe_float(day.get("transport_cost_estimated"), 0.0) + _safe_float(f.get("price_total"), 0.0),
                2,
            )
        elif _needs_transport_placeholder(leg):
            leg_type = _normalize_text(leg.get("leg_type"))
            from_city = _normalize_text(leg.get("from_city"))
            to_city = _normalize_text(leg.get("to_city"))
            event_type = "flight_unavailable" if leg_type in {"outbound", "inbound"} else "transfer_unavailable"
            _append_event(
                day_plan=day,
                event_type=event_type,
                name=f"Flight {from_city} -> {to_city} unavailable",
                duration_h=0.0,
                cost_estimated=0.0,
                meta={
                    "leg_type": leg_type,
                    "from_city": from_city,
                    "to_city": to_city,
                    "selection_status": _normalize_text(leg.get("selection_status")),
                    "requires_manual_selection": True,
                },
            )
            day.setdefault("warnings", []).append(
                f"Required {leg_type or 'transport'} leg has no selected ticket: {from_city}->{to_city}. "
                "Report keeps this row visible for manual selection."
            )
        elif (
            _normalize_text(leg.get("leg_type")) == "intercity"
            or _normalize_text(leg.get("transport_mode")) in {"direct_drive", "rental_car", "drive"}
        ):
            from_city = _normalize_text(leg.get("from_city"))
            to_city = _normalize_text(leg.get("to_city"))

            drive_h_api = _compute_drive_duration_hours_heigit(from_city, to_city)
            drive_h = drive_h_api
            if _safe_float(leg.get("direct_drive_hours"), 0.0) > 0:
                drive_h = _safe_float(leg.get("direct_drive_hours"), 0.0)
                drive_h_api = None
            if drive_h is None:
                drive_h = 3.0
                day.setdefault("warnings", []).append(
                    f"Drive duration fallback used for {from_city}->{to_city} (HEIGIT route duration unavailable)."
                )
            if drive_h > MAX_DRIVE_HOURS_PER_LEG:
                day.setdefault("warnings", []).append(
                    f"Drive duration constraint violated for {from_city}->{to_city}: "
                    f"{round(drive_h, 2)}h > {MAX_DRIVE_HOURS_PER_LEG}h."
                )
            cost = _safe_float(leg.get("rental_car_cost_estimated"), 0.0)
            if cost <= 0:
                rental_car = leg.get("rental_car") or {}
                if isinstance(rental_car, dict):
                    allocated = rental_car.get("allocated_city_estimates") or []
                    if isinstance(allocated, list):
                        cost = sum(
                            _safe_float(item.get("total_car_rental_cost"), 0.0)
                            for item in allocated
                            if isinstance(item, dict)
                        )
            if cost <= 0:
                cost = _estimate_mid_distance_transport_cost("medium")
            meta = {
                "leg_type": _normalize_text(leg.get("leg_type")) or "ground",
                "transport_mode": _normalize_text(leg.get("transport_mode")) or "drive",
                "from_city": from_city,
                "to_city": to_city,
                "rental_car": leg.get("rental_car"),
                "duration_source": (
                    _normalize_text(leg.get("direct_drive_source"))
                    or ("heigit_routes_api" if drive_h_api is not None else "fallback_default")
                ),
            }
            if _normalize_text(leg.get("leg_type")) == "outbound":
                start = "08:00"
                end = _add_minutes_to_time(start, int(round(drive_h * 60))) or ""
                if end:
                    _append_fixed_time_event(
                        day_plan=day,
                        event_type="drive",
                        name=f"Drive { from_city } -> { to_city }",
                        start=start,
                        end=end,
                        duration_h=drive_h,
                        cost_estimated=float(cost),
                        meta=meta,
                    )
                else:
                    _append_event(
                        day_plan=day,
                        event_type="drive",
                        name=f"Drive { from_city } -> { to_city }",
                        duration_h=drive_h,
                        cost_estimated=float(cost),
                        meta=meta,
                    )
            else:
                _append_event(
                    day_plan=day,
                    event_type="drive",
                    name=f"Drive { from_city } -> { to_city }",
                    duration_h=drive_h,
                    cost_estimated=float(cost),
                    meta=meta,
                )
            day["transport_cost_estimated"] = round(
                _safe_float(day.get("transport_cost_estimated"), 0.0) + float(cost),
                2,
            )

    _relocate_hotel_events_for_flights(daily_plans, selected_all_flight_legs)

    for day in daily_plans:
        _ensure_transfer_before_checkin(day)
        _sort_and_recompute_day(day)
        day = _trim_day_to_available_hours(day)

    return daily_plans


def _insert_checkin_checkout_events(
    daily_plans: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not daily_plans:
        return daily_plans

    first = daily_plans[0]
    if first.get("hotel") and _normalize_text(first.get("missing_required_outbound_transport")).lower() != "true":
        start, end = _hotel_checkin_window()
        _append_fixed_time_event(
            day_plan=first,
            event_type="checkin",
            name=f"Check-in { _normalize_text((first.get('hotel') or {}).get('name')) or 'Hotel' }",
            start=start,
            end=end,
            duration_h=0.5,
            cost_estimated=0.0,
            meta={
                "city": _normalize_text(first.get("city")),
                "lat": _safe_float((first.get("hotel") or {}).get("lat"), None),
                "lng": _safe_float((first.get("hotel") or {}).get("lng"), None),
            },
        )

    last = daily_plans[-1]
    if last.get("hotel"):
        start, end = _hotel_checkout_window()
        _append_fixed_time_event(
            day_plan=last,
            event_type="checkout",
            name=f"Check-out { _normalize_text((last.get('hotel') or {}).get('name')) or 'Hotel' }",
            start=start,
            end=end,
            duration_h=0.5,
            cost_estimated=0.0,
            meta={
                "city": _normalize_text(last.get("city")),
                "lat": _safe_float((last.get("hotel") or {}).get("lat"), None),
                "lng": _safe_float((last.get("hotel") or {}).get("lng"), None),
            },
        )

    for day in daily_plans:
        _ensure_transfer_before_checkin(day)
        _sort_and_recompute_day(day)
        _trim_day_to_available_hours(day)

    return daily_plans


def _ensure_transfer_before_checkin(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    kept: List[Dict[str, Any]] = []
    for event in events:
        meta = event.get("meta") or {}
        if (
            _normalize_text(event.get("type")) == "transfer"
            and isinstance(meta, dict)
            and _normalize_text(meta.get("reason")) == "hotel_checkin_navigation"
        ):
            continue
        kept.append(event)
    day_plan["events"] = kept
    _sort_and_recompute_day(day_plan)


def _ensure_hotel_day_boundary_transfers(day_plan: Dict[str, Any]) -> None:
    hotel = day_plan.get("hotel") or {}
    if not isinstance(hotel, dict) or not _normalize_text(hotel.get("name")):
        return

    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    hotel_name = _normalize_text(hotel.get("name")) or "Hotel"
    cleaned: List[Dict[str, Any]] = []
    for event in events:
        event_type = _normalize_text(event.get("type"))
        event_name = _normalize_text(event.get("name"))
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        # Keep at most one check-in / check-out event per day to avoid duplicates.
        if event_type in {"checkin", "checkout"}:
            if any(_normalize_text(kept.get("type")) == event_type for kept in cleaned):
                continue
        if reason in {"hotel_day_start", "hotel_day_end"}:
            continue
        # On checkout day, never keep "return to hotel" transfer rows.
        if event_type == "transfer" and event_name.startswith("Return to"):
            to_name = _normalize_text(meta.get("to")) if isinstance(meta, dict) else ""
            if to_name == hotel_name:
                continue
        cleaned.append(event)
    day_plan["events"] = cleaned
    _sort_and_recompute_day(day_plan)

    place_events = [
        event
        for event in day_plan.get("events", [])
        if _normalize_text(event.get("type")) in {"poi", "meal"}
    ]
    if not place_events:
        return

    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    commute_h = HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES / 60.0
    is_international = bool(day_plan.get("local_transport_is_international"))
    commute_cost = _estimate_local_transfer_cost(commute_h, local_tier, is_international=is_international)

    first_event = place_events[0]
    first_start = _event_start_minutes(first_event)
    checkins = [event for event in day_plan.get("events", []) if _normalize_text(event.get("type")) == "checkin"]
    has_checkout = any(
        _normalize_text(event.get("type")) == "checkout"
        for event in day_plan.get("events", [])
    )
    earliest_checkin_start = None
    earliest_checkin_end = None
    for checkin in checkins:
        start_minutes = _event_start_minutes(checkin)
        end_minutes = _event_end_minutes(checkin)
        if start_minutes is None:
            continue
        if earliest_checkin_start is None or start_minutes < earliest_checkin_start:
            earliest_checkin_start = start_minutes
        if end_minutes is not None and (earliest_checkin_end is None or end_minutes < earliest_checkin_end):
            earliest_checkin_end = end_minutes

    should_add_day_start = True
    if earliest_checkin_end is not None and first_start is not None and earliest_checkin_end > first_start:
        # Do not claim hotel departure before the traveler has completed check-in.
        should_add_day_start = False

    if should_add_day_start and first_start is not None:
        start_minutes = max(0, first_start - HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES)
        _append_fixed_time_event(
            day_plan=day_plan,
            event_type="transfer",
            name=f"Transfer from {hotel_name} to first stop",
            start=f"{start_minutes // 60:02d}:{start_minutes % 60:02d}",
            end=f"{first_start // 60:02d}:{first_start % 60:02d}",
            duration_h=commute_h,
            cost_estimated=commute_cost,
            meta={
                "city": _normalize_text(day_plan.get("city")),
                "from": hotel_name,
                "to": _normalize_text(first_event.get("name")) or "first stop",
                "reason": "hotel_day_start",
                "duration_source": "fixed_hotel_day_boundary",
            },
        )
        day_plan["transport_cost_estimated"] = round(
            _safe_float(day_plan.get("transport_cost_estimated"), 0.0) + commute_cost,
            2,
        )

    if not has_checkout:
        last_event = place_events[-1]
        last_end = _event_end_minutes(last_event)
        if last_end is not None:
            end_minutes = min(23 * 60 + 59, last_end + HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES)
            _append_fixed_time_event(
                day_plan=day_plan,
                event_type="transfer",
                name=f"Return to {hotel_name}",
                start=f"{last_end // 60:02d}:{last_end % 60:02d}",
                end=f"{end_minutes // 60:02d}:{end_minutes % 60:02d}",
                duration_h=commute_h,
                cost_estimated=commute_cost,
                meta={
                    "city": _normalize_text(day_plan.get("city")),
                    "from": _normalize_text(last_event.get("name")) or "last stop",
                    "to": hotel_name,
                    "reason": "hotel_day_end",
                    "duration_source": "fixed_hotel_day_boundary",
                },
            )
            day_plan["transport_cost_estimated"] = round(
                _safe_float(day_plan.get("transport_cost_estimated"), 0.0) + commute_cost,
                2,
            )

    _sort_and_recompute_day(day_plan)


def _is_location_event(event: Dict[str, Any]) -> bool:
    event_type = _normalize_text(event.get("type"))
    return event_type in {"flight", "drive", "poi", "meal", "checkin", "checkout"}


def _event_node_name(event: Dict[str, Any]) -> str:
    event_type = _normalize_text(event.get("type"))
    if event_type == "flight":
        meta = event.get("meta") or {}
        from_city = _normalize_text((meta or {}).get("from_city"))
        to_city = _normalize_text((meta or {}).get("to_city"))
        if from_city or to_city:
            return f"{from_city or 'Origin'} Airport -> {to_city or 'Destination'} Airport"
        return _normalize_text(event.get("name")) or "Flight"
    if event_type in {"checkin", "checkout"}:
        name = _normalize_text(event.get("name"))
        name = name.replace("Check-in ", "").replace("Check-out ", "").strip()
        return name or "Hotel"
    return _normalize_text(event.get("name")) or event_type or "stop"


def _estimate_generic_transfer(day_plan: Dict[str, Any], from_event: Dict[str, Any], to_event: Dict[str, Any]) -> Tuple[float, float]:
    from_meta = _event_place_meta(from_event)
    to_meta = _event_place_meta(to_event)
    commute_h, _, _ = _estimate_commute_hours_between_meta(from_meta, to_meta)
    if commute_h <= 0:
        commute_h = DEFAULT_INTER_EVENT_TRANSFER_MINUTES / 60.0
    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    is_international = bool(day_plan.get("local_transport_is_international"))
    cost = _estimate_local_transfer_cost(commute_h, local_tier, is_international=is_international)
    return commute_h, cost


def _ensure_adjacent_event_transfers(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list) or len(events) <= 1:
        return

    cleaned: List[Dict[str, Any]] = []
    for event in events:
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if reason == "adjacent_event_transfer":
            continue
        cleaned.append(event)
    day_plan["events"] = cleaned

    _sort_and_recompute_day(day_plan)
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    inserts: List[Tuple[int, Dict[str, Any]]] = []
    for index in range(len(events) - 1):
        current = events[index]
        nxt = events[index + 1]
        if _normalize_text(current.get("type")) == "transfer" or _normalize_text(nxt.get("type")) == "transfer":
            continue
        if not _is_location_event(current) or not _is_location_event(nxt):
            continue
        current_type = _normalize_text(current.get("type"))
        next_type = _normalize_text(nxt.get("type"))
        current_meta = current.get("meta") or {}
        if (
            current_type == "flight"
            and next_type == "checkout"
            and isinstance(current_meta, dict)
            and _normalize_text(current_meta.get("leg_type")) == "inbound"
        ):
            continue
        if (
            current_type == "checkout"
            and next_type == "flight"
            and isinstance((nxt.get("meta") or {}), dict)
            and _normalize_text((nxt.get("meta") or {}).get("leg_type")) == "inbound"
        ):
            # Hotel -> airport transfer is valid before an inbound flight home.
            pass

        current_end = _event_end_minutes(current)
        next_start = _event_start_minutes(nxt)
        if current_end is None or next_start is None or next_start <= current_end:
            continue

        gap_h = round((next_start - current_end) / 60.0, 2)
        if gap_h <= 0:
            continue

        estimated_transfer_h, estimated_cost = _estimate_generic_transfer(day_plan, current, nxt)
        # Use estimated commute duration for billing, do not bill the whole schedule gap.
        transfer_h = max(0.08, min(gap_h, estimated_transfer_h))
        transfer_minutes = max(1, int(round(transfer_h * 60)))
        transfer_end = min(next_start, current_end + transfer_minutes)
        transfer_h = round(max(1, transfer_end - current_end) / 60.0, 2)
        local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
        is_international = bool(day_plan.get("local_transport_is_international"))
        transfer_cost = _estimate_local_transfer_cost(transfer_h, local_tier, is_international=is_international)
        if transfer_cost <= 0:
            transfer_cost = estimated_cost
        transfer_event = {
            "type": "transfer",
            "name": f"Transfer {_event_node_name(current)} -> {_event_node_name(nxt)}",
            "start": f"{current_end // 60:02d}:{current_end % 60:02d}",
            "end": f"{transfer_end // 60:02d}:{transfer_end % 60:02d}",
            "duration_h": transfer_h,
            "cost_estimated": round(max(0.0, transfer_cost), 2),
            "meta": {
                "city": _normalize_text(day_plan.get("city")),
                "from": _event_node_name(current),
                "to": _event_node_name(nxt),
                "reason": "adjacent_event_transfer",
                "duration_source": "estimated_commute_capped_by_timeline_gap",
                "scheduled_gap_minutes": max(0, next_start - current_end),
            },
        }
        inserts.append((index + 1, transfer_event))

    if not inserts:
        return

    offset = 0
    for idx, transfer_event in inserts:
        events.insert(idx + offset, transfer_event)
        day_plan["transport_cost_estimated"] = round(
            _safe_float(day_plan.get("transport_cost_estimated"), 0.0) + _safe_float(transfer_event.get("cost_estimated"), 0.0),
            2,
        )
        offset += 1
    _sort_and_recompute_day(day_plan)


def _event_interval(event: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    return _event_start_minutes(event), _event_end_minutes(event)


def _interval_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _find_linked_transfer_index(events: List[Dict[str, Any]], meal_index: int) -> Optional[int]:
    prev_index = meal_index - 1
    if prev_index < 0:
        return None
    prev_event = events[prev_index]
    if _normalize_text(prev_event.get("type")) != "transfer":
        return None
    meta = prev_event.get("meta") or {}
    if not isinstance(meta, dict):
        return None
    meal_name = _normalize_text(events[meal_index].get("name"))
    if _normalize_text(meta.get("to")) != meal_name:
        return None

    previous_location_name = ""
    for idx in range(prev_index - 1, -1, -1):
        candidate = events[idx]
        if _normalize_text(candidate.get("type")) == "transfer":
            continue
        if _is_location_event(candidate):
            previous_location_name = _event_node_name(candidate)
            break

    transfer_from = _normalize_text(meta.get("from"))
    if previous_location_name and transfer_from and transfer_from != previous_location_name:
        return None

    transfer_start = _event_start_minutes(prev_event)
    transfer_end = _event_end_minutes(prev_event)
    meal_start = _event_start_minutes(events[meal_index])
    if (
        transfer_start is None
        or transfer_end is None
        or meal_start is None
        or transfer_end > meal_start
        or transfer_end < transfer_start
    ):
        return None

    return prev_index
    return None


def _sanitize_transfer_timeline(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", []) or []
    if not isinstance(events, list):
        return

    cleaned: List[Dict[str, Any]] = []
    removed_names: List[str] = []

    for index, event in enumerate(events):
        if _normalize_text(event.get("type")) != "transfer":
            cleaned.append(event)
            continue

        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)

        if start is None or end is None or end < start:
            removed_names.append(_normalize_text(event.get("name")) or "transfer")
            continue

        if reason in {"hotel_checkin_navigation", "hotel_day_start", "hotel_day_end", "adjacent_event_transfer"}:
            cleaned.append(event)
            continue

        prev_location = None
        for prev_idx in range(index - 1, -1, -1):
            candidate = events[prev_idx]
            if _normalize_text(candidate.get("type")) == "transfer":
                continue
            if _is_location_event(candidate):
                prev_location = candidate
                break

        next_location = None
        for next_idx in range(index + 1, len(events)):
            candidate = events[next_idx]
            if _normalize_text(candidate.get("type")) == "transfer":
                continue
            if _is_location_event(candidate):
                next_location = candidate
                break

        if not isinstance(meta, dict) or prev_location is None or next_location is None:
            removed_names.append(_normalize_text(event.get("name")) or "transfer")
            continue

        expected_from = _event_node_name(prev_location)
        expected_to = _event_node_name(next_location)
        if (
            _normalize_text(meta.get("from")) != expected_from
            or _normalize_text(meta.get("to")) != expected_to
        ):
            removed_names.append(_normalize_text(event.get("name")) or "transfer")
            continue

        prev_end = _event_end_minutes(prev_location)
        next_start = _event_start_minutes(next_location)
        if prev_end is None or next_start is None or start < prev_end or end > next_start:
            removed_names.append(_normalize_text(event.get("name")) or "transfer")
            continue

        cleaned.append(event)

    if len(cleaned) != len(events):
        day_plan["events"] = cleaned
        if removed_names:
            day_plan.setdefault("warnings", []).append(
                "Removed stale transfer rows after timeline rebalance: " + ", ".join(removed_names)
            )
        _sort_and_recompute_day(day_plan)


def _next_non_meal_event_end(events: List[Dict[str, Any]], end_index: int) -> int:
    for idx in range(end_index - 1, -1, -1):
        event = events[idx]
        if _normalize_text(event.get("type")) == "meal":
            continue
        end = _event_end_minutes(event)
        if end is not None:
            return end
    return _event_minutes_from_hhmm("08:00") or 8 * 60


def _resolve_meal_start_within_window(
    earliest_start: int,
    duration_minutes: int,
    window_start: int,
    window_end: int,
    blocked_ranges: List[Tuple[int, int]],
) -> Optional[int]:
    candidate_start = max(earliest_start, window_start)
    candidate_end = candidate_start + duration_minutes

    if candidate_end > window_end:
        return None

    changed = True
    while changed:
        changed = False
        for block_start, block_end in blocked_ranges:
            if _interval_overlaps(candidate_start, candidate_end, block_start, block_end):
                candidate_start = max(candidate_start, block_end)
                candidate_end = candidate_start + duration_minutes
                changed = True
                break
        if candidate_end > window_end:
            return None

    return candidate_start


def _enforce_meal_time_windows(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    meal_indices = [i for i, event in enumerate(events) if _normalize_text(event.get("type")) == "meal"]
    if not meal_indices:
        return

    target_slots = [
        (NORMAL_BREAKFAST_START, NORMAL_BREAKFAST_END, "breakfast"),
        (NORMAL_LUNCH_START, NORMAL_LUNCH_END, "lunch"),
        (NORMAL_DINNER_START, NORMAL_DINNER_END, "dinner"),
    ]
    fixed_blocks: List[Tuple[int, int]] = []
    flight_blocks: List[Tuple[int, int]] = []
    immovable_types = {"flight", "drive", "checkin", "checkout"}
    for event in events:
        event_type = _normalize_text(event.get("type"))
        if event_type == "meal":
            continue
        if event_type not in immovable_types:
            continue
        start, end = _event_interval(event)
        if start is None or end is None:
            continue
        fixed_blocks.append((start, end))
        if event_type == "flight":
            flight_blocks.append((start, end))

    assigned_meals: List[Tuple[int, int]] = []
    remove_indices: List[int] = []
    last_end = 19 * 60
    for order, index in enumerate(meal_indices):
        event = events[index]
        linked_transfer_index = _find_linked_transfer_index(events, index)
        linked_transfer = events[linked_transfer_index] if linked_transfer_index is not None else None
        linked_transfer_duration = (
            int(round(max(0.0, _safe_float(linked_transfer.get("duration_h"), 0.0)) * 60))
            if linked_transfer is not None
            else 0
        )
        dur_minutes = int(round(max(0.5, _safe_float(event.get("duration_h"), 1.0)) * 60))
        if dur_minutes <= 0:
            dur_minutes = 60
        if order < len(target_slots):
            slot_start = _event_minutes_from_hhmm(target_slots[order][0]) or 0
            slot_end = _event_minutes_from_hhmm(target_slots[order][1]) or (slot_start + 60)
            slot = target_slots[order][2]
            slot_conflicts_with_flight = any(
                _interval_overlaps(slot_start, slot_end, flight_start, flight_end)
                for flight_start, flight_end in flight_blocks
            )
            if slot_conflicts_with_flight:
                meta = event.get("meta") or {}
                if isinstance(meta, dict):
                    meta["skipped_reason"] = "flight_meal_assumed_onboard"
                    meta["meal_slot"] = slot
                else:
                    event["meta"] = {
                        "meal_slot": slot,
                        "skipped_reason": "flight_meal_assumed_onboard",
                    }
                remove_indices.append(index)
                continue
            previous_event_end = _next_non_meal_event_end(events, linked_transfer_index if linked_transfer_index is not None else index)
            earliest_start = max(previous_event_end + linked_transfer_duration, slot_start)
            blocked_ranges = fixed_blocks + assigned_meals
            candidate_start = _resolve_meal_start_within_window(
                earliest_start=earliest_start,
                duration_minutes=dur_minutes,
                window_start=slot_start,
                window_end=slot_end,
                blocked_ranges=blocked_ranges,
            )
            if candidate_start is None:
                candidate_start = _resolve_meal_start_within_window(
                    earliest_start=slot_start,
                    duration_minutes=dur_minutes,
                    window_start=slot_start,
                    window_end=slot_end,
                    blocked_ranges=blocked_ranges,
                )
            if candidate_start is None:
                continue
            candidate_end = candidate_start + dur_minutes
            event["start"] = f"{candidate_start // 60:02d}:{candidate_start % 60:02d}"
            event["end"] = f"{candidate_end // 60:02d}:{candidate_end % 60:02d}"
            block_start = candidate_start
            if linked_transfer is not None and linked_transfer_duration > 0:
                transfer_end = candidate_start
                transfer_start = max(previous_event_end, transfer_end - linked_transfer_duration)
                transfer_end = transfer_start + linked_transfer_duration
                if transfer_end > candidate_start:
                    transfer_end = candidate_start
                    transfer_start = max(previous_event_end, transfer_end - linked_transfer_duration)
                linked_transfer["start"] = f"{transfer_start // 60:02d}:{transfer_start % 60:02d}"
                linked_transfer["end"] = f"{transfer_end // 60:02d}:{transfer_end % 60:02d}"
                block_start = transfer_start
            assigned_meals.append((block_start, candidate_end))
            last_end = candidate_end
            for other_idx, other in enumerate(events):
                if other_idx == index or other_idx == linked_transfer_index:
                    continue
                other_type = _normalize_text(other.get("type"))
                if other_type not in {"poi", "transfer"}:
                    continue
                o_start = _event_start_minutes(other)
                o_end = _event_end_minutes(other)
                if o_start is None or o_end is None:
                    continue
                if not _interval_overlaps(o_start, o_end, block_start, candidate_end):
                    continue
                shift_minutes = candidate_end - o_start
                if shift_minutes <= 0:
                    continue
                new_start = o_start + shift_minutes
                new_end = o_end + shift_minutes
                other["start"] = f"{new_start // 60:02d}:{new_start % 60:02d}"
                other["end"] = f"{new_end // 60:02d}:{new_end % 60:02d}"
        else:
            slot = "extra_meal"
            candidate_start = max(last_end + 60, (_event_minutes_from_hhmm(_normalize_text(event.get("start"))) or last_end + 60))
            candidate_end = candidate_start + dur_minutes
            event["start"] = f"{candidate_start // 60:02d}:{candidate_start % 60:02d}"
            event["end"] = f"{candidate_end // 60:02d}:{candidate_end % 60:02d}"
            assigned_meals.append((candidate_start, candidate_end))
            last_end = candidate_end

        event["duration_h"] = round(dur_minutes / 60.0, 2)
        meta = event.get("meta") or {}
        if isinstance(meta, dict):
            meta["meal_slot"] = slot
        else:
            event["meta"] = {"meal_slot": slot}

    removed_names: List[str] = []
    for idx in sorted(set(remove_indices), reverse=True):
        removed_names.extend(_remove_event_at(day_plan, idx))
    if removed_names:
        day_plan.setdefault("warnings", []).append(
            "Skipped meal due to flight-time conflict (onboard meal assumed): "
            + ", ".join(name for name in removed_names if name)
        )
    _sort_and_recompute_day(day_plan)


def _ensure_checkin_before_first_activity(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    checkin_event = None
    for event in events:
        if _normalize_text(event.get("type")) == "checkin":
            checkin_event = event
            break
    if not checkin_event:
        return

    first_activity = None
    for event in events:
        if _normalize_text(event.get("type")) in {"poi", "meal"}:
            first_activity = event
            break
    if not first_activity:
        return

    first_start = _event_start_minutes(first_activity)
    checkin_start = _event_start_minutes(checkin_event)
    checkin_end = _event_end_minutes(checkin_event)
    if first_start is None or checkin_start is None or checkin_end is None:
        return
    if checkin_end <= first_start:
        return

    arrival_end = None
    for event in events:
        if _normalize_text(event.get("type")) == "flight":
            meta = event.get("meta") or {}
            if _normalize_text((meta or {}).get("leg_type")) == "outbound":
                arrival_end = _event_end_minutes(event)
                break

    desired_end = first_start
    desired_start = max(0, desired_end - 30)
    if arrival_end is not None:
        # Flight-aware check-in time is already set in relocation logic; avoid forcing impossible overlap edits.
        return

    checkin_event["start"] = f"{desired_start // 60:02d}:{desired_start % 60:02d}"
    checkin_event["end"] = f"{desired_end // 60:02d}:{desired_end % 60:02d}"
    checkin_event["duration_h"] = round(max(0.25, (desired_end - desired_start) / 60.0), 2)
    _sort_and_recompute_day(day_plan)


def _shift_flexible_events_after_hotel_windows(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    hotel_windows: List[Tuple[int, int]] = []
    for event in events:
        if _normalize_text(event.get("type")) not in {"checkin", "checkout"}:
            continue
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            continue
        hotel_windows.append((start, end))

    if not hotel_windows:
        return
    hotel_windows.sort(key=lambda x: x[0])

    for window_start, window_end in hotel_windows:
        for event in events:
            event_type = _normalize_text(event.get("type"))
            if event_type not in {"poi", "meal", "transfer"}:
                continue
            meta = event.get("meta") or {}
            reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
            if reason == "hotel_checkin_navigation":
                continue
            start = _event_start_minutes(event)
            end = _event_end_minutes(event)
            if start is None or end is None:
                continue
            if not _interval_overlaps(start, end, window_start, window_end):
                continue
            shift_minutes = max(0, window_end - start)
            new_start = start + shift_minutes
            new_end = end + shift_minutes
            event["start"] = f"{new_start // 60:02d}:{new_start % 60:02d}"
            event["end"] = f"{new_end // 60:02d}:{new_end % 60:02d}"
    _sort_and_recompute_day(day_plan)


def _resolve_timeline_overlaps(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return
    _sort_and_recompute_day(day_plan)
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    fixed_types = {"flight", "drive", "checkin", "checkout"}
    last_end = None
    for event in events:
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            continue
        event_type = _normalize_text(event.get("type"))
        if last_end is None:
            last_end = end
            continue
        if start < last_end and event_type not in fixed_types:
            delta = last_end - start
            new_start = start + delta
            new_end = end + delta
            event["start"] = f"{new_start // 60:02d}:{new_start % 60:02d}"
            event["end"] = f"{new_end // 60:02d}:{new_end % 60:02d}"
            start, end = new_start, new_end
        if end > (last_end or 0):
            last_end = end
    _sort_and_recompute_day(day_plan)


def _compress_flexible_timeline(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list) or not events:
        return

    _sort_and_recompute_day(day_plan)
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    fixed_types = {"flight", "drive", "checkin", "checkout"}
    cursor = _event_minutes_from_hhmm(day_plan.get("schedule_start")) or 8 * 60
    changed = False

    for event in events:
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            continue

        event_type = _normalize_text(event.get("type"))
        duration_minutes = max(1, int(round(max(0.0, _safe_float(event.get("duration_h"), 0.0)) * 60)))

        if event_type in fixed_types:
            cursor = max(cursor, end)
            continue

        target_start = max(0, cursor)
        target_end = target_start + duration_minutes

        if start != target_start or end != target_end:
            event["start"] = f"{target_start // 60:02d}:{target_start % 60:02d}"
            event["end"] = f"{target_end // 60:02d}:{target_end % 60:02d}"
            changed = True

        cursor = target_end

    if changed:
        day_plan.setdefault("warnings", []).append(
            "Compressed flexible events to remove idle gaps from the daytime timeline."
        )
        _sort_and_recompute_day(day_plan)


def _enforce_unique_poi_visits(daily_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    for day in daily_plans:
        events = day.get("events", [])
        if not isinstance(events, list):
            continue
        to_remove: List[int] = []
        for index, event in enumerate(events):
            if _normalize_text(event.get("type")) != "poi":
                continue
            meta = event.get("meta") or {}
            city = _normalize_text(day.get("city")).lower()
            poi_name = ""
            if isinstance(meta, dict):
                poi_name = _normalize_text(meta.get("poi_name"))
            if not poi_name:
                poi_name = _normalize_text(event.get("name"))
            key = f"{city}|{poi_name.lower()}"
            if key in seen:
                to_remove.append(index)
                continue
            seen.add(key)

        if to_remove:
            removed_names: List[str] = []
            for index in reversed(to_remove):
                target_name = _normalize_text((day.get("events", [])[index] or {}).get("name"))
                _remove_event_at(day, index)
                if target_name:
                    removed_names.append(target_name)
            if removed_names:
                day.setdefault("warnings", []).append(
                    "Removed duplicate POI visits to keep each attraction only once in the whole trip: "
                    + ", ".join(reversed(removed_names))
                )
        _sort_and_recompute_day(day)
    return daily_plans


def _collect_scheduled_poi_keys(daily_plans: List[Dict[str, Any]]) -> set:
    seen: set = set()
    for day in daily_plans:
        city = _normalize_text(day.get("city"))
        for event in day.get("events", []) or []:
            if _normalize_text(event.get("type")) != "poi":
                continue
            meta = event.get("meta") or {}
            poi_like = {
                "place_id": meta.get("place_id") if isinstance(meta, dict) else None,
                "name": _normalize_text(meta.get("poi_name")) if isinstance(meta, dict) else _normalize_text(event.get("name")),
                "address": _normalize_text(meta.get("address")) if isinstance(meta, dict) else "",
            }
            key = _poi_dedupe_key(poi_like, city=city)
            if key:
                seen.add(key)
    return seen


def _append_refill_poi(
    day_plan: Dict[str, Any],
    poi: Dict[str, Any],
) -> bool:
    meta = {
        "city": _normalize_text(day_plan.get("city")),
        "poi_name": _normalize_text(poi.get("name")),
        "lat": _safe_float(poi.get("lat"), None),
        "lng": _safe_float(poi.get("lng"), None),
        "place_id": _normalize_text(poi.get("place_id")),
        "address": _normalize_text(poi.get("address")),
    }
    previous = _last_place_event(day_plan)
    commute_h = 0.0
    if previous is not None:
        commute_h, _, _ = _estimate_commute_hours_between_meta(_event_place_meta(previous), meta)
    if _safe_float(day_plan.get("used_hours"), 0.0) + commute_h + POI_VISIT_HOURS > DAY_AVAILABLE_HOURS:
        return False
    _append_commute_before_place(day_plan, _normalize_text(poi.get("name")) or "POI", meta)
    _append_event(
        day_plan=day_plan,
        event_type="poi",
        name=_normalize_text(poi.get("name")) or "POI",
        duration_h=POI_VISIT_HOURS,
        cost_estimated=_safe_float(poi.get("estimated_total_cost"), 0.0),
        meta=meta,
    )
    return True


def _refill_missing_pois_after_cleanup(
    daily_plans: List[Dict[str, Any]],
    city_candidates_map: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    used_poi_keys = _collect_scheduled_poi_keys(daily_plans)

    for day in daily_plans:
        target_quota = _target_poi_quota_for_day(day)
        current_count = _count_poi_events(day)
        if current_count >= target_quota:
            continue

        city = _normalize_text(day.get("city"))
        candidates = (city_candidates_map.get(city) or {}).get("pois", []) or []
        added_names: List[str] = []

        for poi in candidates:
            if _count_poi_events(day) >= target_quota:
                break
            key = _poi_dedupe_key(poi, city=city)
            if key and key in used_poi_keys:
                continue
            if not _append_refill_poi(day, poi):
                continue
            if key:
                used_poi_keys.add(key)
            added_names.append(_normalize_text(poi.get("name")) or "POI")

        while _count_poi_events(day) < target_quota:
            day_marker = _normalize_text(day.get("date")) or f"day-{max(1, int(_safe_float(day.get('day_index_in_city'), 1)))}"
            fallback_index = _count_poi_events(day) + 1
            fallback_poi = {
                "name": f"Self-guided sightseeing in {city or 'city center'} ({day_marker} #{fallback_index})",
                "estimated_total_cost": 0.0,
                "lat": _safe_float((day.get("hotel") or {}).get("lat"), None),
                "lng": _safe_float((day.get("hotel") or {}).get("lng"), None),
            }
            if not _append_refill_poi(day, fallback_poi):
                break
            added_names.append(_normalize_text(fallback_poi.get("name")) or "POI")

        if added_names:
            day.setdefault("warnings", []).append(
                "Refilled missing attraction slots after cleanup: " + ", ".join(added_names)
            )
        _sort_and_recompute_day(day)

    return daily_plans

    checkins = [event for event in day_plan.get("events", []) if _normalize_text(event.get("type")) == "checkin"]
    if not checkins:
        return

    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    commute_h = CHECKIN_TRANSFER_MINUTES / 60.0
    is_international = bool(day_plan.get("local_transport_is_international"))
    cost = _estimate_local_transfer_cost(commute_h, local_tier, is_international=is_international)
    for checkin in checkins:
        checkin_start = _event_start_minutes(checkin)
        if checkin_start is None:
            continue
        start_minutes = max(0, checkin_start - CHECKIN_TRANSFER_MINUTES)
        start = f"{start_minutes // 60:02d}:{start_minutes % 60:02d}"
        end = f"{checkin_start // 60:02d}:{checkin_start % 60:02d}"
        hotel_name = _normalize_text(checkin.get("name")).replace("Check-in ", "").strip() or "Hotel"
        _append_fixed_time_event(
            day_plan=day_plan,
            event_type="transfer",
            name=f"Transfer to {hotel_name} for check-in",
            start=start,
            end=end,
            duration_h=commute_h,
            cost_estimated=cost,
            meta={
                "city": _normalize_text(day_plan.get("city")),
                "to": hotel_name,
                "reason": "hotel_checkin_navigation",
                "duration_source": "fixed_pre_checkin_navigation",
            },
        )
        day_plan["transport_cost_estimated"] = round(
            _safe_float(day_plan.get("transport_cost_estimated"), 0.0) + cost,
            2,
        )
    _sort_and_recompute_day(day_plan)


def _validate_day_constraints(day_plan: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    within_day_hours = _safe_float(day_plan.get("used_hours"), 0.0) <= DAY_AVAILABLE_HOURS
    if not within_day_hours:
        warnings.append(f"Time constraint violated: used {day_plan.get('used_hours')}h > {DAY_AVAILABLE_HOURS}h.")

    max_drive_ok = True
    for ev in day_plan.get("events", []):
        if _normalize_text(ev.get("type")) == "drive":
            d = _safe_float(ev.get("duration_h"), 0.0)
            if d > MAX_DRIVE_HOURS_PER_LEG:
                max_drive_ok = False
                warnings.append(f"Drive constraint violated: {d}h > 5h.")

    return {
        "within_day_hours": within_day_hours,
        "max_drive_leg_le_5h": max_drive_ok,
        "warnings": warnings,
    }


def _validate_budget_constraints(
    daily_plans: List[Dict[str, Any]],
    total_budget: float,
) -> Dict[str, Any]:
    rollup = _build_cost_rollup(daily_plans)
    total = _safe_float(rollup.get("total_selected_estimated"), 0.0)

    if total_budget <= 0:
        return {
            "budget_feasible": True,
            "total_budget": total_budget,
            "estimated_total": total,
            "delta": 0.0,
        }

    delta = round(total_budget - total, 2)
    return {
        "budget_feasible": delta >= 0,
        "total_budget": total_budget,
        "estimated_total": total,
        "delta": delta,
    }


def _remove_event_at(day: Dict[str, Any], index: int) -> List[str]:
    events = day.get("events", [])
    if not isinstance(events, list) or index < 0 or index >= len(events):
        return []

    removed = []
    target = events[index]
    target_name = _normalize_text(target.get("name"))
    keep: List[Dict[str, Any]] = []
    for idx, event in enumerate(events):
        if idx == index:
            removed.append(target_name or _normalize_text(target.get("type")) or "event")
            continue
        event_type = _normalize_text(event.get("type"))
        meta = event.get("meta") or {}
        if event_type == "transfer" and isinstance(meta, dict):
            if _normalize_text(meta.get("to")) == target_name or _normalize_text(meta.get("from")) == target_name:
                removed.append(_normalize_text(event.get("name")) or "linked transfer")
                continue
        keep.append(event)

    day["events"] = keep
    _sort_and_recompute_day(day)
    return removed


def _enforce_total_budget_hard_cap(
    daily_plans: List[Dict[str, Any]],
    total_budget: float,
) -> List[str]:
    warnings: List[str] = []
    if total_budget <= 0:
        return warnings

    def current_total() -> float:
        return _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)

    removable_type_order = ["poi", "transfer", "meal"]
    while current_total() > total_budget:
        best: Optional[Tuple[int, float, int, int]] = None
        for type_rank, event_type in enumerate(removable_type_order):
            for day_index, day in enumerate(daily_plans):
                for event_index, event in enumerate(day.get("events", []) or []):
                    if _normalize_text(event.get("type")) != event_type:
                        continue
                    meta = event.get("meta") or {}
                    if event_type == "poi" and _count_poi_events(day) <= DAILY_MIN_POIS:
                        continue
                    if (
                        event_type == "transfer"
                        and isinstance(meta, dict)
                        and _normalize_text(meta.get("reason")) in {"hotel_checkin_navigation", "hotel_day_start", "hotel_day_end"}
                    ):
                        continue
                    if event_type == "meal" and _count_events_of_type(day, "meal") <= DAILY_REQUIRED_MEALS:
                        continue
                    cost = _safe_float(event.get("cost_estimated"), 0.0)
                    if cost <= 0:
                        continue
                    candidate = (type_rank, -cost, day_index, event_index)
                    if best is None or candidate < best:
                        best = candidate
            if best is not None and best[0] == type_rank:
                break

        if best is None:
            warnings.append(
                f"Budget hard cap could not be reached: fixed transport/hotel costs are {round(current_total(), 2)} CNY against budget {round(total_budget, 2)} CNY."
            )
            break

        _, _, day_index, event_index = best
        day = daily_plans[day_index]
        before = current_total()
        removed_names = _remove_event_at(day, event_index)
        after = current_total()
        if not removed_names or after >= before:
            warnings.append("Budget hard cap trimming stopped because no further removable paid events reduced cost.")
            break
        warnings.append(
            "Budget hard cap adjustment: removed "
            + ", ".join(name for name in removed_names if name)
            + f" to keep total cost within {round(total_budget, 2)} CNY."
        )

    return warnings


def _downgrade_lifestyle_costs_for_budget(
    daily_plans: List[Dict[str, Any]],
    total_budget: float,
) -> List[str]:
    warnings: List[str] = []
    if total_budget <= 0:
        return warnings

    current_total = _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)
    if current_total <= total_budget:
        return warnings

    adjustable: List[Tuple[int, float, Dict[str, Any]]] = []
    for day in daily_plans:
        for event in day.get("events", []) or []:
            event_type = _normalize_text(event.get("type"))
            if event_type not in {"meal", "poi"}:
                continue
            cost = _safe_float(event.get("cost_estimated"), 0.0)
            if cost <= 0:
                continue
            type_rank = 0 if event_type == "poi" else 1
            adjustable.append((type_rank, cost, event))

    adjustable.sort(key=lambda x: (x[0], -x[1]))
    if not adjustable:
        return warnings

    need_reduce = current_total - total_budget
    reduced = 0.0
    for _, _, event in adjustable:
        event_type = _normalize_text(event.get("type"))
        old_cost = _safe_float(event.get("cost_estimated"), 0.0)
        floor = LOW_QUALITY_MEAL_COST_FLOOR if event_type == "meal" else LOW_QUALITY_POI_COST_FLOOR
        can_reduce = max(0.0, old_cost - floor)
        if can_reduce <= 0:
            continue
        cut = min(can_reduce, need_reduce - reduced)
        if cut <= 0:
            break
        event["cost_estimated"] = round(old_cost - cut, 2)
        reduced = round(reduced + cut, 2)
        if reduced >= need_reduce:
            break

    if reduced > 0:
        warnings.append(
            f"Budget quality downgrade applied: reduced meal/attraction quality costs by {round(reduced, 2)} CNY before deleting events."
        )

    return warnings


def _build_itinerary_warnings(
    daily_plans: List[Dict[str, Any]],
    must_visit_uncovered: List[str],
) -> List[str]:
    warnings: List[str] = []

    for day in daily_plans:
        for w in day.get("warnings", []):
            if w:
                warnings.append(str(w))
        constraints = day.get("constraints", {}) or {}
        if constraints.get("within_day_hours") is False:
            warnings.append(f"{day.get('date')} {day.get('city')}: exceeds {DAY_AVAILABLE_HOURS}h day limit.")
        if constraints.get("max_drive_leg_le_5h") is False:
            warnings.append(f"{day.get('date')} {day.get('city')}: contains drive segment over 5h.")

    for m in must_visit_uncovered:
        warnings.append(f"Must-visit attraction warning: '{m}' not scheduled.")

    deduped: List[str] = []
    seen = set()
    for w in warnings:
        key = w.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)

    return deduped


def _build_cost_rollup(daily_plans: List[Dict[str, Any]]) -> Dict[str, float]:
    transport = 0.0
    food = 0.0
    attractions = 0.0
    hotel_reference = 0.0

    seen_city_hotel = set()

    for day in daily_plans:
        city = _normalize_text(day.get("city")).lower()
        hotel = day.get("hotel") or {}
        if city and city not in seen_city_hotel and isinstance(hotel, dict):
            hotel_reference += _safe_float(hotel.get("estimated_total_cost"), 0.0)
            seen_city_hotel.add(city)

        for ev in day.get("events", []):
            t = _normalize_text(ev.get("type"))
            cost = _safe_float(ev.get("cost_estimated"), 0.0)
            if t in {"flight", "drive", "local_transport", "transfer"}:
                transport += cost
            elif t == "meal":
                food += cost
            elif t == "poi":
                attractions += cost

    total = transport + food + attractions + hotel_reference

    return {
        "transport": round(transport, 2),
        "food": round(food, 2),
        "attractions": round(attractions, 2),
        "hotel_reference": round(hotel_reference, 2),
        "total_selected_estimated": round(total, 2),
    }


def _build_itinerary_summary(
    daily_plans: List[Dict[str, Any]],
    budget_check: Dict[str, Any],
    must_visit_total: int,
    must_visit_done: int,
) -> Dict[str, Any]:
    total_days = len(daily_plans)
    time_feasible_days = 0

    for day in daily_plans:
        constraints = day.get("constraints", {}) or {}
        if constraints.get("within_day_hours") and constraints.get("max_drive_leg_le_5h"):
            time_feasible_days += 1

    coverage = 1.0
    if must_visit_total > 0:
        coverage = round(must_visit_done / float(max(1, must_visit_total)), 4)

    return {
        "total_days": total_days,
        "time_feasible_days": time_feasible_days,
        "budget_feasible": bool(budget_check.get("budget_feasible", True)),
        "must_visit_coverage": coverage,
    }


def _format_agent4_output(
    daily_plans: List[Dict[str, Any]],
    itinerary_summary: Dict[str, Any],
    cost_rollup: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "success": True,
        "stage": "agent4_itinerary_planning",
        "itinerary_summary": itinerary_summary,
        "daily_plan": daily_plans,
        "cost_rollup": cost_rollup,
        "budget_check": {
            "budget_feasible": itinerary_summary.get("budget_feasible", True),
            "estimated_total": cost_rollup.get("total_selected_estimated"),
        },
        "warnings": warnings,
    }


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _restaurant_dedupe_key(restaurant: Dict[str, Any], city: str = "") -> str:
    if not isinstance(restaurant, dict):
        return ""
    place_id = _normalize_text(restaurant.get("place_id"))
    if place_id:
        return f"{_normalize_text(city).lower()}|pid:{place_id.lower()}"
    name = _normalize_text(restaurant.get("name")).lower()
    area = _normalize_text(restaurant.get("area")).lower()
    if name or area:
        return f"{_normalize_text(city).lower()}|na:{name}|{area}"
    return ""


def _poi_dedupe_key(poi: Dict[str, Any], city: str = "") -> str:
    if not isinstance(poi, dict):
        return ""
    place_id = _normalize_text(poi.get("place_id"))
    if place_id:
        return f"{_normalize_text(city).lower()}|pid:{place_id.lower()}"
    name = _normalize_text(poi.get("name")).lower()
    address = _normalize_text(poi.get("address")).lower()
    if name or address:
        return f"{_normalize_text(city).lower()}|na:{name}|{address}"
    return ""


def _safe_float(value: Any, default: Any = 0.0) -> Any:
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


def _parse_date(value: str) -> Optional[datetime]:
    value = _normalize_text(value)
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except Exception:
        return None


def _parse_time_value(value: Any) -> Optional[datetime]:
    value = _normalize_text(value)
    if not value:
        return None
    for fmt in [
        "%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%I:%M %p",
        "%Y-%m-%d %I:%M %p",
    ]:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            pass
    match = re.search(r"\b(\d{1,2}):(\d{2})\b", value)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M")
    return None


def _format_time_value(value: Any) -> str:
    parsed = _parse_time_value(value)
    return parsed.strftime("%H:%M") if parsed else ""


def _parse_hhmm(value: str) -> Optional[datetime]:
    value = _normalize_text(value)
    if not value:
        return None
    try:
        return datetime.strptime(value, "%H:%M")
    except Exception:
        return None


def _extract_hour(value: str) -> Optional[int]:
    value = _normalize_text(value)
    if not value:
        return None
    for fmt in ["%H:%M", "%Y-%m-%d %H:%M", "%I:%M %p"]:
        try:
            return datetime.strptime(value, fmt).hour
        except Exception:
            pass
    if len(value) >= 2 and value[:2].isdigit():
        return int(value[:2])
    return None


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    import math

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c
