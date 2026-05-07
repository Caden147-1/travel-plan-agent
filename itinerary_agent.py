from __future__ import annotations

import os
import re
import threading
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DAY_AVAILABLE_HOURS = 13.0
POI_VISIT_HOURS = 1.5
MEAL_HOURS = 1.0
BREAKFAST_HOURS = 0.67
FLIGHT_BUFFER_HOURS = 2.0
REDEYE_TIME_FACTOR = 0.5
DEFAULT_DAY_START = "08:00"
DEFAULT_DAY_END = "20:30"
NORMAL_BREAKFAST_START = "08:00"
NORMAL_BREAKFAST_END = "10:00"
NORMAL_LUNCH_START = "12:00"
NORMAL_LUNCH_END = "13:30"
NORMAL_DINNER_START = "18:00"
NORMAL_DINNER_END = "20:30"
LOCAL_TRANSPORT_RATE_BY_TIER = {"low": 1.0, "medium": 2.0, "high": 8.0, "ultra": 30.0}
MID_DISTANCE_CAR_COST = {"low": 300, "medium": 400, "high": 500, "ultra": 800}
MAX_DRIVE_HOURS_PER_LEG = max(1.0, _env_float("TRAVEL_MAX_DRIVE_HOURS_PER_LEG", 6.0))
HOTEL_CLUSTER_RADIUS_KM = 5.0
HOTEL_CLUSTER_MAX_ROUTE_KM = 10.0
HOTEL_ACTIVITY_RADIUS_KM = 10.0
HOTEL_MEAL_FALLBACK_RADIUS_KM = 15.0
MIN_MEAL_VARIETY_RESTAURANTS = 3
MEAL_ANCHOR_RADIUS_KM = 0.5
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
HOTEL_SWITCH_TRANSFER_MINUTES = 30
HOTEL_PROCESS_MINUTES = 15
LATE_DINNER_TRANSFER_MINUTES = 15
AIRPORT_EXIT_BUFFER_MINUTES = 30
AIRPORT_TRANSFER_MINUTES = 60
MIN_AIRPORT_TRANSFER_MINUTES = 30
AIRPORT_ARRIVAL_BUFFER_MINUTES = 90
EVENING_RETURN_TARGET_MINUTES = 21 * 60
LOW_QUALITY_MEAL_COST_FLOOR = 30.0
LOW_QUALITY_POI_COST_FLOOR = 20.0
BUDGET_SPEND_UP_TARGET_RATIO = 0.99
BUDGET_SPEND_UP_MIN_DELTA = 100.0
HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES = 15
DEFAULT_INTER_EVENT_TRANSFER_MINUTES = 15
WALK_TRANSFER_MINUTES = 5
WALK_DISTANCE_KM = 0.35
MIN_IDLE_GAP_FILL_MINUTES = 30
MIN_REAL_GAP_POI_VISIT_MINUTES = 45
GAP_REAL_POI_RADIUS_KM = 2.0
INTERNATIONAL_LOCAL_TRANSPORT_RATE_PER_MINUTE = 7.0
INTERNATIONAL_ULTRA_LOCAL_TRANSPORT_RATE_PER_MINUTE = 10.0
CHINA_COUNTRY_ALIASES = {"china", "cn", "中国", "中华人民共和国", "people's republic of china", "prc"}
PROTECTED_TRANSFER_REASONS = {
    "hotel_checkin_navigation",
    "hotel_checkout_navigation",
    "hotel_day_start",
    "hotel_day_end",
    "hotel_switch",
    "arrival_to_hotel",
    "departure_from_hotel",
    "late_arrival_dinner_transfer",
    "evening_activity_transfer",
}
AREA_MUST_VISIT_COORDS_BY_CITY = {
    "tokyo": {
        "ginza": {"lat": 35.6717, "lng": 139.7650, "label": "Ginza shopping district walk"},
        "shinjuku": {"lat": 35.6938, "lng": 139.7034, "label": "Shinjuku area walk"},
        "shibuya": {"lat": 35.6595, "lng": 139.7005, "label": "Shibuya area walk"},
        "harajuku": {"lat": 35.6702, "lng": 139.7027, "label": "Harajuku area walk"},
    }
}


def _agent4_timing_enabled() -> bool:
    value = str(os.getenv("TRAVEL_AGENT4_TIMING", "true")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _agent4_log(message: str) -> None:
    if _agent4_timing_enabled():
        print(f"[agent4] {message}", flush=True)


def _agent4_elapsed(started_at: float) -> float:
    return time.perf_counter() - started_at


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
            if drive_h > 0 and leg_type in {"outbound", "intercity"}:
                drive_start = "08:00" if leg_type == "outbound" else "08:15"
                arrival = _add_minutes_to_time(drive_start, int(round(drive_h * 60)))
                if arrival:
                    day_constraints["schedule_start"] = arrival
                    if leg_type == "intercity":
                        day_constraints["intercity_drive_departure_time"] = drive_start
                        day_constraints["intercity_drive_arrival_time"] = arrival
                        day_constraints["intercity_from_city"] = _normalize_text(leg.get("from_city"))
                        day_constraints["intercity_to_city"] = _normalize_text(leg.get("to_city"))
                    if _event_minutes_from_hhmm(arrival) and _event_minutes_from_hhmm(arrival) >= 18 * 60:
                        day_constraints["skip_city_schedule"] = "true"
            continue
        if not flights or not isinstance(flights[0], dict):
            continue
        flight = flights[0]
        day_constraints = constraints.setdefault(dep_date, {})
        if leg_type in {"outbound", "intercity"}:
            dep_minutes = _event_minutes_from_hhmm(flight.get("departure_time"))
            arr_minutes = _event_minutes_from_hhmm(flight.get("arrival_time"))
            arrival = _format_time_value(flight.get("arrival_time"))
            if arrival:
                arrival_date = _flight_arrival_date(dep_date, flight)
                overnight_arrival = arrival_date != dep_date
                if overnight_arrival:
                    day_constraints["skip_city_schedule"] = "true"
                    day_constraints["skip_city_local_transport"] = "true"
                arrival_constraints = constraints.setdefault(arrival_date, {})
                airport_ready = _add_minutes_to_time(arrival, AIRPORT_EXIT_BUFFER_MINUTES) or arrival
                arrival_transfer_minutes = int(round(max(0.0, _safe_float(flight.get("ground_from_arrival_airport_hours"), 0.0)) * 60))
                if arrival_transfer_minutes <= 0:
                    arrival_transfer_minutes = AIRPORT_TRANSFER_MINUTES
                recovery_start = _add_minutes_to_time(airport_ready, arrival_transfer_minutes) or airport_ready
                arrival_constraints["schedule_start"] = _max_time_value(
                    recovery_start,
                    DEFAULT_DAY_START,
                ) or recovery_start
                arrival_constraints["airport_arrival_time"] = airport_ready
                arrival_constraints["flight_landing_time"] = arrival
                arrival_constraints["airport_exit_buffer_minutes"] = str(AIRPORT_EXIT_BUFFER_MINUTES)
                arrival_constraints["arrival_airport_code"] = (
                    _normalize_text(flight.get("arrival_airport_code"))
                    or _normalize_text(flight.get("arrival_code"))
                    or _normalize_text(leg.get("to_airport"))
                )
                arrival_constraints["arrival_airport_candidate_source"] = _normalize_text(leg.get("to_airport_candidate_source"))
                arrival_constraints["arrival_airport_city"] = _normalize_text(leg.get("to_city"))
                if overnight_arrival or _is_redeye_flight(flight):
                    arrival_constraints["overnight_arrival"] = "true"
                    if arr_minutes is not None and 10 * 60 <= arr_minutes < 12 * 60:
                        arrival_constraints["schedule_start"] = _max_time_value(
                            arrival_constraints.get("schedule_start"),
                            "15:00",
                        ) or arrival_constraints.get("schedule_start", "")
        elif leg_type == "inbound":
            departure = _format_time_value(flight.get("departure_time"))
            if departure:
                airport_deadline = _subtract_minutes_from_time(departure, AIRPORT_ARRIVAL_BUFFER_MINUTES)
                day_constraints["schedule_end"] = airport_deadline or departure
                day_constraints["airport_arrival_deadline"] = airport_deadline
                day_constraints["flight_departure_time"] = departure
    return constraints


def run_itinerary_agent(router_output: Dict[str, Any]) -> Dict[str, Any]:
    agent4_started = time.perf_counter()
    stage_started = agent4_started

    def _mark(stage: str) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        _agent4_log(f"{stage} finished in {now - stage_started:.2f}s total={now - agent4_started:.2f}s")
        stage_started = now

    normalized = _normalize_agent3_payload(router_output)
    refinement_context = normalized.get("refinement_context") or {}
    activity_radius_km = _refinement_activity_radius_km(refinement_context)
    local_transport_tier = _local_transport_tier_from_cost_level(normalized.get("cost_level"))
    local_transport_rate = _local_transport_rate_per_minute(
        local_transport_tier,
        is_international=_is_international_trip(normalized),
    )
    daily_slots = _extract_daily_city_slots(normalized.get("city_date_ranges", {}))
    city_budget_map = _build_city_budget_map(
        normalized.get("city_budget_plan", []),
        normalized.get("budget_plan", {}),
        daily_slots,
    )
    flight_day_constraints = _build_flight_day_constraints(normalized.get("selected_all_flight_legs", []))
    departure_airport_coord_by_city = _departure_airport_coords_by_city(normalized.get("selected_all_flight_legs", []))

    if not daily_slots:
        _agent4_log(f"abort no daily slots total={_agent4_elapsed(agent4_started):.2f}s")
        return {
            "success": False,
            "stage": "agent4_itinerary_planning",
            "error": "No daily slots available from city_date_ranges.",
            "daily_plan": [],
            "warnings": ["Missing city_date_ranges or unable to expand to daily slots."]
        }

    hotels = list(normalized.get("selected_hotels", []) or [])
    pois = list(normalized.get("selected_pois", []) or [])
    restaurants = list(normalized.get("selected_restaurants", []) or [])

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
    _agent4_log(
        "start "
        f"days={len(daily_slots)} cities={len(unique_cities)} "
        f"hotels={len(hotels)} pois={len(pois)} restaurants={len(restaurants)}"
    )
    _mark("setup")

    fallback_started = time.perf_counter()
    for city in unique_cities:
        if not _city_has_selected_hotel(hotels, city):
            fallback_hotel = _build_fallback_hotel_for_city(
                city=city,
                city_budget=city_budget_map.get(city, {}),
                pois=pois,
                restaurants=restaurants,
            )
            if fallback_hotel:
                hotels.append(fallback_hotel)
    _agent4_log(f"fallback_hotels count={len(hotels)} elapsed={_agent4_elapsed(fallback_started):.2f}s")
    _mark("fallback hotel prep")

    candidate_prep_started = time.perf_counter()
    for city in unique_cities:
        city_started = time.perf_counter()
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
            activity_radius_km=activity_radius_km,
        )

        must_city, _ = _split_must_visit_and_regular_pois(
            pois=city_candidates.get("pois", []),
            must_visit_names=must_visit_names,
        )
        area_fallbacks = _build_area_must_visit_fallbacks(
            city=city,
            must_visit_names=must_visit_names,
            matched_must_visit_pois=must_city,
            hotel=primary_hotel,
        )
        if area_fallbacks:
            city_candidates["pois"] = area_fallbacks + city_candidates.get("pois", [])
            must_city = area_fallbacks + must_city
        city_candidates_map[city] = city_candidates
        must_visit_remaining_by_city[city] = list(must_city)
        _agent4_log(
            f"candidate_prep city={city} clusters={len(city_hotel_clusters)} "
            f"pois={len(city_candidates.get('pois', []))} "
            f"restaurants={len(city_candidates.get('restaurants', []))} "
            f"elapsed={_agent4_elapsed(city_started):.2f}s"
        )
    _agent4_log(f"candidate_prep total elapsed={_agent4_elapsed(candidate_prep_started):.2f}s")
    _mark("candidate prep")

    daily_plans: List[Dict[str, Any]] = []
    restaurant_last_used_day_index: Dict[str, int] = {}
    restaurant_cooldown_days = 2
    used_poi_keys: set = set()
    used_restaurant_keys: set = set()
    previous_hotel_by_city: Dict[str, str] = {}

    initial_schedule_started = time.perf_counter()
    for day_index, slot in enumerate(daily_slots, start=1):
        day_started = time.perf_counter()
        city = _normalize_text(slot.get("city"))
        selected_cluster = _select_stay_hotel_cluster(
            hotel_clusters=city_hotel_clusters_map.get(city, []),
            day_index_in_city=int(_safe_float(slot.get("day_index_in_city"), 1)),
            days_in_city=int(_safe_float(slot.get("days_in_city"), 1)),
            departure_airport_coord=departure_airport_coord_by_city.get(city),
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
        if (
            _normalize_text(day_plan.get("skip_city_schedule")).lower() == "true"
            and _normalize_text(day_plan.get("overnight_arrival")).lower() != "true"
        ):
            day_plan["hotel"] = None
        if isinstance(hotel, dict):
            previous_hotel_by_city[city] = _normalize_text(hotel.get("name"))

        day_budget = city_budget_map.get(city, {})
        if isinstance(selected_cluster, dict):
            cluster_restaurants = selected_cluster.get("restaurants", []) or []
            if len(cluster_restaurants) < MIN_MEAL_VARIETY_RESTAURANTS:
                seen_cluster_restaurants = {
                    _restaurant_dedupe_key(item, city=city) or (
                        _normalize_text(item.get("name")).lower(),
                        _normalize_text(item.get("area")).lower(),
                    )
                    for item in cluster_restaurants
                    if isinstance(item, dict)
                }
                for item in (city_candidates_map.get(city, {}).get("restaurants", []) or []):
                    if not isinstance(item, dict):
                        continue
                    key = _restaurant_dedupe_key(item, city=city) or (
                        _normalize_text(item.get("name")).lower(),
                        _normalize_text(item.get("area")).lower(),
                    )
                    if key in seen_cluster_restaurants:
                        continue
                    cluster_restaurants.append(item)
                    seen_cluster_restaurants.add(key)
                    if len(cluster_restaurants) >= MIN_MEAL_VARIETY_RESTAURANTS:
                        break
            city_candidates = {
                "pois": selected_cluster.get("pois", []) or [],
                "restaurants": cluster_restaurants,
                "radius_meta": {
                    "poi_limit_km": HOTEL_ACTIVITY_RADIUS_KM,
                    "meal_limit_km": HOTEL_ACTIVITY_RADIUS_KM,
                    "meal_fallback_used": False,
                    "poi_in_radius_count": len(selected_cluster.get("pois", []) or []),
                    "meal_in_10km_count": len(cluster_restaurants),
                    "meal_in_15km_count": len(cluster_restaurants),
                    "meal_variety_backfill_used": len(cluster_restaurants) > len(selected_cluster.get("restaurants", []) or []),
                    "hotel_missing_for_radius_filter": False,
                    "hotel_switched_for_variety": _normalize_text(previous_hotel_by_city.get(city)) != _normalize_text((hotel or {}).get("name")),
                },
            }
        else:
            city_candidates = city_candidates_map.get(city, {"pois": [], "restaurants": [], "radius_meta": {}})
        radius_meta = city_candidates.get("radius_meta", {}) if isinstance(city_candidates, dict) else {}
        if isinstance(radius_meta, dict):
            if isinstance(hotel, dict) and bool(hotel.get("fallback_generated")):
                day_plan.setdefault("warnings", []).append(
                    f"No selected hotel was returned for {city}; using a fallback itinerary base hotel so lodging, check-in, transfers, and hotel costs remain scheduled."
                )
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
                    f"No restaurants found within {int(_safe_float(radius_meta.get('poi_limit_km'), HOTEL_ACTIVITY_RADIUS_KM))}km of hotel; meal search relaxed to {int(_safe_float(radius_meta.get('meal_limit_km'), HOTEL_MEAL_FALLBACK_RADIUS_KM))}km."
                )
            if int(_safe_float(radius_meta.get("poi_in_radius_count"), 0)) <= 0:
                day_plan.setdefault("warnings", []).append(
                    f"No POIs found within {int(_safe_float(radius_meta.get('poi_limit_km'), HOTEL_ACTIVITY_RADIUS_KM))}km of hotel; POI options may be limited."
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
        _agent4_log(
            f"initial_schedule day={day_index} city={city} events={len(day_plan.get('events', []))} "
            f"elapsed={_agent4_elapsed(day_started):.2f}s"
        )
    _agent4_log(f"initial_schedule total elapsed={_agent4_elapsed(initial_schedule_started):.2f}s")
    _mark("initial daily scheduling")

    daily_plans = _insert_checkin_checkout_events(daily_plans=daily_plans)
    daily_plans = _insert_flight_and_transfer_events(
        daily_plans=daily_plans,
        selected_all_flight_legs=normalized.get("selected_all_flight_legs", []),
    )
    daily_plans = _ensure_hotel_transition_events(daily_plans)
    daily_plans = _ensure_late_arrival_dinners(
        daily_plans=daily_plans,
        city_candidates_map=city_candidates_map,
        city_budget_map=city_budget_map,
        restaurant_last_used_day_index=restaurant_last_used_day_index,
        restaurant_cooldown_days=restaurant_cooldown_days,
        used_restaurant_keys=used_restaurant_keys,
    )
    daily_plans = _enforce_unique_poi_visits(daily_plans)
    daily_plans = _refill_missing_pois_after_cleanup(
        daily_plans=daily_plans,
        city_candidates_map=city_candidates_map,
    )
    _mark("flight hotel arrival cleanup")

    def _ensure_contextual_adjacent_transfers(day: Dict[str, Any], fill_idle_gaps: bool = True) -> None:
        city_key = _normalize_text(day.get("city"))
        city_candidates = city_candidates_map.get(city_key, {})
        _ensure_adjacent_event_transfers(
            day,
            gap_poi_candidates=city_candidates.get("pois", []),
            blocked_poi_keys=_collect_scheduled_poi_keys(daily_plans, exclude_generated_day=day),
            fill_idle_gaps=fill_idle_gaps,
        )

    first_pass_started = time.perf_counter()
    for day_index, day in enumerate(daily_plans, start=1):
        day_started = time.perf_counter()
        city_key = _normalize_text(day.get("city"))
        _ensure_required_meals(
            day,
            city_candidates_map.get(city_key, {"restaurants": []}),
            city_budget_map.get(city_key, {}),
            current_day_index=day_index,
            restaurant_last_used_day_index=restaurant_last_used_day_index,
            restaurant_cooldown_days=restaurant_cooldown_days,
            used_restaurant_keys=used_restaurant_keys,
        )
        _ensure_checkin_before_first_activity(day)
        _ensure_transfer_before_checkin(day)
        _shift_flexible_events_after_hotel_windows(day)
        _ensure_hotel_day_boundary_transfers(day)
        _enforce_meal_time_windows(day)
        _remove_events_before_schedule_start(day)
        _remove_extra_meals_when_required_slots_present(day)
        _remove_invalid_same_site_transfers(day)
        _place_daytime_pois_around_meals(day)
        _enforce_hotel_switch_morning_order(day)
        _ensure_evening_activity_after_dinner(day, city_candidates_map.get(_normalize_text(day.get("city")), {"pois": []}))
        _shift_flexible_events_after_hotel_windows(day)
        _enforce_departure_checkout_to_airport_order(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _compress_flexible_timeline(day)
        _ensure_hotel_day_boundary_transfers(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _enforce_departure_checkout_to_airport_order(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _enforce_departure_checkout_to_airport_order(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _cap_evening_activity_return_time(day)
        _realign_evening_activity_transfers(day)
        _remove_late_regular_pois(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _remove_extra_meals_when_required_slots_present(day)
        _remove_invalid_same_site_transfers(day)
        _ensure_hotel_day_boundary_transfers(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _sort_and_recompute_day(day)
        _agent4_log(
            f"normalize_pass1 day={day_index} city={city_key} events={len(day.get('events', []))} "
            f"elapsed={_agent4_elapsed(day_started):.2f}s"
        )
    _agent4_log(f"normalize_pass1 total elapsed={_agent4_elapsed(first_pass_started):.2f}s")
    _mark("normalize pass 1")

    budget_started = time.perf_counter()
    total_budget = float(_safe_float(normalized.get("trip_summary", {}).get("budget", {}).get("total"), 0.0))
    budget_trim_warnings = _enforce_total_budget_hard_cap(daily_plans, total_budget)
    budget_downgrade_warnings = _downgrade_lifestyle_costs_for_budget(daily_plans, total_budget)
    daily_plans = _ensure_hotel_transition_events(daily_plans)
    _agent4_log(
        f"budget_adjust total_budget={round(total_budget, 2)} "
        f"trim_warnings={len(budget_trim_warnings)} downgrade_warnings={len(budget_downgrade_warnings)} "
        f"elapsed={_agent4_elapsed(budget_started):.2f}s"
    )
    _mark("budget adjustment")

    early_budget_check = _validate_budget_constraints(daily_plans, total_budget)
    if not early_budget_check.get("budget_feasible", True):
        cost_rollup = _build_cost_rollup(daily_plans)
        warnings = _build_itinerary_warnings(
            daily_plans=daily_plans,
            must_visit_uncovered=[],
        )
        warnings.extend(budget_downgrade_warnings)
        warnings.extend(budget_trim_warnings)
        warnings.append(
            "Budget hard cap could not be reached after removing flexible paid activities and downgrading lifestyle costs."
        )
        _agent4_log(
            f"budget early stop estimated={early_budget_check.get('estimated_total')} "
            f"budget={early_budget_check.get('total_budget')} total={_agent4_elapsed(agent4_started):.2f}s"
        )
        return {
            "success": False,
            "stage": "budget_feasibility",
            "error": (
                "Selected fixed and required itinerary costs still exceed the total budget after automatic trims."
            ),
            "daily_plan": daily_plans,
            "cost_rollup": cost_rollup,
            "budget_check": early_budget_check,
            "warnings": list(dict.fromkeys(warnings)),
        }

    second_pass_started = time.perf_counter()
    for day_index, day in enumerate(daily_plans, start=1):
        day_started = time.perf_counter()
        city_key = _normalize_text(day.get("city"))
        _ensure_required_meals(
            day,
            city_candidates_map.get(city_key, {"restaurants": []}),
            city_budget_map.get(city_key, {}),
            current_day_index=day_index,
            restaurant_last_used_day_index=restaurant_last_used_day_index,
            restaurant_cooldown_days=restaurant_cooldown_days,
            used_restaurant_keys=used_restaurant_keys,
        )
        _ensure_checkin_before_first_activity(day)
        _ensure_transfer_before_checkin(day)
        _shift_flexible_events_after_hotel_windows(day)
        _ensure_hotel_day_boundary_transfers(day)
        _enforce_meal_time_windows(day)
        _remove_events_before_schedule_start(day)
        _remove_extra_meals_when_required_slots_present(day)
        _remove_invalid_same_site_transfers(day)
        _place_daytime_pois_around_meals(day)
        _enforce_hotel_switch_morning_order(day)
        _ensure_evening_activity_after_dinner(day, city_candidates_map.get(_normalize_text(day.get("city")), {"pois": []}))
        _shift_flexible_events_after_hotel_windows(day)
        _enforce_departure_checkout_to_airport_order(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _compress_flexible_timeline(day)
        _ensure_hotel_day_boundary_transfers(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _cap_evening_activity_return_time(day)
        _realign_evening_activity_transfers(day)
        _remove_late_regular_pois(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _remove_extra_meals_when_required_slots_present(day)
        _remove_invalid_same_site_transfers(day)
        _ensure_hotel_day_boundary_transfers(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _sort_and_recompute_day(day)
        _agent4_log(
            f"normalize_pass2 day={day_index} city={city_key} events={len(day.get('events', []))} "
            f"elapsed={_agent4_elapsed(day_started):.2f}s"
        )
    _agent4_log(f"normalize_pass2 total elapsed={_agent4_elapsed(second_pass_started):.2f}s")
    _mark("normalize pass 2")

    _reapply_flight_event_constraints(
        daily_plans=daily_plans,
        selected_all_flight_legs=normalized.get("selected_all_flight_legs", []),
    )
    daily_plans = _enforce_poi_three_day_cooldown(daily_plans)
    daily_plans = _refill_missing_pois_after_cleanup(
        daily_plans=daily_plans,
        city_candidates_map=city_candidates_map,
    )
    daily_plans = _enforce_poi_three_day_cooldown(daily_plans)
    daily_plans = _ensure_late_arrival_dinners(
        daily_plans=daily_plans,
        city_candidates_map=city_candidates_map,
        city_budget_map=city_budget_map,
        restaurant_last_used_day_index=restaurant_last_used_day_index,
        restaurant_cooldown_days=restaurant_cooldown_days,
        used_restaurant_keys=used_restaurant_keys,
    )
    _mark("flight reapply and refill")

    third_pass_started = time.perf_counter()
    for day_index, day in enumerate(daily_plans, start=1):
        day_started = time.perf_counter()
        city_key = _normalize_text(day.get("city"))
        _ensure_required_meals(
            day,
            city_candidates_map.get(city_key, {"restaurants": []}),
            city_budget_map.get(city_key, {}),
            current_day_index=day_index,
            restaurant_last_used_day_index=restaurant_last_used_day_index,
            restaurant_cooldown_days=restaurant_cooldown_days,
            used_restaurant_keys=used_restaurant_keys,
        )
        _enforce_meal_time_windows(day)
        _remove_events_before_schedule_start(day)
        _remove_extra_meals_when_required_slots_present(day)
        _remove_invalid_same_site_transfers(day)
        _place_daytime_pois_around_meals(day)
        _ensure_hotel_day_boundary_transfers(day)
        _enforce_departure_checkout_to_airport_order(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _enforce_departure_checkout_to_airport_order(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _cap_evening_activity_return_time(day)
        _realign_evening_activity_transfers(day)
        _remove_late_regular_pois(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _remove_extra_meals_when_required_slots_present(day)
        _remove_invalid_same_site_transfers(day)
        _ensure_hotel_day_boundary_transfers(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day)
        _resolve_timeline_overlaps(day)
        _sort_and_recompute_day(day)
        _agent4_log(
            f"normalize_pass3 day={day_index} city={city_key} events={len(day.get('events', []))} "
            f"elapsed={_agent4_elapsed(day_started):.2f}s"
        )
    _agent4_log(f"normalize_pass3 total elapsed={_agent4_elapsed(third_pass_started):.2f}s")
    _mark("normalize pass 3")

    # Evening/gap fillers run late in the pipeline, so do one final POI reuse pass
    # without adding new attractions afterward.
    daily_plans = _enforce_unique_poi_visits(daily_plans)
    daily_plans = _enforce_poi_three_day_cooldown(daily_plans)
    final_cleanup_started = time.perf_counter()
    for day_index, day in enumerate(daily_plans, start=1):
        day_started = time.perf_counter()
        _ensure_hotel_day_boundary_transfers(day)
        _enforce_departure_checkout_to_airport_order(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day, fill_idle_gaps=False)
        _resolve_timeline_overlaps(day)
        _cap_evening_activity_return_time(day)
        _realign_evening_activity_transfers(day)
        _remove_late_regular_pois(day)
        _sanitize_transfer_timeline(day)
        _ensure_contextual_adjacent_transfers(day, fill_idle_gaps=False)
        _remove_invalid_same_site_transfers(day)
        _sort_and_recompute_day(day)
        _agent4_log(
            f"final_cleanup day={day_index} city={_normalize_text(day.get('city'))} events={len(day.get('events', []))} "
            f"elapsed={_agent4_elapsed(day_started):.2f}s"
        )
    _agent4_log(f"final_cleanup total elapsed={_agent4_elapsed(final_cleanup_started):.2f}s")
    _mark("final cleanup")

    _remove_post_inbound_home_events(
        daily_plans=daily_plans,
        selected_all_flight_legs=normalized.get("selected_all_flight_legs", []),
    )
    _mark("remove post inbound")

    validated_plans: List[Dict[str, Any]] = []
    validation_started = time.perf_counter()
    for day in daily_plans:
        checks = _validate_day_constraints(day)
        max_drive_ok = checks.get("max_drive_leg_le_4h", checks.get("max_drive_leg_le_5h", True))
        day["constraints"] = {
            "within_day_hours": checks.get("within_day_hours", True),
            "max_drive_leg_le_4h": max_drive_ok,
            "max_drive_leg_le_5h": max_drive_ok,
        }
        for w in checks.get("warnings", []):
            day.setdefault("warnings", []).append(w)
        validated_plans.append(day)
    _agent4_log(f"validation elapsed={_agent4_elapsed(validation_started):.2f}s")
    _mark("day validation")

    budget_validate_started = time.perf_counter()
    budget_spend_up_warnings = _spend_up_lifestyle_costs_for_budget(
        daily_plans=validated_plans,
        total_budget=total_budget,
        normalized=normalized,
    )
    budget_check = _validate_budget_constraints(
        daily_plans=validated_plans,
        total_budget=total_budget,
    )
    _agent4_log(
        f"budget_validate feasible={budget_check.get('budget_feasible')} "
        f"spend_up_warnings={len(budget_spend_up_warnings)} "
        f"elapsed={_agent4_elapsed(budget_validate_started):.2f}s"
    )
    _mark("budget validation")

    summary_started = time.perf_counter()
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
    warnings.extend(budget_spend_up_warnings)

    cost_rollup = _build_cost_rollup(validated_plans)
    itinerary_summary = _build_itinerary_summary(
        daily_plans=validated_plans,
        budget_check=budget_check,
        must_visit_total=must_visit_total,
        must_visit_done=must_visit_done,
    )
    _agent4_log(
        f"summary warnings={len(warnings)} cost_total={cost_rollup.get('total_cost')} "
        f"elapsed={_agent4_elapsed(summary_started):.2f}s"
    )
    _agent4_log(f"complete total={_agent4_elapsed(agent4_started):.2f}s")

    return _format_agent4_output(
        daily_plans=validated_plans,
        itinerary_summary=itinerary_summary,
        cost_rollup=cost_rollup,
        warnings=warnings,
    )


def _normalize_agent3_payload(router_output: Dict[str, Any]) -> Dict[str, Any]:
    payload = router_output if isinstance(router_output, dict) else {}
    trip_summary = payload.get("trip_summary", {}) or {}
    agent2_summary = payload.get("agent2_summary", {}) or {}
    parsed_request = payload.get("parsed_request", {}) or {}
    preferences = (
        trip_summary.get("preferences")
        or payload.get("preferences")
        or parsed_request.get("preferences")
        or {}
    )
    city_budget_plan = (
        payload.get("city_budget_plan")
        or payload.get("city_budget_plan_used")
        or agent2_summary.get("city_budget_plan")
        or []
    )

    return {
        "trip_summary": trip_summary,
        "country": trip_summary.get("country") or payload.get("country") or parsed_request.get("country"),
        "cost_level": payload.get("cost_level") or agent2_summary.get("cost_level"),
        "budget_strategy": payload.get("budget_strategy") or agent2_summary.get("budget_strategy"),
        "budget_status": payload.get("budget_status") or agent2_summary.get("budget_status"),
        "budget_cap_mode": (
            payload.get("budget_cap_mode")
            or parsed_request.get("budget_cap_mode")
            or trip_summary.get("budget_cap_mode")
        ),
        "budget_maximize_experience": (
            payload.get("budget_maximize_experience")
            or parsed_request.get("budget_maximize_experience")
            or trip_summary.get("budget_maximize_experience")
        ),
        "city_date_ranges": payload.get("city_date_ranges", {}) or {},
        "city_budget_plan": city_budget_plan,
        "budget_plan": (
            payload.get("budget_plan")
            or payload.get("recommended_spend")
            or agent2_summary.get("budget_plan")
            or {}
        ),
        "budget_check": payload.get("budget_check", {}) or {},
        "selected_hotels": payload.get("selected_hotels", []) or [],
        "selected_restaurants": payload.get("selected_restaurants", []) or [],
        "selected_pois": payload.get("selected_pois", []) or [],
        "selected_all_flight_legs": payload.get("selected_all_flight_legs", []) or [],
        "must_visit_attractions": payload.get("must_visit_attractions") or preferences.get("must_visit_attractions", []) or [],
        "refinement_context": payload.get("refinement_context", {}) if isinstance(payload.get("refinement_context"), dict) else {},
    }


def _extract_daily_city_slots(city_date_ranges: Dict[str, Any]) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []

    if not isinstance(city_date_ranges, dict):
        return slots

    city_order = {
        _normalize_text(city): index
        for index, city in enumerate(city_date_ranges.keys())
    }
    city_items = list(city_date_ranges.items())
    last_city_name = _normalize_text(city_items[-1][0]) if city_items else ""
    for city, info in city_items:
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
                    "_city_order": city_order.get(_normalize_text(city), i),
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
        explicit_nights = int(_safe_float(info.get("nights"), 0))
        if _normalize_text(city) == last_city_name and explicit_nights > 0:
            delta = max(delta, (end_dt - start_dt).days + 1)

        for i in range(delta):
            day = start_dt + timedelta(days=i)
            slots.append({
                "date": day.strftime("%Y-%m-%d"),
                "city": _normalize_text(city),
                "day_index_in_city": i + 1,
                "days_in_city": delta,
                "check_in": check_in,
                "check_out": check_out,
                "_city_order": city_order.get(_normalize_text(city), i),
            })

    def _slot_key(s: Dict[str, Any]) -> Tuple[str, str]:
        return (_normalize_text(s.get("date")), f"{int(_safe_float(s.get('_city_order'), 0)):04d}")

    slots.sort(key=_slot_key)

    # City ranges can share a transition date (checkout from city A and
    # check-in to city B). Keep the later city as the actual day card and let
    # hotel-transition logic add the previous-city checkout on that same day.
    by_date: Dict[str, Dict[str, Any]] = {}
    for slot in slots:
        date_key = _normalize_text(slot.get("date"))
        if date_key:
            by_date[date_key] = slot
    if len(by_date) != len(slots):
        slots = [by_date[key] for key in sorted(by_date.keys(), key=lambda x: _parse_date(x) or datetime.max)]

    city_counts: Dict[str, int] = {}
    for slot in slots:
        city_counts[_normalize_text(slot.get("city"))] = city_counts.get(_normalize_text(slot.get("city")), 0) + 1
    city_seen: Dict[str, int] = {}
    for slot in slots:
        city = _normalize_text(slot.get("city"))
        city_seen[city] = city_seen.get(city, 0) + 1
        slot["day_index_in_city"] = city_seen[city]
        slot["days_in_city"] = city_counts.get(city, slot.get("days_in_city") or 1)
        slot.pop("_city_order", None)
    return slots


def _build_city_budget_map(
    city_budget_plan: List[Dict[str, Any]],
    budget_plan: Optional[Dict[str, Any]] = None,
    daily_slots: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    if isinstance(city_budget_plan, list):
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
    if not isinstance(budget_plan, dict) or not isinstance(daily_slots, list) or not daily_slots:
        return result

    slot_counts: Dict[str, int] = {}
    for slot in daily_slots:
        city = _normalize_text(slot.get("city"))
        if city:
            slot_counts[city] = slot_counts.get(city, 0) + 1
    if not slot_counts:
        return result

    total_slot_days = max(1, sum(slot_counts.values()))
    total_hotel = _safe_float(budget_plan.get("hotel"), 0.0)
    total_food = _safe_float(budget_plan.get("food"), 0.0)
    total_attractions = _safe_float(budget_plan.get("attractions"), 0.0)
    for city, days in slot_counts.items():
        existing = result.get(city)
        if existing and any(_safe_float(existing.get(key), 0.0) > 0 for key in ("hotel_budget", "food_budget", "attractions_budget")):
            continue
        ratio = days / total_slot_days
        result[city] = {
            "hotel_budget": round(total_hotel * ratio, 2),
            "food_budget": round(total_food * ratio, 2),
            "attractions_budget": round(total_attractions * ratio, 2),
            "days": float(days),
        }
    return result


def _city_has_selected_hotel(hotels: List[Dict[str, Any]], city: str) -> bool:
    city_key = _normalize_text(city).lower()
    if not city_key:
        return False
    for hotel in hotels:
        if not isinstance(hotel, dict):
            continue
        if _normalize_text(hotel.get("city")).lower() != city_key:
            continue
        if _normalize_text(hotel.get("name")):
            return True
    return False


def _centroid_for_city_places(
    city: str,
    pois: List[Dict[str, Any]],
    restaurants: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    city_key = _normalize_text(city).lower()
    coords: List[Tuple[float, float]] = []
    for place in list(pois or []) + list(restaurants or []):
        if not isinstance(place, dict):
            continue
        if _normalize_text(place.get("city")).lower() != city_key:
            continue
        coord = _place_coord(place)
        if coord:
            coords.append((coord["lat"], coord["lng"]))
    if not coords:
        city_coord = _geocode_city_heigit(city)
        if city_coord:
            return {
                "lat": _safe_float(city_coord.get("lat"), None),
                "lng": _safe_float(city_coord.get("lng"), None),
            }
        return {"lat": None, "lng": None}
    return {
        "lat": round(sum(item[0] for item in coords) / len(coords), 7),
        "lng": round(sum(item[1] for item in coords) / len(coords), 7),
    }


def _build_fallback_hotel_for_city(
    city: str,
    city_budget: Dict[str, Any],
    pois: List[Dict[str, Any]],
    restaurants: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    city_name = _normalize_text(city)
    if not city_name:
        return None
    centroid = _centroid_for_city_places(city_name, pois, restaurants)
    days = max(1.0, _safe_float(city_budget.get("days"), 1.0) if isinstance(city_budget, dict) else 1.0)
    hotel_budget = _safe_float(city_budget.get("hotel_budget"), 0.0) if isinstance(city_budget, dict) else 0.0
    estimated_total = hotel_budget if hotel_budget > 0 else max(800.0, days * 1200.0)
    return {
        "city": city_name,
        "name": f"{city_name} itinerary base hotel",
        "area": "fallback itinerary lodging area",
        "estimated_total_cost": round(estimated_total, 2),
        "lat": centroid.get("lat"),
        "lng": centroid.get("lng"),
        "rating": 0.0,
        "stars": 0.0,
        "fallback_generated": True,
        "selection_note": "Fallback hotel generated because Agent3 did not return a selected hotel for this city.",
    }


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
            keys = _poi_reuse_keys_from_place(poi, city=city_key)
            if not keys or not (keys & used_poi_keys):
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


def _select_stay_hotel_cluster(
    hotel_clusters: List[Dict[str, Any]],
    day_index_in_city: int,
    days_in_city: int,
    departure_airport_coord: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    if not hotel_clusters:
        return None
    if len(hotel_clusters) == 1:
        return hotel_clusters[0]

    day_index = max(1, int(_safe_float(day_index_in_city, 1)))
    total_days = max(1, int(_safe_float(days_in_city, 1)))

    # Keep hotel assignment stable for short stays. If the trip is long enough
    # to justify a district move, shift once in the second half; when airport
    # coordinates are available, make the later stay closer to the airport.
    if total_days >= 5:
        switch_after = max(1, (total_days + 1) // 2)
        if day_index > switch_after:
            if departure_airport_coord:
                ranked = sorted(
                    hotel_clusters,
                    key=lambda cluster: _distance_to_coord((cluster.get("hotel") or {}), departure_airport_coord),
                )
                return ranked[0] if ranked else hotel_clusters[min(1, len(hotel_clusters) - 1)]
            return hotel_clusters[min(1, len(hotel_clusters) - 1)]

    return hotel_clusters[0]


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


def _distance_to_coord(place: Dict[str, Any], coord: Dict[str, float]) -> float:
    place_coord = _place_coord(place)
    if not place_coord or not coord:
        return 999999.0
    return _haversine_km(place_coord["lat"], place_coord["lng"], coord["lat"], coord["lng"])


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


def _load_airport_coord_by_code(code: str) -> Optional[Dict[str, float]]:
    airport = _load_airport_info_by_code(code)
    if not airport:
        return None
    lat = _safe_float(airport.get("lat"), None)
    lng = _safe_float(airport.get("lng"), None)
    if lat is None or lng is None:
        return None
    return {"lat": lat, "lng": lng}


def _load_airport_info_by_code(code: str) -> Optional[Dict[str, Any]]:
    airport_code = _normalize_text(code).upper()
    if not airport_code:
        return None
    path = os.path.join(os.path.dirname(__file__), "global_airports.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    airport = data.get(airport_code)
    if not isinstance(airport, dict):
        return None
    return airport


def _departure_airport_coords_by_city(selected_all_flight_legs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    coords: Dict[str, Dict[str, float]] = {}
    if not isinstance(selected_all_flight_legs, list):
        return coords
    for leg in selected_all_flight_legs:
        if not isinstance(leg, dict) or _normalize_text(leg.get("leg_type")) != "inbound":
            continue
        city = _normalize_text(leg.get("from_city"))
        flights = leg.get("selected_flights") or []
        flight = flights[0] if flights and isinstance(flights[0], dict) else {}
        code = (
            _normalize_text(flight.get("departure_airport_code"))
            or _normalize_text(flight.get("departure_code"))
            or _normalize_text(flight.get("from_airport"))
            or _normalize_text(leg.get("from_airport"))
        )
        coord = _load_airport_coord_by_code(code)
        if city and coord:
            coords[city] = coord
    return coords


def _local_transport_rate_per_minute(tier: str, is_international: bool = False) -> float:
    key = _normalize_text(tier).lower()
    if key == "ultra":
        if is_international:
            return INTERNATIONAL_ULTRA_LOCAL_TRANSPORT_RATE_PER_MINUTE
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


def _refinement_activity_radius_km(refinement_context: Dict[str, Any]) -> Optional[float]:
    if not isinstance(refinement_context, dict):
        return None
    if _normalize_text(refinement_context.get("mode")).lower() != "refine":
        return None
    value = _safe_float(refinement_context.get("max_activity_radius_km"), 0.0)
    if value in {5.0, 10.0, 15.0}:
        return value
    return None


def _filter_city_candidates_by_hotel_radius(
    city_candidates: Dict[str, List[Dict[str, Any]]],
    hotel: Optional[Dict[str, Any]],
    activity_radius_km: Optional[float] = None,
) -> Dict[str, Any]:
    pois = [x for x in (city_candidates.get("pois", []) or []) if isinstance(x, dict)]
    restaurants = [x for x in (city_candidates.get("restaurants", []) or []) if isinstance(x, dict)]
    has_refinement_radius = activity_radius_km is not None
    activity_radius = _safe_float(activity_radius_km, HOTEL_ACTIVITY_RADIUS_KM) if has_refinement_radius else HOTEL_ACTIVITY_RADIUS_KM
    meal_fallback_radius = activity_radius if has_refinement_radius else HOTEL_MEAL_FALLBACK_RADIUS_KM

    if not isinstance(hotel, dict):
        return {
            "pois": pois,
            "restaurants": restaurants,
            "radius_meta": {
                "poi_limit_km": activity_radius,
                "meal_limit_km": activity_radius,
                "meal_fallback_used": False,
                "poi_in_radius_count": len(pois),
                "meal_in_10km_count": len(restaurants),
                "meal_in_15km_count": len(restaurants),
                "hotel_missing_for_radius_filter": True,
            },
        }

    pois_in_radius = [p for p in pois if _is_within_hotel_radius(hotel, p, activity_radius)]
    restaurants_in_radius = [r for r in restaurants if _is_within_hotel_radius(hotel, r, activity_radius)]
    restaurants_in_fallback = [r for r in restaurants if _is_within_hotel_radius(hotel, r, meal_fallback_radius)]

    use_meal_fallback = not restaurants_in_radius and bool(restaurants_in_fallback)
    filtered_restaurants = restaurants_in_radius if restaurants_in_radius else restaurants_in_fallback
    meal_variety_backfill_used = False

    if (
        not has_refinement_radius
        and restaurants
        and len(filtered_restaurants) < MIN_MEAL_VARIETY_RESTAURANTS
    ):
        seen_restaurants = {
            (
                _normalize_text(item.get("name")).lower(),
                _normalize_text(item.get("area")).lower(),
            )
            for item in filtered_restaurants
        }
        backfill_candidates = [
            item
            for item in restaurants
            if (
                _normalize_text(item.get("name")).lower(),
                _normalize_text(item.get("area")).lower(),
            )
            not in seen_restaurants
        ]
        backfill_candidates.sort(
            key=lambda item: (
                _distance_km_between_places(hotel, item)
                if _distance_km_between_places(hotel, item) is not None
                else 9999.0,
                -_safe_float(item.get("rating"), 0.0),
                _normalize_text(item.get("name")).lower(),
            )
        )
        for item in backfill_candidates:
            if len(filtered_restaurants) >= MIN_MEAL_VARIETY_RESTAURANTS:
                break
            filtered_restaurants.append(item)
            meal_variety_backfill_used = True

    return {
        "pois": pois_in_radius,
        "restaurants": filtered_restaurants,
        "radius_meta": {
            "poi_limit_km": activity_radius,
            "meal_limit_km": meal_fallback_radius if use_meal_fallback else activity_radius,
            "meal_fallback_used": use_meal_fallback,
            "meal_variety_backfill_used": meal_variety_backfill_used,
            "poi_in_radius_count": len(pois_in_radius),
            "meal_in_10km_count": len([r for r in restaurants if _is_within_hotel_radius(hotel, r, 10.0)]),
            "meal_in_15km_count": len([r for r in restaurants if _is_within_hotel_radius(hotel, r, 15.0)]),
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


def _build_area_must_visit_fallbacks(
    city: str,
    must_visit_names: List[str],
    matched_must_visit_pois: List[Dict[str, Any]],
    hotel: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    city_key = _normalize_text(city).lower()
    area_map = AREA_MUST_VISIT_COORDS_BY_CITY.get(city_key, {})
    if not area_map:
        return []

    matched = {
        _normalize_text(poi.get("_must_visit_name_norm")).lower()
        for poi in matched_must_visit_pois
        if isinstance(poi, dict) and _normalize_text(poi.get("_must_visit_name_norm"))
    }
    fallbacks: List[Dict[str, Any]] = []
    for raw_name in must_visit_names:
        target = _normalize_text(raw_name).lower()
        if not target or target in matched:
            continue
        area = area_map.get(target)
        if not area:
            continue
        lat = _safe_float(area.get("lat"), None)
        lng = _safe_float(area.get("lng"), None)
        if (lat is None or lng is None) and isinstance(hotel, dict):
            lat = _safe_float(hotel.get("lat"), None)
            lng = _safe_float(hotel.get("lng"), None)
        label = _normalize_text(area.get("label")) or f"{_normalize_text(raw_name)} area walk"
        fallbacks.append(
            {
                "city": _normalize_text(city),
                "name": label,
                "address": f"{_normalize_text(raw_name)}, {_normalize_text(city)}",
                "estimated_total_cost": 0.0,
                "estimated_ticket_per_person": 0.0,
                "lat": lat,
                "lng": lng,
                "types": ["tourist_attraction", "point_of_interest", "shopping_mall"],
                "matched_query": f"{_normalize_text(raw_name)} area walk",
                "why_selected": "Fallback area walk for requested district-style must-visit.",
                "_must_visit_name_norm": target,
                "area_must_visit_fallback": True,
            }
        )
        matched.add(target)
    return fallbacks


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
            "estimated_nightly_per_room": _safe_float((hotel or {}).get("estimated_nightly_per_room"), 0.0),
            "realtime_nightly_per_room": _safe_float((hotel or {}).get("realtime_nightly_per_room"), 0.0),
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
            "max_drive_leg_le_4h": True,
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


def _meal_slot_for_index(index: int) -> str:
    if index == 0:
        return "breakfast"
    if index == 1:
        return "lunch"
    if index == 2:
        return "dinner"
    return "meal"


def _fallback_meal_name(day_plan: Dict[str, Any], city: str, index: int) -> str:
    slot = _meal_slot_for_index(index)
    hotel_name = _hotel_name_from_day(day_plan)
    city_name = city or "City"
    if slot == "breakfast":
        if hotel_name:
            return f"Hotel breakfast cafe at {hotel_name}"
        return f"{city_name} breakfast cafe"
    if slot == "lunch":
        if hotel_name:
            return f"Hotel lunch restaurant at {hotel_name}"
        return f"{city_name} lunch restaurant"
    if slot == "dinner":
        if hotel_name:
            return f"Hotel dinner restaurant at {hotel_name}"
        return f"{city_name} dinner restaurant"
    return f"{city_name} restaurant"


def _meal_event_cost(candidate: Dict[str, Any], fallback_meal_cost: float) -> float:
    estimated_total = _safe_float(candidate.get("estimated_total_cost"), 0.0)
    recommended_visits = max(1.0, _safe_float(candidate.get("recommended_visits"), 1.0))
    if estimated_total > 0:
        return round(max(LOW_QUALITY_MEAL_COST_FLOOR, estimated_total / recommended_visits), 2)

    estimated_meal_per_person = _safe_float(candidate.get("estimated_meal_per_person"), 0.0)
    if estimated_meal_per_person > 0:
        return round(max(LOW_QUALITY_MEAL_COST_FLOOR, estimated_meal_per_person), 2)

    return round(max(LOW_QUALITY_MEAL_COST_FLOOR, fallback_meal_cost), 2)


def _meal_duration_hours_for_index(index: int) -> float:
    return BREAKFAST_HOURS if index == 0 else MEAL_HOURS


def _restaurant_text_blob(restaurant: Dict[str, Any]) -> str:
    parts = [
        _normalize_text(restaurant.get("name")),
        _normalize_text(restaurant.get("area")),
        _normalize_text(restaurant.get("matched_query")),
        _normalize_text(restaurant.get("why_selected")),
    ]
    types = restaurant.get("types")
    if isinstance(types, list):
        parts.extend(_normalize_text(item) for item in types)
    return " ".join(part for part in parts if part).lower()


def _is_breakfast_restaurant(restaurant: Dict[str, Any]) -> bool:
    text = _restaurant_text_blob(restaurant)
    breakfast_terms = {
        "breakfast",
        "brunch",
        "cafe",
        "coffee",
        "bakery",
        "toast",
        "pancake",
        "pastry",
        "sandwich",
        "喫茶",
        "カフェ",
        "ベーカリー",
        "coffee shop",
    }
    dinner_terms = {
        "steak",
        "wagyu",
        "ramen",
        "unagi",
        "鰻",
        "sushi",
        "barbecue",
        "yakiniku",
        "izakaya",
        "焼肉",
        "居酒屋",
    }
    if any(term in text for term in breakfast_terms):
        return True
    if any(term in text for term in dinner_terms):
        return False
    return False


def _sort_restaurants_for_meal_slot(
    restaurants: List[Dict[str, Any]],
    anchors: List[Dict[str, Any]],
    meal_index: int,
) -> List[Dict[str, Any]]:
    slot = _meal_slot_for_index(meal_index)
    candidates = _sort_restaurants_by_meal_anchors(restaurants, anchors)
    if slot != "breakfast":
        return candidates
    breakfast_candidates = [item for item in candidates if _is_breakfast_restaurant(item)]
    return breakfast_candidates


def _restaurant_anchor_score(
    restaurant: Dict[str, Any],
    anchors: List[Dict[str, Any]],
) -> Tuple[int, float, str]:
    restaurant_coord = _place_coord(restaurant)
    if not restaurant_coord:
        return (1, 9999.0, _normalize_text(restaurant.get("name")).lower())

    best_distance = 9999.0
    for anchor in anchors:
        anchor_coord = _place_coord(anchor)
        if not anchor_coord:
            continue
        distance = _haversine_km(
            restaurant_coord["lat"],
            restaurant_coord["lng"],
            anchor_coord["lat"],
            anchor_coord["lng"],
        )
        best_distance = min(best_distance, distance)

    within_anchor = 0 if best_distance <= MEAL_ANCHOR_RADIUS_KM else 1
    return (within_anchor, round(best_distance, 4), _normalize_text(restaurant.get("name")).lower())


def _meal_anchor_places(day_plan: Dict[str, Any], selected_pois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    hotel = day_plan.get("hotel")
    if isinstance(hotel, dict):
        anchors.append(hotel)
    for poi in selected_pois:
        if isinstance(poi, dict):
            anchors.append(poi)
    return anchors


def _sort_restaurants_by_meal_anchors(
    restaurants: List[Dict[str, Any]],
    anchors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    valid_restaurants = [r for r in restaurants if isinstance(r, dict)]
    if not anchors:
        return valid_restaurants
    return sorted(valid_restaurants, key=lambda item: _restaurant_anchor_score(item, anchors))


def _restaurant_key_from_event(event: Dict[str, Any], city: str = "") -> str:
    if not isinstance(event, dict) or _normalize_text(event.get("type")) != "meal":
        return ""
    meta = event.get("meta") or {}
    if isinstance(meta, dict):
        key = _normalize_text(meta.get("restaurant_key"))
        if key:
            return key
        restaurant_name = _normalize_text(meta.get("restaurant_name"))
        if restaurant_name:
            return _restaurant_dedupe_key({"name": restaurant_name}, city=city)
    name = _normalize_text(event.get("name"))
    if name:
        return _restaurant_dedupe_key({"name": name}, city=city)
    return ""


def _restaurant_recently_used(
    restaurant_key: str,
    current_day_index: Optional[int],
    restaurant_last_used_day_index: Optional[Dict[str, int]],
    restaurant_cooldown_days: int,
) -> bool:
    if not restaurant_key or current_day_index is None or not isinstance(restaurant_last_used_day_index, dict):
        return False
    last_used_day = restaurant_last_used_day_index.get(restaurant_key)
    if last_used_day is None:
        return False
    day_gap = abs(current_day_index - last_used_day)
    return day_gap <= max(1, restaurant_cooldown_days)


def _used_restaurant_keys_for_day(day_plan: Dict[str, Any]) -> set:
    city = _normalize_text(day_plan.get("city"))
    keys = set()
    for event in day_plan.get("events", []) or []:
        key = _restaurant_key_from_event(event, city=city)
        if key:
            keys.add(key)
    return keys


def _select_restaurant_with_reuse_policy(
    restaurants: List[Dict[str, Any]],
    city: str,
    used_today_restaurant_keys: set,
    restaurant_last_used_day_index: Optional[Dict[str, int]],
    current_day_index: Optional[int],
    restaurant_cooldown_days: int,
    used_restaurant_keys: Optional[set] = None,
) -> Tuple[Dict[str, Any], str, str]:
    keyed_candidates: List[Tuple[Dict[str, Any], str]] = []
    for candidate in restaurants:
        if not isinstance(candidate, dict):
            continue
        key = _restaurant_dedupe_key(candidate, city=city)
        if not key:
            continue
        keyed_candidates.append((candidate, key))

    if not keyed_candidates:
        return {}, "", "none"

    trip_used = used_restaurant_keys if isinstance(used_restaurant_keys, set) else set()

    for candidate, key in keyed_candidates:
        if key in used_today_restaurant_keys:
            continue
        if _restaurant_recently_used(key, current_day_index, restaurant_last_used_day_index, restaurant_cooldown_days):
            continue
        if key in trip_used:
            continue
        return candidate, key, "fresh"

    for candidate, key in keyed_candidates:
        if key in used_today_restaurant_keys:
            continue
        if _restaurant_recently_used(key, current_day_index, restaurant_last_used_day_index, restaurant_cooldown_days):
            continue
        return candidate, key, "repeat_after_cooldown"

    for candidate, key in keyed_candidates:
        if key not in used_today_restaurant_keys:
            return candidate, key, "reuse_due_to_shortage"

    candidate, key = keyed_candidates[0]
    return candidate, key, "reuse_due_to_shortage"


def _warn_restaurant_reuse_if_needed(day_plan: Dict[str, Any], restaurant_name: str, meal_slot: str, reuse_reason: str) -> None:
    if reuse_reason == "repeat_after_cooldown":
        day_plan.setdefault("warnings", []).append(
            f"Reused {restaurant_name} for {meal_slot} because no unused restaurant candidate was available; the 2-day restaurant cooldown was respected."
        )
    elif reuse_reason == "reuse_due_to_shortage":
        day_plan.setdefault("warnings", []).append(
            f"Reused {restaurant_name} for {meal_slot} within the 2-day restaurant cooldown because no alternative restaurant candidate was available."
        )


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


def _subtract_minutes_from_time(value: Any, minutes: int) -> str:
    dt = _parse_time_value(value)
    if not dt:
        return ""
    total = max(0, dt.hour * 60 + dt.minute - max(0, int(minutes)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _max_time_value(*values: Any) -> str:
    parsed = [_parse_time_value(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return ""
    return max(parsed).strftime("%H:%M")


def _hotel_checkin_window(arrival_time: Any = None) -> Tuple[str, str]:
    if _format_time_value(arrival_time):
        start = _format_time_value(arrival_time) or "14:00"
    else:
        start = "14:00"
    end = _add_minutes_to_time(start, HOTEL_PROCESS_MINUTES) or "14:15"
    return start, end


def _hotel_checkout_window(departure_time: Any = None) -> Tuple[str, str]:
    departure = _parse_time_value(departure_time)
    noon = _parse_time_value("12:00")
    if departure:
        departure_minutes = departure.hour * 60 + departure.minute
        # For very early departures after midnight, keep checkout shortly before leaving
        # instead of wrapping it to the previous evening on the same report day.
        if departure_minutes <= 3 * 60:
            end_minutes = max(0, departure_minutes - AIRPORT_ARRIVAL_BUFFER_MINUTES)
            end = f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
        elif noon and departure < noon:
            end_minutes = max(6 * 60, departure_minutes - AIRPORT_ARRIVAL_BUFFER_MINUTES)
            end = f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
        else:
            end = "12:00"
    else:
        end = "12:00"
    start = _add_minutes_to_time(end, -HOTEL_PROCESS_MINUTES) or "11:45"
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

    dep_date = _normalize_text(leg.get("departure_date"))
    day_date = _normalize_text(day_plan.get("date"))
    if leg_type == "outbound" and dep_date and day_date == dep_date and _flight_arrival_date(dep_date, flight) != dep_date:
        kept: List[Dict[str, Any]] = []
        removed_names: List[str] = []
        for event in day_plan.get("events", []) or []:
            event_type = _normalize_text(event.get("type"))
            if event_type in {"flight", "drive", "local_transport", "flight_unavailable", "transfer_unavailable"}:
                kept.append(event)
                continue
            removed_names.append(_normalize_text(event.get("name")) or event_type or "event")
        if removed_names:
            day_plan["events"] = kept
            day_plan.setdefault("warnings", []).append(
                "Removed destination-city events from the outbound departure day because the selected flight arrives on the next calendar day: "
                + ", ".join(name for name in removed_names if name)
            )
            _sort_and_recompute_day(day_plan)
        return

    dep_dt = _parse_time_value(flight.get("departure_time"))
    arr_dt = _parse_time_value(flight.get("arrival_time"))
    if leg_type == "outbound" and not arr_dt:
        return
    if leg_type == "inbound" and not dep_dt:
        return

    cutoff = (arr_dt.hour * 60 + arr_dt.minute) if leg_type == "outbound" and arr_dt else (dep_dt.hour * 60 + dep_dt.minute)
    if leg_type == "inbound":
        cutoff = max(0, cutoff - AIRPORT_ARRIVAL_BUFFER_MINUTES)
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


def _remove_post_inbound_home_events(
    daily_plans: List[Dict[str, Any]],
    selected_all_flight_legs: List[Dict[str, Any]],
) -> None:
    if not isinstance(daily_plans, list) or not isinstance(selected_all_flight_legs, list):
        return

    def _is_inbound_terminal_event(event: Dict[str, Any]) -> bool:
        event_type = _normalize_text(event.get("type"))
        if event_type not in {"flight", "flight_unavailable", "transfer_unavailable"}:
            return False
        meta = event.get("meta") or {}
        return isinstance(meta, dict) and _normalize_text(meta.get("leg_type")) == "inbound"

    def _remove_events_after_inbound_terminal(day: Dict[str, Any]) -> None:
        events = day.get("events", [])
        if not isinstance(events, list):
            return
        terminal_index = next(
            (
                index
                for index, event in enumerate(events)
                if isinstance(event, dict) and _is_inbound_terminal_event(event)
            ),
            None,
        )
        if terminal_index is None or terminal_index >= len(events) - 1:
            return
        removed = [
            _normalize_text(event.get("name")) or _normalize_text(event.get("type")) or "event"
            for event in events[terminal_index + 1:]
            if isinstance(event, dict)
        ]
        day["events"] = events[:terminal_index + 1]
        if removed:
            day.setdefault("warnings", []).append(
                "Removed all schedule items after the inbound flight returned to the departure city: "
                + ", ".join(name for name in removed if name)
            )
        _sort_and_recompute_day(day)

    by_date = {_normalize_text(day.get("date")): day for day in daily_plans if isinstance(day, dict)}
    index_by_date = {
        _normalize_text(day.get("date")): index
        for index, day in enumerate(daily_plans)
        if isinstance(day, dict) and _normalize_text(day.get("date"))
    }
    first_inbound_day_index: Optional[int] = None
    for leg in selected_all_flight_legs:
        if not isinstance(leg, dict) or _normalize_text(leg.get("leg_type")) != "inbound":
            continue
        flights = leg.get("selected_flights") or []
        has_selected_flight = bool(flights and isinstance(flights[0], dict))
        airport_deadline: Optional[int] = None
        if has_selected_flight:
            departure = _event_minutes_from_hhmm(flights[0].get("departure_time"))
            if departure is None:
                continue
            airport_deadline = max(0, departure - AIRPORT_ARRIVAL_BUFFER_MINUTES)
        day = by_date.get(_normalize_text(leg.get("departure_date"))) or (daily_plans[-1] if daily_plans else None)
        if not isinstance(day, dict):
            continue
        day_index = index_by_date.get(_normalize_text(day.get("date")))
        if day_index is not None:
            if first_inbound_day_index is None or day_index < first_inbound_day_index:
                first_inbound_day_index = day_index

        if not has_selected_flight:
            if _needs_transport_placeholder(leg):
                _remove_events_after_inbound_terminal(day)
            continue

        kept: List[Dict[str, Any]] = []
        removed: List[str] = []
        for event in day.get("events", []) or []:
            event_type = _normalize_text(event.get("type"))
            if event_type == "flight":
                kept.append(event)
                continue
            start = _event_start_minutes(event)
            end = _event_end_minutes(event)
            if event_type == "checkout" and (end is None or end <= airport_deadline):
                kept.append(event)
                continue
            if event_type == "transfer" and (end is None or end <= airport_deadline):
                kept.append(event)
                continue
            if (start is not None and start >= airport_deadline) or (end is not None and end > airport_deadline):
                removed.append(_normalize_text(event.get("name")) or event_type or "event")
                continue
            kept.append(event)

        if removed:
            day["events"] = kept
            day.setdefault("warnings", []).append(
                "Removed destination-city events after the inbound airport deadline: "
                + ", ".join(name for name in removed if name)
            )
            _sort_and_recompute_day(day)
        _remove_events_after_inbound_terminal(day)

    if first_inbound_day_index is None or first_inbound_day_index >= len(daily_plans) - 1:
        return

    removed_days = [
        f"{_normalize_text(day.get('date'))} {_normalize_text(day.get('city'))}".strip()
        for day in daily_plans[first_inbound_day_index + 1:]
        if isinstance(day, dict)
    ]
    cutoff_day = daily_plans[first_inbound_day_index]
    if isinstance(cutoff_day, dict) and removed_days:
        cutoff_day.setdefault("warnings", []).append(
            "Removed all itinerary days after the inbound flight returned to the departure city: "
            + ", ".join(removed_days)
        )
    del daily_plans[first_inbound_day_index + 1:]


def _date_from_time_value(value: Any) -> str:
    value = _normalize_text(value)
    if not value:
        return ""
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", value)
    return match.group(1) if match else ""


def _flight_arrival_date(dep_date: str, flight: Dict[str, Any]) -> str:
    dep_date = _normalize_text(dep_date)
    explicit_arrival_date = (
        _normalize_text(flight.get("arrival_date"))
        or _date_from_time_value(flight.get("arrival_time"))
    )
    if explicit_arrival_date:
        return explicit_arrival_date

    explicit_departure_date = (
        _normalize_text(flight.get("departure_date"))
        or _date_from_time_value(flight.get("departure_time"))
    )
    if explicit_departure_date:
        dep_date = explicit_departure_date

    dep_minutes = _event_minutes_from_hhmm(flight.get("departure_time"))
    arr_minutes = _event_minutes_from_hhmm(flight.get("arrival_time"))
    dep_dt = _parse_date(dep_date)
    if not dep_dt:
        return dep_date

    duration_minutes = _safe_float(flight.get("duration_minutes") or flight.get("total_duration"), 0.0)
    if dep_minutes is not None and duration_minutes > 0 and dep_minutes + duration_minutes >= 24 * 60:
        return (dep_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    if dep_minutes is None or arr_minutes is None or arr_minutes > dep_minutes:
        return dep_date
    return (dep_dt + timedelta(days=1)).strftime("%Y-%m-%d")


def _reapply_flight_event_constraints(
    daily_plans: List[Dict[str, Any]],
    selected_all_flight_legs: List[Dict[str, Any]],
) -> None:
    if not isinstance(daily_plans, list) or not isinstance(selected_all_flight_legs, list):
        return

    by_date = {
        _normalize_text(day.get("date")): day
        for day in daily_plans
        if isinstance(day, dict) and _normalize_text(day.get("date"))
    }
    if not by_date:
        return

    for leg in selected_all_flight_legs:
        if not isinstance(leg, dict):
            continue
        flights = leg.get("selected_flights") or []
        if not flights or not isinstance(flights[0], dict):
            continue
        dep_date = _normalize_text(leg.get("departure_date"))
        day = by_date.get(dep_date)
        if day is None and _normalize_text(leg.get("leg_type")) == "outbound":
            day = by_date.get(_flight_arrival_date(dep_date, flights[0]))
        if day is None:
            continue
        _remove_impossible_city_events(day, leg, flights[0])


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


def _hotel_name_from_day(day_plan: Dict[str, Any]) -> str:
    hotel = day_plan.get("hotel") or {}
    if not isinstance(hotel, dict):
        return ""
    return _normalize_text(hotel.get("name"))


def _hotel_meta_from_day(day_plan: Dict[str, Any]) -> Dict[str, Any]:
    hotel = day_plan.get("hotel") or {}
    if not isinstance(hotel, dict):
        hotel = {}
    return _hotel_meta_from_value(day_plan, hotel)


def _hotel_meta_from_value(day_plan: Dict[str, Any], hotel: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(hotel, dict):
        hotel = {}
    return {
        "city": _normalize_text(day_plan.get("city")),
        "hotel_name": _normalize_text(hotel.get("name")),
        "lat": _safe_float(hotel.get("lat"), None),
        "lng": _safe_float(hotel.get("lng"), None),
    }


def _has_hotel_event(day_plan: Dict[str, Any], event_type: str, hotel_name: str = "") -> bool:
    target = _normalize_text(event_type)
    hotel_key = _normalize_text(hotel_name).lower()
    for event in day_plan.get("events", []) or []:
        if _normalize_text(event.get("type")) != target:
            continue
        if not hotel_key:
            return True
        if hotel_key in _event_node_name(event).lower():
            return True
    return False


def _should_checkout_on_day(day_plan: Dict[str, Any]) -> bool:
    if _normalize_text(day_plan.get("flight_departure_time")):
        return True
    day_dt = _parse_date(day_plan.get("date"))
    checkout_dt = _parse_date(day_plan.get("check_out"))
    if day_dt is not None and checkout_dt is not None and day_dt < checkout_dt:
        return False
    return True


def _place_text_blob(place: Dict[str, Any]) -> str:
    parts = [
        _normalize_text(place.get("name")),
        _normalize_text(place.get("address")),
        _normalize_text(place.get("area")),
        _normalize_text(place.get("matched_query")),
        _normalize_text(place.get("why_selected")),
    ]
    types = place.get("types")
    if isinstance(types, list):
        parts.extend(_normalize_text(item) for item in types)
    return " ".join(part for part in parts if part).lower()


def _is_night_poi(poi: Dict[str, Any]) -> bool:
    text = _place_text_blob(poi)
    night_terms = {
        "night",
        "night view",
        "night light",
        "light",
        "illumination",
        "observation",
        "sky",
        "tower",
        "view",
        "夜景",
        "ライト",
        "イルミ",
        "展望",
    }
    return any(term in text for term in night_terms)


def _distance_between_meta_km(a_meta: Dict[str, Any], b_meta: Dict[str, Any]) -> Optional[float]:
    a_coord = _place_coord(a_meta)
    b_coord = _place_coord(b_meta)
    if not a_coord or not b_coord:
        return None
    return _haversine_km(a_coord["lat"], a_coord["lng"], b_coord["lat"], b_coord["lng"])


def _meta_location_name(meta: Dict[str, Any]) -> str:
    if not isinstance(meta, dict):
        return ""
    return (
        _normalize_text(meta.get("restaurant_name"))
        or _normalize_text(meta.get("poi_name"))
        or _normalize_text(meta.get("hotel_name"))
        or _normalize_text(meta.get("name"))
    )


def _is_hotel_breakfast_event(event: Dict[str, Any], hotel_name: str = "") -> bool:
    if _normalize_text(event.get("type")) != "meal":
        return False
    meta = event.get("meta") or {}
    if not isinstance(meta, dict):
        return False
    if _normalize_text(meta.get("meal_slot")) != "breakfast":
        return False
    name = _normalize_text(event.get("name")).lower()
    restaurant = _normalize_text(meta.get("restaurant_name")).lower()
    hotel_key = _normalize_text(hotel_name).lower()
    return "hotel breakfast" in name or (hotel_key and hotel_key in name) or (hotel_key and hotel_key in restaurant)


def _same_site_transfer_is_unnecessary(current: Dict[str, Any], nxt: Dict[str, Any], hotel_name: str = "") -> bool:
    current_name = _normalize_text(_transfer_endpoint_name(current, as_destination=False)).lower()
    next_name = _normalize_text(_transfer_endpoint_name(nxt, as_destination=True)).lower()
    if current_name and next_name and current_name == next_name:
        return True
    hotel_key = _normalize_text(hotel_name).lower()
    if hotel_key and hotel_key in current_name and hotel_key in next_name:
        return True
    distance = _distance_between_meta_km(_event_place_meta(current), _event_place_meta(nxt))
    if distance is None:
        return False
    current_meta = current.get("meta") or {}
    next_meta = nxt.get("meta") or {}
    current_reason = _normalize_text(current_meta.get("reason")) if isinstance(current_meta, dict) else ""
    next_reason = _normalize_text(next_meta.get("reason")) if isinstance(next_meta, dict) else ""
    if distance <= 0.03:
        if "idle_gap_fill" in {current_reason, next_reason}:
            return True
        if current_name and next_name:
            if current_name.startswith("local area walk near ") and next_name in current_name:
                return True
            if next_name.startswith("local area walk near ") and current_name in next_name:
                return True
        return False
    return False


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
        if distance <= 0.03:
            return 0.0, "same_site", round(distance, 2)
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
                duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
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
                duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
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
        base = 3

    if schedule_end is not None and schedule_end <= 14 * 60:
        base = max(1, base - 1)

    if days_in_city >= 2 and day_index_in_city == 1:
        base = max(1, base - 1)

    return max(DAILY_MIN_POIS, min(3, base))


def _fallback_poi_name(day_plan: Dict[str, Any], index: int) -> str:
    city = _normalize_text(day_plan.get("city")) or "city center"
    day_index = max(1, int(_safe_float(day_plan.get("day_index_in_city"), 1)))
    themes = [
        "neighborhood orientation walk",
        "local park stroll",
        "shopping street walk",
        "riverside photo walk",
        "museum district walk",
        "market and cafe lane",
        "historic street stroll",
        "city viewpoint walk",
        "design and gallery lane",
    ]
    theme = themes[((day_index - 1) * 3 + max(0, index - 1)) % len(themes)]
    return f"{theme.title()} in {city}"


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
    schedule_start_minutes = _event_minutes_from_hhmm(day_plan.get("schedule_start"))
    daytime_schedule = schedule_start_minutes is None or schedule_start_minutes < 17 * 60
    regular = [
        p
        for p in pois
        if _normalize_text(p.get("name")).lower() not in must_ids
        and not (daytime_schedule and _is_night_poi(p))
    ]

    poi_quota = _target_poi_quota_for_day(day_plan)

    selected_pois: List[Dict[str, Any]] = []
    day_city = _normalize_text(day_plan.get("city"))
    for p in must_now:
        if len(selected_pois) >= poi_quota:
            break
        if daytime_schedule and _is_night_poi(p):
            continue
        poi_keys = _poi_reuse_keys_from_place(p, city=day_city)
        if used_poi_keys is not None and poi_keys and (poi_keys & used_poi_keys):
            continue
        selected_pois.append(p)

    if len(selected_pois) < poi_quota:
        for p in regular:
            if len(selected_pois) >= poi_quota:
                break
            poi_keys = _poi_reuse_keys_from_place(p, city=day_city)
            if used_poi_keys is not None and poi_keys and (poi_keys & used_poi_keys):
                continue
            selected_pois.append(p)

    while len(selected_pois) < poi_quota:
        hotel = day_plan.get("hotel") or {}
        fallback_index = len(selected_pois) + 1
        selected_pois.append({
            "name": _fallback_poi_name(day_plan, fallback_index),
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
        poi_keys = _poi_reuse_keys_from_place(poi, city=day_city)
        if used_poi_keys is not None and poi_keys:
            used_poi_keys.update(poi_keys)

    meal_anchors = _meal_anchor_places(day_plan, selected_pois)
    restaurants = _sort_restaurants_by_meal_anchors(restaurants, meal_anchors)
    if restaurants and meal_anchors and _restaurant_anchor_score(restaurants[0], meal_anchors)[0] != 0:
        day_plan.setdefault("warnings", []).append(
            f"No restaurant candidate was within {MEAL_ANCHOR_RADIUS_KM}km of the hotel or scheduled sights; using the nearest available meal option."
        )

    meal_quota = DAILY_REQUIRED_MEALS
    meal_added = 0
    used_today_restaurant_keys = set()
    city_name = _normalize_text(day_plan.get("city"))
    fallback_meal_cost = _fallback_meal_cost(city_budget)
    fallback_restaurants: List[Dict[str, Any]] = [r for r in restaurants if isinstance(r, dict)]
    fallback_cursor = 0
    for r in _sort_restaurants_for_meal_slot(restaurants, meal_anchors, meal_added):
        if meal_added >= meal_quota:
            break
        meal_duration_h = _meal_duration_hours_for_index(meal_added)
        if _safe_float(day_plan.get("used_hours"), 0.0) + meal_duration_h > DAY_AVAILABLE_HOURS:
            break

        restaurant_key = _restaurant_dedupe_key(r, city=_normalize_text(day_plan.get("city")))
        if not restaurant_key:
            continue

        if restaurant_key in used_today_restaurant_keys:
            continue
        if _restaurant_recently_used(
            restaurant_key,
            current_day_index,
            restaurant_last_used_day_index,
            restaurant_cooldown_days,
        ):
            continue
        if isinstance(used_restaurant_keys, set) and restaurant_key in used_restaurant_keys:
            continue

        meta = {
            "city": _normalize_text(day_plan.get("city")),
            "restaurant_key": restaurant_key,
            "restaurant_name": _normalize_text(r.get("name")),
            "lat": _safe_float(r.get("lat"), None),
            "lng": _safe_float(r.get("lng"), None),
            "meal_slot": _meal_slot_for_index(meal_added),
        }
        commute_h = 0.0
        previous = _last_place_event(day_plan)
        if previous is not None:
            commute_h, _, _ = _estimate_commute_hours_between_meta(_event_place_meta(previous), meta)
        if _safe_float(day_plan.get("used_hours"), 0.0) + commute_h + meal_duration_h > DAY_AVAILABLE_HOURS:
            break
        meal_name = _meal_display_name(r, city_name, meal_added)
        meal_cost = _meal_event_cost(r, fallback_meal_cost)
        _append_commute_before_place(day_plan, meal_name, meta)
        _append_event(
            day_plan=day_plan,
            event_type="meal",
            name=meal_name,
            duration_h=meal_duration_h,
            cost_estimated=meal_cost,
            meta=meta,
        )
        restaurant_last_used_day_index[restaurant_key] = current_day_index
        used_today_restaurant_keys.add(restaurant_key)
        if used_restaurant_keys is not None:
            used_restaurant_keys.add(restaurant_key)
        meal_added += 1

    while (
        meal_added < DAILY_REQUIRED_MEALS
        and _safe_float(day_plan.get("used_hours"), 0.0) + _meal_duration_hours_for_index(meal_added) <= DAY_AVAILABLE_HOURS
    ):
        meal_duration_h = _meal_duration_hours_for_index(meal_added)
        if fallback_restaurants:
            fallback = None
            fallback_key = ""
            meal_candidates = _sort_restaurants_for_meal_slot(fallback_restaurants, meal_anchors, meal_added)
            if meal_candidates:
                rotated_candidates = meal_candidates[fallback_cursor % len(meal_candidates):] + meal_candidates[:fallback_cursor % len(meal_candidates)]
                fallback, fallback_key, reuse_reason = _select_restaurant_with_reuse_policy(
                    restaurants=rotated_candidates,
                    city=city_name,
                    used_today_restaurant_keys=used_today_restaurant_keys,
                    restaurant_last_used_day_index=restaurant_last_used_day_index,
                    current_day_index=current_day_index,
                    restaurant_cooldown_days=restaurant_cooldown_days,
                    used_restaurant_keys=used_restaurant_keys,
                )
                fallback_cursor += 1
                if fallback:
                    _warn_restaurant_reuse_if_needed(
                        day_plan,
                        _normalize_text(fallback.get("name")) or "restaurant",
                        _meal_slot_for_index(meal_added),
                        reuse_reason,
                    )
                else:
                    fallback = None
                if fallback is None:
                    fallback = meal_candidates[meal_added % len(meal_candidates)]
                    fallback_key = (
                        _restaurant_dedupe_key(fallback, city=city_name)
                        or f"{city_name.lower()}|fallback-reuse:{meal_added+1}"
                    )
            if fallback is not None:
                meal_name = _meal_display_name(fallback, city_name, meal_added)
                meal_cost = _meal_event_cost(fallback, fallback_meal_cost)
                meta = {
                    "city": city_name,
                    "restaurant_key": fallback_key,
                    "restaurant_name": _normalize_text(fallback.get("name")),
                    "lat": _safe_float(fallback.get("lat"), None),
                    "lng": _safe_float(fallback.get("lng"), None),
                    "meal_slot": _meal_slot_for_index(meal_added),
                }
            elif meal_added != 0 and fallback_restaurants:
                fallback = fallback_restaurants[meal_added % len(fallback_restaurants)]
                fallback_key = (
                    _restaurant_dedupe_key(fallback, city=city_name)
                    or f"{city_name.lower()}|meal-fallback:{meal_added+1}"
                )
                meal_name = _meal_display_name(fallback, city_name, meal_added)
                meal_cost = _meal_event_cost(fallback, fallback_meal_cost)
                meta = {
                    "city": city_name,
                    "restaurant_key": fallback_key,
                    "restaurant_name": _normalize_text(fallback.get("name")),
                    "lat": _safe_float(fallback.get("lat"), None),
                    "lng": _safe_float(fallback.get("lng"), None),
                    "meal_slot": _meal_slot_for_index(meal_added),
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
                    "meal_slot": _meal_slot_for_index(meal_added),
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
                "meal_slot": _meal_slot_for_index(meal_added),
            }
        commute_h = 0.0
        previous = _last_place_event(day_plan)
        if previous is not None:
            commute_h, _, _ = _estimate_commute_hours_between_meta(_event_place_meta(previous), meta)
        if _safe_float(day_plan.get("used_hours"), 0.0) + commute_h + meal_duration_h > DAY_AVAILABLE_HOURS:
            break
        _append_commute_before_place(day_plan, meal_name, meta)
        _append_event(
            day_plan=day_plan,
            event_type="meal",
            name=meal_name,
            duration_h=meal_duration_h,
            cost_estimated=meal_cost,
            meta=meta,
        )
        restaurant_key = _normalize_text(meta.get("restaurant_key"))
        if restaurant_key:
            used_today_restaurant_keys.add(restaurant_key)
            restaurant_last_used_day_index[restaurant_key] = current_day_index
            if used_restaurant_keys is not None:
                used_restaurant_keys.add(restaurant_key)
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
                    and reason not in PROTECTED_TRANSFER_REASONS
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
                if reason in PROTECTED_TRANSFER_REASONS:
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
            leg_type = _normalize_text(leg.get("leg_type"))
            arrival_date = _flight_arrival_date(dep_date, f)
            arrival_day_offset = 0
            dep_dt = _parse_date(dep_date)
            arr_dt = _parse_date(arrival_date)
            if dep_dt and arr_dt:
                arrival_day_offset = max(0, (arr_dt - dep_dt).days)
            departure_airport_code = (
                _normalize_text(f.get("departure_airport_code"))
                or _normalize_text(f.get("departure_code"))
                or _normalize_text(f.get("from_airport"))
                or _normalize_text(leg.get("from_airport"))
            )
            arrival_airport_code = (
                _normalize_text(f.get("arrival_airport_code"))
                or _normalize_text(f.get("arrival_code"))
                or _normalize_text(f.get("to_airport"))
                or _normalize_text(leg.get("to_airport"))
            )
            airport_coord = _load_airport_coord_by_code(
                departure_airport_code if leg_type == "inbound" else arrival_airport_code
            )
            meta = {
                "leg_type": leg_type,
                "from_city": _normalize_text(leg.get("from_city")),
                "to_city": _normalize_text(leg.get("to_city")),
                "departure_airport": _normalize_text(f.get("departure_airport")),
                "departure_airport_code": departure_airport_code,
                "arrival_airport": _normalize_text(f.get("arrival_airport")),
                "arrival_airport_code": arrival_airport_code,
                "from_airport_candidate_source": _normalize_text(leg.get("from_airport_candidate_source")),
                "to_airport_candidate_source": _normalize_text(leg.get("to_airport_candidate_source")),
                "departure_time": _normalize_text(f.get("departure_time")),
                "departure_date": dep_date,
                "arrival_time": _normalize_text(f.get("arrival_time")),
                "arrival_date": arrival_date,
                "arrival_day_offset": arrival_day_offset,
                "travel_class": _normalize_text(f.get("travel_class")).lower(),
                "stops": _safe_int(f.get("stops"), 0),
                "selection_note": _normalize_text(f.get("selection_note")),
                "selection_status": _normalize_text(leg.get("selection_status")),
                "time_source": _normalize_text(f.get("provider")) or "selected_flight",
                "redeye_or_night_flight": _is_redeye_flight(f),
                "time_counting_rule": "night_flight_not_counted_in_day_hours" if _is_redeye_flight(f) else "flight_duration_plus_2h_buffer",
                "is_estimated_fallback": bool(f.get("is_estimated_fallback")),
                "ground_to_departure_airport_hours": _safe_float(f.get("ground_to_departure_airport_hours"), 0.0),
                "ground_from_arrival_airport_hours": _safe_float(f.get("ground_from_arrival_airport_hours"), 0.0),
                "lat": airport_coord.get("lat") if airport_coord else None,
                "lng": airport_coord.get("lng") if airport_coord else None,
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
            for source_key, role in [
                ("from_airport_candidate_source", "departure"),
                ("to_airport_candidate_source", "arrival"),
            ]:
                if _normalize_text(meta.get(source_key)) == "nearby_airport":
                    city_field = "from_city" if role == "departure" else "to_city"
                    day.setdefault("warnings", []).append(
                        f"{_normalize_text(leg.get(city_field))} has no direct airport candidate; "
                        f"using {_airport_role_display(meta, role)} with ground transfer included."
                    )
            if bool(f.get("is_estimated_fallback")):
                day.setdefault("warnings", []).append(
                    f"Flight price for {_normalize_text(leg.get('from_city'))}->{_normalize_text(leg.get('to_city'))} "
                    "is an estimated fallback because no live fare was selected."
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
            if _normalize_text(leg.get("leg_type")) in {"outbound", "intercity"}:
                start = "08:00" if _normalize_text(leg.get("leg_type")) == "outbound" else (
                    _normalize_text(day.get("intercity_drive_departure_time")) or "08:15"
                )
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
            duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
            cost_estimated=0.0,
            meta={
                "city": _normalize_text(first.get("city")),
                "lat": _safe_float((first.get("hotel") or {}).get("lat"), None),
                "lng": _safe_float((first.get("hotel") or {}).get("lng"), None),
            },
        )

    last = daily_plans[-1]
    if last.get("hotel") and _should_checkout_on_day(last):
        start, end = _hotel_checkout_window()
        _append_fixed_time_event(
            day_plan=last,
            event_type="checkout",
            name=f"Check-out { _normalize_text((last.get('hotel') or {}).get('name')) or 'Hotel' }",
            start=start,
            end=end,
            duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
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


def _find_intercity_transport_event(
    day_plan: Dict[str, Any],
    from_city: str = "",
    to_city: str = "",
) -> Optional[Dict[str, Any]]:
    from_key = _normalize_text(from_city).lower()
    to_key = _normalize_text(to_city).lower()
    for event in day_plan.get("events", []) or []:
        event_type = _normalize_text(event.get("type"))
        if event_type not in {"drive", "flight"}:
            continue
        meta = event.get("meta") or {}
        if not isinstance(meta, dict) or _normalize_text(meta.get("leg_type")) != "intercity":
            continue
        if from_key and _normalize_text(meta.get("from_city")).lower() != from_key:
            continue
        if to_key and _normalize_text(meta.get("to_city")).lower() != to_key:
            continue
        return event
    return None


def _checkin_start_after_intercity_arrival(arrival_end: int) -> int:
    lunch_start = _event_minutes_from_hhmm(NORMAL_LUNCH_START) or 12 * 60
    lunch_end = _event_minutes_from_hhmm(NORMAL_LUNCH_END) or (13 * 60 + 30)
    if lunch_start <= arrival_end < lunch_end:
        return max(lunch_end, arrival_end + 60)
    return arrival_end


def _ensure_hotel_transition_events(daily_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not daily_plans:
        return daily_plans

    protected_reasons = {"hotel_switch", "hotel_checkin_navigation", "arrival_to_hotel", "departure_from_hotel"}
    for day in daily_plans:
        cleaned: List[Dict[str, Any]] = []
        for event in day.get("events", []) or []:
            event_type = _normalize_text(event.get("type"))
            meta = event.get("meta") or {}
            reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
            if event_type in {"checkin", "checkout"}:
                continue
            if event_type == "transfer" and reason in protected_reasons:
                continue
            cleaned.append(event)
        day["events"] = cleaned
        _sort_and_recompute_day(day)

    for index, day in enumerate(daily_plans):
        prev_day = daily_plans[index - 1] if index > 0 else None
        hotel_obj = day.get("hotel") if isinstance(day.get("hotel"), dict) else {}
        hotel_name = _normalize_text((hotel_obj or {}).get("name"))
        prev_hotel_obj = prev_day.get("hotel") if isinstance(prev_day, dict) and isinstance(prev_day.get("hotel"), dict) else {}
        prev_hotel = _normalize_text((prev_hotel_obj or {}).get("name"))
        skip_before_arrival = (
            _normalize_text(day.get("skip_city_schedule")).lower() == "true"
            and _normalize_text(day.get("overnight_arrival")).lower() != "true"
        )
        missing_transport = _normalize_text(day.get("missing_required_outbound_transport")).lower() == "true"

        switch_checkin_added = False
        checkout_end_minutes = None
        if prev_hotel and prev_hotel != hotel_name:
            same_city = _normalize_text(prev_day.get("city")).lower() == _normalize_text(day.get("city")).lower()
            prev_city = _normalize_text(prev_day.get("city")) if isinstance(prev_day, dict) else ""
            current_city = _normalize_text(day.get("city"))
            intercity_event = None if same_city else _find_intercity_transport_event(day, from_city=prev_city, to_city=current_city)
            if intercity_event is not None:
                transport_start = _event_start_minutes(intercity_event)
                if transport_start is None:
                    transport_start = _event_minutes_from_hhmm(day.get("intercity_drive_departure_time")) or (8 * 60 + HOTEL_PROCESS_MINUTES)
                if _normalize_text(intercity_event.get("type")) == "flight":
                    airport_deadline = max(0, transport_start - AIRPORT_ARRIVAL_BUFFER_MINUTES)
                    airport_transfer_minutes = _airport_transfer_minutes_for_flight(intercity_event)
                    checkout_end = max(HOTEL_PROCESS_MINUTES, airport_deadline - airport_transfer_minutes)
                else:
                    checkout_end = max(HOTEL_PROCESS_MINUTES, transport_start)
                checkout_start = max(0, checkout_end - HOTEL_PROCESS_MINUTES)
                start, end = (
                    f"{checkout_start // 60:02d}:{checkout_start % 60:02d}",
                    f"{checkout_end // 60:02d}:{checkout_end % 60:02d}",
                )
            elif same_city and hotel_name and not skip_before_arrival and not missing_transport:
                start, end = "10:00", "10:15"
            else:
                start, end = _hotel_checkout_window()
            prev_checkout_meta = (
                _hotel_meta_from_value(prev_day, prev_hotel_obj or {})
                if isinstance(prev_day, dict)
                else _hotel_meta_from_value(day, prev_hotel_obj or {})
            )
            _append_fixed_time_event(
                day_plan=day,
                event_type="checkout",
                name=f"Check-out {prev_hotel}",
                start=start,
                end=end,
                duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
                cost_estimated=0.0,
                meta=prev_checkout_meta,
            )
            checkout_end_minutes = _event_minutes_from_hhmm(end)

            if hotel_name and not skip_before_arrival and not missing_transport:
                if intercity_event is not None:
                    arrival_end = _event_end_minutes(intercity_event)
                    if arrival_end is None:
                        arrival_end = _event_minutes_from_hhmm(day.get("schedule_start")) or (checkout_end_minutes or 8 * 60)
                    schedule_start_minutes = _event_minutes_from_hhmm(day.get("schedule_start"))
                    if _normalize_text(intercity_event.get("type")) == "flight" and schedule_start_minutes is not None:
                        arrival_end = max(arrival_end, schedule_start_minutes)
                    checkin_start = _checkin_start_after_intercity_arrival(arrival_end)
                    checkin_end = checkin_start + HOTEL_PROCESS_MINUTES
                    _append_fixed_time_event(
                        day_plan=day,
                        event_type="checkin",
                        name=f"Check-in {hotel_name}",
                        start=f"{checkin_start // 60:02d}:{checkin_start % 60:02d}",
                        end=f"{checkin_end // 60:02d}:{checkin_end % 60:02d}",
                        duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
                        cost_estimated=0.0,
                        meta={
                            **_hotel_meta_from_day(day),
                            "reason": "intercity_arrival_checkin",
                        },
                    )
                    switch_checkin_added = True
                elif same_city and checkout_end_minutes is not None:
                    duration = HOTEL_SWITCH_TRANSFER_MINUTES
                    transfer_start = checkout_end_minutes
                    transfer_end = transfer_start + duration
                    schedule_start = _event_minutes_from_hhmm(day.get("schedule_start"))
                    if schedule_start is not None and schedule_start > transfer_end:
                        transfer_start = max(checkout_end_minutes, schedule_start - duration)
                        transfer_end = transfer_start + duration
                    local_tier = _normalize_text(day.get("local_transport_tier")) or "medium"
                    is_international = bool(day.get("local_transport_is_international"))
                    commute_h = duration / 60.0
                    cost = _estimate_local_transfer_cost(commute_h, local_tier, is_international=is_international)
                    _append_fixed_time_event(
                        day_plan=day,
                        event_type="transfer",
                        name=f"Transfer {prev_hotel} -> {hotel_name}",
                        start=f"{transfer_start // 60:02d}:{transfer_start % 60:02d}",
                        end=f"{transfer_end // 60:02d}:{transfer_end % 60:02d}",
                        duration_h=commute_h,
                        cost_estimated=cost,
                        meta={
                            "city": _normalize_text(day.get("city")),
                            "from": prev_hotel,
                            "to": hotel_name,
                            "reason": "hotel_switch",
                            "duration_source": "fixed_hotel_switch",
                        },
                    )
                    day["transport_cost_estimated"] = round(
                        _safe_float(day.get("transport_cost_estimated"), 0.0) + cost,
                        2,
                    )
                    checkin_start = transfer_end
                    checkin_end = checkin_start + HOTEL_PROCESS_MINUTES
                    _append_fixed_time_event(
                        day_plan=day,
                        event_type="checkin",
                        name=f"Check-in {hotel_name}",
                        start=f"{checkin_start // 60:02d}:{checkin_start % 60:02d}",
                        end=f"{checkin_end // 60:02d}:{checkin_end % 60:02d}",
                        duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
                        cost_estimated=0.0,
                        meta={
                            **_hotel_meta_from_day(day),
                            "reason": "hotel_switch_checkin",
                        },
                    )
                    switch_checkin_added = True

        begins_new_stay = bool(hotel_name) and (index == 0 or not prev_hotel or prev_hotel != hotel_name)
        if begins_new_stay and not switch_checkin_added and not skip_before_arrival and not missing_transport:
            start, end = _hotel_checkin_window(day.get("schedule_start"))
            _append_fixed_time_event(
                day_plan=day,
                event_type="checkin",
                name=f"Check-in {hotel_name}",
                start=start,
                end=end,
                duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
                cost_estimated=0.0,
                meta=_hotel_meta_from_day(day),
            )

        if (
            hotel_name
            and index == len(daily_plans) - 1
            and not skip_before_arrival
            and not missing_transport
            and _should_checkout_on_day(day)
            and not _has_hotel_event(day, "checkout", hotel_name)
        ):
            start, end = _hotel_checkout_window()
            _append_fixed_time_event(
                day_plan=day,
                event_type="checkout",
                name=f"Check-out {hotel_name}",
                start=start,
                end=end,
                duration_h=round(HOTEL_PROCESS_MINUTES / 60.0, 2),
                cost_estimated=0.0,
                meta=_hotel_meta_from_day(day),
            )

        _sort_and_recompute_day(day)

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
            and _normalize_text(meta.get("reason")) in {"hotel_checkin_navigation", "arrival_to_hotel"}
        ):
            continue
        kept.append(event)
    day_plan["events"] = kept
    _sort_and_recompute_day(day_plan)

    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    is_international = bool(day_plan.get("local_transport_is_international"))

    inserts: List[Tuple[int, Dict[str, Any]]] = []
    for index, event in enumerate(events):
        if _normalize_text(event.get("type")) != "checkin":
            continue
        checkin_start = _event_start_minutes(event)
        if checkin_start is None:
            continue

        previous_event = events[index - 1] if index > 0 else None
        previous_meta = previous_event.get("meta") if isinstance(previous_event, dict) else {}
        if (
            isinstance(previous_event, dict)
            and _normalize_text(previous_event.get("type")) == "transfer"
            and isinstance(previous_meta, dict)
            and _normalize_text(previous_meta.get("to")) == _event_node_name(event)
        ):
            continue

        prev_location = None
        for prev_idx in range(index - 1, -1, -1):
            candidate = events[prev_idx]
            if _normalize_text(candidate.get("type")) == "transfer":
                continue
            if _is_location_event(candidate):
                prev_location = candidate
                break

        hotel_name = _event_node_name(event)
        reason = "hotel_checkin_navigation"
        from_name = "arrival point"
        commute_h = CHECKIN_TRANSFER_MINUTES / 60.0
        source = "fixed_pre_checkin_navigation"
        distance_km = None

        if prev_location is not None:
            from_name = _transfer_endpoint_name(prev_location, as_destination=False)
            commute_h, source, distance_km = _estimate_commute_hours_between_meta(
                _event_place_meta(prev_location),
                _event_place_meta(event),
            )
            if _normalize_text(prev_location.get("type")) == "flight":
                reason = "arrival_to_hotel"
        else:
            airport_arrival = _event_minutes_from_hhmm(day_plan.get("airport_arrival_time"))
            if airport_arrival is not None and airport_arrival < checkin_start:
                from_name = _airport_role_display(
                    {
                        "to_city": _normalize_text(day_plan.get("arrival_airport_city")) or _normalize_text(day_plan.get("city")),
                        "arrival_airport_code": _normalize_text(day_plan.get("arrival_airport_code")),
                        "to_airport_candidate_source": _normalize_text(day_plan.get("arrival_airport_candidate_source")),
                    },
                    "arrival",
                )
                reason = "arrival_to_hotel"
                source = "airport_arrival_to_hotel"
                commute_h = max(1, checkin_start - airport_arrival) / 60.0
            else:
                # On ordinary in-city days, assume the traveler starts at the
                # hotel; do not invent a pre-08:00 "arrival point" transfer.
                continue
            schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start"))
            if airport_arrival is None and schedule_start is not None and schedule_start < checkin_start:
                commute_h = min(commute_h, max(1, checkin_start - schedule_start) / 60.0)

        transfer_minutes = max(1, int(round(commute_h * 60)))
        airport_arrival = _event_minutes_from_hhmm(day_plan.get("airport_arrival_time"))
        if reason == "arrival_to_hotel" and airport_arrival is not None and airport_arrival < checkin_start:
            start_minutes = airport_arrival
        else:
            start_minutes = max(0, checkin_start - transfer_minutes)
            prev_end = _event_end_minutes(prev_location) if prev_location is not None else None
            if prev_end is not None and start_minutes < prev_end:
                start_minutes = prev_end
                if start_minutes + transfer_minutes > checkin_start:
                    checkin_start = start_minutes + transfer_minutes
                    checkin_end = checkin_start + HOTEL_PROCESS_MINUTES
                    event["start"] = f"{checkin_start // 60:02d}:{checkin_start % 60:02d}"
                    event["end"] = f"{checkin_end // 60:02d}:{checkin_end % 60:02d}"
                    event["duration_h"] = round(HOTEL_PROCESS_MINUTES / 60.0, 2)
        transfer_h = round((checkin_start - start_minutes) / 60.0, 2)
        cost = _estimate_local_transfer_cost(transfer_h, local_tier, is_international=is_international)
        inserts.append((
            index,
            {
                "type": "transfer",
                "name": f"Transfer {from_name} -> {hotel_name} for check-in",
                "start": f"{start_minutes // 60:02d}:{start_minutes % 60:02d}",
                "end": f"{checkin_start // 60:02d}:{checkin_start % 60:02d}",
                "duration_h": transfer_h,
                "cost_estimated": round(max(0.0, cost), 2),
                "meta": {
                    "city": _normalize_text(day_plan.get("city")),
                    "from": from_name,
                    "to": hotel_name,
                    "reason": reason,
                    "duration_source": source,
                    "distance_km": distance_km,
                    "duration_minutes": int(round(transfer_h * 60)),
                },
            },
        ))

    if not inserts:
        return

    offset = 0
    for index, transfer_event in inserts:
        events.insert(index + offset, transfer_event)
        day_plan["transport_cost_estimated"] = round(
            _safe_float(day_plan.get("transport_cost_estimated"), 0.0)
            + _safe_float(transfer_event.get("cost_estimated"), 0.0),
            2,
        )
        offset += 1
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
    if _is_hotel_breakfast_event(first_event, hotel_name):
        should_add_day_start = False
    first_meta = first_event.get("meta") or {}
    if (
        checkins
        and _normalize_text(first_event.get("type")) == "meal"
        and isinstance(first_meta, dict)
        and _normalize_text(first_meta.get("meal_slot")) == "breakfast"
    ):
        should_add_day_start = False
    if should_add_day_start and first_start is not None:
        first_name = _event_node_name(first_event)
        for event in day_plan.get("events", []):
            if _normalize_text(event.get("type")) != "transfer":
                continue
            meta = event.get("meta") or {}
            if not isinstance(meta, dict):
                continue
            transfer_end = _event_end_minutes(event)
            if transfer_end is None or transfer_end > first_start:
                continue
            if _normalize_text(meta.get("to")) == first_name:
                should_add_day_start = False
                break

    if should_add_day_start and first_start is not None:
        start_minutes = max(0, first_start - HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES)
        end_minutes = first_start
        if earliest_checkin_end is not None and start_minutes < earliest_checkin_end:
            start_minutes = earliest_checkin_end
            end_minutes = start_minutes + HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES
            first_index = next((idx for idx, event in enumerate(day_plan.get("events", [])) if event is first_event), None)
            if first_index is not None and end_minutes > first_start:
                _shift_flexible_tail(day_plan.get("events", []), first_index, end_minutes - first_start)
        _append_fixed_time_event(
            day_plan=day_plan,
            event_type="transfer",
            name=f"Transfer from {hotel_name} to first stop",
            start=f"{start_minutes // 60:02d}:{start_minutes % 60:02d}",
            end=f"{end_minutes // 60:02d}:{end_minutes % 60:02d}",
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

    should_add_day_end = not (has_checkout and not checkins)
    if should_add_day_end:
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


def _airport_role_display(meta: Dict[str, Any], role: str) -> str:
    if not isinstance(meta, dict):
        return "airport"
    role_key = "departure" if role == "departure" else "arrival"
    city_key = "from_city" if role_key == "departure" else "to_city"
    source_key = "from_airport_candidate_source" if role_key == "departure" else "to_airport_candidate_source"
    code = (
        _normalize_text(meta.get(f"{role_key}_airport_code"))
        or _normalize_text(meta.get(f"{role_key}_code"))
    )
    airport_name = _normalize_text(meta.get(f"{role_key}_airport"))
    city = _normalize_text(meta.get(city_key))
    source = _normalize_text(meta.get(source_key))
    label = airport_name or (f"{code} Airport" if code else f"{city or role_key.title()} Airport")
    airport_info = _load_airport_info_by_code(code) if code else None
    airport_city = _normalize_text((airport_info or {}).get("city")) if airport_info else ""
    inferred_nearby = bool(city and airport_city and city.lower() != airport_city.lower())
    if source == "nearby_airport" or inferred_nearby:
        if city:
            return f"{label} (nearby airport for {city})"
        return f"{label} (nearby airport)"
    return label


def _transfer_endpoint_name(event: Dict[str, Any], as_destination: bool) -> str:
    if _normalize_text(event.get("type")) == "flight":
        meta = event.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        return _airport_role_display(meta, "departure" if as_destination else "arrival")
    return _event_node_name(event)


def _event_node_name(event: Dict[str, Any]) -> str:
    event_type = _normalize_text(event.get("type"))
    if event_type == "flight":
        meta = event.get("meta") or {}
        leg_type = _normalize_text((meta or {}).get("leg_type"))
        if leg_type == "inbound":
            return _airport_role_display(meta, "departure")
        if leg_type in {"outbound", "intercity"}:
            return _airport_role_display(meta, "arrival")
        return _normalize_text(event.get("name")) or "Flight"
    if event_type in {"checkin", "checkout"}:
        name = _normalize_text(event.get("name"))
        name = name.replace("Check-in ", "").replace("Check-out ", "").strip()
        return name or "Hotel"
    return _normalize_text(event.get("name")) or event_type or "stop"


def _estimate_generic_transfer(day_plan: Dict[str, Any], from_event: Dict[str, Any], to_event: Dict[str, Any]) -> Tuple[float, float]:
    from_meta = _event_place_meta(from_event)
    to_meta = _event_place_meta(to_event)
    commute_h, _, distance_km = _estimate_commute_hours_between_meta(from_meta, to_meta)
    if commute_h <= 0 or (distance_km is not None and distance_km <= WALK_DISTANCE_KM):
        return WALK_TRANSFER_MINUTES / 60.0, 0.0
    if commute_h <= 0:
        commute_h = DEFAULT_INTER_EVENT_TRANSFER_MINUTES / 60.0
    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    is_international = bool(day_plan.get("local_transport_is_international"))
    cost = _estimate_local_transfer_cost(commute_h, local_tier, is_international=is_international)
    return commute_h, cost


def _airport_transfer_minutes_for_flight(event: Dict[str, Any]) -> int:
    meta = event.get("meta") or {}
    if isinstance(meta, dict):
        ground_h = _safe_float(meta.get("ground_to_departure_airport_hours"), 0.0)
        if ground_h > 0:
            return max(MIN_AIRPORT_TRANSFER_MINUTES, int(round(ground_h * 60)))
    return AIRPORT_TRANSFER_MINUTES


def _shift_flexible_tail(events: List[Dict[str, Any]], start_index: int, delta_minutes: int) -> None:
    if delta_minutes <= 0:
        return
    fixed_types = {"flight", "drive", "checkin", "checkout"}
    for idx in range(start_index, len(events)):
        event = events[idx]
        event_type = _normalize_text(event.get("type"))
        if event_type in fixed_types:
            break
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if event_type == "transfer" and reason in PROTECTED_TRANSFER_REASONS:
            break
        if event_type == "poi" and reason == "evening_after_dinner":
            break
        if event_type == "meal" and isinstance(meta, dict) and _normalize_text(meta.get("meal_slot")) in {"breakfast", "lunch", "dinner"}:
            break
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            continue
        new_start = start + delta_minutes
        new_end = end + delta_minutes
        event["start"] = f"{new_start // 60:02d}:{new_start % 60:02d}"
        event["end"] = f"{new_end // 60:02d}:{new_end % 60:02d}"


def _poi_event_key_from_event(event: Dict[str, Any], city: str = "") -> str:
    meta = event.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    poi_name = _normalize_text(meta.get("poi_name")) or _normalize_text(event.get("name"))
    address = _normalize_text(meta.get("address"))
    return _poi_dedupe_key({"name": poi_name, "address": address, "place_id": meta.get("place_id")}, city=city)


def _normalize_poi_reuse_name(name: Any) -> str:
    normalized = _normalize_text(name).lower()
    normalized = re.sub(r"\s*\(\d{4}-\d{2}-\d{2}(?:\s*#\d+)?\)", "", normalized).strip()
    normalized = re.sub(r"\s*\(day-\d+(?:\s*#\d+)?\)", "", normalized).strip()
    normalized = re.sub(r"\s+#\d+$", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _poi_name_reuse_key(name: Any, city: str = "") -> str:
    normalized = _normalize_poi_reuse_name(name)
    if not normalized:
        return ""
    return f"{_normalize_text(city).lower()}|name:{normalized}"


def _poi_reuse_keys_from_place(place: Dict[str, Any], city: str = "") -> set:
    if not isinstance(place, dict):
        return set()
    keys = set()
    full_key = _poi_dedupe_key(place, city=city)
    if full_key:
        keys.add(full_key)
    name_key = _poi_name_reuse_key(place.get("name") or place.get("poi_name"), city=city)
    if name_key:
        keys.add(name_key)
    return keys


def _poi_reuse_keys_from_event(event: Dict[str, Any], city: str = "") -> set:
    meta = event.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    poi_like = {
        "place_id": meta.get("place_id"),
        "name": _normalize_text(meta.get("poi_name")) or _normalize_text(event.get("name")),
        "address": _normalize_text(meta.get("address")),
    }
    return _poi_reuse_keys_from_place(poi_like, city=city)


def _poi_counts_for_reuse(event: Dict[str, Any]) -> bool:
    if _normalize_text(event.get("type")) != "poi":
        return False
    meta = event.get("meta") or {}
    reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
    if reason == "idle_gap_fill":
        return bool(isinstance(meta, dict) and meta.get("real_gap_fill") is True)
    return True


def _collect_scheduled_poi_keys(
    day_plans: List[Dict[str, Any]],
    exclude_generated_day: Optional[Dict[str, Any]] = None,
) -> set:
    keys = set()
    for day in day_plans or []:
        city = _normalize_text(day.get("city"))
        for event in day.get("events", []) or []:
            if not _poi_counts_for_reuse(event):
                continue
            meta = event.get("meta") or {}
            reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
            if day is exclude_generated_day and reason == "idle_gap_fill":
                continue
            keys.update(_poi_reuse_keys_from_event(event, city=city))
    return keys


def _candidate_to_gap_poi_event(candidate: Dict[str, Any], city: str, start: int, end: int) -> Dict[str, Any]:
    name = _normalize_text(candidate.get("name")) or "Nearby attraction"
    meta = {
        "city": _normalize_text(city),
        "poi_name": name,
        "address": _normalize_text(candidate.get("address")),
        "lat": _safe_float(candidate.get("lat"), None),
        "lng": _safe_float(candidate.get("lng"), None),
        "place_id": _normalize_text(candidate.get("place_id")),
        "reason": "idle_gap_fill",
        "real_gap_fill": True,
    }
    must_visit_name = _normalize_text(candidate.get("_must_visit_name_norm"))
    if must_visit_name:
        meta["must_visit_name"] = must_visit_name
    return {
        "type": "poi",
        "name": name,
        "start": f"{start // 60:02d}:{start % 60:02d}",
        "end": f"{end // 60:02d}:{end % 60:02d}",
        "duration_h": round(max(1, end - start) / 60.0, 2),
        "cost_estimated": round(max(0.0, _safe_float(candidate.get("estimated_total_cost"), 0.0)), 2),
        "meta": meta,
    }


def _build_adjacent_transfer_event(
    day_plan: Dict[str, Any],
    from_event: Dict[str, Any],
    to_event: Dict[str, Any],
    start: int,
    end: int,
    reason: str = "adjacent_event_transfer",
) -> Dict[str, Any]:
    transfer_h = round(max(1, end - start) / 60.0, 2)
    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    is_international = bool(day_plan.get("local_transport_is_international"))
    _, estimated_cost = _estimate_generic_transfer(day_plan, from_event, to_event)
    transfer_cost = estimated_cost
    if transfer_cost <= 0 and (end - start) > WALK_TRANSFER_MINUTES:
        transfer_cost = _estimate_local_transfer_cost(transfer_h, local_tier, is_international=is_international)
    transfer_kind = "Walk" if transfer_cost <= 0 and (end - start) <= WALK_TRANSFER_MINUTES else "Transfer"
    from_name = _transfer_endpoint_name(from_event, as_destination=False)
    to_name = _transfer_endpoint_name(to_event, as_destination=True)
    return {
        "type": "transfer",
        "name": f"{transfer_kind} {from_name} -> {to_name}",
        "start": f"{start // 60:02d}:{start % 60:02d}",
        "end": f"{end // 60:02d}:{end % 60:02d}",
        "duration_h": transfer_h,
        "cost_estimated": round(max(0.0, transfer_cost), 2),
        "meta": {
            "city": _normalize_text(day_plan.get("city")),
            "from": from_name,
            "to": to_name,
            "reason": reason,
            "duration_source": "estimated_commute_capped_by_timeline_gap",
        },
    }


def _select_real_gap_poi(
    day_plan: Dict[str, Any],
    current: Dict[str, Any],
    nxt: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    blocked_poi_keys: set,
    current_end: int,
    next_start: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(candidates, list):
        return None
    city = _normalize_text(day_plan.get("city"))
    current_meta = _event_place_meta(current)
    next_meta = _event_place_meta(nxt)
    ranked: List[Tuple[float, int, Dict[str, Any], int, int]] = []
    local_blocked = set(blocked_poi_keys or set())
    for event in day_plan.get("events", []) or []:
        if _poi_counts_for_reuse(event):
            local_blocked.update(_poi_reuse_keys_from_event(event, city=city))

    for candidate in candidates:
        if not isinstance(candidate, dict) or not _place_coord(candidate):
            continue
        candidate_keys = _poi_reuse_keys_from_place(candidate, city=city)
        if candidate_keys and (candidate_keys & local_blocked):
            continue
        if _is_night_poi(candidate) and next_start <= 18 * 60:
            continue

        poi_event = _candidate_to_gap_poi_event(candidate, city, current_end, current_end + MIN_REAL_GAP_POI_VISIT_MINUTES)
        to_poi_h, _ = _estimate_generic_transfer(day_plan, current, poi_event)
        from_poi_h, _ = _estimate_generic_transfer(day_plan, poi_event, nxt)
        to_poi_min = max(1, int(round(max(0.08, to_poi_h) * 60)))
        from_poi_min = max(1, int(round(max(0.08, from_poi_h) * 60)))
        visit_start = current_end + to_poi_min
        visit_end = next_start - from_poi_min
        visit_minutes = visit_end - visit_start
        if visit_minutes < MIN_REAL_GAP_POI_VISIT_MINUTES:
            continue

        current_distance = _distance_between_meta_km(current_meta, poi_event.get("meta") or {})
        next_distance = _distance_between_meta_km(poi_event.get("meta") or {}, next_meta)
        known_distances = [d for d in [current_distance, next_distance] if d is not None]
        if known_distances and min(known_distances) > GAP_REAL_POI_RADIUS_KM:
            continue

        route_minutes = to_poi_min + from_poi_min
        nearest_distance = min(known_distances) if known_distances else 999.0
        ranked.append((nearest_distance, route_minutes, candidate, to_poi_min, from_poi_min))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], _normalize_text(item[2].get("name")).lower()))
    _, _, candidate, to_poi_min, from_poi_min = ranked[0]
    visit_start = current_end + to_poi_min
    visit_end = next_start - from_poi_min
    return {
        "candidate": candidate,
        "to_poi_minutes": to_poi_min,
        "from_poi_minutes": from_poi_min,
        "visit_start": visit_start,
        "visit_end": visit_end,
    }


def _ensure_adjacent_event_transfers(
    day_plan: Dict[str, Any],
    gap_poi_candidates: Optional[List[Dict[str, Any]]] = None,
    blocked_poi_keys: Optional[set] = None,
    fill_idle_gaps: bool = True,
) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list) or len(events) <= 1:
        return

    cleaned: List[Dict[str, Any]] = []
    for event in events:
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if reason == "adjacent_event_transfer" or (fill_idle_gaps and reason == "idle_gap_fill"):
            continue
        cleaned.append(event)
    day_plan["events"] = cleaned

    _sort_and_recompute_day(day_plan)
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    inserts: List[Tuple[int, Dict[str, Any]]] = []
    gap_selected_keys = set()
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
        if current_end is None or next_start is None:
            continue
        hotel_name = _hotel_name_from_day(day_plan)
        if _same_site_transfer_is_unnecessary(current, nxt, hotel_name):
            continue

        estimated_transfer_h, estimated_cost = _estimate_generic_transfer(day_plan, current, nxt)
        estimated_transfer_minutes = max(1, int(round(max(0.08, estimated_transfer_h) * 60)))
        next_meta = nxt.get("meta") or {}
        next_leg_type = _normalize_text(next_meta.get("leg_type")) if isinstance(next_meta, dict) else ""
        if next_type == "flight" and next_leg_type in {"outbound", "inbound", "intercity"}:
            airport_transfer_minutes = _airport_transfer_minutes_for_flight(nxt)
            estimated_transfer_minutes = max(estimated_transfer_minutes, airport_transfer_minutes)
            estimated_transfer_h = max(estimated_transfer_h, airport_transfer_minutes / 60.0)
        if next_type == "flight" and next_leg_type in {"outbound", "inbound", "intercity"}:
            airport_deadline = max(0, next_start - AIRPORT_ARRIVAL_BUFFER_MINUTES)
            latest_previous_end = max(0, airport_deadline - estimated_transfer_minutes)
            if current_end > latest_previous_end and current_type in {"poi", "meal"}:
                current_start = _event_start_minutes(current)
                if current_start is None or latest_previous_end <= current_start:
                    continue
                current["end"] = f"{latest_previous_end // 60:02d}:{latest_previous_end % 60:02d}"
                current["duration_h"] = round((latest_previous_end - current_start) / 60.0, 2)
                current_end = latest_previous_end
            if current_end > airport_deadline:
                continue

        gap_h = round(max(0, next_start - current_end) / 60.0, 2)

        # Use estimated commute duration for billing, do not bill the whole schedule gap.
        transfer_h = max(0.08, estimated_transfer_h if gap_h <= 0 else min(gap_h, estimated_transfer_h))
        transfer_minutes = max(1, int(round(transfer_h * 60)))
        if next_type == "flight" and next_leg_type in {"outbound", "inbound", "intercity"}:
            transfer_end = max(0, next_start - AIRPORT_ARRIVAL_BUFFER_MINUTES)
            transfer_start = max(current_end, transfer_end - transfer_minutes)
        else:
            original_gap_minutes = max(0, next_start - current_end)
            can_fill_pre_meal_gap = (
                fill_idle_gaps
                and
                next_type in {"poi", "meal"}
                and current_type in {"poi", "meal", "checkin", "checkout"}
                and original_gap_minutes >= transfer_minutes + MIN_IDLE_GAP_FILL_MINUTES
            )
            if can_fill_pre_meal_gap:
                real_gap_poi = _select_real_gap_poi(
                    day_plan=day_plan,
                    current=current,
                    nxt=nxt,
                    candidates=gap_poi_candidates or [],
                    blocked_poi_keys=set(blocked_poi_keys or set()) | gap_selected_keys,
                    current_end=current_end,
                    next_start=next_start,
                )
                if real_gap_poi:
                    candidate = real_gap_poi["candidate"]
                    candidate_keys = _poi_reuse_keys_from_place(candidate, city=_normalize_text(day_plan.get("city")))
                    if candidate_keys:
                        gap_selected_keys.update(candidate_keys)
                    to_poi_minutes = int(real_gap_poi["to_poi_minutes"])
                    from_poi_minutes = int(real_gap_poi["from_poi_minutes"])
                    visit_start = int(real_gap_poi["visit_start"])
                    visit_end = int(real_gap_poi["visit_end"])
                    gap_poi_event = _candidate_to_gap_poi_event(
                        candidate=candidate,
                        city=_normalize_text(day_plan.get("city")),
                        start=visit_start,
                        end=visit_end,
                    )
                    inserts.append((
                        index + 1,
                        _build_adjacent_transfer_event(
                            day_plan=day_plan,
                            from_event=current,
                            to_event=gap_poi_event,
                            start=current_end,
                            end=current_end + to_poi_minutes,
                        ),
                    ))
                    inserts.append((index + 1, gap_poi_event))
                    inserts.append((
                        index + 1,
                        _build_adjacent_transfer_event(
                            day_plan=day_plan,
                            from_event=gap_poi_event,
                            to_event=nxt,
                            start=visit_end,
                            end=visit_end + from_poi_minutes,
                        ),
                    ))
                    continue
                transfer_end = next_start
                transfer_start = max(current_end, transfer_end - transfer_minutes)
                filler_minutes = transfer_start - current_end
                if filler_minutes >= MIN_IDLE_GAP_FILL_MINUTES:
                    current_place_meta = _event_place_meta(current)
                    meal_slot = _normalize_text(next_meta.get("meal_slot")) if isinstance(next_meta, dict) else ""
                    next_label = meal_slot or _event_node_name(nxt)
                    filler_name = f"Local area walk near {_event_node_name(current)} before {next_label}"
                    inserts.append((
                        index + 1,
                        {
                            "type": "poi",
                            "name": filler_name,
                            "start": f"{current_end // 60:02d}:{current_end % 60:02d}",
                            "end": f"{transfer_start // 60:02d}:{transfer_start % 60:02d}",
                            "duration_h": round(filler_minutes / 60.0, 2),
                            "cost_estimated": 0.0,
                            "meta": {
                                "city": _normalize_text(day_plan.get("city")),
                                "poi_name": filler_name,
                                "lat": _safe_float(current_place_meta.get("lat"), None),
                                "lng": _safe_float(current_place_meta.get("lng"), None),
                                "reason": "idle_gap_fill",
                                "from": _event_node_name(current),
                                "before": _event_node_name(nxt),
                            },
                        },
                    ))
            else:
                transfer_start = current_end
                transfer_end = current_end + transfer_minutes
            if transfer_end > next_start:
                if next_type in {"flight", "drive", "checkin", "checkout"}:
                    continue
                _shift_flexible_tail(events, index + 1, transfer_end - next_start)
                next_start = transfer_end
        transfer_h = round(max(1, transfer_end - transfer_start) / 60.0, 2)
        local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
        is_international = bool(day_plan.get("local_transport_is_international"))
        transfer_cost = _estimate_local_transfer_cost(transfer_h, local_tier, is_international=is_international)
        if transfer_cost <= 0:
            transfer_cost = estimated_cost
        transfer_kind = "Walk" if transfer_cost <= 0 and transfer_minutes <= WALK_TRANSFER_MINUTES else "Transfer"
        from_name = _transfer_endpoint_name(current, as_destination=False)
        to_name = _transfer_endpoint_name(nxt, as_destination=True)
        transfer_event = {
            "type": "transfer",
            "name": f"{transfer_kind} {from_name} -> {to_name}",
            "start": f"{transfer_start // 60:02d}:{transfer_start % 60:02d}",
            "end": f"{transfer_end // 60:02d}:{transfer_end % 60:02d}",
            "duration_h": transfer_h,
            "cost_estimated": round(max(0.0, transfer_cost), 2),
            "meta": {
                "city": _normalize_text(day_plan.get("city")),
                "from": from_name,
                "to": to_name,
                "reason": "adjacent_event_transfer",
                "duration_source": "estimated_commute_capped_by_timeline_gap",
                "scheduled_gap_minutes": max(0, next_start - current_end),
                "airport_arrival_buffer_minutes": AIRPORT_ARRIVAL_BUFFER_MINUTES if next_type == "flight" else None,
            },
        }
        inserts.append((index + 1, transfer_event))

    if not inserts:
        return

    offset = 0
    for idx, transfer_event in inserts:
        events.insert(idx + offset, transfer_event)
        if _normalize_text(transfer_event.get("type")) == "transfer":
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


def _transfer_endpoint_key(value: Any) -> str:
    text = _normalize_text(value).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _transfer_endpoint_equal(actual: Any, expected: Any) -> bool:
    actual_key = _transfer_endpoint_key(actual)
    expected_key = _transfer_endpoint_key(expected)
    return bool(actual_key and expected_key and actual_key == expected_key)


def _neighbor_location_before(events: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    for prev_idx in range(index - 1, -1, -1):
        candidate = events[prev_idx]
        if _normalize_text(candidate.get("type")) == "transfer":
            continue
        if _is_location_event(candidate):
            return candidate
    return None


def _neighbor_location_after(events: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    for next_idx in range(index + 1, len(events)):
        candidate = events[next_idx]
        if _normalize_text(candidate.get("type")) == "transfer":
            continue
        if _is_location_event(candidate):
            return candidate
    return None


def _transfer_endpoint_field_matches(
    meta: Dict[str, Any],
    key: str,
    expected: Optional[str],
    required: bool,
) -> bool:
    actual = _normalize_text(meta.get(key)) if isinstance(meta, dict) else ""
    if not expected:
        return not required
    if actual:
        return _transfer_endpoint_equal(actual, expected)
    return not required


def _transfer_matches_neighbor_context(
    event: Dict[str, Any],
    reason: str,
    prev_location: Optional[Dict[str, Any]],
    next_location: Optional[Dict[str, Any]],
) -> bool:
    meta = event.get("meta") or {}
    if not isinstance(meta, dict):
        return False

    start = _event_start_minutes(event)
    end = _event_end_minutes(event)
    if start is None or end is None or end < start:
        return False

    prev_end = _event_end_minutes(prev_location) if prev_location is not None else None
    next_start = _event_start_minutes(next_location) if next_location is not None else None
    if prev_end is not None and start < prev_end:
        return False
    if next_start is not None and end > next_start:
        return False

    expected_from = (
        _transfer_endpoint_name(prev_location, as_destination=False)
        if prev_location is not None
        else None
    )
    expected_to = (
        _transfer_endpoint_name(next_location, as_destination=True)
        if next_location is not None
        else None
    )

    if prev_location is not None and next_location is not None:
        return (
            _transfer_endpoint_field_matches(meta, "from", expected_from, required=True)
            and _transfer_endpoint_field_matches(meta, "to", expected_to, required=True)
        )

    if reason not in PROTECTED_TRANSFER_REASONS:
        return False

    if prev_location is None and next_location is None:
        return False

    return (
        _transfer_endpoint_field_matches(meta, "from", expected_from, required=False)
        and _transfer_endpoint_field_matches(meta, "to", expected_to, required=False)
    )


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

        prev_location = _neighbor_location_before(events, index)
        next_location = _neighbor_location_after(events, index)

        if not _transfer_matches_neighbor_context(event, reason, prev_location, next_location):
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
    day_schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start"))
    for order, index in enumerate(meal_indices):
        event = events[index]
        event_meta = event.get("meta") or {}
        fixed_meal_slot = _normalize_text(event_meta.get("meal_slot")) if isinstance(event_meta, dict) else ""
        is_locked_meal = bool(
            isinstance(event_meta, dict)
            and (
                event_meta.get("fixed_meal_slot")
                or event_meta.get("required_meal_added")
                or event_meta.get("late_arrival_added")
            )
        )
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
        fixed_slot_index = next((i for i, item in enumerate(target_slots) if item[2] == fixed_meal_slot), None)
        target_index = fixed_slot_index if fixed_slot_index is not None else order
        if target_index < len(target_slots):
            slot_start = _event_minutes_from_hhmm(target_slots[target_index][0]) or 0
            slot_end = _event_minutes_from_hhmm(target_slots[target_index][1]) or (slot_start + 60)
            slot = target_slots[target_index][2]
            if (
                not is_locked_meal
                and day_schedule_start is not None
                and slot_end <= day_schedule_start
            ):
                meta = event.get("meta") or {}
                if isinstance(meta, dict):
                    meta["skipped_reason"] = "before_available_day_start"
                    meta["meal_slot"] = slot
                else:
                    event["meta"] = {
                        "meal_slot": slot,
                        "skipped_reason": "before_available_day_start",
                    }
                remove_indices.append(index)
                continue
            if slot == "breakfast":
                dur_minutes = min(dur_minutes, int(round(BREAKFAST_HOURS * 60)))
            slot_conflicts_with_flight = any(
                _interval_overlaps(slot_start, slot_end, flight_start, flight_end)
                for flight_start, flight_end in flight_blocks
            )
            if slot_conflicts_with_flight and not is_locked_meal:
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
            if slot == "breakfast":
                linked_transfer = None
                linked_transfer_duration = 0
                previous_event_end = day_schedule_start if day_schedule_start is not None else slot_start
                for block_start, block_end in fixed_blocks:
                    if block_end <= slot_end:
                        previous_event_end = max(previous_event_end, block_end)
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
                    earliest_start=max(
                        slot_start,
                        _event_start_minutes(event) or slot_start,
                    ) if is_locked_meal else slot_start,
                    duration_minutes=dur_minutes,
                    window_start=slot_start,
                    window_end=max(slot_end, (_event_end_minutes(event) or slot_end)) if is_locked_meal else slot_end,
                    blocked_ranges=blocked_ranges,
                )
            if candidate_start is None:
                if not is_locked_meal:
                    continue
                current_start = _event_start_minutes(event)
                if current_start is None:
                    continue
                candidate_start = current_start
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
    skipped_meal_names = {
        _normalize_text(event.get("name"))
        for event in day_plan.get("events", []) or []
        if _normalize_text(event.get("type")) == "meal"
        and isinstance(event.get("meta"), dict)
        and _normalize_text((event.get("meta") or {}).get("skipped_reason"))
    }
    if skipped_meal_names:
        kept_events: List[Dict[str, Any]] = []
        for event in day_plan.get("events", []) or []:
            event_type = _normalize_text(event.get("type"))
            event_name = _normalize_text(event.get("name"))
            meta = event.get("meta") or {}
            if event_type == "meal" and event_name in skipped_meal_names:
                continue
            if (
                event_type == "transfer"
                and isinstance(meta, dict)
                and (
                    _normalize_text(meta.get("to")) in skipped_meal_names
                    or _normalize_text(meta.get("from")) in skipped_meal_names
                )
            ):
                continue
            kept_events.append(event)
        day_plan["events"] = kept_events
    _sort_and_recompute_day(day_plan)


def _remove_events_before_schedule_start(day_plan: Dict[str, Any]) -> None:
    schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start"))
    if schedule_start is None or schedule_start <= 0:
        return
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    keep_transfer_reasons = {"arrival_to_hotel", "hotel_checkin_navigation"}
    removed: List[str] = []
    kept: List[Dict[str, Any]] = []
    for event in events:
        event_type = _normalize_text(event.get("type"))
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            kept.append(event)
            continue
        if event_type in {"flight", "drive", "checkin", "checkout"}:
            kept.append(event)
            continue
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if event_type == "transfer" and reason in keep_transfer_reasons and end <= schedule_start:
            kept.append(event)
            continue
        if event_type in {"poi", "meal", "transfer"} and start < schedule_start:
            removed.append(_normalize_text(event.get("name")) or event_type)
            continue
        kept.append(event)

    if removed:
        day_plan["events"] = kept
        day_plan.setdefault("warnings", []).append(
            "Removed activities scheduled before the traveler is available after arrival: "
            + ", ".join(removed)
        )
        _sort_and_recompute_day(day_plan)


def _remove_late_regular_pois(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    allowed_late_terms = {
        "night",
        "show",
        "theatre",
        "theater",
        "concert",
        "bar",
        "pub",
        "club",
        "whisky",
        "whiskey",
    }
    remove_indices: List[int] = []
    for index, event in enumerate(events):
        if _normalize_text(event.get("type")) != "poi":
            continue
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            continue
        if start < EVENING_RETURN_TARGET_MINUTES and end <= EVENING_RETURN_TARGET_MINUTES + 30:
            continue
        name = _normalize_text(event.get("name")).lower()
        if any(term in name for term in allowed_late_terms):
            continue
        remove_indices.append(index)

    removed: List[str] = []
    for index in sorted(remove_indices, reverse=True):
        removed.extend(_remove_event_at(day_plan, index))
    if removed:
        day_plan.setdefault("warnings", []).append(
            "Removed regular sightseeing scheduled too late in the evening: "
            + ", ".join(name for name in removed if name)
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
    checkin_meta = checkin_event.get("meta") or {}
    if isinstance(checkin_meta, dict) and _normalize_text(checkin_meta.get("reason")) in {"hotel_switch_checkin", "intercity_arrival_checkin"}:
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
    airport_arrival = _event_minutes_from_hhmm(day_plan.get("airport_arrival_time"))
    if airport_arrival is not None:
        schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start")) or airport_arrival
        desired_start = max(schedule_start, airport_arrival)
        desired_end = desired_start + HOTEL_PROCESS_MINUTES
        if first_start < desired_end:
            delta = desired_end - first_start
            for event in events:
                if event is checkin_event:
                    continue
                event_type = _normalize_text(event.get("type"))
                if event_type not in {"poi", "meal", "transfer"}:
                    continue
                start = _event_start_minutes(event)
                end = _event_end_minutes(event)
                if start is None or end is None or start < first_start:
                    continue
                event["start"] = f"{(start + delta) // 60:02d}:{(start + delta) % 60:02d}"
                event["end"] = f"{(end + delta) // 60:02d}:{(end + delta) % 60:02d}"
        checkin_event["start"] = f"{desired_start // 60:02d}:{desired_start % 60:02d}"
        checkin_event["end"] = f"{desired_end // 60:02d}:{desired_end % 60:02d}"
        checkin_event["duration_h"] = round(HOTEL_PROCESS_MINUTES / 60.0, 2)
        _sort_and_recompute_day(day_plan)
        return

    desired_end = first_start
    schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start")) or 0
    desired_start = max(schedule_start, desired_end - HOTEL_PROCESS_MINUTES)
    desired_end = desired_start + HOTEL_PROCESS_MINUTES
    if arrival_end is not None:
        # Flight-aware check-in time is already set in relocation logic; avoid forcing impossible overlap edits.
        return

    checkin_event["start"] = f"{desired_start // 60:02d}:{desired_start % 60:02d}"
    checkin_event["end"] = f"{desired_end // 60:02d}:{desired_end % 60:02d}"
    checkin_event["duration_h"] = round(max(0.25, (desired_end - desired_start) / 60.0), 2)
    _sort_and_recompute_day(day_plan)


def _enforce_departure_checkout_to_airport_order(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    inbound_flight = next(
        (
            event
            for event in events
            if _normalize_text(event.get("type")) == "flight"
            and isinstance(event.get("meta"), dict)
            and _normalize_text((event.get("meta") or {}).get("leg_type")) == "inbound"
        ),
        None,
    )
    checkout = next((event for event in events if _normalize_text(event.get("type")) == "checkout"), None)
    if inbound_flight is None:
        return
    if checkout is None and day_plan.get("hotel"):
        flight_start_for_checkout = _event_start_minutes(inbound_flight)
        if flight_start_for_checkout is None:
            return
        airport_deadline_for_checkout = max(0, flight_start_for_checkout - AIRPORT_ARRIVAL_BUFFER_MINUTES)
        airport_transfer_minutes_for_checkout = _airport_transfer_minutes_for_flight(inbound_flight)
        checkout_end_for_checkout = max(0, airport_deadline_for_checkout - airport_transfer_minutes_for_checkout)
        checkout_start_for_checkout = max(0, checkout_end_for_checkout - HOTEL_PROCESS_MINUTES)
        hotel = day_plan.get("hotel") or {}
        checkout = {
            "type": "checkout",
            "name": f"Check-out { _normalize_text(hotel.get('name')) or 'Hotel' }",
            "start": f"{checkout_start_for_checkout // 60:02d}:{checkout_start_for_checkout % 60:02d}",
            "end": f"{checkout_end_for_checkout // 60:02d}:{checkout_end_for_checkout % 60:02d}",
            "duration_h": round(HOTEL_PROCESS_MINUTES / 60.0, 2),
            "cost_estimated": 0.0,
            "meta": {
                "city": _normalize_text(day_plan.get("city")),
                "hotel_name": _normalize_text(hotel.get("name")),
                "lat": _safe_float(hotel.get("lat"), None),
                "lng": _safe_float(hotel.get("lng"), None),
            },
        }
        events.append(checkout)
        _sort_and_recompute_day(day_plan)
        events = day_plan.get("events", [])
        if not isinstance(events, list):
            return

    checkout_end = _event_end_minutes(checkout)
    checkout_start = _event_start_minutes(checkout)
    flight_start = _event_start_minutes(inbound_flight)
    if checkout_end is None or checkout_start is None or flight_start is None:
        return
    airport_deadline = max(0, flight_start - AIRPORT_ARRIVAL_BUFFER_MINUTES)
    airport_transfer_minutes = _airport_transfer_minutes_for_flight(inbound_flight)
    desired_checkout_end = max(0, airport_deadline - airport_transfer_minutes)
    desired_checkout_start = max(0, desired_checkout_end - HOTEL_PROCESS_MINUTES)
    if checkout_end != desired_checkout_end:
        checkout["start"] = f"{desired_checkout_start // 60:02d}:{desired_checkout_start % 60:02d}"
        checkout["end"] = f"{desired_checkout_end // 60:02d}:{desired_checkout_end % 60:02d}"
        checkout["duration_h"] = round(HOTEL_PROCESS_MINUTES / 60.0, 2)
        checkout_end = desired_checkout_end
        _sort_and_recompute_day(day_plan)
        events = day_plan.get("events", [])
        if not isinstance(events, list):
            return

    removed: List[str] = []
    kept: List[Dict[str, Any]] = []
    for event in events:
        if event is checkout or event is inbound_flight:
            kept.append(event)
            continue
        event_type = _normalize_text(event.get("type"))
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if (
            event_type == "checkin"
            and end is not None
            and end <= checkout_start
            and not any(
                _normalize_text(candidate.get("type")) == "flight"
                and isinstance(candidate.get("meta"), dict)
                and _normalize_text((candidate.get("meta") or {}).get("leg_type")) == "outbound"
                for candidate in events
            )
        ):
            removed.append(_normalize_text(event.get("name")) or "checkin")
            continue
        if (
            event_type == "checkin"
            and start is not None
            and end is not None
            and start >= checkout_end
            and end <= flight_start
        ):
            removed.append(_normalize_text(event.get("name")) or "checkin")
            continue
        if (
            event_type in {"poi", "meal"}
            and start is not None
            and end is not None
            and end > checkout_start
            and start < flight_start
        ):
            removed.append(_normalize_text(event.get("name")) or event_type)
            continue
        if event_type == "transfer":
            meta = event.get("meta") or {}
            reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
            if "airport" in _normalize_text(event.get("name")).lower() or (
                isinstance(meta, dict) and "airport" in _normalize_text(meta.get("to")).lower()
            ):
                kept.append(event)
                continue
            if reason in {"adjacent_event_transfer", "hotel_checkin_navigation", "arrival_to_hotel", "hotel_day_end"} and start is not None and end is not None and end > checkout_start and start < flight_start:
                removed.append(_normalize_text(event.get("name")) or "transfer")
                continue
        kept.append(event)

    if removed:
        day_plan["events"] = kept
        day_plan.setdefault("warnings", []).append(
            "Removed post-checkout activities before airport departure: " + ", ".join(removed)
        )
        _sort_and_recompute_day(day_plan)


def _cap_evening_activity_return_time(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return
    changed = False
    for event in events:
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if reason != "evening_after_dinner":
            continue
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            continue
        target_end = EVENING_RETURN_TARGET_MINUTES
        if target_end <= start:
            continue
        if end != target_end:
            event["end"] = f"{target_end // 60:02d}:{target_end % 60:02d}"
            event["duration_h"] = round((target_end - start) / 60.0, 2)
            changed = True
    if changed:
        _sort_and_recompute_day(day_plan)
        events = day_plan.get("events", [])
    for index, event in enumerate(events):
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if reason != "hotel_day_end":
            continue
        prev = events[index - 1] if index > 0 else None
        prev_meta = prev.get("meta") if isinstance(prev, dict) else {}
        if not isinstance(prev_meta, dict) or _normalize_text(prev_meta.get("reason")) != "evening_after_dinner":
            continue
        start = _event_end_minutes(prev)
        if start is None:
            continue
        end = start + HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES
        event["start"] = f"{start // 60:02d}:{start % 60:02d}"
        event["end"] = f"{end // 60:02d}:{end % 60:02d}"
        event["duration_h"] = round(HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES / 60.0, 2)
        changed = True
    if changed:
        _sort_and_recompute_day(day_plan)


def _realign_evening_activity_transfers(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return
    _sort_and_recompute_day(day_plan)
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    changed = False
    for index, event in enumerate(events):
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if _normalize_text(event.get("type")) != "transfer" or reason != "evening_activity_transfer":
            continue

        next_poi_index = None
        for next_index in range(index + 1, len(events)):
            next_event = events[next_index]
            next_meta = next_event.get("meta") or {}
            if (
                _normalize_text(next_event.get("type")) == "poi"
                and isinstance(next_meta, dict)
                and _normalize_text(next_meta.get("reason")) == "evening_after_dinner"
            ):
                next_poi_index = next_index
                break
        if next_poi_index is None:
            continue

        dinner = None
        for prev_index in range(index - 1, -1, -1):
            candidate = events[prev_index]
            candidate_meta = candidate.get("meta") or {}
            if (
                _normalize_text(candidate.get("type")) == "meal"
                and isinstance(candidate_meta, dict)
                and _normalize_text(candidate_meta.get("meal_slot")) == "dinner"
            ):
                dinner = candidate
                break
        if dinner is None:
            continue

        dinner_end = _event_end_minutes(dinner)
        if dinner_end is None:
            continue
        old_start = _event_start_minutes(event)
        old_end = _event_end_minutes(event)
        transfer_minutes = HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES
        if old_start is not None and old_end is not None and old_end > old_start:
            transfer_minutes = max(1, old_end - old_start)
        transfer_start = dinner_end
        transfer_end = min(EVENING_RETURN_TARGET_MINUTES - 30, transfer_start + transfer_minutes)
        if transfer_end <= transfer_start:
            continue

        event["start"] = f"{transfer_start // 60:02d}:{transfer_start % 60:02d}"
        event["end"] = f"{transfer_end // 60:02d}:{transfer_end % 60:02d}"
        event["duration_h"] = round((transfer_end - transfer_start) / 60.0, 2)

        poi = events[next_poi_index]
        poi_end = max(transfer_end + 30, EVENING_RETURN_TARGET_MINUTES)
        poi_end = min(poi_end, EVENING_RETURN_TARGET_MINUTES)
        poi["start"] = f"{transfer_end // 60:02d}:{transfer_end % 60:02d}"
        poi["end"] = f"{poi_end // 60:02d}:{poi_end % 60:02d}"
        poi["duration_h"] = round(max(0, poi_end - transfer_end) / 60.0, 2)
        changed = True

    if changed:
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


def _enforce_hotel_switch_morning_order(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    switch_transfer = next(
        (
            event
            for event in events
            if _normalize_text(event.get("type")) == "transfer"
            and _normalize_text((event.get("meta") or {}).get("reason")) == "hotel_switch"
        ),
        None,
    )
    if not switch_transfer:
        return

    checkout = None
    checkin = None
    switch_to = _normalize_text((switch_transfer.get("meta") or {}).get("to"))
    for event in events:
        event_type = _normalize_text(event.get("type"))
        if event_type == "checkout" and checkout is None:
            checkout = event
        if event_type == "checkin" and switch_to and switch_to in _event_node_name(event):
            checkin = event

    checkout_start = _event_start_minutes(checkout) if checkout else None
    checkin_end = _event_end_minutes(checkin) if checkin else None
    if checkout_start is None or checkin_end is None:
        return

    switch_breakfast = None
    for event in events:
        if _normalize_text(event.get("type")) != "meal":
            continue
        meta = event.get("meta") or {}
        meal_slot = _normalize_text(meta.get("meal_slot")) if isinstance(meta, dict) else ""
        end = _event_end_minutes(event)
        if meal_slot == "breakfast" and end is not None and end <= checkout_start:
            switch_breakfast = event
            break

    if switch_breakfast is not None and checkout is not None and checkin is not None:
        breakfast_distance_from_old = _distance_between_meta_km(
            _event_place_meta(switch_breakfast),
            _event_place_meta(checkout),
        )
        if breakfast_distance_from_old is None or breakfast_distance_from_old > MEAL_ANCHOR_RADIUS_KM:
            breakfast_start = _event_start_minutes(switch_breakfast)
            if breakfast_start is not None:
                checkout_start_new = max(6 * 60, breakfast_start - HOTEL_PROCESS_MINUTES - HOTEL_SWITCH_TRANSFER_MINUTES - HOTEL_PROCESS_MINUTES)
                checkout_end_new = checkout_start_new + HOTEL_PROCESS_MINUTES
                transfer_start_new = checkout_end_new
                transfer_end_new = transfer_start_new + HOTEL_SWITCH_TRANSFER_MINUTES
                checkin_start_new = transfer_end_new
                checkin_end_new = checkin_start_new + HOTEL_PROCESS_MINUTES
                checkout["start"] = f"{checkout_start_new // 60:02d}:{checkout_start_new % 60:02d}"
                checkout["end"] = f"{checkout_end_new // 60:02d}:{checkout_end_new % 60:02d}"
                switch_transfer["start"] = f"{transfer_start_new // 60:02d}:{transfer_start_new % 60:02d}"
                switch_transfer["end"] = f"{transfer_end_new // 60:02d}:{transfer_end_new % 60:02d}"
                checkin["start"] = f"{checkin_start_new // 60:02d}:{checkin_start_new % 60:02d}"
                checkin["end"] = f"{checkin_end_new // 60:02d}:{checkin_end_new % 60:02d}"
                day_plan.setdefault("warnings", []).append(
                    "Moved hotel switch before breakfast because the breakfast restaurant is closer to the next hotel."
                )
                _sort_and_recompute_day(day_plan)
                events = day_plan.get("events", [])
                checkout_start = _event_start_minutes(checkout)
                checkin_end = _event_end_minutes(checkin)
                if checkout_start is None or checkin_end is None:
                    return

    cursor = checkin_end
    changed = False
    for event in events:
        event_type = _normalize_text(event.get("type"))
        if event_type in {"flight", "drive", "checkin", "checkout"}:
            continue
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if event is switch_transfer or reason == "hotel_switch":
            continue
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None or start >= checkout_start:
            continue
        meal_slot = _normalize_text(meta.get("meal_slot")) if isinstance(meta, dict) else ""
        if event_type == "meal" and meal_slot == "breakfast" and end <= checkout_start:
            if isinstance(meta, dict):
                restaurant_name = _normalize_text(meta.get("restaurant_name"))
                if restaurant_name and restaurant_name.lower() not in {"breakfast", "breakfast near hotel"}:
                    event["name"] = restaurant_name
                meta["reason"] = "breakfast_before_hotel_switch"
            continue
        duration = max(1, end - start)
        event["start"] = f"{cursor // 60:02d}:{cursor % 60:02d}"
        cursor += duration
        event["end"] = f"{cursor // 60:02d}:{cursor % 60:02d}"
        changed = True

    if changed:
        day_plan.setdefault("warnings", []).append(
            "Moved pre-switch activities after hotel checkout, transfer, and check-in to avoid hotel teleporting."
        )
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
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if last_end is None:
            last_end = end
            continue
        is_protected_transfer = event_type == "transfer" and reason in PROTECTED_TRANSFER_REASONS
        can_shift_hotel_process = event_type in {"checkin", "checkout"}
        if start < last_end and (event_type not in fixed_types or can_shift_hotel_process) and not is_protected_transfer:
            delta = last_end - start
            new_start = start + delta
            new_end = end + delta
            event["start"] = f"{new_start // 60:02d}:{new_start % 60:02d}"
            event["end"] = f"{new_end // 60:02d}:{new_end % 60:02d}"
            start, end = new_start, new_end
        if end > (last_end or 0):
            last_end = end
    _sort_and_recompute_day(day_plan)


def _meal_slot_window_minutes(slot: str) -> Optional[Tuple[int, int]]:
    windows = {
        "breakfast": (NORMAL_BREAKFAST_START, NORMAL_BREAKFAST_END),
        "lunch": (NORMAL_LUNCH_START, NORMAL_LUNCH_END),
        "dinner": (NORMAL_DINNER_START, NORMAL_DINNER_END),
    }
    if slot not in windows:
        return None
    start = _event_minutes_from_hhmm(windows[slot][0])
    end = _event_minutes_from_hhmm(windows[slot][1])
    if start is None or end is None:
        return None
    return start, end


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

        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if event_type == "transfer" and reason in PROTECTED_TRANSFER_REASONS:
            cursor = max(cursor, end)
            continue

        if event_type in fixed_types:
            cursor = max(cursor, end)
            continue

        target_start = max(0, cursor)
        if event_type == "meal":
            meal_slot = _normalize_text(meta.get("meal_slot")) if isinstance(meta, dict) else ""
            window = _meal_slot_window_minutes(meal_slot)
            if window is not None:
                window_start, window_end = window
                latest_start = max(window_start, window_end - duration_minutes)
                target_start = max(target_start, window_start)
                if target_start + duration_minutes > window_end:
                    current_start = _event_start_minutes(event)
                    current_end = _event_end_minutes(event)
                    if (
                        current_start is not None
                        and current_end is not None
                        and current_start >= window_start
                        and current_end <= window_end
                    ):
                        target_start = current_start
                    else:
                        target_start = latest_start
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


def _place_daytime_pois_around_meals(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return
    _sort_and_recompute_day(day_plan)
    events = day_plan.get("events", [])

    meals: Dict[str, Dict[str, Any]] = {}
    daytime_pois: List[Dict[str, Any]] = []
    for event in events:
        event_type = _normalize_text(event.get("type"))
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if event_type == "meal" and isinstance(meta, dict):
            slot = _normalize_text(meta.get("meal_slot"))
            if slot in {"breakfast", "lunch", "dinner"}:
                meals[slot] = event
        elif event_type == "poi" and reason != "evening_after_dinner" and not _is_night_poi({"name": event.get("name"), **(meta if isinstance(meta, dict) else {})}):
            daytime_pois.append(event)

    if not daytime_pois:
        return

    breakfast_end = _event_end_minutes(meals.get("breakfast", {})) if meals.get("breakfast") else None
    lunch_start = _event_start_minutes(meals.get("lunch", {})) if meals.get("lunch") else None
    lunch_end = _event_end_minutes(meals.get("lunch", {})) if meals.get("lunch") else None
    dinner_start = _event_start_minutes(meals.get("dinner", {})) if meals.get("dinner") else None

    schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start")) or 8 * 60
    cursor = max(schedule_start, breakfast_end or schedule_start)
    changed = False
    remaining = list(daytime_pois)

    def place_next_before(limit: Optional[int]) -> bool:
        nonlocal cursor, changed
        if not remaining:
            return False
        poi = remaining[0]
        duration = max(1, int(round(_safe_float(poi.get("duration_h"), POI_VISIT_HOURS) * 60)))
        latest_end = limit - DEFAULT_INTER_EVENT_TRANSFER_MINUTES if limit is not None else None
        if latest_end is not None and cursor + duration > latest_end:
            return False
        old_start = _event_start_minutes(poi)
        old_end = _event_end_minutes(poi)
        new_start = cursor
        new_end = cursor + duration
        poi["start"] = f"{new_start // 60:02d}:{new_start % 60:02d}"
        poi["end"] = f"{new_end // 60:02d}:{new_end % 60:02d}"
        if old_start != new_start or old_end != new_end:
            changed = True
        cursor = new_end
        remaining.pop(0)
        return True

    # Always try to put at least one daytime attraction between breakfast and lunch.
    if lunch_start is not None:
        place_next_before(lunch_start)

    if lunch_end is not None:
        cursor = max(cursor, lunch_end)
    while remaining:
        if dinner_start is not None and cursor >= dinner_start - DEFAULT_INTER_EVENT_TRANSFER_MINUTES:
            break
        if not place_next_before(dinner_start):
            break

    if changed:
        day_plan.setdefault("warnings", []).append(
            "Reordered daytime attractions around meals so the morning and afternoon are not left idle."
        )
        _sort_and_recompute_day(day_plan)


def _enforce_unique_poi_visits(daily_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for day in daily_plans:
        seen: set = set()
        events = day.get("events", [])
        if not isinstance(events, list):
            continue
        to_remove: List[int] = []
        for index, event in enumerate(events):
            if not _poi_counts_for_reuse(event):
                continue
            event_keys = _poi_reuse_keys_from_event(event, city=_normalize_text(day.get("city")))
            if event_keys and (event_keys & seen):
                to_remove.append(index)
                continue
            seen.update(event_keys)

        if to_remove:
            removed_names: List[str] = []
            for index in reversed(to_remove):
                target_name = _normalize_text((day.get("events", [])[index] or {}).get("name"))
                _remove_event_at(day, index)
                if target_name:
                    removed_names.append(target_name)
            if removed_names:
                day.setdefault("warnings", []).append(
                    "Removed duplicate POI visits within the same day: "
                    + ", ".join(reversed(removed_names))
                )
        _sort_and_recompute_day(day)
    return daily_plans


def _collect_scheduled_poi_keys(
    daily_plans: List[Dict[str, Any]],
    exclude_generated_day: Optional[Dict[str, Any]] = None,
) -> set:
    seen: set = set()
    for day in daily_plans:
        city = _normalize_text(day.get("city"))
        for event in day.get("events", []) or []:
            if not _poi_counts_for_reuse(event):
                continue
            meta = event.get("meta") or {}
            reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
            if day is exclude_generated_day and reason == "idle_gap_fill":
                continue
            seen.update(_poi_reuse_keys_from_event(event, city=city))
    return seen


def _enforce_poi_three_day_cooldown(daily_plans: List[Dict[str, Any]], cooldown_days: int = 3) -> List[Dict[str, Any]]:
    last_seen: Dict[str, int] = {}

    def cooldown_key(day: Dict[str, Any], event: Dict[str, Any]) -> str:
        meta = event.get("meta") or {}
        name = _normalize_text(meta.get("poi_name")) if isinstance(meta, dict) else ""
        if not name:
            name = _normalize_text(event.get("name"))
        return _poi_name_reuse_key(name, city=_normalize_text(day.get("city")))

    for day_index, day in enumerate(daily_plans):
        events = day.get("events", [])
        if not isinstance(events, list):
            continue
        remove_indices: List[int] = []
        for event_index, event in enumerate(events):
            if not _poi_counts_for_reuse(event):
                continue
            meta = event.get("meta") or {}
            name = _normalize_text(meta.get("poi_name")) if isinstance(meta, dict) else ""
            if not name:
                name = _normalize_text(event.get("name"))
            key = cooldown_key(day, event)
            if not name:
                continue
            previous_day_index = last_seen.get(key)
            if previous_day_index is not None and (day_index - previous_day_index) < cooldown_days:
                remove_indices.append(event_index)
                continue
            last_seen[key] = day_index

        if remove_indices:
            removed: List[str] = []
            for event_index in sorted(remove_indices, reverse=True):
                removed.extend(_remove_event_at(day, event_index))
            if removed:
                day.setdefault("warnings", []).append(
                    f"Removed attractions repeated within {cooldown_days} days: " + ", ".join(name for name in removed if name)
                )
        _sort_and_recompute_day(day)
    return daily_plans


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
        schedule_start = _event_minutes_from_hhmm(day.get("schedule_start"))
        schedule_end = _event_minutes_from_hhmm(day.get("schedule_end"))
        if (
            _normalize_text(day.get("skip_city_schedule")).lower() == "true"
            or (schedule_start is not None and schedule_start >= 18 * 60)
            or (schedule_end is not None and schedule_end <= 12 * 60)
        ):
            continue
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
            poi_keys = _poi_reuse_keys_from_place(poi, city=city)
            if poi_keys and (poi_keys & used_poi_keys):
                continue
            if not _append_refill_poi(day, poi):
                continue
            if poi_keys:
                used_poi_keys.update(poi_keys)
            added_names.append(_normalize_text(poi.get("name")) or "POI")

        while _count_poi_events(day) < target_quota:
            fallback_index = _count_poi_events(day) + 1
            fallback_poi = {
                "name": _fallback_poi_name(day, fallback_index),
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


def _night_activity_score(poi: Dict[str, Any]) -> Tuple[int, str]:
    text_parts = [
        _normalize_text(poi.get("name")),
        _normalize_text(poi.get("address")),
        _normalize_text(poi.get("matched_query")),
        _normalize_text(poi.get("why_selected")),
    ]
    types = poi.get("types")
    if isinstance(types, list):
        text_parts.extend(_normalize_text(item) for item in types)
    text = " ".join(part for part in text_parts if part).lower()
    night_terms = {
        "night",
        "evening",
        "view",
        "observation",
        "tower",
        "sky",
        "river",
        "bridge",
        "market",
        "shopping",
        "district",
        "crossing",
        "夜景",
        "展望",
        "タワー",
        "スカイ",
    }
    score = 0 if any(term in text for term in night_terms) else 1
    return score, _normalize_text(poi.get("name")).lower()


def _choose_evening_poi(day_plan: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    city = _normalize_text(day_plan.get("city"))
    scheduled: set = set()
    for event in day_plan.get("events", []) or []:
        if _poi_counts_for_reuse(event):
            scheduled.update(_poi_reuse_keys_from_event(event, city=city))
    valid = [
        poi
        for poi in candidates
        if isinstance(poi, dict)
        and _normalize_text(poi.get("name"))
        and not (_poi_reuse_keys_from_place(poi, city=city) & scheduled)
    ]
    if valid:
        return sorted(valid, key=_night_activity_score)[0]

    hotel = day_plan.get("hotel") or {}
    city = _normalize_text(day_plan.get("city")) or "city"
    hotel_name = _normalize_text(hotel.get("name")) if isinstance(hotel, dict) else ""
    return {
        "name": f"{city} night view walk near {hotel_name or 'hotel'}",
        "estimated_total_cost": 0.0,
        "lat": _safe_float(hotel.get("lat"), None) if isinstance(hotel, dict) else None,
        "lng": _safe_float(hotel.get("lng"), None) if isinstance(hotel, dict) else None,
    }


def _ensure_evening_activity_after_dinner(
    day_plan: Dict[str, Any],
    city_candidates: Dict[str, List[Dict[str, Any]]],
) -> None:
    if not isinstance(day_plan.get("events"), list) or not day_plan.get("hotel"):
        return
    if any(
        _normalize_text(event.get("type")) == "flight"
        and (_event_start_minutes(event) or 0) < EVENING_RETURN_TARGET_MINUTES
        for event in day_plan.get("events", [])
    ):
        return

    events = day_plan.get("events", [])
    dinners = [
        event
        for event in events
        if _normalize_text(event.get("type")) == "meal"
        and isinstance(event.get("meta"), dict)
        and _normalize_text((event.get("meta") or {}).get("meal_slot")) == "dinner"
    ]
    if not dinners:
        return
    dinner = max(dinners, key=lambda event: _event_end_minutes(event) or 0)
    dinner_end = _event_end_minutes(dinner)
    if dinner_end is None or dinner_end >= EVENING_RETURN_TARGET_MINUTES - 30:
        return

    kept_events: List[Dict[str, Any]] = []
    removed_return = False
    for event in events:
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        start = _event_start_minutes(event)
        if reason == "hotel_day_end" and start is not None and start >= dinner_end:
            removed_return = True
            continue
        kept_events.append(event)
    if removed_return:
        day_plan["events"] = kept_events
        _sort_and_recompute_day(day_plan)
        events = day_plan.get("events", [])

    for event in events:
        if event is dinner:
            continue
        event_type = _normalize_text(event.get("type"))
        if event_type != "poi":
            continue
        start = _event_start_minutes(event)
        if start is not None and start >= dinner_end:
            return

    candidates = (city_candidates.get("pois", []) if isinstance(city_candidates, dict) else []) or []
    poi = _choose_evening_poi(day_plan, candidates)
    poi_name = _normalize_text(poi.get("name")) or "Evening viewpoint"
    poi_meta = {
        "city": _normalize_text(day_plan.get("city")),
        "poi_name": poi_name,
        "lat": _safe_float(poi.get("lat"), None),
        "lng": _safe_float(poi.get("lng"), None),
        "reason": "evening_after_dinner",
    }
    transfer_start = dinner_end
    transfer_end = min(EVENING_RETURN_TARGET_MINUTES - 30, dinner_end + HOTEL_DAY_BOUNDARY_TRANSFER_MINUTES)
    if transfer_end <= transfer_start:
        return
    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    is_international = bool(day_plan.get("local_transport_is_international"))
    commute_h = round((transfer_end - transfer_start) / 60.0, 2)
    transfer_cost = _estimate_local_transfer_cost(commute_h, local_tier, is_international=is_international)
    _append_fixed_time_event(
        day_plan=day_plan,
        event_type="transfer",
        name=f"Transfer {_event_node_name(dinner)} -> {poi_name}",
        start=f"{transfer_start // 60:02d}:{transfer_start % 60:02d}",
        end=f"{transfer_end // 60:02d}:{transfer_end % 60:02d}",
        duration_h=commute_h,
        cost_estimated=transfer_cost,
        meta={
            "city": _normalize_text(day_plan.get("city")),
            "from": _event_node_name(dinner),
            "to": poi_name,
            "reason": "evening_activity_transfer",
            "duration_source": "fixed_evening_activity_transfer",
        },
    )
    day_plan["transport_cost_estimated"] = round(
        _safe_float(day_plan.get("transport_cost_estimated"), 0.0) + transfer_cost,
        2,
    )
    poi_start = transfer_end
    poi_end = EVENING_RETURN_TARGET_MINUTES
    if poi_end - poi_start < 30:
        return
    _append_fixed_time_event(
        day_plan=day_plan,
        event_type="poi",
        name=poi_name,
        start=f"{poi_start // 60:02d}:{poi_start % 60:02d}",
        end=f"{poi_end // 60:02d}:{poi_end % 60:02d}",
        duration_h=round((poi_end - poi_start) / 60.0, 2),
        cost_estimated=_safe_float(poi.get("estimated_total_cost"), 0.0),
        meta=poi_meta,
    )
    day_plan.setdefault("warnings", []).append(
        "Added an evening activity after dinner before returning to the hotel."
    )
    _sort_and_recompute_day(day_plan)


def _ensure_evening_activities_after_dinner(
    daily_plans: List[Dict[str, Any]],
    city_candidates_map: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    for day in daily_plans:
        city = _normalize_text(day.get("city"))
        _ensure_evening_activity_after_dinner(
            day_plan=day,
            city_candidates=city_candidates_map.get(city, {"pois": []}),
        )
    return daily_plans


def _has_dinner(day_plan: Dict[str, Any]) -> bool:
    for event in day_plan.get("events", []) or []:
        if _normalize_text(event.get("type")) != "meal":
            continue
        meta = event.get("meta") or {}
        meal_slot = _normalize_text(meta.get("meal_slot")) if isinstance(meta, dict) else ""
        if meal_slot == "dinner":
            return True
        name = _normalize_text(event.get("name")).lower()
        start = _event_start_minutes(event)
        if "dinner" in name or (start is not None and start >= 17 * 60):
            return True
    return False


def _flight_covers_dinner_window(day_plan: Dict[str, Any]) -> bool:
    dinner_start = _event_minutes_from_hhmm(NORMAL_DINNER_START) or 18 * 60
    dinner_end = _event_minutes_from_hhmm(NORMAL_DINNER_END) or (20 * 60)
    extended_dinner_end = max(dinner_end, 20 * 60 + 30)
    for event in day_plan.get("events", []) or []:
        if _normalize_text(event.get("type")) != "flight":
            continue
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            continue
        if _interval_overlaps(start, end, dinner_start, extended_dinner_end):
            return True
    return False


def _flight_covers_meal_window(day_plan: Dict[str, Any], slot: str) -> bool:
    window = _meal_slot_window_minutes(slot)
    if window is None:
        return False
    meal_start, meal_end = window
    if slot == "dinner":
        meal_end = max(meal_end, 20 * 60 + 30)
    for event in day_plan.get("events", []) or []:
        if _normalize_text(event.get("type")) != "flight":
            continue
        start = _event_start_minutes(event)
        end = _event_end_minutes(event)
        if start is None or end is None:
            continue
        if _interval_overlaps(start, end, meal_start, meal_end):
            return True
    return False


def _meal_slot_index(slot: str) -> int:
    slot = _normalize_text(slot)
    if slot == "breakfast":
        return 0
    if slot == "lunch":
        return 1
    if slot == "dinner":
        return 2
    return 0


def _has_meal_slot(day_plan: Dict[str, Any], slot: str) -> bool:
    slot = _normalize_text(slot)
    window = _meal_slot_window_minutes(slot)
    for event in day_plan.get("events", []) or []:
        if _normalize_text(event.get("type")) != "meal":
            continue
        meta = event.get("meta") or {}
        meal_slot = _normalize_text(meta.get("meal_slot")) if isinstance(meta, dict) else ""
        if meal_slot == slot:
            return True
        name = _normalize_text(event.get("name")).lower()
        if slot and slot in name:
            return True
        if window is not None:
            start = _event_start_minutes(event)
            end = _event_end_minutes(event)
            if start is not None and end is not None and _interval_overlaps(start, end, window[0], window[1]):
                return True
    return False


def _required_meal_block_ranges(day_plan: Dict[str, Any]) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    for event in day_plan.get("events", []) or []:
        event_type = _normalize_text(event.get("type"))
        meta = event.get("meta") or {}
        reason = _normalize_text(meta.get("reason")) if isinstance(meta, dict) else ""
        if event_type in {"flight", "drive", "checkin", "checkout", "meal"}:
            pass
        elif event_type == "transfer" and reason in PROTECTED_TRANSFER_REASONS:
            pass
        else:
            continue
        start, end = _event_interval(event)
        if start is None or end is None:
            continue
        blocks.append((start, end))
    return blocks


def _meal_anchor_places_from_day(day_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    anchors = _meal_anchor_places(day_plan, [])
    for event in day_plan.get("events", []) or []:
        if not _is_location_event(event):
            continue
        if _normalize_text(event.get("type")) == "flight":
            continue
        meta = _event_place_meta(event)
        if meta:
            anchors.append(meta)
    return anchors


def _choose_required_meal_restaurant(
    day_plan: Dict[str, Any],
    city_candidates: Dict[str, List[Dict[str, Any]]],
    slot: str,
    current_day_index: Optional[int] = None,
    restaurant_last_used_day_index: Optional[Dict[str, int]] = None,
    restaurant_cooldown_days: int = 2,
    used_restaurant_keys: Optional[set] = None,
) -> Tuple[Dict[str, Any], str, str]:
    restaurants = [
        item
        for item in (city_candidates.get("restaurants", []) if isinstance(city_candidates, dict) else [])
        if isinstance(item, dict)
    ]
    if not restaurants:
        return {}, "", "none"
    meal_index = _meal_slot_index(slot)
    anchors = _meal_anchor_places_from_day(day_plan)
    preferred = _sort_restaurants_for_meal_slot(restaurants, anchors, meal_index)
    if preferred:
        return _select_restaurant_with_reuse_policy(
            restaurants=preferred,
            city=_normalize_text(day_plan.get("city")),
            used_today_restaurant_keys=_used_restaurant_keys_for_day(day_plan),
            restaurant_last_used_day_index=restaurant_last_used_day_index,
            current_day_index=current_day_index,
            restaurant_cooldown_days=restaurant_cooldown_days,
            used_restaurant_keys=used_restaurant_keys,
        )
    if slot == "breakfast":
        return {}, "", "none"
    sorted_restaurants = _sort_restaurants_by_meal_anchors(restaurants, anchors)
    return _select_restaurant_with_reuse_policy(
        restaurants=sorted_restaurants,
        city=_normalize_text(day_plan.get("city")),
        used_today_restaurant_keys=_used_restaurant_keys_for_day(day_plan),
        restaurant_last_used_day_index=restaurant_last_used_day_index,
        current_day_index=current_day_index,
        restaurant_cooldown_days=restaurant_cooldown_days,
        used_restaurant_keys=used_restaurant_keys,
    )


def _ensure_required_meals(
    day_plan: Dict[str, Any],
    city_candidates: Dict[str, List[Dict[str, Any]]],
    city_budget: Dict[str, float],
    current_day_index: Optional[int] = None,
    restaurant_last_used_day_index: Optional[Dict[str, int]] = None,
    restaurant_cooldown_days: int = 2,
    used_restaurant_keys: Optional[set] = None,
) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return
    if (
        _normalize_text(day_plan.get("skip_city_schedule")).lower() == "true"
        and _normalize_text(day_plan.get("overnight_arrival")).lower() != "true"
    ):
        return

    city_name = _normalize_text(day_plan.get("city"))
    fallback_cost = _fallback_meal_cost(city_budget)
    schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start")) or _event_minutes_from_hhmm(DEFAULT_DAY_START) or 8 * 60
    schedule_end = _event_minutes_from_hhmm(day_plan.get("schedule_end")) or _event_minutes_from_hhmm(DEFAULT_DAY_END) or 20 * 60 + 30
    hotel = day_plan.get("hotel") or {}
    hotel_lat = _safe_float(hotel.get("lat"), None) if isinstance(hotel, dict) else None
    hotel_lng = _safe_float(hotel.get("lng"), None) if isinstance(hotel, dict) else None

    for slot in ("breakfast", "lunch", "dinner"):
        if _has_meal_slot(day_plan, slot):
            continue
        if _flight_covers_meal_window(day_plan, slot):
            day_plan.setdefault("warnings", []).append(
                f"{slot.title()} treated as covered by the selected flight because it overlaps the normal meal window."
            )
            continue

        window = _meal_slot_window_minutes(slot)
        if window is None:
            continue
        window_start, window_end = window
        if window_end <= schedule_start or window_start >= schedule_end:
            continue

        durations = [int(round(BREAKFAST_HOURS * 60)), 30] if slot == "breakfast" else [int(round(MEAL_HOURS * 60)), 45]
        candidate_start = None
        duration_minutes = durations[0]
        blocked_ranges = _required_meal_block_ranges(day_plan)
        earliest_start = max(schedule_start, window_start)
        latest_window_end = min(schedule_end, window_end)
        for candidate_duration in durations:
            candidate_start = _resolve_meal_start_within_window(
                earliest_start=earliest_start,
                duration_minutes=candidate_duration,
                window_start=window_start,
                window_end=latest_window_end,
                blocked_ranges=blocked_ranges,
            )
            if candidate_start is not None:
                duration_minutes = candidate_duration
                break
        if candidate_start is None:
            day_plan.setdefault("warnings", []).append(
                f"Could not place {slot} inside the normal meal window without conflicting with fixed transport or hotel timing."
            )
            continue

        meal_index = _meal_slot_index(slot)
        restaurant, restaurant_key, reuse_reason = _choose_required_meal_restaurant(
            day_plan,
            city_candidates,
            slot,
            current_day_index=current_day_index,
            restaurant_last_used_day_index=restaurant_last_used_day_index,
            restaurant_cooldown_days=restaurant_cooldown_days,
            used_restaurant_keys=used_restaurant_keys,
        )
        meal_name = (
            _meal_display_name(restaurant, city_name, meal_index)
            if restaurant
            else _fallback_meal_name(day_plan, city_name, meal_index)
        )
        meal_cost = _meal_event_cost(restaurant, fallback_cost) if restaurant else fallback_cost
        meal_end = candidate_start + duration_minutes
        _append_fixed_time_event(
            day_plan=day_plan,
            event_type="meal",
            name=meal_name,
            start=f"{candidate_start // 60:02d}:{candidate_start % 60:02d}",
            end=f"{meal_end // 60:02d}:{meal_end % 60:02d}",
            duration_h=round(duration_minutes / 60.0, 2),
            cost_estimated=meal_cost,
            meta={
                "city": city_name,
                "restaurant_key": restaurant_key if restaurant else f"{city_name.lower()}|required-{slot}",
                "restaurant_name": _normalize_text(restaurant.get("name")) if restaurant else meal_name,
                "lat": _safe_float(restaurant.get("lat"), None) if restaurant else hotel_lat,
                "lng": _safe_float(restaurant.get("lng"), None) if restaurant else hotel_lng,
                "meal_slot": slot,
                "fixed_meal_slot": True,
                "required_meal_added": True,
            },
        )
        day_plan.setdefault("warnings", []).append(
            f"Added required {slot} because no selected flight covers that meal window."
        )
        if restaurant and restaurant_key:
            if isinstance(restaurant_last_used_day_index, dict) and current_day_index is not None:
                restaurant_last_used_day_index[restaurant_key] = current_day_index
            if isinstance(used_restaurant_keys, set):
                used_restaurant_keys.add(restaurant_key)
            _warn_restaurant_reuse_if_needed(day_plan, meal_name, slot, reuse_reason)
        _sort_and_recompute_day(day_plan)


def _remove_extra_meals_when_required_slots_present(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    duplicate_slot_indices: List[int] = []
    seen_slots: set = set()
    for index, event in enumerate(events):
        if _normalize_text(event.get("type")) != "meal" or not isinstance(event.get("meta"), dict):
            continue
        slot = _normalize_text((event.get("meta") or {}).get("meal_slot"))
        if slot not in {"breakfast", "lunch", "dinner"}:
            continue
        if slot in seen_slots:
            duplicate_slot_indices.append(index)
            continue
        seen_slots.add(slot)

    if duplicate_slot_indices:
        removed_duplicates: List[str] = []
        for index in sorted(duplicate_slot_indices, reverse=True):
            removed_duplicates.extend(_remove_event_at(day_plan, index))
        if removed_duplicates:
            day_plan.setdefault("warnings", []).append(
                "Removed duplicate required meal slots: "
                + ", ".join(name for name in removed_duplicates if name)
            )
        events = day_plan.get("events", [])
        if not isinstance(events, list):
            return

    slots = {
        _normalize_text((event.get("meta") or {}).get("meal_slot"))
        for event in events
        if _normalize_text(event.get("type")) == "meal" and isinstance(event.get("meta"), dict)
    }
    if not {"breakfast", "lunch", "dinner"}.issubset(slots):
        return

    extra_names = {
        _normalize_text(event.get("name"))
        for event in events
        if _normalize_text(event.get("type")) == "meal"
        and isinstance(event.get("meta"), dict)
        and _normalize_text((event.get("meta") or {}).get("meal_slot")) == "extra_meal"
    }
    if not extra_names:
        return

    removed: List[str] = []
    kept: List[Dict[str, Any]] = []
    for event in events:
        event_type = _normalize_text(event.get("type"))
        event_name = _normalize_text(event.get("name"))
        meta = event.get("meta") or {}
        if event_type == "meal" and event_name in extra_names:
            removed.append(event_name)
            continue
        if (
            event_type == "transfer"
            and isinstance(meta, dict)
            and (
                _normalize_text(meta.get("to")) in extra_names
                or _normalize_text(meta.get("from")) in extra_names
            )
        ):
            removed.append(event_name or "linked transfer")
            continue
        kept.append(event)
    day_plan["events"] = kept
    _sort_and_recompute_day(day_plan)
    if removed:
        day_plan.setdefault("warnings", []).append(
            "Removed extra meal after breakfast, lunch, and dinner were already scheduled: "
            + ", ".join(name for name in removed if name)
        )


def _remove_invalid_same_site_transfers(day_plan: Dict[str, Any]) -> None:
    events = day_plan.get("events", [])
    if not isinstance(events, list):
        return

    kept: List[Dict[str, Any]] = []
    removed: List[str] = []
    for event in events:
        if _normalize_text(event.get("type")) != "transfer":
            kept.append(event)
            continue
        meta = event.get("meta") or {}
        from_name = _normalize_text(meta.get("from")) if isinstance(meta, dict) else ""
        to_name = _normalize_text(meta.get("to")) if isinstance(meta, dict) else ""
        if from_name and to_name and from_name.lower() == to_name.lower():
            removed.append(_normalize_text(event.get("name")) or "same-site transfer")
            continue
        kept.append(event)

    if len(kept) != len(events):
        day_plan["events"] = kept
        if removed:
            day_plan.setdefault("warnings", []).append(
                "Removed same-site transfer rows: " + ", ".join(name for name in removed if name)
            )
        _sort_and_recompute_day(day_plan)


def _last_location_before(day_plan: Dict[str, Any], cutoff_minutes: int) -> Optional[Dict[str, Any]]:
    last = None
    for event in day_plan.get("events", []) or []:
        if not _is_location_event(event):
            continue
        end = _event_end_minutes(event)
        if end is None or end <= cutoff_minutes:
            last = event
    return last


def _append_fixed_transfer(
    day_plan: Dict[str, Any],
    from_event: Optional[Dict[str, Any]],
    to_name: str,
    to_meta: Dict[str, Any],
    transfer_end_minutes: int,
    reason: str,
) -> int:
    local_tier = _normalize_text(day_plan.get("local_transport_tier")) or "medium"
    is_international = bool(day_plan.get("local_transport_is_international"))
    from_name = "previous stop"
    commute_h = LATE_DINNER_TRANSFER_MINUTES / 60.0
    source = "fixed_late_arrival_transfer"
    distance_km = None
    if from_event is not None:
        from_name = _event_node_name(from_event)
        commute_h, source, distance_km = _estimate_commute_hours_between_meta(
            _event_place_meta(from_event),
            to_meta,
        )
    transfer_minutes = max(1, int(round(commute_h * 60)))
    transfer_start = max(0, transfer_end_minutes - transfer_minutes)
    commute_h = round((transfer_end_minutes - transfer_start) / 60.0, 2)
    cost = _estimate_local_transfer_cost(commute_h, local_tier, is_international=is_international)
    _append_fixed_time_event(
        day_plan=day_plan,
        event_type="transfer",
        name=f"Transfer {from_name} -> {to_name}",
        start=f"{transfer_start // 60:02d}:{transfer_start % 60:02d}",
        end=f"{transfer_end_minutes // 60:02d}:{transfer_end_minutes % 60:02d}",
        duration_h=commute_h,
        cost_estimated=cost,
        meta={
            "city": _normalize_text(day_plan.get("city")),
            "from": from_name,
            "to": to_name,
            "reason": reason,
            "duration_source": source,
            "distance_km": distance_km,
            "duration_minutes": int(round(commute_h * 60)),
        },
    )
    day_plan["transport_cost_estimated"] = round(
        _safe_float(day_plan.get("transport_cost_estimated"), 0.0) + cost,
        2,
    )
    return transfer_start


def _ensure_late_arrival_dinner(
    day_plan: Dict[str, Any],
    city_candidates: Dict[str, List[Dict[str, Any]]],
    city_budget: Dict[str, float],
    current_day_index: Optional[int] = None,
    restaurant_last_used_day_index: Optional[Dict[str, int]] = None,
    restaurant_cooldown_days: int = 2,
    used_restaurant_keys: Optional[set] = None,
) -> None:
    if _has_dinner(day_plan):
        return
    if _flight_covers_dinner_window(day_plan):
        day_plan.setdefault("warnings", []).append(
            "Dinner treated as covered by the selected flight because it overlaps the 18:00-20:30 dinner window."
        )
        return

    schedule_start = _event_minutes_from_hhmm(day_plan.get("schedule_start"))
    latest_fixed_end = None
    for event in day_plan.get("events", []) or []:
        if _normalize_text(event.get("type")) not in {"flight", "drive", "checkin"}:
            continue
        end = _event_end_minutes(event)
        if end is not None and (latest_fixed_end is None or end > latest_fixed_end):
            latest_fixed_end = end

    candidate_start = max(
        18 * 60,
        schedule_start if schedule_start is not None else 0,
        latest_fixed_end if latest_fixed_end is not None else 0,
    )
    if candidate_start < 18 * 60:
        return

    if candidate_start > 23 * 60:
        day_plan.setdefault("warnings", []).append(
            "Dinner could not be scheduled because arrival/check-in finished after 23:00."
        )
        return

    restaurants = [
        item
        for item in (city_candidates.get("restaurants", []) if isinstance(city_candidates, dict) else [])
        if isinstance(item, dict)
    ]
    city_name = _normalize_text(day_plan.get("city"))
    fallback_meal_cost = _fallback_meal_cost(city_budget)
    restaurants = _sort_restaurants_for_meal_slot(restaurants, _meal_anchor_places(day_plan, []), 2)
    restaurant, restaurant_key, reuse_reason = _select_restaurant_with_reuse_policy(
        restaurants=restaurants,
        city=city_name,
        used_today_restaurant_keys=_used_restaurant_keys_for_day(day_plan),
        restaurant_last_used_day_index=restaurant_last_used_day_index,
        current_day_index=current_day_index,
        restaurant_cooldown_days=restaurant_cooldown_days,
        used_restaurant_keys=used_restaurant_keys,
    )
    meal_name = _meal_display_name(restaurant, city_name, 2) if restaurant else _fallback_meal_name(day_plan, city_name, 2)
    meal_cost = _meal_event_cost(restaurant, fallback_meal_cost) if restaurant else fallback_meal_cost
    hotel = day_plan.get("hotel") or {}
    meal_meta = {
        "city": city_name,
        "restaurant_key": restaurant_key if restaurant else f"{city_name.lower()}|late-dinner",
        "restaurant_name": _normalize_text(restaurant.get("name")) if restaurant else meal_name,
        "lat": _safe_float(restaurant.get("lat"), None) if restaurant else (_safe_float(hotel.get("lat"), None) if isinstance(hotel, dict) else None),
        "lng": _safe_float(restaurant.get("lng"), None) if restaurant else (_safe_float(hotel.get("lng"), None) if isinstance(hotel, dict) else None),
        "meal_slot": "dinner",
        "fixed_meal_slot": True,
        "late_arrival_added": True,
    }

    duration_minutes = 60 if candidate_start <= 21 * 60 + 30 else 45
    meal_start = min(candidate_start, 23 * 60 - duration_minutes)
    previous = _last_location_before(day_plan, meal_start)
    transfer_start = meal_start
    if previous is not None:
        transfer_start = _append_fixed_transfer(
            day_plan=day_plan,
            from_event=previous,
            to_name=meal_name,
            to_meta=meal_meta,
            transfer_end_minutes=meal_start,
            reason="late_arrival_dinner_transfer",
        )
    meal_end = meal_start + duration_minutes
    _append_fixed_time_event(
        day_plan=day_plan,
        event_type="meal",
        name=meal_name,
        start=f"{meal_start // 60:02d}:{meal_start % 60:02d}",
        end=f"{meal_end // 60:02d}:{meal_end % 60:02d}",
        duration_h=round(duration_minutes / 60.0, 2),
        cost_estimated=meal_cost,
        meta=meal_meta,
    )
    day_plan.setdefault("warnings", []).append(
        "Added dinner after late arrival instead of leaving the arrival day without an evening meal."
    )
    if restaurant and restaurant_key:
        if isinstance(restaurant_last_used_day_index, dict) and current_day_index is not None:
            restaurant_last_used_day_index[restaurant_key] = current_day_index
        if isinstance(used_restaurant_keys, set):
            used_restaurant_keys.add(restaurant_key)
        _warn_restaurant_reuse_if_needed(day_plan, meal_name, "dinner", reuse_reason)
    if transfer_start < 0:
        day_plan.setdefault("warnings", []).append("Late-arrival dinner transfer was clipped to the start of the day.")
    _sort_and_recompute_day(day_plan)


def _ensure_late_arrival_dinners(
    daily_plans: List[Dict[str, Any]],
    city_candidates_map: Dict[str, Dict[str, List[Dict[str, Any]]]],
    city_budget_map: Dict[str, Dict[str, float]],
    restaurant_last_used_day_index: Optional[Dict[str, int]] = None,
    restaurant_cooldown_days: int = 2,
    used_restaurant_keys: Optional[set] = None,
) -> List[Dict[str, Any]]:
    for day_index, day in enumerate(daily_plans, start=1):
        schedule_start = _event_minutes_from_hhmm(day.get("schedule_start"))
        has_late_arrival_signal = schedule_start is not None and schedule_start >= 18 * 60
        if not has_late_arrival_signal:
            has_late_arrival_signal = any(
                _normalize_text(event.get("type")) in {"flight", "drive", "checkin"}
                and (_event_end_minutes(event) or 0) >= 18 * 60
                for event in day.get("events", []) or []
            )
        if not has_late_arrival_signal:
            has_late_arrival_signal = (
                _normalize_text(day.get("skip_city_schedule")).lower() == "true"
                and _normalize_text(day.get("overnight_arrival")).lower() == "true"
                and bool(day.get("hotel"))
                and schedule_start is not None
                and schedule_start <= (_event_minutes_from_hhmm(NORMAL_DINNER_END) or 20 * 60)
            )
        if not has_late_arrival_signal:
            continue
        city = _normalize_text(day.get("city"))
        _ensure_late_arrival_dinner(
            day_plan=day,
            city_candidates=city_candidates_map.get(city, {"restaurants": []}),
            city_budget=city_budget_map.get(city, {}),
            current_day_index=day_index,
            restaurant_last_used_day_index=restaurant_last_used_day_index,
            restaurant_cooldown_days=restaurant_cooldown_days,
            used_restaurant_keys=used_restaurant_keys,
        )
    return daily_plans


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
                warnings.append(f"Drive constraint violated: {d}h > {MAX_DRIVE_HOURS_PER_LEG:g}h.")

    return {
        "within_day_hours": within_day_hours,
        "max_drive_leg_le_4h": max_drive_ok,
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
                        and _normalize_text(meta.get("reason")) in PROTECTED_TRANSFER_REASONS
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


def _budget_spend_up_suppressed(normalized: Dict[str, Any]) -> bool:
    if not isinstance(normalized, dict):
        return False
    if bool(normalized.get("budget_maximize_experience")):
        return False
    strategy = _normalize_text(normalized.get("budget_strategy")).lower()
    status = _normalize_text(normalized.get("budget_status")).lower()
    cap_mode = normalized.get("budget_cap_mode")
    if isinstance(cap_mode, str):
        cap_mode = cap_mode.strip().lower() in {"true", "yes", "1", "cap"}
    return bool(cap_mode) or strategy == "tight" or status in {"over_budget", "infeasible"}


def _hotel_spend_up_groups(daily_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for day in daily_plans:
        if not isinstance(day, dict):
            continue
        hotel = day.get("hotel") or {}
        if not isinstance(hotel, dict):
            continue
        name = _normalize_text(hotel.get("name"))
        if not name:
            continue
        contribution = _hotel_cost_for_planned_night(day)
        if contribution <= 0:
            continue
        key = (
            _normalize_text(day.get("city")),
            name,
            _normalize_text(day.get("check_in")),
            _normalize_text(day.get("check_out")),
        )
        group = grouped.setdefault(
            key,
            {
                "days": [],
                "current_total": 0.0,
                "rollup_contribution": 0.0,
            },
        )
        group["days"].append(day)
        group["current_total"] = max(
            _safe_float(group.get("current_total"), 0.0),
            _safe_float(hotel.get("estimated_total_cost"), 0.0),
        )
        group["rollup_contribution"] = round(
            _safe_float(group.get("rollup_contribution"), 0.0) + contribution,
            2,
        )
    return list(grouped.values())


def _allocate_budget_spend_up(amount: float, weights: List[float]) -> List[float]:
    amount = round(max(0.0, _safe_float(amount, 0.0)), 2)
    if amount <= 0 or not weights:
        return []
    total_weight = sum(max(0.0, _safe_float(weight, 0.0)) for weight in weights)
    if total_weight <= 0:
        total_weight = float(len(weights))
        weights = [1.0 for _ in weights]

    allocated: List[float] = []
    remaining = amount
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            share = remaining
        else:
            share = round(amount * max(0.0, _safe_float(weight, 0.0)) / total_weight, 2)
            share = min(share, remaining)
        allocated.append(round(share, 2))
        remaining = round(remaining - share, 2)
    return allocated


def _stable_text_multiplier(text: Any, low: float = 0.85, high: float = 1.18) -> float:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return 1.0
    checksum = sum((idx + 1) * ord(ch) for idx, ch in enumerate(normalized))
    span = max(0.0, high - low)
    return round(low + ((checksum % 101) / 100.0) * span, 3)


def _spend_up_hotels(daily_plans: List[Dict[str, Any]], amount: float) -> float:
    groups = _hotel_spend_up_groups(daily_plans)
    if not groups or amount <= 0:
        return 0.0

    before_total = _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)
    allocations = _allocate_budget_spend_up(
        amount,
        [_safe_float(group.get("rollup_contribution"), 0.0) for group in groups],
    )
    for group, rollup_amount in zip(groups, allocations):
        if rollup_amount <= 0:
            continue
        current_total = _safe_float(group.get("current_total"), 0.0)
        rollup_contribution = _safe_float(group.get("rollup_contribution"), 0.0)
        if current_total <= 0 or rollup_contribution <= 0:
            continue
        effect_ratio = rollup_contribution / current_total
        if effect_ratio <= 0:
            continue
        hotel_total_increase = round(rollup_amount / effect_ratio, 2)
        upgraded_total = round(current_total + hotel_total_increase, 2)
        for day in group.get("days", []):
            hotel = day.get("hotel") or {}
            if not isinstance(hotel, dict):
                continue
            previous_total = _safe_float(hotel.get("estimated_total_cost"), 0.0)
            hotel["pre_budget_spend_up_estimated_total_cost"] = round(previous_total, 2)
            hotel["estimated_total_cost"] = upgraded_total
            hotel["budget_spend_up_upgrade"] = True

    after_total = _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)
    return round(max(0.0, after_total - before_total), 2)


def _spend_up_events(daily_plans: List[Dict[str, Any]], amount: float, event_type: str) -> float:
    if amount <= 0:
        return 0.0
    events: List[Dict[str, Any]] = []
    for day in daily_plans:
        for event in day.get("events", []) or []:
            if _normalize_text(event.get("type")) != event_type:
                continue
            cost = _safe_float(event.get("cost_estimated"), 0.0)
            if cost <= 0:
                continue
            events.append(event)
    if not events:
        return 0.0

    before_total = _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)
    weights: List[float] = []
    for event in events:
        base_cost = _safe_float(event.get("cost_estimated"), 0.0)
        if event_type == "meal":
            meta = event.get("meta") if isinstance(event.get("meta"), dict) else {}
            weight_key = "|".join([
                _normalize_text(event.get("name")),
                _normalize_text(meta.get("meal_slot")),
                _normalize_text(meta.get("restaurant_key")),
            ])
            base_cost *= _stable_text_multiplier(weight_key)
        weights.append(base_cost)
    allocations = _allocate_budget_spend_up(amount, weights)
    for event, add_amount in zip(events, allocations):
        if add_amount <= 0:
            continue
        old_cost = _safe_float(event.get("cost_estimated"), 0.0)
        event["pre_budget_spend_up_cost_estimated"] = round(old_cost, 2)
        event["cost_estimated"] = round(old_cost + add_amount, 2)
        meta = event.get("meta") if isinstance(event.get("meta"), dict) else {}
        meta["budget_spend_up_upgrade"] = True
        event["meta"] = meta

    after_total = _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)
    return round(max(0.0, after_total - before_total), 2)


def _spend_up_lifestyle_costs_for_budget(
    daily_plans: List[Dict[str, Any]],
    total_budget: float,
    normalized: Dict[str, Any],
) -> List[str]:
    if total_budget <= 0 or _budget_spend_up_suppressed(normalized):
        return []

    current_total = _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)
    target_total = round(total_budget * BUDGET_SPEND_UP_TARGET_RATIO, 2)
    if current_total <= 0 or current_total >= target_total:
        return []

    shortfall = round(target_total - current_total, 2)
    if shortfall < BUDGET_SPEND_UP_MIN_DELTA:
        return []

    upgraded = 0.0
    upgraded += _spend_up_hotels(daily_plans, shortfall)
    remaining = round(target_total - _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0), 2)
    if remaining >= BUDGET_SPEND_UP_MIN_DELTA:
        upgraded += _spend_up_events(daily_plans, remaining, "meal")
    remaining = round(target_total - _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0), 2)
    if remaining >= BUDGET_SPEND_UP_MIN_DELTA:
        upgraded += _spend_up_events(daily_plans, remaining, "poi")

    final_total = _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)
    if final_total > total_budget:
        _downgrade_lifestyle_costs_for_budget(daily_plans, total_budget)
        final_total = _safe_float(_build_cost_rollup(daily_plans).get("total_selected_estimated"), 0.0)

    upgraded = round(max(0.0, final_total - current_total), 2)
    if upgraded <= 0:
        return []
    return [
        (
            f"Budget utilization upgrade: raised hotel/meal/activity quality estimates by {upgraded} CNY "
            f"so the final plan uses about {round(final_total / total_budget * 100, 1)}% of the total budget without exceeding the cap."
        )
    ]


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
        max_drive_ok = constraints.get("max_drive_leg_le_4h", constraints.get("max_drive_leg_le_5h", True))
        if max_drive_ok is False:
            warnings.append(f"{day.get('date')} {day.get('city')}: contains drive segment over {MAX_DRIVE_HOURS_PER_LEG:g}h.")

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

    for day in daily_plans:
        hotel_reference += _hotel_cost_for_planned_night(day)

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


def _hotel_cost_for_planned_night(day: Dict[str, Any]) -> float:
    hotel = day.get("hotel") or {}
    if not isinstance(hotel, dict) or not _normalize_text(hotel.get("name")):
        return 0.0
    if _normalize_text(day.get("missing_required_outbound_transport")).lower() == "true":
        return 0.0

    nightly = _safe_float(hotel.get("estimated_nightly_per_room"), 0.0)
    if nightly <= 0:
        nightly = _safe_float(hotel.get("realtime_nightly_per_room"), 0.0)
    if nightly > 0:
        return nightly

    day_dt = _parse_date(day.get("date"))
    check_in = _parse_date(day.get("check_in"))
    check_out = _parse_date(day.get("check_out"))
    if day_dt and check_in and check_out:
        if not (check_in <= day_dt < check_out):
            return 0.0
        nights = max(1, (check_out - check_in).days)
    else:
        day_index = int(_safe_float(day.get("day_index_in_city"), 0))
        days_in_city = int(_safe_float(day.get("days_in_city"), 0))
        if day_index > 0 and days_in_city > 0 and day_index >= days_in_city:
            return 0.0
        nights = max(1, days_in_city - 1)

    return _safe_float(hotel.get("estimated_total_cost"), 0.0) / float(max(1, nights))


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
        max_drive_ok = constraints.get("max_drive_leg_le_4h", constraints.get("max_drive_leg_le_5h", True))
        if constraints.get("within_day_hours") and max_drive_ok:
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        numeric = _safe_float(value, default)
        return int(numeric)
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
