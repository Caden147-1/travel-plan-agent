from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

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
HEIGIT_API_KEY = os.getenv("HEIGIT_API_KEY", "")
HEIGIT_BASE_URL = os.getenv("HEIGIT_BASE_URL", "https://api.openrouteservice.org")
_CITY_COORD_CACHE: Dict[str, Dict[str, float]] = {}
_ROUTE_METRICS_CACHE: Dict[str, Dict[str, float]] = {}

PLACE_EVENT_TYPES = {"poi", "meal", "checkin", "checkout"}
FIXED_TIME_EVENT_TYPES = {"flight", "drive"}


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
                constraints.setdefault(arrival_date, {})["schedule_start"] = _add_minutes_to_time(arrival, 30) or arrival
        elif leg_type == "inbound":
            departure = _format_time_value(flight.get("departure_time"))
            if departure:
                day_constraints["schedule_end"] = departure
    return constraints


def run_itinerary_agent(router_output: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_agent3_payload(router_output)
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
        primary_hotel = _select_primary_hotel_for_city(city=city, hotel_clusters=city_hotel_clusters)
        city_hotel_map[city] = primary_hotel

        city_candidates = _collect_city_candidates(
            city=city,
            selected_pois=pois,
            selected_restaurants=restaurants,
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

    for day_index, slot in enumerate(daily_slots, start=1):
        city = _normalize_text(slot.get("city"))
        hotel = city_hotel_map.get(city)
        day_plan = _init_day_plan(slot=slot, hotel=hotel)
        day_plan.update(flight_day_constraints.get(_normalize_text(slot.get("date")), {}))

        if _normalize_text(day_plan.get("skip_city_local_transport")).lower() != "true":
            _append_event(
                day_plan=day_plan,
                event_type="local_transport",
                name=f"Local transport in {city}",
                duration_h=0.0,
                cost_estimated=float(_estimate_local_transport_cost_per_day()),
                meta={"city": city, "all_day_cost": True},
            )
            day_plan["transport_cost_estimated"] = float(day_plan.get("transport_cost_estimated", 0.0)) + float(_estimate_local_transport_cost_per_day())

        day_budget = city_budget_map.get(city, {})
        city_candidates = city_candidates_map.get(city, {"pois": [], "restaurants": []})
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

    total_budget = float(_safe_float(normalized.get("trip_summary", {}).get("budget", {}).get("total"), 0.0))
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
    if key in _ROUTE_METRICS_CACHE:
        cached = _ROUTE_METRICS_CACHE[key]
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
            _ROUTE_METRICS_CACHE[key] = {}
            return None
        data = resp.json() if resp.text else {}
        routes = data.get("routes", []) if isinstance(data, dict) else []
        if not routes:
            _ROUTE_METRICS_CACHE[key] = {}
            return None
        summary = routes[0].get("summary", {}) if isinstance(routes[0], dict) else {}
        duration_sec = _safe_float(summary.get("duration"), 0.0)
        distance_m = _safe_float(summary.get("distance"), 0.0)
        if duration_sec <= 0 or distance_m <= 0:
            _ROUTE_METRICS_CACHE[key] = {}
            return None
        result = {
            "duration_h": round(duration_sec / 3600.0, 2),
            "distance_km": round(distance_m / 1000.0, 2),
        }
        _ROUTE_METRICS_CACHE[key] = result
        return dict(result)
    except Exception:
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
    if city_key in _CITY_COORD_CACHE:
        cached = _CITY_COORD_CACHE[city_key]
        if cached:
            return cached
        return None
    if not HEIGIT_API_KEY:
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
            _CITY_COORD_CACHE[city_key] = {}
            return None
        data = resp.json() if resp.text else {}
        features = data.get("features", []) if isinstance(data, dict) else []
        if not features:
            _CITY_COORD_CACHE[city_key] = {}
            return None
        geometry = features[0].get("geometry", {}) if isinstance(features[0], dict) else {}
        coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
        if not isinstance(coords, list) or len(coords) < 2:
            _CITY_COORD_CACHE[city_key] = {}
            return None
        lng = _safe_float(coords[0], 0.0)
        lat = _safe_float(coords[1], 0.0)
        if lat == 0.0 and lng == 0.0:
            _CITY_COORD_CACHE[city_key] = {}
            return None
        result = {"lat": lat, "lng": lng}
        _CITY_COORD_CACHE[city_key] = result
        return result
    except Exception:
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


def _estimate_local_transport_cost_per_day() -> int:
    return int(LOCAL_TRANSPORT_PER_DAY)


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
        start = _max_time_value("15:00", _add_minutes_to_time(arrival_time, 30))
    else:
        start = "15:00"
    end = _add_minutes_to_time(start, 30) or "15:30"
    return start, end


def _hotel_checkout_window(departure_time: Any = None) -> Tuple[str, str]:
    departure = _parse_time_value(departure_time)
    noon = _parse_time_value("12:00")
    if departure and noon and departure < noon:
        end = (departure - timedelta(minutes=120)).strftime("%H:%M")
        if end < "06:00":
            end = "06:00"
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
    _append_event(
        day_plan=day_plan,
        event_type="transfer",
        name=f"Transfer {from_name} -> {to_name}",
        duration_h=commute_h,
        cost_estimated=0.0,
        meta={
            "city": _normalize_text(day_plan.get("city")),
            "from": from_name,
            "to": _normalize_text(to_name),
            "duration_source": source,
            "distance_km": distance_km,
        },
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


def _schedule_city_day(
    day_plan: Dict[str, Any],
    city_candidates: Dict[str, List[Dict[str, Any]]],
    city_budget: Dict[str, float],
    must_visit_left: List[Dict[str, Any]],
    current_day_index: int,
    restaurant_last_used_day_index: Dict[str, int],
    restaurant_cooldown_days: int = 2,
    used_poi_keys: Optional[set] = None,
) -> Dict[str, Any]:
    pois = city_candidates.get("pois", []) or []
    restaurants = city_candidates.get("restaurants", []) or []

    must_now = list(must_visit_left)
    must_ids = {_normalize_text(x.get("name")).lower() for x in must_now}
    regular = [p for p in pois if _normalize_text(p.get("name")).lower() not in must_ids]

    poi_quota = 3

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

    meal_quota = 2
    meal_added = 0
    used_today_restaurant_keys = set()
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
            "lat": _safe_float(r.get("lat"), None),
            "lng": _safe_float(r.get("lng"), None),
        }
        commute_h = 0.0
        previous = _last_place_event(day_plan)
        if previous is not None:
            commute_h, _, _ = _estimate_commute_hours_between_meta(_event_place_meta(previous), meta)
        if _safe_float(day_plan.get("used_hours"), 0.0) + commute_h + MEAL_HOURS > DAY_AVAILABLE_HOURS:
            break
        _append_commute_before_place(day_plan, _normalize_text(r.get("name")) or "Meal", meta)
        _append_event(
            day_plan=day_plan,
            event_type="meal",
            name=_normalize_text(r.get("name")) or "Meal",
            duration_h=MEAL_HOURS,
            cost_estimated=_safe_float(r.get("estimated_total_cost"), 0.0),
            meta=meta,
        )
        restaurant_last_used_day_index[restaurant_key] = current_day_index
        used_today_restaurant_keys.add(restaurant_key)
        meal_added += 1

    while meal_added < 2 and _safe_float(day_plan.get("used_hours"), 0.0) + MEAL_HOURS <= DAY_AVAILABLE_HOURS:
        meta = {"city": _normalize_text(day_plan.get("city"))}
        commute_h = 0.0
        previous = _last_place_event(day_plan)
        if previous is not None:
            commute_h, _, _ = _estimate_commute_hours_between_meta(_event_place_meta(previous), meta)
        if _safe_float(day_plan.get("used_hours"), 0.0) + commute_h + MEAL_HOURS > DAY_AVAILABLE_HOURS:
            break
        _append_commute_before_place(day_plan, "Meal", meta)
        _append_event(
            day_plan=day_plan,
            event_type="meal",
            name="Meal",
            duration_h=MEAL_HOURS,
            cost_estimated=0.0,
            meta=meta,
        )
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
        for i in range(len(events) - 1, -1, -1):
            if _normalize_text(events[i].get("type")) in {"poi", "meal"}:
                idx_to_remove = i
                break
        if idx_to_remove is None:
            idx_to_remove = len(events) - 1
        removed = events.pop(idx_to_remove)
        dur = _safe_float(removed.get("duration_h"), 0.0)
        day_plan["used_hours"] = round(max(0.0, _safe_float(day_plan.get("used_hours"), 0.0) - dur), 2)
        for neighbor_idx in sorted([idx_to_remove, idx_to_remove - 1], reverse=True):
            if 0 <= neighbor_idx < len(events) and _normalize_text(events[neighbor_idx].get("type")) == "transfer":
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
        _sort_and_recompute_day(day)
        _trim_day_to_available_hours(day)

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
