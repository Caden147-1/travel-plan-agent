from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from itinerary_cluster import collect_city_candidates, cluster_points_by_hotel_radius, select_primary_hotel_for_city, split_must_visit_and_regular_pois
from itinerary_rules import (
    append_event,
    build_day_template,
    estimate_flight_time_cost_hours,
    estimate_local_transport_cost_per_day,
    estimate_mid_distance_transport_cost,
    meal_hours,
    poi_visit_hours,
    trim_day_events_to_available_hours,
)
from planner_schema import DAY_AVAILABLE_HOURS, HOTEL_CLUSTER_MAX_ROUTE_KM, HOTEL_CLUSTER_RADIUS_KM, normalize_text, safe_float


def parse_date(value: Any) -> datetime | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except Exception:
        return None


def extract_daily_city_slots(city_date_ranges: Dict[str, Any]) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []

    if not isinstance(city_date_ranges, dict):
        return slots

    for city, info in city_date_ranges.items():
        if not isinstance(info, dict):
            continue
        check_in = normalize_text(info.get("check_in"))
        check_out = normalize_text(info.get("check_out"))

        if not check_in or not check_out:
            days = max(1, int(safe_float(info.get("days"), 1)))
            for i in range(days):
                slots.append({
                    "date": f"day_{i+1}",
                    "city": normalize_text(city),
                    "day_index_in_city": i + 1,
                    "days_in_city": days,
                    "check_in": check_in or None,
                    "check_out": check_out or None,
                })
            continue

        start_dt = parse_date(check_in)
        end_dt = parse_date(check_out)
        if not start_dt or not end_dt:
            continue

        span = (end_dt - start_dt).days
        if span <= 0:
            span = 1

        for i in range(span):
            day = start_dt + timedelta(days=i)
            slots.append({
                "date": day.strftime("%Y-%m-%d"),
                "city": normalize_text(city),
                "day_index_in_city": i + 1,
                "days_in_city": span,
                "check_in": check_in,
                "check_out": check_out,
            })

    slots.sort(key=lambda x: (normalize_text(x.get("date")), normalize_text(x.get("city"))))
    return slots


def build_city_budget_map(city_budget_plan: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    mapping: Dict[str, Dict[str, float]] = {}
    if not isinstance(city_budget_plan, list):
        return mapping

    for item in city_budget_plan:
        if not isinstance(item, dict):
            continue
        city = normalize_text(item.get("city"))
        if not city:
            continue
        mapping[city] = {
            "hotel_budget": safe_float(item.get("hotel_budget"), 0.0),
            "food_budget": safe_float(item.get("food_budget"), 0.0),
            "attractions_budget": safe_float(item.get("attractions_budget"), 0.0),
            "days": max(1.0, safe_float(item.get("days"), 1.0)),
        }

    return mapping


def schedule_city_day(day_plan: Dict[str, Any], city_candidates: Dict[str, List[Dict[str, Any]]], city_budget: Dict[str, float], must_visit_left: List[Dict[str, Any]]) -> Dict[str, Any]:
    pois = city_candidates.get("pois", []) or []
    restaurants = city_candidates.get("restaurants", []) or []

    must_now = list(must_visit_left)
    must_ids = {normalize_text(x.get("name")).lower() for x in must_now}
    regular = [p for p in pois if normalize_text(p.get("name")).lower() not in must_ids]

    poi_quota = 3
    meal_quota = 2

    selected_pois = must_now[:poi_quota]
    if len(selected_pois) < poi_quota:
        for p in regular:
            if len(selected_pois) >= poi_quota:
                break
            selected_pois.append(p)

    for poi in selected_pois:
        if safe_float(day_plan.get("used_hours"), 0.0) + poi_visit_hours() > DAY_AVAILABLE_HOURS:
            break
        meta = {
            "city": normalize_text(day_plan.get("city")),
            "poi_name": normalize_text(poi.get("name")),
        }
        mv = poi.get("_must_visit_name_norm")
        if mv:
            meta["must_visit_name"] = mv
        append_event(
            day_plan=day_plan,
            event_type="poi",
            name=normalize_text(poi.get("name")) or "POI",
            duration_h=poi_visit_hours(),
            cost_estimated=safe_float(poi.get("estimated_total_cost"), 0.0),
            meta=meta,
        )

    added_meals = 0
    for r in restaurants:
        if added_meals >= meal_quota:
            break
        if safe_float(day_plan.get("used_hours"), 0.0) + meal_hours() > DAY_AVAILABLE_HOURS:
            break
        append_event(
            day_plan=day_plan,
            event_type="meal",
            name=normalize_text(r.get("name")) or "Meal",
            duration_h=meal_hours(),
            cost_estimated=safe_float(r.get("estimated_total_cost"), 0.0),
            meta={"city": normalize_text(day_plan.get("city"))},
        )
        added_meals += 1

    while added_meals < meal_quota and safe_float(day_plan.get("used_hours"), 0.0) + meal_hours() <= DAY_AVAILABLE_HOURS:
        append_event(
            day_plan=day_plan,
            event_type="meal",
            name="Meal",
            duration_h=meal_hours(),
            cost_estimated=0.0,
            meta={"city": normalize_text(day_plan.get("city"))},
        )
        added_meals += 1

    days = max(1.0, safe_float(city_budget.get("days"), 1.0))
    day_food_budget = safe_float(city_budget.get("food_budget"), 0.0) / days
    day_attr_budget = safe_float(city_budget.get("attractions_budget"), 0.0) / days

    food_spend = 0.0
    attr_spend = 0.0
    for ev in day_plan.get("events", []) or []:
        t = normalize_text(ev.get("type"))
        if t == "meal":
            food_spend += safe_float(ev.get("cost_estimated"), 0.0)
        if t == "poi":
            attr_spend += safe_float(ev.get("cost_estimated"), 0.0)

    if day_food_budget > 0 and food_spend > day_food_budget * 1.2:
        day_plan.setdefault("warnings", []).append(
            f"Food daily spend warning: {round(food_spend,2)} exceeds daily food budget reference {round(day_food_budget,2)}."
        )

    if day_attr_budget > 0 and attr_spend > day_attr_budget * 1.2:
        day_plan.setdefault("warnings", []).append(
            f"Attractions daily spend warning: {round(attr_spend,2)} exceeds daily attractions budget reference {round(day_attr_budget,2)}."
        )

    return day_plan


def insert_checkin_checkout_events(daily_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not daily_plans:
        return daily_plans

    first = daily_plans[0]
    if first.get("hotel"):
        append_event(
            day_plan=first,
            event_type="checkin",
            name=f"Check-in {normalize_text((first.get('hotel') or {}).get('name')) or 'Hotel'}",
            duration_h=0.5,
            cost_estimated=0.0,
            meta={"city": normalize_text(first.get("city"))},
        )

    last = daily_plans[-1]
    if last.get("hotel"):
        append_event(
            day_plan=last,
            event_type="checkout",
            name=f"Check-out {normalize_text((last.get('hotel') or {}).get('name')) or 'Hotel'}",
            duration_h=0.5,
            cost_estimated=0.0,
            meta={"city": normalize_text(last.get("city"))},
        )

    for d in daily_plans:
        trim_day_events_to_available_hours(d)

    return daily_plans


def insert_flight_and_transfer_events(daily_plans: List[Dict[str, Any]], selected_all_flight_legs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not daily_plans or not isinstance(selected_all_flight_legs, list):
        return daily_plans

    by_date = {normalize_text(d.get("date")): d for d in daily_plans}

    for leg in selected_all_flight_legs:
        if not isinstance(leg, dict):
            continue
        dep_date = normalize_text(leg.get("departure_date"))
        day = by_date.get(dep_date)
        if not day:
            continue

        flights = leg.get("selected_flights", []) or []
        if flights:
            f = flights[0]
            dur = estimate_flight_time_cost_hours(f)
            cost = safe_float(f.get("price_total"), 0.0)
            append_event(
                day_plan=day,
                event_type="flight",
                name=f"Flight {normalize_text(leg.get('from_city'))} -> {normalize_text(leg.get('to_city'))}",
                duration_h=dur,
                cost_estimated=cost,
                meta={
                    "leg_type": normalize_text(leg.get("leg_type")),
                    "from_city": normalize_text(leg.get("from_city")),
                    "to_city": normalize_text(leg.get("to_city")),
                },
            )
            day["transport_cost_estimated"] = round(safe_float(day.get("transport_cost_estimated"), 0.0) + cost, 2)
        elif normalize_text(leg.get("leg_type")) == "intercity":
            cost = safe_float(leg.get("rental_car_cost_estimated"), 0.0)
            if cost <= 0:
                rental_car = leg.get("rental_car") or {}
                if isinstance(rental_car, dict):
                    allocated = rental_car.get("allocated_city_estimates") or []
                    if isinstance(allocated, list):
                        cost = sum(
                            safe_float(item.get("total_car_rental_cost"), 0.0)
                            for item in allocated
                            if isinstance(item, dict)
                        )
            if cost <= 0:
                cost = float(estimate_mid_distance_transport_cost("medium"))
            append_event(
                day_plan=day,
                event_type="drive",
                name=f"Drive {normalize_text(leg.get('from_city'))} -> {normalize_text(leg.get('to_city'))}",
                duration_h=3.0,
                cost_estimated=cost,
                meta={
                    "leg_type": "intercity",
                    "transport_mode": normalize_text(leg.get("transport_mode")) or "drive",
                    "from_city": normalize_text(leg.get("from_city")),
                    "to_city": normalize_text(leg.get("to_city")),
                    "rental_car": leg.get("rental_car"),
                },
            )
            day["transport_cost_estimated"] = round(safe_float(day.get("transport_cost_estimated"), 0.0) + cost, 2)

    for d in daily_plans:
        trim_day_events_to_available_hours(d)

    return daily_plans


def build_daily_plans(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    slots = extract_daily_city_slots(payload.get("city_date_ranges", {}) or {})
    if not slots:
        return [], []

    budget_map = build_city_budget_map(payload.get("city_budget_plan", []) or [])
    hotels = payload.get("selected_hotels", []) or []
    pois = payload.get("selected_pois", []) or []
    restaurants = payload.get("selected_restaurants", []) or []
    must_visit_names = payload.get("must_visit_attractions", []) or []

    unique_cities: List[str] = []
    for s in slots:
        c = normalize_text(s.get("city"))
        if c and c not in unique_cities:
            unique_cities.append(c)

    city_hotel_map: Dict[str, Dict[str, Any] | None] = {}
    city_candidates_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    city_must_map: Dict[str, List[Dict[str, Any]]] = {}

    for city in unique_cities:
        clusters = cluster_points_by_hotel_radius(
            hotels=hotels,
            pois=pois,
            restaurants=restaurants,
            city=city,
            radius_km=HOTEL_CLUSTER_RADIUS_KM,
            max_route_km=HOTEL_CLUSTER_MAX_ROUTE_KM,
        )
        city_hotel_map[city] = select_primary_hotel_for_city(city=city, hotel_clusters=clusters)

        candidates = collect_city_candidates(city, pois, restaurants)
        city_candidates_map[city] = candidates

        must_city, _ = split_must_visit_and_regular_pois(candidates.get("pois", []), must_visit_names)
        city_must_map[city] = list(must_city)

    daily_plans: List[Dict[str, Any]] = []

    for slot in slots:
        city = normalize_text(slot.get("city"))
        hotel = city_hotel_map.get(city)
        day_plan = build_day_template(slot, hotel)

        local_cost = float(estimate_local_transport_cost_per_day())
        append_event(
            day_plan=day_plan,
            event_type="local_transport",
            name=f"Local transport in {city}",
            duration_h=0.5,
            cost_estimated=local_cost,
            meta={"city": city},
        )
        day_plan["transport_cost_estimated"] = round(safe_float(day_plan.get("transport_cost_estimated"), 0.0) + local_cost, 2)

        day_plan = schedule_city_day(
            day_plan=day_plan,
            city_candidates=city_candidates_map.get(city, {"pois": [], "restaurants": []}),
            city_budget=budget_map.get(city, {}),
            must_visit_left=city_must_map.get(city, []),
        )

        trim_day_events_to_available_hours(day_plan)
        daily_plans.append(day_plan)

    insert_checkin_checkout_events(daily_plans)
    insert_flight_and_transfer_events(daily_plans, payload.get("selected_all_flight_legs", []) or [])

    return daily_plans, [normalize_text(x) for x in must_visit_names if normalize_text(x)]
