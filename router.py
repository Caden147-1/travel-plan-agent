from __future__ import annotations

import json
import copy
import math
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from travel_requirement_parser_llm_agent import (
    get_llm_backend_status,
    travel_requirement_parser_llm_agent,
)
from travel_time import get_today_date
from travel_budget_agent import (
    get_budget_agent_backend_status,
    travel_budget_agent,
    _estimate_food_cost,
    _estimate_hotel_cost,
    _estimate_transport_cost,
    _get_primary_profile,
    _get_season_factor,
    _get_weighted_cost_profile,
    _normalize_preference_levels,
)
from city_consumption_tier_agent import (
    estimate_mid_distance_car_rental,
    get_city_consumption_tier_backend_status,
)
from agent3_real import Agent3Selection
from itinerary_agent import run_itinerary_agent
from travel_report_generator import write_travel_report


ROUTER_PROCESS_STARTED_AT = datetime.now().isoformat(timespec="seconds")
ROUTER_CODE_VERSION = datetime.fromtimestamp(
    Path(__file__).resolve().stat().st_mtime
).isoformat(timespec="seconds")

REFINEMENT_CATEGORY_MAP = {
    "food": "food",
    "restaurant": "food",
    "restaurants": "food",
    "meal": "food",
    "meals": "food",
    "attraction": "attractions",
    "attractions": "attractions",
    "poi": "attractions",
    "pois": "attractions",
    "sightseeing": "attractions",
    "flight": "flight",
    "flights": "flight",
    "transport": "flight",
    "hotel": "hotel",
    "hotels": "hotel",
    "stay": "hotel",
}

REFINEMENT_BUDGET_KEYS = {
    "food": ("food", "food_budget"),
    "attractions": ("attractions", "attractions_budget"),
    "hotel": ("hotel", "hotel_budget"),
    "flight": ("transport", None),
}


def _parse_iso_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _one_year_from_today():
    today = get_today_date()
    try:
        return today.replace(year=today.year + 1)
    except ValueError:
        return today.replace(year=today.year + 1, day=28)


def _validate_request_for_planning(parsed: dict) -> str | None:
    travel_dates = parsed.get("travel_dates") or {}
    start = _parse_iso_date(travel_dates.get("start"))
    end = _parse_iso_date(travel_dates.get("end"))
    today = get_today_date()

    if not start:
        return (
            "An exact departure date is required before planning flights. "
            "If the trip starts tomorrow, confirm that. Otherwise provide the departure date in YYYY-MM-DD format."
        )

    if start and start < today:
        return f"Travel start date {start.isoformat()} is in the past. Please provide a date from {today.isoformat()} onward."
    if start and start > _one_year_from_today():
        return "Travel dates must be within the next year."
    if start and end and end < start:
        return "Travel end date cannot be earlier than the start date."

    budget = (parsed.get("budget") or {}).get("total")
    if budget is not None and budget <= 0:
        return "Total budget must be greater than 0."

    num_people = parsed.get("num_people")
    if num_people is not None and (num_people <= 0 or num_people > 50):
        return "Number of travelers must be between 1 and 50."

    duration_days = parsed.get("duration_days")
    if duration_days is not None and (duration_days <= 0 or duration_days > 60):
        return "Trip length must be between 1 and 60 days."

    return None


def _normalize_effective_cities(parsed: dict) -> list:
    city_plan = parsed.get("city_plan", {}) or {}
    planned_cities = city_plan.get("cities", [])

    normalized = []

    if isinstance(planned_cities, list) and planned_cities:
        for item in planned_cities:
            if isinstance(item, dict):
                city_name = item.get("city")
                if city_name:
                    normalized.append({
                        "city": city_name,
                        "days": item.get("days"),
                        "source": item.get("source"),
                    })
            elif item:
                normalized.append({
                    "city": str(item),
                    "days": None,
                    "source": "destination",
                })
        if normalized:
            return normalized

    destination = parsed.get("destination", [])
    if isinstance(destination, list) and destination:
        for city in destination:
            if city:
                normalized.append({
                    "city": str(city),
                    "days": None,
                    "source": "destination",
                })

    return normalized


def _clean_flight(flight: dict) -> dict:
    if not isinstance(flight, dict):
        return {}

    return {
        "airline": flight.get("airline"),
        "departure_airport": flight.get("departure_airport"),
        "departure_code": flight.get("departure_code"),
        "arrival_airport": flight.get("arrival_airport"),
        "arrival_code": flight.get("arrival_code"),
        "departure_time": flight.get("departure_time"),
        "arrival_time": flight.get("arrival_time"),
        "duration_minutes": flight.get("duration_minutes"),
        "stops": flight.get("stops"),
        "price_total": flight.get("price_total"),
        "currency": flight.get("currency"),
        "travel_class": flight.get("travel_class"),
        "provider": flight.get("provider"),
        "why_selected": flight.get("why_selected"),
        "is_estimated_fallback": flight.get("is_estimated_fallback"),
        "ground_to_departure_airport_hours": flight.get("ground_to_departure_airport_hours"),
        "ground_from_arrival_airport_hours": flight.get("ground_from_arrival_airport_hours"),
        "flight_buffer_hours": flight.get("flight_buffer_hours"),
        "flight_total_comparable_hours": flight.get("flight_total_comparable_hours"),
        "direct_drive_comparable_hours": flight.get("direct_drive_comparable_hours"),
        "airport_time_comparison": flight.get("airport_time_comparison"),
    }


def _clean_flight_leg(leg: dict) -> dict:
    if not isinstance(leg, dict):
        return {}

    return {
        "leg_type": leg.get("leg_type"),
        "leg_index": leg.get("leg_index"),
        "from_city": leg.get("from_city"),
        "to_city": leg.get("to_city"),
        "from_airport": leg.get("from_airport"),
        "to_airport": leg.get("to_airport"),
        "candidate_from_airports": leg.get("candidate_from_airports", []),
        "candidate_to_airports": leg.get("candidate_to_airports", []),
        "from_airport_candidate_source": leg.get("from_airport_candidate_source"),
        "to_airport_candidate_source": leg.get("to_airport_candidate_source"),
        "departure_date": leg.get("departure_date"),
        "budget": leg.get("budget"),
        "selection_status": leg.get("selection_status"),
        "transport_mode": leg.get("transport_mode"),
        "rental_car": leg.get("rental_car"),
        "rental_car_cost_estimated": leg.get("rental_car_cost_estimated"),
        "direct_drive_hours": leg.get("direct_drive_hours"),
        "direct_drive_distance_km": leg.get("direct_drive_distance_km"),
        "direct_drive_source": leg.get("direct_drive_source"),
        "selected_flights": [
            _clean_flight(x) for x in leg.get("selected_flights", [])
        ],
    }


def _clean_hotel(hotel: dict) -> dict:
    if not isinstance(hotel, dict):
        return {}

    return {
        "city": hotel.get("city"),
        "name": hotel.get("name"),
        "place_id": hotel.get("place_id"),
        "area": hotel.get("area"),
        "rating": hotel.get("rating"),
        "stars": hotel.get("stars"),
        "lat": hotel.get("lat"),
        "lng": hotel.get("lng"),
        "realtime_price_available": hotel.get("realtime_price_available"),
        "realtime_nightly_per_room": hotel.get("realtime_nightly_per_room"),
        "estimated_total_cost": hotel.get("estimated_total_cost"),
        "why_selected": hotel.get("why_selected"),
    }


def _clean_poi(poi: dict) -> dict:
    if not isinstance(poi, dict):
        return {}

    return {
        "city": poi.get("city"),
        "name": poi.get("name"),
        "place_id": poi.get("place_id"),
        "address": poi.get("address"),
        "rating": poi.get("rating"),
        "lat": poi.get("lat"),
        "lng": poi.get("lng"),
        "types": poi.get("types", []),
        "matched_query": poi.get("matched_query"),
        "estimated_ticket_per_person": poi.get("estimated_ticket_per_person"),
        "estimated_total_cost": poi.get("estimated_total_cost"),
        "why_selected": poi.get("why_selected"),
    }


def _clean_restaurant(restaurant: dict) -> dict:
    if not isinstance(restaurant, dict):
        return {}

    return {
        "city": restaurant.get("city"),
        "name": restaurant.get("name"),
        "place_id": restaurant.get("place_id"),
        "area": restaurant.get("area"),
        "rating": restaurant.get("rating"),
        "lat": restaurant.get("lat"),
        "lng": restaurant.get("lng"),
        "types": restaurant.get("types", []),
        "matched_query": restaurant.get("matched_query"),
        "local_food_match_score": restaurant.get("local_food_match_score"),
        "estimated_meal_per_person": restaurant.get("estimated_meal_per_person"),
        "recommended_visits": restaurant.get("recommended_visits"),
        "estimated_total_cost": restaurant.get("estimated_total_cost"),
        "why_selected": restaurant.get("why_selected"),
    }


def _clean_city_selection_summary(city_summary: dict) -> dict:
    if not isinstance(city_summary, dict):
        return {}

    return {
        "city": city_summary.get("city"),
        "days": city_summary.get("days"),
        "check_in": city_summary.get("check_in"),
        "check_out": city_summary.get("check_out"),
        "hotel_budget": city_summary.get("hotel_budget"),
        "food_budget": city_summary.get("food_budget"),
        "attractions_budget": city_summary.get("attractions_budget"),
        "selected_hotels": city_summary.get("selected_hotels"),
        "selected_restaurants": city_summary.get("selected_restaurants"),
        "selected_pois": city_summary.get("selected_pois"),
        "hotel_selection_status": city_summary.get("hotel_selection_status"),
        "restaurant_selection_status": city_summary.get("restaurant_selection_status"),
        "poi_selection_status": city_summary.get("poi_selection_status"),
        "hotel_provider": city_summary.get("hotel_provider"),
    }


def _clean_route_plan(route_plan: dict) -> dict:
    if not isinstance(route_plan, dict):
        return {}

    def _clean_route(route: dict) -> dict:
        if not isinstance(route, dict):
            return {}
        return {
            "leg_type": route.get("leg_type"),
            "leg_index": route.get("leg_index"),
            "from_city": route.get("from_city"),
            "to_city": route.get("to_city"),
            "from_airport": route.get("from_airport"),
            "to_airport": route.get("to_airport"),
            "candidate_from_airports": route.get("candidate_from_airports", []),
            "candidate_to_airports": route.get("candidate_to_airports", []),
            "from_airport_candidate_source": route.get("from_airport_candidate_source"),
            "to_airport_candidate_source": route.get("to_airport_candidate_source"),
            "departure_date": route.get("departure_date"),
            "direct_drive_hours": route.get("direct_drive_hours"),
            "direct_drive_distance_km": route.get("direct_drive_distance_km"),
            "direct_drive_source": route.get("direct_drive_source"),
            "flight_skip_reason": route.get("flight_skip_reason"),
            "flight_skip_distance_km": route.get("flight_skip_distance_km"),
        }

    return {
        "outbound": _clean_route(route_plan.get("outbound", {})),
        "intercity": [_clean_route(x) for x in route_plan.get("intercity", [])],
        "inbound": _clean_route(route_plan.get("inbound", {})),
    }


def _clean_final_output(agent3_output: dict, parsed: dict, budget_result: dict | None = None) -> dict:
    """
    Keep only the necessary fields for final presentation.
    """
    if not isinstance(agent3_output, dict):
        return {
            "success": False,
            "stage": "agent3_real_selection",
            "error": "Invalid agent3 output."
        }

    budget_result = budget_result or {}
    budget_check = agent3_output.get("budget_check") or {}

    return {
        "success": agent3_output.get("success", False),
        "stage": agent3_output.get("stage", "agent3_real_selection"),
        "trip_summary": {
            "origin": agent3_output.get("origin"),
            "destination": agent3_output.get("destination", []),
            "country": parsed.get("country"),
            "num_people": parsed.get("num_people"),
            "duration_days": parsed.get("duration_days"),
            "travel_dates": parsed.get("travel_dates", {}),
            "budget": parsed.get("budget", {}),
            "preferences": parsed.get("preferences", {}),
            "planning_signals": parsed.get("planning_signals", {}),
        },
        "effective_cities": agent3_output.get("effective_cities", []),
        "city_date_ranges": agent3_output.get("city_date_ranges", {}),
        "mid_distance_car_rental": agent3_output.get("mid_distance_car_rental", {}),
        "route_plan": _clean_route_plan(agent3_output.get("route_plan", {})),

        "selected_outbound_flights": [
            _clean_flight(x) for x in agent3_output.get("selected_outbound_flights", [])
        ],
        "selected_intercity_flights": [
            _clean_flight_leg(x) for x in agent3_output.get("selected_intercity_flights", [])
        ],
        "selected_inbound_flights": [
            _clean_flight(x) for x in agent3_output.get("selected_inbound_flights", [])
        ],
        "selected_all_flight_legs": [
            _clean_flight_leg(x) for x in agent3_output.get("selected_all_flight_legs", [])
        ],

        "selected_hotels": [
            _clean_hotel(x) for x in agent3_output.get("selected_hotels", [])
        ],
        "selected_pois": [
            _clean_poi(x) for x in agent3_output.get("selected_pois", [])
        ],
        "selected_restaurants": [
            _clean_restaurant(x) for x in agent3_output.get("selected_restaurants", [])
        ],

        "city_selection_summary": [
            _clean_city_selection_summary(x)
            for x in agent3_output.get("city_selection_summary", [])
        ],

        "budget_check": {
            "total_budget_input": budget_check.get("total_budget_input"),
            "transport_budget_input": budget_check.get("transport_budget_input"),
            "flight_leg_budget_map": budget_check.get("flight_leg_budget_map", {}),
            "hotel_budget_input": budget_check.get("hotel_budget_input"),
            "food_budget_input": budget_check.get("food_budget_input"),
            "attractions_budget_input": budget_check.get("attractions_budget_input"),
            "selected_outbound_flight_spend": budget_check.get("selected_outbound_flight_spend"),
            "selected_intercity_flight_spend": budget_check.get("selected_intercity_flight_spend"),
            "selected_mid_distance_car_rental_spend": budget_check.get("selected_mid_distance_car_rental_spend"),
            "selected_inbound_flight_spend": budget_check.get("selected_inbound_flight_spend"),
            "selected_total_flight_spend": budget_check.get("selected_total_flight_spend"),
            "selected_total_transport_spend": budget_check.get("selected_total_transport_spend"),
            "selected_hotel_total_reference": budget_check.get("selected_hotel_total_reference"),
            "selected_restaurant_total": budget_check.get("selected_restaurant_total"),
            "selected_poi_total": budget_check.get("selected_poi_total"),
        },

        "budget_plan": budget_result.get("budget_plan", {}),
        "city_budget_plan": (
            budget_result.get("city_budget_plan")
            or agent3_output.get("city_budget_plan_used")
            or []
        ),
        "budget_strategy": budget_result.get("budget_strategy"),
        "cost_level": budget_result.get("cost_level"),
        "normalized_preferences": budget_result.get("normalized_preferences", {}),
        "allocation_basis": budget_result.get("allocation_basis", {}),

        "budget_fit_status": agent3_output.get("budget_fit_status", {}),
        "warnings": agent3_output.get("warnings", []),
    }


def _selected_total_from_budget_check(budget_check: dict) -> float:
    if not isinstance(budget_check, dict):
        return 0.0
    keys = [
        "selected_total_transport_spend",
        "selected_hotel_total_reference",
        "selected_restaurant_total",
        "selected_poi_total",
    ]
    total = 0.0
    for key in keys:
        try:
            total += float(budget_check.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return round(total, 2)


def _category_budget_spend(budget_check: dict, category: str) -> tuple[float, float]:
    if category == "food":
        return (
            _safe_float(budget_check.get("selected_restaurant_total"), 0.0),
            _safe_float(budget_check.get("food_budget_input"), 0.0),
        )
    if category == "attractions":
        return (
            _safe_float(budget_check.get("selected_poi_total"), 0.0),
            _safe_float(budget_check.get("attractions_budget_input"), 0.0),
        )
    if category == "hotel":
        return (
            _safe_float(budget_check.get("selected_hotel_total_reference"), 0.0),
            _safe_float(budget_check.get("hotel_budget_input"), 0.0),
        )
    if category == "flight":
        return (
            _safe_float(budget_check.get("selected_total_transport_spend"), 0.0),
            _safe_float(budget_check.get("transport_budget_input"), 0.0),
        )
    return 0.0, 0.0


def _build_refinement_state(final_output: dict, refinement_context: dict) -> tuple[dict, list[dict]]:
    previous_state = refinement_context.get("previous_refinement_state") if isinstance(refinement_context, dict) else {}
    if not isinstance(previous_state, dict):
        previous_state = {}

    category_ceilings = dict(previous_state.get("categoryCeilings") or previous_state.get("category_ceilings") or {})
    category_spend = dict(previous_state.get("categorySpend") or previous_state.get("category_spend") or {})
    target_categories = refinement_context.get("target_categories") or []
    budget_check = final_output.get("budget_check") or {}
    budget_plan = final_output.get("budget_plan") or {}
    total_budget = _safe_float(budget_check.get("total_budget_input"), 0.0)
    selected_total = _selected_total_from_budget_check(budget_check)
    statuses: list[dict] = []

    for category in ["food", "attractions", "flight", "hotel"]:
        spend, budget = _category_budget_spend(budget_check, category)
        ratio = round(spend / budget, 4) if budget > 0 else 0.0
        remaining = max(0.0, budget - spend)
        budget_key, _ = REFINEMENT_BUDGET_KEYS.get(category, (None, None))
        buffer_left = _safe_float(budget_plan.get("buffer"), 0.0) if isinstance(budget_plan, dict) else 0.0
        is_ceiling = bool(budget > 0 and (ratio >= 0.98 or (remaining <= max(100.0, budget * 0.02) and buffer_left <= 0)))
        if is_ceiling:
            category_ceilings[category] = {
                "reason": f"{category} is already at the maximum under the current budget.",
                "spend": round(spend, 2),
                "budget": round(budget, 2),
                "ratio": ratio,
            }
        elif category in category_ceilings and category in target_categories:
            category_ceilings.pop(category, None)

        category_spend[category] = {
            "spend": round(spend, 2),
            "budget": round(budget, 2),
            "ratio": ratio,
            "budget_key": budget_key,
        }

        if category in target_categories:
            statuses.append({
                "category": category,
                "status": "maxed" if is_ceiling else "upgraded",
                "message": (
                    f"{category.title()} is already at the maximum under the current budget."
                    if is_ceiling
                    else f"{category.title()} upgraded under the current planning budget."
                ),
                "spend": round(spend, 2),
                "budget": round(budget, 2),
                "ratio": ratio,
            })

    state = {
        "categoryCeilings": category_ceilings,
        "categorySpend": category_spend,
        "lastTargetCategories": list(target_categories),
        "maxActivityRadiusKm": refinement_context.get("max_activity_radius_km"),
        "allowTotalBudgetIncrease": bool(refinement_context.get("allow_total_budget_increase")),
        "totalBudget": round(total_budget, 2),
        "selectedTotal": round(selected_total, 2),
    }
    return state, statuses


def _build_planning_baseline(final_output: dict, user_request: str) -> dict:
    budget_check = final_output.get("budget_check") or {}
    total_budget = _safe_float(budget_check.get("total_budget_input"), 0.0)
    if total_budget <= 0:
        total_budget = _safe_float((((final_output.get("trip_summary") or {}).get("budget") or {}).get("total")), 0.0)
    selected_total = _selected_total_from_budget_check(budget_check)
    return {
        "request": str(user_request or "").strip(),
        "total_budget": round(total_budget, 2) if total_budget > 0 else None,
        "selected_total": round(selected_total, 2) if selected_total > 0 else None,
    }


def _refinement_non_decrease_error(
    final_output: dict,
    refinement_context: dict,
    rough_budget_check: dict,
    precheck_warnings: list[str] | None = None,
) -> dict | None:
    if not refinement_context:
        return None
    previous_selected_total = _safe_float(refinement_context.get("previous_selected_total"), 0.0)
    if previous_selected_total <= 0:
        return None
    budget_check = final_output.get("budget_check") or {}
    selected_total = _selected_total_from_budget_check(budget_check)
    if selected_total + 1.0 >= previous_selected_total:
        return None

    state, statuses = _build_refinement_state(final_output, refinement_context)
    return {
        "success": False,
        "stage": "refinement_non_decrease",
        "error": (
            f"Generated refinement would reduce total selected spend from {round(previous_selected_total, 2)} CNY "
            f"to {round(selected_total, 2)} CNY. Keeping the previous plan instead; increase the budget "
            "or choose another upgrade target if you want a higher-spec version."
        ),
        "trip_summary": final_output.get("trip_summary", {}),
        "budget_check": budget_check,
        "rough_budget_check": rough_budget_check,
        "refinement_state": state,
        "upgrade_status": statuses,
        "warnings": (final_output.get("warnings", []) + (precheck_warnings or [])),
    }


def _build_mid_distance_car_rental_plan(parsed: dict, budget_result: dict) -> dict:
    city_days = budget_result.get("city_budget_plan") or _normalize_effective_cities(parsed)
    if not isinstance(city_days, list) or len(city_days) <= 1:
        return {}

    currency = ((parsed.get("budget") or {}).get("currency")) or "CNY"
    return estimate_mid_distance_car_rental(
        city_days=city_days,
        country=parsed.get("country"),
        currency=currency,
        use_llm=True,
    )


def _log_stage(stage: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    print(f"[pipeline] {stage} finished in {elapsed:.2f}s")


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_refinement_categories(values) -> list[str]:
    if isinstance(values, str):
        raw_values = re.split(r"[,/|\s]+", values)
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []

    categories: list[str] = []
    for value in raw_values:
        key = str(value or "").strip().lower()
        normalized = REFINEMENT_CATEGORY_MAP.get(key)
        if normalized and normalized not in categories:
            categories.append(normalized)
    return categories


def _normalize_refinement_context(refinement_context: dict | None) -> dict:
    if not isinstance(refinement_context, dict):
        return {}
    if str(refinement_context.get("mode") or "").lower() != "refine":
        return {}

    max_radius = _safe_float(refinement_context.get("max_activity_radius_km"), 0.0)
    if max_radius not in {5.0, 10.0, 15.0}:
        max_radius = 0.0

    allow_budget_increase = bool(refinement_context.get("allow_total_budget_increase"))
    new_total_budget = _safe_float(refinement_context.get("new_total_budget"), 0.0)
    previous_total_budget = _safe_float(refinement_context.get("previous_total_budget"), 0.0)
    previous_selected_total = _safe_float(refinement_context.get("previous_selected_total"), 0.0)

    return {
        "mode": "refine",
        "previous_request": str(refinement_context.get("previous_request") or "").strip(),
        "followup_message": str(refinement_context.get("followup_message") or "").strip(),
        "target_categories": _normalize_refinement_categories(refinement_context.get("target_categories")),
        "allow_total_budget_increase": allow_budget_increase,
        "new_total_budget": new_total_budget if allow_budget_increase and new_total_budget > 0 else None,
        "previous_total_budget": previous_total_budget if previous_total_budget > 0 else None,
        "previous_selected_total": previous_selected_total if previous_selected_total > 0 else None,
        "max_activity_radius_km": max_radius or None,
        "previous_refinement_state": (
            refinement_context.get("previous_refinement_state")
            if isinstance(refinement_context.get("previous_refinement_state"), dict)
            else {}
        ),
    }


def _apply_refinement_to_parsed(parsed: dict, refinement_context: dict) -> dict:
    if not refinement_context:
        return parsed

    parsed = copy.deepcopy(parsed)
    categories = set(refinement_context.get("target_categories") or [])
    preferences = parsed.setdefault("preferences", {})
    planning_signals = parsed.setdefault("planning_signals", {})

    new_total_budget = refinement_context.get("new_total_budget")
    if new_total_budget:
        budget = parsed.setdefault("budget", {})
        budget["total"] = int(round(float(new_total_budget)))
    else:
        previous_total_budget = refinement_context.get("previous_total_budget")
        if previous_total_budget:
            budget = parsed.setdefault("budget", {})
            budget["total"] = int(round(float(previous_total_budget)))

    if "food" in categories:
        preferences["food"] = "fine_dining"
        food_preferences = preferences.get("food_preferences")
        if not isinstance(food_preferences, list):
            food_preferences = []
        lower_food_preferences = [str(x).lower() for x in food_preferences]
        if "fine_dining" not in lower_food_preferences:
            food_preferences.append("fine_dining")
        preferences["food_preferences"] = food_preferences
        planning_signals["food_priority"] = "high"

    if "attractions" in categories:
        planning_signals["attraction_intensity"] = "heavy"

    if "flight" in categories:
        preferences["flight"] = "comfort"
        preferences["flight_cabin"] = preferences.get("flight_cabin") or "business"
        planning_signals["flight_priority"] = "high"

    if "hotel" in categories:
        preferences["hotel"] = "luxury"
        preferences["hotel_level"] = "luxury"
        planning_signals["hotel_level"] = "luxury"
        planning_signals["hotel_priority"] = "high"

    parsed["refinement_context"] = refinement_context
    return parsed


def _scale_city_budget_field(city_budget_plan: list, field: str, new_total: float) -> None:
    if not city_budget_plan or not field:
        return
    current_total = sum(_safe_float(item.get(field), 0.0) for item in city_budget_plan if isinstance(item, dict))
    total_days = sum(max(1, _safe_int(item.get("days"), 1)) for item in city_budget_plan if isinstance(item, dict)) or 1
    for item in city_budget_plan:
        if not isinstance(item, dict):
            continue
        if current_total > 0:
            share = _safe_float(item.get(field), 0.0) / current_total
        else:
            share = max(1, _safe_int(item.get("days"), 1)) / total_days
        item[field] = round(max(0.0, new_total * share), 2)


def _apply_refinement_budget_bias(budget_result: dict, refinement_context: dict) -> dict:
    if not refinement_context or not isinstance(budget_result, dict):
        return budget_result

    categories = [c for c in refinement_context.get("target_categories", []) if c in REFINEMENT_BUDGET_KEYS]
    if not categories:
        return budget_result

    budget_result = copy.deepcopy(budget_result)
    budget_plan = budget_result.get("budget_plan")
    if not isinstance(budget_plan, dict):
        return budget_result

    buffer_value = max(0.0, _safe_float(budget_plan.get("buffer"), 0.0))
    move_amount = buffer_value * 0.8 if buffer_value > 0 else 0.0
    if move_amount <= 0 and refinement_context.get("allow_total_budget_increase"):
        total_budget = _safe_float((budget_result.get("budget") or {}).get("total"), 0.0)
        plan_total = sum(_safe_float(budget_plan.get(k), 0.0) for k in ["transport", "hotel", "food", "attractions", "buffer"])
        move_amount = max(0.0, total_budget - plan_total)

    if move_amount <= 0:
        budget_result.setdefault("allocation_basis", {})["refinement_budget_bias"] = {
            "applied": False,
            "reason": "no_buffer_or_new_budget_available",
            "target_categories": categories,
        }
        return budget_result

    per_category = move_amount / max(1, len(categories))
    for category in categories:
        budget_key, city_field = REFINEMENT_BUDGET_KEYS[category]
        budget_plan[budget_key] = round(_safe_float(budget_plan.get(budget_key), 0.0) + per_category, 2)
        if city_field:
            _scale_city_budget_field(
                budget_result.get("city_budget_plan") or [],
                city_field,
                _safe_float(budget_plan.get(budget_key), 0.0),
            )

    budget_plan["buffer"] = round(max(0.0, buffer_value - move_amount), 2)
    budget_result["recommended_spend"] = dict(budget_plan)
    budget_result.setdefault("allocation_basis", {})["refinement_budget_bias"] = {
        "applied": True,
        "moved_from_buffer": round(move_amount, 2),
        "target_categories": categories,
    }
    return budget_result


def _round_up_to_100(value: float) -> int:
    amount = max(0.0, float(value))
    return int(math.ceil(amount / 100.0) * 100)


def _minimum_budget_from_core_estimate(core_total: float, core_ratio: float) -> int:
    ratio = max(0.01, float(core_ratio))
    return _round_up_to_100(float(core_total) / ratio)


def _minimum_budget_from_rough_precheck(core_total: float, core_ratio: float) -> int:
    # Rough precheck uses static cost tables before live flight/hotel search.
    # Add a safety margin so the first suggested budget does not immediately fail
    # when live flight or hotel prices come back higher than static estimates.
    live_search_safety_multiplier = 1.30
    return _minimum_budget_from_core_estimate(float(core_total) * live_search_safety_multiplier, core_ratio)


def _build_minimum_execution_prefs(normalized_prefs: dict) -> dict:
    prefs = dict(normalized_prefs or {})
    prefs["flight_type"] = "budget"
    prefs["hotel_level"] = "budget"
    prefs["hotel_level_detail"] = "budget"
    prefs["hotel_central"] = "no"
    prefs["hotel_clean"] = "no"
    prefs["food_level"] = "budget"
    prefs["wants_fine_dining"] = "no"
    return prefs


def _estimate_rough_core_cost(parsed: dict, normalized_prefs: dict) -> dict:
    weighted_profile = _get_weighted_cost_profile(parsed)
    _, destination_profile, _ = _get_primary_profile(parsed)
    season_factor = _get_season_factor(parsed, destination_profile)

    transport = _estimate_transport_cost(parsed, weighted_profile, normalized_prefs, season_factor)
    hotel = _estimate_hotel_cost(parsed, weighted_profile, normalized_prefs, season_factor)
    food = _estimate_food_cost(parsed, weighted_profile, normalized_prefs, season_factor)

    return {
        "transport": int(transport),
        "hotel": int(hotel),
        "food_three_meals": int(food),
        "total": int(transport + hotel + food),
        "cost_level_reference": str(weighted_profile.get("cost_level") or "medium"),
    }


def _run_rough_budget_precheck(parsed: dict) -> dict:
    total_budget = _safe_int(((parsed.get("budget") or {}).get("total")), 0)
    if total_budget <= 0:
        return {
            "checked": False,
            "status": "skip_invalid_budget",
            "total_budget": total_budget,
        }

    core_threshold_ratio = 0.70
    core_threshold_amount = round(total_budget * core_threshold_ratio, 2)

    normal_prefs = _normalize_preference_levels(parsed)
    normal_estimate = _estimate_rough_core_cost(parsed, normal_prefs)
    normal_total = float(normal_estimate.get("total") or 0.0)
    normal_ratio = round(normal_total / float(total_budget), 4) if total_budget > 0 else 0.0

    if normal_total <= core_threshold_amount:
        return {
            "checked": True,
            "status": "normal_feasible",
            "total_budget": total_budget,
            "core_threshold_ratio": core_threshold_ratio,
            "core_threshold_amount": core_threshold_amount,
            "normal_estimate": normal_estimate,
            "normal_ratio_to_budget": normal_ratio,
            "minimum_estimate": None,
            "minimum_ratio_to_budget": None,
        }

    minimum_prefs = _build_minimum_execution_prefs(normal_prefs)
    minimum_estimate = _estimate_rough_core_cost(parsed, minimum_prefs)
    minimum_total = float(minimum_estimate.get("total") or 0.0)
    minimum_ratio = round(minimum_total / float(total_budget), 4) if total_budget > 0 else 0.0
    status = "minimum_feasible_only" if minimum_total <= core_threshold_amount else "minimum_infeasible"
    minimum_budget_required = _minimum_budget_from_rough_precheck(minimum_total, core_threshold_ratio)
    return {
        "checked": True,
        "status": status,
        "total_budget": total_budget,
        "core_threshold_ratio": core_threshold_ratio,
        "core_threshold_amount": core_threshold_amount,
        "normal_estimate": normal_estimate,
        "normal_ratio_to_budget": normal_ratio,
        "minimum_estimate": minimum_estimate,
        "minimum_ratio_to_budget": minimum_ratio,
        "minimum_budget_required": minimum_budget_required,
        "minimum_budget_basis": "core_three_components_with_70pct_guard_plus_live_search_safety_margin",
    }


def _is_budget_cap_followup(user_request: str) -> bool:
    text = (user_request or "").lower()
    if re.search(
        r"\brerun\s+the\s+full\s+planning\s+pipeline\b|\bfull\s+planning\s+pipeline\s+with\s+the\s+new\s+budget\b",
        text,
    ):
        return False
    patterns = [
        r"\bkeep\s+(?:the\s+)?same\s+plan\b",
        r"\baccept\s+(?:a\s+)?small\s+over",
        r"\bfit\s+(?:the\s+)?budget\b",
        r"\bstay\s+at\s+budget\b",
        r"\bbudget\s+hotels?\b",
        r"\bminimum-cost\b",
        r"\bcheapest\s+executable\s+plan\b",
        r"\bdo\s+not\s+upgrade\b",
        r"\bbudget-feasibility\s+failure\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _requests_remove_paid_activities(user_request: str) -> bool:
    text = (user_request or "").lower()
    return bool(
        re.search(
            r"remove all paid|remove paid|free attractions?|no paid attractions?|skip paid|不要付费|去掉付费|免费景点|不要门票|去掉门票|不买门票",
            text,
        )
    )


def _apply_budget_cap_mode(parsed: dict, user_request: str) -> dict:
    if not _is_budget_cap_followup(user_request):
        return parsed

    parsed = dict(parsed)
    preferences = dict(parsed.get("preferences") or {})
    planning_signals = dict(parsed.get("planning_signals") or {})

    preferences.setdefault("hotel_features", [])
    preferences.setdefault("food_preferences", [])
    preferences.setdefault("attraction_styles", [])
    preferences["hotel_level"] = "budget"
    if not preferences.get("hotel") or str(preferences.get("hotel")).lower() in {"standard", "clean", "central"}:
        preferences["hotel"] = "budget"

    planning_signals["budget_discipline"] = "cap"
    planning_signals["hotel_level"] = "low"
    planning_signals.setdefault("food_priority", "medium")
    planning_signals.setdefault("attraction_intensity", "low")
    planning_signals["allow_paid_attractions"] = "no" if _requests_remove_paid_activities(user_request) else "yes"

    parsed["preferences"] = preferences
    parsed["planning_signals"] = planning_signals
    parsed["budget_cap_mode"] = True
    return parsed


def _apply_comfortable_budget_fit_mode(parsed: dict) -> dict:
    parsed = dict(parsed or {})
    preferences = dict(parsed.get("preferences") or {})
    planning_signals = dict(parsed.get("planning_signals") or {})

    if not preferences.get("flight_cabin"):
        preferences["flight_cabin"] = "business"
        preferences.setdefault("flight", "comfort")

    hotel_value = str(preferences.get("hotel") or "").lower()
    if not hotel_value or hotel_value in {"standard", "clean", "central", "budget"}:
        preferences["hotel"] = "luxury"
    preferences["hotel_level"] = "luxury"

    food_value = str(preferences.get("food") or "").lower()
    if not food_value or food_value in {"standard", "mixed", "budget"}:
        preferences["food"] = "fine_dining"

    food_preferences = preferences.get("food_preferences")
    if not isinstance(food_preferences, list):
        food_preferences = []
    if "fine_dining" not in food_preferences:
        food_preferences.append("fine_dining")
    preferences["food_preferences"] = food_preferences

    planning_signals["flight_priority"] = "high"
    planning_signals["hotel_level"] = "high"
    planning_signals["hotel_priority"] = "high"
    planning_signals["food_priority"] = "high"
    planning_signals["attraction_intensity"] = "high"

    parsed["preferences"] = preferences
    parsed["planning_signals"] = planning_signals
    parsed["budget_fit_mode"] = "comfortable"
    parsed.pop("budget_cap_mode", None)
    return parsed


def _apply_budget_maximize_experience_mode(parsed: dict, flight_cabin: str | None = None) -> dict:
    parsed = dict(parsed or {})
    preferences = dict(parsed.get("preferences") or {})
    planning_signals = dict(parsed.get("planning_signals") or {})

    cabin = str(flight_cabin or preferences.get("flight_cabin") or "").strip().lower()
    if cabin:
        preferences["flight_cabin"] = cabin
    elif not preferences.get("flight_cabin"):
        preferences["flight_cabin"] = "business"
    preferences.setdefault("flight", "comfort")

    preferences["hotel"] = "luxury"
    preferences["hotel_level"] = "luxury"

    preferences["food"] = "fine_dining"
    food_preferences = preferences.get("food_preferences")
    if not isinstance(food_preferences, list):
        food_preferences = []
    if "fine_dining" not in food_preferences:
        food_preferences.append("fine_dining")
    preferences["food_preferences"] = food_preferences

    planning_signals["budget_discipline"] = "maximize_experience"
    planning_signals["flight_priority"] = "high"
    planning_signals["hotel_level"] = "high"
    planning_signals["hotel_priority"] = "high"
    planning_signals["food_priority"] = "high"
    planning_signals["attraction_intensity"] = "high"
    planning_signals.setdefault("allow_paid_attractions", "yes")

    parsed["preferences"] = preferences
    parsed["planning_signals"] = planning_signals
    parsed["budget_maximize_experience"] = True
    parsed.pop("budget_cap_mode", None)
    return parsed


def _finalize_success_output(final_output: dict, user_request: str, auto_open_report: bool) -> dict:
    started = time.perf_counter()
    try:
        itinerary_output = run_itinerary_agent(final_output)
        final_output["itinerary"] = itinerary_output
    except Exception as e:
        print("[pipeline] agent4_itinerary exception", flush=True)
        traceback.print_exc()
        final_output.setdefault("warnings", [])
        final_output["warnings"].append(f"Agent4 itinerary planning failed: {str(e)}")
        final_output["itinerary"] = {
            "success": False,
            "stage": "agent4_itinerary_planning",
            "error": str(e),
        }
    finally:
        _log_stage("agent4_itinerary", started)

    itinerary = final_output.get("itinerary") or {}
    if not itinerary.get("success", False):
        final_output["success"] = False
        final_output["stage"] = itinerary.get("stage", "agent4_itinerary_planning")
        final_output["error"] = itinerary.get("error", "Agent4 itinerary planning failed.")
        final_output.setdefault("warnings", [])
        final_output["runtime"] = {
            "router_process_started_at": ROUTER_PROCESS_STARTED_AT,
            "router_code_version": ROUTER_CODE_VERSION,
        }
        return final_output

    itinerary_summary = itinerary.get("itinerary_summary") or {}
    if itinerary_summary.get("budget_feasible") is False:
        cost_rollup = itinerary.get("cost_rollup") or {}
        total_budget = ((final_output.get("trip_summary") or {}).get("budget") or {}).get("total")
        estimated_total = _safe_float(cost_rollup.get("total_selected_estimated"), 0.0)
        rough_minimum_budget_required = _safe_int(
            (final_output.get("rough_budget_check") or {}).get("minimum_budget_required"),
            0,
        )
        minimum_budget_required = max(_round_up_to_100(estimated_total), rough_minimum_budget_required)
        final_output["success"] = False
        final_output["stage"] = "budget_feasibility"
        final_output["minimum_budget_required"] = minimum_budget_required
        final_output["budget_check"] = {
            **(final_output.get("budget_check") or {}),
            "total_budget_input": total_budget,
            "selected_total": estimated_total,
            "itinerary_total_selected_estimated": estimated_total,
            "budget_feasible": False,
            "minimum_budget_required": minimum_budget_required,
        }
        final_output["error"] = (
            f"Even after trimming optional meals and paid activities, fixed costs are "
            f"{cost_rollup.get('total_selected_estimated')} CNY against budget {total_budget} CNY. "
            f"Current minimum suggested budget is {minimum_budget_required} CNY. "
            "Please increase the budget, shorten the trip, or change dates/destination."
        )
        return final_output

    started = time.perf_counter()
    try:
        final_output["report"] = write_travel_report(
            final_output,
            user_request,
            auto_open=auto_open_report,
        )
    except Exception as e:
        final_output.setdefault("warnings", [])
        final_output["warnings"].append(f"HTML report generation failed: {str(e)}")
        final_output["report"] = {
            "success": False,
            "error": str(e),
        }
    finally:
        _log_stage("html_report", started)

    final_output["runtime"] = {
        "router_process_started_at": ROUTER_PROCESS_STARTED_AT,
        "router_code_version": ROUTER_CODE_VERSION,
    }
    return final_output


def _run_agent2_agent3(parsed: dict, refinement_context: dict | None = None) -> tuple[dict | None, dict | None, dict | None, dict]:
    started = time.perf_counter()
    budget_result = travel_budget_agent(parsed)
    budget_result = _apply_refinement_budget_bias(budget_result, refinement_context or {})
    _log_stage("agent2_budget", started)

    if budget_result.get("meta", {}).get("error"):
        return None, None, {
            "success": False,
            "stage": "agent2_budget",
            "error": budget_result.get("meta", {}).get("error"),
            "parsed_request": parsed,
            "budget_result": budget_result
        }, budget_result

    started = time.perf_counter()
    mid_distance_car_rental = _build_mid_distance_car_rental_plan(parsed, budget_result)
    _log_stage("mid_distance_car_rental", started)

    agent3_input = {
        "parsed_request": parsed,
        "budget_result": budget_result,
        "mid_distance_car_rental": mid_distance_car_rental,
        "city_budget_plan": budget_result.get("city_budget_plan", []),
        "recommended_spend": budget_result.get("recommended_spend", budget_result.get("budget_plan", {})),
        "budget_plan": budget_result.get("budget_plan", {}),
        "allocation_basis": budget_result.get("allocation_basis", {}),
        "normalized_preferences": budget_result.get("normalized_preferences", {}),
        "budget_strategy": budget_result.get("budget_strategy"),
        "cost_level": budget_result.get("cost_level"),
        "budget_status": budget_result.get("budget_status"),
        "agent2_summary": {
            "budget_strategy": budget_result.get("budget_strategy"),
            "cost_level": budget_result.get("cost_level"),
            "budget_status": budget_result.get("budget_status"),
            "normalized_preferences": budget_result.get("normalized_preferences", {}),
            "allocation_basis": budget_result.get("allocation_basis", {}),
            "budget_plan": budget_result.get("budget_plan", {}),
            "city_budget_plan": budget_result.get("city_budget_plan", []),
        },
        "num_people": parsed.get("num_people"),
        "duration_days": parsed.get("duration_days"),
        "budget": parsed.get("budget"),
        "origin": parsed.get("origin"),
        "destination": parsed.get("destination"),
        "preferences": parsed.get("preferences", {}),
        "planning_signals": parsed.get("planning_signals", {}),
        "refinement_context": refinement_context or {},
    }

    started = time.perf_counter()
    try:
        agent3_agent = Agent3Selection(agent3_input)
        agent3_output = agent3_agent.generate_shortlist()
    except Exception as e:
        return None, None, {
            "success": False,
            "stage": "agent3_real_selection",
            "error": f"agent3 selection failed: {str(e)}",
            "parsed_request": parsed,
            "budget_result": budget_result,
            "agent3_input_preview": {
                "city_budget_plan": agent3_input.get("city_budget_plan", []),
                "normalized_preferences": agent3_input.get("normalized_preferences", {}),
                "allocation_basis": agent3_input.get("allocation_basis", {}),
            }
        }, budget_result
    finally:
        _log_stage("agent3_realtime_selection", started)

    if not agent3_output.get("success"):
        return None, agent3_output, {
            "success": False,
            "stage": agent3_output.get("stage", "agent3_real_selection"),
            "error": agent3_output.get("error", "agent3 failed"),
            "parsed_request": parsed,
            "budget_result": budget_result
        }, budget_result

    return budget_result, agent3_output, None, budget_result


def run_budget_pipeline(user_request: str, auto_open_report: bool = True, refinement_context: dict | None = None) -> dict:
    """
    Full pipeline: agent1 -> agent2 -> agent3
    Final output: return clean final result only.
    """
    refinement_context = _normalize_refinement_context(refinement_context)
    planning_request = refinement_context.get("previous_request") or user_request
    parsed = travel_requirement_parser_llm_agent(planning_request)

    # Agent1 parse failed
    if parsed.get("parse_status") == "failed":
        return {
            "success": False,
            "stage": "agent1_parse",
            "error": parsed.get("meta", {}).get("error"),
            "parsed_request": parsed
        }

    # Agent1 parsed but not ready
    if not parsed.get("is_ready_for_planning", False):
        return {
            "success": False,
            "stage": "agent1_validation",
            "error": "Parsed request is not ready for budget planning.",
            "parsed_request": parsed
        }

    validation_error = _validate_request_for_planning(parsed)
    if validation_error:
        return {
            "success": False,
            "stage": "agent1_validation",
            "error": validation_error,
            "parsed_request": parsed
        }

    parsed = _apply_refinement_to_parsed(parsed, refinement_context)
    parsed = _apply_budget_cap_mode(parsed, user_request)
    rough_budget_check = _run_rough_budget_precheck(parsed)
    rough_status = rough_budget_check.get("status")
    if False and rough_status == "minimum_infeasible":
        normal_total = ((rough_budget_check.get("normal_estimate") or {}).get("total"))
        minimum_total = ((rough_budget_check.get("minimum_estimate") or {}).get("total"))
        budget_total = rough_budget_check.get("total_budget")
        core_threshold_amount = rough_budget_check.get("core_threshold_amount")
        core_threshold_ratio = rough_budget_check.get("core_threshold_ratio")
        minimum_budget_required = rough_budget_check.get("minimum_budget_required")
        return {
            "success": False,
            "stage": "budget_feasibility",
            "error": (
                f"Rough budget precheck failed before realtime search: under current preferences, core spend "
                f"(flight + hotel + three meals) is about {normal_total} CNY. After minimum downgrades, "
                f"it is still about {minimum_total} CNY, which is above the 70% core threshold "
                f"({core_threshold_amount} CNY) of total budget {budget_total} CNY. "
                f"Current minimum suggested budget is {minimum_budget_required} CNY."
            ),
            "parsed_request": parsed,
            "trip_summary": {
                "origin": parsed.get("origin"),
                "destination": parsed.get("destination", []),
                "duration_days": parsed.get("duration_days"),
                "num_people": parsed.get("num_people"),
                "budget": parsed.get("budget", {}),
            },
            "rough_budget_check": rough_budget_check,
            "minimum_budget_required": minimum_budget_required,
            "warnings": [
                (
                    "Core precheck rule triggered: flight + hotel + three meals should stay within "
                    f"{int(float(core_threshold_ratio) * 100)}% of total budget before detailed planning. "
                    "Please increase budget, shorten trip, or relax route/hotel/food constraints."
                ),
            ],
        }

    precheck_warnings: list[str] = []
    if rough_status in {"minimum_feasible_only", "minimum_infeasible"} and not parsed.get("budget_cap_mode"):
        parsed = _apply_budget_cap_mode(
            parsed,
            "fit the budget with cheap flights, budget hotels, and lower-cost meals",
        )
        precheck_warnings.append(
            "Rough budget precheck: core spend (flight + hotel + three meals) exceeded 70% of total budget, so planning switched to lower-cost execution mode before realtime search."
        )

    budget_result, agent3_output, pipeline_error, _ = _run_agent2_agent3(parsed, refinement_context)
    if pipeline_error:
        pipeline_error["rough_budget_check"] = rough_budget_check
        return pipeline_error

    # Clean final output
    final_output = _clean_final_output(agent3_output, parsed, budget_result)
    final_output["original_request"] = user_request
    if refinement_context:
        final_output["refinement_context"] = refinement_context
    final_output["rough_budget_check"] = rough_budget_check

    budget_check = final_output.get("budget_check") or {}
    total_budget = float(budget_check.get("total_budget_input") or 0.0)
    selected_total = _selected_total_from_budget_check(budget_check)
    if total_budget > 0 and selected_total > total_budget:
        final_failed_total = selected_total
        rough_min_budget = _safe_int(rough_budget_check.get("minimum_budget_required"), 0)
        if not parsed.get("budget_cap_mode"):
            fit_parsed = _apply_comfortable_budget_fit_mode(parsed)
            fit_budget_result, fit_agent3_output, fit_error, _ = _run_agent2_agent3(fit_parsed, refinement_context)
            if not fit_error:
                fit_final_output = _clean_final_output(fit_agent3_output, fit_parsed, fit_budget_result)
                fit_budget_check = fit_final_output.get("budget_check") or {}
                fit_total_budget = float(fit_budget_check.get("total_budget_input") or 0.0)
                fit_selected_total = _selected_total_from_budget_check(fit_budget_check)
                if fit_total_budget > 0 and 0 < fit_selected_total <= fit_total_budget:
                    fit_final_output["original_request"] = user_request
                    if refinement_context:
                        fit_final_output["refinement_context"] = refinement_context
                    fit_final_output["rough_budget_check"] = rough_budget_check
                    fit_final_output.setdefault("warnings", [])
                    fit_final_output["warnings"].extend(precheck_warnings)
                    fit_final_output["warnings"].append(
                        "Budget fit adjustment: the first selected plan exceeded the budget, so the system retried with business-class flights and high-comfort hotel/dining execution instead of switching to minimum-cost mode."
                    )
                    fit_final_output["agent2_summary"] = {
                        "budget_strategy": fit_budget_result.get("budget_strategy"),
                        "cost_level": fit_budget_result.get("cost_level"),
                        "budget_status": fit_budget_result.get("budget_status"),
                        "normalized_preferences": fit_budget_result.get("normalized_preferences", {}),
                        "allocation_basis": fit_budget_result.get("allocation_basis", {}),
                        "budget_plan": fit_budget_result.get("budget_plan", {}),
                        "city_budget_plan": fit_budget_result.get("city_budget_plan", []),
                    }
                    if refinement_context:
                        fit_state, fit_status = _build_refinement_state(fit_final_output, refinement_context)
                        fit_final_output["refinement_state"] = fit_state
                        fit_final_output["upgrade_status"] = fit_status
                    fit_final_output["planning_baseline"] = _build_planning_baseline(
                        fit_final_output,
                        planning_request,
                    )
                    finalized_fit = _finalize_success_output(fit_final_output, user_request, auto_open_report)
                    if finalized_fit.get("success"):
                        regression_error = _refinement_non_decrease_error(
                            finalized_fit,
                            refinement_context,
                            rough_budget_check,
                            precheck_warnings,
                        )
                        if regression_error:
                            return regression_error
                        return finalized_fit

            experience_parsed = _apply_budget_maximize_experience_mode(parsed, flight_cabin="economy")
            experience_budget_result, experience_agent3_output, experience_error, _ = _run_agent2_agent3(
                experience_parsed,
                refinement_context,
            )
            if not experience_error:
                experience_final_output = _clean_final_output(
                    experience_agent3_output,
                    experience_parsed,
                    experience_budget_result,
                )
                experience_budget_check = experience_final_output.get("budget_check") or {}
                experience_total_budget = float(experience_budget_check.get("total_budget_input") or 0.0)
                experience_selected_total = _selected_total_from_budget_check(experience_budget_check)
                if experience_total_budget > 0 and 0 < experience_selected_total <= experience_total_budget:
                    experience_final_output["original_request"] = user_request
                    experience_final_output["budget_maximize_experience"] = True
                    if refinement_context:
                        experience_final_output["refinement_context"] = refinement_context
                    experience_final_output["rough_budget_check"] = rough_budget_check
                    experience_final_output.setdefault("warnings", [])
                    experience_final_output["warnings"].extend(precheck_warnings)
                    experience_final_output["warnings"].append(
                        "Budget experience maximization: higher-cabin attempts exceeded the budget, so the system kept economy flights but maximized hotels, dining, activities, and final budget utilization within the cap."
                    )
                    experience_final_output["agent2_summary"] = {
                        "budget_strategy": experience_budget_result.get("budget_strategy"),
                        "cost_level": experience_budget_result.get("cost_level"),
                        "budget_status": experience_budget_result.get("budget_status"),
                        "normalized_preferences": experience_budget_result.get("normalized_preferences", {}),
                        "allocation_basis": experience_budget_result.get("allocation_basis", {}),
                        "budget_plan": experience_budget_result.get("budget_plan", {}),
                        "city_budget_plan": experience_budget_result.get("city_budget_plan", []),
                    }
                    if refinement_context:
                        experience_state, experience_status = _build_refinement_state(
                            experience_final_output,
                            refinement_context,
                        )
                        experience_final_output["refinement_state"] = experience_state
                        experience_final_output["upgrade_status"] = experience_status
                    experience_final_output["planning_baseline"] = _build_planning_baseline(
                        experience_final_output,
                        planning_request,
                    )
                    finalized_experience = _finalize_success_output(
                        experience_final_output,
                        user_request,
                        auto_open_report,
                    )
                    if finalized_experience.get("success"):
                        regression_error = _refinement_non_decrease_error(
                            finalized_experience,
                            refinement_context,
                            rough_budget_check,
                            precheck_warnings,
                        )
                        if regression_error:
                            return regression_error
                        return finalized_experience

            retry_parsed = _apply_budget_cap_mode(
                parsed,
                "fit the budget with cheap flights, budget hotels, fewer paid activities, and no comfort upgrades",
            )
            retry_budget_result, retry_agent3_output, retry_error, _ = _run_agent2_agent3(retry_parsed, refinement_context)
            if not retry_error:
                retry_final_output = _clean_final_output(retry_agent3_output, retry_parsed, retry_budget_result)
                retry_final_output["original_request"] = user_request
                if refinement_context:
                    retry_final_output["refinement_context"] = refinement_context
                retry_budget_check = retry_final_output.get("budget_check") or {}
                retry_total_budget = float(retry_budget_check.get("total_budget_input") or 0.0)
                retry_selected_total = _selected_total_from_budget_check(retry_budget_check)
                final_failed_total = retry_selected_total or final_failed_total
                if retry_total_budget > 0:
                    retry_final_output.setdefault("warnings", [])
                    retry_final_output["warnings"].append(
                        "Budget pressure adjustment: the first selected plan exceeded the budget, so the system automatically retried with cheaper flights, budget hotels, fewer paid activities, and no comfort upgrades."
                    )
                    retry_final_output["warnings"].extend(precheck_warnings)
                    retry_final_output["agent2_summary"] = {
                        "budget_strategy": retry_budget_result.get("budget_strategy"),
                        "cost_level": retry_budget_result.get("cost_level"),
                        "budget_status": retry_budget_result.get("budget_status"),
                        "normalized_preferences": retry_budget_result.get("normalized_preferences", {}),
                        "allocation_basis": retry_budget_result.get("allocation_basis", {}),
                        "budget_plan": retry_budget_result.get("budget_plan", {}),
                        "city_budget_plan": retry_budget_result.get("city_budget_plan", []),
                    }
                    if refinement_context:
                        retry_state, retry_status = _build_refinement_state(retry_final_output, refinement_context)
                        retry_final_output["refinement_state"] = retry_state
                        retry_final_output["upgrade_status"] = retry_status
                    retry_final_output["planning_baseline"] = _build_planning_baseline(
                        retry_final_output,
                        planning_request,
                    )
                    finalized_retry = _finalize_success_output(retry_final_output, user_request, auto_open_report)
                    if finalized_retry.get("success") or finalized_retry.get("stage") == "budget_feasibility":
                        regression_error = _refinement_non_decrease_error(
                            finalized_retry,
                            refinement_context,
                            rough_budget_check,
                            precheck_warnings,
                        )
                        if regression_error:
                            return regression_error
                        return finalized_retry

        final_output.setdefault("warnings", [])
        final_output["warnings"].append(
            "Selected realtime candidates exceeded the budget before itinerary assembly, so the system continued to build the minimum executable itinerary and will use that final itinerary cost for any budget guidance."
        )

    final_output["agent2_summary"] = {
        "budget_strategy": budget_result.get("budget_strategy"),
        "cost_level": budget_result.get("cost_level"),
        "budget_status": budget_result.get("budget_status"),
        "normalized_preferences": budget_result.get("normalized_preferences", {}),
        "allocation_basis": budget_result.get("allocation_basis", {}),
        "budget_plan": budget_result.get("budget_plan", {}),
        "city_budget_plan": budget_result.get("city_budget_plan", []),
    }
    final_output.setdefault("warnings", [])
    regression_error = _refinement_non_decrease_error(
        final_output,
        refinement_context,
        rough_budget_check,
        precheck_warnings,
    )
    if regression_error:
        return regression_error
    final_output["warnings"].extend(precheck_warnings)
    if refinement_context:
        refinement_state, upgrade_status = _build_refinement_state(final_output, refinement_context)
        final_output["refinement_state"] = refinement_state
        final_output["upgrade_status"] = upgrade_status
    final_output["planning_baseline"] = _build_planning_baseline(final_output, planning_request)

    return _finalize_success_output(final_output, user_request, auto_open_report)

def get_router_backend_status() -> dict:
    google_maps_ready = bool(os.getenv("GOOGLE_MAPS_API_KEY"))
    searchapi_ready = bool(os.getenv("SEARCHAPI_API_KEY"))

    return {
        "runtime": {
            "router_process_started_at": ROUTER_PROCESS_STARTED_AT,
            "router_code_version": ROUTER_CODE_VERSION,
        },
        "agent1_llm": get_llm_backend_status(),
        "agent2_llm": get_budget_agent_backend_status(),
        "agent3": {
            "provider": "searchapi_google_flights + searchapi_google_hotels + google_places",
            "google_places_configured": google_maps_ready,
            "searchapi_configured": searchapi_ready,
            "ready": google_maps_ready and searchapi_ready
        },
        "city_consumption_tier_llm": get_city_consumption_tier_backend_status(),
    }


if __name__ == "__main__":
    test_input = (
     "I want to travel  from Beijing to Chengdu 4.19-4.21 with a budget of 6000 CNY. I love local food."
    )
    result = run_budget_pipeline(test_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
