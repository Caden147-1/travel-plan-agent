from __future__ import annotations

import os
from typing import Any, Dict

import router
from destination import build_city_plan, extract_explicit_city_days
from agent3_real import Agent3Selection
from travel_requirement_parser_llm_agent import (
    _extract_explicit_city_list_from_text,
    _extract_hotel_area_from_text,
    _normalize_travel_dates,
)
from itinerary_agent import (
    _airport_role_display,
    _build_area_must_visit_fallbacks,
    _build_city_budget_map,
    _compress_flexible_timeline,
    _ensure_adjacent_event_transfers,
    _ensure_hotel_transition_events,
    _ensure_transfer_before_checkin,
    _enforce_unique_poi_visits,
    _enforce_poi_three_day_cooldown,
    _extract_daily_city_slots,
    _ensure_required_meals,
    _realign_evening_activity_transfers,
    _remove_invalid_same_site_transfers,
    _resolve_timeline_overlaps,
    _sanitize_transfer_timeline,
    _poi_dedupe_key,
    _normalize_agent3_payload,
    _filter_city_candidates_by_hotel_radius,
    _schedule_city_day,
    run_itinerary_agent,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_schedule_city_day_accepts_new_kwargs() -> None:
    day_plan: Dict[str, Any] = {
        "date": "2026-04-26",
        "city": "Tokyo",
        "day_index_in_city": 1,
        "days_in_city": 3,
        "schedule_start": "09:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Test Hotel", "lat": 35.68, "lng": 139.76},
        "used_hours": 0.0,
        "events": [],
        "warnings": [],
        "local_transport_tier": "medium",
        "local_transport_is_international": True,
    }
    city_candidates = {
        "pois": [],
        "restaurants": [],
    }
    city_budget = {"food_budget": 1000.0, "attractions_budget": 500.0, "days": 3.0}

    result = _schedule_city_day(
        day_plan=day_plan,
        city_candidates=city_candidates,
        city_budget=city_budget,
        must_visit_left=[],
        current_day_index=1,
        restaurant_last_used_day_index={},
        restaurant_cooldown_days=2,
        used_poi_keys=set(),
        used_restaurant_keys=set(),
    )

    _assert(isinstance(result, dict), "_schedule_city_day should return a day plan dict")


def test_refinement_state_marks_food_budget_ceiling() -> None:
    final_output = {
        "budget_check": {
            "selected_restaurant_total": 980.0,
            "food_budget_input": 1000.0,
            "selected_poi_total": 100.0,
            "attractions_budget_input": 500.0,
            "selected_total_transport_spend": 1000.0,
            "transport_budget_input": 3000.0,
            "selected_hotel_total_reference": 2000.0,
            "hotel_budget_input": 4000.0,
        },
        "budget_plan": {"buffer": 0.0},
    }
    state, statuses = router._build_refinement_state(
        final_output,
        {"mode": "refine", "target_categories": ["food"], "previous_refinement_state": {}},
    )

    _assert("food" in state.get("categoryCeilings", {}), "Food should be marked maxed at 98% of category budget")
    _assert(statuses and statuses[0].get("status") == "maxed", "Food upgrade status should be maxed")


def test_refinement_state_keeps_food_upgrade_open_when_under_budget() -> None:
    final_output = {
        "budget_check": {
            "selected_restaurant_total": 500.0,
            "food_budget_input": 1000.0,
        },
        "budget_plan": {"buffer": 300.0},
    }
    state, statuses = router._build_refinement_state(
        final_output,
        {"mode": "refine", "target_categories": ["food"], "previous_refinement_state": {}},
    )

    _assert("food" not in state.get("categoryCeilings", {}), "Food should not be maxed when spend is well below budget")
    _assert(statuses and statuses[0].get("status") == "upgraded", "Food upgrade should remain open under budget")
    _assert(state.get("totalBudget") == 0.0, "Refinement state should include a total budget snapshot")
    _assert(state.get("selectedTotal") == 500.0, "Refinement state should include a selected spend snapshot")


def test_refinement_uses_previous_budget_when_user_does_not_increase_again() -> None:
    parsed = {
        "budget": {"total": 80000, "currency": "CNY"},
        "preferences": {},
        "planning_signals": {},
    }
    context = router._normalize_refinement_context({
        "mode": "refine",
        "previous_request": "Plan Tokyo with 80000 CNY budget",
        "followup_message": "upgrade food again within current budget",
        "target_categories": ["food"],
        "allow_total_budget_increase": False,
        "previous_total_budget": 100000,
        "previous_selected_total": 76000,
    })

    refined = router._apply_refinement_to_parsed(parsed, context)

    _assert(refined["budget"]["total"] == 100000, "Refinement should use the last successful total budget")
    _assert(refined["refinement_context"]["previous_selected_total"] == 76000, "Previous spend baseline should be preserved")


def test_refinement_new_budget_overrides_previous_budget() -> None:
    parsed = {
        "budget": {"total": 80000, "currency": "CNY"},
        "preferences": {},
        "planning_signals": {},
    }
    context = router._normalize_refinement_context({
        "mode": "refine",
        "previous_request": "Plan Tokyo with 80000 CNY budget",
        "followup_message": "upgrade food with 120000 total budget",
        "target_categories": ["food"],
        "allow_total_budget_increase": True,
        "new_total_budget": 120000,
        "previous_total_budget": 100000,
        "previous_selected_total": 76000,
    })

    refined = router._apply_refinement_to_parsed(parsed, context)

    _assert(refined["budget"]["total"] == 120000, "Explicit new budget should become the planning budget")


def test_refinement_rejects_lower_selected_total_than_previous_plan() -> None:
    final_output = {
        "trip_summary": {"budget": {"total": 100000, "currency": "CNY"}},
        "budget_check": {
            "total_budget_input": 100000,
            "selected_total_transport_spend": 20000,
            "selected_hotel_total_reference": 25000,
            "selected_restaurant_total": 10000,
            "selected_poi_total": 5000,
        },
        "budget_plan": {"buffer": 40000},
        "warnings": [],
    }
    context = {
        "mode": "refine",
        "target_categories": ["food"],
        "previous_selected_total": 70000,
        "previous_refinement_state": {},
    }

    error = router._refinement_non_decrease_error(final_output, context, {}, [])

    _assert(error is not None, "Refinement should reject a generated plan with lower total spend")
    _assert(error.get("stage") == "refinement_non_decrease", "Lower-spend refinement should use the non-decrease stage")


def test_refinement_radius_controls_food_and_poi_filtering() -> None:
    hotel = {"name": "Hotel", "lat": 0.0, "lng": 0.0}
    near = {"name": "Near", "lat": 0.0, "lng": 0.03}
    far = {"name": "Far", "lat": 0.0, "lng": 0.09}
    city_candidates = {
        "pois": [near, far],
        "restaurants": [near, far],
    }

    strict = _filter_city_candidates_by_hotel_radius(city_candidates, hotel, activity_radius_km=5.0)
    relaxed = _filter_city_candidates_by_hotel_radius(city_candidates, hotel, activity_radius_km=15.0)

    _assert(len(strict.get("pois", [])) == 1, "5km refinement radius should keep only near POIs")
    _assert(len(strict.get("restaurants", [])) == 1, "5km refinement radius should keep only near restaurants")
    _assert(len(relaxed.get("pois", [])) == 2, "15km refinement radius should include farther POIs")
    _assert(len(relaxed.get("restaurants", [])) == 2, "15km refinement radius should include farther restaurants")


def test_refinement_state_marks_flight_and_hotel_budget_ceilings() -> None:
    final_output = {
        "budget_check": {
            "selected_total_transport_spend": 4900.0,
            "transport_budget_input": 5000.0,
            "selected_hotel_total_reference": 3920.0,
            "hotel_budget_input": 4000.0,
        },
        "budget_plan": {"buffer": 0.0},
    }
    state, statuses = router._build_refinement_state(
        final_output,
        {"mode": "refine", "target_categories": ["flight", "hotel"], "previous_refinement_state": {}},
    )

    ceilings = state.get("categoryCeilings", {})
    status_by_category = {item.get("category"): item.get("status") for item in statuses}
    _assert("flight" in ceilings, "Flight should be marked maxed at 98% of transport budget")
    _assert("hotel" in ceilings, "Hotel should be marked maxed at 98% of hotel budget")
    _assert(status_by_category.get("flight") == "maxed", "Flight upgrade status should be maxed")
    _assert(status_by_category.get("hotel") == "maxed", "Hotel upgrade status should be maxed")


def test_finalize_success_output_fails_when_itinerary_fails() -> None:
    original_run_itinerary_agent = router.run_itinerary_agent
    original_write_travel_report = router.write_travel_report
    try:
        router.run_itinerary_agent = lambda _: {
            "success": False,
            "stage": "agent4_itinerary_planning",
            "error": "synthetic itinerary failure",
        }
        router.write_travel_report = lambda *_args, **_kwargs: {
            "success": True,
            "path": "unused.html",
        }
        result = router._finalize_success_output(
            {
                "success": True,
                "stage": "agent3_real_selection",
                "trip_summary": {"budget": {"total": 15000}},
                "warnings": [],
            },
            "Plan a 5 day Tokyo trip from Shanghai for 2 people with a 15000 CNY budget.",
            auto_open_report=False,
        )
    finally:
        router.run_itinerary_agent = original_run_itinerary_agent
        router.write_travel_report = original_write_travel_report

    _assert(result.get("success") is False, "Agent4 failure must mark the whole pipeline as failed")
    _assert(result.get("stage") == "agent4_itinerary_planning", "Agent4 failure stage should be preserved")
    _assert("runtime" in result, "Runtime metadata should be attached to final output")


def test_agent4_normalize_accepts_agent3_city_budget_plan_used() -> None:
    normalized = _normalize_agent3_payload(
        {
            "trip_summary": {
                "destination": ["Paris"],
                "budget": {"total": 30000, "currency": "CNY"},
            },
            "parsed_request": {
                "country": "france",
                "preferences": {"must_visit_attractions": ["Louvre"]},
            },
            "cost_level": "high",
            "city_date_ranges": {
                "Paris": {"check_in": "2026-05-02", "check_out": "2026-05-05", "days": 3},
            },
            "city_budget_plan_used": [
                {
                    "city": "Paris",
                    "days": 3,
                    "hotel_budget": 14200,
                    "food_budget": 4000,
                    "attractions_budget": 1400,
                }
            ],
        }
    )

    _assert(normalized["country"] == "france", "Agent4 normalize should fall back to parsed_request.country")
    _assert(normalized["city_budget_plan"], "Agent4 normalize should accept Agent3 city_budget_plan_used")
    _assert(
        normalized["city_budget_plan"][0]["hotel_budget"] == 14200,
        "Agent4 should preserve city hotel budget from city_budget_plan_used",
    )
    _assert(
        normalized["must_visit_attractions"] == ["Louvre"],
        "Agent4 normalize should preserve preferences from parsed_request fallback",
    )


def test_agent4_normalize_preserves_budget_plan_for_fallback_budget_map() -> None:
    normalized = _normalize_agent3_payload(
        {
            "trip_summary": {
                "destination": ["Tokyo"],
                "budget": {"total": 80000, "currency": "CNY"},
            },
            "agent2_summary": {
                "budget_plan": {
                    "transport": 16000,
                    "hotel": 42000,
                    "food": 12000,
                    "attractions": 6000,
                    "buffer": 4000,
                }
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-06-08", "check_out": "2026-06-10", "days": 2},
                "Osaka": {"check_in": "2026-06-10", "check_out": "2026-06-14", "days": 4},
            },
        }
    )
    slots = _extract_daily_city_slots(normalized["city_date_ranges"])
    budget_map = _build_city_budget_map([], normalized["budget_plan"], slots)

    _assert(normalized["budget_plan"]["hotel"] == 42000, "Agent4 normalize should preserve Agent2 budget_plan")
    _assert(budget_map["Tokyo"]["hotel_budget"] > 0, "Budget map should distribute hotel budget to Tokyo")
    _assert(budget_map["Osaka"]["food_budget"] > 0, "Budget map should distribute food budget to Osaka")


def test_agent4_spend_up_targets_budget_ceiling() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 2,
                "budget": {"total": 20000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-06-08", "check_out": "2026-06-10", "days": 2},
            },
            "city_budget_plan_used": [
                {"city": "Tokyo", "days": 2, "hotel_budget": 12000, "food_budget": 3000, "attractions_budget": 1000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 2000, "lat": 35.67, "lng": 139.76},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Breakfast Cafe", "estimated_total_cost": 300, "recommended_visits": 1, "types": ["cafe"], "lat": 35.671, "lng": 139.761},
                {"city": "Tokyo", "name": "Lunch Sushi", "estimated_total_cost": 400, "recommended_visits": 1, "lat": 35.672, "lng": 139.762},
                {"city": "Tokyo", "name": "Dinner Steak", "estimated_total_cost": 500, "recommended_visits": 1, "lat": 35.673, "lng": 139.763},
            ],
            "selected_pois": [
                {"city": "Tokyo", "name": "Tokyo Tower", "estimated_total_cost": 100, "lat": 35.6586, "lng": 139.7454},
            ],
            "selected_all_flight_legs": [],
        }
    )

    total = (result.get("cost_rollup") or {}).get("total_selected_estimated", 0)
    _assert(result.get("success") is True, "Agent4 spend-up fixture should generate an itinerary")
    _assert(19500 <= total <= 20000, "Agent4 should spend up close to the budget ceiling without exceeding it")
    _assert(
        (result.get("cost_rollup") or {}).get("hotel_reference", 0) > 2000,
        "Agent4 should use hotel quality upgrades to close a large remaining budget gap",
    )
    _assert(
        any("Budget utilization upgrade" in warning for warning in result.get("warnings", [])),
        "Agent4 should explain when it raises quality estimates to use the budget",
    )


def test_fallback_poi_names_do_not_include_date_suffix() -> None:
    day = {
        "date": "2026-06-09",
        "city": "Tokyo",
        "day_index_in_city": 2,
    }

    name = _fallback_poi_name(day, 2)

    _assert(name == "Market And Cafe Lane in Tokyo", "Fallback POI names should not include a parenthesized date suffix")


def test_parser_and_city_plan_keep_explicit_three_city_nights() -> None:
    prompt = (
        "Plan a 7-day, 6-night multi-city trip from 6-8 to 6-14 in Japan for 1 person, "
        "departing from Shanghai, with a total budget of 80,000 CNY. "
        "The cities are Tokyo, Kyoto, and Osaka. Stay 2 nights in Tokyo, 2 nights in Kyoto, and 2 nights in Osaka."
    )
    explicit_cities = _extract_explicit_city_list_from_text(prompt)
    explicit_stays = extract_explicit_city_days(prompt, ["Tokyo", "Kyoto", "Osaka"])
    planned = build_city_plan(
        {
            "origin": "Shanghai",
            "destination": explicit_cities,
            "country": "Japan",
            "duration_days": 7,
            "travel_dates": {"start": "2026-06-08", "end": "2026-06-14"},
            "user_request": prompt,
        }
    )
    cities = planned.get("city_plan", {}).get("cities", [])

    _assert(explicit_cities == ["Tokyo", "Kyoto", "Osaka"], "Parser fallback should keep all explicit cities")
    _assert([item["city"] for item in explicit_stays] == ["Tokyo", "Kyoto", "Osaka"], "Night stays should match all explicit cities")
    _assert([item.get("days") for item in cities] == [2, 2, 2], "Explicit 2-night city stays should not be reallocated")


def test_city_plan_orders_japan_cities_by_gateway() -> None:
    original_env = os.environ.get("TRAVEL_CITY_ORDER_USE_MAP_API")
    try:
        os.environ["TRAVEL_CITY_ORDER_USE_MAP_API"] = "false"
        planned = build_city_plan(
            {
                "origin": "Shanghai",
                "destination": ["Kyoto", "Tokyo", "Osaka"],
                "country": "Japan",
                "duration_days": 6,
                "travel_dates": {},
                "user_request": "I want to visit Kyoto, Tokyo and Osaka in Japan from Shanghai.",
            }
        )
    finally:
        if original_env is None:
            os.environ.pop("TRAVEL_CITY_ORDER_USE_MAP_API", None)
        else:
            os.environ["TRAVEL_CITY_ORDER_USE_MAP_API"] = original_env

    city_plan = planned.get("city_plan", {})
    order = [item.get("city") for item in city_plan.get("cities", [])]
    route_ordering = city_plan.get("route_ordering", {})
    kyoto_meta = next((item for item in route_ordering.get("cities", []) if item.get("city") == "Kyoto"), {})

    _assert(order == ["Tokyo", "Osaka", "Kyoto"], "Gateway-aware Japan order should be Tokyo -> Osaka -> Kyoto")
    _assert(route_ordering.get("applied") is True, "Route ordering metadata should record automatic optimization")
    _assert(kyoto_meta.get("gateway_city") == "Osaka", "Kyoto should borrow Osaka as its gateway airport city")


def test_city_plan_preserves_explicit_sequence_markers() -> None:
    original_env = os.environ.get("TRAVEL_CITY_ORDER_USE_MAP_API")
    try:
        os.environ["TRAVEL_CITY_ORDER_USE_MAP_API"] = "false"
        planned = build_city_plan(
            {
                "origin": "Shanghai",
                "destination": ["Kyoto", "Tokyo", "Osaka"],
                "country": "Japan",
                "duration_days": 6,
                "travel_dates": {},
                "user_request": "First Kyoto, then Tokyo, finally Osaka from Shanghai.",
            }
        )
    finally:
        if original_env is None:
            os.environ.pop("TRAVEL_CITY_ORDER_USE_MAP_API", None)
        else:
            os.environ["TRAVEL_CITY_ORDER_USE_MAP_API"] = original_env

    city_plan = planned.get("city_plan", {})
    order = [item.get("city") for item in city_plan.get("cities", [])]
    route_ordering = city_plan.get("route_ordering", {})

    _assert(order == ["Kyoto", "Tokyo", "Osaka"], "Explicit first/then/finally sequence should be preserved")
    _assert(route_ordering.get("applied") is False, "Explicit sequence should disable automatic route ordering")
    _assert(route_ordering.get("reason") == "explicit_user_sequence_detected", "Route metadata should explain why sorting was skipped")


def test_hotel_area_parser_extracts_only_lodging_area() -> None:
    _assert(
        _extract_hotel_area_from_text("I want a hotel near Kyoto Station for 3 nights") == "Kyoto Station",
        "English hotel-area expression should be extracted",
    )
    _assert(
        _extract_hotel_area_from_text("酒店在大阪梅田附近，预算80000") == "大阪梅田",
        "Chinese hotel-area expression should be extracted",
    )
    _assert(
        _extract_hotel_area_from_text("I want to visit Ginza and Tokyo Skytree") is None,
        "Attraction mentions alone should not become hotel_area",
    )


def test_last_city_checkout_date_becomes_departure_day_slot() -> None:
    slots = _extract_daily_city_slots(
        {
            "Tokyo": {"check_in": "2026-06-08", "check_out": "2026-06-10", "days": 2, "nights": 2},
            "Kyoto": {"check_in": "2026-06-10", "check_out": "2026-06-12", "days": 2, "nights": 2},
            "Osaka": {"check_in": "2026-06-12", "check_out": "2026-06-14", "days": 2, "nights": 2},
        }
    )
    dates = [(slot["date"], slot["city"]) for slot in slots]

    _assert(len(slots) == 7, "7-day date range should expand to seven day cards")
    _assert(("2026-06-14", "Osaka") in dates, "Last city checkout date should be the final departure day")
    _assert(("2026-06-12", "Osaka") in dates, "Transition date should belong to the destination city")


def test_final_transfer_pass_adds_transfer_without_walk_filler() -> None:
    day = {
        "date": "2026-06-09",
        "city": "Tokyo",
        "hotel": {"name": "Hotel"},
        "local_transport_tier": "medium",
        "events": [
            {
                "type": "poi",
                "name": "Tokyo Tower",
                "start": "09:00",
                "end": "10:00",
                "duration_h": 1.0,
                "meta": {"lat": 35.6586, "lng": 139.7454, "poi_name": "Tokyo Tower"},
            },
            {
                "type": "poi",
                "name": "Ueno Park",
                "start": "10:00",
                "end": "11:30",
                "duration_h": 1.5,
                "meta": {"lat": 35.7156, "lng": 139.7745, "poi_name": "Ueno Park"},
            },
        ],
        "warnings": [],
    }

    _ensure_adjacent_event_transfers(day, fill_idle_gaps=False)

    names = [event.get("name") for event in day["events"]]
    _assert(any("Transfer Tokyo Tower -> Ueno Park" in name for name in names), "Final pass should add transfer between different real POIs")
    _assert(not any("Local area walk" in name for name in names), "Final transfer pass should not add new area-walk fillers")


def test_airport_display_infers_nearby_airport_from_airport_city() -> None:
    label = _airport_role_display(
        {
            "to_city": "Kyoto",
            "arrival_airport": "Osaka Itami International Airport",
            "arrival_airport_code": "ITM",
            "to_airport_candidate_source": "city_airport",
        },
        "arrival",
    )
    _assert("nearby airport for Kyoto" in label, "Kyoto itinerary should label Osaka airports as nearby airports")


def test_missing_city_hotel_gets_fallback_hotel_and_checkin() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 2,
                "budget": {"total": 30000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-06-08", "check_out": "2026-06-10", "days": 2, "nights": 2},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 2, "food_budget": 2000, "attractions_budget": 1000, "hotel_budget": 6000},
            ],
            "selected_hotels": [],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Hinomaru", "estimated_meal_per_person": 150, "lat": 35.6938, "lng": 139.7086},
                {"city": "Tokyo", "name": "Morning Cafe", "estimated_meal_per_person": 80, "types": ["cafe"], "lat": 35.694, "lng": 139.709},
            ],
            "selected_pois": [
                {"city": "Tokyo", "name": "Sensō-ji", "estimated_total_cost": 80, "lat": 35.7148, "lng": 139.8027},
            ],
            "selected_all_flight_legs": [
                {
                    "leg_type": "outbound",
                    "from_city": "Shanghai",
                    "to_city": "Tokyo",
                    "departure_date": "2026-06-08",
                    "to_airport": "HND",
                    "selected_flights": [
                        {
                            "departure_time": "13:35",
                            "arrival_time": "17:45",
                            "duration_minutes": 250,
                            "price_total": 3000,
                            "arrival_airport_code": "HND",
                            "arrival_airport": "Haneda Airport",
                        }
                    ],
                }
            ],
        }
    )
    first_day = result.get("daily_plan", [])[0]
    hotel = first_day.get("hotel") or {}
    event_types = [event.get("type") for event in first_day.get("events", [])]
    event_names = [event.get("name", "") for event in first_day.get("events", [])]

    _assert("Tokyo itinerary base hotel" == hotel.get("name"), "Agent4 should create a fallback hotel when Agent3 returns no city hotel")
    _assert("checkin" in event_types, "Fallback hotel should still get check-in")
    _assert(any("Haneda Airport" in name and "Tokyo itinerary base hotel" in name for name in event_names), "Arrival should transfer to fallback hotel before city activities")


def test_distinct_synthetic_pois_get_transfer_even_with_close_estimated_coords() -> None:
    day = {
        "date": "2026-06-11",
        "city": "Kyoto",
        "hotel": {"name": "Kyoto Hotel"},
        "local_transport_tier": "medium",
        "events": [
            {
                "type": "poi",
                "name": "Museum District Walk in Kyoto (2026-06-11)",
                "start": "14:45",
                "end": "16:15",
                "duration_h": 1.5,
                "meta": {"poi_name": "Museum District Walk in Kyoto (2026-06-11)", "lat": 35.0000, "lng": 135.0000},
            },
            {
                "type": "poi",
                "name": "Market And Cafe Lane in Kyoto (2026-06-11)",
                "start": "16:15",
                "end": "17:45",
                "duration_h": 1.5,
                "meta": {"poi_name": "Market And Cafe Lane in Kyoto (2026-06-11)", "lat": 35.0001, "lng": 135.0001},
            },
        ],
        "warnings": [],
    }

    _ensure_adjacent_event_transfers(day, fill_idle_gaps=False)

    _assert(
        any(event.get("type") == "transfer" and "Museum District Walk" in event.get("name", "") and "Market And Cafe Lane" in event.get("name", "") for event in day["events"]),
        "Different synthetic POI names should still get a transfer even when estimated coordinates are close",
    )


def test_evening_transfer_starts_after_dinner_end() -> None:
    day = {
        "date": "2026-06-12",
        "city": "Osaka",
        "events": [
            {"type": "meal", "name": "Dinner", "start": "18:25", "end": "19:25", "duration_h": 1.0, "meta": {"meal_slot": "dinner"}},
            {"type": "transfer", "name": "Transfer Dinner -> Umeda Sky Building", "start": "19:20", "end": "19:35", "duration_h": 0.25, "meta": {"reason": "evening_activity_transfer"}},
            {"type": "poi", "name": "Umeda Sky Building", "start": "19:35", "end": "21:00", "duration_h": 1.42, "meta": {"reason": "evening_after_dinner"}},
        ],
        "warnings": [],
    }

    _realign_evening_activity_transfers(day)
    transfer = next(event for event in day["events"] if event.get("type") == "transfer")
    poi = next(event for event in day["events"] if event.get("type") == "poi")

    _assert(transfer.get("start") == "19:25", "Evening transfer should start after dinner ends")
    _assert(poi.get("start") == transfer.get("end"), "Evening POI should start after the transfer")


def test_checkin_after_poi_overlap_is_shifted_after_previous_event() -> None:
    day = {
        "date": "2026-06-10",
        "city": "Kyoto",
        "schedule_start": "09:00",
        "schedule_end": "20:30",
        "events": [
            {"type": "poi", "name": "Fushimi Inari Taisha", "start": "12:59", "end": "14:29", "duration_h": 1.5},
            {"type": "checkin", "name": "Check-in Higashiyama Shikikaboku", "start": "13:44", "end": "13:59", "duration_h": 0.25},
            {"type": "poi", "name": "Local area walk before Nijo Castle", "start": "14:29", "end": "15:29", "duration_h": 1.0},
        ],
    }

    _resolve_timeline_overlaps(day)
    events = day["events"]
    first_poi = next(event for event in events if event.get("name") == "Fushimi Inari Taisha")
    checkin = next(event for event in events if event.get("type") == "checkin")
    walk = next(event for event in events if event.get("name") == "Local area walk before Nijo Castle")

    _assert(checkin.get("start") >= first_poi.get("end"), "Check-in must not overlap the previous POI")
    _assert(walk.get("start") >= checkin.get("end"), "Following flexible activity must move after shifted check-in")


def test_stale_evening_transfer_is_removed_and_rebuilt() -> None:
    day = {
        "date": "2026-06-12",
        "city": "Tokyo",
        "hotel": {"name": "Tokyo Test Hotel", "lat": 35.67, "lng": 139.76},
        "local_transport_tier": "medium",
        "local_transport_is_international": True,
        "events": [
            {
                "type": "poi",
                "name": "Afternoon Garden",
                "start": "17:00",
                "end": "18:00",
                "duration_h": 1.0,
                "meta": {"poi_name": "Afternoon Garden", "lat": 35.6600, "lng": 139.7500},
            },
            {
                "type": "transfer",
                "name": "Transfer Narisawa -> Tokyo Skytree",
                "start": "19:00",
                "end": "19:15",
                "duration_h": 0.25,
                "meta": {"from": "Narisawa", "to": "Tokyo Skytree", "reason": "evening_activity_transfer"},
            },
            {
                "type": "meal",
                "name": "Narisawa",
                "start": "19:18",
                "end": "20:18",
                "duration_h": 1.0,
                "meta": {"meal_slot": "dinner", "restaurant_name": "Narisawa", "lat": 35.6620, "lng": 139.7200},
            },
            {
                "type": "poi",
                "name": "Tokyo Skytree",
                "start": "20:30",
                "end": "21:00",
                "duration_h": 0.5,
                "meta": {"reason": "evening_after_dinner", "poi_name": "Tokyo Skytree", "lat": 35.7100, "lng": 139.8107},
            },
        ],
        "warnings": [],
    }

    _sanitize_transfer_timeline(day)
    _ensure_adjacent_event_transfers(day, fill_idle_gaps=False)

    transfers = [event for event in day["events"] if event.get("type") == "transfer"]
    stale = [event for event in transfers if event.get("start") == "19:00" and "Narisawa -> Tokyo Skytree" in event.get("name", "")]
    rebuilt = next(
        (
            event for event in transfers
            if (event.get("meta") or {}).get("from") == "Narisawa"
            and (event.get("meta") or {}).get("to") == "Tokyo Skytree"
        ),
        None,
    )

    _assert(not stale, "Stale evening transfer before dinner should be removed")
    _assert(rebuilt is not None, "Adjacent transfer should be rebuilt from dinner to evening POI")
    _assert(rebuilt.get("start") >= "20:18", "Rebuilt evening transfer should start after dinner ends")


def test_checkin_transfer_after_drive_does_not_overlap_drive() -> None:
    day = {
        "date": "2026-06-12",
        "city": "Osaka",
        "hotel": {"name": "Conrad Osaka", "lat": 34.6933, "lng": 135.5016},
        "local_transport_tier": "medium",
        "events": [
            {"type": "drive", "name": "Drive Kyoto -> Osaka", "start": "08:15", "end": "08:58", "duration_h": 0.72, "meta": {"lat": 34.6938, "lng": 135.5011}},
            {"type": "checkin", "name": "Check-in Conrad Osaka", "start": "08:58", "end": "09:13", "duration_h": 0.25, "meta": {"lat": 34.6933, "lng": 135.5016}},
        ],
        "warnings": [],
    }

    _ensure_transfer_before_checkin(day)
    drive = next(event for event in day["events"] if event.get("type") == "drive")
    transfer = next(event for event in day["events"] if event.get("type") == "transfer")
    checkin = next(event for event in day["events"] if event.get("type") == "checkin")

    _assert(transfer.get("start") >= drive.get("end"), "Transfer from intercity drive to hotel must not overlap the drive")
    _assert(checkin.get("start") >= transfer.get("end"), "Check-in must start after the hotel transfer")


def test_agent3_drive_under_4h_skips_flight_selection() -> None:
    selector = Agent3Selection({})
    leg = {
        "from_city": "City A",
        "to_city": "City B",
        "from_airport": "AAA",
        "to_airport": "BBB",
        "direct_drive_hours": 3.9,
    }

    skip, hours, reason = selector.should_skip_flight_search(leg)

    _assert(skip is True, "Agent3 should skip flight search when city-to-city drive is within 4h")
    _assert(hours == 3.9, "Agent3 should return the direct drive hours used for the decision")
    _assert(reason == "direct_drive_under_4h", "Agent3 should expose the 4h ground-transport rule reason")
    _assert(
        selector._should_select_direct_drive(leg, f"skipped_{reason}") is True,
        "Agent3 should select direct drive when the 4h rule skipped flight search",
    )


def test_agent3_drive_over_4h_allows_flight_selection() -> None:
    selector = Agent3Selection({})
    leg = {
        "from_city": "City A",
        "to_city": "City B",
        "from_airport": "AAA",
        "to_airport": "BBB",
        "direct_drive_hours": 4.1,
    }

    skip, _hours, reason = selector.should_skip_flight_search(leg)

    _assert(skip is False, "Agent3 should not skip flight search solely by drive time when drive exceeds 4h")
    _assert(reason != "direct_drive_under_4h", "Drive over 4h should not use the under-4h skip reason")


def test_agent3_known_city_pair_over_4h_does_not_skip_flights() -> None:
    selector = Agent3Selection({})
    leg = {
        "from_city": "Tokyo",
        "to_city": "Osaka",
        "from_airport": "HND",
        "to_airport": "KIX",
        "direct_drive_hours": 5.5,
    }

    _assert(
        selector._leg_is_short_distance(leg) is False,
        "Known city-pair shortcut must not override the 4h ground-transport rule",
    )


def _agent3_budget_spend_up_fixture() -> Dict[str, Any]:
    city_budget_plan = [
        {
            "city": "Tokyo",
            "days": 4,
            "hotel_budget": 60000,
            "food_budget": 12000,
            "attractions_budget": 4000,
        }
    ]
    return {
        "parsed_request": {
            "origin": "",
            "destination": ["Tokyo"],
            "country": "Japan",
            "num_people": 1,
            "duration_days": 4,
            "travel_dates": {"start": "2026-06-08", "end": "2026-06-11"},
            "budget": {"total": 80000, "currency": "CNY"},
            "preferences": {"hotel_area": "Ginza"},
            "city_plan": {"mode": "user_specified", "cities": [{"city": "Tokyo", "days": 4}]},
        },
        "budget_result": {
            "city_budget_plan": city_budget_plan,
            "budget_plan": {"transport": 4000, "hotel": 60000, "food": 12000, "attractions": 4000, "buffer": 0},
            "budget_strategy": "balanced",
        },
        "city_budget_plan": city_budget_plan,
        "city_date_ranges": {"Tokyo": {"check_in": "2026-06-08", "check_out": "2026-06-12", "days": 4, "nights": 4}},
    }


def test_agent3_budget_utilization_fields_and_spend_up() -> None:
    selector = Agent3Selection(_agent3_budget_spend_up_fixture())
    selector._build_route_plan = lambda: {"flight_legs": []}
    selector.fetch_hotel_candidates = lambda city, city_days, max_results=20: (
        [
            {"city": city, "name": "Ginza Test Hotel A", "rating": 4.7, "stars": 4.5, "lat": 35.67, "lng": 139.76},
            {"city": city, "name": "Ginza Test Hotel B", "rating": 4.6, "stars": 4.5, "lat": 35.68, "lng": 139.77},
        ],
        "mock_hotels",
    )
    selector.fetch_restaurant_candidates = lambda city, max_results=20: [
        {"city": city, "name": f"Restaurant {idx}", "rating": 4.5, "price_level": "PRICE_LEVEL_MODERATE"}
        for idx in range(10)
    ]
    selector.fetch_poi_candidates = lambda city, max_results=20: [
        {"city": city, "name": f"POI {idx}", "rating": 4.4, "price_level": "PRICE_LEVEL_MODERATE"}
        for idx in range(8)
    ]
    selector._get_restaurant_tier = lambda restaurant: {"tier_price_per_person": 300, "source": "mock"}

    result = selector.generate_shortlist()
    budget_check = result.get("budget_check", {})

    _assert(result.get("success") is True, "Agent3 spend-up fixture should generate a shortlist")
    _assert("budget_utilization_ratio" in budget_check, "Budget check should expose utilization ratio")
    _assert("budget_utilization_status" in budget_check, "Budget check should expose utilization status")
    _assert(
        0.75 <= budget_check.get("budget_utilization_ratio", 0) <= 0.90,
        "Non-hard-cap selection should spend toward 75%-90% of the total budget",
    )
    _assert(
        budget_check.get("budget_utilization_status") == "target_met",
        "Spend-up fixture should hit the target utilization band",
    )


def test_hotel_area_query_fallback_metadata() -> None:
    selector = Agent3Selection(_agent3_budget_spend_up_fixture())
    calls: list[tuple[str, str]] = []

    def fake_search(city: str, check_in_date: str, check_out_date: str, max_results: int = 10, hotel_area: str | None = None) -> list[Dict[str, Any]]:
        calls.append(("area" if hotel_area else "city", hotel_area or city))
        if hotel_area:
            return []
        return [
            {
                "city": city,
                "name": "Tokyo City-Level Hotel",
                "rating": 4.4,
                "stars": 4.0,
                "estimated_total_cost": 1000,
            }
        ]

    selector.search_hotels_searchapi = fake_search
    selector.fetch_hotel_candidates_from_google_places = lambda city, max_results=20, hotel_area=None: []

    hotels, provider = selector.fetch_hotel_candidates("Tokyo", city_days=4, max_results=5)

    _assert(calls[0] == ("area", "Ginza"), "Hotel search should try requested area before city-level search")
    _assert(provider == "searchapi_google_hotels", "City-level SearchAPI fallback should still return usable hotels")
    _assert(hotels and hotels[0].get("hotel_area_fallback") is True, "Fallback hotels should carry hotel_area_fallback metadata")
    _assert("Tokyo" in selector.hotel_area_fallbacks, "Fallback should be recorded in selection debug metadata")


def test_hotel_area_query_failure_reason_is_preserved() -> None:
    selector = Agent3Selection(_agent3_budget_spend_up_fixture())

    def fake_search(city: str, check_in_date: str, check_out_date: str, max_results: int = 10, hotel_area: str | None = None) -> list[Dict[str, Any]]:
        if hotel_area:
            raise RuntimeError("area search unavailable")
        return [{"city": city, "name": "Tokyo City-Level Hotel", "rating": 4.4, "stars": 4.0, "estimated_total_cost": 1000}]

    selector.search_hotels_searchapi = fake_search
    selector.fetch_hotel_candidates_from_google_places = lambda city, max_results=20, hotel_area=None: []

    hotels, _provider = selector.fetch_hotel_candidates("Tokyo", city_days=4, max_results=5)

    _assert(hotels, "City-level fallback should still return hotels after an area-search failure")
    _assert(
        selector.hotel_area_fallbacks["Tokyo"].get("reason") == "area_search_failed",
        "Fallback metadata should distinguish API/search failure from no area results",
    )


def test_unknown_multi_city_hotel_area_applies_to_first_city() -> None:
    payload = _agent3_budget_spend_up_fixture()
    payload["parsed_request"]["destination"] = ["Seoul", "Busan"]
    payload["parsed_request"]["preferences"] = {"hotel_area": "Hongdae"}
    payload["parsed_request"]["city_plan"] = {
        "mode": "user_specified",
        "cities": [{"city": "Seoul", "days": 2}, {"city": "Busan", "days": 2}],
    }
    payload["city_budget_plan"] = [
        {"city": "Seoul", "days": 2, "hotel_budget": 10000, "food_budget": 3000, "attractions_budget": 1000},
        {"city": "Busan", "days": 2, "hotel_budget": 10000, "food_budget": 3000, "attractions_budget": 1000},
    ]
    payload["budget_result"]["city_budget_plan"] = payload["city_budget_plan"]
    selector = Agent3Selection(payload)

    _assert(selector._hotel_area_for_city("Seoul") == "Hongdae", "Unknown explicit hotel area should apply to the first stay city")
    _assert(selector._hotel_area_for_city("Busan") is None, "Unknown explicit hotel area should not be applied to every city")


def test_realtime_hotel_cost_spend_up_and_total_budget_cap() -> None:
    selector = Agent3Selection(_agent3_budget_spend_up_fixture())
    realtime_hotel = {
        "city": "Tokyo",
        "name": "Realtime Hotel",
        "realtime_total_cost": 2000,
        "realtime_nightly_per_room": 500,
        "rating": 4.7,
        "stars": 4.5,
    }

    estimated = selector._estimate_hotel_cost(dict(realtime_hotel), city_days=4, city_nights=4)
    _assert(estimated.get("hotel_cost_spend_up") is True, "Realtime-priced hotels should also participate in spend-up")
    _assert(estimated.get("estimated_total_cost") == 51000.0, "Realtime hotel spend-up should target 85% of city hotel budget")

    # Exercise the public flow with network calls mocked out; total selected
    # would exceed 80,000 without the final spend-up cap.
    cap_payload = _agent3_budget_spend_up_fixture()
    cap_payload["city_budget_plan"][0]["hotel_budget"] = 90000
    cap_payload["city_budget_plan"][0]["food_budget"] = 30000
    cap_payload["budget_result"]["city_budget_plan"] = cap_payload["city_budget_plan"]
    cap_payload["budget_result"]["budget_plan"] = {
        "transport": 0,
        "hotel": 90000,
        "food": 30000,
        "attractions": 0,
        "buffer": -40000,
    }
    selector = Agent3Selection(cap_payload)
    selector._build_route_plan = lambda: {"flight_legs": []}
    selector.fetch_hotel_candidates = lambda city, city_days, max_results=20: ([dict(realtime_hotel)], "mock_hotels")
    selector.fetch_restaurant_candidates = lambda city, max_results=20: [
        {"city": city, "name": f"Restaurant {idx}", "rating": 4.5, "price_level": "PRICE_LEVEL_EXPENSIVE"}
        for idx in range(10)
    ]
    selector.fetch_poi_candidates = lambda city, max_results=20: []
    selector._get_restaurant_tier = lambda restaurant: {"tier_price_per_person": 1800, "source": "mock"}
    result = selector.generate_shortlist()

    _assert(
        result["budget_check"]["overall_selected_total"] <= result["budget_check"]["total_budget_input"],
        "Final spend-up cap should keep selected total within the user budget when reducible upgrades caused the overage",
    )


def test_yearless_numeric_date_range_overrides_wrong_llm_dates() -> None:
    dates = _normalize_travel_dates(
        {
            "start": "2026-06-01",
            "end": "2026-06-08",
            "text": "6-1 to 6-8",
        },
        user_request=(
            "Plan a 7-day, 6-night multi-city trip from 6-8 to 6-14 in Japan "
            "for 1 person, departing from Shanghai."
        ),
    )

    _assert(dates["start"] == "2026-06-08", "Parser fallback should preserve the explicit 6-8 start date from raw text")
    _assert(dates["end"] == "2026-06-14", "Parser fallback should preserve the explicit 6-14 end date from raw text")


def _late_arrival_payload(departure_time: str, arrival_time: str) -> Dict[str, Any]:
    return {
        "trip_summary": {
            "origin": "Shanghai",
            "destination": ["Tokyo"],
            "country": "japan",
            "num_people": 1,
            "duration_days": 1,
            "budget": {"total": 15000, "currency": "CNY"},
            "preferences": {},
        },
        "city_date_ranges": {
            "Tokyo": {"check_in": "2026-05-01", "check_out": "2026-05-02", "days": 1},
        },
        "city_budget_plan": [
            {"city": "Tokyo", "days": 1, "food_budget": 600, "attractions_budget": 500, "hotel_budget": 1000},
        ],
        "selected_hotels": [
            {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 800},
        ],
        "selected_restaurants": [
            {"city": "Tokyo", "name": "Late Ramen", "estimated_meal_per_person": 80},
        ],
        "selected_pois": [],
        "selected_all_flight_legs": [
            {
                "leg_type": "outbound",
                "from_city": "Shanghai",
                "to_city": "Tokyo",
                "departure_date": "2026-05-01",
                "selected_flights": [
                    {
                        "departure_time": departure_time,
                        "arrival_time": arrival_time,
                        "duration_minutes": 210,
                        "price_total": 2000,
                        "currency": "CNY",
                    }
                ],
            }
        ],
    }


def test_flight_covering_dinner_window_assumes_onboard_dinner() -> None:
    result = run_itinerary_agent(_late_arrival_payload("16:00", "19:30"))

    _assert(result.get("success") is True, "Late arrival itinerary should still be generated")
    day = result.get("daily_plan", [])[0]
    dinner_events = [
        event
        for event in day.get("events", [])
        if event.get("type") == "meal" and (event.get("meta") or {}).get("meal_slot") == "dinner"
    ]
    _assert(not dinner_events, "Flight covering dinner time should not add a separate restaurant dinner")
    _assert(
        any("Dinner treated as covered by the selected flight" in warning for warning in day.get("warnings", [])),
        "Flight dinner assumption should be visible in warnings",
    )


def test_late_arrival_after_non_dinner_flight_gets_dinner() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 2,
                "budget": {"total": 15000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-01", "check_out": "2026-05-02", "days": 1},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 1, "food_budget": 600, "attractions_budget": 500, "hotel_budget": 1000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 800},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Late Ramen", "estimated_meal_per_person": 80},
            ],
            "selected_pois": [],
            "selected_all_flight_legs": [
                {
                    "leg_type": "outbound",
                    "from_city": "Shanghai",
                    "to_city": "Tokyo",
                    "departure_date": "2026-05-01",
                    "selected_flights": [
                        {
                            "departure_time": "14:00",
                            "arrival_time": "17:30",
                            "duration_minutes": 210,
                            "price_total": 2000,
                            "currency": "CNY",
                        }
                    ],
                }
            ],
        }
    )

    _assert(result.get("success") is True, "Late arrival itinerary should still be generated")
    day = result.get("daily_plan", [])[0]
    dinner_events = [
        event
        for event in day.get("events", [])
        if event.get("type") == "meal" and (event.get("meta") or {}).get("meal_slot") == "dinner"
    ]
    _assert(dinner_events, "Late arrival day should include a dinner meal")


def test_hotel_switch_gets_checkout_checkin_and_transfer() -> None:
    days = [
        {
            "date": "2026-05-01",
            "city": "Tokyo",
            "schedule_start": "09:00",
            "schedule_end": "20:30",
            "hotel": {"name": "Hotel A", "lat": None, "lng": None},
            "events": [],
            "used_hours": 0.0,
            "transport_cost_estimated": 0.0,
            "warnings": [],
            "local_transport_tier": "medium",
        },
        {
            "date": "2026-05-02",
            "city": "Tokyo",
            "schedule_start": "09:00",
            "schedule_end": "20:30",
            "hotel": {"name": "Hotel B", "lat": None, "lng": None},
            "events": [],
            "used_hours": 0.0,
            "transport_cost_estimated": 0.0,
            "warnings": [],
            "local_transport_tier": "medium",
        },
    ]

    _ensure_hotel_transition_events(days)

    day2_events = days[1].get("events", [])
    day2_types = [event.get("type") for event in day2_events]
    day2_checkout_names = [
        event.get("name", "")
        for event in day2_events
        if event.get("type") == "checkout"
    ]
    switch_transfers = [
        event
        for event in day2_events
        if event.get("type") == "transfer" and (event.get("meta") or {}).get("reason") == "hotel_switch"
    ]

    _assert(any("Hotel A" in name for name in day2_checkout_names), "Switch day should checkout from the previous night's hotel")
    _assert("checkin" in day2_types, "New hotel should get checkin after switching hotels")
    _assert(switch_transfers, "Hotel switch should include transfer from old hotel to new hotel")


def test_overnight_outbound_has_no_prearrival_checkin() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 2,
                "budget": {"total": 20000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-01", "check_out": "2026-05-03", "days": 2},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 2, "food_budget": 800, "attractions_budget": 600, "hotel_budget": 2000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 1600},
            ],
            "selected_restaurants": [],
            "selected_pois": [],
            "selected_all_flight_legs": [
                {
                    "leg_type": "outbound",
                    "from_city": "Shanghai",
                    "to_city": "Tokyo",
                    "departure_date": "2026-05-01",
                    "selected_flights": [
                        {
                            "departure_time": "22:05",
                            "arrival_time": "14:20",
                            "duration_minutes": 975,
                            "price_total": 3000,
                            "currency": "CNY",
                        }
                    ],
                }
            ],
        }
    )

    _assert(result.get("success") is True, "Overnight outbound itinerary should be generated")
    first_day = result.get("daily_plan", [])[0]
    first_day_types = [event.get("type") for event in first_day.get("events", [])]
    _assert("checkin" not in first_day_types, "Pre-arrival day should not check in at the destination hotel")


def test_overnight_afternoon_arrival_gets_dinner() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 2,
                "budget": {"total": 20000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-01", "check_out": "2026-05-03", "days": 2},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 2, "food_budget": 800, "attractions_budget": 600, "hotel_budget": 2000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 1600},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Arrival Dinner", "estimated_meal_per_person": 80},
            ],
            "selected_pois": [],
            "selected_all_flight_legs": [
                {
                    "leg_type": "outbound",
                    "from_city": "Shanghai",
                    "to_city": "Tokyo",
                    "departure_date": "2026-05-01",
                    "selected_flights": [
                        {
                            "departure_time": "22:05",
                            "arrival_time": "14:20",
                            "duration_minutes": 975,
                            "price_total": 3000,
                            "currency": "CNY",
                        }
                    ],
                }
            ],
        }
    )

    _assert(result.get("success") is True, "Overnight afternoon-arrival itinerary should be generated")
    arrival_day = result.get("daily_plan", [])[1]
    dinner_events = [
        event
        for event in arrival_day.get("events", [])
        if event.get("type") == "meal" and (event.get("meta") or {}).get("meal_slot") == "dinner"
    ]
    _assert(dinner_events, "Afternoon arrival recovery day should still get dinner if flight does not cover dinner")


def test_arrival_day_has_airport_transfer_and_hotel_cost() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 2,
                "budget": {"total": 20000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-01", "check_out": "2026-05-03", "days": 2},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 2, "food_budget": 800, "attractions_budget": 600, "hotel_budget": 2000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 1600},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Arrival Dinner", "estimated_meal_per_person": 80},
            ],
            "selected_pois": [],
            "selected_all_flight_legs": [
                {
                    "leg_type": "outbound",
                    "from_city": "Shanghai",
                    "to_city": "Tokyo",
                    "departure_date": "2026-05-01",
                    "selected_flights": [
                        {
                            "departure_time": "22:05",
                            "arrival_time": "14:20",
                            "duration_minutes": 975,
                            "price_total": 3000,
                            "currency": "CNY",
                        }
                    ],
                }
            ],
        }
    )

    arrival_day = result.get("daily_plan", [])[1]
    arrival_transfers = [
        event
        for event in arrival_day.get("events", [])
        if event.get("type") == "transfer" and (event.get("meta") or {}).get("reason") == "arrival_to_hotel"
    ]
    _assert(arrival_transfers, "Arrival day should include airport-to-hotel transfer")
    _assert(arrival_transfers[0].get("start") == "14:50", "Airport transfer should start after 30 minutes to exit the airport")
    _assert(
        (result.get("cost_rollup") or {}).get("hotel_reference", 0) > 0,
        "Hotel reference cost should be included in summary rollup",
    )


def test_nearby_airport_arrival_is_visible_in_itinerary() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Kyoto"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 2,
                "budget": {"total": 20000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Kyoto": {"check_in": "2026-05-01", "check_out": "2026-05-03", "days": 2},
            },
            "city_budget_plan": [
                {"city": "Kyoto", "days": 2, "food_budget": 800, "attractions_budget": 600, "hotel_budget": 2000},
            ],
            "selected_hotels": [
                {"city": "Kyoto", "name": "Kyoto Test Hotel", "estimated_total_cost": 1600},
            ],
            "selected_restaurants": [
                {"city": "Kyoto", "name": "Arrival Dinner", "estimated_meal_per_person": 80},
            ],
            "selected_pois": [],
            "selected_all_flight_legs": [
                {
                    "leg_type": "outbound",
                    "from_city": "Shanghai",
                    "to_city": "Kyoto",
                    "departure_date": "2026-05-01",
                    "to_airport": "KIX",
                    "to_airport_candidate_source": "nearby_airport",
                    "selected_flights": [
                        {
                            "departure_time": "10:00",
                            "arrival_time": "14:00",
                            "duration_minutes": 240,
                            "arrival_code": "KIX",
                            "price_total": 3000,
                            "currency": "CNY",
                            "ground_from_arrival_airport_hours": 1.25,
                        }
                    ],
                }
            ],
        }
    )

    day = result.get("daily_plan", [])[0]
    transfer_names = [
        event.get("name", "")
        for event in day.get("events", [])
        if event.get("type") == "transfer"
    ]
    _assert(
        any("KIX Airport (nearby airport for Kyoto)" in name for name in transfer_names),
        "Nearby airport used for a no-airport city should be visible in the transfer itinerary",
    )
    warnings = " ".join(day.get("warnings", []))
    _assert("nearby airport for Kyoto" in warnings, "Nearby airport fallback should be surfaced in warnings")


def test_router_preserves_nearby_airport_route_fields() -> None:
    cleaned = router._clean_flight_leg(
        {
            "leg_type": "outbound",
            "from_city": "Shanghai",
            "to_city": "Kyoto",
            "from_airport": "PVG",
            "to_airport": "KIX",
            "candidate_from_airports": ["PVG"],
            "candidate_to_airports": ["KIX", "ITM"],
            "from_airport_candidate_source": "city_airport",
            "to_airport_candidate_source": "nearby_airport",
            "selected_flights": [],
        }
    )

    _assert(cleaned.get("to_airport") == "KIX", "Router should preserve selected nearby airport")
    _assert(
        cleaned.get("to_airport_candidate_source") == "nearby_airport",
        "Router should preserve nearby airport candidate source for Agent4",
    )


def test_inbound_airport_transfer_arrives_90_minutes_early() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 1,
                "budget": {"total": 20000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-03", "check_out": "2026-05-05", "days": 2},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 2, "food_budget": 800, "attractions_budget": 600, "hotel_budget": 2000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 1600},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Breakfast Cafe", "estimated_meal_per_person": 80},
            ],
            "selected_pois": [],
            "selected_all_flight_legs": [
                {
                    "leg_type": "inbound",
                    "from_city": "Tokyo",
                    "to_city": "Shanghai",
                    "departure_date": "2026-05-04",
                    "selected_flights": [
                        {
                            "departure_time": "14:30",
                            "arrival_time": "17:30",
                            "duration_minutes": 180,
                            "price_total": 3000,
                            "currency": "CNY",
                        }
                    ],
                }
            ],
        }
    )

    day = result.get("daily_plan", [])[1]
    airport_transfers = [
        event
        for event in day.get("events", [])
        if event.get("type") == "transfer" and "Airport" in event.get("name", "")
    ]
    _assert(airport_transfers, "Departure day should include transfer to airport")
    _assert(airport_transfers[-1].get("end") == "13:00", "Airport transfer should arrive 90 minutes before 14:30 flight")
    _assert("Shanghai" not in airport_transfers[-1].get("name", ""), "Hotel-to-airport transfer should not include the arrival city airport")
    post_deadline_events = [
        event
        for event in day.get("events", [])
        if event.get("type") != "flight"
        and (event.get("end") or "") > "13:00"
    ]
    _assert(not post_deadline_events, "No destination-city activities should remain after the inbound airport deadline")


def test_early_inbound_checkout_allows_airport_transfer_time() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 1,
                "budget": {"total": 20000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-04", "check_out": "2026-05-04", "days": 1},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 1, "food_budget": 800, "attractions_budget": 600, "hotel_budget": 2000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 1600},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Breakfast Cafe", "estimated_meal_per_person": 80, "types": ["cafe"]},
            ],
            "selected_pois": [],
            "selected_all_flight_legs": [
                {
                    "leg_type": "inbound",
                    "from_city": "Tokyo",
                    "to_city": "Shanghai",
                    "departure_date": "2026-05-04",
                    "selected_flights": [
                        {
                            "departure_time": "10:50",
                            "arrival_time": "13:00",
                            "duration_minutes": 130,
                            "price_total": 3000,
                            "currency": "CNY",
                            "ground_to_departure_airport_hours": 0.5,
                        }
                    ],
                }
            ],
        }
    )

    day = result.get("daily_plan", [])[0]
    checkout = next(event for event in day.get("events", []) if event.get("type") == "checkout")
    airport_transfer = next(event for event in day.get("events", []) if event.get("type") == "transfer" and "Airport" in event.get("name", ""))
    _assert(checkout.get("end") == "08:50", "Checkout should finish before the hotel-to-airport transfer")
    _assert(airport_transfer.get("start") == "08:50", "Airport transfer should start immediately after checkout")
    _assert(airport_transfer.get("end") == "09:20", "Airport transfer should arrive 90 minutes before 10:50 flight")


def test_breakfast_uses_breakfast_restaurant_and_evening_returns_at_21() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 1,
                "budget": {"total": 20000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-02", "check_out": "2026-05-03", "days": 1},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 1, "food_budget": 1200, "attractions_budget": 600, "hotel_budget": 2000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 1600, "lat": 35.67, "lng": 139.76},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Serious Steak House", "estimated_meal_per_person": 250, "lat": 35.67, "lng": 139.76},
                {"city": "Tokyo", "name": "Morning Bakery Cafe", "estimated_meal_per_person": 80, "lat": 35.671, "lng": 139.761, "types": ["cafe", "bakery"]},
                {"city": "Tokyo", "name": "Dinner Izakaya", "estimated_meal_per_person": 180, "lat": 35.672, "lng": 139.762},
            ],
            "selected_pois": [
                {"city": "Tokyo", "name": "Tokyo Tower Night View", "estimated_total_cost": 100, "lat": 35.6586, "lng": 139.7454, "types": ["observation_deck"]},
            ],
            "selected_all_flight_legs": [],
        }
    )

    day = result.get("daily_plan", [])[0]
    breakfast = next(
        event
        for event in day.get("events", [])
        if event.get("type") == "meal" and (event.get("meta") or {}).get("meal_slot") == "breakfast"
    )
    _assert("Cafe" in breakfast.get("name", "") or "Bakery" in breakfast.get("name", ""), "Breakfast should use a breakfast-style restaurant")
    evening_pois = [
        event
        for event in day.get("events", [])
        if event.get("type") == "poi" and (event.get("meta") or {}).get("reason") == "evening_after_dinner"
    ]
    _assert(evening_pois, "Dinner should be followed by an evening activity when the day allows it")
    _assert(evening_pois[-1].get("end") == "21:00", "Evening activity should end at 21:00 before returning to hotel")


def test_night_poi_is_not_scheduled_in_afternoon_and_not_repeated_within_three_days() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 3,
                "budget": {"total": 30000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-02", "check_out": "2026-05-05", "days": 3},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 3, "food_budget": 2000, "attractions_budget": 1000, "hotel_budget": 3000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Tokyo Test Hotel", "estimated_total_cost": 2400, "lat": 35.67, "lng": 139.76},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Morning Cafe", "estimated_meal_per_person": 80, "types": ["cafe"], "lat": 35.67, "lng": 139.76},
                {"city": "Tokyo", "name": "Lunch Sushi", "estimated_meal_per_person": 120, "lat": 35.671, "lng": 139.761},
                {"city": "Tokyo", "name": "Dinner Place", "estimated_meal_per_person": 160, "lat": 35.672, "lng": 139.762},
            ],
            "selected_pois": [
                {"city": "Tokyo", "name": "Tokyo Night & Light", "estimated_total_cost": 100, "types": ["night_view"], "lat": 35.68, "lng": 139.77},
                {"city": "Tokyo", "name": "Morning Garden", "estimated_total_cost": 50, "lat": 35.67, "lng": 139.76},
            ],
            "selected_all_flight_legs": [],
        }
    )

    seen_night_days = 0
    for day in result.get("daily_plan", []):
        for event in day.get("events", []):
            if event.get("type") != "poi" or "Tokyo Night & Light" not in event.get("name", ""):
                continue
            seen_night_days += 1
            start = event.get("start")
            _assert(start >= "19:00", "Night/light attraction should not be scheduled in the afternoon")
    _assert(seen_night_days <= 1, "Same night attraction should not repeat within three days")


def test_adjacent_places_get_transfer_even_without_existing_gap() -> None:
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "08:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Hotel A", "lat": 35.67, "lng": 139.76},
        "events": [
            {
                "type": "checkin",
                "name": "Check-in Hotel A",
                "start": "08:00",
                "end": "08:15",
                "duration_h": 0.25,
                "meta": {"hotel_name": "Hotel A", "lat": 35.67, "lng": 139.76},
            },
            {
                "type": "poi",
                "name": "Nearby Garden",
                "start": "08:15",
                "end": "09:45",
                "duration_h": 1.5,
                "meta": {"poi_name": "Nearby Garden", "lat": 35.671, "lng": 139.761},
            },
        ],
        "warnings": [],
        "transport_cost_estimated": 0.0,
        "local_transport_tier": "medium",
        "local_transport_is_international": True,
    }

    _ensure_adjacent_event_transfers(day)

    transfers = [event for event in day["events"] if event.get("type") == "transfer"]
    poi = next(event for event in day["events"] if event.get("type") == "poi")
    _assert(transfers, "Adjacent location events should get a walk/transfer even when no gap existed")
    _assert(transfers[0].get("start") == "08:15", "Transfer should start after check-in")
    _assert(poi.get("start") >= transfers[0].get("end"), "POI should be shifted after the inserted transfer")


def test_pre_meal_gap_gets_activity_and_late_transfer() -> None:
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "08:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Hotel A", "lat": 35.67, "lng": 139.76},
        "events": [
            {
                "type": "poi",
                "name": "Tokyo Metropolitan Government Building No.1",
                "start": "09:00",
                "end": "10:21",
                "duration_h": 1.35,
                "meta": {
                    "poi_name": "Tokyo Metropolitan Government Building No.1",
                    "lat": 35.6896,
                    "lng": 139.6921,
                },
            },
            {
                "type": "meal",
                "name": "Japanese food Wagyu Restaurant",
                "start": "12:00",
                "end": "13:00",
                "duration_h": 1.0,
                "meta": {
                    "restaurant_name": "Japanese food Wagyu Restaurant",
                    "meal_slot": "lunch",
                    "lat": 35.6984,
                    "lng": 139.7730,
                },
            },
        ],
        "warnings": [],
        "transport_cost_estimated": 0.0,
        "local_transport_tier": "medium",
        "local_transport_is_international": True,
    }

    _ensure_adjacent_event_transfers(day)

    filler = next(
        event
        for event in day["events"]
        if event.get("type") == "poi" and (event.get("meta") or {}).get("reason") == "idle_gap_fill"
    )
    transfer = next(event for event in day["events"] if event.get("type") == "transfer")
    lunch = next(event for event in day["events"] if event.get("type") == "meal")

    _assert(filler.get("start") == "10:21", "Large pre-lunch gap should be filled after the morning POI")
    _assert(transfer.get("end") == lunch.get("start"), "Restaurant transfer should arrive at the meal time, not 90 minutes early")
    _assert(filler.get("end") == transfer.get("start"), "The filler activity should connect directly to the meal transfer")


def test_pre_meal_gap_prefers_nearby_real_poi() -> None:
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "08:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Hotel A", "lat": 35.67, "lng": 139.76},
        "events": [
            {
                "type": "poi",
                "name": "Tokyo Metropolitan Government Building No.1",
                "start": "09:00",
                "end": "10:21",
                "duration_h": 1.35,
                "meta": {
                    "poi_name": "Tokyo Metropolitan Government Building No.1",
                    "lat": 35.6896,
                    "lng": 139.6921,
                },
            },
            {
                "type": "meal",
                "name": "Japanese food Wagyu Restaurant",
                "start": "12:00",
                "end": "13:00",
                "duration_h": 1.0,
                "meta": {
                    "restaurant_name": "Japanese food Wagyu Restaurant",
                    "meal_slot": "lunch",
                    "lat": 35.6984,
                    "lng": 139.7730,
                },
            },
        ],
        "warnings": [],
        "transport_cost_estimated": 0.0,
        "local_transport_tier": "medium",
        "local_transport_is_international": True,
    }

    _ensure_adjacent_event_transfers(
        day,
        gap_poi_candidates=[
            {
                "city": "Tokyo",
                "name": "Shinjuku Central Park",
                "address": "2-11 Nishishinjuku, Tokyo",
                "lat": 35.6900,
                "lng": 139.6910,
                "estimated_total_cost": 0,
            }
        ],
    )

    filler = next(
        event
        for event in day["events"]
        if event.get("type") == "poi" and (event.get("meta") or {}).get("reason") == "idle_gap_fill"
    )
    transfers = [event for event in day["events"] if event.get("type") == "transfer"]
    _assert(filler.get("name") == "Shinjuku Central Park", "Large gap should prefer an unused nearby real POI")
    _assert((filler.get("meta") or {}).get("real_gap_fill") is True, "Real POI gap filler should be marked")
    _assert(len(transfers) == 2, "Real POI gap filler should get transfers in and out")


def test_real_gap_poi_respects_existing_poi_blocklist() -> None:
    candidate = {
        "city": "Tokyo",
        "name": "Shinjuku Central Park",
        "address": "2-11 Nishishinjuku, Tokyo",
        "lat": 35.6900,
        "lng": 139.6910,
        "estimated_total_cost": 0,
    }
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "08:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Hotel A", "lat": 35.67, "lng": 139.76},
        "events": [
            {
                "type": "poi",
                "name": "Tokyo Metropolitan Government Building No.1",
                "start": "09:00",
                "end": "10:21",
                "duration_h": 1.35,
                "meta": {"poi_name": "Tokyo Metropolitan Government Building No.1", "lat": 35.6896, "lng": 139.6921},
            },
            {
                "type": "meal",
                "name": "Japanese food Wagyu Restaurant",
                "start": "12:00",
                "end": "13:00",
                "duration_h": 1.0,
                "meta": {"restaurant_name": "Japanese food Wagyu Restaurant", "meal_slot": "lunch", "lat": 35.6984, "lng": 139.7730},
            },
        ],
        "warnings": [],
        "transport_cost_estimated": 0.0,
        "local_transport_tier": "medium",
        "local_transport_is_international": True,
    }

    _ensure_adjacent_event_transfers(
        day,
        gap_poi_candidates=[candidate],
        blocked_poi_keys={_poi_dedupe_key(candidate, city="Tokyo")},
    )

    fillers = [
        event
        for event in day["events"]
        if event.get("type") == "poi" and (event.get("meta") or {}).get("reason") == "idle_gap_fill"
    ]
    _assert(fillers, "Blocked real POI should fall back to an area walk filler")
    _assert(fillers[0].get("name") != "Shinjuku Central Park", "Blocked real POI must not be reused as a gap filler")
    _assert((fillers[0].get("meta") or {}).get("real_gap_fill") is not True, "Fallback area walk should not be marked as a real POI")


def test_real_gap_poi_does_not_duplicate_current_stop_by_name() -> None:
    candidate = {
        "city": "Tokyo",
        "name": "Shinjuku Gyoen National Garden",
        "address": "11 Naitomachi, Shinjuku City, Tokyo",
        "lat": 35.6852,
        "lng": 139.7101,
        "estimated_total_cost": 80,
    }
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "08:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Hotel A", "lat": 35.67, "lng": 139.76},
        "events": [
            {
                "type": "poi",
                "name": "Shinjuku Gyoen National Garden",
                "start": "09:00",
                "end": "10:20",
                "duration_h": 1.33,
                "meta": {"poi_name": "Shinjuku Gyoen National Garden", "lat": 35.6852, "lng": 139.7101},
            },
            {
                "type": "meal",
                "name": "Lunch Restaurant",
                "start": "12:00",
                "end": "13:00",
                "duration_h": 1.0,
                "meta": {"restaurant_name": "Lunch Restaurant", "meal_slot": "lunch", "lat": 35.6900, "lng": 139.7050},
            },
        ],
        "warnings": [],
        "transport_cost_estimated": 0.0,
        "local_transport_tier": "medium",
        "local_transport_is_international": True,
    }

    _ensure_adjacent_event_transfers(day, gap_poi_candidates=[candidate])

    fillers = [
        event
        for event in day["events"]
        if event.get("type") == "poi" and (event.get("meta") or {}).get("reason") == "idle_gap_fill"
    ]
    _assert(fillers, "Large pre-meal gap should still be filled")
    _assert(
        fillers[0].get("name") != "Shinjuku Gyoen National Garden",
        "Gap filler must not duplicate the current attraction just because address/place_id differ",
    )
    _assert((fillers[0].get("meta") or {}).get("real_gap_fill") is not True, "Duplicate real POI should fall back to an area walk")


def test_unique_poi_visits_removes_real_gap_duplicate_by_name_only() -> None:
    daily_plans = [
        {
            "date": "2026-05-02",
            "city": "Tokyo",
            "events": [
                {
                    "type": "poi",
                    "name": "Godzilla Head",
                    "start": "10:00",
                    "end": "11:00",
                    "meta": {"poi_name": "Godzilla Head"},
                },
                {
                    "type": "poi",
                    "name": "Godzilla Head",
                    "start": "12:00",
                    "end": "13:00",
                    "meta": {
                        "poi_name": "Godzilla Head",
                        "address": "1-19-1 Kabukicho, Tokyo",
                        "reason": "idle_gap_fill",
                        "real_gap_fill": True,
                    },
                },
                {
                    "type": "poi",
                    "name": "Local area walk near Godzilla Head",
                    "start": "14:00",
                    "end": "15:00",
                    "meta": {
                        "poi_name": "Local area walk near Godzilla Head",
                        "reason": "idle_gap_fill",
                    },
                },
            ],
            "warnings": [],
        }
    ]

    _enforce_unique_poi_visits(daily_plans)

    names = [event.get("name") for event in daily_plans[0]["events"] if event.get("type") == "poi"]
    _assert(names.count("Godzilla Head") == 1, "Real duplicate POIs should be removed by normalized name")
    _assert(
        "Local area walk near Godzilla Head" in names,
        "Synthetic area-walk fillers should not be treated as real attraction reuse",
    )


def test_same_site_transfer_rows_are_removed() -> None:
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "events": [
            {
                "type": "meal",
                "name": "Dinner Place",
                "start": "18:00",
                "end": "19:00",
                "meta": {"restaurant_name": "Dinner Place", "meal_slot": "dinner"},
            },
            {
                "type": "transfer",
                "name": "Transfer Dinner Place -> Dinner Place",
                "start": "19:00",
                "end": "19:05",
                "meta": {"from": "Dinner Place", "to": "Dinner Place"},
            },
        ],
        "warnings": [],
    }

    _remove_invalid_same_site_transfers(day)

    _assert(
        not any(event.get("type") == "transfer" for event in day["events"]),
        "Transfer rows whose from/to endpoint is the same place should be removed",
    )


def test_overlapping_transition_date_keeps_destination_city_slot() -> None:
    slots = _extract_daily_city_slots(
        {
            "Kyoto": {"check_in": "2026-06-04", "check_out": "2026-06-07", "days": 3},
            "Osaka": {"check_in": "2026-06-06", "check_out": "2026-06-08", "days": 2},
        }
    )

    date_to_city = {slot["date"]: slot["city"] for slot in slots}
    _assert(date_to_city["2026-06-06"] == "Osaka", "Shared transition date should become the destination-city day")
    _assert(
        len([slot for slot in slots if slot["date"] == "2026-06-06"]) == 1,
        "A transition date should not generate two day cards",
    )


def test_intercity_drive_orders_checkout_drive_lunch_then_checkin() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Kyoto", "Osaka"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 4,
                "budget": {"total": 40000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Kyoto": {"check_in": "2026-06-04", "check_out": "2026-06-07", "days": 3},
                "Osaka": {"check_in": "2026-06-06", "check_out": "2026-06-08", "days": 2},
            },
            "city_budget_plan": [
                {"city": "Kyoto", "days": 2, "food_budget": 2000, "attractions_budget": 500, "hotel_budget": 3000},
                {"city": "Osaka", "days": 2, "food_budget": 2000, "attractions_budget": 500, "hotel_budget": 3000},
            ],
            "selected_hotels": [
                {"city": "Kyoto", "name": "Kyoto Hotel", "estimated_total_cost": 2000, "lat": 35.0, "lng": 135.76},
                {"city": "Osaka", "name": "Osaka Hotel", "estimated_total_cost": 2000, "lat": 34.70, "lng": 135.50},
            ],
            "selected_restaurants": [
                {"city": "Kyoto", "name": "Kyoto Breakfast Cafe", "estimated_meal_per_person": 80, "types": ["cafe"], "lat": 35.0, "lng": 135.76},
                {"city": "Kyoto", "name": "Kyoto Dinner", "estimated_meal_per_person": 150, "lat": 35.01, "lng": 135.77},
                {"city": "Osaka", "name": "Osaka Breakfast Cafe", "estimated_meal_per_person": 80, "types": ["cafe"], "lat": 34.70, "lng": 135.50},
                {"city": "Osaka", "name": "Osaka Lunch", "estimated_meal_per_person": 120, "lat": 34.70, "lng": 135.50},
                {"city": "Osaka", "name": "Osaka Dinner", "estimated_meal_per_person": 150, "lat": 34.71, "lng": 135.51},
            ],
            "selected_pois": [
                {"city": "Kyoto", "name": "Kyoto Garden", "estimated_total_cost": 80, "lat": 35.01, "lng": 135.77},
                {"city": "Osaka", "name": "Osaka Castle", "estimated_total_cost": 80, "lat": 34.6873, "lng": 135.5262},
            ],
            "selected_all_flight_legs": [
                {
                    "leg_type": "intercity",
                    "from_city": "Kyoto",
                    "to_city": "Osaka",
                    "departure_date": "2026-06-06",
                    "transport_mode": "direct_drive",
                    "selected_flights": [],
                    "direct_drive_hours": 4.0,
                    "rental_car_cost_estimated": 800,
                }
            ],
        }
    )

    transition_day = next(day for day in result.get("daily_plan", []) if day.get("date") == "2026-06-06")
    events = transition_day.get("events", [])
    checkout = next(event for event in events if event.get("type") == "checkout")
    drive = next(event for event in events if event.get("type") == "drive")
    lunch = next(event for event in events if event.get("type") == "meal" and (event.get("meta") or {}).get("meal_slot") == "lunch")
    checkin = next(event for event in events if event.get("type") == "checkin")

    _assert(checkout.get("end") <= drive.get("start"), "Previous-city checkout should happen before intercity drive")
    _assert(drive.get("end") <= lunch.get("start"), "Lunch should happen after arriving in the new city")
    _assert(lunch.get("end") <= checkin.get("start"), "If arrival is during lunch time, lunch should come before check-in")


def test_real_gap_poi_participates_in_three_day_cooldown() -> None:
    daily_plans = [
        {
            "date": "2026-05-01",
            "city": "Tokyo",
            "events": [
                {
                    "type": "poi",
                    "name": "Shinjuku Central Park",
                    "start": "09:00",
                    "end": "10:00",
                    "duration_h": 1.0,
                    "meta": {"poi_name": "Shinjuku Central Park", "address": "2-11 Nishishinjuku"},
                }
            ],
        },
        {
            "date": "2026-05-02",
            "city": "Tokyo",
            "events": [
                {
                    "type": "poi",
                    "name": "Shinjuku Central Park",
                    "start": "10:30",
                    "end": "11:30",
                    "duration_h": 1.0,
                    "meta": {
                        "poi_name": "Shinjuku Central Park",
                        "address": "2-11 Nishishinjuku",
                        "reason": "idle_gap_fill",
                        "real_gap_fill": True,
                    },
                },
                {
                    "type": "poi",
                    "name": "Local area walk near Lunch",
                    "start": "11:30",
                    "end": "12:00",
                    "duration_h": 0.5,
                    "meta": {"poi_name": "Local area walk near Lunch", "reason": "idle_gap_fill"},
                },
            ],
        },
    ]

    _enforce_poi_three_day_cooldown(daily_plans)

    day2_pois = [event.get("name") for event in daily_plans[1].get("events", []) if event.get("type") == "poi"]
    _assert("Shinjuku Central Park" not in day2_pois, "Real gap filler should be removed by three-day POI cooldown")
    _assert("Local area walk near Lunch" in day2_pois, "Synthetic area walk fallback should not be removed by POI cooldown")


def test_area_must_visit_fallback_is_created_for_tokyo_district() -> None:
    fallbacks = _build_area_must_visit_fallbacks(
        city="Tokyo",
        must_visit_names=["Ginza"],
        matched_must_visit_pois=[],
        hotel={"name": "Hotel A", "lat": 35.67, "lng": 139.76},
    )

    _assert(fallbacks, "District-style must visit should create a fallback POI")
    _assert(fallbacks[0]["name"] == "Ginza shopping district walk", "Fallback should use the configured district walk name")
    _assert(fallbacks[0]["_must_visit_name_norm"] == "ginza", "Fallback should keep must_visit_name coverage metadata")


def test_large_gap_before_next_poi_gets_real_poi() -> None:
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "08:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Hotel A", "lat": 35.67, "lng": 139.76},
        "events": [
            {
                "type": "meal",
                "name": "Morning Bakery",
                "start": "08:00",
                "end": "08:40",
                "duration_h": 0.67,
                "meta": {"restaurant_name": "Morning Bakery", "meal_slot": "breakfast", "lat": 35.6700, "lng": 139.7600},
            },
            {
                "type": "poi",
                "name": "Afternoon Museum",
                "start": "11:00",
                "end": "12:30",
                "duration_h": 1.5,
                "meta": {"poi_name": "Afternoon Museum", "lat": 35.6730, "lng": 139.7630},
            },
        ],
        "warnings": [],
        "transport_cost_estimated": 0.0,
        "local_transport_tier": "medium",
        "local_transport_is_international": True,
    }

    _ensure_adjacent_event_transfers(
        day,
        gap_poi_candidates=[
            {"city": "Tokyo", "name": "Nearby Shrine", "lat": 35.6710, "lng": 139.7610, "estimated_total_cost": 0}
        ],
    )

    filler = next(
        event
        for event in day["events"]
        if event.get("type") == "poi" and (event.get("meta") or {}).get("reason") == "idle_gap_fill"
    )
    _assert(filler.get("name") == "Nearby Shrine", "Large gap before a POI should also prefer a real nearby POI")


def test_required_meals_are_added_when_not_covered_by_flight() -> None:
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "08:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Tokyo Test Hotel", "lat": 35.67, "lng": 139.76},
        "events": [],
        "warnings": [],
        "used_hours": 0.0,
    }
    candidates = {
        "restaurants": [
            {"city": "Tokyo", "name": "Morning Bakery Cafe", "estimated_meal_per_person": 80, "types": ["cafe"], "lat": 35.671, "lng": 139.761},
            {"city": "Tokyo", "name": "Lunch Sushi", "estimated_meal_per_person": 120, "lat": 35.672, "lng": 139.762},
            {"city": "Tokyo", "name": "Dinner Izakaya", "estimated_meal_per_person": 160, "lat": 35.673, "lng": 139.763},
        ],
    }

    _ensure_required_meals(day, candidates, {"food_budget": 2000, "days": 1})

    slots = {
        (event.get("meta") or {}).get("meal_slot")
        for event in day["events"]
        if event.get("type") == "meal"
    }
    _assert(slots == {"breakfast", "lunch", "dinner"}, "A full non-flight day should include breakfast, lunch, and dinner")
    breakfast = next(event for event in day["events"] if (event.get("meta") or {}).get("meal_slot") == "breakfast")
    _assert("Morning Bakery Cafe" in breakfast.get("name", ""), "Breakfast should use a concrete breakfast restaurant when available")


def test_required_lunch_is_skipped_when_flight_covers_lunch() -> None:
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "08:00",
        "schedule_end": "20:30",
        "hotel": {"name": "Tokyo Test Hotel", "lat": 35.67, "lng": 139.76},
        "events": [
            {
                "type": "flight",
                "name": "Flight Tokyo -> Osaka",
                "start": "12:10",
                "end": "13:00",
                "duration_h": 0.83,
                "cost_estimated": 0.0,
                "meta": {"leg_type": "intercity"},
            },
        ],
        "warnings": [],
        "used_hours": 0.0,
    }
    candidates = {
        "restaurants": [
            {"city": "Tokyo", "name": "Morning Bakery Cafe", "estimated_meal_per_person": 80, "types": ["cafe"], "lat": 35.671, "lng": 139.761},
            {"city": "Tokyo", "name": "Lunch Sushi", "estimated_meal_per_person": 120, "lat": 35.672, "lng": 139.762},
            {"city": "Tokyo", "name": "Dinner Izakaya", "estimated_meal_per_person": 160, "lat": 35.673, "lng": 139.763},
        ],
    }

    _ensure_required_meals(day, candidates, {"food_budget": 2000, "days": 1})

    slots = [
        (event.get("meta") or {}).get("meal_slot")
        for event in day["events"]
        if event.get("type") == "meal"
    ]
    _assert("lunch" not in slots, "Lunch should be treated as onboard when a flight overlaps 12:00-13:30")
    _assert({"breakfast", "dinner"}.issubset(set(slots)), "Other non-covered meals should still be scheduled")


def test_short_city_stay_keeps_one_hotel_area() -> None:
    result = run_itinerary_agent(
        {
            "trip_summary": {
                "origin": "Shanghai",
                "destination": ["Tokyo"],
                "country": "japan",
                "num_people": 1,
                "duration_days": 4,
                "budget": {"total": 30000, "currency": "CNY"},
                "preferences": {},
            },
            "city_date_ranges": {
                "Tokyo": {"check_in": "2026-05-02", "check_out": "2026-05-06", "days": 4},
            },
            "city_budget_plan": [
                {"city": "Tokyo", "days": 4, "food_budget": 3000, "attractions_budget": 1000, "hotel_budget": 8000},
            ],
            "selected_hotels": [
                {"city": "Tokyo", "name": "Dense Area Hotel", "estimated_total_cost": 4000, "lat": 35.67, "lng": 139.76, "rating": 4.5},
                {"city": "Tokyo", "name": "Second Area Hotel", "estimated_total_cost": 4000, "lat": 35.80, "lng": 139.90, "rating": 4.4},
            ],
            "selected_restaurants": [
                {"city": "Tokyo", "name": "Morning Cafe", "estimated_meal_per_person": 80, "types": ["cafe"], "lat": 35.671, "lng": 139.761},
                {"city": "Tokyo", "name": "Lunch Place", "estimated_meal_per_person": 120, "lat": 35.672, "lng": 139.762},
                {"city": "Tokyo", "name": "Dinner Place", "estimated_meal_per_person": 160, "lat": 35.673, "lng": 139.763},
            ],
            "selected_pois": [
                {"city": "Tokyo", "name": "Garden A", "estimated_total_cost": 50, "lat": 35.674, "lng": 139.764},
                {"city": "Tokyo", "name": "Museum A", "estimated_total_cost": 80, "lat": 35.675, "lng": 139.765},
            ],
            "selected_all_flight_legs": [],
        }
    )

    hotel_names = {
        (day.get("hotel") or {}).get("name")
        for day in result.get("daily_plan", [])
        if day.get("hotel")
    }
    _assert(hotel_names == {"Dense Area Hotel"}, "Stays under five city days should not switch hotel areas")


def test_ultra_international_hotel_switch_cost_is_capped() -> None:
    days = [
        {
            "date": "2026-05-01",
            "city": "Tokyo",
            "schedule_start": "08:00",
            "schedule_end": "20:30",
            "hotel": {"name": "Hotel A", "lat": None, "lng": None},
            "events": [],
            "used_hours": 0.0,
            "transport_cost_estimated": 0.0,
            "warnings": [],
            "local_transport_tier": "ultra",
            "local_transport_is_international": True,
        },
        {
            "date": "2026-05-02",
            "city": "Tokyo",
            "schedule_start": "08:00",
            "schedule_end": "20:30",
            "hotel": {"name": "Hotel B", "lat": None, "lng": None},
            "events": [],
            "used_hours": 0.0,
            "transport_cost_estimated": 0.0,
            "warnings": [],
            "local_transport_tier": "ultra",
            "local_transport_is_international": True,
        },
    ]

    _ensure_hotel_transition_events(days)
    switch_transfer = next(
        event
        for event in days[1]["events"]
        if event.get("type") == "transfer" and (event.get("meta") or {}).get("reason") == "hotel_switch"
    )
    _assert(switch_transfer.get("cost_estimated") == 300.0, "30-minute international ultra hotel transfer should be capped at 300 CNY")


def test_compress_timeline_keeps_lunch_in_window() -> None:
    day = {
        "date": "2026-05-02",
        "city": "Tokyo",
        "schedule_start": "09:00",
        "schedule_end": "20:30",
        "events": [
            {"type": "checkin", "name": "Check-in Hotel B", "start": "11:00", "end": "11:30", "duration_h": 0.5},
            {
                "type": "meal",
                "name": "Lunch near Hotel B",
                "start": "12:00",
                "end": "13:00",
                "duration_h": 1.0,
                "meta": {"meal_slot": "lunch"},
            },
        ],
        "warnings": [],
    }

    _compress_flexible_timeline(day)

    lunch = next(event for event in day["events"] if event.get("type") == "meal")
    _assert(lunch.get("start") == "12:00", "Timeline compression must not move lunch before 12:00")


def main() -> None:
    test_schedule_city_day_accepts_new_kwargs()
    test_refinement_state_marks_food_budget_ceiling()
    test_refinement_state_keeps_food_upgrade_open_when_under_budget()
    test_refinement_uses_previous_budget_when_user_does_not_increase_again()
    test_refinement_new_budget_overrides_previous_budget()
    test_refinement_rejects_lower_selected_total_than_previous_plan()
    test_refinement_radius_controls_food_and_poi_filtering()
    test_refinement_state_marks_flight_and_hotel_budget_ceilings()
    test_finalize_success_output_fails_when_itinerary_fails()
    test_agent4_normalize_accepts_agent3_city_budget_plan_used()
    test_agent4_normalize_preserves_budget_plan_for_fallback_budget_map()
    test_agent4_spend_up_targets_budget_ceiling()
    test_parser_and_city_plan_keep_explicit_three_city_nights()
    test_city_plan_orders_japan_cities_by_gateway()
    test_city_plan_preserves_explicit_sequence_markers()
    test_hotel_area_parser_extracts_only_lodging_area()
    test_last_city_checkout_date_becomes_departure_day_slot()
    test_final_transfer_pass_adds_transfer_without_walk_filler()
    test_airport_display_infers_nearby_airport_from_airport_city()
    test_missing_city_hotel_gets_fallback_hotel_and_checkin()
    test_distinct_synthetic_pois_get_transfer_even_with_close_estimated_coords()
    test_evening_transfer_starts_after_dinner_end()
    test_checkin_after_poi_overlap_is_shifted_after_previous_event()
    test_stale_evening_transfer_is_removed_and_rebuilt()
    test_checkin_transfer_after_drive_does_not_overlap_drive()
    test_agent3_drive_under_4h_skips_flight_selection()
    test_agent3_drive_over_4h_allows_flight_selection()
    test_agent3_known_city_pair_over_4h_does_not_skip_flights()
    test_agent3_budget_utilization_fields_and_spend_up()
    test_hotel_area_query_fallback_metadata()
    test_hotel_area_query_failure_reason_is_preserved()
    test_unknown_multi_city_hotel_area_applies_to_first_city()
    test_realtime_hotel_cost_spend_up_and_total_budget_cap()
    test_yearless_numeric_date_range_overrides_wrong_llm_dates()
    test_flight_covering_dinner_window_assumes_onboard_dinner()
    test_late_arrival_after_non_dinner_flight_gets_dinner()
    test_hotel_switch_gets_checkout_checkin_and_transfer()
    test_overnight_outbound_has_no_prearrival_checkin()
    test_overnight_afternoon_arrival_gets_dinner()
    test_arrival_day_has_airport_transfer_and_hotel_cost()
    test_nearby_airport_arrival_is_visible_in_itinerary()
    test_router_preserves_nearby_airport_route_fields()
    test_inbound_airport_transfer_arrives_90_minutes_early()
    test_early_inbound_checkout_allows_airport_transfer_time()
    test_breakfast_uses_breakfast_restaurant_and_evening_returns_at_21()
    test_night_poi_is_not_scheduled_in_afternoon_and_not_repeated_within_three_days()
    test_adjacent_places_get_transfer_even_without_existing_gap()
    test_pre_meal_gap_gets_activity_and_late_transfer()
    test_pre_meal_gap_prefers_nearby_real_poi()
    test_real_gap_poi_respects_existing_poi_blocklist()
    test_real_gap_poi_does_not_duplicate_current_stop_by_name()
    test_unique_poi_visits_removes_real_gap_duplicate_by_name_only()
    test_same_site_transfer_rows_are_removed()
    test_overlapping_transition_date_keeps_destination_city_slot()
    test_intercity_drive_orders_checkout_drive_lunch_then_checkin()
    test_real_gap_poi_participates_in_three_day_cooldown()
    test_area_must_visit_fallback_is_created_for_tokyo_district()
    test_large_gap_before_next_poi_gets_real_poi()
    test_required_meals_are_added_when_not_covered_by_flight()
    test_required_lunch_is_skipped_when_flight_covers_lunch()
    test_short_city_stay_keeps_one_hotel_area()
    test_ultra_international_hotel_switch_cost_is_capped()
    test_compress_timeline_keeps_lunch_in_window()
    print("smoke_regression: OK")


if __name__ == "__main__":
    main()
