from __future__ import annotations

from typing import Any, Dict

import router
from itinerary_agent import _schedule_city_day


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


def main() -> None:
    test_schedule_city_day_accepts_new_kwargs()
    test_finalize_success_output_fails_when_itinerary_fails()
    print("smoke_regression: OK")


if __name__ == "__main__":
    main()
