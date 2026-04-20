from __future__ import annotations

from typing import Any, Dict, List, Tuple

from itinerary_rules import validate_day_time_and_drive
from planner_schema import normalize_text, safe_float


def collect_must_visit_coverage(daily_plans: List[Dict[str, Any]], must_visit_names: List[str]) -> Tuple[int, List[str]]:
    target_norm = [normalize_text(x).lower() for x in must_visit_names if normalize_text(x)]
    seen = set()

    for day in daily_plans:
        for event in day.get("events", []) or []:
            meta = event.get("meta", {}) or {}
            mv = normalize_text(meta.get("must_visit_name")).lower()
            if mv:
                seen.add(mv)

    done = 0
    uncovered: List[str] = []
    for raw in must_visit_names:
        n = normalize_text(raw).lower()
        if n in seen:
            done += 1
        else:
            uncovered.append(raw)

    if not must_visit_names and not target_norm:
        return 0, []

    return done, uncovered


def apply_day_constraints(daily_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for day in daily_plans:
        checks = validate_day_time_and_drive(day)
        day["constraints"] = {
            "within_12h": checks.get("within_12h", True),
            "max_drive_leg_le_5h": checks.get("max_drive_leg_le_5h", True),
        }
        for w in checks.get("warnings", []):
            day.setdefault("warnings", []).append(w)
        out.append(day)
    return out


def build_cost_rollup(daily_plans: List[Dict[str, Any]]) -> Dict[str, float]:
    transport = 0.0
    food = 0.0
    attractions = 0.0
    hotel_reference = 0.0

    city_hotel_seen = set()

    for day in daily_plans:
        city = normalize_text(day.get("city")).lower()
        hotel = day.get("hotel") or {}
        if city and city not in city_hotel_seen and isinstance(hotel, dict):
            hotel_reference += safe_float(hotel.get("estimated_total_cost"), 0.0)
            city_hotel_seen.add(city)

        for ev in day.get("events", []) or []:
            t = normalize_text(ev.get("type"))
            cost = safe_float(ev.get("cost_estimated"), 0.0)
            if t in {"flight", "drive", "local_transport"}:
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


def validate_budget_constraints(daily_plans: List[Dict[str, Any]], total_budget: float) -> Dict[str, Any]:
    rollup = build_cost_rollup(daily_plans)
    estimated = safe_float(rollup.get("total_selected_estimated"), 0.0)

    if total_budget <= 0:
        return {
            "budget_feasible": True,
            "total_budget": total_budget,
            "estimated_total": estimated,
            "delta": 0.0,
        }

    delta = round(total_budget - estimated, 2)
    return {
        "budget_feasible": delta >= 0,
        "total_budget": total_budget,
        "estimated_total": estimated,
        "delta": delta,
    }


def build_itinerary_warnings(daily_plans: List[Dict[str, Any]], must_visit_uncovered: List[str], budget_check: Dict[str, Any] | None = None) -> List[str]:
    warnings: List[str] = []

    for day in daily_plans:
        for w in day.get("warnings", []) or []:
            if normalize_text(w):
                warnings.append(normalize_text(w))

        constraints = day.get("constraints", {}) or {}
        if constraints.get("within_12h") is False:
            warnings.append(f"{day.get('date')} {day.get('city')}: exceeds 12h day limit.")
        if constraints.get("max_drive_leg_le_5h") is False:
            warnings.append(f"{day.get('date')} {day.get('city')}: contains drive segment over 5h.")

    for m in must_visit_uncovered:
        warnings.append(f"Must-visit attraction warning: '{m}' not scheduled.")

    if budget_check and budget_check.get("budget_feasible") is False:
        warnings.append(
            f"Budget warning: estimated total {budget_check.get('estimated_total')} exceeds total budget {budget_check.get('total_budget')}."
        )

    deduped: List[str] = []
    seen = set()
    for w in warnings:
        key = normalize_text(w)
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)

    return deduped


def build_itinerary_summary(daily_plans: List[Dict[str, Any]], budget_check: Dict[str, Any], must_visit_total: int, must_visit_done: int) -> Dict[str, Any]:
    total_days = len(daily_plans)
    time_feasible_days = 0

    for day in daily_plans:
        c = day.get("constraints", {}) or {}
        if c.get("within_12h") and c.get("max_drive_leg_le_5h"):
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


def format_agent4_output(daily_plans: List[Dict[str, Any]], itinerary_summary: Dict[str, Any], cost_rollup: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    return {
        "success": True,
        "stage": "agent4_itinerary_planning",
        "itinerary_summary": itinerary_summary,
        "daily_plan": daily_plans,
        "cost_rollup": cost_rollup,
        "warnings": warnings,
    }
