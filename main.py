from __future__ import annotations

import json
import mimetypes
import os
import re
import copy
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from router import get_router_backend_status, run_budget_pipeline
from travel_time import get_today_date
from travel_report_generator import write_travel_report
from travel_requirement_parser_llm_agent import _call_llm
from travel_requirement_parser_llm_agent import travel_requirement_parser_llm_agent


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "front-end"
REPORTS_DIR = BASE_DIR / "travel_reports"
API_HOST = os.getenv("TRAVEL_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("TRAVEL_API_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("TRAVEL_FRONTEND_PORT", "3000"))
API_PROCESS_STARTED_AT = get_today_date().isoformat() + " " + time.strftime("%H:%M:%S")
API_CODE_VERSION = datetime.fromtimestamp(Path(__file__).resolve().stat().st_mtime).isoformat(timespec="seconds")
RECOVERABLE_BUDGET_RESULTS: Dict[str, Dict[str, Any]] = {}
RECOVERABLE_BUDGET_ORDER: list[str] = []


def _compact_json(value: Any, limit: int = 2500) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _example_departure_date() -> str:
    return (get_today_date() + timedelta(days=7)).isoformat()


def _example_date_range() -> tuple[str, str]:
    start = get_today_date() + timedelta(days=14)
    end = start + timedelta(days=5 - 1)
    return start.isoformat(), end.isoformat()


def _build_preference_hint(user_request: str, parsed: dict) -> str:
    text = (user_request or "").lower()
    hints: list[str] = []

    if any(token in text for token in ["budget", "budget-friendly", "affordable", "cheap", "省钱", "预算"]):
        hints.append("I can help keep it budget-friendly by favoring better-value flights, stays, and dining choices.")
    if any(token in text for token in ["clean hotel", "clean hotels", "clean", "central hotel", "central hotels", "central"]):
        hints.append("I can prioritize clean, well-rated hotels in convenient central areas.")
    if any(token in text for token in ["local restaurant", "local restaurants", "local food", "food", "restaurant", "restaurants", "美食", "当地"]):
        hints.append("I can also lean toward local restaurants instead of tourist-focused places.")

    destination = parsed.get("destination")
    if not hints and destination:
        destination_text = ", ".join(destination) if isinstance(destination, list) else str(destination)
        hints.append(f"I can help shape this into a practical {destination_text} trip with budget, hotel, food, and pace suggestions.")

    if not hints:
        hints.append("I can help turn this into a practical trip plan and suggest good tradeoffs on budget, stay location, and pace.")

    return " ".join(hints[:2])


def _fallback_guidance(user_request: str, result: dict) -> str:
    example_departure_date = _example_departure_date()
    example_start_date, example_end_date = _example_date_range()
    def money_text(amount: Any, currency: str) -> str:
        if isinstance(amount, (int, float)):
            numeric = float(amount)
            if numeric.is_integer():
                return f"{int(numeric)} {currency}"
            return f"{round(numeric, 2)} {currency}"
        return f"{amount} {currency}"

    if result.get("stage") == "agent4_itinerary_planning":
        return (
            "The request had enough information, but the itinerary assembly step failed internally before a report could be generated. "
            "This is a system issue, not something you need to answer with another preference. Please retry after the backend fix is applied."
        )
    if result.get("stage") == "budget_feasibility":
        user_text = (user_request or "").lower()
        trip = result.get("trip_summary") or {}
        budget = (trip.get("budget") or {}).get("total")
        budget_currency = (trip.get("budget") or {}).get("currency") or "CNY"
        duration_days = trip.get("duration_days")
        budget_text = money_text(budget, budget_currency) if budget else f"your {budget_currency} cap"

        budget_check = result.get("budget_check") or {}
        flight_spend = budget_check.get("selected_total_flight_spend")
        if flight_spend is None:
            flight_spend = ((result.get("rough_budget_check") or {}).get("normal_estimate") or {}).get("transport")
        flight_line = "Flight costs are too high for this budget."
        if isinstance(flight_spend, (int, float)) and float(flight_spend) > 0:
            flight_line = f"Flight costs alone are about {int(round(float(flight_spend)))} {budget_currency} roundtrip."
        flight_exceeds_budget = (
            isinstance(flight_spend, (int, float))
            and isinstance(budget, (int, float))
            and float(flight_spend) > float(budget)
        )
        warning_text = "\n".join(str(item) for item in result.get("warnings", []) or []).lower()
        paid_attractions_already_removed = (
            "removed paid attraction" in warning_text
            or "paid attraction removed" in warning_text
            or "remove all paid" in user_text
            or "remove paid" in user_text
        )
        shortened_trip_already_applied = bool(
            re.search(
                r"\b(shorten|shorter|fewer days?|reduce.*days?|cut.*days?)\b|\b\d+\s*days?\s+instead of\s+\d+\s*days?\b|缩短|少.*天|改成.*天",
                user_text,
            )
        )
        dates_already_changed = bool(
            re.search(
                r"\b(change|changed|shift|moved|try|use)\b.{0,24}\b(date|dates|departure|departing|start|leave|leaving)\b|different dates?|new dates?|改日期|换日期|调整日期|换时间|改时间",
                user_text,
            )
        )

        minimum_budget_required = result.get("minimum_budget_required")
        rough_minimum_budget_required = (result.get("rough_budget_check") or {}).get("minimum_budget_required")
        if isinstance(minimum_budget_required, (int, float)) and isinstance(rough_minimum_budget_required, (int, float)):
            minimum_budget_required = max(float(minimum_budget_required), float(rough_minimum_budget_required))
        elif minimum_budget_required is None:
            minimum_budget_required = rough_minimum_budget_required

        suggested_budget_text = f"more than {budget_text}"
        budget_reason_line = "The current budget is below the minimum needed by the budget precheck."
        applied_budget_line = ""
        if isinstance(minimum_budget_required, (int, float)) and float(minimum_budget_required) > 0:
            suggested_budget_text = money_text(round(float(minimum_budget_required)), budget_currency)
            budget_reason_line = (
                "The budget precheck keeps core transport, hotels, and meals under about 70% of the total budget, "
                f"so the suggested minimum total budget is about {suggested_budget_text}."
            )
            if isinstance(budget, (int, float)) and float(budget) > 0:
                gap = int(round(float(minimum_budget_required) - float(budget)))
                if gap > 0:
                    applied_budget_line = (
                        f"I applied the current budget as {budget_text}, but it is still about "
                        f"{gap} {budget_currency} below that minimum.\n\n"
                    )
        elif isinstance(budget, (int, float)) and float(budget) > 0:
            suggested_budget = int(round(float(budget) * 1.2 / 1000.0) * 1000)
            if suggested_budget <= int(round(float(budget))):
                suggested_budget = int(round(float(budget))) + 1000
            suggested_budget_text = f"{suggested_budget} {budget_currency}"

        shorten_example = "a shorter trip"
        if isinstance(duration_days, (int, float)) and int(duration_days) > 1:
            current_days = int(duration_days)
            shorter_days = max(1, current_days - 1)
            day_label = "day" if shorter_days == 1 else "days"
            current_day_label = "day" if current_days == 1 else "days"
            shorten_example = f"{shorter_days} {day_label} instead of {current_days} {current_day_label}"

        option_lines = [
            f"- Increase your budget to at least {suggested_budget_text}",
        ]
        if not shortened_trip_already_applied:
            option_lines.append(f"- Shorten the trip (for example, {shorten_example})")
        if not paid_attractions_already_removed and not flight_exceeds_budget:
            option_lines.append("- Remove paid attractions")

        already_tried_line = ""
        if paid_attractions_already_removed:
            already_tried_line = (
                "I removed paid attraction costs, but the remaining fixed costs are still above the budget.\n\n"
            )
        elif flight_exceeds_budget:
            already_tried_line = (
                "Removing paid attractions will not be enough because the flight cost alone is already above the current budget.\n\n"
            )
        if shortened_trip_already_applied:
            already_tried_line += "I applied the shorter trip length, but the remaining fixed costs are still above the budget.\n\n"
        if dates_already_changed:
            already_tried_line += "I applied the changed dates, but the trip is still above the budget.\n\n"

        example_options = [f'"Increase budget to {suggested_budget_text}"']
        if not shortened_trip_already_applied:
            example_options.append(f'"Shorten the trip to {shorten_example}."')
        elif not paid_attractions_already_removed and not flight_exceeds_budget:
            example_options.append('"Remove paid attractions."')

        return (
            f"This trip is over budget.\n\n"
            f"{flight_line}\n\n"
            f"{budget_reason_line}\n\n"
            f"{applied_budget_line}"
            f"{already_tried_line}"
            "Options:\n"
            + "\n".join(option_lines)
            + "\n\n"
            "Reply with one option.\n"
            f"For example: {' or '.join(example_options)}"
        )

    parsed = result.get("parsed_request") or {}
    hint = _build_preference_hint(user_request, parsed)
    missing = (parsed.get("missing_fields") or {}).get("critical") or []
    labels = {
        "origin": "departure city",
        "destination_or_country": "destination city or country",
        "travel_dates_or_duration": "travel dates or trip length",
        "travel_dates.start": "exact departure date",
        "budget.total": "total budget",
    }
    missing_text = ", ".join(labels.get(item, item) for item in missing)
    if missing_text:
        if missing == ["origin"]:
            return (
                f"{hint}\n\n"
                "Please provide the departure city.\n\n"
                "For example: from Shanghai."
            )
        if missing == ["travel_dates.start"] and (parsed.get("duration_days") or (parsed.get("travel_dates") or {}).get("text")):
            return (
                f"{hint}\n\n"
                "The departure date is missing. If this trip is starting tomorrow, "
                "please reply with \"starting tomorrow\".\n\n"
                "If not, please provide the exact departure date in YYYY-MM-DD format.\n\n"
                f"For example: {example_departure_date}."
            )
        return (
            f"{hint}\n\n"
            f"Please provide {missing_text}.\n\n"
            f"For example: {example_start_date} to {example_end_date}, 2 people, 15000 CNY."
        )
    return (
        f"{hint}\n\n"
        "Please provide your departure city, destination, travel dates or trip length, traveler count, and budget."
    )


def _build_guidance_prompt(user_request: str, result: dict) -> str:
    default_example_date = _example_departure_date()
    return f"""
You are a friendly travel planning intake assistant.

The planner could not create a report from the user's request yet. Your job is to help the user revise it into a complete request for the downstream travel planning system.

If you need an example explicit departure date, use {default_example_date}.

Rules:
1. Reply in English.
2. Do not say the request is simply rejected.
3. Do not rewrite or normalize the user's wording, number style, punctuation, capitalization, or formatting.
4. Identify missing or unreasonable information in plain language.
5. Ask focused guiding questions only when required information is missing or unreasonable. Do not ask the user to confirm information that is already present.
6. Do not ask confirmation questions such as "Is this trip starting tomorrow?" or "Do you mean...?" unless the request is genuinely ambiguous and cannot proceed safely.
7. When one exact field is missing, ask directly for that field in one sentence, without restating the whole request unless absolutely necessary.
8. If you give an example departure date, use {default_example_date} by default. Do not give past dates.
9. Do not produce a rewritten full request template unless the user explicitly asks for help rewriting.
10. If the user is continuing a previous request or only added one missing detail, ask only for the next missing detail and keep the reply short.
11. Cover these required downstream information points when relevant:
   - departure city
   - destination city or country
   - travel dates or trip length
   - number of travelers
   - total budget and currency
   - useful preferences such as hotel level/location, food, transport, attractions, pace
12. Keep it concise, under 80 words when only one field is missing, otherwise under 140 words.
13. If the failure stage is "budget_feasibility", treat the user's budget as a hard cap. Say the system already tried cheaper flights, budget hotels, lower-cost meals, and fewer paid activities when applicable. Do not suggest accepting an over-budget plan. Ask the user to either increase the budget, shorten the trip, remove paid activities, or relax destination/date constraints.
14. When the user already gave meaningful preferences or intent, briefly offer 1 to 2 helpful planning suggestions before asking for missing details.
15. Do not respond with only an example. The user should feel you are actively helping shape the trip.

User request:
{user_request}

Planner failure result:
{_compact_json(result)}
""".strip()


def _build_guidance(user_request: str, result: dict) -> str:
    if result.get("stage") in {"agent4_itinerary_planning", "budget_feasibility"}:
        return _fallback_guidance(user_request, result)
    try:
        guidance = _call_llm(_build_guidance_prompt(user_request, result)).strip()
        return guidance or _fallback_guidance(user_request, result)
    except Exception:
        return _fallback_guidance(user_request, result)


def _run_intake_check(user_request: str) -> Dict[str, Any]:
    try:
        parsed_request = travel_requirement_parser_llm_agent(user_request)
        parse_status = str(parsed_request.get("parse_status") or "").lower()
        parse_error = ((parsed_request.get("meta") or {}).get("error") or "").strip()
        if parse_status == "failed" or parse_error:
            return {
                "success": True,
                "ready": True,
                "guidance": None,
                "parsed_request": parsed_request,
                "missing_fields": parsed_request.get("missing_fields", {}),
                "intake_error": parse_error or "Travel intake parsing failed.",
            }
        missing_fields = (parsed_request.get("missing_fields") or {}).get("critical") or []
        is_ready = bool(parsed_request.get("is_ready_for_planning")) and not missing_fields
        result = {
            "success": is_ready,
            "stage": "agent1_parse" if not is_ready else "intake_ready",
            "error": None if is_ready else "The request needs clarification before planning.",
            "parsed_request": parsed_request,
        }
        return {
            "success": True,
            "ready": is_ready,
            "guidance": None if is_ready else _build_guidance(user_request, result),
            "parsed_request": parsed_request,
            "missing_fields": parsed_request.get("missing_fields", {}),
        }
    except Exception as exc:
        result = {
            "success": False,
            "stage": "frontend_validation",
            "error": str(exc),
        }
        return {
            "success": True,
            "ready": True,
            "guidance": None,
            "parsed_request": {},
            "missing_fields": {"critical": [], "optional": []},
            "intake_error": str(exc),
        }


def _fallback_frontend_html() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TravelAI Planner</title>
  <style>
    :root { color-scheme: light; --ink:#18201f; --muted:#65716f; --line:#d9e0dd; --paper:#f7f8f4; --panel:#fff; --accent:#007c89; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, Helvetica, sans-serif; background:var(--paper); color:var(--ink); }
    main { min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px; }
    .shell { width:min(900px, 100%); }
    h1 { margin:0 0 8px; font-size:32px; }
    p { color:var(--muted); line-height:1.5; }
    form { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin-top:18px; }
    textarea { width:100%; min-height:140px; resize:vertical; border:1px solid var(--line); border-radius:8px; padding:12px; font:inherit; }
    button { margin-top:12px; border:0; border-radius:8px; background:var(--accent); color:#fff; padding:11px 16px; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.6; cursor:not-allowed; }
    .result { margin-top:16px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; white-space:pre-wrap; }
    a { color:var(--accent); font-weight:700; }
  </style>
</head>
<body>
  <main>
    <div class="shell">
      <h1>TravelAI Planner</h1>
      <p>This fallback page is served by Python because Node.js/npm is not available. It still calls the same router pipeline and opens the generated report.</p>
      <form id="form">
        <textarea id="message" placeholder="Describe your trip, for example: I want to travel from Beijing to Chengdu from 6.19 to 6.21 with a budget of 6000 CNY. I love local food."></textarea>
        <br>
        <button id="submit" type="submit">Generate plan</button>
      </form>
      <div id="result" class="result" style="display:none;"></div>
    </div>
  </main>
  <script>
    const form = document.getElementById('form');
    const message = document.getElementById('message');
    const submit = document.getElementById('submit');
    const result = document.getElementById('result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = message.value.trim();
      if (!text) return;
      submit.disabled = true;
      result.style.display = 'block';
      result.textContent = 'Running router pipeline and generating report...';
      try {
        const response = await fetch('/api/plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'Planning failed.');
        const trip = data.trip_summary || {};
        const destination = Array.isArray(trip.destination) ? trip.destination.join(', ') : '';
        result.innerHTML = [
          '<strong>Plan generated.</strong>',
          '',
          'Route: ' + (trip.origin || 'Origin') + ' -> ' + (destination || 'destination'),
          'Days: ' + (trip.duration_days || 'unknown'),
          data.report_url ? '<a href="' + data.report_url + '" target="_blank" rel="noreferrer">Open report</a>' : 'No report URL returned.'
        ].join('<br>');
      } catch (error) {
        result.textContent = 'The plan could not be generated.\\n\\n' + (error && error.message ? error.message : String(error));
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _report_url_from_result(result: Dict[str, Any]) -> Optional[str]:
    report = result.get("report") or {}
    if not isinstance(report, dict) or not report.get("success"):
        return None

    path_text = report.get("path") or report.get("file")
    if not path_text:
        return None

    path = Path(path_text)
    if not path.is_absolute():
        path = BASE_DIR / path

    try:
        relative = path.resolve().relative_to(REPORTS_DIR.resolve())
    except Exception:
        return None

    return f"http://{API_HOST}:{API_PORT}/reports/{relative.as_posix()}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_up_to_100(value: float) -> int:
    if value <= 0:
        return 0
    return int(((value + 99) // 100) * 100)


def _event_type(event: Dict[str, Any]) -> str:
    return str(event.get("type") or "").strip().lower()


def _hotel_cost_for_recovery_day(day: Dict[str, Any]) -> float:
    hotel = day.get("hotel") or {}
    if not isinstance(hotel, dict):
        return 0.0

    nightly = _safe_float(hotel.get("estimated_nightly_per_room"), 0.0)
    if nightly <= 0:
        nightly = _safe_float(hotel.get("realtime_nightly_per_room"), 0.0)
    if nightly > 0:
        return nightly

    total = _safe_float(hotel.get("estimated_total_cost"), 0.0)
    if total <= 0:
        return 0.0

    nights = 1
    check_in = str(day.get("check_in") or "").strip()
    check_out = str(day.get("check_out") or "").strip()
    try:
        if check_in and check_out:
            nights = max(1, (datetime.strptime(check_out, "%Y-%m-%d") - datetime.strptime(check_in, "%Y-%m-%d")).days)
        else:
            days_in_city = int(_safe_float(day.get("days_in_city"), 0.0))
            if days_in_city > 1:
                nights = max(1, days_in_city - 1)
    except Exception:
        nights = 1
    return total / float(max(1, nights))


def _traveler_count(result: Dict[str, Any]) -> int:
    trip = result.get("trip_summary") or {}
    travelers = trip.get("travelers") or trip.get("traveler_count") or trip.get("people")
    if isinstance(travelers, dict):
        travelers = travelers.get("total") or travelers.get("count")
    count = int(_safe_float(travelers, 1.0))
    return max(1, count)


def _rebuild_cost_rollup(daily_plan: list[dict]) -> Dict[str, float]:
    transport = 0.0
    food = 0.0
    attractions = 0.0
    hotels = 0.0
    for day in daily_plan:
        if not isinstance(day, dict):
            continue
        hotels += _hotel_cost_for_recovery_day(day)
        for event in day.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            cost = _safe_float(event.get("cost_estimated"), 0.0)
            kind = _event_type(event)
            if kind in {"meal", "restaurant"}:
                food += cost
            elif kind in {"poi", "attraction", "activity"}:
                attractions += cost
            elif kind in {"flight", "transfer", "drive", "rental_car", "intercity", "local_transport"}:
                transport += cost
    total = transport + food + attractions + hotels
    return {
        "transport": round(transport, 2),
        "food": round(food, 2),
        "attractions": round(attractions, 2),
        "hotel_reference": round(hotels, 2),
        "total_selected_estimated": round(total, 2),
        "total_cost": round(total, 2),
    }


def _cache_recoverable_budget_result(request_id: str, result: Dict[str, Any]) -> None:
    itinerary = result.get("itinerary") or {}
    daily_plan = itinerary.get("daily_plan")
    if result.get("stage") != "budget_feasibility" or not isinstance(daily_plan, list) or not daily_plan:
        return
    RECOVERABLE_BUDGET_RESULTS[request_id] = copy.deepcopy(result)
    RECOVERABLE_BUDGET_ORDER.append(request_id)
    while len(RECOVERABLE_BUDGET_ORDER) > 20:
        old_id = RECOVERABLE_BUDGET_ORDER.pop(0)
        RECOVERABLE_BUDGET_RESULTS.pop(old_id, None)


def _budget_recovery_actions(payload: Dict[str, Any]) -> list[str]:
    raw = payload.get("budget_recovery_actions")
    if isinstance(raw, list):
        actions = [str(item).strip().lower() for item in raw if str(item).strip()]
    else:
        actions = []
    return [
        item for item in actions
        if item in {"increase_budget", "lower_hotels", "lower_food", "remove_paid_attractions", "use_public_transport"}
    ]


def _apply_fast_budget_recovery(result: Dict[str, Any], actions: list[str], new_total_budget: Optional[float]) -> list[str]:
    notes: list[str] = []
    itinerary = result.setdefault("itinerary", {})
    daily_plan = itinerary.get("daily_plan")
    if not isinstance(daily_plan, list):
        return notes

    travelers = _traveler_count(result)
    action_set = set(actions)

    if "lower_hotels" in action_set:
        for day in daily_plan:
            if not isinstance(day, dict):
                continue
            hotel = day.get("hotel")
            if not isinstance(hotel, dict):
                continue
            old_cost = _safe_float(hotel.get("estimated_total_cost"), 0.0)
            if old_cost <= 0:
                continue
            hotel["estimated_total_cost"] = round(max(old_cost * 0.62, 380.0 * travelers), 2)
            hotel["name"] = hotel.get("name") or f"Budget hotel in {day.get('city') or 'destination city'}"
            hotel["fast_recovery_note"] = "Downgraded to a budget-level hotel estimate without rerunning hotel search."
        notes.append("Downgraded hotel estimates to budget-level accommodation.")

    if "lower_food" in action_set:
        for day in daily_plan:
            if not isinstance(day, dict):
                continue
            for event in day.get("events", []) or []:
                if not isinstance(event, dict) or _event_type(event) not in {"meal", "restaurant"}:
                    continue
                old_cost = _safe_float(event.get("cost_estimated"), 0.0)
                if old_cost <= 0:
                    continue
                budget_meal = 85.0 * travelers
                event["cost_estimated"] = round(min(old_cost * 0.55, budget_meal), 2)
                event.setdefault("metadata", {})
                if isinstance(event["metadata"], dict):
                    event["metadata"]["fast_recovery_note"] = "Downgraded to a budget/local meal estimate without rerunning restaurant search."
        notes.append("Downgraded meal estimates to budget local meals.")

    if "remove_paid_attractions" in action_set:
        for day in daily_plan:
            if not isinstance(day, dict):
                continue
            kept_events = []
            for event in day.get("events", []) or []:
                if isinstance(event, dict) and _event_type(event) in {"poi", "attraction", "activity"} and _safe_float(event.get("cost_estimated"), 0.0) > 0:
                    event["cost_estimated"] = 0
                    event.setdefault("metadata", {})
                    if isinstance(event["metadata"], dict):
                        event["metadata"]["fast_recovery_note"] = "Paid attraction removed from budget; use free exterior visit or nearby free alternative."
                kept_events.append(event)
            day["events"] = kept_events
        notes.append("Removed paid attraction costs and kept free/exterior alternatives.")

    if "use_public_transport" in action_set:
        for day in daily_plan:
            if not isinstance(day, dict):
                continue
            for event in day.get("events", []) or []:
                if not isinstance(event, dict) or _event_type(event) not in {"drive", "rental_car"}:
                    continue
                old_cost = _safe_float(event.get("cost_estimated"), 0.0)
                if old_cost > 0:
                    event["cost_estimated"] = round(max(old_cost * 0.35, 90.0 * travelers), 2)
                event["type"] = "intercity"
                name = str(event.get("name") or "").strip()
                event["name"] = name.replace("Drive", "Public transport").replace("Rental car", "Public transport") or "Public intercity transport"
                event.setdefault("metadata", {})
                if isinstance(event["metadata"], dict):
                    event["metadata"]["transport_mode"] = "public_intercity_transport"
                    event["metadata"]["fast_recovery_note"] = "Changed rental car/drive segment to public intercity transport without rerunning live searches."
        notes.append("Changed rental-car or drive segments to public intercity transport estimates.")

    itinerary["cost_rollup"] = _rebuild_cost_rollup(daily_plan)
    budget_total = new_total_budget
    trip_budget = result.setdefault("trip_summary", {}).setdefault("budget", {})
    if budget_total is None:
        budget_total = _safe_float(trip_budget.get("total"), 0.0)
    if budget_total and budget_total > 0:
        trip_budget["total"] = budget_total
        trip_budget.setdefault("currency", "CNY")

    estimated_total = _safe_float((itinerary.get("cost_rollup") or {}).get("total_selected_estimated"), 0.0)
    feasible = bool(budget_total and estimated_total <= budget_total)
    budget_check = itinerary.setdefault("budget_check", {})
    budget_check["budget_feasible"] = feasible
    budget_check["total_budget"] = budget_total
    budget_check["selected_total"] = estimated_total
    budget_check["delta"] = round(estimated_total - _safe_float(budget_total, 0.0), 2)
    summary = itinerary.setdefault("itinerary_summary", {})
    summary["budget_feasible"] = feasible
    summary["estimated_total_cost"] = estimated_total
    result["planning_baseline"] = {
        **(result.get("planning_baseline") or {}),
        "request": result.get("planning_baseline", {}).get("request") or "",
        "total_budget": budget_total,
        "selected_total": estimated_total,
    }
    return notes


def _recover_budget_plan(payload: Dict[str, Any], user_request: str) -> tuple[Dict[str, Any], int]:
    recover_id = str(payload.get("recover_budget_request_id") or "").strip()
    cached = RECOVERABLE_BUDGET_RESULTS.get(recover_id)
    if not cached:
        return {
            "success": False,
            "stage": "budget_recovery",
            "error": "The previous budget-failed plan is no longer available. Please rerun the full request.",
        }, 404

    new_total_budget = payload.get("new_total_budget")
    parsed_budget = _safe_float(new_total_budget, 0.0) if new_total_budget is not None else None
    if parsed_budget is not None and parsed_budget <= 0:
        parsed_budget = None

    actions = _budget_recovery_actions(payload)
    if parsed_budget is not None and "increase_budget" not in actions:
        actions.append("increase_budget")
    if not actions:
        return {
            "success": False,
            "stage": "budget_recovery",
            "error": "No fast budget recovery action was provided.",
        }, 400
    cached_minimum = _safe_float(cached.get("minimum_budget_required"), 0.0)
    if cached_minimum <= 0:
        cached_minimum = _safe_float((cached.get("budget_check") or {}).get("minimum_budget_required"), 0.0)
    if parsed_budget is not None and cached_minimum > 0 and parsed_budget > cached_minimum:
        return {
            "success": False,
            "stage": "budget_recovery",
            "error": "The new budget is above the previous minimum executable budget, so the full planning pipeline should rerun instead of reusing the minimum-cost recovery plan.",
        }, 404
    if actions == ["increase_budget"]:
        return {
            "success": False,
            "stage": "budget_recovery",
            "error": "Pure budget increases should rerun the full planning pipeline instead of reusing the previous minimum-cost recovery plan.",
        }, 404

    result = copy.deepcopy(cached)
    notes = _apply_fast_budget_recovery(result, actions, parsed_budget)
    itinerary = result.get("itinerary") or {}
    estimated_total = _safe_float((itinerary.get("cost_rollup") or {}).get("total_selected_estimated"), 0.0)
    budget_total = _safe_float(((result.get("trip_summary") or {}).get("budget") or {}).get("total"), 0.0)
    if not budget_total or estimated_total > budget_total:
        minimum = _round_up_to_100(estimated_total)
        result["success"] = False
        result["stage"] = "budget_feasibility"
        result["error"] = (
            f"Fast budget recovery still exceeds the budget: {round(estimated_total, 2)} CNY "
            f"against budget {round(budget_total, 2)} CNY."
        )
        result["minimum_budget_required"] = minimum
        result.setdefault("warnings", []).extend(notes)
        return result, 422

    result["success"] = True
    result["stage"] = "fast_budget_recovery"
    result["error"] = None
    result.setdefault("warnings", []).extend(notes)
    result["warnings"].append("Fast budget recovery reused the previous plan and regenerated the report without rerunning live searches.")
    result["report"] = write_travel_report(result, user_request, auto_open=False)
    return result, 200


class TravelApiHandler(BaseHTTPRequestHandler):
    server_version = "TravelAI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[api] " + fmt % args + "\n")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            data = _fallback_frontend_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/status":
            _json_response(self, 200, {
                "success": True,
                "status": get_router_backend_status(),
                "runtime": {
                    "api_process_started_at": API_PROCESS_STARTED_AT,
                    "api_code_version": API_CODE_VERSION,
                },
            })
            return

        if parsed.path.startswith("/reports/"):
            self._serve_report(parsed.path[len("/reports/"):])
            return

        _json_response(self, 404, {"success": False, "error": "Unknown endpoint."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        request_id = _new_request_id()
        if parsed.path not in {"/api/plan", "/api/guidance"}:
            _json_response(self, 404, {"success": False, "error": "Unknown endpoint.", "request_id": request_id})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            _json_response(self, 400, {"success": False, "error": "Invalid JSON body.", "request_id": request_id})
            return

        user_request = str(payload.get("message") or payload.get("request") or "").strip()
        refinement_context = payload.get("refinement_context")
        if not isinstance(refinement_context, dict):
            refinement_context = None
        if not user_request:
            _json_response(self, 400, {"success": False, "error": "Message is required.", "request_id": request_id})
            return

        sys.stdout.write(f"[api] [{request_id}] {parsed.path} started\n")

        if parsed.path == "/api/guidance":
            guidance_response = _run_intake_check(user_request)
            guidance_response["request_id"] = request_id
            _json_response(self, 200, guidance_response)
            sys.stdout.write(f"[api] [{request_id}] {parsed.path} finished\n")
            return

        try:
            if payload.get("recover_budget_request_id"):
                result, recovery_status = _recover_budget_plan(payload, user_request)
                if recovery_status == 404 and result.get("stage") == "budget_recovery":
                    result = run_budget_pipeline(user_request, auto_open_report=False, refinement_context=refinement_context)
                    recovery_status = None
                _cache_recoverable_budget_result(request_id, result)
            else:
                result = run_budget_pipeline(user_request, auto_open_report=False, refinement_context=refinement_context)
                recovery_status = None
                _cache_recoverable_budget_result(request_id, result)
            report_url = _report_url_from_result(result)
            success = bool(result.get("success"))
            itinerary = result.get("itinerary") or {}
            cost_rollup = itinerary.get("cost_rollup") or {}
            estimated_total = _safe_float(cost_rollup.get("total_selected_estimated"), 0.0)
            minimum_budget_required = result.get("minimum_budget_required")
            if result.get("stage") == "budget_feasibility" and estimated_total > 0:
                minimum_budget_required = minimum_budget_required or _round_up_to_100(estimated_total)
            response = {
                "request_id": request_id,
                "success": success,
                "stage": result.get("stage"),
                "error": result.get("error"),
                "guidance": None if success else _build_guidance(user_request, result),
                "warnings": result.get("warnings", []),
                "trip_summary": result.get("trip_summary", {}),
                "itinerary_summary": itinerary.get("itinerary_summary", {}),
                "refinement_state": result.get("refinement_state", {}),
                "upgrade_status": result.get("upgrade_status", []),
                "planning_baseline": result.get("planning_baseline", {}),
                "minimum_budget_required": minimum_budget_required,
                "recoverable_budget_request_id": request_id if result.get("stage") == "budget_feasibility" and estimated_total > 0 else None,
                "recoverable_estimated_total": estimated_total or None,
                "report": result.get("report", {}),
                "report_url": report_url,
                "runtime": {
                    "api_process_started_at": API_PROCESS_STARTED_AT,
                    "api_code_version": API_CODE_VERSION,
                    "router_runtime": result.get("runtime", {}),
                },
            }
            failure_status = 422 if response["stage"] in {"agent1_parse", "agent1_validation", "budget_feasibility", "refinement_non_decrease"} else 500
            _json_response(self, 200 if success else (recovery_status or failure_status), response)
            sys.stdout.write(
                f"[api] [{request_id}] {parsed.path} finished success={success} stage={response.get('stage')}\n"
            )
        except Exception as exc:
            sys.stdout.write(f"[api] [{request_id}] {parsed.path} failed error={exc}\n")
            _json_response(self, 500, {"success": False, "error": str(exc), "request_id": request_id})

    def _serve_report(self, report_path: str) -> None:
        safe_name = unquote(report_path).replace("\\", "/").lstrip("/")
        target = (REPORTS_DIR / safe_name).resolve()
        try:
            target.relative_to(REPORTS_DIR.resolve())
        except Exception:
            _json_response(self, 403, {"success": False, "error": "Forbidden report path."})
            return

        if not target.is_file():
            _json_response(self, 404, {"success": False, "error": "Report not found."})
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def _start_api_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((API_HOST, API_PORT), TravelApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _node_command(name: str) -> str | None:
    command = shutil.which(name)
    if command:
        return command
    if os.name == "nt":
        return shutil.which(f"{name}.cmd")
    return None


def _frontend_command() -> list[str]:
    pnpm = _node_command("pnpm")
    if pnpm:
        return [pnpm, "dev", "--hostname", "127.0.0.1", "--port", str(FRONTEND_PORT)]
    npm = _node_command("npm")
    if npm:
        return [npm, "run", "dev", "--", "--hostname", "127.0.0.1", "--port", str(FRONTEND_PORT)]
    raise RuntimeError("Neither pnpm nor npm was found. Please install Node.js and pnpm/npm.")


def _install_command() -> list[str]:
    pnpm = _node_command("pnpm")
    if pnpm:
        return [pnpm, "install"]
    npm = _node_command("npm")
    if npm:
        return [npm, "install"]
    raise RuntimeError("Neither pnpm nor npm was found. Please install Node.js and pnpm/npm.")


def _frontend_dependencies_ready() -> bool:
    node_modules_dir = FRONTEND_DIR / "node_modules"
    if not node_modules_dir.exists():
        return False

    next_package = node_modules_dir / "next" / "package.json"
    if not next_package.exists():
        return False

    bin_dir = node_modules_dir / ".bin"
    next_bin = bin_dir / ("next.cmd" if os.name == "nt" else "next")
    return next_bin.exists()


def _ensure_frontend_dependencies() -> None:
    if _frontend_dependencies_ready():
        return
    print("Frontend dependencies are missing or incomplete. Installing them now...")
    subprocess.check_call(_install_command(), cwd=str(FRONTEND_DIR))


def _wait_for_port(host: str, port: int, timeout_s: float = 45.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _start_frontend() -> subprocess.Popen[Any]:
    if not FRONTEND_DIR.exists():
        raise RuntimeError(f"Frontend folder not found: {FRONTEND_DIR}")

    _ensure_frontend_dependencies()
    command = _frontend_command()
    env = os.environ.copy()
    env["NEXT_PUBLIC_TRAVEL_API_BASE"] = f"http://{API_HOST}:{API_PORT}"
    return subprocess.Popen(command, cwd=str(FRONTEND_DIR), env=env)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    api_server = _start_api_server()
    frontend_process: Optional[subprocess.Popen[Any]] = None
    frontend_url = f"http://127.0.0.1:{FRONTEND_PORT}"
    fallback_url = f"http://{API_HOST}:{API_PORT}/"

    try:
        print(f"TravelAI API: http://{API_HOST}:{API_PORT}")
        try:
            if _wait_for_port("127.0.0.1", FRONTEND_PORT, timeout_s=1.0):
                print(f"TravelAI frontend already running: {frontend_url}")
                webbrowser.open(frontend_url)
            else:
                frontend_process = _start_frontend()
                print(f"TravelAI frontend: {frontend_url}")

                if _wait_for_port("127.0.0.1", FRONTEND_PORT):
                    webbrowser.open(frontend_url)
                else:
                    print("Frontend did not become ready in time. Open this URL after it finishes compiling:")
                    print(frontend_url)
        except RuntimeError as exc:
            print(f"Next frontend unavailable: {exc}")
            print(f"Opening Python fallback frontend: {fallback_url}")
            webbrowser.open(fallback_url)

        while True:
            if frontend_process is not None and frontend_process.poll() is not None:
                print("Frontend process exited. Keeping API server alive so an existing frontend can still connect.")
                frontend_process = None
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping TravelAI...")
    finally:
        api_server.shutdown()
        if frontend_process is not None:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()


if __name__ == "__main__":
    main()
