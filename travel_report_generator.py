from __future__ import annotations

import html
import json
import math
import os
import re
import webbrowser
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "travel_reports"
CITY_CENTROIDS_PATH = BASE_DIR / "cities15000.txt"


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _esc(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except Exception:
        return default


def _money(value: Any, currency: str = "CNY") -> str:
    amount = _safe_float(value, 0.0)
    if abs(amount - round(amount)) < 0.005:
        return f"{int(round(amount))} {currency}"
    return f"{amount:.2f} {currency}"


def _flight_class_label(value: Any) -> str:
    text = _text(value).lower()
    if text in {"economy", "econ"}:
        return "Economy"
    if text in {"business", "biz"}:
        return "Business"
    if text in {"first", "first_class", "first class"}:
        return "First"
    if text:
        return text.replace("_", " ").title()
    return "Economy"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return text[:70] or "travel_report"


def _norm_key(*parts: Any) -> str:
    return "|".join(_text(p).lower().strip() for p in parts)


def _city_key(value: Any) -> str:
    text = _text(value).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(city|airport|international airport|intl airport)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _load_city_centroids() -> Dict[str, Dict[str, float]]:
    index: Dict[str, Dict[str, float]] = {}
    if not CITY_CENTROIDS_PATH.exists():
        return index

    try:
        with CITY_CENTROIDS_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 15:
                    continue

                names = [parts[1], parts[2]]
                if parts[3]:
                    names.extend(parts[3].split(","))

                try:
                    lat = float(parts[4])
                    lng = float(parts[5])
                    population = int(float(parts[14] or 0))
                except Exception:
                    continue

                record = {
                    "lat": lat,
                    "lng": lng,
                    "population": float(population),
                }
                for name in names:
                    key = _city_key(name)
                    if not key:
                        continue
                    existing = index.get(key)
                    if existing is None or record["population"] > existing.get("population", 0.0):
                        index[key] = record
    except Exception:
        return {}

    return index


def _city_coord(city: Any) -> Optional[Dict[str, float]]:
    record = _load_city_centroids().get(_city_key(city))
    if not record:
        return None
    return {"lat": float(record["lat"]), "lng": float(record["lng"])}


def _offset_coord(coord: Dict[str, float], index: int, radius: float = 0.006) -> Dict[str, float]:
    angle = (index * 137.508) * math.pi / 180.0
    ring = 1 + (index // 8)
    distance = radius * ring
    return {
        "lat": coord["lat"] + math.sin(angle) * distance,
        "lng": coord["lng"] + math.cos(angle) * distance,
    }


def _valid_coord(lat: Any, lng: Any) -> bool:
    lat_f = _safe_float(lat, 999.0)
    lng_f = _safe_float(lng, 999.0)
    return -90 <= lat_f <= 90 and -180 <= lng_f <= 180


def _coord(lat: Any, lng: Any) -> Optional[Dict[str, float]]:
    if not _valid_coord(lat, lng):
        return None
    return {"lat": _safe_float(lat), "lng": _safe_float(lng)}


def _build_place_index(router_output: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    index: Dict[str, Dict[str, float]] = {}

    for collection, name_field in [
        ("selected_hotels", "name"),
        ("selected_pois", "name"),
        ("selected_restaurants", "name"),
    ]:
        for item in router_output.get(collection, []) or []:
            if not isinstance(item, dict):
                continue
            city = _text(item.get("city"))
            name = _text(item.get(name_field))
            coord = _coord(item.get("lat"), item.get("lng"))
            if city and name and coord:
                index[_norm_key(city, name)] = coord
                index[_norm_key(name)] = coord

    itinerary = router_output.get("itinerary") or {}
    for day in itinerary.get("daily_plan", []) or []:
        if not isinstance(day, dict):
            continue
        city = _text(day.get("city"))
        hotel = day.get("hotel") or {}
        if isinstance(hotel, dict):
            name = _text(hotel.get("name"))
            coord = _coord(hotel.get("lat"), hotel.get("lng"))
            if city and name and coord:
                index[_norm_key(city, name)] = coord
                index[_norm_key(name)] = coord

    return index


def _event_coord(event: Dict[str, Any], city: str, place_index: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    meta = event.get("meta") or {}
    if isinstance(meta, dict):
        coord = _coord(meta.get("lat"), meta.get("lng"))
        if coord:
            return coord

    name = _text(event.get("name"))
    for prefix in ["Check-in ", "Check-out "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return (
        place_index.get(_norm_key(city, name))
        or place_index.get(_norm_key(name))
    )


def _append_city_route_point(
    points: List[Dict[str, Any]],
    city: Any,
    name: str,
    point_type: str,
    day: int = 0,
    date: str = "",
    cost: float = 0.0,
) -> None:
    city_name = _text(city)
    coord = _city_coord(city_name)
    if not city_name or not coord:
        return

    points.append({
        "lat": coord["lat"],
        "lng": coord["lng"],
        "day": day,
        "date": date,
        "city": city_name,
        "name": name or city_name,
        "type": point_type,
        "time": "",
        "cost": cost,
        "route_point": True,
    })


def _collect_city_route_points(router_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    trip = router_output.get("trip_summary") or {}
    origin = _text(trip.get("origin"))

    if origin:
        _append_city_route_point(points, origin, f"Start: {origin}", "origin")

    legs = router_output.get("selected_all_flight_legs") or []
    if not legs:
        legs = []
        route_plan = router_output.get("route_plan") or {}
        for key in ["outbound", "intercity", "inbound"]:
            value = route_plan.get(key)
            if isinstance(value, list):
                legs.extend(value)
            elif isinstance(value, dict) and value:
                legs.append(value)

    if legs:
        for idx, leg in enumerate(legs, start=1):
            if not isinstance(leg, dict):
                continue
            from_city = _text(leg.get("from_city"))
            to_city = _text(leg.get("to_city"))
            dep_date = _text(leg.get("departure_date"))
            mode = _text(leg.get("transport_mode")) or ("flight" if leg.get("selected_flights") else _text(leg.get("leg_type"), "transfer"))
            leg_type = _text(leg.get("leg_type")).lower()
            departure_code = _text((leg.get("selected_flights") or [{}])[0].get("departure_code")) if leg.get("selected_flights") else ""
            arrival_code = _text((leg.get("selected_flights") or [{}])[0].get("arrival_code")) if leg.get("selected_flights") else ""
            travel_class = ""
            flights = leg.get("selected_flights") or []
            if flights and isinstance(flights[0], dict):
                travel_class = _flight_class_label(flights[0].get("travel_class"))
            cost = _safe_float(leg.get("rental_car_cost_estimated"), 0.0)
            if cost <= 0:
                if flights and isinstance(flights[0], dict):
                    cost = _safe_float(flights[0].get("price_total"), 0.0)

            if mode == "rental_car":
                arrive_label = f"Drive {from_city} -> {to_city}"
            elif mode == "flight":
                from_label = f"{from_city} ({departure_code})" if departure_code else from_city
                to_label = f"{to_city} ({arrival_code})" if arrival_code else to_city
                class_suffix = f" [{travel_class}]" if travel_class else ""
                arrive_label = f"Fly {from_label} -> {to_label}{class_suffix}"
            else:
                arrive_label = f"Transfer {from_city} -> {to_city}"

            _append_city_route_point(
                points,
                to_city,
                arrive_label,
                mode,
                day=idx,
                date=dep_date,
                cost=cost,
            )
    else:
        for idx, item in enumerate(router_output.get("effective_cities", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            city = _text(item.get("city"))
            _append_city_route_point(points, city, f"{idx}. {city}", "city", day=idx)

    return _dedupe_consecutive_points(points)


def _dedupe_consecutive_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for point in points:
        if not deduped:
            deduped.append(point)
            continue
        prev = deduped[-1]
        if (
            abs(_safe_float(prev.get("lat")) - _safe_float(point.get("lat"))) < 0.00001
            and abs(_safe_float(prev.get("lng")) - _safe_float(point.get("lng"))) < 0.00001
            and _text(prev.get("city")).lower() == _text(point.get("city")).lower()
        ):
            prev_name = _text(prev.get("name"))
            point_name = _text(point.get("name"))
            if (
                (prev_name.startswith("Arrive ") or prev_name.startswith("Arrive in "))
                and (
                    point_name.startswith("Depart ")
                    or point_name.startswith("Fly from ")
                    or point_name.startswith("Drive from ")
                )
            ):
                prev["name"] = prev_name
            elif point_name:
                prev["name"] = point_name
            prev["type"] = point.get("type") or prev.get("type")
            prev["date"] = point.get("date") or prev.get("date")
            prev["cost"] = _safe_float(prev.get("cost"), 0.0) + _safe_float(point.get("cost"), 0.0)
            continue
        deduped.append(point)
    return deduped


def _spread_duplicate_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[int]] = {}
    for idx, point in enumerate(points):
        key = f"{_safe_float(point.get('lat')):.5f}|{_safe_float(point.get('lng')):.5f}"
        groups.setdefault(key, []).append(idx)

    for indexes in groups.values():
        if len(indexes) <= 1:
            continue
        base = {
            "lat": _safe_float(points[indexes[0]].get("lat")),
            "lng": _safe_float(points[indexes[0]].get("lng")),
        }
        for offset_index, point_index in enumerate(indexes):
            coord = _offset_coord(base, offset_index, radius=0.006)
            points[point_index]["lat"] = coord["lat"]
            points[point_index]["lng"] = coord["lng"]
            if offset_index > 0:
                points[point_index]["estimated_coord"] = (
                    bool(points[point_index].get("estimated_coord"))
                    or not bool(points[point_index].get("route_point"))
                )

    return points


def _route_sort_key(point: Dict[str, Any], index: int) -> Tuple[str, str, int]:
    date = _text(point.get("date"))
    time_text = _text(point.get("time"))
    start_time = time_text.split("-", 1)[0].strip() if time_text else ""

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        date = "0000-00-00" if _text(point.get("type")) == "origin" else "9999-99-99"
    if not re.match(r"^\d{2}:\d{2}$", start_time):
        start_time = "00:00"

    return date, start_time, index


def _sort_route_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        point
        for _, point in sorted(
            enumerate(points),
            key=lambda item: _route_sort_key(item[1], item[0]),
        )
    ]


def _collect_route_points(router_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    place_index = _build_place_index(router_output)
    itinerary = router_output.get("itinerary") or {}
    points: List[Dict[str, Any]] = _collect_city_route_points(router_output)
    fallback_counter_by_city: Dict[str, int] = {}

    for day_index, day in enumerate(itinerary.get("daily_plan", []) or [], start=1):
        if not isinstance(day, dict):
            continue
        city = _text(day.get("city"))
        date = _text(day.get("date"), f"Day {day_index}")
        for event_index, event in enumerate(day.get("events", []) or [], start=1):
            if not isinstance(event, dict):
                continue
            event_type = _text(event.get("type")).lower()
            if event_type in {"flight", "drive"}:
                continue
            coord = _event_coord(event, city, place_index)
            estimated_coord = False
            if not coord:
                base = _city_coord(city)
                if not base:
                    continue
                counter_key = _city_key(city)
                fallback_index = fallback_counter_by_city.get(counter_key, 0)
                fallback_counter_by_city[counter_key] = fallback_index + 1
                coord = _offset_coord(base, fallback_index)
                estimated_coord = True
            points.append({
                "lat": coord["lat"],
                "lng": coord["lng"],
                "day": day_index,
                "date": date,
                "city": city,
                "name": f"Day {day_index}.{event_index} {_text(event.get('name'), 'Stop')}",
                "type": event_type or "event",
                "time": f"{_text(event.get('start'))}-{_text(event.get('end'))}".strip("-"),
                "cost": _safe_float(event.get("cost_estimated"), 0.0),
                "estimated_coord": estimated_coord,
            })

    return _spread_duplicate_points(_sort_route_points(points))


def _day_total(day: Dict[str, Any], actual_checkin_dates: Optional[Dict[str, str]] = None) -> float:
    events_total = sum(
        _safe_float(ev.get("cost_estimated"), 0.0)
        for ev in day.get("events", []) or []
        if isinstance(ev, dict)
    )
    hotel_total = _hotel_cost_for_day(day, actual_checkin_dates)
    return events_total + hotel_total


def _has_event(day: Dict[str, Any], event_type: str) -> bool:
    target = _text(event_type)
    return any(
        isinstance(ev, dict) and _text(ev.get("type")) == target
        for ev in day.get("events", []) or []
    )


def _parse_report_date(value: Any) -> Optional[datetime]:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except Exception:
        return None


def _hotel_nights(day: Dict[str, Any]) -> int:
    check_in = _parse_report_date(day.get("check_in"))
    check_out = _parse_report_date(day.get("check_out"))
    if check_in and check_out and check_out > check_in:
        return max(1, (check_out - check_in).days)
    days = _safe_float(day.get("days_in_city"), 0.0)
    if days > 1:
        return max(1, int(days - 1))
    return 1


def _is_hotel_night(day: Dict[str, Any], actual_checkin_dates: Optional[Dict[str, str]] = None) -> bool:
    if _text(day.get("missing_required_outbound_transport")).lower() == "true":
        return False

    hotel = day.get("hotel") or {}
    if not isinstance(hotel, dict) or not _text(hotel.get("name")):
        return False

    day_date = _parse_report_date(day.get("date"))
    check_in = _parse_report_date(day.get("check_in"))
    check_out = _parse_report_date(day.get("check_out"))
    if day_date and check_in and check_out:
        actual_start_text = (actual_checkin_dates or {}).get(_text(day.get("city")).lower())
        actual_start = _parse_report_date(actual_start_text) if actual_start_text else None
        if actual_start and day_date < actual_start:
            return False
        return check_in <= day_date < check_out

    day_index = int(_safe_float(day.get("day_index_in_city"), 0))
    days_in_city = int(_safe_float(day.get("days_in_city"), 0))
    if day_index > 0 and days_in_city > 0:
        return day_index < days_in_city

    return _has_event(day, "checkin")


def _hotel_cost_for_day(day: Dict[str, Any], actual_checkin_dates: Optional[Dict[str, str]] = None) -> float:
    hotel = day.get("hotel") or {}
    if not isinstance(hotel, dict) or not _text(hotel.get("name")):
        return 0.0
    if not _is_hotel_night(day, actual_checkin_dates):
        return 0.0
    return _safe_float(hotel.get("estimated_total_cost"), 0.0) / max(1, _hotel_nights(day))


def _should_render_hotel_for_day(day: Dict[str, Any], actual_checkin_dates: Optional[Dict[str, str]] = None) -> bool:
    return _hotel_cost_for_day(day, actual_checkin_dates) > 0


def _actual_checkin_dates(daily_plan: List[Dict[str, Any]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for day in daily_plan:
        if not isinstance(day, dict):
            continue
        city_key = _text(day.get("city")).lower()
        if not city_key or city_key in result:
            continue
        for event in day.get("events", []) or []:
            if isinstance(event, dict) and _text(event.get("type")) == "checkin":
                result[city_key] = _text(day.get("date"))
                break
    return result


def _selection_status_note(status: Any) -> str:
    text = _text(status)
    if not text:
        return ""
    if "flight_date_outside_booking_window" in text:
        return "No live ticket was selected because the flight date is outside the next-six-month search window."
    if "flight_date_in_past" in text:
        return "No live ticket was selected because the requested flight date is in the past."
    if "no_flights_to_compare" in text or "no_candidates" in text:
        return "No live ticket candidate was returned by the flight search provider."
    if "missing_route_info" in text:
        return "No live ticket was selected because airport or route information was incomplete."
    if "flight_search_failed" in text:
        return "No live ticket was selected because the flight search request failed."
    return text


def _total_cost(router_output: Dict[str, Any]) -> float:
    itinerary = router_output.get("itinerary") or {}
    rollup = itinerary.get("cost_rollup") or {}
    if isinstance(rollup, dict) and _safe_float(rollup.get("total_selected_estimated"), 0.0) > 0:
        return _safe_float(rollup.get("total_selected_estimated"), 0.0)
    if isinstance(rollup, dict) and _safe_float(rollup.get("total"), 0.0) > 0:
        return _safe_float(rollup.get("total"), 0.0)
    return sum(
        _day_total(day)
        for day in itinerary.get("daily_plan", []) or []
        if isinstance(day, dict)
    )


def _render_day_cards(router_output: Dict[str, Any], currency: str) -> str:
    itinerary = router_output.get("itinerary") or {}
    cards: List[str] = []
    daily_plan = itinerary.get("daily_plan", []) or []
    checkin_dates = _actual_checkin_dates(daily_plan if isinstance(daily_plan, list) else [])

    for day_index, day in enumerate(daily_plan, start=1):
        if not isinstance(day, dict):
            continue

        events_html: List[str] = []
        for event in day.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            event_type = _text(event.get("type"), "event").replace("_", " ").title()
            meta = event.get("meta") or {}
            raw_event_type = _text(event.get("type"))
            if raw_event_type == "flight" and isinstance(meta, dict):
                event_type = f"Flight ({_flight_class_label(meta.get('travel_class'))})"
            event_cost = _safe_float(event.get("cost_estimated"), 0.0)
            if raw_event_type in {"flight_unavailable", "transfer_unavailable"}:
                time_label = "Needs selection"
            elif raw_event_type == "local_transport" and isinstance(meta, dict) and meta.get("all_day_cost"):
                time_label = "Daily budget"
            elif _text(event.get("start")) and _text(event.get("end")):
                time_label = f"{_text(event.get('start'))} - {_text(event.get('end'))}"
            else:
                time_label = _text(event.get("start")) or _text(event.get("end")) or ""
            if raw_event_type == "transfer" and event_cost <= 0:
                cost_label = "Included"
            else:
                cost_label = _money(event.get("cost_estimated"), currency)

            event_note = ""
            if raw_event_type in {"flight_unavailable", "transfer_unavailable"} and isinstance(meta, dict):
                note = _selection_status_note(meta.get("selection_status"))
                if note:
                    event_note = f"<small class=\"event-note\">{_esc(note)}</small>"

            events_html.append(
                "<li>"
                f"<span class=\"time\">{_esc(time_label)}</span>"
                f"<span class=\"event-name\">{_esc(event.get('name'))}{event_note}</span>"
                f"<span class=\"event-type\">{_esc(event_type)}</span>"
                f"<span class=\"cost\">{_esc(cost_label)}</span>"
                "</li>"
            )

        hotel = day.get("hotel") or {}
        hotel_html = ""
        if isinstance(hotel, dict) and _text(hotel.get("name")) and _should_render_hotel_for_day(day, checkin_dates):
            hotel_html = (
                "<div class=\"hotel-line\">"
                f"Hotel night: <strong>{_esc(hotel.get('name'))}</strong>"
                f"<span>{_esc(_money(_hotel_cost_for_day(day, checkin_dates), currency))}</span>"
                "</div>"
            )

        cards.append(
            "<section class=\"day-card\">"
            f"<div class=\"day-head\"><div><p>Day {day_index}</p>"
            f"<h2>{_esc(day.get('date'))} | {_esc(day.get('city'))}</h2></div>"
            f"<strong>{_esc(_money(_day_total(day, checkin_dates), currency))}</strong></div>"
            f"{hotel_html}"
            f"<ul class=\"event-list\">{''.join(events_html)}</ul>"
            "</section>"
        )

    return "".join(cards)


def _render_cost_rollup(router_output: Dict[str, Any], currency: str) -> str:
    itinerary = router_output.get("itinerary") or {}
    rollup = itinerary.get("cost_rollup") or {}
    if not isinstance(rollup, dict):
        rollup = {}

    labels = [
        ("transport", "Transport"),
        ("hotel_reference", "Hotels"),
        ("food", "Food"),
        ("attractions", "Attractions"),
        ("total_selected_estimated", "Total"),
    ]
    items = []
    for key, label in labels:
        if key in rollup:
            items.append(
                f"<div class=\"cost-chip\"><span>{_esc(label)}</span><strong>{_esc(_money(rollup.get(key), currency))}</strong></div>"
            )
    if not items:
        items.append(
            f"<div class=\"cost-chip total\"><span>Total</span><strong>{_esc(_money(_total_cost(router_output), currency))}</strong></div>"
        )
    return "".join(items)


def render_travel_report_html(router_output: Dict[str, Any], user_request: str) -> str:
    trip = router_output.get("trip_summary") or {}
    budget = trip.get("budget") or {}
    currency = _text(budget.get("currency"), "CNY")
    points = _collect_route_points(router_output)
    points_json = (
        json.dumps(points, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    destination = ", ".join(_text(x) for x in trip.get("destination", []) or [] if _text(x))
    total = _money(_total_cost(router_output), currency)
    budget_total = _money(budget.get("total"), currency) if budget.get("total") is not None else "Not specified"
    if budget.get("source_total") and budget.get("source_currency") and budget.get("source_currency") != currency:
        source_budget = _money(budget.get("source_total"), _text(budget.get("source_currency"), ""))
        budget_total = f"{source_budget} (approx. {budget_total} for planning)"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Travel Plan Report</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
  <style>
    :root {{
      --ink: #18201f;
      --muted: #64716e;
      --line: #d9e0dd;
      --paper: #f7f8f4;
      --panel: #ffffff;
      --accent: #007c89;
      --accent-2: #d64f35;
      --soft: #e6f2ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--paper);
      color: var(--ink);
      letter-spacing: 0;
    }}
    header {{
      padding: 32px 24px 24px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    .eyebrow {{ margin: 0 0 8px; color: var(--accent); font-weight: 700; }}
    h1 {{ margin: 0; font-size: 32px; line-height: 1.15; }}
    .request {{
      margin-top: 18px;
      padding: 14px 16px;
      border-left: 4px solid var(--accent);
      background: var(--soft);
      border-radius: 6px;
      color: #263331;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .summary div, .cost-chip {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .summary span, .cost-chip span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
    .summary strong, .cost-chip strong {{ font-size: 18px; }}
    main {{ padding: 24px; }}
    #map {{
      width: 100%;
      height: 520px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #dfe8e5;
    }}
    .map-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .map-controls button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: #ffffff;
      border-radius: 6px;
      padding: 8px 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .map-controls button.secondary {{
      background: #ffffff;
      color: var(--accent);
    }}
    .map-controls input[type="range"] {{
      flex: 1;
      min-width: 180px;
      accent-color: var(--accent);
    }}
    .step-panel {{
      margin-top: 10px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent-2);
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
    }}
    .step-panel strong {{ display: block; margin-bottom: 4px; }}
    .step-panel span {{ color: var(--muted); }}
    .step-cost {{ white-space: nowrap; font-weight: 700; }}
    .moving-dot {{
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--accent-2);
      border: 3px solid #ffffff;
      box-shadow: 0 0 0 4px rgba(214, 79, 53, 0.22), 0 2px 8px rgba(0,0,0,0.25);
    }}
    .section-title {{ margin: 30px 0 14px; font-size: 24px; }}
    .cost-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .cost-chip.total {{ border-color: var(--accent); }}
    .day-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
    }}
    .day-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
      margin-bottom: 12px;
    }}
    .day-head p {{ margin: 0 0 4px; color: var(--accent); font-weight: 700; }}
    .day-head h2 {{ margin: 0; font-size: 22px; }}
    .hotel-line {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 10px 0;
      color: var(--muted);
    }}
    .event-list {{ list-style: none; padding: 0; margin: 0; }}
    .event-list li {{
      display: grid;
      grid-template-columns: 110px minmax(0, 1fr) 120px 120px;
      gap: 12px;
      padding: 10px 0;
      border-top: 1px solid #eef2f0;
      align-items: center;
    }}
    .time, .event-type {{ color: var(--muted); font-size: 14px; }}
    .event-name {{ font-weight: 700; overflow-wrap: anywhere; }}
    .event-note {{ display: block; margin-top: 4px; color: var(--muted); font-weight: 400; line-height: 1.35; }}
    .cost {{ text-align: right; font-weight: 700; }}
    .empty-map {{
      padding: 14px;
      margin-top: 10px;
      color: var(--muted);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    @media (max-width: 820px) {{
      h1 {{ font-size: 26px; }}
      .summary, .cost-grid {{ grid-template-columns: 1fr 1fr; }}
      #map {{ height: 380px; }}
      .step-panel {{ flex-direction: column; }}
      .event-list li {{ grid-template-columns: 1fr; gap: 4px; }}
      .cost {{ text-align: left; }}
      .day-head, .hotel-line {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <p class="eyebrow">Travel Plan Report</p>
      <h1>{_esc(trip.get("origin"))} to {_esc(destination or "your destinations")}</h1>
      <div class="request"><strong>Your request:</strong> {_esc(user_request)}</div>
      <div class="summary">
        <div><span>Dates</span><strong>{_esc((trip.get("travel_dates") or {}).get("start"))} - {_esc((trip.get("travel_dates") or {}).get("end"))}</strong></div>
        <div><span>Travelers</span><strong>{_esc(trip.get("num_people"))}</strong></div>
        <div><span>Budget</span><strong>{_esc(budget_total)}</strong></div>
        <div><span>Estimated Spend</span><strong>{_esc(total)}</strong></div>
      </div>
    </div>
  </header>
  <main>
    <div class="wrap">
      <h2 class="section-title">Route Map</h2>
      <div id="map"></div>
      <div class="map-controls">
        <button id="play-route" type="button">Play</button>
        <button id="pause-route" class="secondary" type="button">Pause</button>
        <button id="replay-route" class="secondary" type="button">Replay</button>
        <button id="toggle-stops" class="secondary" type="button">Show all stops</button>
        <input id="route-step" type="range" min="0" value="0">
      </div>
      <div id="step-panel" class="step-panel">
        <div>
          <strong id="step-title">Route ready</strong>
          <span id="step-detail">Press Play to review the trip from start to finish.</span>
        </div>
        <div id="step-cost" class="step-cost"></div>
      </div>
      <div id="empty-map" class="empty-map" style="display:none;">No coordinates were available for the selected stops, so the map route could not be drawn.</div>

      <h2 class="section-title">Cost Summary</h2>
      <div class="cost-grid">{_render_cost_rollup(router_output, currency)}</div>

      <h2 class="section-title">Daily Checklist</h2>
      {_render_day_cards(router_output, currency)}
    </div>
  </main>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <script>
    const routePoints = {points_json};
    const map = L.map('map');
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    const playBtn = document.getElementById('play-route');
    const pauseBtn = document.getElementById('pause-route');
    const replayBtn = document.getElementById('replay-route');
    const toggleStopsBtn = document.getElementById('toggle-stops');
    const routeStep = document.getElementById('route-step');
    const stepTitle = document.getElementById('step-title');
    const stepDetail = document.getElementById('step-detail');
    const stepCost = document.getElementById('step-cost');
    let timer = null;
    let currentIndex = 0;
    let allStopsVisible = false;
    let allStopLayer = L.layerGroup();
    let completedLayer = L.layerGroup().addTo(map);
    let progressLine = null;
    let currentMarker = null;

    function pointLabel(p, idx) {{
      return `<strong>${{idx + 1}}. ${{p.name}}</strong><br>${{p.date}} | ${{p.city}}<br>${{p.time}} | ${{p.type}}<br>${{p.cost}} {currency}${{p.estimated_coord ? '<br><em>Approximate location</em>' : ''}}`;
    }}

    function updateStep(idx, pan = true) {{
      if (!routePoints.length) return;
      currentIndex = Math.max(0, Math.min(idx, routePoints.length - 1));
      routeStep.value = currentIndex;
      const p = routePoints[currentIndex];
      const activeLatLng = [p.lat, p.lng];
      const previous = currentIndex > 0 ? routePoints[currentIndex - 1] : null;
      const isLocalStop = !['origin', 'flight', 'rental_car', 'drive', 'transfer', 'intercity', 'outbound', 'inbound'].includes(String(p.type || '').toLowerCase());
      const sameCityAsPrevious = previous && String(previous.city || '').toLowerCase() === String(p.city || '').toLowerCase();
      const segmentLatLngs = (previous && (!p.estimated_coord || !sameCityAsPrevious))
        ? [[previous.lat, previous.lng], activeLatLng]
        : [activeLatLng];

      if (progressLine) progressLine.remove();
      progressLine = L.polyline(segmentLatLngs, {{ color: '#d64f35', weight: 5, opacity: 0.95 }}).addTo(map);

      completedLayer.clearLayers();
      if (previous) {{
        L.circleMarker([previous.lat, previous.lng], {{
          radius: 5,
          color: '#007c89',
          fillColor: '#ffffff',
          fillOpacity: 1,
          weight: 2
        }}).addTo(completedLayer).bindPopup(pointLabel(previous, currentIndex - 1));
      }}
      L.circleMarker(activeLatLng, {{
        radius: 8,
        color: '#d64f35',
        fillColor: '#d64f35',
        fillOpacity: 1,
        weight: 2
      }}).addTo(completedLayer).bindPopup(pointLabel(p, currentIndex));

      if (!currentMarker) {{
        currentMarker = L.marker(activeLatLng, {{
          icon: L.divIcon({{
            className: 'moving-dot',
            html: '',
            iconSize: [18, 18],
            iconAnchor: [9, 9]
          }})
        }}).addTo(map);
      }} else {{
        currentMarker.setLatLng(activeLatLng);
      }}
      currentMarker.bindPopup(pointLabel(p, currentIndex));
      stepTitle.textContent = `${{currentIndex + 1}} / ${{routePoints.length}} | ${{p.name}}`;
      stepDetail.textContent = `${{p.date || ''}} ${{p.time || ''}} | ${{p.city}} | ${{p.type}}${{p.estimated_coord ? ' | approximate location' : ''}}`;
      stepCost.textContent = `${{p.cost || 0}} {currency}`;
      if (pan) {{
        const targetZoom = isLocalStop ? (p.estimated_coord ? 13 : 16) : 10;
        map.flyTo(activeLatLng, targetZoom, {{ animate: true, duration: 1.35 }});
      }}
    }}

    function startAnimation() {{
      if (!routePoints.length) return;
      if (timer) clearInterval(timer);
      timer = setInterval(() => {{
        if (currentIndex >= routePoints.length - 1) {{
          clearInterval(timer);
          timer = null;
          return;
        }}
        updateStep(currentIndex + 1);
      }}, 2600);
    }}

    function pauseAnimation() {{
      if (timer) clearInterval(timer);
      timer = null;
    }}

    function replayAnimation() {{
      pauseAnimation();
      updateStep(0);
      startAnimation();
    }}

    function toggleAllStops() {{
      if (!routePoints.length) return;
      allStopsVisible = !allStopsVisible;
      toggleStopsBtn.textContent = allStopsVisible ? 'Hide all stops' : 'Show all stops';
      allStopLayer.clearLayers();
      if (!allStopsVisible) {{
        map.removeLayer(allStopLayer);
        return;
      }}
      routePoints.forEach((p, idx) => {{
        L.circleMarker([p.lat, p.lng], {{
          radius: 5,
          color: '#18201f',
          fillColor: p.estimated_coord ? '#f6c945' : '#ffffff',
          fillOpacity: 0.9,
          weight: 1.5
        }}).addTo(allStopLayer).bindPopup(pointLabel(p, idx));
      }});
      allStopLayer.addTo(map);
    }}

    if (!routePoints.length) {{
      map.setView([20, 0], 2);
      document.getElementById('empty-map').style.display = 'block';
    }} else {{
      const latLngs = routePoints.map(p => [p.lat, p.lng]);
      routeStep.max = String(routePoints.length - 1);
      false && routePoints.forEach((p, idx) => {{
        L.marker([p.lat, p.lng]).addTo(map).bindPopup(
          `<strong>${{idx + 1}}. ${{p.name}}</strong><br>${{p.date}} | ${{p.city}}<br>${{p.time}} | ${{p.type}}<br>${{p.cost}} {currency}${{p.estimated_coord ? '<br><em>Approximate location</em>' : ''}}`
        );
      }});
      if (latLngs.length > 1) {{
        L.polyline(latLngs, {{ color: '#007c89', weight: 3, opacity: 0.28, dashArray: '8 8' }}).addTo(map);
      }}
      map.fitBounds(latLngs, {{ padding: [30, 30] }});
      updateStep(0, false);
      map.setView(latLngs[0], 10);
    }}
    playBtn.addEventListener('click', startAnimation);
    pauseBtn.addEventListener('click', pauseAnimation);
    replayBtn.addEventListener('click', replayAnimation);
    toggleStopsBtn.addEventListener('click', toggleAllStops);
    routeStep.addEventListener('input', (event) => {{
      pauseAnimation();
      updateStep(Number(event.target.value));
    }});
  </script>
</body>
</html>"""


def write_travel_report(
    router_output: Dict[str, Any],
    user_request: str,
    output_dir: Optional[Path] = None,
    auto_open: bool = False,
) -> Dict[str, Any]:
    output_root = output_dir or REPORTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    trip = router_output.get("trip_summary") or {}
    destination = "_".join(_text(x) for x in trip.get("destination", []) or [] if _text(x))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{_slug(destination)}.html"
    path = output_root / filename
    html_text = render_travel_report_html(router_output, user_request)
    path.write_text(html_text, encoding="utf-8")
    opened = False
    if auto_open:
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(path.resolve()))
                opened = True
            else:
                opened = bool(webbrowser.open(path.resolve().as_uri()))
        except Exception:
            opened = False

    return {
        "success": True,
        "path": str(path),
        "filename": filename,
        "route_points": len(_collect_route_points(router_output)),
        "opened": opened,
    }
