import copy
import json
import re
from datetime import datetime
from typing import Any, Dict, List


COUNTRY_NAME_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "korea": "south korea",
    "republic of korea": "south korea",
    "south korea": "south korea",
    "north korea": "north korea",
    "uae": "united arab emirates",
    "u.a.e.": "united arab emirates",
    "russia": "russian federation",
    "czech republic": "czechia",
    "viet nam": "vietnam",
}


def load_country_city_db(json_path="country_default_city_plan.json"):
    """
    Load the country-to-city mapping database from a JSON file.

    Expected format:
    {
        "japan": [
            {"city": "Tokyo", "days_ratio": 0.35},
            {"city": "Osaka", "days_ratio": 0.25}
        ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_country_name(text: str) -> str:
    """
    Normalize country name for dictionary lookup.
    """
    key = _normalize_text(text).lower()
    key = re.sub(r"\s+", " ", key)
    return COUNTRY_NAME_ALIASES.get(key, key)


def normalize_city_name(text: str) -> str:
    """
    Normalize a city name for comparison.
    """
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def compute_duration_days(travel_dates):
    """
    Automatically compute travel duration (inclusive of start and end date)
    if duration_days is not provided by the parser agent.

    Example:
    2026-05-01 to 2026-05-05 -> 5 days
    """
    if not isinstance(travel_dates, dict):
        return 0

    start = travel_dates.get("start")
    end = travel_dates.get("end")

    if not start or not end:
        return 0

    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        return (end_date - start_date).days + 1
    except Exception:
        return 0


def _get_duration_days(parsed_request: Dict[str, Any]) -> int:
    duration_days = _safe_int(parsed_request.get("duration_days"), 0)
    if duration_days > 0:
        return duration_days

    travel_dates = parsed_request.get("travel_dates", {}) or {}
    computed = compute_duration_days(travel_dates)
    return max(1, computed) if computed > 0 else 1


def _max_city_count_by_duration(total_days: int) -> int:
    total_days = max(1, total_days)

    if total_days <= 4:
        return 1
    if total_days <= 7:
        return 2
    if total_days <= 11:
        return 3
    if total_days <= 15:
        return 4
    return 5


def _allocate_days_by_ratio(total_days: int, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []

    total_days = max(1, total_days)

    if len(items) == 1:
        item = dict(items[0])
        item["days"] = total_days
        item.pop("days_ratio", None)
        item["source"] = _normalize_text(item.get("source")) or "system"
        return [item]

    raw_days = []
    for item in items:
        ratio = _safe_float(item.get("days_ratio"), 0.0)
        raw_days.append(max(0.0, ratio) * total_days)

    assigned = [max(1, int(x)) for x in raw_days]
    current_sum = sum(assigned)

    while current_sum > total_days:
        idx = max(range(len(assigned)), key=lambda i: assigned[i])
        if assigned[idx] > 1:
            assigned[idx] -= 1
            current_sum -= 1
        else:
            break

    while current_sum < total_days:
        idx = min(range(len(assigned)), key=lambda i: assigned[i])
        assigned[idx] += 1
        current_sum += 1

    result = []
    for i, item in enumerate(items):
        new_item = dict(item)
        new_item["city"] = _normalize_text(new_item.get("city"))
        new_item["days"] = assigned[i]
        new_item["source"] = _normalize_text(new_item.get("source")) or "system"
        new_item.pop("days_ratio", None)
        result.append(new_item)

    return result


def recommend_cities_by_country(country: str, duration_days: int, db: dict):
    """
    Recommend country default cities based on trip duration.

    New DB format:
    [
        {"city": "...", "days_ratio": ...},
        ...
    ]
    """
    country_key = normalize_country_name(country)
    candidate_cities = db.get(country_key, [])

    if not candidate_cities:
        return []

    city_count = _max_city_count_by_duration(duration_days)
    return candidate_cities[:city_count]


def is_country_input(destination_list, db: dict):
    """
    Determine whether the destination field actually contains a country name.

    Conditions:
    - destination must be a list
    - list must contain exactly one element
    - the element must exist in the country-city database
    """
    if not isinstance(destination_list, list) or not destination_list:
        return False

    if len(destination_list) != 1:
        return False

    item = normalize_country_name(destination_list[0])
    return item in db


def extract_explicit_city_days(user_request: str, destination_list):
    """
    Extract explicit city-day mapping from raw user request.

    Supported examples:
    - "New York for 3 days and Las Vegas for 5 days"
    - "go to Tokyo for 2 days, Osaka for 3 days"
    - "visit Seoul for 4 days and Busan for 2 days"

    Only cities already present in destination_list will be matched,
    to avoid extracting unrelated places.

    Return format:
    [
        {"city": "New York", "days": 3, "source": "user_explicit"},
        {"city": "Las Vegas", "days": 5, "source": "user_explicit"}
    ]
    """
    if not user_request or not isinstance(destination_list, list) or not destination_list:
        return []

    request_text = str(user_request)

    request_text = re.sub(r"(?i)\bdays?(from|to|with|and)\b", r"days \1", request_text)
    request_text = re.sub(r"(?i)\bday(from|to|with|and)\b", r"day \1", request_text)

    lowered_request = request_text.lower()
    results = []

    for city in destination_list:
        if not city:
            continue

        city_text = str(city).strip()
        city_key = normalize_city_name(city_text)
        city_pattern = re.escape(city_key)

        patterns = [
            rf"\b{city_pattern}\b\s*for\s*(\d+)\s*days?\b",
            rf"\b{city_pattern}\b\s+(\d+)\s*days?\b",
            rf"(?:stay in|visit|go to|travel to)\s+\b{city_pattern}\b\s*(?:for\s*)?(\d+)\s*days?\b",
        ]

        matched_days = None
        for pattern in patterns:
            match = re.search(pattern, lowered_request, flags=re.IGNORECASE)
            if match:
                matched_days = int(match.group(1))
                break

        if matched_days is not None:
            results.append({
                "city": city_text,
                "days": matched_days,
                "source": "user_explicit"
            })

    if not results:
        return []

    ordered = []
    matched_map = {normalize_city_name(item["city"]): item for item in results}

    for city in destination_list:
        key = normalize_city_name(city)
        if key in matched_map:
            ordered.append(matched_map[key])

    return ordered


def _get_effective_cities(parsed_request: Dict[str, Any], db: Dict[str, Any]) -> List[Dict[str, Any]]:
    city_plan = parsed_request.get("city_plan", {}) or {}
    planned_cities = city_plan.get("cities", [])

    total_days = _get_duration_days(parsed_request)
    max_city_count = _max_city_count_by_duration(total_days)

    def _trim_items(items: List[Any], max_count: int) -> List[Any]:
        return items[:max(1, min(len(items), max_count))]

    # 1) Highest priority: explicit city_plan
    if isinstance(planned_cities, list) and planned_cities:
        trimmed_planned_cities = _trim_items(planned_cities, max_city_count)

        ratio_items: List[Dict[str, Any]] = []
        explicit_day_values: List[int] = []
        all_have_days = True

        for item in trimmed_planned_cities:
            if isinstance(item, dict):
                city_name = _normalize_text(item.get("city"))
                if not city_name:
                    continue

                raw_days = item.get("days")
                day_val = _safe_int(raw_days, 0)
                if day_val <= 0:
                    all_have_days = False

                explicit_day_values.append(max(0, day_val))

                ratio_items.append({
                    "city": city_name,
                    "days_ratio": float(day_val) if day_val > 0 else 1.0,
                    "source": _normalize_text(item.get("source")) or "system"
                })
            else:
                city_name = _normalize_text(item)
                if not city_name:
                    continue
                all_have_days = False
                ratio_items.append({
                    "city": city_name,
                    "days_ratio": 1.0,
                    "source": "destination"
                })

        if ratio_items:
            if all_have_days and sum(explicit_day_values) > 0:
                total_explicit = float(sum(explicit_day_values))
                for item in ratio_items:
                    item["days_ratio"] = item["days_ratio"] / total_explicit
            else:
                equal_ratio = 1.0 / len(ratio_items)
                for item in ratio_items:
                    item["days_ratio"] = equal_ratio

            return _allocate_days_by_ratio(total_days, ratio_items)

    # 2) Second priority: explicit destination list
    dests = _ensure_list(parsed_request.get("destination"))
    if dests:
        trimmed_dests = _trim_items(dests, max_city_count)

        if len(trimmed_dests) == 1:
            single_item = trimmed_dests[0]

            # If the single destination is actually a country name, handle it in branch 3
            if normalize_country_name(single_item) in db:
                pass
            else:
                return [{
                    "city": _normalize_text(single_item),
                    "days": total_days,
                    "source": "destination"
                }]

        # If this looks like multiple explicit cities, allocate evenly.
        # If there is only one item and it is a country, skip to branch 3 below.
        if len(trimmed_dests) > 1 or normalize_country_name(trimmed_dests[0]) not in db:
            equal_ratio_items = []
            for city_name in trimmed_dests:
                equal_ratio_items.append({
                    "city": _normalize_text(city_name),
                    "days_ratio": 1.0 / len(trimmed_dests),
                    "source": "destination"
                })

            return _allocate_days_by_ratio(total_days, equal_ratio_items)

    # 3) Third priority: country default city plan
    country_raw = parsed_request.get("country")
    if not country_raw:
        dests = _ensure_list(parsed_request.get("destination"))
        if len(dests) == 1:
            candidate = normalize_country_name(dests[0])
            if candidate in db:
                country_raw = candidate

    country = normalize_country_name(country_raw)

    if country in db:
        default_items = db[country]
        trimmed_items = _trim_items(default_items, max_city_count)

        if not trimmed_items:
            return []

        normalized_default_items = []
        for item in trimmed_items:
            if not isinstance(item, dict):
                continue

            city_name = _normalize_text(item.get("city"))
            if not city_name:
                continue

            normalized_default_items.append({
                "city": city_name,
                "days_ratio": max(0.0, _safe_float(item.get("days_ratio"), 0.0)),
                "source": _normalize_text(item.get("source")) or "country_default"
            })

        if not normalized_default_items:
            return []

        total_ratio = sum(x.get("days_ratio", 0.0) for x in normalized_default_items)
        if total_ratio <= 0:
            equal_ratio = 1.0 / len(normalized_default_items)
            for item in normalized_default_items:
                item["days_ratio"] = equal_ratio
        else:
            for item in normalized_default_items:
                item["days_ratio"] = item["days_ratio"] / total_ratio

        return _allocate_days_by_ratio(total_days, normalized_default_items)

    return []


def _get_effective_primary_destination(parsed_request: Dict[str, Any], db: Dict[str, Any]) -> str:
    cities = _get_effective_cities(parsed_request, db)
    if cities:
        return (cities[0].get("city") or "").strip().lower()

    country = normalize_country_name(parsed_request.get("country"))
    return (country or "").strip().lower()


def _get_city_plan_mode(parsed_request: Dict[str, Any], db: Dict[str, Any]) -> str:
    city_plan = parsed_request.get("city_plan", {}) or {}
    mode = _normalize_text(city_plan.get("mode"))
    if mode:
        return mode

    if _get_effective_cities(parsed_request, db) and not _ensure_list(parsed_request.get("destination")):
        return "country_recommendation"

    return "unknown"


def build_city_plan(input_json: dict, db_path="country_default_city_plan.json"):
    """
    Generate a city-level travel plan based on structured travel input.

    Logic:
    1. If destination contains city names and explicit city-day mapping exists
       in user_request -> use that mapping directly
    2. If destination contains city names but no explicit city-day mapping ->
       allocate days evenly
    3. If destination contains a country name -> recommend cities automatically
    4. If destination is empty but country is provided -> recommend cities
    5. Append city_plan back into the original JSON structure
    """
    result = copy.deepcopy(input_json)
    db = load_country_city_db(db_path)

    destination = result.get("destination", [])
    country = result.get("country")
    duration_days = _safe_int(result.get("duration_days"), 0)

    user_request = (
        result.get("user_request")
        or result.get("raw_user_request")
        or result.get("meta", {}).get("user_request")
        or ""
    )

    if not duration_days or duration_days <= 0:
        duration_days = compute_duration_days(result.get("travel_dates", {}))
        result["duration_days"] = duration_days

    if not duration_days or duration_days <= 0:
        duration_days = 1
        result["duration_days"] = duration_days

    city_plan = {
        "mode": None,
        "cities": []
    }

    # Case 1: destination contains one country name
    if is_country_input(destination, db):
        country_name_raw = destination[0]
        country_key = normalize_country_name(country_name_raw)

        recommended_cities = recommend_cities_by_country(
            country_key,
            duration_days,
            db
        )

        city_plan["mode"] = "country_recommendation"
        city_plan["cities"] = _allocate_days_by_ratio(
            duration_days,
            [
                {
                    "city": _normalize_text(item.get("city")),
                    "days_ratio": _safe_float(item.get("days_ratio"), 0.0),
                    "source": _normalize_text(item.get("source")) or "country_default"
                }
                for item in recommended_cities
                if isinstance(item, dict) and _normalize_text(item.get("city"))
            ]
        )

        if not country:
            result["country"] = country_key

    # Case 2: explicit destination city list
    elif isinstance(destination, list) and len(destination) > 0:
        explicit_city_days = extract_explicit_city_days(user_request, destination)

        if explicit_city_days and len(explicit_city_days) == len(destination):
            explicit_total = sum(item["days"] for item in explicit_city_days)

            city_plan["mode"] = "user_explicit_city_days"
            city_plan["cities"] = explicit_city_days

            result["duration_days"] = explicit_total
        else:
            city_plan["mode"] = "user_specified"

            if len(destination) == 1:
                city_plan["cities"] = [{
                    "city": _normalize_text(destination[0]),
                    "days": result.get("duration_days", duration_days),
                    "source": "user"
                }]
            else:
                equal_ratio_items = []
                for city_name in destination[:_max_city_count_by_duration(result.get("duration_days", duration_days))]:
                    equal_ratio_items.append({
                        "city": _normalize_text(city_name),
                        "days_ratio": 1.0 / len(destination[:_max_city_count_by_duration(result.get("duration_days", duration_days))]),
                        "source": "user"
                    })

                city_plan["cities"] = _allocate_days_by_ratio(
                    result.get("duration_days", duration_days),
                    equal_ratio_items
                )

    # Case 3: destination empty but country exists
    elif country:
        country_key = normalize_country_name(country)
        recommended_cities = recommend_cities_by_country(
            country_key,
            duration_days,
            db
        )

        city_plan["mode"] = "country_recommendation"
        city_plan["cities"] = _allocate_days_by_ratio(
            duration_days,
            [
                {
                    "city": _normalize_text(item.get("city")),
                    "days_ratio": _safe_float(item.get("days_ratio"), 0.0),
                    "source": _normalize_text(item.get("source")) or "country_default"
                }
                for item in recommended_cities
                if isinstance(item, dict) and _normalize_text(item.get("city"))
            ]
        )

        result["country"] = country_key

    else:
        city_plan["mode"] = "unknown"
        city_plan["cities"] = []

    result["city_plan"] = city_plan
    result["primary_destination"] = _get_effective_primary_destination(result, db)
    return result


def run_city_planning_agent(parsed_json: dict, db_path="country_default_city_plan.json"):
    """
    Entry function for the City Planning Agent.

    This function is called by the main orchestrator
    after the parser agent produces structured travel data.
    """
    return build_city_plan(parsed_json, db_path=db_path)


if __name__ == "__main__":
    input_json = {
        "origin": "Beijing",
        "destination": ["New York", "Las Vegas"],
        "country": "",
        "num_people": 2,
        "travel_dates": {
            "start": None,
            "end": None,
            "text": None
        },
        "duration_days": 8,
        "budget": {
            "total": 20000,
            "currency": "CNY"
        },
        "preferences": {
            "flight": "direct",
            "hotel": "budget",
            "food": None,
            "attraction_style": None
        },
        "user_request": "My friend and I want to travel from Beijing to New York for 3 days and Las Vegas for 5 days"
    }

    output_json = run_city_planning_agent(
        input_json,
        db_path="country_default_city_plan.json"
    )

    print(json.dumps(output_json, ensure_ascii=False, indent=2))