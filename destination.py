import copy
import json
import math
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE_DIR = Path(__file__).resolve().parent
CITY_CENTROIDS_PATH = BASE_DIR / "cities15000.txt"
AIRPORT_MAPPING_PATH = BASE_DIR / "global_airports.json"
GOOGLE_DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
NEARBY_AIRPORT_RADIUS_KM = 220.0


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
    - "Stay 2 nights in Tokyo, 2 nights in Kyoto, and 2 nights in Osaka"
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
            (rf"\b{city_pattern}\b\s*for\s*(\d+)\s*days?\b", "days"),
            (rf"\b{city_pattern}\b\s+(\d+)\s*days?\b", "days"),
            (rf"(?:stay in|visit|go to|travel to)\s+\b{city_pattern}\b\s*(?:for\s*)?(\d+)\s*days?\b", "days"),
            (rf"\b{city_pattern}\b\s*for\s*(\d+)\s*nights?\b", "nights"),
            (rf"\b{city_pattern}\b\s+(\d+)\s*nights?\b", "nights"),
            (rf"(?:stay in|staying in)\s+\b{city_pattern}\b\s*(?:for\s*)?(\d+)\s*nights?\b", "nights"),
            (rf"(?:stay|staying)\s+(\d+)\s*nights?\s+in\s+\b{city_pattern}\b", "nights"),
            (rf"\b(\d+)\s*nights?\s+in\s+\b{city_pattern}\b", "nights"),
        ]

        matched_days = None
        matched_unit = "days"
        for pattern, unit in patterns:
            match = re.search(pattern, lowered_request, flags=re.IGNORECASE)
            if match:
                matched_days = int(match.group(1))
                matched_unit = unit
                break

        if matched_days is not None:
            results.append({
                "city": city_text,
                "days": matched_days,
                "source": "user_explicit",
                "unit": matched_unit,
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
        explicit_mode = _normalize_text(city_plan.get("mode")) in {"user_specified", "user_explicit_city_days"}
        trimmed_planned_cities = planned_cities if explicit_mode else _trim_items(planned_cities, max_city_count)

        ratio_items: List[Dict[str, Any]] = []
        explicit_day_values: List[int] = []
        all_have_days = True
        direct_explicit_nights: List[Dict[str, Any]] = []

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
                if explicit_mode and _normalize_text(item.get("unit")) == "nights" and day_val > 0:
                    direct_explicit_nights.append({
                        "city": city_name,
                        "days": day_val,
                        "source": _normalize_text(item.get("source")) or "user_explicit",
                        "unit": "nights",
                    })

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

        if direct_explicit_nights and len(direct_explicit_nights) == len(ratio_items):
            return direct_explicit_nights

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


def _normalize_alias_key(text: Any) -> str:
    text = _normalize_text(text).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_route_city_key(text: Any) -> str:
    text = _normalize_alias_key(text)
    text = re.sub(r"\b(city|airport|international airport|intl airport)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _geonames_feature_rank(feature_code: Any) -> int:
    rank_map = {
        "PPLC": 6,
        "PPLA": 5,
        "PPLA2": 4,
        "PPLA3": 3,
        "PPLA4": 2,
        "PPL": 1,
        "PPLX": 0,
    }
    return rank_map.get(_normalize_text(feature_code).upper(), -1)


@lru_cache(maxsize=1)
def _load_city_centroids() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    if not CITY_CENTROIDS_PATH.exists():
        return index

    try:
        with CITY_CENTROIDS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or _normalize_text(parts[6]).upper() != "P":
                    continue
                try:
                    lat = float(parts[4])
                    lng = float(parts[5])
                except Exception:
                    continue

                record = {
                    "name": _normalize_text(parts[1]),
                    "asciiname": _normalize_text(parts[2]),
                    "lat": lat,
                    "lng": lng,
                    "feature_code": _normalize_text(parts[7]).upper(),
                    "country_code": _normalize_text(parts[8]).upper(),
                    "population": _safe_int(parts[14], 0) if len(parts) > 14 else 0,
                }
                aliases = {_normalize_alias_key(record["name"]), _normalize_alias_key(record["asciiname"])}
                aliases.update(_normalize_alias_key(x) for x in (_normalize_text(parts[3]) or "").split(","))
                for alias in aliases:
                    if not alias:
                        continue
                    existing = index.get(alias)
                    if existing is None:
                        index[alias] = record
                        continue
                    old_rank = _geonames_feature_rank(existing.get("feature_code"))
                    new_rank = _geonames_feature_rank(record.get("feature_code"))
                    if (new_rank > old_rank) or (
                        new_rank == old_rank
                        and record["population"] > _safe_int(existing.get("population"), 0)
                    ):
                        index[alias] = record
    except Exception:
        return {}

    return index


def _get_city_centroid(city: Any) -> Optional[Dict[str, Any]]:
    key = _normalize_alias_key(city)
    if not key:
        return None
    return _load_city_centroids().get(key)


def _city_coord(city: Any) -> Optional[Dict[str, float]]:
    centroid = _get_city_centroid(city)
    if not centroid:
        return None
    lat = _safe_float(centroid.get("lat"), 0.0)
    lng = _safe_float(centroid.get("lng"), 0.0)
    if lat == 0.0 and lng == 0.0:
        return None
    return {"lat": lat, "lng": lng}


def _haversine_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _coord_key(coord: Dict[str, float]) -> str:
    return f"{_safe_float(coord.get('lat'), 0.0):.5f},{_safe_float(coord.get('lng'), 0.0):.5f}"


def _distance_between_coords(origin: Dict[str, float], destination: Dict[str, float]) -> float:
    return _haversine_distance_km(
        _safe_float(origin.get("lat"), 0.0),
        _safe_float(origin.get("lng"), 0.0),
        _safe_float(destination.get("lat"), 0.0),
        _safe_float(destination.get("lng"), 0.0),
    )


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _airport_type_rank(airport_type: Any) -> int:
    return {
        "large_airport": 5,
        "medium_airport": 4,
        "small_airport": 3,
    }.get(_normalize_text(airport_type).lower(), 0)


@lru_cache(maxsize=1)
def _load_airports_by_iata() -> Dict[str, Dict[str, Any]]:
    if not AIRPORT_MAPPING_PATH.exists():
        return {}

    try:
        with AIRPORT_MAPPING_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return result

    for iata_key, airport in raw.items():
        if not isinstance(airport, dict):
            continue
        iata = _normalize_text(airport.get("iata") or iata_key).upper()
        lat = _safe_float(airport.get("lat"), 0.0)
        lng = _safe_float(airport.get("lng"), 0.0)
        airport_type = _normalize_text(airport.get("type")).lower()
        if not iata or (lat == 0.0 and lng == 0.0) or airport_type not in {"large_airport", "medium_airport", "small_airport"}:
            continue
        result[iata] = {
            "iata": iata,
            "name": _normalize_text(airport.get("name")),
            "city": _normalize_text(airport.get("city")),
            "country": _normalize_text(airport.get("country")).upper(),
            "lat": lat,
            "lng": lng,
            "type": airport_type,
        }
    return result


@lru_cache(maxsize=1)
def _build_city_airport_index() -> Dict[str, List[Dict[str, Any]]]:
    city_index: Dict[str, List[Dict[str, Any]]] = {}
    for airport in _load_airports_by_iata().values():
        key = _normalize_route_city_key(airport.get("city"))
        if key:
            city_index.setdefault(key, []).append(airport)

    for airports in city_index.values():
        airports.sort(key=lambda x: (-_airport_type_rank(x.get("type")), _normalize_text(x.get("name"))))
    return city_index


def _get_direct_city_airports(city: Any) -> List[Dict[str, Any]]:
    city_key = _normalize_route_city_key(city)
    if not city_key:
        return []
    city_index = _build_city_airport_index()
    direct = list(city_index.get(city_key, []))
    if direct:
        return direct

    matched: Dict[str, Dict[str, Any]] = {}
    for indexed_key, airports in city_index.items():
        if city_key in indexed_key or indexed_key in city_key:
            for airport in airports:
                iata = _normalize_text(airport.get("iata")).upper()
                if iata:
                    matched[iata] = airport
    result = list(matched.values())
    result.sort(key=lambda x: (-_airport_type_rank(x.get("type")), _normalize_text(x.get("name"))))
    return result


def _coord_for_place(place: Any) -> Optional[Dict[str, float]]:
    text = _normalize_text(place)
    if not text:
        return None
    city_coord = _city_coord(text)
    if city_coord:
        return city_coord
    airport = _load_airports_by_iata().get(text.upper())
    if airport:
        return {"lat": _safe_float(airport.get("lat"), 0.0), "lng": _safe_float(airport.get("lng"), 0.0)}
    return None


def _find_gateway_for_city(city: Any, selected_city_keys: Dict[str, str]) -> Dict[str, Any]:
    city_name = _normalize_text(city)
    city_key = _normalize_route_city_key(city_name)
    direct_airports = _get_direct_city_airports(city_name)
    if direct_airports:
        return {
            "gateway_city": city_name,
            "gateway_city_key": city_key,
            "airport": direct_airports[0],
            "uses_other_selected_city_airport": False,
            "has_direct_airport": True,
        }

    city_coord = _city_coord(city_name)
    if not city_coord:
        return {
            "gateway_city": city_name,
            "gateway_city_key": city_key,
            "airport": None,
            "uses_other_selected_city_airport": False,
            "has_direct_airport": False,
        }

    city_country = _normalize_text((_get_city_centroid(city_name) or {}).get("country_code")).upper()
    nearby: List[Dict[str, Any]] = []
    for airport in _load_airports_by_iata().values():
        airport_city_key = _normalize_route_city_key(airport.get("city"))
        if not airport_city_key:
            continue
        if city_country and _normalize_text(airport.get("country")).upper() != city_country:
            continue
        distance_km = _haversine_distance_km(
            city_coord["lat"],
            city_coord["lng"],
            _safe_float(airport.get("lat"), 0.0),
            _safe_float(airport.get("lng"), 0.0),
        )
        if distance_km <= NEARBY_AIRPORT_RADIUS_KM:
            nearby.append({
                "airport": airport,
                "airport_city_key": airport_city_key,
                "distance_km": distance_km,
            })

    nearby.sort(key=lambda x: (x["distance_km"], -_airport_type_rank((x.get("airport") or {}).get("type"))))
    for candidate in nearby:
        gateway_key = _normalize_text(candidate.get("airport_city_key"))
        if gateway_key in selected_city_keys and gateway_key != city_key:
            return {
                "gateway_city": selected_city_keys[gateway_key],
                "gateway_city_key": gateway_key,
                "airport": candidate.get("airport"),
                "airport_distance_km": round(candidate["distance_km"], 2),
                "uses_other_selected_city_airport": True,
                "has_direct_airport": False,
            }

    if nearby:
        candidate = nearby[0]
        airport = candidate.get("airport") or {}
        return {
            "gateway_city": _normalize_text(airport.get("city")) or city_name,
            "gateway_city_key": candidate.get("airport_city_key") or city_key,
            "airport": airport,
            "airport_distance_km": round(candidate["distance_km"], 2),
            "uses_other_selected_city_airport": False,
            "has_direct_airport": False,
        }

    return {
        "gateway_city": city_name,
        "gateway_city_key": city_key,
        "airport": None,
        "uses_other_selected_city_airport": False,
        "has_direct_airport": False,
    }


def _distance_map_from_origin(
    origin_coord: Dict[str, float],
    destination_coords: List[Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {
        _coord_key(coord): {
            "distance_km": round(_distance_between_coords(origin_coord, coord), 2),
            "source": "haversine_fallback",
        }
        for coord in destination_coords
    }
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key or not _env_bool("TRAVEL_CITY_ORDER_USE_MAP_API", True):
        return result

    try:
        response = requests.get(
            GOOGLE_DISTANCE_MATRIX_URL,
            params={
                "origins": f"{origin_coord['lat']},{origin_coord['lng']}",
                "destinations": "|".join(f"{coord['lat']},{coord['lng']}" for coord in destination_coords),
                "mode": "driving",
                "key": api_key,
            },
            timeout=5,
        )
        if response.status_code != 200:
            return result
        data = response.json() if response.text else {}
        rows = data.get("rows") if isinstance(data, dict) else []
        elements = rows[0].get("elements", []) if rows and isinstance(rows[0], dict) else []
        for idx, element in enumerate(elements[:len(destination_coords)]):
            if not isinstance(element, dict) or element.get("status") != "OK":
                continue
            distance_m = _safe_float((element.get("distance") or {}).get("value"), 0.0)
            if distance_m > 0:
                coord = destination_coords[idx]
                result[_coord_key(coord)] = {
                    "distance_km": round(distance_m / 1000.0, 2),
                    "source": "google_distance_matrix",
                }
    except Exception:
        pass
    return result


def _request_has_explicit_route_sequence(user_request: Any, city_names: List[str]) -> bool:
    text = _normalize_text(user_request)
    if not text or len(city_names) < 2:
        return False
    markers = [
        r"(?i)\b(first|second|third|then|next|after that|finally|last)\b",
        r"(?i)\b(before|after)\b",
        r"\u7b2c\s*[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\d]+\s*\u7ad9",
        r"\u5148\u53bb|\u5148\u5230|\u7136\u540e|\u518d\u53bb|\u63a5\u7740|\u4e4b\u540e|\u6700\u540e",
    ]
    if not any(re.search(pattern, text) for pattern in markers):
        return False
    lowered = text.lower()
    return sum(1 for city in city_names if _normalize_text(city).lower() in lowered) >= 2


def _route_order_city_items(
    city_items: List[Dict[str, Any]],
    origin: Any,
    city_plan_mode: Any,
    user_request: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    original_items = [dict(item) for item in city_items if isinstance(item, dict)]
    original_order = [_normalize_text(item.get("city")) for item in original_items if _normalize_text(item.get("city"))]
    meta: Dict[str, Any] = {
        "applied": False,
        "strategy": "gateway_aware_far_to_near_from_origin",
        "origin": _normalize_text(origin),
        "original_order": original_order,
        "optimized_order": original_order,
    }

    if len(original_items) < 2 or len(original_order) < 2:
        meta["reason"] = "single_city_or_empty"
        return original_items, meta
    if _normalize_text(city_plan_mode) == "user_explicit_city_days":
        meta["reason"] = "explicit_city_day_or_night_allocation"
        return original_items, meta
    if _request_has_explicit_route_sequence(user_request, original_order):
        meta["reason"] = "explicit_user_sequence_detected"
        return original_items, meta

    origin_coord = _coord_for_place(origin)
    if not origin_coord:
        meta["reason"] = "origin_coordinate_unavailable"
        return original_items, meta

    selected_city_keys: Dict[str, str] = {}
    for city in original_order:
        city_key = _normalize_route_city_key(city)
        if city_key and city_key not in selected_city_keys:
            selected_city_keys[city_key] = city

    records: List[Dict[str, Any]] = []
    anchor_coords: Dict[str, Dict[str, float]] = {}
    for index, item in enumerate(original_items):
        city_name = _normalize_text(item.get("city"))
        if not city_name:
            continue
        city_key = _normalize_route_city_key(city_name)
        gateway = _find_gateway_for_city(city_name, selected_city_keys)
        gateway_city = _normalize_text(gateway.get("gateway_city")) or city_name
        anchor_coord = _city_coord(gateway_city) or _city_coord(city_name)
        airport = gateway.get("airport") if isinstance(gateway.get("airport"), dict) else None
        if not anchor_coord and airport:
            anchor_coord = {"lat": _safe_float(airport.get("lat"), 0.0), "lng": _safe_float(airport.get("lng"), 0.0)}
        if anchor_coord:
            anchor_coords[_coord_key(anchor_coord)] = anchor_coord
        gateway_key = _normalize_text(gateway.get("gateway_city_key")) or city_key
        records.append({
            "index": index,
            "item": item,
            "city": city_name,
            "city_key": city_key,
            "gateway": gateway,
            "gateway_city": gateway_city,
            "gateway_key": gateway_key,
            "anchor_coord": anchor_coord,
            "gateway_rank": 1 if gateway.get("uses_other_selected_city_airport") else 0,
        })

    distance_map = _distance_map_from_origin(origin_coord, list(anchor_coords.values()))
    sortable: List[Dict[str, Any]] = []
    unsortable: List[Dict[str, Any]] = []
    for record in records:
        coord = record.get("anchor_coord")
        if not coord:
            unsortable.append(record)
            continue
        distance_info = distance_map.get(_coord_key(coord))
        if not distance_info:
            unsortable.append(record)
            continue
        record["distance_km"] = _safe_float(distance_info.get("distance_km"), 0.0)
        record["distance_source"] = _normalize_text(distance_info.get("source")) or "unknown"
        sortable.append(record)

    if len(sortable) < 2:
        meta["reason"] = "insufficient_distance_data"
        return original_items, meta

    group_distances: Dict[str, float] = {}
    for record in sortable:
        group_key = _normalize_text(record.get("gateway_key")) or _normalize_text(record.get("city_key"))
        group_distances[group_key] = max(
            group_distances.get(group_key, 0.0),
            _safe_float(record.get("distance_km"), 0.0),
        )

    sortable.sort(
        key=lambda record: (
            -group_distances.get(_normalize_text(record.get("gateway_key")), _safe_float(record.get("distance_km"), 0.0)),
            _safe_int(record.get("gateway_rank"), 0),
            _safe_int(record.get("index"), 0),
        )
    )

    ordered_records = sortable + unsortable
    ordered_items = [dict(record["item"]) for record in ordered_records]
    optimized_order = [_normalize_text(item.get("city")) for item in ordered_items if _normalize_text(item.get("city"))]
    meta["optimized_order"] = optimized_order
    if optimized_order != original_order:
        meta["applied"] = True
    else:
        meta["reason"] = "already_ordered"

    sources = sorted({
        _normalize_text(record.get("distance_source"))
        for record in sortable
        if _normalize_text(record.get("distance_source"))
    })
    meta.update({
        "distance_unit": "km",
        "distance_source": sources[0] if len(sources) == 1 else "mixed",
        "gateway_city_priority": True,
        "cities": [
            {
                "city": record.get("city"),
                "distance_from_origin_km": (
                    round(_safe_float(record.get("distance_km"), 0.0), 2)
                    if record.get("distance_km") is not None
                    else None
                ),
                "distance_anchor_city": record.get("gateway_city"),
                "gateway_city": record.get("gateway_city"),
                "gateway_airport": _normalize_text(((record.get("gateway") or {}).get("airport") or {}).get("iata")),
                "uses_other_selected_city_airport": bool((record.get("gateway") or {}).get("uses_other_selected_city_airport")),
                "has_direct_airport": bool((record.get("gateway") or {}).get("has_direct_airport")),
            }
            for record in ordered_records
        ],
    })
    return ordered_items, meta


def _apply_route_ordering_to_city_plan(
    result: Dict[str, Any],
    city_plan: Dict[str, Any],
    user_request: Any,
) -> None:
    cities = city_plan.get("cities")
    if not isinstance(cities, list) or len(cities) < 2:
        return

    ordered_cities, route_meta = _route_order_city_items(
        city_items=cities,
        origin=result.get("origin"),
        city_plan_mode=city_plan.get("mode"),
        user_request=user_request,
    )
    city_plan["route_ordering"] = route_meta
    if not route_meta.get("applied"):
        return

    city_plan["cities"] = ordered_cities
    destination = result.get("destination")
    if isinstance(destination, list) and len(destination) > 1:
        destination_keys = {_normalize_route_city_key(city) for city in destination}
        ordered_keys = {_normalize_route_city_key(item.get("city")) for item in ordered_cities}
        if destination_keys == ordered_keys:
            result["destination"] = [_normalize_text(item.get("city")) for item in ordered_cities]


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
            explicit_units = {_normalize_text(item.get("unit")) for item in explicit_city_days}

            city_plan["mode"] = "user_explicit_city_days"
            city_plan["cities"] = explicit_city_days

            if explicit_units <= {"days"}:
                result["duration_days"] = explicit_total
            elif explicit_units <= {"nights"} and not result.get("duration_days"):
                result["duration_days"] = explicit_total + 1
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
                explicit_destinations = destination
                for city_name in explicit_destinations:
                    equal_ratio_items.append({
                        "city": _normalize_text(city_name),
                        "days_ratio": 1.0 / len(explicit_destinations),
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

    _apply_route_ordering_to_city_plan(result, city_plan, user_request=user_request)

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
