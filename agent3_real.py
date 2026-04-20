from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from travel_time import get_today_date

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SEARCHAPI_API_KEY = os.getenv("SEARCHAPI_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
TRAVEL_LLM_MODEL = os.getenv("TRAVEL_LLM_MODEL", "qwen-plus")
DASHSCOPE_CHAT_BASE_URL = os.getenv(
    "DASHSCOPE_CHAT_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
FLIGHT_LOOKAHEAD_DAYS = int(os.getenv("TRAVEL_FLIGHT_LOOKAHEAD_DAYS", "183"))

BASE_DIR = Path(__file__).resolve().parent
AIRPORT_MAPPING_PATH = BASE_DIR / "global_airports.json"
CITY_CENTROIDS_PATH = BASE_DIR / "cities15000.txt"

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("Please set GOOGLE_MAPS_API_KEY in your environment variables first.")

if not SEARCHAPI_API_KEY:
    raise ValueError("Please set SEARCHAPI_API_KEY in your environment variables first.")


class Agent3Selection:
    """
    Agent3:
    - Input: Agent2 output or merged pipeline output
    - Flights: SearchAPI Google Flights
    - Hotels: SearchAPI Google Hotels (preferred) + Google Places fallback
    - POIs / Restaurants: Google Places API (New)

    Key alignment in this version:
    1) Fully aligned with Agent1 controlled vocabulary:
       - hotel: budget / standard / luxury / central / clean
       - food: local / fine_dining / street_food / mixed
       - attraction_style: historical / nature / shopping / entertainment / art / mixed
       - planning_signals are consumed directly
    2) Fully aligned with Agent2 city_budget_plan / normalized_preferences / final_ratio.
    3) Hotel candidates are NOT summed together anymore.
       Each hotel candidate is checked independently against its city hotel budget.
    4) Restaurants default to days * 2.
    5) POIs default to days * 2, then adjusted by attraction intensity.
    6) Multi-city intercity flight legs are supported.
    """

    def __init__(self, agent2_output: Dict[str, Any]):
        self.raw_input = agent2_output or {}
        self.parsed_request = self.raw_input.get("parsed_request", {}) or {}
        self.budget_result = self.raw_input.get("budget_result", {}) or self.raw_input

        self.city_plan = self.parsed_request.get("city_plan", {}) or {}
        self.city_plan_mode = self._normalize_text(self.city_plan.get("mode")) or "unknown"

        self.num_people = self._extract_num_people()
        self.duration_days = self._extract_duration_days()
        self.total_budget = self._extract_total_budget()

        self.origin = self._normalize_text(
            self.parsed_request.get("origin") or self.raw_input.get("origin")
        )
        self.currency = (
            (self.parsed_request.get("budget") or {}).get("currency")
            or (self.raw_input.get("budget") or {}).get("currency")
            or "CNY"
        )

        self.raw_preferences = (self.parsed_request.get("preferences") or {})
        self.raw_planning_signals = (self.parsed_request.get("planning_signals") or {})

        # Agent1 controlled vocabulary (legacy scalar fields)
        self.flight_pref = self._normalize_text(self.raw_preferences.get("flight")).lower()
        self.hotel_pref = self._normalize_text(self.raw_preferences.get("hotel")).lower()
        self.food_pref = self._normalize_text(self.raw_preferences.get("food")).lower()
        self.attraction_style = (
            self._normalize_text(self.raw_preferences.get("attraction_style")).lower() or "mixed"
        )

        # Agent1 richer fields
        self.hotel_level_rich = self._normalize_text(
            self.raw_preferences.get("hotel_level")
        ).lower()
        self.hotel_features = self._extract_string_list(
            self.raw_preferences.get("hotel_features")
        )
        self.food_preferences = self._extract_string_list(
            self.raw_preferences.get("food_preferences")
        )
        self.attraction_styles = self._extract_string_list(
            self.raw_preferences.get("attraction_styles")
        )
        self.must_visit_attractions = self._extract_named_list(
            self.raw_preferences.get("must_visit_attractions")
        )

        if not self.food_preferences and self.food_pref:
            self.food_preferences = [self.food_pref]
        if not self.attraction_styles:
            self.attraction_styles = [self.attraction_style or "mixed"]
        if not self.hotel_level_rich:
            if self.hotel_pref == "budget":
                self.hotel_level_rich = "budget"
            elif self.hotel_pref == "standard":
                self.hotel_level_rich = "standard"
            elif self.hotel_pref == "luxury":
                self.hotel_level_rich = "luxury"
            elif self.hotel_pref in {"central", "clean"}:
                self.hotel_level_rich = "standard"
        if self.hotel_pref == "central" and "central" not in self.hotel_features:
            self.hotel_features.append("central")
        if self.hotel_pref == "clean" and "clean" not in self.hotel_features:
            self.hotel_features.append("clean")

        # Agent1 planning signals
        self.flight_priority = self._normalize_text(
            self.raw_planning_signals.get("flight_priority")
        ).lower()
        self.hotel_priority = self._normalize_text(
            self.raw_planning_signals.get("hotel_priority") or self.raw_planning_signals.get("hotel_level")
        ).lower()
        self.food_priority = self._normalize_text(
            self.raw_planning_signals.get("food_priority")
        ).lower()
        self.attraction_priority = self._normalize_text(
            self.raw_planning_signals.get("attraction_intensity")
        ).lower()

        # Agent2 normalized preferences
        self.normalized_preferences = (
            self.raw_input.get("normalized_preferences")
            or self.budget_result.get("normalized_preferences")
            or {}
        )
        self.mid_distance_car_rental = (
            self.raw_input.get("mid_distance_car_rental")
            or self.budget_result.get("mid_distance_car_rental")
            or {}
        )

        self.hotel_level = self._normalize_text(
            self.normalized_preferences.get("hotel_level")
        ).lower() or "standard"
        self.food_level = self._normalize_text(
            self.normalized_preferences.get("food_level")
        ).lower() or "standard"
        self.attraction_intensity = self._normalize_text(
            self.normalized_preferences.get("attraction_intensity")
        ).lower() or "medium"
        self.flight_type = self._normalize_text(
            self.normalized_preferences.get("flight_type")
        ).lower() or "standard"

        if not self.hotel_level_rich:
            self.hotel_level_rich = self.hotel_level
        if not self.food_preferences and self.food_level == "premium":
            self.food_preferences = ["fine_dining"]
        elif not self.food_preferences and self.food_level in {"standard", "budget"}:
            self.food_preferences = [self.food_pref or "mixed"] if (self.food_pref or "mixed") else ["mixed"]
        if not self.attraction_styles:
            self.attraction_styles = [self.attraction_style or "mixed"]

        self.final_ratio = self._extract_ratio("final_ratio")
        self.baseline_ratio = self._extract_ratio("baseline_ratio")
        self.llm_ratio = self._extract_ratio("llm_ratio")
        self.transport_budget = self._extract_transport_budget()

        self.budget_strategy = (
            self.raw_input.get("budget_strategy")
            or self.budget_result.get("budget_strategy")
            or "balanced"
        )
        self.cost_level = (
            self.raw_input.get("cost_level")
            or self.budget_result.get("cost_level")
            or "medium"
        )

        self.prefers_local_food = (self.food_pref == "local") or ("local" in self.food_preferences)
        self.local_food_llm_enabled = bool(DASHSCOPE_API_KEY)
        self.local_food_keyword_cache: Dict[str, Dict[str, Any]] = {}
        self.restaurant_tier_cache: Dict[str, Dict[str, Any]] = {}

        self.airports_by_iata = self._load_airport_mapping()
        self.city_airport_index = self._build_city_airport_index()
        self.city_centroid_index = self._load_city_centroids()

        self.effective_cities = self._extract_effective_cities()
        self.destination = [item["city"] for item in self.effective_cities]

        self.city_budget_plan, self.budget_source = self._extract_city_budget_plan()
        self.city_date_ranges = self._build_city_date_ranges()

        self.rooms_needed = max(1, math.ceil(self.num_people / 2))

        self.flight_selected_limit = 1
        self.flight_budget_tolerance = 1.15
        self.flight_relaxed_budget_tolerance = 1.25
        self.hotel_budget_tolerance = 1.20
        self.poi_budget_tolerance = 1.00
        self.restaurant_budget_tolerance = 1.00
        self.min_flight_distance_km = 180.0
        self.flight_buffer_hours = 2.0
        self.max_direct_drive_hours_for_auto_drive = 5.0
        self.nearby_airport_radius_km = 220.0
        self.max_nearby_airport_candidates = 5
        self.drive_metrics_cache: Dict[str, Dict[str, Any]] = {}

        self.prefer_direct_price_gap_ratio = 0.15
        self.prefer_direct_duration_ratio = 2.0
        self.hotel_clean_rating_threshold = 4.3
        self.hotel_central_keywords = [
            "central", "downtown", "city center", "city centre", "centre", "center",
            "old town", "inner city", "innere stadt", "historic center", "historical center",
            "historical centre", "near cathedral", "near station"
        ]
        self.strict_preference_filtering = True

        self.selection_policy = self._build_selection_policy()

    # =========================================================
    # basic helpers
    # =========================================================
    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(float(value))
        except Exception:
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _extract_string_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        results: List[str] = []
        seen = set()
        for item in value:
            text = self._normalize_text(item).lower()
            if text and text not in seen:
                results.append(text)
                seen.add(text)
        return results

    def _extract_named_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        results: List[str] = []
        seen = set()
        for item in value:
            text = self._normalize_text(item)
            if text and text.lower() not in seen:
                results.append(text)
                seen.add(text.lower())
        return results

    def _extract_num_people(self) -> int:
        value = self.raw_input.get("num_people") or self.parsed_request.get("num_people") or 1
        return max(1, self._safe_int(value, 1))

    def _extract_duration_days(self) -> int:
        value = self.raw_input.get("duration_days") or self.parsed_request.get("duration_days") or 1
        return max(1, self._safe_int(value, 1))

    def _extract_total_budget(self) -> float:
        budget_obj = self.raw_input.get("budget") or self.parsed_request.get("budget") or {}
        total = budget_obj.get("total", 0)
        return max(0.0, self._safe_float(total, 0.0))

    def _normalize_ratio_map(self, value: Any) -> Optional[Dict[str, float]]:
        if not isinstance(value, dict):
            return None

        keys = ["transport", "hotel", "food", "attractions", "buffer"]
        cleaned: Dict[str, float] = {}
        for key in keys:
            cleaned[key] = max(0.0, self._safe_float(value.get(key), 0.0))

        total = sum(cleaned.values())
        if total <= 0:
            return None

        return {k: round(v / total, 4) for k, v in cleaned.items()}

    def _extract_ratio(self, ratio_name: str) -> Optional[Dict[str, float]]:
        ratio = (
            (self.raw_input.get("allocation_basis") or {}).get(ratio_name)
            or (self.budget_result.get("allocation_basis") or {}).get(ratio_name)
        )
        return self._normalize_ratio_map(ratio)

    def _extract_transport_budget(self) -> float:
        explicit_transport = (
            (self.raw_input.get("recommended_spend") or {}).get("transport")
            or (self.budget_result.get("recommended_spend") or {}).get("transport")
            or (self.raw_input.get("budget_plan") or {}).get("transport")
            or (self.budget_result.get("budget_plan") or {}).get("transport")
        )

        if explicit_transport is not None:
            return max(0.0, self._safe_float(explicit_transport, 0.0))

        if self.total_budget > 0 and self.final_ratio:
            return round(self.total_budget * self.final_ratio.get("transport", 0.0), 2)

        return 0.0

    def _get_single_selected_total(self, items: List[Dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return round(self._safe_float(items[0].get(key), 0.0), 2)

    def _get_sum_selected_total(self, items: List[Dict[str, Any]], key: str) -> float:
        return round(sum(self._safe_float(x.get(key), 0.0) for x in items), 2)

    def _budget_fit_label(self, selected_total: float, budget: float, tolerance: float = 1.0) -> str:
        if budget <= 0:
            return "no_budget_defined"
        if selected_total <= 0:
            return "not_selected"
        if selected_total <= budget:
            return "within_budget"
        if selected_total <= budget * tolerance:
            return "slightly_over_budget"
        return "over_budget"

    def _normalize_city_key(self, city: str) -> str:
        text = self._normalize_text(city).lower().strip()
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b(city|airport|international airport|intl airport)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_alias_key(self, text: str) -> str:
        text = self._normalize_text(text).lower().strip()
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _geonames_feature_rank(self, feature_code: str) -> int:
        """Prefer capital / admin centers / populated places."""
        rank_map = {
            "PPLC": 6,
            "PPLA": 5,
            "PPLA2": 4,
            "PPLA3": 3,
            "PPLA4": 2,
            "PPL": 1,
            "PPLX": 0,
        }
        return rank_map.get(self._normalize_text(feature_code).upper(), -1)

    def _load_city_centroids(self) -> Dict[str, Dict[str, Any]]:
        """
        Load GeoNames cities15000.txt into an alias-based centroid index.
        """
        index: Dict[str, Dict[str, Any]] = {}

        if not CITY_CENTROIDS_PATH.exists():
            return index

        try:
            with open(CITY_CENTROIDS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) < 9:
                        continue

                    geonameid = parts[0]
                    name = parts[1]
                    asciiname = parts[2]
                    alternatenames = parts[3]
                    latitude = parts[4]
                    longitude = parts[5]
                    feature_class = parts[6]
                    feature_code = parts[7]
                    country_code = parts[8]

                    if self._normalize_text(feature_class).upper() != "P":
                        continue

                    try:
                        lat = float(latitude)
                        lng = float(longitude)
                    except Exception:
                        continue

                    population = self._safe_int(parts[14], 0) if len(parts) > 14 else 0

                    record = {
                        "geonameid": geonameid,
                        "name": self._normalize_text(name),
                        "asciiname": self._normalize_text(asciiname),
                        "lat": lat,
                        "lng": lng,
                        "feature_code": self._normalize_text(feature_code).upper(),
                        "country_code": self._normalize_text(country_code).upper(),
                        "population": population,
                    }

                    aliases = set()
                    if record["name"]:
                        aliases.add(self._normalize_alias_key(record["name"]))
                    if record["asciiname"]:
                        aliases.add(self._normalize_alias_key(record["asciiname"]))

                    for alias in (alternatenames or "").split(","):
                        alias = self._normalize_text(alias)
                        if alias:
                            aliases.add(self._normalize_alias_key(alias))

                    for alias_key in aliases:
                        if not alias_key:
                            continue

                        existing = index.get(alias_key)
                        if existing is None:
                            index[alias_key] = record
                            continue

                        old_rank = self._geonames_feature_rank(existing.get("feature_code", ""))
                        new_rank = self._geonames_feature_rank(record.get("feature_code", ""))

                        if (new_rank > old_rank) or (
                            new_rank == old_rank and record["population"] > existing.get("population", 0)
                        ):
                            index[alias_key] = record
        except Exception:
            return {}

        return index

    def _get_city_centroid(self, city: str) -> Optional[Dict[str, Any]]:
        city_key = self._normalize_alias_key(city)
        if not city_key:
            return None
        return self.city_centroid_index.get(city_key)

    def _get_city_location_bias(self, city: str) -> Optional[Dict[str, float]]:
        centroid = self._get_city_centroid(city)
        if not centroid:
            return None

        return {
            "lat": centroid["lat"],
            "lng": centroid["lng"],
            "radius_m": 30000
        }

    def _city_coord(self, city: str) -> Optional[Dict[str, float]]:
        centroid = self._get_city_centroid(city)
        if not centroid:
            return None
        return {
            "lat": self._safe_float(centroid.get("lat"), 0.0),
            "lng": self._safe_float(centroid.get("lng"), 0.0),
        }

    def _airport_coord(self, iata: Optional[str]) -> Optional[Dict[str, float]]:
        airport = self.get_airport_info(iata)
        if not airport:
            return None
        lat = self._safe_float(airport.get("lat"), 0.0)
        lng = self._safe_float(airport.get("lng"), 0.0)
        if lat == 0.0 and lng == 0.0:
            return None
        return {"lat": lat, "lng": lng}

    def _extract_place_lat_lng(self, place: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        location = place.get("location") or {}
        lat = location.get("latitude")
        lng = location.get("longitude")

        try:
            if lat is not None and lng is not None:
                return float(lat), float(lng)
        except Exception:
            pass

        return None, None

    def _haversine_distance_km(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
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

    def _coord_key(self, coord: Dict[str, float]) -> str:
        return f"{self._safe_float(coord.get('lat'), 0.0):.5f},{self._safe_float(coord.get('lng'), 0.0):.5f}"

    def _estimate_drive_metrics_between_coords(
        self,
        origin: Optional[Dict[str, float]],
        destination: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        if not origin or not destination:
            return None

        o_lat = self._safe_float(origin.get("lat"), 0.0)
        o_lng = self._safe_float(origin.get("lng"), 0.0)
        d_lat = self._safe_float(destination.get("lat"), 0.0)
        d_lng = self._safe_float(destination.get("lng"), 0.0)
        if (o_lat == 0.0 and o_lng == 0.0) or (d_lat == 0.0 and d_lng == 0.0):
            return None

        cache_key = f"{self._coord_key(origin)}->{self._coord_key(destination)}"
        cached = self.drive_metrics_cache.get(cache_key)
        if cached is not None:
            return cached or None

        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": f"{o_lat},{o_lng}",
            "destinations": f"{d_lat},{d_lng}",
            "mode": "driving",
            "key": GOOGLE_MAPS_API_KEY,
        }
        try:
            response = requests.get(url, params=params, timeout=12)
            if response.status_code == 200:
                data = response.json() if response.text else {}
                rows = data.get("rows") if isinstance(data, dict) else []
                elements = rows[0].get("elements", []) if rows and isinstance(rows[0], dict) else []
                element = elements[0] if elements and isinstance(elements[0], dict) else {}
                if element.get("status") == "OK":
                    distance_m = self._safe_float((element.get("distance") or {}).get("value"), 0.0)
                    duration_s = self._safe_float((element.get("duration") or {}).get("value"), 0.0)
                    if distance_m > 0 and duration_s > 0:
                        result = {
                            "distance_km": round(distance_m / 1000.0, 2),
                            "duration_h": round(duration_s / 3600.0, 2),
                            "source": "google_distance_matrix",
                        }
                        self.drive_metrics_cache[cache_key] = result
                        return result
        except Exception:
            pass

        distance_km = self._haversine_distance_km(o_lat, o_lng, d_lat, d_lng)
        result = {
            "distance_km": round(distance_km, 2),
            "duration_h": round(max(0.1, distance_km / 60.0), 2),
            "source": "haversine_60kmh_fallback",
        }
        self.drive_metrics_cache[cache_key] = result
        return result

    def _estimate_city_drive_metrics(self, from_city: str, to_city: str) -> Optional[Dict[str, Any]]:
        return self._estimate_drive_metrics_between_coords(
            self._city_coord(from_city),
            self._city_coord(to_city),
        )

    def _estimate_city_airport_drive_metrics(self, city: str, iata: Optional[str]) -> Optional[Dict[str, Any]]:
        return self._estimate_drive_metrics_between_coords(
            self._city_coord(city),
            self._airport_coord(iata),
        )

    def _address_geo_signal(self, address: str, city: str) -> str:
        """Return strong_match / weak_match / mismatch / unknown."""
        address_text = self._normalize_text(address).lower()
        if not address_text:
            return "unknown"

        centroid = self._get_city_centroid(city)
        city_key = self._normalize_alias_key(city)

        if not centroid:
            if city_key and city_key in address_text:
                return "weak_match"
            return "unknown"

        country_code = self._normalize_text(centroid.get("country_code")).lower()
        city_name = self._normalize_text(centroid.get("name")).lower()
        city_ascii = self._normalize_text(centroid.get("asciiname")).lower()

        has_city_name = (
            (city_key and city_key in address_text)
            or (city_name and city_name in address_text)
            or (city_ascii and city_ascii in address_text)
        )

        country_name_map = {
            "jp": ["japan"],
            "sg": ["singapore"],
            "kr": ["korea", "south korea"],
            "th": ["thailand"],
            "hk": ["hong kong"],
            "cn": ["china"],
            "us": ["usa", "united states"],
            "gb": ["uk", "united kingdom", "england"],
            "fr": ["france"],
            "it": ["italy"],
        }
        country_keywords = country_name_map.get(country_code, [])
        has_country_signal = any(keyword in address_text for keyword in country_keywords)

        strong_negative_markers = [
            "hong kong", "kowloon",
            "singapore",
            "bangkok", "thailand",
            "seoul", "korea",
            "taipei", "taiwan",
            "shanghai", "china",
            "beijing", "china",
        ]

        allowed_tokens = set([city_key, city_name, city_ascii] + country_keywords)
        for marker in strong_negative_markers:
            if marker in address_text and marker not in allowed_tokens:
                return "mismatch"

        if has_city_name and has_country_signal:
            return "strong_match"
        if has_city_name or has_country_signal:
            return "weak_match"

        return "unknown"

    def _place_is_near_target_city(
        self,
        place: Dict[str, Any],
        city: str,
        strict_km: float = 40.0,
        loose_km: float = 80.0
    ) -> str:
        """Return near / far / unknown."""
        centroid = self._get_city_centroid(city)
        if not centroid:
            return "unknown"

        lat, lng = self._extract_place_lat_lng(place)
        if lat is None or lng is None:
            return "unknown"

        distance = self._haversine_distance_km(lat, lng, centroid["lat"], centroid["lng"])
        if distance <= strict_km:
            return "near"
        if distance > loose_km:
            return "far"
        return "unknown"

    def _restaurant_matches_target_city(self, place: Dict[str, Any], city: str) -> bool:
        """Priority: coordinate proximity, then address city/country signal."""
        address = self._normalize_text(place.get("formattedAddress"))
        distance_signal = self._place_is_near_target_city(place, city)
        address_signal = self._address_geo_signal(address, city)

        if distance_signal == "near":
            return True
        if distance_signal == "far":
            return False

        if address_signal == "strong_match":
            return True
        if address_signal == "mismatch":
            return False
        if address_signal == "weak_match":
            return True

        return False

    # =========================================================
    # policy
    # =========================================================
    def _build_selection_policy(self) -> Dict[str, Any]:
        return {
            "flight": {
                "preference": self.flight_pref,
                "priority": self.flight_priority,
                "type": self.flight_type,
                "budget_tolerance": self.flight_budget_tolerance,
                "relaxed_budget_tolerance": self.flight_relaxed_budget_tolerance,
            },
            "hotel": {
                "preference": self.hotel_pref,
                "priority": self.hotel_priority,
                "level": self.hotel_level,
                "hotel_level_rich": self.hotel_level_rich,
                "features": list(self.hotel_features),
                "minimum_candidates_per_city": 2,
                "budget_tolerance": self.hotel_budget_tolerance,
            },
            "restaurant": {
                "preference": self.food_pref,
                "preferences": list(self.food_preferences),
                "priority": self.food_priority,
                "level": self.food_level,
                "target_per_day": 2,
                "budget_tolerance": self.restaurant_budget_tolerance,
            },
            "poi": {
                "style": self.attraction_style,
                "styles": list(self.attraction_styles),
                "priority": self.attraction_priority or self.attraction_intensity,
                "must_visit_attractions": list(self.must_visit_attractions),
                "target_base_per_day": 2,
                "budget_tolerance": self.poi_budget_tolerance,
            },
        }

    def _candidate_limits_by_days(self, city_days: int) -> Dict[str, int]:
        city_days = max(1, city_days)
        return {
            "hotel_candidates": max(6, city_days * 3),
            "restaurant_candidates": max(8, city_days * 4),
            "poi_candidates": max(8, city_days * 4),
            "hotel_selected": 2,
            "restaurant_selected": self._restaurant_target_count(city_days),
            "poi_selected": self._poi_limit_by_intensity(city_days),
        }

    # =========================================================
    # airport mapping
    # =========================================================
    def _load_airport_mapping(self) -> Dict[str, Any]:
        if not AIRPORT_MAPPING_PATH.exists():
            return {}

        with open(AIRPORT_MAPPING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        cleaned: Dict[str, Dict[str, Any]] = {}
        for iata_key, airport in data.items():
            if not isinstance(airport, dict):
                continue

            iata = self._normalize_text(airport.get("iata") or iata_key).upper()
            if not iata:
                continue

            cleaned[iata] = {
                "iata": iata,
                "icao": self._normalize_text(airport.get("icao")),
                "name": self._normalize_text(airport.get("name")),
                "city": self._normalize_text(airport.get("city")),
                "country": self._normalize_text(airport.get("country")),
                "lat": self._safe_float(airport.get("lat"), 0.0),
                "lng": self._safe_float(airport.get("lng"), 0.0),
                "type": self._normalize_text(airport.get("type")).lower(),
            }
        return cleaned

    def _airport_type_rank(self, airport_type: str) -> int:
        rank_map = {
            "large_airport": 5,
            "medium_airport": 4,
            "small_airport": 3,
            "seaplane_base": 2,
            "heliport": 1,
        }
        return rank_map.get(self._normalize_text(airport_type).lower(), 0)

    def _build_city_airport_index(self) -> Dict[str, List[Dict[str, Any]]]:
        city_index: Dict[str, List[Dict[str, Any]]] = {}

        for airport in self.airports_by_iata.values():
            city = self._normalize_text(airport.get("city"))
            if not city:
                continue

            city_key = self._normalize_city_key(city)
            if not city_key:
                continue

            city_index.setdefault(city_key, []).append(airport)

        for city_key, airports in city_index.items():
            airports.sort(
                key=lambda x: (
                    self._airport_type_rank(x.get("type", "")),
                    len(self._normalize_text(x.get("name"))),
                    self._normalize_text(x.get("iata")),
                ),
                reverse=True,
            )
        return city_index

    def _get_city_airport_candidates(self, city: str) -> List[Dict[str, Any]]:
        city_key = self._normalize_city_key(city)
        direct = self.city_airport_index.get(city_key, [])
        if direct:
            preferred = [
                a for a in direct
                if a.get("type") in {"large_airport", "medium_airport", "small_airport"}
            ]
            return preferred or direct

        matched: List[Dict[str, Any]] = []
        for indexed_city_key, airports in self.city_airport_index.items():
            if city_key and (city_key in indexed_city_key or indexed_city_key in city_key):
                matched.extend(airports)

        dedup: Dict[str, Dict[str, Any]] = {}
        for airport in matched:
            iata = self._normalize_text(airport.get("iata")).upper()
            if iata and iata not in dedup:
                dedup[iata] = airport

        result = list(dedup.values())
        result.sort(
            key=lambda x: (
                self._airport_type_rank(x.get("type", "")),
                len(self._normalize_text(x.get("name"))),
                self._normalize_text(x.get("iata")),
            ),
            reverse=True,
        )
        return result

    def city_to_primary_airport(self, city: str) -> Optional[str]:
        candidates = self._get_city_airport_candidates(city)
        if not candidates:
            return None
        return self._normalize_text(candidates[0].get("iata")).upper() or None

    def city_to_candidate_airports(self, city: str) -> List[str]:
        candidates = self._get_city_airport_candidates(city)
        return [
            self._normalize_text(item.get("iata")).upper()
            for item in candidates
            if self._normalize_text(item.get("iata"))
        ][:8]

    def city_to_nearby_airport_candidates(self, city: str) -> List[Dict[str, Any]]:
        city_coord = self._city_coord(city)
        if not city_coord:
            return []

        ranked: List[Dict[str, Any]] = []
        for airport in self.airports_by_iata.values():
            iata = self._normalize_text(airport.get("iata")).upper()
            if not iata:
                continue
            airport_type = self._normalize_text(airport.get("type")).lower()
            if airport_type not in {"large_airport", "medium_airport", "small_airport"}:
                continue
            airport_coord = {
                "lat": self._safe_float(airport.get("lat"), 0.0),
                "lng": self._safe_float(airport.get("lng"), 0.0),
            }
            if airport_coord["lat"] == 0.0 and airport_coord["lng"] == 0.0:
                continue
            distance_km = self._haversine_distance_km(
                city_coord["lat"],
                city_coord["lng"],
                airport_coord["lat"],
                airport_coord["lng"],
            )
            if distance_km > self.nearby_airport_radius_km:
                continue
            ranked.append({
                "iata": iata,
                "distance_km": round(distance_km, 2),
                "airport": airport,
            })

        ranked.sort(
            key=lambda item: (
                item.get("distance_km", 99999.0),
                -self._airport_type_rank((item.get("airport") or {}).get("type", "")),
                self._normalize_text(item.get("iata")),
            )
        )
        return ranked[:self.max_nearby_airport_candidates]

    def city_to_search_airport_candidates(self, city: str) -> Tuple[List[str], str]:
        direct = self.city_to_candidate_airports(city)
        if direct:
            return direct, "city_airport"
        nearby = self.city_to_nearby_airport_candidates(city)
        return [item["iata"] for item in nearby], "nearby_airport" if nearby else "no_airport"

    def get_airport_info(self, iata: Optional[str]) -> Optional[Dict[str, Any]]:
        if not iata:
            return None
        return self.airports_by_iata.get(self._normalize_text(iata).upper())

    def _to_radians(self, value: float) -> float:
        return value * math.pi / 180.0

    def calculate_airport_distance_km(self, from_iata: Optional[str], to_iata: Optional[str]) -> Optional[float]:
        from_airport = self.get_airport_info(from_iata)
        to_airport = self.get_airport_info(to_iata)

        if not from_airport or not to_airport:
            return None

        lat1 = self._safe_float(from_airport.get("lat"), 0.0)
        lng1 = self._safe_float(from_airport.get("lng"), 0.0)
        lat2 = self._safe_float(to_airport.get("lat"), 0.0)
        lng2 = self._safe_float(to_airport.get("lng"), 0.0)

        if lat1 == 0.0 and lng1 == 0.0:
            return None
        if lat2 == 0.0 and lng2 == 0.0:
            return None

        r = 6371.0
        dlat = self._to_radians(lat2 - lat1)
        dlng = self._to_radians(lng2 - lng1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(self._to_radians(lat1))
            * math.cos(self._to_radians(lat2))
            * math.sin(dlng / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(r * c, 2)

    def should_skip_flight_search(self, route: Dict[str, Any]) -> Tuple[bool, Optional[float], str]:
        if not route:
            return False, None, "empty_route"

        from_airport = route.get("from_airport")
        to_airport = route.get("to_airport")

        if not from_airport or not to_airport:
            return False, None, "missing_airport"

        if from_airport == to_airport:
            return True, 0.0, "same_airport"

        distance_km = self.calculate_airport_distance_km(from_airport, to_airport)
        if distance_km is None:
            return False, None, "distance_unavailable"

        if distance_km < self.min_flight_distance_km:
            return True, distance_km, "below_distance_threshold"

        return False, distance_km, "distance_ok"

    # =========================================================
    # Agent1 / Agent2 alignment
    # =========================================================
    def _extract_effective_cities(self) -> List[Dict[str, Any]]:
        city_budget_plan = (
            self.raw_input.get("city_budget_plan")
            or self.budget_result.get("city_budget_plan")
            or []
        )

        normalized: List[Dict[str, Any]] = []

        if isinstance(city_budget_plan, list) and city_budget_plan:
            for item in city_budget_plan:
                if not isinstance(item, dict):
                    continue
                city_name = self._normalize_text(item.get("city"))
                if city_name:
                    normalized.append({
                        "city": city_name,
                        "days": max(1, self._safe_int(item.get("days"), 1)),
                        "source": "agent2_city_budget_plan"
                    })
            if normalized:
                return normalized

        city_plan = self.parsed_request.get("city_plan", {}) or {}
        planned_cities = city_plan.get("cities", [])

        if isinstance(planned_cities, list) and planned_cities:
            for item in planned_cities:
                if isinstance(item, dict):
                    city_name = self._normalize_text(item.get("city"))
                    if city_name:
                        normalized.append({
                            "city": city_name,
                            "days": max(1, self._safe_int(item.get("days"), 1)),
                            "source": self._normalize_text(item.get("source")) or "system"
                        })
                else:
                    city_name = self._normalize_text(item)
                    if city_name:
                        normalized.append({
                            "city": city_name,
                            "days": 1,
                            "source": "destination"
                        })
            if normalized:
                return normalized

        destination = self.parsed_request.get("destination") or self.raw_input.get("destination") or []
        if isinstance(destination, list) and destination:
            total_days = self.duration_days
            if len(destination) == 1:
                return [{
                    "city": self._normalize_text(destination[0]),
                    "days": total_days,
                    "source": "destination"
                }]

            avg_days = max(1, total_days // len(destination))
            result = []
            used = 0
            for i, city in enumerate(destination):
                if i == len(destination) - 1:
                    days = max(1, total_days - used)
                else:
                    days = avg_days
                    used += days

                result.append({
                    "city": self._normalize_text(city),
                    "days": days,
                    "source": "destination"
                })
            return result

        return []

    def _extract_city_budget_plan(self) -> Tuple[List[Dict[str, Any]], str]:
        raw_city_budget_plan = (
            self.raw_input.get("city_budget_plan")
            or self.budget_result.get("city_budget_plan")
            or []
        )

        if isinstance(raw_city_budget_plan, list) and raw_city_budget_plan:
            cleaned = []
            for item in raw_city_budget_plan:
                if not isinstance(item, dict):
                    continue

                city_name = self._normalize_text(item.get("city"))
                if not city_name:
                    continue

                cleaned.append({
                    "city": city_name,
                    "days": max(1, self._safe_int(item.get("days"), 1)),
                    "hotel_budget": max(0.0, self._safe_float(item.get("hotel_budget"), 0.0)),
                    "food_budget": max(0.0, self._safe_float(item.get("food_budget"), 0.0)),
                    "attractions_budget": max(0.0, self._safe_float(item.get("attractions_budget"), 0.0)),
                    "profile_tier": self._normalize_text(item.get("profile_tier")) or None,
                    "matched_profile": self._normalize_text(item.get("matched_profile")) or None,
                    "cost_weights": item.get("cost_weights", {}) if isinstance(item.get("cost_weights"), dict) else {}
                })
            if cleaned:
                return cleaned, "city_budget_plan"

        if self.effective_cities and self.total_budget > 0 and self.final_ratio:
            total_days = sum(item["days"] for item in self.effective_cities) or 1
            hotel_total = self.total_budget * self.final_ratio.get("hotel", 0.0)
            food_total = self.total_budget * self.final_ratio.get("food", 0.0)
            attractions_total = self.total_budget * self.final_ratio.get("attractions", 0.0)

            derived = []
            for item in self.effective_cities:
                weight = item["days"] / total_days
                derived.append({
                    "city": item["city"],
                    "days": item["days"],
                    "hotel_budget": round(hotel_total * weight, 2),
                    "food_budget": round(food_total * weight, 2),
                    "attractions_budget": round(attractions_total * weight, 2),
                    "profile_tier": None,
                    "matched_profile": None,
                    "cost_weights": {}
                })
            return derived, "final_ratio_days_fallback"

        return [], "none"

    # =========================================================
    # date / route helpers
    # =========================================================
    def _resolve_departure_date(self) -> Optional[str]:
        travel_dates = self.parsed_request.get("travel_dates", {}) or {}
        start = self._normalize_text(travel_dates.get("start"))
        if start and re.fullmatch(r"\d{4}-\d{2}-\d{2}", start):
            return start

        text = self._normalize_text(travel_dates.get("text"))
        if not text:
            return None

        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if match:
            return match.group(0)

        return None

    def _resolve_return_date(self) -> Optional[str]:
        travel_dates = self.parsed_request.get("travel_dates", {}) or {}
        end = self._normalize_text(travel_dates.get("end"))
        if end and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end):
            return end
        return None

    def _parse_iso_date(self, value: Optional[str]) -> Optional[datetime]:
        value = self._normalize_text(value)
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except Exception:
            return None

    def _allocate_city_nights(self) -> List[Dict[str, Any]]:
        if not self.effective_cities:
            return []

        trip_start = self._parse_iso_date(self._resolve_departure_date())
        trip_end = self._parse_iso_date(self._resolve_return_date())

        cities: List[Dict[str, Any]] = []
        for idx, item in enumerate(self.effective_cities):
            city = self._normalize_text(item.get("city"))
            days = max(1, self._safe_int(item.get("days"), 1))
            if city:
                cities.append({
                    "city": city,
                    "days": days,
                    "original_index": idx
                })

        if not cities:
            return []

        total_days = sum(x["days"] for x in cities)
        if trip_start and trip_end:
            total_nights = (trip_end - trip_start).days
        else:
            total_nights = total_days - 1

        total_nights = max(len(cities), total_nights)

        if len(cities) == 1:
            return [{
                "city": cities[0]["city"],
                "days": cities[0]["days"],
                "nights": total_nights
            }]

        allocations: List[Dict[str, Any]] = []
        base_sum = 0

        for item in cities:
            raw_nights = total_nights * item["days"] / total_days
            base_nights = max(1, int(math.floor(raw_nights)))
            remainder = raw_nights - math.floor(raw_nights)

            allocations.append({
                "city": item["city"],
                "days": item["days"],
                "original_index": item["original_index"],
                "raw_nights": raw_nights,
                "nights": base_nights,
                "remainder": remainder
            })
            base_sum += base_nights

        remaining = total_nights - base_sum
        if remaining > 0:
            ranked = sorted(
                allocations,
                key=lambda x: (x["remainder"], x["days"], -x["original_index"]),
                reverse=True
            )
            i = 0
            while remaining > 0:
                ranked[i % len(ranked)]["nights"] += 1
                remaining -= 1
                i += 1
        elif remaining < 0:
            ranked = sorted(
                [x for x in allocations if x["nights"] > 1],
                key=lambda x: (x["remainder"], x["days"], x["original_index"])
            )
            i = 0
            while remaining < 0 and ranked:
                target = ranked[i % len(ranked)]
                if target["nights"] > 1:
                    target["nights"] -= 1
                    remaining += 1
                i += 1
                ranked = sorted(
                    [x for x in allocations if x["nights"] > 1],
                    key=lambda x: (x["remainder"], x["days"], x["original_index"])
                )

        allocations.sort(key=lambda x: x["original_index"])
        final_sum = sum(x["nights"] for x in allocations)
        diff = total_nights - final_sum
        if diff != 0:
            allocations[-1]["nights"] += diff

        return [{
            "city": x["city"],
            "days": x["days"],
            "nights": max(1, x["nights"])
        } for x in allocations]

    def _build_city_date_ranges(self) -> Dict[str, Dict[str, Optional[str]]]:
        result: Dict[str, Dict[str, Optional[str]]] = {}
        trip_start = self._parse_iso_date(self._resolve_departure_date())
        trip_end = self._parse_iso_date(self._resolve_return_date())

        allocated_city_nights = self._allocate_city_nights()
        if not allocated_city_nights:
            return result

        if not trip_start:
            for item in allocated_city_nights:
                result[item["city"]] = {
                    "check_in": None,
                    "check_out": None,
                    "days": item["days"],
                    "nights": item["nights"]
                }
            return result

        current_check_in = trip_start
        for idx, item in enumerate(allocated_city_nights):
            city = item["city"]
            days = max(1, self._safe_int(item.get("days"), 1))
            nights = max(1, self._safe_int(item.get("nights"), 1))

            if idx == len(allocated_city_nights) - 1 and trip_end:
                check_out = trip_end
            else:
                check_out = current_check_in + timedelta(days=nights)

            result[city] = {
                "check_in": current_check_in.strftime("%Y-%m-%d"),
                "check_out": check_out.strftime("%Y-%m-%d"),
                "days": days,
                "nights": nights
            }
            current_check_in = check_out

        return result

    def _build_intercity_departure_date(self, from_city: str) -> Optional[str]:
        date_info = self.city_date_ranges.get(from_city, {})
        return date_info.get("check_out")

    def _build_route(self, from_city: str, to_city: str, departure_date: Optional[str], leg_type: str, leg_index: int) -> Dict[str, Any]:
        from_candidates, from_candidate_source = self.city_to_search_airport_candidates(from_city)
        to_candidates, to_candidate_source = self.city_to_search_airport_candidates(to_city)
        direct_drive_metrics = self._estimate_city_drive_metrics(from_city, to_city)
        direct_drive_hours = (
            self._safe_float(direct_drive_metrics.get("duration_h"), 0.0)
            if isinstance(direct_drive_metrics, dict)
            else 0.0
        )
        return {
            "leg_type": leg_type,
            "leg_index": leg_index,
            "from_city": from_city,
            "to_city": to_city,
            "from_airport": from_candidates[0] if from_candidates else None,
            "to_airport": to_candidates[0] if to_candidates else None,
            "candidate_from_airports": from_candidates,
            "candidate_to_airports": to_candidates,
            "from_airport_candidate_source": from_candidate_source,
            "to_airport_candidate_source": to_candidate_source,
            "departure_date": departure_date,
            "direct_drive_hours": round(direct_drive_hours, 2) if direct_drive_hours > 0 else None,
            "direct_drive_distance_km": (
                round(self._safe_float(direct_drive_metrics.get("distance_km"), 0.0), 2)
                if isinstance(direct_drive_metrics, dict)
                else None
            ),
            "direct_drive_source": (
                self._normalize_text(direct_drive_metrics.get("source"))
                if isinstance(direct_drive_metrics, dict)
                else None
            ),
        }

    def _build_route_plan(self) -> Dict[str, Any]:
        if not self.origin or not self.effective_cities:
            return {"outbound": None, "inbound": None, "intercity": [], "flight_legs": []}

        first_city = self.effective_cities[0]["city"]
        last_city = self.effective_cities[-1]["city"]

        outbound_route = self._build_route(
            from_city=self.origin,
            to_city=first_city,
            departure_date=self._resolve_departure_date(),
            leg_type="outbound",
            leg_index=0
        )

        inbound_route = self._build_route(
            from_city=last_city,
            to_city=self.origin,
            departure_date=self._resolve_return_date(),
            leg_type="inbound",
            leg_index=max(1, len(self.effective_cities))
        )

        intercity_routes: List[Dict[str, Any]] = []
        for idx in range(len(self.effective_cities) - 1):
            from_city = self.effective_cities[idx]["city"]
            to_city = self.effective_cities[idx + 1]["city"]
            departure_date = self._build_intercity_departure_date(from_city)

            intercity_routes.append(
                self._build_route(
                    from_city=from_city,
                    to_city=to_city,
                    departure_date=departure_date,
                    leg_type="intercity",
                    leg_index=idx + 1
                )
            )

        flight_legs: List[Dict[str, Any]] = [outbound_route]
        if not self._has_mid_distance_car_rental_plan():
            flight_legs.extend(intercity_routes)
        if inbound_route.get("departure_date"):
            flight_legs.append(inbound_route)

        return {
            "outbound": outbound_route,
            "inbound": inbound_route,
            "intercity": intercity_routes,
            "flight_legs": flight_legs
        }

    def _has_mid_distance_car_rental_plan(self) -> bool:
        if not isinstance(self.mid_distance_car_rental, dict):
            return False
        estimates = self.mid_distance_car_rental.get("city_estimates")
        return isinstance(estimates, list) and bool(estimates)

    def _mid_distance_car_estimate_by_city(self) -> Dict[str, Dict[str, Any]]:
        if not self._has_mid_distance_car_rental_plan():
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for item in self.mid_distance_car_rental.get("city_estimates", []):
            if not isinstance(item, dict):
                continue
            city = self._normalize_text(item.get("city"))
            if city:
                result[self._normalize_city_key(city)] = item
        return result

    def _build_mid_distance_rental_legs(self, intercity_routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not intercity_routes or not self._has_mid_distance_car_rental_plan():
            return []

        estimates_by_city = self._mid_distance_car_estimate_by_city()
        rental_legs: List[Dict[str, Any]] = []

        for idx, route in enumerate(intercity_routes):
            allocated_city_estimates: List[Dict[str, Any]] = []
            if idx == 0:
                from_estimate = estimates_by_city.get(
                    self._normalize_city_key(route.get("from_city"))
                )
                if from_estimate:
                    allocated_city_estimates.append(from_estimate)

            to_estimate = estimates_by_city.get(
                self._normalize_city_key(route.get("to_city"))
            )
            if to_estimate:
                allocated_city_estimates.append(to_estimate)

            rental_cost = round(
                sum(
                    self._safe_float(item.get("total_car_rental_cost"), 0.0)
                    for item in allocated_city_estimates
                ),
                2,
            )

            rental_legs.append({
                "leg_type": "intercity",
                "leg_index": route.get("leg_index"),
                "from_city": route.get("from_city"),
                "to_city": route.get("to_city"),
                "departure_date": route.get("departure_date"),
                "budget": None,
                "selection_status": "rental_car_selected|middle_distance_skip_flight_search",
                "transport_mode": "rental_car",
                "selected_flights": [],
                "rental_car_cost_estimated": rental_cost,
                "rental_car": {
                    "scenario": self.mid_distance_car_rental.get("scenario"),
                    "cost_rule": self.mid_distance_car_rental.get("cost_rule"),
                    "allocated_city_estimates": allocated_city_estimates,
                    "allocation_note": (
                        "First rental leg carries the first city plus destination city rental-days; "
                        "later legs carry the next destination city rental-days, so all city day totals "
                        "sum to mid_distance_car_rental.total_car_rental_cost."
                    ),
                },
            })

        return rental_legs

    # =========================================================
    # local food helpers
    # =========================================================
    def _fallback_local_food_map(self) -> Dict[str, List[str]]:
        return {
            "tokyo": ["sushi", "ramen", "izakaya", "yakitori", "tempura"],
            "osaka": ["takoyaki", "okonomiyaki", "kushikatsu", "izakaya", "ramen"],
            "kyoto": ["kaiseki", "yudofu", "matcha sweets", "tofu cuisine", "tempura"],
            "guangzhou": ["cantonese cuisine", "dim sum", "yum cha", "roast meat", "seafood"],
            "shenzhen": ["cantonese cuisine", "seafood", "teochew cuisine", "roast meat", "hotpot"],
            "zhuhai": ["cantonese cuisine", "seafood", "oyster", "hotpot", "dim sum"],
            "beijing": ["peking duck", "beijing cuisine", "zhajiangmian", "hotpot", "dumplings"],
            "shanghai": ["xiaolongbao", "shengjianbao", "shanghainese cuisine", "noodles", "crab roe noodles"],
            "chengdu": ["sichuan cuisine", "hotpot", "skewers", "dan dan noodles", "mapo tofu"],
            "seoul": ["korean bbq", "bibimbap", "kimchi", "gukbap", "tteokbokki"],
            "bangkok": ["pad thai", "tom yum", "thai curry", "street food", "boat noodles"],
            "singapore": ["hawker food", "laksa", "hainanese chicken rice", "bak kut teh", "satay"],
        }

    def _city_local_food_keywords_fallback(self, city: str) -> Dict[str, Any]:
        city_key = self._normalize_city_key(city)
        keywords = self._fallback_local_food_map().get(
            city_key,
            ["local cuisine", "traditional food", "signature dish"]
        )
        queries = [
            f"local cuisine in {city}",
            f"authentic restaurants in {city}",
        ] + [f"{kw} in {city}" for kw in keywords[:3]]
        return {
            "city": city,
            "keywords": keywords[:5],
            "queries": queries[:5],
            "source": "fallback_map",
        }

    def _call_local_food_llm(self, city: str) -> Optional[Dict[str, Any]]:
        if not self.local_food_llm_enabled:
            return None

        url = DASHSCOPE_CHAT_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        }
        prompt = (
            "You are helping a travel recommender system. "
            f"For the city '{city}', return JSON only with 5 concise local food keywords and 5 short restaurant search queries. "
            "Focus on authentic local cuisine and representative dishes. "
            "No markdown. Output format: "
            '{"city":"...","keywords":["..."],"queries":["..."]}'
        )
        payload = {
            "model": TRAVEL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                return None

            data = resp.json()
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            if not content:
                return None

            match = re.search(r"\{.*\}", content, re.DOTALL)
            json_text = match.group(0) if match else content
            parsed = json.loads(json_text)

            keywords = parsed.get("keywords") or []
            queries = parsed.get("queries") or []
            if not isinstance(keywords, list) or not isinstance(queries, list):
                return None

            cleaned_keywords = [self._normalize_text(x) for x in keywords if self._normalize_text(x)][:5]
            cleaned_queries = [self._normalize_text(x) for x in queries if self._normalize_text(x)][:5]
            if not cleaned_keywords:
                return None

            return {
                "city": city,
                "keywords": cleaned_keywords,
                "queries": cleaned_queries or [f"{cleaned_keywords[0]} in {city}"],
                "source": "llm",
            }
        except Exception:
            return None

    def get_local_food_profile(self, city: str) -> Dict[str, Any]:
        city_key = self._normalize_city_key(city)
        if city_key in self.local_food_keyword_cache:
            return self.local_food_keyword_cache[city_key]

        profile = self._call_local_food_llm(city)
        if not profile:
            profile = self._city_local_food_keywords_fallback(city)

        self.local_food_keyword_cache[city_key] = profile
        return profile

    def _fallback_restaurant_tier_from_price_level(self, price_level: Any) -> Dict[str, Any]:
        text = self._normalize_text(price_level).lower()
        mapping = {
            "price_level_inexpensive": 100,
            "inexpensive": 100,
            "price_level_moderate": 300,
            "moderate": 300,
            "price_level_expensive": 500,
            "expensive": 500,
            "price_level_very_expensive": 500,
            "very_expensive": 500,
        }
        tier_price = mapping.get(text, 300)
        return {
            "tier_price_per_person": tier_price,
            "source": "fallback_price_level",
        }

    def _call_restaurant_tier_llm(self, restaurant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.local_food_llm_enabled:
            return None

        url = DASHSCOPE_CHAT_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        }
        city = self._normalize_text(restaurant.get("city"))
        name = self._normalize_text(restaurant.get("name"))
        matched_query = self._normalize_text(restaurant.get("matched_query"))
        price_level = self._normalize_text(restaurant.get("price_level"))
        types = [self._normalize_text(x) for x in (restaurant.get("types") or []) if self._normalize_text(x)]
        rating = self._safe_float(restaurant.get("rating"), 0.0)
        prompt = (
            "You are a travel budgeting assistant. "
            "Classify this restaurant into exactly one per-person meal tier: 100, 300, or 500. "
            "Output JSON only, no markdown. "
            "Format: {\"tier_price_per_person\":100|300|500,\"reason\":\"...\"}. "
            f"City: {city}; Restaurant: {name}; Query: {matched_query}; "
            f"price_level: {price_level}; rating: {rating}; types: {', '.join(types[:6])}"
        )
        payload = {
            "model": TRAVEL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            if not content:
                return None
            match = re.search(r"\{.*\}", content, re.DOTALL)
            json_text = match.group(0) if match else content
            parsed = json.loads(json_text)
            tier_price = self._safe_int(parsed.get("tier_price_per_person"), 0)
            if tier_price not in {100, 300, 500}:
                return None
            return {
                "tier_price_per_person": tier_price,
                "source": "llm_qwen",
                "reason": self._normalize_text(parsed.get("reason")),
            }
        except Exception:
            return None

    def _get_restaurant_tier(self, restaurant: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = (
            f"{self._normalize_city_key(restaurant.get('city', ''))}|"
            f"{self._normalize_text(restaurant.get('name')).lower()}|"
            f"{self._normalize_text(restaurant.get('area')).lower()}"
        )
        if cache_key in self.restaurant_tier_cache:
            return self.restaurant_tier_cache[cache_key]

        tier_result = self._call_restaurant_tier_llm(restaurant)
        if not tier_result:
            tier_result = self._fallback_restaurant_tier_from_price_level(restaurant.get("price_level"))
        self.restaurant_tier_cache[cache_key] = tier_result
        return tier_result

    def _restaurant_local_match_score(self, restaurant: Dict[str, Any]) -> float:
        city = restaurant.get("city", "")
        profile = self.get_local_food_profile(city)
        keywords = [self._normalize_text(x).lower() for x in (profile.get("keywords") or [])]
        text_parts = [
            self._normalize_text(restaurant.get("name")).lower(),
            self._normalize_text(restaurant.get("area")).lower(),
            " ".join([self._normalize_text(x).lower() for x in (restaurant.get("types") or [])]),
            self._normalize_text(restaurant.get("matched_query")).lower(),
        ]
        full_text = " | ".join(text_parts)
        matches = [kw for kw in keywords if kw and kw in full_text]

        if len(matches) >= 3:
            return 1.0
        if len(matches) == 2:
            return 0.8
        if len(matches) == 1:
            return 0.55
        return 0.0

    # =========================================================
    # providers
    # =========================================================
    def search_places_new(
        self,
        text_query: str,
        max_results: int = 10,
        location_bias: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
            "X-Goog-FieldMask": (
                "places.displayName,"
                "places.id,"
                "places.formattedAddress,"
                "places.rating,"
                "places.userRatingCount,"
                "places.types,"
                "places.priceLevel,"
                "places.location"
            ),
        }

        payload: Dict[str, Any] = {
            "textQuery": text_query,
            "pageSize": max_results
        }

        if location_bias and location_bias.get("lat") is not None and location_bias.get("lng") is not None:
            payload["locationBias"] = {
                "circle": {
                    "center": {
                        "latitude": float(location_bias["lat"]),
                        "longitude": float(location_bias["lng"]),
                    },
                    "radius": float(location_bias.get("radius_m", 30000))
                }
            }

        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(f"Places API request failed: {response.status_code}, {response.text}")

        data = response.json()
        return data.get("places", []) or []

    # =========================================================
    # query builders
    # =========================================================
    def _build_hotel_query(self, city: str) -> str:
        level = self.hotel_level_rich or self.hotel_level or self.hotel_pref or "standard"
        features = set(self.hotel_features)

        if level == "budget":
            if "central" in features:
                return f"budget central hotels in {city}"
            return f"budget hotels in {city}"
        if level == "luxury":
            if "central" in features:
                return f"luxury central hotels in {city}"
            return f"luxury hotels in {city}"
        if level == "premium":
            if "central" in features and "clean" in features:
                return f"well-rated central 4 star hotels in {city}"
            if "central" in features:
                return f"central 4 star hotels in {city}"
            if "clean" in features:
                return f"well-rated clean 4 star hotels in {city}"
            return f"4 star hotels in {city}"
        if "central" in features:
            return f"central hotels in {city}"
        if "clean" in features:
            return f"well-rated clean hotels in {city}"
        if self.hotel_pref == "central":
            return f"central hotels in {city}"
        if self.hotel_pref == "clean":
            return f"well-rated clean hotels in {city}"
        return f"Hotels in {city}"

    def _poi_style_query(self, city: str, style: str) -> str:
        if style == "historical":
            return f"historical landmarks museums heritage sites in {city}"
        if style == "nature":
            return f"nature parks scenic spots in {city}"
        if style == "shopping":
            return f"shopping districts malls markets in {city}"
        if style == "art":
            return f"art museums galleries exhibitions in {city}"
        if style == "entertainment":
            return f"entertainment attractions theme parks nightlife in {city}"
        return f"top attractions in {city}"

    def _poi_primary_query(self, city: str) -> str:
        style = (self.attraction_styles[0] if self.attraction_styles else self.attraction_style) or "mixed"
        return self._poi_style_query(city, style)

    def _poi_general_query(self, city: str) -> str:
        return f"top attractions in {city}"

    def _poi_mix_ratio(self) -> float:
        if "mixed" in self.attraction_styles or self.attraction_style == "mixed":
            return 0.5
        if len(self.attraction_styles) > 1:
            return 0.7
        return 0.6

    def _poi_limit_by_intensity(self, city_days: int) -> int:
        base = max(2, city_days * 2)
        signal = self.attraction_priority or self.attraction_intensity
        if signal == "high":
            return base + max(1, city_days // 2)
        if signal == "low":
            return max(2, base - max(1, city_days // 2))
        return base

    def _restaurant_target_count(self, city_days: int) -> int:
        return max(2, city_days * 2)

    # =========================================================
    # POIs / restaurants
    # =========================================================
    def fetch_poi_candidates(self, city: str, max_results: int = 20) -> List[Dict[str, Any]]:
        try:
            style_queries: List[Tuple[str, str]] = []
            active_styles = [s for s in self.attraction_styles if s and s != "mixed"]
            if not active_styles:
                active_styles = [self.attraction_style or "mixed"]

            must_visit_queries: List[Tuple[str, str]] = []
            for target in self.must_visit_attractions[:5]:
                target_text = self._normalize_text(target)
                if target_text:
                    must_visit_queries.append(("must_visit", f"{target_text} in {city}"))

            for style in active_styles[:3]:
                style_queries.append((style, self._poi_style_query(city, style)))

            general_query = self._poi_general_query(city)
            style_count = len(style_queries)
            primary_ratio = self._poi_mix_ratio()
            primary_total = max(1, int(max_results * primary_ratio))
            per_style_count = max(1, primary_total // max(1, style_count))
            general_count = max(1, max_results - per_style_count * style_count)

            merged_raw: List[Tuple[Dict[str, Any], str, str]] = []
            seen = set()

            for matched_style, query in must_visit_queries:
                for place in self.search_places_new(query, max_results=min(4, max_results)):
                    name = self._normalize_text((place.get("displayName") or {}).get("text"))
                    address = self._normalize_text(place.get("formattedAddress"))
                    dedup_key = (name.lower(), address.lower())
                    if not name or dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    merged_raw.append((place, matched_style, query))

            for style, query in style_queries:
                for place in self.search_places_new(query, max_results=per_style_count):
                    name = self._normalize_text((place.get("displayName") or {}).get("text"))
                    address = self._normalize_text(place.get("formattedAddress"))
                    dedup_key = (name.lower(), address.lower())
                    if not name or dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    merged_raw.append((place, style, query))

            for place in self.search_places_new(general_query, max_results=general_count):
                name = self._normalize_text((place.get("displayName") or {}).get("text"))
                address = self._normalize_text(place.get("formattedAddress"))
                dedup_key = (name.lower(), address.lower())
                if not name or dedup_key in seen:
                    continue
                seen.add(dedup_key)
                merged_raw.append((place, "general", general_query))

            results: List[Dict[str, Any]] = []
            for place, matched_style, matched_query in merged_raw[:max_results * 3]:
                lat, lng = self._extract_place_lat_lng(place)
                results.append({
                    "name": place.get("displayName", {}).get("text", ""),
                    "place_id": place.get("id"),
                    "category": "landmark",
                    "rating": place.get("rating", 0),
                    "user_ratings_total": place.get("userRatingCount", 0),
                    "address": place.get("formattedAddress", ""),
                    "city": city,
                    "lat": lat,
                    "lng": lng,
                    "types": place.get("types", []),
                    "price_level": place.get("priceLevel"),
                    "source": "google_places",
                    "attraction_style": matched_style,
                    "matched_query": matched_query,
                })

            return results[:max_results * 2]
        except Exception:
            return []

    
    def fetch_restaurant_candidates(self, city: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch restaurant candidates with stronger recall and stricter restaurant-type filtering.

        Improvements:
        1) Keep the current geographic validation logic.
        2) Expand query coverage to improve candidate count.
        3) Increase per-query recall.
        4) Prefer actual restaurant-like places over districts / nightlife areas / generic POIs.
        5) Preserve downstream fields and overall structure.
        """
        try:
            food_modes = [m for m in self.food_preferences if m and m != "mixed"]
            if not food_modes:
                food_modes = [self.food_pref or "mixed"]

            query_plan: List[Tuple[str, str, str, List[str]]] = []
            seen_queries = set()

            def add_query(mode: str, query: str, keyword_source: str, keywords: List[str]) -> None:
                q = self._normalize_text(query)
                if not q:
                    return
                q_key = q.lower()
                if q_key in seen_queries:
                    return
                seen_queries.add(q_key)
                query_plan.append((mode, q, keyword_source, keywords))

            # Build richer query plan
            for mode in food_modes[:3]:
                if mode == "local":
                    profile = self.get_local_food_profile(city)
                    keywords = [
                        self._normalize_text(x)
                        for x in (profile.get("keywords") or [])
                        if self._normalize_text(x)
                    ]
                    llm_queries = [
                        self._normalize_text(x)
                        for x in (profile.get("queries") or [])
                        if self._normalize_text(x)
                    ]

                    # Primary local-food queries
                    for query in llm_queries[:4]:
                        add_query(mode, query, profile.get("source", "fallback_map"), keywords)

                    # Generic local-food recall queries
                    add_query(mode, f"local cuisine in {city}", "generic_local", keywords)
                    add_query(mode, f"authentic restaurants in {city}", "generic_local", keywords)
                    add_query(mode, f"best local food in {city}", "generic_local", keywords)
                    add_query(mode, f"top rated restaurants in {city}", "generic_local", keywords)

                    # Keyword-specific queries
                    for kw in keywords[:4]:
                        add_query(mode, f"{kw} in {city}", "keyword_local", keywords)

                elif mode == "fine_dining":
                    add_query(mode, f"fine dining restaurants in {city}", "default", [])
                    add_query(mode, f"best fine dining in {city}", "default", [])
                    add_query(mode, f"top rated restaurants in {city}", "default", [])
                    add_query(mode, f"best restaurants in {city}", "default", [])

                elif mode == "street_food":
                    add_query(mode, f"street food in {city}", "default", [])
                    add_query(mode, f"local snacks in {city}", "default", [])
                    add_query(mode, f"cheap eats in {city}", "default", [])
                    add_query(mode, f"casual restaurants in {city}", "default", [])

                else:
                    # mixed / standard fallback
                    add_query(mode, f"restaurants in {city}", "default", [])
                    add_query(mode, f"best restaurants in {city}", "default", [])
                    add_query(mode, f"top rated restaurants in {city}", "default", [])
                    add_query(mode, f"popular restaurants in {city}", "default", [])

            if not query_plan:
                add_query("mixed", f"restaurants in {city}", "default", [])
                add_query("mixed", f"best restaurants in {city}", "default", [])

            # Allow a few more searches now that geo filter is stricter
            query_plan = query_plan[:10]
            per_query_limit = 8
            location_bias = self._get_city_location_bias(city)

            seen_places = set()
            candidates: List[Dict[str, Any]] = []

            # Helper: classify whether a place really looks like a restaurant
            restaurant_positive_types = {
                "restaurant",
                "japanese_restaurant",
                "sushi_restaurant",
                "ramen_restaurant",
                "bar_and_grill",
                "meal_takeaway",
                "food",
            }

            restaurant_negative_types = {
                "tourist_attraction",
                "park",
                "museum",
                "night_club",
                "art_gallery",
                "shopping_mall",
                "movie_theater",
                "event_venue",
                "bar_area",
                "neighborhood",
                "subway_station",
                "train_station",
            }

            def restaurant_type_score(types: List[str]) -> float:
                normalized = {self._normalize_text(t).lower() for t in (types or []) if self._normalize_text(t)}
                score = 0.0

                if "restaurant" in normalized:
                    score += 2.0

                if normalized.intersection(restaurant_positive_types):
                    score += 1.5

                if normalized.intersection(restaurant_negative_types):
                    score -= 1.5

                return score

            def looks_like_area_not_restaurant(name: str, types: List[str]) -> bool:
                n = self._normalize_text(name).lower()
                normalized = {self._normalize_text(t).lower() for t in (types or []) if self._normalize_text(t)}

                suspicious_terms = [
                    "golden gai", "memory lane", "district", "street", "lane",
                    "market", "park", "tower", "garden", "shrine", "temple"
                ]
                if any(term in n for term in suspicious_terms):
                    if "restaurant" not in normalized:
                        return True

                if normalized.intersection(restaurant_negative_types) and "restaurant" not in normalized:
                    return True

                return False

            for mode, query, keyword_source, keywords in query_plan:
                places = self.search_places_new(
                    text_query=query,
                    max_results=per_query_limit,
                    location_bias=location_bias
                )

                for restaurant in places:
                    # Keep your current geographic protection
                    if not self._restaurant_matches_target_city(restaurant, city):
                        continue

                    name = self._normalize_text((restaurant.get("displayName") or {}).get("text"))
                    area = self._normalize_text(restaurant.get("formattedAddress"))
                    types = restaurant.get("types", []) or []

                    if not name or not area:
                        continue

                    if looks_like_area_not_restaurant(name, types):
                        continue

                    dedup_key = (name.lower(), area.lower())
                    if dedup_key in seen_places:
                        continue
                    seen_places.add(dedup_key)

                    lat, lng = self._extract_place_lat_lng(restaurant)
                    item = {
                        "name": name,
                        "place_id": restaurant.get("id"),
                        "area": area,
                        "rating": restaurant.get("rating", 0),
                        "user_ratings_total": restaurant.get("userRatingCount", 0),
                        "city": city,
                        "lat": lat,
                        "lng": lng,
                        "types": types,
                        "price_level": restaurant.get("priceLevel"),
                        "source": "google_places",
                        "matched_query": query,
                        "matched_food_mode": mode,
                        "keyword_source": keyword_source,
                        "local_food_keywords": keywords,
                    }
                    item["local_food_match_score"] = self._restaurant_local_match_score(item)
                    item["restaurant_type_score"] = restaurant_type_score(types)
                    candidates.append(item)

            # Sort candidates more intelligently
            # Priority:
            # 1) local match score
            # 2) restaurant type confidence
            # 3) rating
            # 4) review count
            candidates.sort(
                key=lambda x: (
                    self._safe_float(x.get("local_food_match_score"), 0.0),
                    self._safe_float(x.get("restaurant_type_score"), 0.0),
                    self._safe_float(x.get("rating"), 0.0),
                    self._safe_int(x.get("user_ratings_total"), 0),
                ),
                reverse=True
            )

            return candidates[: max_results * 3]

        except Exception:
            return []

    # =========================================================
    # hotels
    # =========================================================
    def _extract_numeric_price(self, value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            cleaned = value.replace(",", "")
            match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    return None
            return None

        if isinstance(value, dict):
            preferred_keys = [
                "amount", "price", "value", "extracted_price", "lowest", "current_price",
                "nightly_rate", "rate_per_night", "total_rate", "total_price", "price_total"
            ]
            for key in preferred_keys:
                if key in value:
                    extracted = self._extract_numeric_price(value.get(key))
                    if extracted is not None:
                        return extracted

            for v in value.values():
                extracted = self._extract_numeric_price(v)
                if extracted is not None:
                    return extracted

        if isinstance(value, list):
            for item in value:
                extracted = self._extract_numeric_price(item)
                if extracted is not None:
                    return extracted

        return None

    def _normalize_star_value(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return round(float(value), 1)
        if isinstance(value, str):
            text = value.strip().lower()
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                try:
                    return round(float(match.group(1)), 1)
                except Exception:
                    return 0.0
            return 0.0
        if isinstance(value, dict):
            for key in ["stars", "star_rating", "hotel_class", "classification", "rating", "property_class"]:
                if key in value:
                    normalized = self._normalize_star_value(value.get(key))
                    if normalized > 0:
                        return normalized
            for v in value.values():
                normalized = self._normalize_star_value(v)
                if normalized > 0:
                    return normalized
            return 0.0
        return 0.0

    def _extract_star_from_text(self, text: str) -> float:
        text = self._normalize_text(text).lower()
        if not text:
            return 0.0

        patterns = [
            r"(\d(?:\.\d)?)\s*star",
            r"(\d(?:\.\d)?)\s*stars",
            r"(\d(?:\.\d)?)\s*星",
            r"(\d(?:\.\d)?)\s*星级",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return round(float(match.group(1)), 1)
                except Exception:
                    continue

        if "five star" in text or "五星" in text:
            return 5.0
        if "four star" in text or "四星" in text:
            return 4.0
        if "three star" in text or "三星" in text:
            return 3.0
        return 0.0

    def _derive_hotel_stars(self, item: Dict[str, Any]) -> float:
        direct_keys = ["stars", "star_rating", "hotel_class", "classification", "property_class"]
        for key in direct_keys:
            if key in item:
                stars = self._normalize_star_value(item.get(key))
                if stars > 0:
                    return stars

        text_candidates = [
            item.get("name"),
            item.get("description"),
            item.get("type"),
            item.get("category"),
            item.get("category_description"),
            item.get("summary"),
        ]
        for candidate in text_candidates:
            stars = self._extract_star_from_text(self._normalize_text(candidate))
            if stars > 0:
                return stars

        raw_item_str = json.dumps(item, ensure_ascii=False)
        stars = self._extract_star_from_text(raw_item_str)
        if stars > 0:
            return stars

        return 0.0

    def _parse_searchapi_hotel_item(
        self,
        item: Dict[str, Any],
        city: str,
        nights: int
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        name = (
            item.get("name")
            or item.get("hotel_name")
            or item.get("property_name")
            or ((item.get("display_name") or {}).get("text") if isinstance(item.get("display_name"), dict) else item.get("display_name"))
            or ""
        )
        name = self._normalize_text(name)
        if not name:
            return None

        address = (
            item.get("address")
            or item.get("full_address")
            or item.get("formatted_address")
            or item.get("address_line")
            or item.get("location")
            or item.get("neighborhood")
            or item.get("district")
            or ""
        )
        address = self._normalize_text(address)

        user_rating = self._safe_float(item.get("overall_rating") or item.get("rating"), 0.0)
        hotel_stars = self._derive_hotel_stars(item)

        reviews = self._safe_int(
            item.get("reviews")
            or item.get("reviews_count")
            or item.get("user_ratings_total")
            or item.get("review_count"),
            0
        )

        price_level = item.get("price_level")
        total_price = None
        nightly_price = None

        total_candidates = [
            item.get("total_rate"),
            item.get("total_price"),
            item.get("price_total"),
            item.get("gross_price"),
            item.get("rate_total"),
            item.get("prices"),
        ]
        nightly_candidates = [
            item.get("rate_per_night"),
            item.get("nightly_rate"),
            item.get("price"),
            item.get("lowest_price"),
            item.get("price_per_night"),
        ]

        for candidate in total_candidates:
            extracted = self._extract_numeric_price(candidate)
            if extracted is not None:
                total_price = extracted
                break

        for candidate in nightly_candidates:
            extracted = self._extract_numeric_price(candidate)
            if extracted is not None:
                nightly_price = extracted
                break

        if total_price is None and nightly_price is not None:
            total_price = nightly_price * max(1, nights)

        if nightly_price is None and total_price is not None:
            nightly_price = total_price / max(1, nights)

        realtime_total_cost = None
        realtime_nightly_per_room = None
        coordinates = item.get("gps_coordinates") or item.get("coordinates") or item.get("location") or {}
        lat = (
            coordinates.get("latitude")
            if isinstance(coordinates, dict)
            else None
        )
        lng = (
            coordinates.get("longitude")
            if isinstance(coordinates, dict)
            else None
        )
        if lat is None and isinstance(coordinates, dict):
            lat = coordinates.get("lat")
        if lng is None and isinstance(coordinates, dict):
            lng = coordinates.get("lng")

        if total_price is not None:
            realtime_total_cost = round(float(total_price) * self.rooms_needed, 2)
        if nightly_price is not None:
            realtime_nightly_per_room = round(float(nightly_price), 2)

        return {
            "name": name,
            "place_id": item.get("property_token") or item.get("hotel_id") or item.get("id"),
            "area": address,
            "rating": user_rating,
            "stars": hotel_stars,
            "user_ratings_total": reviews,
            "city": city,
            "lat": self._safe_float(lat, None),
            "lng": self._safe_float(lng, None),
            "types": item.get("type", []),
            "price_level": price_level,
            "source": "searchapi_google_hotels",
            "realtime_price_available": realtime_total_cost is not None,
            "realtime_nightly_per_room": realtime_nightly_per_room,
            "realtime_total_cost": realtime_total_cost,
            "raw_hotel": item,
        }

    def search_hotels_searchapi(
        self,
        city: str,
        check_in_date: str,
        check_out_date: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google_hotels",
            "q": self._build_hotel_query(city),
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "currency": self.currency,
            "api_key": SEARCHAPI_API_KEY,
        }

        response = requests.get(url, params=params, timeout=40)
        if response.status_code != 200:
            raise RuntimeError(
                f"SearchAPI hotel request failed: {response.status_code}\n"
                f"params={json.dumps(params, ensure_ascii=False)}\n"
                f"response_text={response.text[:1500]}"
            )

        root = response.json() if response.text else {}
        raw_items = (
            root.get("properties")
            or root.get("hotels")
            or root.get("hotel_results")
            or root.get("results")
            or []
        )

        check_in_dt = self._parse_iso_date(check_in_date)
        check_out_dt = self._parse_iso_date(check_out_date)
        if check_in_dt and check_out_dt:
            nights = max(1, (check_out_dt - check_in_dt).days)
        else:
            nights = max(1, self.duration_days - 1)

        normalized: List[Dict[str, Any]] = []
        seen = set()

        for item in raw_items:
            parsed = self._parse_searchapi_hotel_item(item, city=city, nights=nights)
            if not parsed:
                continue

            dedup_key = (
                parsed.get("name"),
                parsed.get("area"),
                parsed.get("realtime_total_cost"),
            )
            if dedup_key in seen:
                continue

            seen.add(dedup_key)
            normalized.append(parsed)
            if len(normalized) >= max_results:
                break

        return normalized

    def fetch_hotel_candidates_from_google_places(self, city: str, max_results: int = 20) -> List[Dict[str, Any]]:
        try:
            query = self._build_hotel_query(city)
            places = self.search_places_new(query, max_results=max_results)
            hotels = []
            for hotel in places:
                lat, lng = self._extract_place_lat_lng(hotel)
                hotels.append({
                    "name": hotel.get("displayName", {}).get("text", ""),
                    "place_id": hotel.get("id"),
                    "area": hotel.get("formattedAddress", ""),
                    "rating": hotel.get("rating", 0),
                    "stars": 0.0,
                    "user_ratings_total": hotel.get("userRatingCount", 0),
                    "city": city,
                    "lat": lat,
                    "lng": lng,
                    "types": hotel.get("types", []),
                    "price_level": hotel.get("priceLevel"),
                    "source": "google_places_fallback",
                    "realtime_price_available": False,
                    "realtime_nightly_per_room": None,
                    "realtime_total_cost": None,
                })
            return hotels
        except Exception:
            return []

    def fetch_hotel_candidates(self, city: str, city_days: int, max_results: int = 20) -> Tuple[List[Dict[str, Any]], str]:
        date_info = self.city_date_ranges.get(city, {})
        check_in = date_info.get("check_in")
        check_out = date_info.get("check_out")

        if check_in and check_out:
            try:
                hotels = self.search_hotels_searchapi(
                    city=city,
                    check_in_date=check_in,
                    check_out_date=check_out,
                    max_results=max_results
                )
                if hotels:
                    return hotels, "searchapi_google_hotels"
            except Exception:
                pass

        fallback_hotels = self.fetch_hotel_candidates_from_google_places(city, max_results=max_results)
        return fallback_hotels, "google_places_fallback"

    # =========================================================
    # flights
    # =========================================================
    def _searchapi_travel_class(self) -> str:
        if self.flight_pref == "comfort":
            return "business"
        return "economy"

    def _searchapi_stops(self, strict_direct: bool = True) -> str:
        if self.flight_pref == "direct" and strict_direct:
            return "nonstop"
        return "any"

    def _extract_flight_price_value(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            v = float(value)
            return v if v > 0 else None
        if isinstance(value, str):
            text = value.replace(",", "").strip()
            price_patterns = [
                r"(?:CNY|USD|RMB|JPY|EUR|SGD|HKD|¥|\$|€|£)\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:CNY|USD|RMB|JPY|EUR|SGD|HKD)",
            ]
            for pattern in price_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        v = float(match.group(1))
                        return v if v > 0 else None
                    except Exception:
                        return None
            return None
        if isinstance(value, dict):
            preferred_keys = [
                "price", "amount", "value", "total_price", "displayed_price",
                "best_price", "current_price", "lowest_price", "price_total", "fare"
            ]
            for key in preferred_keys:
                if key in value:
                    extracted = self._extract_flight_price_value(value.get(key))
                    if extracted is not None:
                        return extracted
            return None
        if isinstance(value, list):
            for item in value:
                extracted = self._extract_flight_price_value(item)
                if extracted is not None:
                    return extracted
            return None
        return None

    def _resolve_searchapi_flight_price(self, item: Dict[str, Any], root: Dict[str, Any]) -> float:
        candidate_sources = [
            item.get("price"),
            item.get("price_total"),
            item.get("total_price"),
            item.get("displayed_price"),
            item.get("fare"),
            item.get("best_price"),
            item.get("current_price"),
            item.get("lowest_price"),
        ]
        candidate_sources.extend([root.get("price_insights")])

        for source in candidate_sources:
            extracted = self._extract_flight_price_value(source)
            if extracted is not None and extracted > 0:
                if extracted < 100:
                    continue
                return round(extracted, 2)
        return 0.0

    def _normalize_searchapi_flight(
        self,
        item: Dict[str, Any],
        root: Dict[str, Any],
        flight_type: str
    ) -> Dict[str, Any]:
        flights = item.get("flights", []) or []
        first_flight = flights[0] if flights else {}
        last_flight = flights[-1] if flights else {}

        departure_airport = first_flight.get("departure_airport", {}) or {}
        arrival_airport = last_flight.get("arrival_airport", {}) or {}

        airline = first_flight.get("airline") or item.get("airline") or "Unknown"
        airline_logo = item.get("airline_logo") or first_flight.get("airline_logo")
        travel_class = first_flight.get("travel_class")
        layovers = item.get("layovers", []) or []
        stops = len(layovers)

        total_duration = self._safe_int(item.get("total_duration"), 0)
        if total_duration <= 0 and flights:
            total_duration = sum(self._safe_int(seg.get("duration"), 0) for seg in flights)

        price_total = self._resolve_searchapi_flight_price(item, root)

        currency = (
            self._normalize_text(item.get("currency"))
            or self._normalize_text((root.get("search_parameters") or {}).get("currency"))
            or self.currency
            or "CNY"
        )

        departure_code = self._normalize_text(departure_airport.get("id")).upper()
        arrival_code = self._normalize_text(arrival_airport.get("id")).upper()

        if not departure_code:
            departure_code = self._normalize_text(
                (root.get("search_parameters") or {}).get("departure_id")
            ).upper()

        if not arrival_code:
            arrival_code = self._normalize_text(
                (root.get("search_parameters") or {}).get("arrival_id")
            ).upper()

        return {
            "airline": airline,
            "airline_logo": airline_logo,
            "departure_airport": self._normalize_text(departure_airport.get("name")),
            "departure_airport_code": departure_code,
            "departure_time": self._normalize_text(departure_airport.get("time")),
            "arrival_airport": self._normalize_text(arrival_airport.get("name")),
            "arrival_airport_code": arrival_code,
            "arrival_time": self._normalize_text(arrival_airport.get("time")),
            "travel_class": travel_class or self._searchapi_travel_class(),
            "stops": stops,
            "total_duration": total_duration,
            "duration_minutes": total_duration,
            "price_total": price_total,
            "currency": currency,
            "provider": "searchapi_google_flights",
            "flight_type": flight_type,
            "flights": flights,
        }

    def search_flights_searchapi(
        self,
        departure_id: str,
        arrival_id: str,
        outbound_date: str,
        flight_type: str = "one_way",
        return_date: Optional[str] = None,
        adults: int = 1,
        max_results: int = 10,
        strict_direct: bool = True,
    ) -> List[Dict[str, Any]]:
        url = "https://www.searchapi.io/api/v1/search"
        params: Dict[str, Any] = {
            "engine": "google_flights",
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": outbound_date,
            "flight_type": flight_type,
            "currency": self.currency,
            "adults": max(1, self._safe_int(adults, 1)),
            "travel_class": self._searchapi_travel_class(),
            "stops": self._searchapi_stops(strict_direct=strict_direct),
            "sort_by": "price",
            "api_key": SEARCHAPI_API_KEY,
        }
        if return_date and flight_type == "round_trip":
            params["return_date"] = return_date

        response = requests.get(url, params=params, timeout=40)
        if response.status_code != 200:
            raise RuntimeError(
                f"SearchAPI request failed: {response.status_code}\n"
                f"params={json.dumps(params, ensure_ascii=False)}\n"
                f"response_text={response.text[:2000]}"
            )

        root = response.json() if response.text else {}
        best_flights = root.get("best_flights", []) or []
        other_flights = root.get("other_flights", []) or []
        combined = best_flights + other_flights

        normalized: List[Dict[str, Any]] = []
        seen_keys = set()

        for item in combined:
            try:
                normalized_item = self._normalize_searchapi_flight(
                    item=item,
                    root=root,
                    flight_type=flight_type
                )
                normalized_item["direct_query_used"] = strict_direct

                dedup_key = (
                    normalized_item.get("airline"),
                    normalized_item.get("departure_time"),
                    normalized_item.get("arrival_time"),
                    normalized_item.get("price_total"),
                    normalized_item.get("stops")
                )
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                normalized.append(normalized_item)
                if len(normalized) >= max_results:
                    break
            except Exception:
                continue

        return normalized

    # =========================================================
    # estimators
    # =========================================================
    def _price_level_to_multiplier(self, price_level: Any) -> float:
        text = self._normalize_text(price_level).lower()
        mapping = {
            "price_level_inexpensive": 0.75,
            "price_level_moderate": 1.0,
            "price_level_expensive": 1.45,
            "price_level_very_expensive": 2.1,
            "inexpensive": 0.75,
            "moderate": 1.0,
            "expensive": 1.45,
            "very_expensive": 2.1,
        }
        if text in mapping:
            return mapping[text]
        return 1.0

    def _get_city_food_budget_entry(self, city: str) -> Dict[str, Any]:
        city_key = self._normalize_city_key(city)
        for item in self.city_budget_plan:
            if self._normalize_city_key(item.get("city", "")) == city_key:
                return item
        return {}

    def _get_city_food_daily_base(self, city: str, city_days: int) -> float:
        city_days = max(1, city_days)
        city_budget_entry = self._get_city_food_budget_entry(city)
        city_food_budget = self._safe_float(city_budget_entry.get("food_budget"), 0.0)

        if city_food_budget > 0:
            daily_base = city_food_budget / city_days
        elif self.total_budget > 0 and self.final_ratio:
            fallback_food_budget = self.total_budget * self.final_ratio.get("food", 0.0)
            daily_base = fallback_food_budget / max(1, self.duration_days)
        else:
            daily_base = 180.0

        return max(40.0, daily_base)

    def _restaurant_preference_multiplier(self) -> float:
        multiplier = 1.0

        if self.food_level == "premium":
            multiplier *= 1.15
        elif self.food_level == "budget":
            multiplier *= 0.85

        if self.food_pref == "fine_dining":
            multiplier *= 1.25
        elif self.food_pref == "street_food":
            multiplier *= 0.75
        elif self.food_pref == "local":
            multiplier *= 0.95

        return multiplier

    def _estimate_poi_cost(self, poi: Dict[str, Any], city_days: int) -> Dict[str, Any]:
        base_per_person = 80.0
        style = self.attraction_style

        if style == "historical":
            base_per_person = 100.0
        elif style == "nature":
            base_per_person = 60.0
        elif style == "art":
            base_per_person = 110.0
        elif style == "entertainment":
            base_per_person = 150.0
        elif style == "shopping":
            base_per_person = 30.0

        multiplier = self._price_level_to_multiplier(poi.get("price_level"))
        estimated_ticket_per_person = round(base_per_person * multiplier, 2)
        estimated_total_cost = round(estimated_ticket_per_person * self.num_people, 2)

        poi["estimated_ticket_per_person"] = estimated_ticket_per_person
        poi["estimated_total_cost"] = estimated_total_cost
        return poi

    def _estimate_restaurant_cost(self, restaurant: Dict[str, Any], city_days: int) -> Dict[str, Any]:
        tier_result = self._get_restaurant_tier(restaurant)
        estimated_meal_per_person = float(tier_result.get("tier_price_per_person", 300))
        recommended_visits = 1
        estimated_total_cost = round(estimated_meal_per_person * self.num_people * recommended_visits, 2)
        restaurant["estimated_meal_per_person"] = estimated_meal_per_person
        restaurant["recommended_visits"] = recommended_visits
        restaurant["estimated_total_cost"] = estimated_total_cost
        restaurant["meal_tier_per_person"] = int(estimated_meal_per_person)
        restaurant["meal_tier_source"] = self._normalize_text(tier_result.get("source")) or "fallback_price_level"
        if tier_result.get("reason"):
            restaurant["meal_tier_reason"] = self._normalize_text(tier_result.get("reason"))
        return restaurant

    def _estimate_hotel_cost(self, hotel: Dict[str, Any], city_days: int, city_nights: int) -> Dict[str, Any]:
        city_nights = max(1, city_nights)

        if hotel.get("realtime_total_cost") is not None:
            estimated_total_cost = round(self._safe_float(hotel.get("realtime_total_cost"), 0.0), 2)
            nightly_per_room = self._safe_float(hotel.get("realtime_nightly_per_room"), 0.0)
            hotel["estimated_total_cost"] = estimated_total_cost
            hotel["estimated_nightly_per_room"] = nightly_per_room
            return hotel

        base_per_room_per_night = 420.0
        if self.hotel_pref == "budget":
            base_per_room_per_night = 220.0
        elif self.hotel_pref == "luxury":
            base_per_room_per_night = 900.0
        elif self.hotel_pref == "standard":
            base_per_room_per_night = 420.0
        elif self.hotel_pref == "central":
            base_per_room_per_night = 520.0
        elif self.hotel_pref == "clean":
            base_per_room_per_night = 470.0

        if self.hotel_level == "budget":
            base_per_room_per_night *= 0.9
        elif self.hotel_level == "premium":
            base_per_room_per_night *= 1.15
        elif self.hotel_level == "luxury":
            base_per_room_per_night *= 1.35

        if self._safe_float(hotel.get("stars"), 0.0) >= 4.5:
            base_per_room_per_night *= 1.15

        estimated_nightly_per_room = round(base_per_room_per_night, 2)
        estimated_total_cost = round(
            estimated_nightly_per_room * self.rooms_needed * city_nights,
            2
        )

        hotel["estimated_total_cost"] = estimated_total_cost
        hotel["estimated_nightly_per_room"] = estimated_nightly_per_room
        return hotel

    def _attach_city_budget_reference(self, items: List[Dict[str, Any]], city_budget: float, cost_key: str) -> List[Dict[str, Any]]:
        enriched = []
        for item in items:
            copied = dict(item)
            copied["city_budget_reference"] = round(city_budget, 2)
            copied["budget_fit"] = self._safe_float(copied.get(cost_key), 0.0) <= city_budget if city_budget > 0 else True
            enriched.append(copied)
        return enriched


    def _text_contains_any(self, text: str, keywords: List[str]) -> bool:
        text = self._normalize_text(text).lower()
        return any(keyword in text for keyword in keywords if keyword)

    def _hotel_matches_central(self, hotel: Dict[str, Any]) -> bool:
        searchable = " ".join([
            self._normalize_text(hotel.get("name")),
            self._normalize_text(hotel.get("area")),
        ]).lower()
        return self._text_contains_any(searchable, self.hotel_central_keywords)

    def _hotel_matches_clean(self, hotel: Dict[str, Any]) -> bool:
        rating = self._safe_float(hotel.get("rating"), 0.0)
        return rating >= self.hotel_clean_rating_threshold

    def _hotel_matches_level(self, hotel: Dict[str, Any]) -> bool:
        level = (self.hotel_level_rich or self.hotel_level or self.hotel_pref or "standard").lower()
        stars = self._safe_float(hotel.get("stars"), 0.0)
        rating = self._safe_float(hotel.get("rating"), 0.0)
        total = self._safe_float(hotel.get("estimated_total_cost"), 0.0)
        ref = self._safe_float(hotel.get("city_budget_reference"), 0.0)

        if level == "luxury":
            return stars >= 4.5 or rating >= 4.6
        if level == "premium":
            return stars >= 4.0 or rating >= 4.2
        if level == "budget":
            return ref <= 0 or (0 < total <= ref)
        if level == "standard":
            return rating >= 3.8 or stars >= 3.0
        return True

    def _hotel_hard_filter_reasons(self, hotel: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        if not self._hotel_matches_level(hotel):
            reasons.append(f"hotel_level:{self.hotel_level_rich or self.hotel_level or self.hotel_pref or 'standard'}")

        if "central" in self.hotel_features or self.hotel_pref == "central":
            if not self._hotel_matches_central(hotel):
                reasons.append("central")

        if "clean" in self.hotel_features or self.hotel_pref == "clean":
            if not self._hotel_matches_clean(hotel):
                reasons.append("clean")

        return reasons

    def _apply_hotel_hard_filters(
        self,
        hotels: List[Dict[str, Any]],
        fallback_mode: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        if not self.strict_preference_filtering:
            return hotels, "disabled"

        filtered: List[Dict[str, Any]] = []
        for hotel in hotels:
            item = dict(hotel)
            failure_reasons = self._hotel_hard_filter_reasons(item)
            item["hard_filter_failures"] = failure_reasons

            if not fallback_mode:
                if not failure_reasons:
                    filtered.append(item)
                continue

            # Fallback mode: keep budget discipline and a minimum quality floor,
            # but relax preference failures so the system can still output hotels.
            rating = self._safe_float(item.get("rating"), 0.0)
            stars = self._safe_float(item.get("stars"), 0.0)
            total_cost = self._safe_float(item.get("estimated_total_cost"), 0.0)
            budget_ref = self._safe_float(item.get("city_budget_reference"), 0.0)
            central_match = self._hotel_matches_central(item)
            clean_match = self._hotel_matches_clean(item)
            level = (self.hotel_level_rich or self.hotel_level or self.hotel_pref or "standard").lower()

            # Hard keep: never allow obviously poor or wildly over-budget fallback hotels.
            if rating < 4.0 and stars < 3.5:
                continue
            if budget_ref > 0 and total_cost > budget_ref * max(self.hotel_budget_tolerance, 1.15):
                continue

            # Level-aware relaxed thresholds.
            if level == "luxury":
                if not (stars >= 4.0 or rating >= 4.5):
                    continue
            elif level == "premium":
                if not (stars >= 4.0 or rating >= 4.3):
                    continue
            elif level == "budget":
                if budget_ref > 0 and total_cost > budget_ref:
                    continue
            elif level == "standard":
                if rating < 4.0 and stars < 3.0:
                    continue

            # Relax central/clean from hard gates into strong preferences.
            item["fallback_mode"] = True
            item["fallback_relaxed_central"] = ("central" in failure_reasons and not central_match)
            item["fallback_relaxed_clean"] = ("clean" in failure_reasons and not clean_match)
            filtered.append(item)

        return filtered, "applied|fallback" if fallback_mode else "applied"

    def _get_active_food_modes(self) -> List[str]:
        active_modes = [m for m in self.food_preferences if m and m != "mixed"]
        if not active_modes and self.food_pref and self.food_pref != "mixed":
            active_modes = [self.food_pref]
        return active_modes

    def _restaurant_matches_user_preferences(self, restaurant: Dict[str, Any]) -> bool:
        active_modes = self._get_active_food_modes()
        if not active_modes:
            return True
        return any(self._restaurant_matches_food_mode(restaurant, mode) for mode in active_modes)

    def _restaurant_matches_food_mode(self, restaurant: Dict[str, Any], mode: str) -> bool:
        mode = self._normalize_text(mode).lower()
        matched_mode = self._normalize_text(restaurant.get("matched_food_mode")).lower()
        price_multiplier = self._price_level_to_multiplier(restaurant.get("price_level"))
        types = [self._normalize_text(x).lower() for x in restaurant.get("types", [])]
        local_score = self._safe_float(restaurant.get("local_food_match_score"), 0.0)

        if mode == "local":
            return local_score > 0
        if mode == "fine_dining":
            return matched_mode == "fine_dining" or price_multiplier >= 1.35
        if mode == "street_food":
            return matched_mode == "street_food" or "meal_takeaway" in types or "food" in types
        if mode == "mixed":
            return True
        return matched_mode == mode

    def _restaurant_matches_food_level(self, restaurant: Dict[str, Any]) -> bool:
        level = self._normalize_text(self.food_level).lower()
        price_multiplier = self._price_level_to_multiplier(restaurant.get("price_level"))
        if level == "premium":
            return price_multiplier >= 1.35
        if level == "budget":
            return price_multiplier <= 1.0
        return True

    def _restaurant_hard_filter_reasons(self, restaurant: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []

        if not self._restaurant_matches_user_preferences(restaurant):
            reasons.append("food_preferences")

        if not self._restaurant_matches_food_level(restaurant):
            reasons.append(f"food_level:{self.food_level}")

        return reasons

    def _apply_restaurant_hard_filters(self, restaurants: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        if not self.strict_preference_filtering:
            return restaurants, "disabled"

        strict_pass: List[Dict[str, Any]] = []
        user_pref_pass: List[Dict[str, Any]] = []
        food_level_pass: List[Dict[str, Any]] = []
        fallback_all: List[Dict[str, Any]] = []

        for restaurant in restaurants:
            item = dict(restaurant)

            user_pref_ok = self._restaurant_matches_user_preferences(item)
            food_level_ok = self._restaurant_matches_food_level(item)

            failure_reasons: List[str] = []
            if not user_pref_ok:
                failure_reasons.append("food_preferences")
            if not food_level_ok:
                failure_reasons.append(f"food_level:{self.food_level}")

            item["hard_filter_failures"] = failure_reasons
            item["user_preference_match"] = user_pref_ok
            item["food_level_match"] = food_level_ok

            fallback_all.append(item)

            if user_pref_ok and food_level_ok:
                strict_pass.append(item)
            elif user_pref_ok:
                item["fallback_mode"] = True
                item["restaurant_fallback_reason"] = "kept_user_preference_relaxed_food_level"
                user_pref_pass.append(item)
            elif food_level_ok:
                item["fallback_mode"] = True
                item["restaurant_fallback_reason"] = "kept_food_level_relaxed_user_preference"
                food_level_pass.append(item)

        if strict_pass:
            return strict_pass, "applied_strict"

        if user_pref_pass:
            return user_pref_pass, "applied_fallback_user_preference_first"

        if food_level_pass:
            return food_level_pass, "applied_fallback_food_level_only"

        for item in fallback_all:
            item["fallback_mode"] = True
            item["restaurant_fallback_reason"] = "kept_all_candidates_no_strict_match"

        return fallback_all, "applied_fallback_all"

    def _poi_matches_style(self, poi: Dict[str, Any], style: str) -> bool:
        style = self._normalize_text(style).lower()
        types_text = " ".join([self._normalize_text(x).lower() for x in (poi.get("types") or [])])
        name_text = self._normalize_text(poi.get("name")).lower()
        full = f"{name_text} {types_text}"

        if style == "historical":
            return any(k in full for k in ["museum", "heritage", "monument", "historic", "history", "palace", "temple", "castle"])
        if style == "nature":
            return any(k in full for k in ["park", "garden", "scenic", "mountain", "nature", "lake"])
        if style == "shopping":
            return any(k in full for k in ["mall", "market", "shopping", "outlet"])
        if style == "art":
            return any(k in full for k in ["gallery", "museum", "art", "exhibition"])
        if style == "entertainment":
            return any(k in full for k in ["theme", "entertainment", "night", "show", "theater", "theatre"])
        return True

    def _poi_hard_filter_reasons(self, poi: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        styles = [s for s in self.attraction_styles if s and s != "mixed"]
        if not styles and self.attraction_style and self.attraction_style != "mixed":
            styles = [self.attraction_style]
        if styles and not any(self._poi_matches_style(poi, style) for style in styles):
            reasons.append("attraction_styles")
        return reasons

    def _apply_poi_hard_filters(self, pois: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        if not self.strict_preference_filtering:
            return pois, "disabled"
        filtered: List[Dict[str, Any]] = []
        for poi in pois:
            item = dict(poi)
            failure_reasons = self._poi_hard_filter_reasons(item)
            item["hard_filter_failures"] = failure_reasons
            if not failure_reasons:
                filtered.append(item)
        return filtered, "applied"

    def _flight_hard_filter_reasons(self, flight: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        pref = self._normalize_text(self.flight_pref).lower()
        travel_class = self._normalize_text(flight.get("travel_class")).lower()
        stops = self._safe_int(flight.get("stops"), 0)

        if pref == "direct" and stops != 0:
            reasons.append("direct")
        if pref == "comfort" and travel_class != "business":
            reasons.append("comfort")
        return reasons

    def _apply_flight_hard_filters(self, flights: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        if not self.strict_preference_filtering:
            return flights, "disabled"
        filtered: List[Dict[str, Any]] = []
        for flight in flights:
            item = dict(flight)
            failure_reasons = self._flight_hard_filter_reasons(item)
            item["hard_filter_failures"] = failure_reasons
            if not failure_reasons:
                filtered.append(item)
        return filtered, "applied"

    def _dedupe_flights(self, flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for flight in flights:
            key = (
                self._normalize_text(flight.get("airline")).lower(),
                self._normalize_text(flight.get("departure_time")),
                self._normalize_text(flight.get("arrival_time")),
                round(self._safe_float(flight.get("price_total"), 0.0), 2),
                self._safe_int(flight.get("stops"), 0),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(flight)
        return deduped

    def _choose_direct_over_layover(self, flights: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        directs = [f for f in flights if self._safe_int(f.get("stops"), 0) == 0 and self._safe_float(f.get("price_total"), 0.0) > 0]
        layovers = [f for f in flights if self._safe_int(f.get("stops"), 0) > 0 and self._safe_float(f.get("price_total"), 0.0) > 0]
        if not directs or not layovers:
            return None

        best_direct = sorted(directs, key=lambda x: (-self._safe_float(x.get("score"), 0.0), self._safe_float(x.get("price_total"), 0.0)))[0]
        best_layover = sorted(layovers, key=lambda x: (-self._safe_float(x.get("score"), 0.0), self._safe_float(x.get("price_total"), 0.0)))[0]

        direct_price = self._safe_float(best_direct.get("price_total"), 0.0)
        layover_price = self._safe_float(best_layover.get("price_total"), 0.0)
        direct_duration = self._safe_float(best_direct.get("total_duration"), 0.0)
        layover_duration = self._safe_float(best_layover.get("total_duration"), 0.0)

        if direct_price <= 0 or layover_price <= 0 or direct_duration <= 0 or layover_duration <= 0:
            return None

        price_gap_ratio = (direct_price - layover_price) / layover_price
        if price_gap_ratio <= self.prefer_direct_price_gap_ratio and layover_duration >= direct_duration * self.prefer_direct_duration_ratio:
            chosen = dict(best_direct)
            chosen["why_selected"] = chosen.get("why_selected", "") + "; direct preferred because price gap is within 15% and layover duration is at least 2x"
            return chosen
        return None

    # =========================================================
    # ranking
    # =========================================================
    def _hotel_preference_match_score(self, hotel: Dict[str, Any]) -> float:
        score = 0.0
        stars = self._safe_float(hotel.get("stars"), 0.0)
        area = self._normalize_text(hotel.get("area")).lower()
        rating = self._safe_float(hotel.get("rating"), 0.0)
        ref = self._safe_float(hotel.get("city_budget_reference"), 0.0)
        total = self._safe_float(hotel.get("estimated_total_cost"), 0.0)

        level = self.hotel_level_rich or self.hotel_level or self.hotel_pref or "standard"
        if level == "luxury":
            if stars >= 4.7:
                score += 1.2
            elif stars >= 4.3:
                score += 0.9
        elif level == "premium":
            if stars >= 4.0:
                score += 1.0
            elif rating >= 4.2:
                score += 0.7
        elif level == "budget":
            if ref <= 0 or (0 < total <= ref):
                score += 1.0
            if ref > 0 and 0 < total <= ref * 0.9:
                score += 0.3
        else:
            if rating >= 4.0:
                score += 0.6

        if "central" in self.hotel_features:
            if any(k in area for k in ["central", "downtown", "city center", "centre"]):
                score += 0.8
        elif self.hotel_pref == "central":
            if any(k in area for k in ["central", "downtown", "city center", "centre"]):
                score += 0.8

        if "clean" in self.hotel_features:
            if rating >= 4.3:
                score += 0.6
        elif self.hotel_pref == "clean" and rating >= 4.3:
            score += 0.6

        return score

    def _rank_hotels(self, hotels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []
        for hotel in hotels:
            item = dict(hotel)
            pref_score = self._hotel_preference_match_score(item)
            rating = self._safe_float(item.get("rating"), 0.0)
            stars = self._safe_float(item.get("stars"), 0.0)
            reviews = self._safe_int(item.get("user_ratings_total"), 0)
            budget_fit = 1.0 if item.get("budget_fit") else 0.0
            realtime = 1.0 if item.get("realtime_price_available") else 0.0

            score = (
                3.0 * pref_score
                + 1.8 * rating
                + 0.6 * stars
                + 0.5 * budget_fit
                + 0.4 * realtime
                + min(reviews / 1000.0, 0.8)
            )
            item["score"] = round(score, 4)

            reasons = []
            if pref_score > 0:
                reasons.append(f"matches hotel preference '{self.hotel_level_rich or self.hotel_pref or self.hotel_level or 'standard'}'")
            if "central" in self.hotel_features:
                reasons.append("considers central location preference")
            if "clean" in self.hotel_features:
                reasons.append("considers clean hotel preference")
            if item.get("budget_fit"):
                reasons.append("fits city hotel budget")
            if rating > 0:
                reasons.append(f"well rated ({rating})")
            if stars > 0:
                reasons.append(f"hotel stars {stars}")
            if realtime:
                reasons.append("real-time hotel price available")

            item["why_selected"] = "; ".join(reasons) if reasons else "selected by balanced hotel ranking"
            ranked.append(item)

        ranked.sort(
            key=lambda x: (x.get("score", 0), -self._safe_float(x.get("estimated_total_cost"), 0.0)),
            reverse=True
        )
        return ranked

    def _restaurant_preference_match_score(self, restaurant: Dict[str, Any]) -> float:
        score = 0.0
        active_modes = [m for m in self.food_preferences if m and m != "mixed"]
        if not active_modes:
            active_modes = [self.food_pref or "mixed"]

        matched_mode = self._normalize_text(restaurant.get("matched_food_mode")).lower()
        price_multiplier = self._price_level_to_multiplier(restaurant.get("price_level"))
        types = [self._normalize_text(x).lower() for x in restaurant.get("types", [])]

        for mode in active_modes:
            if mode == "local":
                score += 1.6 * self._safe_float(restaurant.get("local_food_match_score"), 0.0)
            elif mode == "fine_dining":
                if price_multiplier >= 1.35:
                    score += 0.9
                if matched_mode == "fine_dining":
                    score += 0.3
            elif mode == "street_food":
                if "meal_takeaway" in types or "food" in types:
                    score += 0.8
                if matched_mode == "street_food":
                    score += 0.2
            elif mode == "mixed":
                score += 0.3

        return score

    def _rank_restaurants(self, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []
        for restaurant in restaurants:
            item = dict(restaurant)
            pref_score = self._restaurant_preference_match_score(item)
            rating = self._safe_float(item.get("rating"), 0.0)
            reviews = self._safe_int(item.get("user_ratings_total"), 0)
            budget_fit = 1.0 if item.get("budget_fit") else 0.0

            score = (
                3.0 * pref_score
                + 1.8 * rating
                + 0.5 * budget_fit
                + min(reviews / 1000.0, 0.8)
            )
            item["score"] = round(score, 4)

            reasons = []
            fallback_reason = self._normalize_text(item.get("restaurant_fallback_reason"))
            if fallback_reason == "kept_user_preference_relaxed_food_level":
                reasons.append("kept user food preference after relaxing Agent2 food level")
            elif fallback_reason == "kept_food_level_relaxed_user_preference":
                reasons.append("kept Agent2 food level after relaxing user food preference")
            elif fallback_reason == "kept_all_candidates_no_strict_match":
                reasons.append("no strict restaurant match found, used broad fallback")

            if self._safe_float(item.get("local_food_match_score"), 0.0) > 0:
                reasons.append("matches local food preference")
            if item.get("budget_fit"):
                reasons.append("fits city food budget")
            if rating > 0:
                reasons.append(f"well rated ({rating})")

            item["why_selected"] = "; ".join(reasons) if reasons else "selected by balanced restaurant ranking"
            ranked.append(item)

        ranked.sort(
            key=lambda x: (x.get("score", 0), -self._safe_float(x.get("estimated_total_cost"), 0.0)),
            reverse=True
        )
        return ranked

    def _normalize_place_name_for_match(self, text: str) -> str:
        text = self._normalize_text(text).lower()
        text = re.sub(r"[^a-z0-9一-鿿\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _poi_must_visit_match_score(self, poi: Dict[str, Any]) -> float:
        if not self.must_visit_attractions:
            return 0.0

        name_text = self._normalize_place_name_for_match(poi.get("name"))
        address_text = self._normalize_place_name_for_match(poi.get("address"))
        query_text = self._normalize_place_name_for_match(poi.get("matched_query"))
        full_text = f"{name_text} {address_text} {query_text}"

        best_score = 0.0

        for target in self.must_visit_attractions:
            target_norm = self._normalize_place_name_for_match(target)
            if not target_norm:
                continue

            if target_norm == name_text:
                best_score = max(best_score, 1.0)
            elif target_norm in name_text:
                best_score = max(best_score, 0.95)
            elif target_norm in full_text:
                best_score = max(best_score, 0.85)

            if target_norm in {"great wall", "the great wall", "长城"}:
                if any(k in full_text for k in [
                    "great wall", "长城", "mutianyu", "badaling", "jinshanling", "simatai"
                ]):
                    best_score = max(best_score, 0.95)

        return best_score

    def _poi_style_match_score(self, poi: Dict[str, Any]) -> float:
        styles = [s for s in self.attraction_styles if s and s != "mixed"]
        if not styles:
            styles = [self.attraction_style or "mixed"]
        types_text = " ".join([self._normalize_text(x).lower() for x in (poi.get("types") or [])])
        name_text = self._normalize_text(poi.get("name")).lower()
        full = f"{name_text} {types_text}"

        style_scores = []
        for style in styles:
            if style == "historical":
                style_scores.append(1.0 if any(k in full for k in ["museum", "heritage", "monument", "historic", "history", "palace", "temple"]) else 0.2)
            elif style == "nature":
                style_scores.append(1.0 if any(k in full for k in ["park", "garden", "scenic", "mountain", "nature", "lake"]) else 0.2)
            elif style == "shopping":
                style_scores.append(1.0 if any(k in full for k in ["mall", "market", "shopping"]) else 0.2)
            elif style == "art":
                style_scores.append(1.0 if any(k in full for k in ["gallery", "museum", "art", "exhibition"]) else 0.2)
            elif style == "entertainment":
                style_scores.append(1.0 if any(k in full for k in ["theme", "entertainment", "night", "show", "theater"]) else 0.2)
            else:
                style_scores.append(0.5)

        return max(style_scores) if style_scores else 0.5

    def _rank_pois(self, pois: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []
        for poi in pois:
            item = dict(poi)
            style_score = self._poi_style_match_score(item)
            must_visit_score = self._poi_must_visit_match_score(item)
            rating = self._safe_float(item.get("rating"), 0.0)
            reviews = self._safe_int(item.get("user_ratings_total"), 0)
            budget_fit = 1.0 if item.get("budget_fit") else 0.0

            score = (
                5.0 * must_visit_score
                + 2.8 * style_score
                + 1.8 * rating
                + 0.5 * budget_fit
                + min(reviews / 1000.0, 0.8)
            )
            item["score"] = round(score, 4)
            item["must_visit_match_score"] = round(must_visit_score, 4)
            item["is_must_visit"] = must_visit_score >= 0.85

            reasons = []
            if item["is_must_visit"]:
                reasons.append("matches user-specified must-visit attraction")
            if style_score >= 1.0:
                reasons.append(f"matches attraction style '{', '.join(self.attraction_styles) if self.attraction_styles else self.attraction_style}'")
            if item.get("budget_fit"):
                reasons.append("fits city attraction budget")
            if rating > 0:
                reasons.append(f"well rated ({rating})")

            item["why_selected"] = "; ".join(reasons) if reasons else "selected by balanced POI ranking"
            ranked.append(item)

        ranked.sort(
            key=lambda x: (
                x.get("is_must_visit", False),
                x.get("score", 0),
                -self._safe_float(x.get("estimated_total_cost"), 0.0)
            ),
            reverse=True
        )
        return ranked

    # =========================================================
    # selectors
    # =========================================================
    def select_flights_with_budget(
        self,
        flights: List[Dict[str, Any]],
        budget: float,
        max_items: int = 1,
        tolerance: float = 1.15,
        relaxed_tolerance: float = 1.25,
    ) -> Tuple[List[Dict[str, Any]], str]:
        if not flights:
            return [], "no_candidates"

        ranked = sorted(
            flights,
            key=lambda x: (x.get("score", 0), -self._safe_float(x.get("price_total"), 0.0)),
            reverse=True
        )

        strict = [f for f in ranked if 0 < self._safe_float(f.get("price_total"), 0.0) <= budget]
        if strict:
            return strict[:max_items], "within_budget"

        slightly_over = [f for f in ranked if 0 < self._safe_float(f.get("price_total"), 0.0) <= budget * tolerance]
        if slightly_over:
            return slightly_over[:max_items], "slightly_over_budget"

        relaxed = [f for f in ranked if 0 < self._safe_float(f.get("price_total"), 0.0) <= budget * relaxed_tolerance]
        if relaxed:
            return relaxed[:max_items], "relaxed_over_budget"

        priced = [f for f in ranked if self._safe_float(f.get("price_total"), 0.0) > 0]
        if priced:
            closest = sorted(priced, key=lambda x: abs(self._safe_float(x.get("price_total"), 0.0) - budget))
            return closest[:max_items], "closest_to_budget"

        return ranked[:max_items], "price_unavailable"

    def select_with_budget(
        self,
        items: List[Dict[str, Any]],
        budget: float,
        cost_key: str,
        max_items: int,
        tolerance: float = 1.0
    ) -> Tuple[List[Dict[str, Any]], str]:
        if not items:
            return [], "no_candidates"

        ranked = sorted(
            items,
            key=lambda x: (x.get("score", 0), -self._safe_float(x.get(cost_key), 0.0)),
            reverse=True
        )

        selected = []
        spent = 0.0

        for item in ranked:
            item_cost = self._safe_float(item.get(cost_key), 0.0)
            if len(selected) >= max_items:
                break
            if item_cost <= 0:
                continue
            if budget <= 0 or spent + item_cost <= budget:
                selected.append(item)
                spent += item_cost

        if selected:
            return selected, "within_budget"

        relaxed = []
        for item in ranked:
            item_cost = self._safe_float(item.get(cost_key), 0.0)
            if item_cost <= 0:
                continue
            if budget <= 0 or item_cost <= budget * tolerance:
                relaxed.append(item)
            if len(relaxed) >= max_items:
                break

        if relaxed:
            return relaxed, "slightly_over_budget"

        return ranked[:max_items], "not_found_within_tolerance"

    def select_pois_with_must_visit(
        self,
        items: List[Dict[str, Any]],
        budget: float,
        cost_key: str,
        max_items: int,
        tolerance: float = 1.0
    ) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        warnings: List[str] = []
        if not items:
            return [], "no_candidates", warnings

        ranked = sorted(
            items,
            key=lambda x: (
                x.get("is_must_visit", False),
                x.get("score", 0),
                -self._safe_float(x.get(cost_key), 0.0)
            ),
            reverse=True
        )

        selected: List[Dict[str, Any]] = []
        used_names = set()
        spent = 0.0
        effective_budget = budget * tolerance if budget > 0 else 0.0

        must_visit_candidates = [x for x in ranked if x.get("is_must_visit")]
        for poi in must_visit_candidates:
            if len(selected) >= max_items:
                break

            name_key = self._normalize_place_name_for_match(poi.get("name"))
            if not name_key or name_key in used_names:
                continue

            item_cost = self._safe_float(poi.get(cost_key), 0.0)
            if item_cost <= 0:
                continue

            can_take = False
            if budget <= 0:
                can_take = True
            elif spent + item_cost <= effective_budget:
                can_take = True
            elif not selected and spent == 0:
                can_take = True

            if can_take:
                selected.append(poi)
                used_names.add(name_key)
                spent += item_cost

        for item in ranked:
            if len(selected) >= max_items:
                break

            name_key = self._normalize_place_name_for_match(item.get("name"))
            if not name_key or name_key in used_names:
                continue

            item_cost = self._safe_float(item.get(cost_key), 0.0)
            if item_cost <= 0:
                continue
            if budget > 0 and spent + item_cost > effective_budget:
                continue

            selected.append(item)
            used_names.add(name_key)
            spent += item_cost

        matched_targets = set()
        for poi in selected:
            poi_text = self._normalize_place_name_for_match(
                f"{poi.get('name', '')} {poi.get('address', '')} {poi.get('matched_query', '')}"
            )
            for target in self.must_visit_attractions:
                target_norm = self._normalize_place_name_for_match(target)
                if not target_norm:
                    continue
                if target_norm in poi_text:
                    matched_targets.add(target)
                elif target_norm in {"great wall", "the great wall", "长城"} and any(
                    k in poi_text for k in ["great wall", "长城", "mutianyu", "badaling", "jinshanling", "simatai"]
                ):
                    matched_targets.add(target)

        unmet = [x for x in self.must_visit_attractions if x not in matched_targets]
        for target in unmet:
            warnings.append(
                f"Must-visit attraction warning: '{target}' was requested but could not be selected from available candidates."
            )

        if selected:
            if unmet:
                return selected, "partial_with_must_visit_missing", warnings
            if budget > 0 and spent > budget:
                return selected, "slightly_over_budget", warnings
            return selected, "within_budget", warnings

        relaxed = []
        for item in ranked:
            item_cost = self._safe_float(item.get(cost_key), 0.0)
            if item_cost <= 0:
                continue
            if budget <= 0 or item_cost <= budget * tolerance:
                relaxed.append(item)
            if len(relaxed) >= max_items:
                break

        if relaxed:
            return relaxed, "not_found_within_tolerance", warnings

        return ranked[:max_items], "not_found_within_tolerance", warnings

    def select_hotels_with_minimum(
        self,
        hotels: List[Dict[str, Any]],
        budget: float,
        max_items: int,
        minimum_items: int = 2,
        tolerance: float = 1.2
    ) -> Tuple[List[Dict[str, Any]], str]:
        if not hotels:
            return [], "no_candidates"

        ranked = sorted(
            hotels,
            key=lambda x: (x.get("score", 0), -self._safe_float(x.get("estimated_total_cost"), 0.0)),
            reverse=True
        )

        within_budget = []
        slightly_over_budget = []

        for hotel in ranked:
            hotel_cost = self._safe_float(hotel.get("estimated_total_cost"), 0.0)

            if hotel_cost <= 0:
                within_budget.append(hotel)
            elif budget <= 0:
                within_budget.append(hotel)
            elif hotel_cost <= budget:
                within_budget.append(hotel)
            elif hotel_cost <= budget * tolerance:
                slightly_over_budget.append(hotel)

        if len(within_budget) >= minimum_items:
            return within_budget[:max_items], "within_budget"

        combined = within_budget + slightly_over_budget
        if len(combined) >= minimum_items:
            return combined[:max_items], "slightly_over_budget"

        if combined:
            return combined[:max_items], "partial_selection"

        return ranked[:max_items], "not_found_within_tolerance"

    def _get_hotel_reference_total_by_city(self, selected_hotels: List[Dict[str, Any]]) -> float:
        city_min_map: Dict[str, float] = {}

        for hotel in selected_hotels:
            city = self._normalize_text(hotel.get("city"))
            cost = self._safe_float(hotel.get("estimated_total_cost"), 0.0)
            if not city or cost <= 0:
                continue

            if city not in city_min_map or cost < city_min_map[city]:
                city_min_map[city] = cost

        return round(sum(city_min_map.values()), 2)

    # =========================================================
    # flight leg helpers
    # =========================================================
    def _leg_key(self, leg: Dict[str, Any]) -> str:
        return f"{leg.get('leg_type', 'unknown')}::{leg.get('leg_index', 0)}::{leg.get('from_city', '')}->{leg.get('to_city', '')}"

    def _time_to_minutes(self, value: Any) -> Optional[int]:
        text = self._normalize_text(value)
        if not text:
            return None
        for fmt in ("%H:%M", "%Y-%m-%d %H:%M", "%I:%M %p"):
            try:
                dt = datetime.strptime(text, fmt)
                return dt.hour * 60 + dt.minute
            except Exception:
                pass
        match = re.search(r"(\d{1,2}):(\d{2})", text)
        if match:
            hour = self._safe_int(match.group(1), -1)
            minute = self._safe_int(match.group(2), -1)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return hour * 60 + minute
        return None

    def _flight_timing_score(self, flight: Dict[str, Any], leg_type: str) -> float:
        dep = self._time_to_minutes(flight.get("departure_time"))
        arr = self._time_to_minutes(flight.get("arrival_time"))
        leg = self._normalize_text(leg_type).lower()

        if leg == "outbound":
            if arr is None:
                return 0.0
            if arr <= 12 * 60:
                return 1.3
            if arr <= 15 * 60:
                return 0.9
            if arr <= 17 * 60 + 30:
                return 0.35
            if arr >= 20 * 60:
                return -0.8
            return -0.25

        if leg == "inbound":
            if dep is None:
                return 0.0
            if 16 * 60 <= dep <= 20 * 60:
                return 1.0
            if 13 * 60 <= dep < 16 * 60:
                return 0.45
            if dep < 11 * 60:
                return -0.65
            if dep > 21 * 60:
                return -0.25
            return 0.15

        if dep is None:
            return 0.0
        if 8 * 60 <= dep <= 12 * 60:
            return 0.8
        if 12 * 60 < dep <= 15 * 60:
            return 0.35
        if dep >= 18 * 60:
            return -0.45
        return 0.0

    def _score_flight(self, flight: Dict[str, Any], leg_type: str) -> Dict[str, Any]:
        item = dict(flight)

        direct_bonus = 1.2 if self._safe_int(item.get("stops"), 0) == 0 else 0.0
        comfort_bonus = 0.8 if (
            self.flight_pref == "comfort"
            and self._normalize_text(item.get("travel_class")).lower() == "business"
        ) else 0.0
        cheap_bonus = 0.5 if self.flight_pref == "cheap" else 0.0

        rating_like = 0.0
        if item.get("price_total", 0) > 0:
            rating_like = 1.0 / max(1.0, self._safe_float(item.get("price_total"), 1.0) / 1000.0)

        timing_score = self._flight_timing_score(item, leg_type)
        timing_weight = 0.55 if self.flight_pref == "cheap" or self.flight_priority == "save_money" else 1.0

        score = direct_bonus + comfort_bonus + cheap_bonus + rating_like + timing_score * timing_weight
        item["score"] = round(score, 4)

        item["why_selected"] = (
            f"selected for {leg_type} leg based on flight preference "
            f"'{self.flight_pref or self.flight_type or 'balanced'}'; "
            f"stops={self._safe_int(item.get('stops'), 0)}, "
            f"duration={self._safe_int(item.get('duration_minutes') or item.get('total_duration'), 0)} minutes, "
            f"timing_score={round(timing_score, 2)}, "
            f"price={round(self._safe_float(item.get('price_total'), 0.0), 2)} "
            f"{item.get('currency') or self.currency}"
        )
        return item

    def _annotate_flight_ground_time(self, flight: Dict[str, Any], leg: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(flight)
        from_city = self._normalize_text(leg.get("from_city"))
        to_city = self._normalize_text(leg.get("to_city"))
        departure_code = (
            self._normalize_text(item.get("departure_code"))
            or self._normalize_text(item.get("departure_airport_code"))
        ).upper()
        arrival_code = (
            self._normalize_text(item.get("arrival_code"))
            or self._normalize_text(item.get("arrival_airport_code"))
        ).upper()

        origin_ground = self._estimate_city_airport_drive_metrics(from_city, departure_code)
        destination_ground = self._estimate_city_airport_drive_metrics(to_city, arrival_code)
        origin_ground_h = self._safe_float((origin_ground or {}).get("duration_h"), 0.0)
        destination_ground_h = self._safe_float((destination_ground or {}).get("duration_h"), 0.0)
        flight_h = self._safe_float(item.get("duration_minutes") or item.get("total_duration"), 0.0) / 60.0
        comparable_total_h = origin_ground_h + self.flight_buffer_hours + flight_h + destination_ground_h

        item["ground_to_departure_airport_hours"] = round(origin_ground_h, 2)
        item["ground_from_arrival_airport_hours"] = round(destination_ground_h, 2)
        item["flight_buffer_hours"] = round(self.flight_buffer_hours, 2)
        item["flight_total_comparable_hours"] = round(comparable_total_h, 2)
        item["direct_drive_comparable_hours"] = leg.get("direct_drive_hours")
        item["direct_drive_distance_km"] = leg.get("direct_drive_distance_km")
        item["airport_time_comparison"] = {
            "direct_drive_hours": leg.get("direct_drive_hours"),
            "direct_drive_distance_km": leg.get("direct_drive_distance_km"),
            "direct_drive_source": leg.get("direct_drive_source"),
            "ground_to_departure_airport_hours": round(origin_ground_h, 2),
            "flight_buffer_hours": round(self.flight_buffer_hours, 2),
            "flight_hours": round(flight_h, 2),
            "ground_from_arrival_airport_hours": round(destination_ground_h, 2),
            "flight_total_hours": round(comparable_total_h, 2),
            "departure_airport": departure_code,
            "arrival_airport": arrival_code,
        }
        return item

    def _filter_flights_by_drive_time_comparison(
        self,
        flights: List[Dict[str, Any]],
        leg: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], str]:
        if not flights:
            return [], "no_flights_to_compare"

        direct_drive_h = self._safe_float(leg.get("direct_drive_hours"), 0.0)
        if direct_drive_h <= 0:
            return flights, "drive_time_unavailable"

        annotated = [self._annotate_flight_ground_time(flight, leg) for flight in flights]
        faster_or_equal_flights = [
            flight for flight in annotated
            if self._safe_float(flight.get("flight_total_comparable_hours"), 99999.0) <= direct_drive_h
        ]
        if faster_or_equal_flights:
            return faster_or_equal_flights, "flight_faster_than_direct_drive"

        if direct_drive_h <= self.max_direct_drive_hours_for_auto_drive:
            return [], "direct_drive_faster"

        # For long direct drives, keep the least-bad flight candidates but mark the comparison.
        best = sorted(
            annotated,
            key=lambda item: self._safe_float(item.get("flight_total_comparable_hours"), 99999.0),
        )
        for item in best:
            item["why_selected"] = (
                self._normalize_text(item.get("why_selected"))
                + "; flight kept although direct drive is estimated faster because direct drive exceeds the 5h drive constraint"
            ).strip()
        return best, "direct_drive_faster_but_over_drive_limit"

    def _leg_is_short_distance(self, leg: Dict[str, Any]) -> bool:
        from_city = self._normalize_city_key(leg.get("from_city"))
        to_city = self._normalize_city_key(leg.get("to_city"))

        known_near_pairs = {
            ("tokyo", "osaka"): True,
            ("osaka", "kyoto"): True,
            ("paris", "brussels"): True,
            ("vienna", "prague"): True,
        }

        if (from_city, to_city) in known_near_pairs:
            return known_near_pairs[(from_city, to_city)]
        if (to_city, from_city) in known_near_pairs:
            return known_near_pairs[(to_city, from_city)]

        return False

    def _direct_drive_cost_estimate(self) -> float:
        level = self._normalize_text(self.cost_level).lower()
        if "low" in level or "budget" in level:
            return 300.0
        if "high" in level or "luxury" in level:
            return 500.0
        return 400.0

    def _should_select_direct_drive(self, leg: Dict[str, Any], fetch_status: str) -> bool:
        direct_drive_h = self._safe_float(leg.get("direct_drive_hours"), 0.0)
        if direct_drive_h <= 0:
            return False
        if "direct_drive_faster" in self._normalize_text(fetch_status):
            return direct_drive_h <= self.max_direct_drive_hours_for_auto_drive
        if "missing_route_info" in self._normalize_text(fetch_status):
            return direct_drive_h <= self.max_direct_drive_hours_for_auto_drive
        return False

    def _build_direct_drive_leg_output(
        self,
        leg: Dict[str, Any],
        flight_leg_budget: float,
        selection_status: str,
    ) -> Dict[str, Any]:
        drive_h = self._safe_float(leg.get("direct_drive_hours"), 0.0)
        drive_km = self._safe_float(leg.get("direct_drive_distance_km"), 0.0)
        cost = self._direct_drive_cost_estimate()
        return {
            "leg_type": leg.get("leg_type"),
            "leg_index": leg.get("leg_index"),
            "from_city": leg.get("from_city"),
            "to_city": leg.get("to_city"),
            "departure_date": leg.get("departure_date"),
            "budget": flight_leg_budget if flight_leg_budget > 0 else None,
            "selection_status": selection_status,
            "transport_mode": "direct_drive",
            "selected_flights": [],
            "rental_car_cost_estimated": cost,
            "direct_drive_hours": round(drive_h, 2) if drive_h > 0 else None,
            "direct_drive_distance_km": round(drive_km, 2) if drive_km > 0 else None,
            "direct_drive_source": leg.get("direct_drive_source"),
            "rental_car": {
                "scenario": "direct_drive_selected_by_time_comparison",
                "cost_rule": "300-500 CNY by regional cost level",
                "estimated_drive_hours": round(drive_h, 2) if drive_h > 0 else None,
                "estimated_drive_distance_km": round(drive_km, 2) if drive_km > 0 else None,
                "comparison_note": (
                    "Direct driving was selected because city-to-city driving time was shorter "
                    "than candidate flight time including airport transfer and 2h airport buffer."
                ),
            },
        }

    def _format_selected_flight_output(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "airline": flight.get("airline"),
            "departure_airport": flight.get("departure_airport"),
            "departure_code": (
                flight.get("departure_code")
                or flight.get("departure_airport_code")
            ),
            "arrival_airport": flight.get("arrival_airport"),
            "arrival_code": (
                flight.get("arrival_code")
                or flight.get("arrival_airport_code")
            ),
            "departure_time": flight.get("departure_time"),
            "arrival_time": flight.get("arrival_time"),
            "duration_minutes": (
                flight.get("duration_minutes")
                if flight.get("duration_minutes") is not None
                else flight.get("total_duration")
            ),
            "stops": self._safe_int(flight.get("stops"), 0),
            "price_total": round(self._safe_float(flight.get("price_total"), 0.0), 2),
            "currency": flight.get("currency") or self.currency,
            "travel_class": flight.get("travel_class"),
            "provider": flight.get("provider") or "searchapi_google_flights",
            "why_selected": flight.get("why_selected"),
            "is_estimated_fallback": bool(flight.get("is_estimated_fallback")),
            "ground_to_departure_airport_hours": flight.get("ground_to_departure_airport_hours"),
            "ground_from_arrival_airport_hours": flight.get("ground_from_arrival_airport_hours"),
            "flight_buffer_hours": flight.get("flight_buffer_hours"),
            "flight_total_comparable_hours": flight.get("flight_total_comparable_hours"),
            "direct_drive_comparable_hours": flight.get("direct_drive_comparable_hours"),
            "airport_time_comparison": flight.get("airport_time_comparison"),
        }
    
    def _dedupe_flight_candidates(self, flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()

        for item in flights:
            key = (
                self._normalize_text(item.get("airline")),
                self._normalize_text(item.get("departure_time")),
                self._normalize_text(item.get("arrival_time")),
                round(self._safe_float(item.get("price_total"), 0.0), 2),
                self._safe_int(item.get("stops"), 0),
                self._normalize_text(item.get("departure_airport_code")),
                self._normalize_text(item.get("arrival_airport_code")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped 

    def _flight_date_window_status(self, date_text: Any) -> str:
        date_value = self._parse_iso_date(self._normalize_text(date_text))
        if not date_value:
            return ""
        days_ahead = (date_value.date() - get_today_date()).days
        if days_ahead < 0:
            return f"flight_date_in_past(days={abs(days_ahead)})"
        if days_ahead > FLIGHT_LOOKAHEAD_DAYS:
            return f"flight_date_outside_booking_window(days_ahead={days_ahead})"
        return ""
    
    
    def _search_leg_flights(self, leg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        if not leg:
            return [], "empty_leg"

        skip, _, skip_reason = self.should_skip_flight_search(leg)
        if skip or self._leg_is_short_distance(leg):
            return [], f"skipped_{skip_reason}"

        outbound_date = leg.get("departure_date")
        if not outbound_date:
            return [], "missing_route_info"

        date_window_status = self._flight_date_window_status(outbound_date)
        if date_window_status:
            return [], date_window_status

        candidate_from_airports = leg.get("candidate_from_airports") or []
        candidate_to_airports = leg.get("candidate_to_airports") or []

        # fallback to primary airport fields if candidate lists are empty
        if not candidate_from_airports and leg.get("from_airport"):
            candidate_from_airports = [leg.get("from_airport")]
        if not candidate_to_airports and leg.get("to_airport"):
            candidate_to_airports = [leg.get("to_airport")]

        candidate_from_airports = [
            self._normalize_text(x).upper()
            for x in candidate_from_airports
            if self._normalize_text(x)
        ]
        candidate_to_airports = [
            self._normalize_text(x).upper()
            for x in candidate_to_airports
            if self._normalize_text(x)
        ]

        if not candidate_from_airports or not candidate_to_airports:
            return [], "missing_route_info"

        strict_all: List[Dict[str, Any]] = []
        relaxed_all: List[Dict[str, Any]] = []
        tried_pairs: List[str] = []

        try:
            # Query factual flight candidates first, then rank the combined set.
            # Direct flights still receive a score bonus, but they should not
            # prevent cheaper or better-timed one-stop options from being seen.
            for departure_id in candidate_from_airports:
                for arrival_id in candidate_to_airports:
                    if not departure_id or not arrival_id or departure_id == arrival_id:
                        continue

                    tried_pairs.append(f"{departure_id}->{arrival_id}")

                    strict = self.search_flights_searchapi(
                        departure_id=departure_id,
                        arrival_id=arrival_id,
                        outbound_date=outbound_date,
                        flight_type="one_way",
                        adults=self.num_people,
                        max_results=8,
                        strict_direct=True,
                    )

                    if strict:
                        scored = [
                            self._score_flight(x, leg.get("leg_type", "leg"))
                            for x in strict
                        ]
                        for item in scored:
                            item["searched_airport_pair"] = f"{departure_id}->{arrival_id}"
                        strict_all.extend(scored)

            strict_all = self._dedupe_flight_candidates(strict_all)

            # 2) relaxed any-stop search across all airport pairs
            relaxed_pairs: List[str] = []
            for departure_id in candidate_from_airports:
                for arrival_id in candidate_to_airports:
                    if not departure_id or not arrival_id or departure_id == arrival_id:
                        continue

                    relaxed_pairs.append(f"{departure_id}->{arrival_id}")

                    relaxed = self.search_flights_searchapi(
                        departure_id=departure_id,
                        arrival_id=arrival_id,
                        outbound_date=outbound_date,
                        flight_type="one_way",
                        adults=self.num_people,
                        max_results=8,
                        strict_direct=False,
                    )

                    if relaxed:
                        scored = [
                            self._score_flight(x, leg.get("leg_type", "leg"))
                            for x in relaxed
                        ]
                        for item in scored:
                            item["searched_airport_pair"] = f"{departure_id}->{arrival_id}"
                        relaxed_all.extend(scored)

            combined = self._dedupe_flight_candidates(strict_all + relaxed_all)
            compared, comparison_status = self._filter_flights_by_drive_time_comparison(combined, leg)
            return (
                compared,
                f"live_flight_search("
                f"direct_pairs={','.join(tried_pairs)};"
                f"any_stop_pairs={','.join(relaxed_pairs)};"
                f"direct_candidates={len(strict_all)};"
                f"all_candidates={len(combined)}"
                f")|{comparison_status}",
            )

        except Exception:
            return [], "flight_search_failed"

    # =========================================================
    # main
    # =========================================================
    def generate_shortlist(self) -> Dict[str, Any]:
        if not self.effective_cities:
            return {
                "success": False,
                "stage": "agent3_real_selection",
                "error": "No destination found in agent2 output.",
                "debug": {
                    "received_keys": list(self.raw_input.keys()),
                    "city_plan_mode": self.city_plan_mode,
                    "budget_source": self.budget_source
                }
            }

        route_plan = self._build_route_plan()
        flight_legs = route_plan.get("flight_legs", [])

        selected_outbound_flights: List[Dict[str, Any]] = []
        selected_inbound_flights: List[Dict[str, Any]] = []
        selected_intercity_flights: List[Dict[str, Any]] = []
        selected_all_flight_legs: List[Dict[str, Any]] = []

        selected_pois: List[Dict[str, Any]] = []
        selected_hotels: List[Dict[str, Any]] = []
        selected_restaurants: List[Dict[str, Any]] = []

        city_selection_summary: List[Dict[str, Any]] = []
        warnings: List[str] = []

        hotel_provider_used: Dict[str, str] = {}
        all_poi_candidates_count: Dict[str, int] = {}
        all_hotel_candidates_count: Dict[str, int] = {}
        all_restaurant_candidates_count: Dict[str, int] = {}
        leg_selection_status_map: Dict[str, str] = {}
        leg_budget_map: Dict[str, float] = {}

        # -------------------------
        # Flights
        # -------------------------
        flight_leg_budget = 0.0
        if flight_legs:
            flight_leg_budget = round(self.transport_budget / len(flight_legs), 2) if self.transport_budget > 0 else 0.0

        for leg in flight_legs:
            leg_key = self._leg_key(leg)
            leg_budget_map[leg_key] = flight_leg_budget

            candidates, fetch_status = self._search_leg_flights(leg)
            selected_leg_flights, select_status = self.select_flights_with_budget(
                flights=candidates,
                budget=flight_leg_budget,
                max_items=self.flight_selected_limit,
                tolerance=self.flight_budget_tolerance,
                relaxed_tolerance=self.flight_relaxed_budget_tolerance,
            )

            leg_selection_status_map[leg_key] = f"{fetch_status}|{select_status}"

            formatted_selected_leg_flights = [
                self._format_selected_flight_output(flight)
                for flight in selected_leg_flights
            ]

            if not formatted_selected_leg_flights and self._should_select_direct_drive(leg, fetch_status):
                leg_output = self._build_direct_drive_leg_output(
                    leg=leg,
                    flight_leg_budget=flight_leg_budget,
                    selection_status=f"{fetch_status}|direct_drive_selected",
                )
                leg_selection_status_map[leg_key] = leg_output["selection_status"]
                formatted_selected_leg_flights = []
            else:
                leg_output = {
                    "leg_type": leg.get("leg_type"),
                    "leg_index": leg.get("leg_index"),
                    "from_city": leg.get("from_city"),
                    "to_city": leg.get("to_city"),
                    "departure_date": leg.get("departure_date"),
                    "budget": flight_leg_budget if flight_leg_budget > 0 else None,
                    "selection_status": f"{fetch_status}|{select_status}",
                    "transport_mode": None,
                    "selected_flights": formatted_selected_leg_flights,
                    "direct_drive_hours": leg.get("direct_drive_hours"),
                    "direct_drive_distance_km": leg.get("direct_drive_distance_km"),
                    "direct_drive_source": leg.get("direct_drive_source"),
                }

                if (
                    not formatted_selected_leg_flights
                    and self._normalize_text(leg.get("leg_type")) in {"outbound", "inbound"}
                ):
                    warnings.append(
                        f"No live flight candidate selected for {leg.get('leg_type')} leg "
                        f"{leg.get('from_city')}->{leg.get('to_city')} on {leg.get('departure_date')}; "
                        "kept as a manual-selection transport row instead of generating an estimated ticket."
                    )

            if leg.get("leg_type") == "outbound":
                selected_outbound_flights = formatted_selected_leg_flights
            elif leg.get("leg_type") == "inbound":
                selected_inbound_flights = formatted_selected_leg_flights
            else:
                selected_intercity_flights.append(leg_output)

            selected_all_flight_legs.append(leg_output)

        rental_intercity_legs = self._build_mid_distance_rental_legs(
            route_plan.get("intercity", []) or []
        )
        if rental_intercity_legs:
            selected_intercity_flights.extend(rental_intercity_legs)
            selected_all_flight_legs.extend(rental_intercity_legs)
            for leg in rental_intercity_legs:
                leg_budget_map[self._leg_key(leg)] = self._safe_float(
                    leg.get("rental_car_cost_estimated"),
                    0.0,
                )

        # -------------------------
        # City-by-city selection
        # -------------------------
        for city_item in self.city_budget_plan:
            city = self._normalize_text(city_item.get("city"))
            if not city:
                continue

            city_days = max(1, self._safe_int(city_item.get("days"), 1))
            city_dates = self.city_date_ranges.get(city, {})
            city_nights = max(1, self._safe_int(city_dates.get("nights"), max(1, city_days - 1)))

            hotel_budget = self._safe_float(city_item.get("hotel_budget"), 0.0)
            food_budget = self._safe_float(city_item.get("food_budget"), 0.0)
            attractions_budget = self._safe_float(city_item.get("attractions_budget"), 0.0)

            limits = self._candidate_limits_by_days(city_days)

            # Hotels
            hotel_candidates, hotel_provider = self.fetch_hotel_candidates(
                city=city,
                city_days=city_days,
                max_results=limits["hotel_candidates"]
            )
            hotel_provider_used[city] = hotel_provider

            estimated_hotels = [
                self._estimate_hotel_cost(h, city_days=city_days, city_nights=city_nights)
                for h in hotel_candidates
            ]
            estimated_hotels = self._attach_city_budget_reference(
                estimated_hotels,
                city_budget=hotel_budget,
                cost_key="estimated_total_cost"
            )
            filtered_hotels, hotel_filter_status = self._apply_hotel_hard_filters(estimated_hotels)
            fallback_hotel_applied = False
            if not filtered_hotels and estimated_hotels:
                filtered_hotels, hotel_filter_status = self._apply_hotel_hard_filters(
                    estimated_hotels,
                    fallback_mode=True,
                )
                fallback_hotel_applied = bool(filtered_hotels)

            ranked_hotels = self._rank_hotels(filtered_hotels)
            all_hotel_candidates_count[city] = len(ranked_hotels)

            selected_city_hotels, hotel_status = self.select_hotels_with_minimum(
                hotels=ranked_hotels,
                budget=hotel_budget,
                max_items=limits["hotel_selected"],
                minimum_items=2,
                tolerance=self.hotel_budget_tolerance
            )

            if not selected_city_hotels and estimated_hotels:
                fallback_ranked_hotels = self._rank_hotels(estimated_hotels)
                selected_city_hotels, hotel_status = self.select_hotels_with_minimum(
                    hotels=fallback_ranked_hotels,
                    budget=hotel_budget,
                    max_items=limits["hotel_selected"],
                    minimum_items=1,
                    tolerance=max(self.hotel_budget_tolerance, 1.15),
                )
                if selected_city_hotels:
                    fallback_hotel_applied = True
                    hotel_filter_status = "applied|guaranteed_output"
                    hotel_status = f"fallback_guaranteed|{hotel_status}"

            # Restaurants
            restaurant_candidates = self.fetch_restaurant_candidates(
                city=city,
                max_results=limits["restaurant_candidates"]
            )
            estimated_restaurants = [
                self._estimate_restaurant_cost(r, city_days=city_days)
                for r in restaurant_candidates
            ]
            estimated_restaurants = self._attach_city_budget_reference(
                estimated_restaurants,
                city_budget=food_budget,
                cost_key="estimated_total_cost"
            )
            filtered_restaurants, restaurant_filter_status = self._apply_restaurant_hard_filters(estimated_restaurants)

            if not filtered_restaurants and estimated_restaurants:
                user_pref_restaurants = [
                    r for r in estimated_restaurants
                    if self._restaurant_matches_user_preferences(r)
                ]
                if user_pref_restaurants:
                    filtered_restaurants = user_pref_restaurants
                    restaurant_filter_status = "fallback_user_preference_first"

            if not filtered_restaurants and estimated_restaurants:
                food_level_restaurants = [
                    r for r in estimated_restaurants
                    if self._restaurant_matches_food_level(r)
                ]
                if food_level_restaurants:
                    filtered_restaurants = food_level_restaurants
                    restaurant_filter_status = "fallback_food_level_only"

            if not filtered_restaurants and estimated_restaurants:
                filtered_restaurants = estimated_restaurants
                restaurant_filter_status = "fallback_all_candidates"

            ranked_restaurants = self._rank_restaurants(filtered_restaurants)
            all_restaurant_candidates_count[city] = len(ranked_restaurants)

            selected_city_restaurants, restaurant_status = self.select_with_budget(
                items=ranked_restaurants,
                budget=food_budget,
                cost_key="estimated_total_cost",
                max_items=limits["restaurant_selected"],
                tolerance=self.restaurant_budget_tolerance
            )

            # POIs
            poi_candidates = self.fetch_poi_candidates(
                city=city,
                max_results=limits["poi_candidates"]
            )
            estimated_pois = [
                self._estimate_poi_cost(p, city_days=city_days)
                for p in poi_candidates
            ]
            estimated_pois = self._attach_city_budget_reference(
                estimated_pois,
                city_budget=attractions_budget,
                cost_key="estimated_total_cost"
            )
            filtered_pois, poi_filter_status = self._apply_poi_hard_filters(estimated_pois)
            ranked_pois = self._rank_pois(filtered_pois)
            all_poi_candidates_count[city] = len(ranked_pois)

            selected_city_pois, poi_status, poi_warnings = self.select_pois_with_must_visit(
                items=ranked_pois,
                budget=attractions_budget,
                cost_key="estimated_total_cost",
                max_items=limits["poi_selected"],
                tolerance=self.poi_budget_tolerance
            )
            warnings.extend(poi_warnings)

            # Output merge
            for hotel in selected_city_hotels:
                selected_hotels.append({
                    "city": hotel.get("city"),
                    "name": hotel.get("name"),
                    "place_id": hotel.get("place_id"),
                    "area": hotel.get("area"),
                    "rating": hotel.get("rating"),
                    "stars": hotel.get("stars"),
                    "lat": hotel.get("lat"),
                    "lng": hotel.get("lng"),
                    "estimated_nightly_per_room": hotel.get("estimated_nightly_per_room"),
                    "estimated_total_cost": hotel.get("estimated_total_cost"),
                    "realtime_price_available": hotel.get("realtime_price_available"),
                    "realtime_nightly_per_room": hotel.get("realtime_nightly_per_room"),
                    "why_selected": hotel.get("why_selected"),
                })

            for poi in selected_city_pois:
                selected_pois.append({
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
                })

            for restaurant in selected_city_restaurants:
                selected_restaurants.append({
                    "city": restaurant.get("city"),
                    "name": restaurant.get("name"),
                    "place_id": restaurant.get("place_id"),
                    "area": restaurant.get("area"),
                    "rating": restaurant.get("rating"),
                    "lat": restaurant.get("lat"),
                    "lng": restaurant.get("lng"),
                    "types": restaurant.get("types", []),
                    "estimated_meal_per_person": restaurant.get("estimated_meal_per_person"),
                    "recommended_visits": restaurant.get("recommended_visits"),
                    "estimated_total_cost": restaurant.get("estimated_total_cost"),
                    "matched_query": restaurant.get("matched_query"),
                    "local_food_match_score": restaurant.get("local_food_match_score"),
                    "why_selected": restaurant.get("why_selected"),
                })

            city_selection_summary.append({
                "city": city,
                "days": city_days,
                "check_in": city_dates.get("check_in"),
                "check_out": city_dates.get("check_out"),
                "hotel_budget": hotel_budget,
                "food_budget": food_budget,
                "attractions_budget": attractions_budget,
                "selected_hotels": len(selected_city_hotels),
                "selected_restaurants": len(selected_city_restaurants),
                "selected_pois": len(selected_city_pois),
                "hotel_selection_status": f"{hotel_filter_status}|{hotel_status}",
                "restaurant_selection_status": f"{restaurant_filter_status}|{restaurant_status}",
                "poi_selection_status": f"{poi_filter_status}|{poi_status}",
                "hotel_provider": hotel_provider
            })

            if not selected_city_hotels:
                warnings.append(f"No hotel candidates selected for {city}.")
            elif any(h.get("fallback_mode") for h in selected_city_hotels) or "fallback" in str(hotel_filter_status) or "fallback" in str(hotel_status):
                warnings.append(
                    f"Hotel fallback mode applied for {city}: no candidates fully satisfied all hotel preferences, so relaxed selection was used to guarantee output."
                )

            if not selected_city_restaurants:
                warnings.append(f"No restaurant candidates selected for {city}.")

            if not selected_city_pois:
                warnings.append(f"No POI candidates selected for {city}.")

        # -------------------------
        # Budget summary
        # -------------------------
        transport_budget_input = self.transport_budget
        hotel_budget_input = round(sum(self._safe_float(x.get("hotel_budget"), 0.0) for x in self.city_budget_plan), 2)
        food_budget_input = round(sum(self._safe_float(x.get("food_budget"), 0.0) for x in self.city_budget_plan), 2)
        attractions_budget_input = round(sum(self._safe_float(x.get("attractions_budget"), 0.0) for x in self.city_budget_plan), 2)

        selected_outbound_total = self._get_sum_selected_total(selected_outbound_flights, "price_total")
        selected_intercity_total = round(
            sum(
                self._get_sum_selected_total(leg.get("selected_flights", []), "price_total")
                for leg in selected_intercity_flights
            ),
            2
        )
        selected_mid_distance_car_rental_total = round(
            sum(
                self._safe_float(leg.get("rental_car_cost_estimated"), 0.0)
                for leg in selected_intercity_flights
            ),
            2
        )
        selected_inbound_total = self._get_sum_selected_total(selected_inbound_flights, "price_total")
        total_flight_spend = round(selected_outbound_total + selected_intercity_total + selected_inbound_total, 2)
        total_transport_spend = round(total_flight_spend + selected_mid_distance_car_rental_total, 2)

        selected_hotel_total = self._get_hotel_reference_total_by_city(selected_hotels)
        selected_restaurant_total = self._get_sum_selected_total(selected_restaurants, "estimated_total_cost")
        selected_poi_total = self._get_sum_selected_total(selected_pois, "estimated_total_cost")

        budget_fit_status = {
            "flight_total": self._budget_fit_label(
                total_transport_spend,
                transport_budget_input,
                tolerance=self.flight_relaxed_budget_tolerance
            ),
            "hotel": "candidate_based_not_summed",
            "food": self._budget_fit_label(selected_restaurant_total, food_budget_input),
            "attractions": self._budget_fit_label(selected_poi_total, attractions_budget_input)
        }

        # -------------------------
        # Budget warnings
        # -------------------------
        agent2_summary = self.raw_input.get("agent2_summary") or {}
        agent2_budget_plan = (
            agent2_summary.get("budget_plan")
            or self.raw_input.get("budget_plan")
            or self.budget_result.get("budget_plan")
            or {}
        )

        agent2_budget_status = str(
            agent2_summary.get("budget_status")
            or self.raw_input.get("budget_status")
            or ""
        ).lower()

        agent2_buffer = self._safe_float(agent2_budget_plan.get("buffer"), 0.0)

        agent2_total_allocated = round(
            self._safe_float(agent2_budget_plan.get("transport"), 0.0)
            + self._safe_float(agent2_budget_plan.get("hotel"), 0.0)
            + self._safe_float(agent2_budget_plan.get("food"), 0.0)
            + self._safe_float(agent2_budget_plan.get("attractions"), 0.0)
            + self._safe_float(agent2_budget_plan.get("buffer"), 0.0),
            2
        )

        overall_budget_warning_triggered = False

        # Rule 1:
        # If Agent2 already says the allocation itself is infeasible, keep the overall warning.
        if (
            agent2_budget_status == "over_budget"
            or agent2_buffer < 0
            or (self.total_budget > 0 and agent2_total_allocated > self.total_budget + 1)
        ):
            warnings.append(
                "Overall budget warning: Agent2 indicates the trip budget is already infeasible or over-allocated before Agent3 selection, so downstream recommendations may exceed the user's budget."
            )
            overall_budget_warning_triggered = True

        # Rule 2:
        # If Agent2 allocation is OK, only check the FINAL selected total.
        if not overall_budget_warning_triggered:
            overall_selected_total = round(
                total_flight_spend
                + selected_mid_distance_car_rental_total
                + selected_hotel_total
                + selected_restaurant_total
                + selected_poi_total,
                2
            )

            if self.total_budget > 0 and overall_selected_total > self.total_budget:
                warnings.append(
                    f"Overall selected budget warning: total selected spend is {overall_selected_total}, "
                    f"which exceeds the user's total budget {self.total_budget}."
                )

        # De-duplicate warnings while preserving order
        warnings = list(dict.fromkeys(warnings))

        return {
            "success": True,
            "stage": "agent3_real_selection",
            "trip_summary": {
                "origin": self.origin,
                "destination": self.destination,
                "num_people": self.num_people,
                "duration_days": self.duration_days,
                "travel_dates": {
                    "start": self._resolve_departure_date(),
                    "end": self._resolve_return_date(),
                    "text": (self.parsed_request.get("travel_dates", {}) or {}).get("text")
                },
                "budget": {
                    "total": self.total_budget,
                    "currency": self.currency
                }
            },
            "origin": self.origin,
            "destination": self.destination,
            "effective_cities": self.effective_cities,
            "city_plan_mode": self.city_plan_mode,
            "city_budget_plan_used": self.city_budget_plan,
            "city_date_ranges": self.city_date_ranges,
            "route_plan": route_plan,
            "mid_distance_car_rental": self.mid_distance_car_rental if self._has_mid_distance_car_rental_plan() else {},
            "selected_outbound_flights": selected_outbound_flights,
            "selected_inbound_flights": selected_inbound_flights,
            "selected_intercity_flights": selected_intercity_flights,
            "selected_all_flight_legs": selected_all_flight_legs,
            "selected_pois": selected_pois,
            "selected_hotels": selected_hotels,
            "selected_restaurants": selected_restaurants,
            "city_selection_summary": city_selection_summary,
            "warnings": warnings,
            "budget_fit_status": budget_fit_status,
            "budget_check": {
                "budget_source": self.budget_source,
                "total_budget_input": self.total_budget,
                "transport_budget_input": transport_budget_input,
                "flight_leg_budget_map": leg_budget_map,
                "hotel_budget_input": hotel_budget_input,
                "attractions_budget_input": attractions_budget_input,
                "food_budget_input": food_budget_input,
                "selected_outbound_flight_spend": selected_outbound_total,
                "selected_intercity_flight_spend": selected_intercity_total,
                "selected_mid_distance_car_rental_spend": selected_mid_distance_car_rental_total,
                "selected_inbound_flight_spend": selected_inbound_total,
                "selected_total_flight_spend": total_flight_spend,
                "selected_total_transport_spend": total_transport_spend,
                "selected_hotel_total_reference": selected_hotel_total,
                "selected_restaurant_total": selected_restaurant_total,
                "selected_poi_total": selected_poi_total
            },
            "selection_debug": {
                "hotel_provider_used": hotel_provider_used,
                "all_poi_candidates_count": all_poi_candidates_count,
                "all_hotel_candidates_count": all_hotel_candidates_count,
                "all_restaurant_candidates_count": all_restaurant_candidates_count,
                "hotel_level": self.hotel_level,
                "hotel_preference_used": self.hotel_pref,
                "food_level": self.food_level,
                "food_preference_used": self.food_pref,
                "attraction_style_used": self.attraction_style,
                "planning_signals_used": {
                    "flight_priority": self.flight_priority,
                    "hotel_level": self.hotel_priority,
                    "food_priority": self.food_priority,
                    "attraction_intensity": self.attraction_priority,
                },
                "prefers_local_food": self.prefers_local_food,
                "local_food_llm_enabled": self.local_food_llm_enabled,
                "flight_type": self.flight_type,
                "flight_leg_selection_status_map": leg_selection_status_map,
                "selection_policy": self.selection_policy
            }
        }


def run_agent3(agent2_output: Dict[str, Any]) -> Dict[str, Any]:
    selector = Agent3Selection(agent2_output)
    return selector.generate_shortlist()
