from __future__ import annotations

import os
import re
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from destination import run_city_planning_agent
from travel_time import get_today_date

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "country_default_city_plan.json"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)


CURRENCY_ALIASES = {
    "cny": "CNY",
    "rmb": "CNY",
    "人民币": "CNY",
    "元": "CNY",
    "yuan": "CNY",
    "¥": "CNY",
    "usd": "USD",
    "us$": "USD",
    "$": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "美元": "USD",
    "eur": "EUR",
    "€": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "欧元": "EUR",
    "gbp": "GBP",
    "£": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "英镑": "GBP",
    "jpy": "JPY",
    "yen": "JPY",
    "日元": "JPY",
    "hkd": "HKD",
    "港币": "HKD",
    "sgd": "SGD",
    "新币": "SGD",
}

# The budgeting engine's city profiles and downstream cost estimates are CNY-based.
# Keep rates explicit so non-CNY input is not silently mislabeled as CNY.
CURRENCY_TO_CNY = {
    "CNY": 1.0,
    "USD": 7.1,
    "EUR": 7.7,
    "GBP": 9.0,
    "JPY": 0.048,
    "HKD": 0.91,
    "SGD": 5.3,
}

PLACE_STOPWORDS = {
    "for",
    "with",
    "within",
    "under",
    "around",
    "about",
    "starting",
    "leaving",
    "departing",
    "tomorrow",
    "today",
    "next",
    "this",
    "people",
    "person",
    "traveler",
    "travellers",
    "travelers",
    "budget",
    "trip",
    "days",
    "day",
}


def get_llm_backend_status() -> dict:
    """
    Return the current LLM backend status.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("TRAVEL_LLM_MODEL", "qwen-plus")
    base_url = os.getenv(
        "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    return {
        "provider": "dashscope-compatible",
        "configured": bool(api_key),
        "model": model,
        "base_url": base_url,
        "api_key_present": bool(api_key),
    }


def _default_result() -> Dict[str, Any]:
    return {
        "origin": None,
        "destination": [],
        "country": None,
        "num_people": 1,
        "travel_dates": {
            "start": None,
            "end": None,
            "text": None
        },
        "duration_days": None,
        "budget": {
            "total": None,
            "currency": "CNY"
        },
        "preferences": {
            # Backward-compatible legacy scalar fields
            "flight": None,                 # direct / cheap / comfort
            "hotel": None,                  # budget / standard / luxury / central / clean
            "food": None,                   # local / fine_dining / street_food / mixed
            "attraction_style": None,       # historical / nature / shopping / entertainment / art / mixed

            # New richer fields
            "hotel_level": None,            # budget / standard / premium / luxury
            "hotel_features": [],           # clean / central
            "food_preferences": [],         # local / fine_dining / street_food / mixed
            "attraction_styles": [],        # historical / nature / shopping / entertainment / art / mixed
            "must_visit_attractions": []    # explicit named attractions, e.g. Great Wall / Louvre Museum
        },
        "planning_signals": {
            # Keep old keys for downstream compatibility
            "flight_priority": None,
            "hotel_level": None,            # low / medium / high
            "food_priority": None,
            "attraction_intensity": None,

            # Optional alias for readability
            "hotel_priority": None
        },
        "city_plan": {
            "mode": "unknown",
            "cities": []
        },
        "missing_fields": {
            "critical": [],
            "optional": []
        },
        "is_ready_for_planning": False,
        "parse_status": "failed",
        "meta": {
            "llm_called": False,
            "llm_call_success": False,
            "raw_response_preview": None,
            "error": None
        }
    }


def _build_prompt(user_request: str) -> str:
    """
    Build a strict prompt for travel requirement extraction.
    Agent1 should return controlled vocabulary directly for downstream agents.
    This upgraded version supports richer multi-label preference extraction while
    still preserving backward-compatible legacy fields.
    """
    today = get_today_date()
    tomorrow = today + timedelta(days=1)
    return f"""
You are a travel requirement extraction agent.

Your task is to extract structured travel requirements from the user's request
and return ONLY one valid JSON object.

Reference date for all relative date expressions:
- today = {today.isoformat()}
- tomorrow = {tomorrow.isoformat()}

Strict rules:
1. Return JSON only.
2. Do not use markdown.
3. Do not wrap the output in triple backticks.
4. If a field is missing, use null for scalar fields and [] for list fields.
5. "destination" must always be a JSON array.
6. If only one destination is found, still return it as an array.
7. If the user mentions a country but not a city, put the country in "country" and keep "destination" as [].
8. "budget.total" must be a number if possible.
9. Preserve the user's budget currency. For example, "2000dollar", "2000 dollars", "$2000", and "2000 USD" must use "USD", not "CNY".
10. "num_people" must be an integer if possible.
11. "duration_days" must be an integer if possible.
12. Only fill "travel_dates.start" and "travel_dates.end" when the date is explicit enough.
13. If the user gives month/day expressions without a year, resolve them to the current calendar year by default using today = {today.isoformat()}.
14. If a missing year must be resolved, do not roll the date into next year just because the month/day has already passed.
15. If exact dates still cannot be safely determined, set "start" and "end" to null and keep the original phrase in "travel_dates.text".
16. Do not invent any information not explicitly mentioned by the user, except for resolving a missing year in travel dates to the current calendar year.
17. Use controlled vocabulary only for preference categories. Do not output free-form preference text for those category fields.
18. Preserve both:
   - legacy scalar fields for backward compatibility
   - richer new fields for more detailed downstream planning
19. If multiple styles/preferences are mentioned, keep all of them in the new list fields.
20. If the user explicitly names a specific attraction, landmark, museum, park, temple, tower, palace, castle, district, street, site, or destination spot they want to visit, put it into preferences.must_visit_attractions.
21. Do not convert a specific named attraction into only a style label. Keep both if applicable.
22. preferences.must_visit_attractions must be a JSON array of strings.
23. Only include attractions explicitly mentioned by the user. Do not guess.
24. For relative date expressions such as "today", "tomorrow", "starting tomorrow", and "leaving tomorrow", you must use the reference date above, not your own internal date.

Allowed values:

preferences.flight:
- "direct"
- "cheap"
- "comfort"
- null

preferences.hotel (legacy scalar):
- "budget"
- "standard"
- "luxury"
- "central"
- "clean"
- null

preferences.food (legacy scalar):
- "local"
- "fine_dining"
- "street_food"
- "mixed"
- null

preferences.attraction_style (legacy scalar):
- "historical"
- "nature"
- "shopping"
- "entertainment"
- "art"
- "mixed"
- null

preferences.hotel_level (new richer scalar):
- "budget"
- "standard"
- "premium"
- "luxury"
- null

preferences.hotel_features (new list):
allowed items:
- "clean"
- "central"

preferences.food_preferences (new list):
allowed items:
- "local"
- "fine_dining"
- "street_food"
- "mixed"

preferences.attraction_styles (new list):
allowed items:
- "historical"
- "nature"
- "shopping"
- "entertainment"
- "art"
- "mixed"

preferences.must_visit_attractions (new list):
- a list of explicitly mentioned attraction names from the user request
- examples:
  - "Great Wall"
  - "Tokyo Disneyland"
  - "Louvre Museum"
  - "Mount Fuji"
- if none are explicitly mentioned, use []

planning_signals.flight_priority:
- "low"
- "medium"
- "high"
- null

planning_signals.hotel_level:
- "low"
- "medium"
- "high"
- null

planning_signals.food_priority:
- "low"
- "medium"
- "high"
- null

planning_signals.attraction_intensity:
- "low"
- "medium"
- "high"
- null

planning_signals.hotel_priority:
- "low"
- "medium"
- "high"
- null

Interpretation examples:

FLIGHTS
- "prefer direct flights" -> preferences.flight = "direct", planning_signals.flight_priority = "high"
- "direct flights would be nice but not necessary" -> preferences.flight = "direct", planning_signals.flight_priority = "medium"
- "cheap flights" -> preferences.flight = "cheap"
- "comfortable flight" / "better flight experience" -> preferences.flight = "comfort"

HOTELS
- "budget hotel" -> preferences.hotel = "budget", preferences.hotel_level = "budget", planning_signals.hotel_level = "low"
- "standard hotel" -> preferences.hotel = "standard", preferences.hotel_level = "standard", planning_signals.hotel_level = "medium"
- "four star hotel" / "4 star hotel" -> preferences.hotel = "luxury", preferences.hotel_level = "premium", planning_signals.hotel_level = "high"
- "five star hotel" / "5 star hotel" -> preferences.hotel = "luxury", preferences.hotel_level = "luxury", planning_signals.hotel_level = "high"
- "good hotel" / "better hotel" -> preferences.hotel_level = "premium" or "luxury" depending on wording strength
- "clean hotel" -> add "clean" to preferences.hotel_features; legacy preferences.hotel may be "clean"
- "centrally located hotel" / "city center hotel" -> add "central" to preferences.hotel_features; legacy preferences.hotel may be "central"
- "clean and centrally located hotel" -> preferences.hotel_features = ["clean", "central"]

FOOD
- "love local food" -> preferences.food = "local", preferences.food_preferences = ["local"], planning_signals.food_priority = "high"
- "street food" -> preferences.food = "street_food", preferences.food_preferences = ["street_food"]
- "fine dining" -> preferences.food = "fine_dining", preferences.food_preferences = ["fine_dining"]
- "local food and one fine dining meal" -> preferences.food_preferences = ["local", "fine_dining"], and choose the stronger primary one for legacy preferences.food

ATTRACTIONS
- "like history" / "museums" / "old towns" / "cultural heritage" -> include "historical"
- "art galleries" / "exhibitions" -> include "art"
- "nature" / "mountains" / "parks" -> include "nature"
- "shopping" / "mall" / "market" -> include "shopping"
- "nightlife" / "theme parks" / "shows" -> include "entertainment"
- "many attractions" / "packed itinerary" / "lots of attractions" -> planning_signals.attraction_intensity = "high"
- "relaxed trip" / "not too many attractions" -> planning_signals.attraction_intensity = "low"
- if the user wants attractions but does not specify a type, use legacy preferences.attraction_style = "mixed" and preferences.attraction_styles = ["mixed"]
- "I want to visit the Great Wall" -> preferences.must_visit_attractions = ["Great Wall"]
- "I must go to Tokyo Disneyland" -> preferences.must_visit_attractions = ["Tokyo Disneyland"]
- "I want historical attractions and also the Great Wall" -> preferences.attraction_style = "historical", preferences.attraction_styles = ["historical"], preferences.must_visit_attractions = ["Great Wall"]

Important compatibility rule:
- Always fill legacy scalar fields too.
- When multiple hotel features exist, keep them in "hotel_features", but legacy "hotel" should use the most operationally useful single value:
  priority for legacy hotel fallback:
  budget > luxury > standard > central > clean
- When multiple food preferences exist, choose one primary value for legacy "food" using:
  local > fine_dining > street_food > mixed
- When multiple attraction styles exist, choose one primary value for legacy "attraction_style" using:
  historical > nature > shopping > entertainment > art > mixed

Output schema:
{{
  "origin": null,
  "destination": [],
  "country": null,
  "num_people": null,
  "travel_dates": {{
    "start": null,
    "end": null,
    "text": null
  }},
  "duration_days": null,
  "budget": {{
    "total": null,
    "currency": "CNY"
  }},
  "preferences": {{
    "flight": null,
    "hotel": null,
    "food": null,
    "attraction_style": null,
    "hotel_level": null,
    "hotel_features": [],
    "food_preferences": [],
    "attraction_styles": [],
    "must_visit_attractions": []
  }},
  "planning_signals": {{
    "flight_priority": null,
    "hotel_level": null,
    "food_priority": null,
    "attraction_intensity": null,
    "hotel_priority": null
  }}
}}

User request:
{user_request}
""".strip()


def _prepare_user_request(user_request: str) -> str:
    """
    Collapse the frontend's multi-turn merge prompt into the actual user content.
    This keeps Agent1 focused on the trip details instead of the wrapper instructions.
    """
    text = (user_request or "").strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered_messages = []

    for line in lines:
        match = re.match(r"^\d+\.\s+(.*)$", line)
        if match:
            numbered_messages.append(match.group(1).strip())

    if numbered_messages:
        return " ".join(message for message in numbered_messages if message).strip()

    return text


def _call_llm(prompt: str) -> str:
    """
    Call the DashScope compatible chat completion API.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("TRAVEL_LLM_MODEL", "qwen-plus")
    base_url = os.getenv(
        "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY in .env")

    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise information extraction assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM API request failed: {response.status_code} - {response.text}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected LLM response format: {json.dumps(data, ensure_ascii=False)}"
        )


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from raw LLM text.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError("No JSON object found in LLM response.")

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON from LLM response: {exc}") from exc


def _normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None

    return str(value).strip() or None


def _normalize_currency(value: Any) -> str:
    text = _normalize_string(value)
    if not text:
        return "CNY"
    compact = text.strip().lower()
    return CURRENCY_ALIASES.get(compact, compact.upper())


def _currency_token_pattern() -> str:
    tokens = sorted(CURRENCY_ALIASES.keys(), key=len, reverse=True)
    return "|".join(re.escape(token) for token in tokens)


def _infer_budget_currency_from_text(user_request: str, budget_total: Optional[int]) -> Optional[str]:
    if budget_total is None:
        return None

    text = user_request or ""
    number = re.escape(str(budget_total))
    currency = _currency_token_pattern()
    patterns = [
        rf"(?i)(?:{currency})\s*{number}(?:\.0+)?\b",
        rf"(?i)\b{number}(?:\.0+)?\s*(?:{currency})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            matched_text = match.group(0).lower()
            for alias in sorted(CURRENCY_ALIASES.keys(), key=len, reverse=True):
                if alias.lower() in matched_text:
                    return CURRENCY_ALIASES[alias]

    budget_nearby = re.search(
        rf"(?i)(?:budget|预算|费用|cost|spend|total)\D{{0,20}}(\d+(?:\.\d+)?)\s*({currency})",
        text,
    )
    if budget_nearby:
        return CURRENCY_ALIASES.get(budget_nearby.group(2).lower())

    return None


def _normalize_budget_for_planning(
    budget: Dict[str, Any],
    user_request: str,
) -> Dict[str, Any]:
    total = _extract_number(budget.get("total"))
    llm_currency = _normalize_currency(budget.get("currency"))
    text_currency = _infer_budget_currency_from_text(user_request, total)
    source_currency = text_currency or llm_currency or "CNY"

    normalized = {
        "total": total,
        "currency": source_currency,
    }
    if total is None:
        return normalized

    rate = CURRENCY_TO_CNY.get(source_currency)
    if source_currency != "CNY" and rate:
        converted_total = int(round(total * rate))
        normalized.update({
            "total": converted_total,
            "currency": "CNY",
            "source_total": total,
            "source_currency": source_currency,
            "conversion_rate_to_cny": rate,
            "conversion_note": (
                f"Converted from {total} {source_currency} to approximately "
                f"{converted_total} CNY for CNY-based planning estimates."
            ),
        })
    return normalized


def _extract_place_fragment(fragment: Optional[str]) -> Optional[str]:
    if not fragment:
        return None

    cleaned = re.sub(r"^[\s,.;:()\-]+|[\s,.;:()\-]+$", "", fragment.strip())
    cleaned = re.split(
        r"(?i)\b(?:for|with|within|under|around|about|starting|leaving|departing|tomorrow|today|budget|days?|people|travellers?|travelers?)\b",
        cleaned,
        maxsplit=1,
    )[0].strip(" ,.;:-")

    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in PLACE_STOPWORDS:
        return None

    return " ".join(part for part in cleaned.split() if part).strip() or None


def _extract_origin_destination_from_text(user_request: str) -> Dict[str, Any]:
    text = _prepare_user_request(user_request)
    if not text:
        return {"origin": None, "destination": []}

    origin = None
    destination: List[str] = []

    patterns = [
        re.compile(
            r"(?i)\b(?:travel|trip|go|going|plan|planning|visit|vacation|holiday)\s+from\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)(?:[,.]|$)"
        ),
        re.compile(
            r"(?i)\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)(?:[,.]|$)"
        ),
        re.compile(
            r"(?i)\b(?:plan|planning|want|wanna|need|help me plan)\b.*?\b(?P<destination>[a-z][a-z\s.'-]*?)\s+trip\s+from\s+(?P<origin>.+?)(?:[,.]|$)"
        ),
        re.compile(
            r"(?i)\btrip\s+to\s+(?P<destination>.+?)\s+from\s+(?P<origin>.+?)(?:[,.]|$)"
        ),
    ]

    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue

        extracted_origin = _extract_place_fragment(match.groupdict().get("origin"))
        extracted_destination = _extract_place_fragment(match.groupdict().get("destination"))

        if extracted_origin and not origin:
            origin = extracted_origin
        if extracted_destination:
            destination = [extracted_destination]
        if origin or destination:
            break

    return {
        "origin": origin,
        "destination": destination,
    }


def _merge_missing_fields_from_text(result: Dict[str, Any], user_request: str) -> Dict[str, Any]:
    """
    Second-pass fallback: if Agent1 misses core route/date details, recover them from raw text.
    """
    merged = json.loads(json.dumps(result))
    prepared_request = _prepare_user_request(user_request)
    route_fallback = _extract_origin_destination_from_text(prepared_request)

    if not merged.get("origin") and route_fallback.get("origin"):
        merged["origin"] = route_fallback["origin"]

    if not merged.get("destination") and route_fallback.get("destination"):
        merged["destination"] = route_fallback["destination"]

    travel_dates = merged.get("travel_dates") or {}
    if not travel_dates.get("start"):
        relative_start = _extract_relative_start_date_from_text(prepared_request)
        if relative_start:
            travel_dates["start"] = relative_start
            merged["travel_dates"] = travel_dates

    merged["missing_fields"] = _detect_missing_fields(merged)
    merged["is_ready_for_planning"] = len(merged["missing_fields"]["critical"]) == 0
    merged["parse_status"] = "success" if merged["is_ready_for_planning"] else "partial"
    return merged


def _extract_number(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        match = re.search(r"\d+(\.\d+)?", cleaned)
        if match:
            return int(float(match.group(0)))

    return None


def _normalize_destination(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    cleaned_destinations = []
    seen = set()

    for item in value:
        normalized = _normalize_string(item)
        if normalized and normalized.lower() not in seen:
            cleaned_destinations.append(normalized)
            seen.add(normalized.lower())

    return cleaned_destinations


def _normalize_list_of_strings(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    results = []
    seen = set()

    for item in value:
        normalized = _normalize_string(item)
        if normalized:
            lowered = normalized.lower()
            if lowered not in seen:
                results.append(lowered)
                seen.add(lowered)

    return results


def _pick_primary_value(values: List[str], priority_order: List[str]) -> Optional[str]:
    if not values:
        return None

    value_set = set(values)
    for item in priority_order:
        if item in value_set:
            return item

    return values[0] if values else None


def _map_hotel_level_to_legacy_hotel(hotel_level: Optional[str], hotel_features: List[str]) -> Optional[str]:
    if hotel_level == "budget":
        return "budget"
    if hotel_level in {"premium", "luxury"}:
        return "luxury"
    if hotel_level == "standard":
        return "standard"

    if "central" in hotel_features:
        return "central"
    if "clean" in hotel_features:
        return "clean"

    return None


def _normalize_preferences(value: Any) -> Dict[str, Any]:
    default_preferences = {
        "flight": None,
        "hotel": None,
        "food": None,
        "attraction_style": None,
        "hotel_level": None,
        "hotel_features": [],
        "food_preferences": [],
        "attraction_styles": [],
        "must_visit_attractions": []
    }

    allowed_flight = {"direct", "cheap", "comfort"}
    allowed_legacy_hotel = {"budget", "standard", "luxury", "central", "clean"}
    allowed_food = {"local", "fine_dining", "street_food", "mixed"}
    allowed_attraction = {"historical", "nature", "shopping", "entertainment", "art", "mixed"}
    allowed_hotel_level = {"budget", "standard", "premium", "luxury"}
    allowed_hotel_features = {"clean", "central"}

    if not isinstance(value, dict):
        return default_preferences

    raw_flight = _normalize_string(value.get("flight"))
    raw_flight = raw_flight.lower() if raw_flight else None
    default_preferences["flight"] = raw_flight if raw_flight in allowed_flight else None

    raw_hotel_level = _normalize_string(value.get("hotel_level"))
    raw_hotel_level = raw_hotel_level.lower() if raw_hotel_level else None
    default_preferences["hotel_level"] = raw_hotel_level if raw_hotel_level in allowed_hotel_level else None

    raw_hotel_features = _normalize_list_of_strings(value.get("hotel_features"))
    default_preferences["hotel_features"] = [x for x in raw_hotel_features if x in allowed_hotel_features]

    raw_food_preferences = _normalize_list_of_strings(value.get("food_preferences"))
    default_preferences["food_preferences"] = [x for x in raw_food_preferences if x in allowed_food]

    raw_attraction_styles = _normalize_list_of_strings(value.get("attraction_styles"))
    default_preferences["attraction_styles"] = [x for x in raw_attraction_styles if x in allowed_attraction]

    # Keep explicit attraction names in original casing where possible
    raw_must_visit = _normalize_destination(value.get("must_visit_attractions"))
    default_preferences["must_visit_attractions"] = raw_must_visit

    raw_hotel_legacy = _normalize_string(value.get("hotel"))
    raw_hotel_legacy = raw_hotel_legacy.lower() if raw_hotel_legacy else None
    if raw_hotel_legacy in allowed_legacy_hotel:
        default_preferences["hotel"] = raw_hotel_legacy
    else:
        default_preferences["hotel"] = _map_hotel_level_to_legacy_hotel(
            default_preferences["hotel_level"],
            default_preferences["hotel_features"]
        )

    raw_food_legacy = _normalize_string(value.get("food"))
    raw_food_legacy = raw_food_legacy.lower() if raw_food_legacy else None
    if raw_food_legacy in allowed_food:
        default_preferences["food"] = raw_food_legacy
    else:
        default_preferences["food"] = _pick_primary_value(
            default_preferences["food_preferences"],
            ["local", "fine_dining", "street_food", "mixed"]
        )

    raw_attraction_legacy = _normalize_string(value.get("attraction_style"))
    raw_attraction_legacy = raw_attraction_legacy.lower() if raw_attraction_legacy else None
    if raw_attraction_legacy in allowed_attraction:
        default_preferences["attraction_style"] = raw_attraction_legacy
    else:
        default_preferences["attraction_style"] = _pick_primary_value(
            default_preferences["attraction_styles"],
            ["historical", "nature", "shopping", "entertainment", "art", "mixed"]
        )

    if not default_preferences["food_preferences"] and default_preferences["food"]:
        default_preferences["food_preferences"] = [default_preferences["food"]]

    if not default_preferences["attraction_styles"] and default_preferences["attraction_style"]:
        default_preferences["attraction_styles"] = [default_preferences["attraction_style"]]

    if default_preferences["hotel"] == "central" and "central" not in default_preferences["hotel_features"]:
        default_preferences["hotel_features"].append("central")
    if default_preferences["hotel"] == "clean" and "clean" not in default_preferences["hotel_features"]:
        default_preferences["hotel_features"].append("clean")

    return default_preferences


def _normalize_planning_signals(value: Any) -> Dict[str, Optional[str]]:
    default_signals = {
        "flight_priority": None,
        "hotel_level": None,
        "food_priority": None,
        "attraction_intensity": None,
        "hotel_priority": None
    }

    allowed = {"low", "medium", "high"}

    if not isinstance(value, dict):
        return default_signals

    for key in default_signals:
        raw = _normalize_string(value.get(key))
        raw = raw.lower() if raw else None
        default_signals[key] = raw if raw in allowed else None

    if default_signals["hotel_priority"] and not default_signals["hotel_level"]:
        default_signals["hotel_level"] = default_signals["hotel_priority"]
    elif default_signals["hotel_level"] and not default_signals["hotel_priority"]:
        default_signals["hotel_priority"] = default_signals["hotel_level"]

    return default_signals


def _looks_like_full_date(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def _text_contains_explicit_year(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r"\b(19|20)\d{2}\b", text))


def _request_contains_explicit_year(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r"\b(19|20)\d{2}\b", text))


def _extract_relative_start_date_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    lowered = text.lower()
    today = get_today_date()
    relative_patterns = [
        (r"\b(?:starting|start|leaving|leave|departing|depart|from)\s+tomorrow\b", 1),
        (r"\btomorrow\b", 1),
        (r"\b(?:starting|start|leaving|leave|departing|depart)\s+today\b", 0),
        (r"\btoday\b", 0),
    ]

    for pattern, offset_days in relative_patterns:
        if re.search(pattern, lowered):
            return (today + timedelta(days=offset_days)).strftime("%Y-%m-%d")

    return None


def _resolve_date_to_default_year(date_str: Optional[str], today: Optional[datetime] = None) -> Optional[str]:
    """
    If date_str is a full date inferred from a yearless request,
    keep month/day but resolve the year to the current calendar year.
    """
    if not date_str or not _looks_like_full_date(date_str):
        return date_str

    today_date = today.date() if isinstance(today, datetime) else get_today_date()
    dt = _parse_date(date_str)
    if not dt:
        return date_str

    try:
        candidate = dt.replace(year=today_date.year)
    except ValueError:
        if dt.month == 2 and dt.day == 29:
            candidate = datetime(today_date.year, 3, 1)
        else:
            return date_str

    return candidate.strftime("%Y-%m-%d")


def _normalize_travel_dates(
    value: Any,
    user_request: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Normalize travel date fields.
    """
    default_dates = {
        "start": None,
        "end": None,
        "text": None
    }

    if not isinstance(value, dict):
        return default_dates

    for key in default_dates:
        default_dates[key] = _normalize_string(value.get(key))

    start = default_dates["start"]
    end = default_dates["end"]
    text = default_dates["text"] or ""
    relative_start = _extract_relative_start_date_from_text(user_request or text)

    if relative_start:
        default_dates["start"] = relative_start
        start = default_dates["start"]

    start_has_full_year = _looks_like_full_date(start)
    end_has_full_year = _looks_like_full_date(end)

    request_has_year = _request_contains_explicit_year(user_request)
    text_has_year = _text_contains_explicit_year(text)

    if request_has_year or text_has_year:
        return default_dates

    if start_has_full_year:
        default_dates["start"] = _resolve_date_to_default_year(start)
    if end_has_full_year:
        default_dates["end"] = _resolve_date_to_default_year(end)

    # Keep a valid start date even if the end date is missing.
    # Downstream logic can derive the end date from duration_days.
    if default_dates["end"] and not default_dates["start"]:
        default_dates["end"] = None
    elif default_dates["start"] and default_dates["end"]:
        start_dt = _parse_date(default_dates["start"])
        end_dt = _parse_date(default_dates["end"])
        if start_dt and end_dt and end_dt < start_dt:
            try:
                same_year_end = end_dt.replace(year=start_dt.year)
            except ValueError:
                same_year_end = None

            if same_year_end and same_year_end >= start_dt:
                default_dates["end"] = same_year_end.strftime("%Y-%m-%d")
            else:
                try:
                    next_year_end = end_dt.replace(year=start_dt.year + 1)
                except ValueError:
                    next_year_end = None
                if next_year_end and next_year_end >= start_dt:
                    default_dates["end"] = next_year_end.strftime("%Y-%m-%d")
                else:
                    default_dates["start"] = None
                    default_dates["end"] = None

    return default_dates


def _compute_duration_days(start: Optional[str], end: Optional[str]) -> Optional[int]:
    """
    Compute duration days from explicit start and end dates.
    Inclusive counting is used for travel duration.
    """
    if not start or not end:
        return None

    if not _looks_like_full_date(start) or not _looks_like_full_date(end):
        return None

    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        return None

    delta = (end_dt - start_dt).days + 1
    if delta <= 0:
        return None

    return delta


def _compute_end_date_from_start_and_duration(start: Optional[str], duration_days: Any) -> Optional[str]:
    if not start or not _looks_like_full_date(start):
        return None

    try:
        duration = int(duration_days)
    except Exception:
        return None

    if duration <= 0:
        return None

    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        return None

    return (start_dt + timedelta(days=duration - 1)).strftime("%Y-%m-%d")


def _detect_missing_fields(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Detect missing fields and split them into critical / optional.
    """
    critical = []
    optional = []

    if not data.get("origin"):
        critical.append("origin")

    if not data.get("destination") and not data.get("country"):
        critical.append("destination_or_country")

    travel_dates = data.get("travel_dates", {})
    has_explicit_start = bool(travel_dates.get("start"))
    has_dates = (
        travel_dates.get("start")
        or travel_dates.get("end")
        or travel_dates.get("text")
    )
    has_duration = data.get("duration_days") is not None

    if not has_dates and not has_duration:
        critical.append("travel_dates_or_duration")
    elif not has_explicit_start:
        critical.append("travel_dates.start")

    if data.get("budget", {}).get("total") is None:
        critical.append("budget.total")

    preferences = data.get("preferences", {})
    if not preferences.get("flight"):
        optional.append("preferences.flight")
    if not preferences.get("hotel") and not preferences.get("hotel_level"):
        optional.append("preferences.hotel_or_hotel_level")
    if not preferences.get("food") and not preferences.get("food_preferences"):
        optional.append("preferences.food")
    if not preferences.get("attraction_style") and not preferences.get("attraction_styles"):
        optional.append("preferences.attraction_style")

    return {
        "critical": critical,
        "optional": optional
    }


def _determine_parse_status(result: Dict[str, Any]) -> str:
    if result["meta"]["error"] is not None:
        return "failed"
    if result["missing_fields"]["critical"]:
        return "partial"
    return "success"


def _normalize_result(parsed: Dict[str, Any], user_request: str) -> Dict[str, Any]:
    """
    Normalize the parsed LLM result into a stable schema.
    """
    result = _default_result()
    prepared_request = _prepare_user_request(user_request)

    result["origin"] = _normalize_string(parsed.get("origin"))
    result["destination"] = _normalize_destination(parsed.get("destination"))
    result["country"] = _normalize_string(parsed.get("country"))

    extracted_num_people = _extract_number(parsed.get("num_people"))
    result["num_people"] = extracted_num_people if extracted_num_people is not None else 1
    result["duration_days"] = _extract_number(parsed.get("duration_days"))

    result["travel_dates"] = _normalize_travel_dates(
        parsed.get("travel_dates"),
        user_request=prepared_request
    )

    budget = parsed.get("budget", {})
    if isinstance(budget, dict):
        result["budget"].update(_normalize_budget_for_planning(budget, prepared_request))

    result["preferences"] = _normalize_preferences(
        parsed.get("preferences") or {}
    )

    result["planning_signals"] = _normalize_planning_signals(
        parsed.get("planning_signals") or {}
    )

    computed_duration = _compute_duration_days(
        result["travel_dates"].get("start"),
        result["travel_dates"].get("end")
    )
    if computed_duration is not None:
        result["duration_days"] = computed_duration
    elif result["travel_dates"].get("start") and result["duration_days"] and not result["travel_dates"].get("end"):
        computed_end = _compute_end_date_from_start_and_duration(
            result["travel_dates"].get("start"),
            result["duration_days"],
        )
        if computed_end:
            result["travel_dates"]["end"] = computed_end

    result["missing_fields"] = _detect_missing_fields(result)
    result["is_ready_for_planning"] = len(result["missing_fields"]["critical"]) == 0
    result["parse_status"] = "success" if result["is_ready_for_planning"] else "partial"

    return _merge_missing_fields_from_text(result, prepared_request)


def travel_requirement_parser_llm_agent(user_request: str) -> Dict[str, Any]:
    """
    Main entry point for Agent 1.
    Parse a natural language travel request into structured JSON-like data.
    """
    result = _default_result()
    result["meta"]["llm_called"] = True
    prepared_request = _prepare_user_request(user_request)

    try:
        prompt = _build_prompt(prepared_request)
        raw_text = _call_llm(prompt)

        parsed = _extract_json_from_text(raw_text)
        normalized = _normalize_result(parsed, user_request=prepared_request)
        normalized["user_request"] = prepared_request

        enriched = run_city_planning_agent(
            normalized,
            db_path=str(DEFAULT_DB_PATH)
        )

        enriched = _merge_missing_fields_from_text(enriched, prepared_request)

        enriched["missing_fields"] = _detect_missing_fields(enriched)
        enriched["is_ready_for_planning"] = len(enriched["missing_fields"]["critical"]) == 0

        enriched["meta"]["llm_called"] = True
        enriched["meta"]["llm_call_success"] = True
        enriched["meta"]["raw_response_preview"] = raw_text[:500]
        enriched["meta"]["error"] = None
        enriched["parse_status"] = _determine_parse_status(enriched)

        return enriched

    except Exception as exc:
        result["meta"]["error"] = str(exc)
        result["parse_status"] = "failed"
        result["is_ready_for_planning"] = False
        return result


if __name__ == "__main__":
    test_inputs = [
        "My teacher and I want to travel from Zhuhai to Tokyo and Osaka in early May for five days with a budget of 20000 CNY. We prefer direct flights and budget hotels.",
        "I want to visit Korea in April and keep the trip cheap.",
        "Two people want to travel from Shanghai to Paris from July 10 to July 18 with a budget of 30000 CNY. We want better food options.",
        "I want to travel from Singapore to Seoul in 2026-08-10 to 2026-08-15 with a budget of 12000 CNY and I prefer local food.",
        "My wife and I want to travel from Shenzhen to Rome for seven days with a budget of fifteen thousand CNY.",
        "My teacher and I want to travel from Hong Kong to Tokyo from april 4 to april 9 for 5 days with a budget of 30000 CNY. We prefer direct flights, five star hotels, and better food options.",
        "My friend and I want a clean and centrally located four star hotel, local food but one fine dining dinner, lots of history museums and art galleries, and I definitely want to visit the Great Wall."
    ]

    print("=== LLM Backend Status ===")
    print(json.dumps(get_llm_backend_status(), ensure_ascii=False, indent=2))

    for index, text in enumerate(test_inputs, start=1):
        print(f"\\n\\n=== Test Case {{index}} ===")
        print(f"User request: {{text}}")
        result = travel_requirement_parser_llm_agent(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
