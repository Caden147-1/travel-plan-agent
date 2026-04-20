import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


BASE_DIR = Path(__file__).resolve().parent

CITIES_FILE = BASE_DIR / "cities15000.txt"
COUNTRY_INFO_FILE = BASE_DIR / "countryInfo.txt"
OUTPUT_FILE = BASE_DIR / "country_default_city_plan.json"


# =========================================================
# Manual overrides for tourism-friendly defaults
# =========================================================
MANUAL_OVERRIDES: Dict[str, List[Dict[str, Any]]] = {
    "japan": [
        {"city": "Tokyo", "days_ratio": 0.35},
        {"city": "Osaka", "days_ratio": 0.25},
        {"city": "Kyoto", "days_ratio": 0.20},
        {"city": "Sapporo", "days_ratio": 0.12},
        {"city": "Fukuoka", "days_ratio": 0.08},
    ],
    "south korea": [
        {"city": "Seoul", "days_ratio": 0.45},
        {"city": "Busan", "days_ratio": 0.22},
        {"city": "Jeju", "days_ratio": 0.18},
        {"city": "Incheon", "days_ratio": 0.08},
        {"city": "Daegu", "days_ratio": 0.07},
    ],
    "china": [
        {"city": "Beijing", "days_ratio": 0.22},
        {"city": "Shanghai", "days_ratio": 0.20},
        {"city": "Chengdu", "days_ratio": 0.18},
        {"city": "Guangzhou", "days_ratio": 0.16},
        {"city": "Shenzhen", "days_ratio": 0.12},
    ],
    "thailand": [
        {"city": "Bangkok", "days_ratio": 0.35},
        {"city": "Chiang Mai", "days_ratio": 0.20},
        {"city": "Phuket", "days_ratio": 0.20},
        {"city": "Krabi", "days_ratio": 0.15},
        {"city": "Pattaya", "days_ratio": 0.10},
    ],
    "singapore": [
        {"city": "Singapore", "days_ratio": 1.0},
    ],
    "malaysia": [
        {"city": "Kuala Lumpur", "days_ratio": 0.35},
        {"city": "Penang", "days_ratio": 0.22},
        {"city": "Langkawi", "days_ratio": 0.18},
        {"city": "Malacca", "days_ratio": 0.13},
        {"city": "Kota Kinabalu", "days_ratio": 0.12},
    ],
    "indonesia": [
        {"city": "Jakarta", "days_ratio": 0.18},
        {"city": "Denpasar", "days_ratio": 0.30},
        {"city": "Yogyakarta", "days_ratio": 0.20},
        {"city": "Surabaya", "days_ratio": 0.17},
        {"city": "Bandung", "days_ratio": 0.15},
    ],
    "vietnam": [
        {"city": "Hanoi", "days_ratio": 0.30},
        {"city": "Ho Chi Minh City", "days_ratio": 0.28},
        {"city": "Da Nang", "days_ratio": 0.18},
        {"city": "Hoi An", "days_ratio": 0.14},
        {"city": "Ha Long", "days_ratio": 0.10},
    ],
    "philippines": [
        {"city": "Manila", "days_ratio": 0.28},
        {"city": "Cebu City", "days_ratio": 0.22},
        {"city": "Boracay", "days_ratio": 0.20},
        {"city": "Palawan", "days_ratio": 0.18},
        {"city": "Davao", "days_ratio": 0.12},
    ],
    "india": [
        {"city": "Delhi", "days_ratio": 0.28},
        {"city": "Mumbai", "days_ratio": 0.22},
        {"city": "Jaipur", "days_ratio": 0.18},
        {"city": "Agra", "days_ratio": 0.18},
        {"city": "Bengaluru", "days_ratio": 0.14},
    ],
    "nepal": [
        {"city": "Kathmandu", "days_ratio": 0.55},
        {"city": "Pokhara", "days_ratio": 0.30},
        {"city": "Bhaktapur", "days_ratio": 0.15},
    ],
    "sri lanka": [
        {"city": "Colombo", "days_ratio": 0.32},
        {"city": "Kandy", "days_ratio": 0.24},
        {"city": "Galle", "days_ratio": 0.18},
        {"city": "Ella", "days_ratio": 0.14},
        {"city": "Sigiriya", "days_ratio": 0.12},
    ],
    "turkey": [
        {"city": "Istanbul", "days_ratio": 0.34},
        {"city": "Antalya", "days_ratio": 0.20},
        {"city": "Nevşehir", "days_ratio": 0.18},
        {"city": "Izmir", "days_ratio": 0.16},
        {"city": "Ankara", "days_ratio": 0.12},
    ],
    "france": [
        {"city": "Paris", "days_ratio": 0.40},
        {"city": "Nice", "days_ratio": 0.18},
        {"city": "Lyon", "days_ratio": 0.16},
        {"city": "Marseille", "days_ratio": 0.14},
        {"city": "Bordeaux", "days_ratio": 0.12},
    ],
    "italy": [
        {"city": "Rome", "days_ratio": 0.30},
        {"city": "Florence", "days_ratio": 0.22},
        {"city": "Venice", "days_ratio": 0.20},
        {"city": "Milan", "days_ratio": 0.16},
        {"city": "Naples", "days_ratio": 0.12},
    ],
    "spain": [
        {"city": "Barcelona", "days_ratio": 0.28},
        {"city": "Madrid", "days_ratio": 0.25},
        {"city": "Seville", "days_ratio": 0.18},
        {"city": "Valencia", "days_ratio": 0.16},
        {"city": "Granada", "days_ratio": 0.13},
    ],
    "germany": [
        {"city": "Berlin", "days_ratio": 0.28},
        {"city": "Munich", "days_ratio": 0.24},
        {"city": "Frankfurt am Main", "days_ratio": 0.18},
        {"city": "Hamburg", "days_ratio": 0.16},
        {"city": "Cologne", "days_ratio": 0.14},
    ],
    "switzerland": [
        {"city": "Zurich", "days_ratio": 0.26},
        {"city": "Lucerne", "days_ratio": 0.22},
        {"city": "Interlaken", "days_ratio": 0.22},
        {"city": "Geneva", "days_ratio": 0.16},
        {"city": "Zermatt", "days_ratio": 0.14},
    ],
    "austria": [
        {"city": "Vienna", "days_ratio": 0.45},
        {"city": "Salzburg", "days_ratio": 0.25},
        {"city": "Innsbruck", "days_ratio": 0.18},
        {"city": "Hallstatt", "days_ratio": 0.12},
    ],
    "netherlands": [
        {"city": "Amsterdam", "days_ratio": 0.42},
        {"city": "Rotterdam", "days_ratio": 0.22},
        {"city": "The Hague", "days_ratio": 0.18},
        {"city": "Utrecht", "days_ratio": 0.18},
    ],
    "belgium": [
        {"city": "Brussels", "days_ratio": 0.40},
        {"city": "Bruges", "days_ratio": 0.26},
        {"city": "Antwerp", "days_ratio": 0.18},
        {"city": "Ghent", "days_ratio": 0.16},
    ],
    "portugal": [
        {"city": "Lisbon", "days_ratio": 0.42},
        {"city": "Porto", "days_ratio": 0.28},
        {"city": "Faro", "days_ratio": 0.16},
        {"city": "Coimbra", "days_ratio": 0.14},
    ],
    "greece": [
        {"city": "Athens", "days_ratio": 0.32},
        {"city": "Santorini", "days_ratio": 0.24},
        {"city": "Mykonos", "days_ratio": 0.18},
        {"city": "Thessaloniki", "days_ratio": 0.14},
        {"city": "Heraklion", "days_ratio": 0.12},
    ],
    "norway": [
        {"city": "Oslo", "days_ratio": 0.36},
        {"city": "Bergen", "days_ratio": 0.28},
        {"city": "Tromsø", "days_ratio": 0.18},
        {"city": "Stavanger", "days_ratio": 0.18},
    ],
    "sweden": [
        {"city": "Stockholm", "days_ratio": 0.45},
        {"city": "Gothenburg", "days_ratio": 0.25},
        {"city": "Malmö", "days_ratio": 0.18},
        {"city": "Kiruna", "days_ratio": 0.12},
    ],
    "denmark": [
        {"city": "Copenhagen", "days_ratio": 0.60},
        {"city": "Aarhus", "days_ratio": 0.22},
        {"city": "Odense", "days_ratio": 0.18},
    ],
    "finland": [
        {"city": "Helsinki", "days_ratio": 0.48},
        {"city": "Rovaniemi", "days_ratio": 0.24},
        {"city": "Turku", "days_ratio": 0.16},
        {"city": "Tampere", "days_ratio": 0.12},
    ],
    "iceland": [
        {"city": "Reykjavik", "days_ratio": 0.65},
        {"city": "Akureyri", "days_ratio": 0.20},
        {"city": "Vik", "days_ratio": 0.15},
    ],
    "ireland": [
        {"city": "Dublin", "days_ratio": 0.48},
        {"city": "Galway", "days_ratio": 0.22},
        {"city": "Cork", "days_ratio": 0.18},
        {"city": "Limerick", "days_ratio": 0.12},
    ],
    "united kingdom": [
        {"city": "London", "days_ratio": 0.34},
        {"city": "Edinburgh", "days_ratio": 0.22},
        {"city": "Manchester", "days_ratio": 0.18},
        {"city": "Oxford", "days_ratio": 0.14},
        {"city": "Liverpool", "days_ratio": 0.12},
    ],
    "united states": [
        {"city": "New York", "days_ratio": 0.24},
        {"city": "Los Angeles", "days_ratio": 0.20},
        {"city": "San Francisco", "days_ratio": 0.18},
        {"city": "Las Vegas", "days_ratio": 0.18},
        {"city": "Chicago", "days_ratio": 0.12},
        {"city": "Seattle", "days_ratio": 0.08},
    ],
    "canada": [
        {"city": "Toronto", "days_ratio": 0.26},
        {"city": "Vancouver", "days_ratio": 0.24},
        {"city": "Montreal", "days_ratio": 0.20},
        {"city": "Quebec", "days_ratio": 0.16},
        {"city": "Calgary", "days_ratio": 0.14},
    ],
    "mexico": [
        {"city": "Mexico City", "days_ratio": 0.40},
        {"city": "Cancún", "days_ratio": 0.26},
        {"city": "Guadalajara", "days_ratio": 0.18},
        {"city": "Mérida", "days_ratio": 0.16},
    ],
    "australia": [
        {"city": "Sydney", "days_ratio": 0.28},
        {"city": "Melbourne", "days_ratio": 0.24},
        {"city": "Brisbane", "days_ratio": 0.18},
        {"city": "Perth", "days_ratio": 0.16},
        {"city": "Gold Coast", "days_ratio": 0.14},
    ],
    "new zealand": [
        {"city": "Auckland", "days_ratio": 0.34},
        {"city": "Queenstown", "days_ratio": 0.24},
        {"city": "Christchurch", "days_ratio": 0.20},
        {"city": "Wellington", "days_ratio": 0.12},
        {"city": "Rotorua", "days_ratio": 0.10},
    ],
}


# =========================================================
# Optional blacklist for cities that are often not ideal
# for default tourism planning or may be duplicated weirdly
# =========================================================
CITY_BLACKLIST_BY_COUNTRY: Dict[str, set] = {
    "japan": {"yokohama"},
    "south korea": set(),
    "china": set(),
}


# =========================================================
# Helpers
# =========================================================
def normalize_country_name(name: str) -> str:
    return (name or "").strip().lower()


def normalize_city_name(name: str) -> str:
    return (name or "").strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def round_ratio_list(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []

    rounded = []
    total = 0.0

    for item in items:
        ratio = max(0.0, float(item.get("days_ratio", 0.0)))
        rounded_ratio = round(ratio, 4)
        rounded.append({
            "city": item["city"],
            "days_ratio": rounded_ratio
        })
        total += rounded_ratio

    diff = round(1.0 - total, 4)
    if rounded:
        rounded[0]["days_ratio"] = round(rounded[0]["days_ratio"] + diff, 4)

    return rounded


def normalize_override_ratios(overrides: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}

    for country, cities in overrides.items():
        if not cities:
            continue

        total = 0.0
        cleaned = []
        for item in cities:
            city = normalize_city_name(item.get("city"))
            ratio = float(item.get("days_ratio", 0.0))
            if not city or ratio <= 0:
                continue
            cleaned.append({"city": city, "days_ratio": ratio})
            total += ratio

        if not cleaned or total <= 0:
            continue

        normalized = []
        for item in cleaned:
            normalized.append({
                "city": item["city"],
                "days_ratio": item["days_ratio"] / total
            })

        result[normalize_country_name(country)] = round_ratio_list(normalized)

    return result


# =========================================================
# GeoNames loaders
# =========================================================
def load_country_code_to_name(country_info_path: Path) -> Dict[str, str]:
    """
    Load country code -> country name from GeoNames countryInfo.txt
    """
    mapping: Dict[str, str] = {}

    with open(country_info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                continue

            iso_code = parts[0].strip()
            country_name = parts[4].strip()

            if iso_code and country_name:
                mapping[iso_code] = country_name

    return mapping


def should_keep_city(country_name: str, city_name: str, feature_code: str, population: int) -> bool:
    """
    Basic filtering rule for default city candidates.
    """
    if not city_name:
        return False

    if population <= 0:
        return False

    blacklist = CITY_BLACKLIST_BY_COUNTRY.get(country_name, set())
    if city_name.strip().lower() in blacklist:
        return False

    # Keep only populated place feature codes
    allowed_codes = {
        "PPLC",   # capital
        "PPLA",   # seat of admin division
        "PPLA2",
        "PPLA3",
        "PPLA4",
        "PPL",    # populated place
        "PPLG",
        "PPLL",
        "PPLS",
    }

    return feature_code in allowed_codes


def load_cities(cities_path: Path, country_code_to_name: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load GeoNames cities15000.txt and group cities by normalized country name.

    GeoNames fields:
    0 geonameid
    1 name
    2 asciiname
    3 alternatenames
    4 latitude
    5 longitude
    6 feature class
    7 feature code
    8 country code
    9 cc2
    10 admin1 code
    11 admin2 code
    12 admin3 code
    13 admin4 code
    14 population
    15 elevation
    16 dem
    17 timezone
    18 modification date
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    with open(cities_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 19:
                continue

            name = row[1].strip()
            asciiname = row[2].strip()
            feature_class = row[6].strip()
            feature_code = row[7].strip()
            country_code = row[8].strip()
            population = safe_int(row[14], 0)

            if feature_class != "P":
                continue

            country_name_raw = country_code_to_name.get(country_code)
            if not country_name_raw:
                continue

            country_name = normalize_country_name(country_name_raw)
            city_name = normalize_city_name(asciiname or name)

            if not should_keep_city(country_name, city_name, feature_code, population):
                continue

            city_item = {
                "city": city_name,
                "population": population,
                "feature_code": feature_code,
                "is_capital": feature_code == "PPLC",
            }

            grouped.setdefault(country_name, []).append(city_item)

    return grouped


# =========================================================
# Ranking and ratio generation
# =========================================================
def deduplicate_cities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate by lowercase city name and keep the stronger candidate.
    """
    best_by_city: Dict[str, Dict[str, Any]] = {}

    for item in items:
        city = normalize_city_name(item.get("city"))
        if not city:
            continue

        key = city.lower()
        old = best_by_city.get(key)

        if old is None:
            best_by_city[key] = item
            continue

        old_score = (
            1 if old.get("is_capital") else 0,
            old.get("population", 0),
        )
        new_score = (
            1 if item.get("is_capital") else 0,
            item.get("population", 0),
        )

        if new_score > old_score:
            best_by_city[key] = item

    return list(best_by_city.values())


def city_rank_score(item: Dict[str, Any]) -> float:
    """
    Ranking score for default candidates.
    """
    population = max(1, item.get("population", 1))
    capital_boost = 1.20 if item.get("is_capital") else 1.0

    # Mild admin seat preference
    feature_code = item.get("feature_code", "")
    admin_boost = 1.0
    if feature_code in {"PPLA", "PPLA2"}:
        admin_boost = 1.05

    return population * capital_boost * admin_boost


def rank_cities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=city_rank_score, reverse=True)


def assign_days_ratio(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign weighted days_ratio based on ranking score.
    """
    if not items:
        return []

    scores = [max(1.0, city_rank_score(item)) for item in items]
    total_score = sum(scores)

    if total_score <= 0:
        equal_ratio = round(1.0 / len(items), 4)
        return [{"city": item["city"], "days_ratio": equal_ratio} for item in items]

    raw_result = []
    for item, score in zip(items, scores):
        raw_result.append({
            "city": item["city"],
            "days_ratio": score / total_score
        })

    return round_ratio_list(raw_result)


def build_country_default_city_plan(
    grouped_cities: Dict[str, List[Dict[str, Any]]],
    max_cities_per_country: int = 5,
    min_population_threshold: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}

    for country_name, items in grouped_cities.items():
        cleaned = deduplicate_cities(items)

        if min_population_threshold is not None:
            cleaned = [
                item for item in cleaned
                if item.get("population", 0) >= min_population_threshold or item.get("is_capital")
            ]

        ranked = rank_cities(cleaned)
        selected = ranked[:max_cities_per_country]

        if not selected:
            continue

        result[country_name] = assign_days_ratio(selected)

    return dict(sorted(result.items(), key=lambda x: x[0]))


# =========================================================
# Merge manual overrides
# =========================================================
def merge_with_manual_overrides(
    auto_result: Dict[str, List[Dict[str, Any]]],
    manual_overrides: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    merged = dict(auto_result)

    for country_name, city_list in manual_overrides.items():
        merged[country_name] = city_list

    return dict(sorted(merged.items(), key=lambda x: x[0]))


# =========================================================
# Optional Python export for direct import
# =========================================================
def export_python_module(
    output_path: Path,
    data: Dict[str, List[Dict[str, Any]]],
    variable_name: str = "COUNTRY_DEFAULT_CITY_PLAN"
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{variable_name} = ")
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
        f.write("\n")


# =========================================================
# Main
# =========================================================
def main():
    if not CITIES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {CITIES_FILE}")

    if not COUNTRY_INFO_FILE.exists():
        raise FileNotFoundError(f"Missing file: {COUNTRY_INFO_FILE}")

    normalized_overrides = normalize_override_ratios(MANUAL_OVERRIDES)

    country_code_to_name = load_country_code_to_name(COUNTRY_INFO_FILE)
    grouped_cities = load_cities(CITIES_FILE, country_code_to_name)

    auto_result = build_country_default_city_plan(
        grouped_cities=grouped_cities,
        max_cities_per_country=5,
        min_population_threshold=15000,
    )

    final_result = merge_with_manual_overrides(auto_result, normalized_overrides)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    python_output = BASE_DIR / "country_default_city_plan.py"
    export_python_module(python_output, final_result)

    print(f"Done. JSON output saved to: {OUTPUT_FILE}")
    print(f"Done. Python output saved to: {python_output}")
    print("\nSample preview:")
    preview_keys = list(final_result.keys())[:8]
    preview = {k: final_result[k] for k in preview_keys}
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()