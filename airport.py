import csv
import json
from pathlib import Path

INPUT_CSV = Path("airports.csv")
OUTPUT_JSON = Path("global_airports.json")

def clean(value):
    if value is None:
        return ""
    return str(value).strip()

def to_float(value):
    try:
        return float(value)
    except Exception:
        return None

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError("airports.csv not found in current directory.")

    airport_map = {}

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iata = clean(row.get("iata_code")).upper()
            if not iata:
                continue

            lat = to_float(row.get("latitude_deg"))
            lng = to_float(row.get("longitude_deg"))
            if lat is None or lng is None:
                continue

            airport_map[iata] = {
                "iata": iata,
                "icao": clean(row.get("ident")).upper(),
                "name": clean(row.get("name")),
                "city": clean(row.get("municipality")),
                "country": clean(row.get("iso_country")).upper(),
                "lat": lat,
                "lng": lng,
                "type": clean(row.get("type")),
            }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(airport_map, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(airport_map)} airports to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()