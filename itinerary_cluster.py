from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from planner_schema import HOTEL_CLUSTER_MAX_ROUTE_KM, HOTEL_CLUSTER_RADIUS_KM, normalize_text, safe_float
from itinerary_rules import haversine_km


def cluster_points_by_hotel_radius(
    hotels: List[Dict[str, Any]],
    pois: List[Dict[str, Any]],
    restaurants: List[Dict[str, Any]],
    city: str,
    radius_km: float = HOTEL_CLUSTER_RADIUS_KM,
    max_route_km: float = HOTEL_CLUSTER_MAX_ROUTE_KM,
) -> List[Dict[str, Any]]:
    city_l = normalize_text(city).lower()
    city_hotels = [h for h in hotels if normalize_text(h.get("city")).lower() == city_l]
    city_pois = [p for p in pois if normalize_text(p.get("city")).lower() == city_l]
    city_restaurants = [r for r in restaurants if normalize_text(r.get("city")).lower() == city_l]

    clusters: List[Dict[str, Any]] = []

    for hotel in city_hotels:
        h_lat = safe_float(hotel.get("lat"), None)
        h_lng = safe_float(hotel.get("lng"), None)

        poi_in: List[Dict[str, Any]] = []
        for p in city_pois:
            p_lat = safe_float(p.get("lat"), None)
            p_lng = safe_float(p.get("lng"), None)
            if h_lat is None or h_lng is None or p_lat is None or p_lng is None:
                poi_in.append(p)
                continue
            d = haversine_km(h_lat, h_lng, p_lat, p_lng)
            if d <= radius_km and d <= max_route_km:
                poi_in.append(p)

        rest_in: List[Dict[str, Any]] = []
        for r in city_restaurants:
            r_lat = safe_float(r.get("lat"), None)
            r_lng = safe_float(r.get("lng"), None)
            if h_lat is None or h_lng is None or r_lat is None or r_lng is None:
                rest_in.append(r)
                continue
            d = haversine_km(h_lat, h_lng, r_lat, r_lng)
            if d <= radius_km and d <= max_route_km:
                rest_in.append(r)

        clusters.append({
            "city": city,
            "hotel": hotel,
            "pois": poi_in,
            "restaurants": rest_in,
            "coverage_count": len(poi_in) + len(rest_in),
        })

    clusters.sort(
        key=lambda x: (
            int(x.get("coverage_count", 0)),
            safe_float((x.get("hotel") or {}).get("rating"), 0.0),
            safe_float((x.get("hotel") or {}).get("stars"), 0.0),
        ),
        reverse=True,
    )

    return clusters


def select_primary_hotel_for_city(city: str, hotel_clusters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not hotel_clusters:
        return None
    top = hotel_clusters[0]
    hotel = top.get("hotel")
    if isinstance(hotel, dict):
        return hotel
    return None


def collect_city_candidates(city: str, selected_pois: List[Dict[str, Any]], selected_restaurants: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    city_l = normalize_text(city).lower()
    pois = [x for x in selected_pois if normalize_text(x.get("city")).lower() == city_l]
    restaurants = [x for x in selected_restaurants if normalize_text(x.get("city")).lower() == city_l]

    pois = sorted(
        pois,
        key=lambda x: (
            safe_float(x.get("rating"), 0.0),
            -safe_float(x.get("estimated_total_cost"), 0.0),
        ),
        reverse=True,
    )

    restaurants = sorted(
        restaurants,
        key=lambda x: (
            safe_float(x.get("rating"), 0.0),
            -safe_float(x.get("estimated_total_cost"), 0.0),
        ),
        reverse=True,
    )

    return {
        "pois": pois,
        "restaurants": restaurants,
    }


def split_must_visit_and_regular_pois(pois: List[Dict[str, Any]], must_visit_names: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    targets = [normalize_text(x).lower() for x in must_visit_names if normalize_text(x)]

    must: List[Dict[str, Any]] = []
    regular: List[Dict[str, Any]] = []

    for poi in pois:
        text = f"{normalize_text(poi.get('name')).lower()} {normalize_text(poi.get('address')).lower()}"
        matched = None
        for t in targets:
            if t and t in text:
                matched = t
                break
        if matched:
            item = dict(poi)
            item["_must_visit_name_norm"] = matched
            must.append(item)
        else:
            regular.append(poi)

    return must, regular
