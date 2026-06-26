"""
tools/distance_tool.py
─────────────────────────────────────────────────────────────────────────────
חישוב מרחק וזמן הגעה מתחנת הכבאות הקרובה לכתובת האירוע.
מקור נתונים: Overpass API (OpenStreetMap) — תחנות כבאות עדכניות בכל ישראל.
Geocoding: Nominatim (OpenStreetMap) — תמיכה מלאה בעברית.
Routing: OpenRouteService — זמן נסיעה אמיתי לפי תנועה.
─────────────────────────────────────────────────────────────────────────────
"""
import math
import requests
from config import ORS_API_KEY

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_nearest_fire_station",
        "description": (
            "Find the nearest fire station to the incident address using live OpenStreetMap data. "
            "Returns station name, distance in km, and estimated driving time in minutes. "
            "Call immediately after collecting full address (street, number, city)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "כתובת מלאה — רחוב מספר, עיר. לדוגמה: 'וולפסון 10, ראש העין'"},
                "city":    {"type": "string", "description": "שם העיר בלבד, לדוגמה: 'ראש העין'"},
            },
            "required": ["address", "city"],
        },
    },
}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _geocode(address: str) -> tuple | None:
    """ממיר כתובת עברית לקואורדינטות GPS דרך Nominatim."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address + ", ישראל", "format": "json", "limit": 1, "countrycodes": "il"},
            headers={"User-Agent": "DisasterGuard/1.0 (emergency-response-system)"},
            timeout=8,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def _get_fire_stations_near(lat: float, lon: float, radius_m: int = 30000) -> list:
    """
    מחזיר תחנות כבאות בטווח של radius_m מטר מהנקודה הנתונה.
    מקור: Overpass API (OpenStreetMap) — נתונים עדכניים בזמן אמת.
    """
    query = f"""
    [out:json][timeout:10];
    (
      node["amenity"="fire_station"](around:{radius_m},{lat},{lon});
      way["amenity"="fire_station"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=12,
        )
        elements = resp.json().get("elements", [])
        stations = []
        for el in elements:
            s_lat = el.get("lat") or el.get("center", {}).get("lat")
            s_lon = el.get("lon") or el.get("center", {}).get("lon")
            name = el.get("tags", {}).get("name:he") or el.get("tags", {}).get("name") or "תחנת כבאות"
            if s_lat and s_lon:
                stations.append({"name": name, "lat": s_lat, "lon": s_lon})
        return stations
    except Exception:
        return []


def _get_driving_time(origin_lat, origin_lon, dest_lat, dest_lon) -> tuple | None:
    """זמן נסיעה ומרחק אמיתי דרך OpenRouteService."""
    try:
        resp = requests.post(
            "https://api.openrouteservice.org/v2/matrix/driving-car",
            headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
            json={
                "locations": [[origin_lon, origin_lat], [dest_lon, dest_lat]],
                "metrics": ["duration", "distance"],
            },
            timeout=8,
        )
        data = resp.json()
        return data["durations"][0][1] / 60, data["distances"][0][1] / 1000
    except Exception:
        return None


def get_nearest_fire_station(address: str, city: str = "") -> str:
    # שלב 1: Geocoding של כתובת האירוע
    incident_coords = _geocode(address)
    if not incident_coords:
        incident_coords = _geocode(city + ", ישראל") if city else None

    if not incident_coords:
        return "🚒 נפתחה פנייה ב-102. הכוחות בדרך אליך."

    incident_lat, incident_lon = incident_coords

    # שלב 2: שליפת תחנות כבאות חיות מ-OpenStreetMap
    stations = _get_fire_stations_near(incident_lat, incident_lon, radius_m=30000)

    if not stations:
        # הרחב חיפוש ל-60 ק"מ
        stations = _get_fire_stations_near(incident_lat, incident_lon, radius_m=60000)

    if not stations:
        return "🚒 נפתחה פנייה ב-102. הכוחות בדרך אליך."

    # שלב 3: מיין לפי מרחק אווירי ובדוק זמן נסיעה אמיתי
    stations_sorted = sorted(
        stations,
        key=lambda s: _haversine(incident_lat, incident_lon, s["lat"], s["lon"])
    )

    for station in stations_sorted[:3]:
        result = _get_driving_time(station["lat"], station["lon"], incident_lat, incident_lon)
        if result:
            minutes, km = result
            return (
                f"🚒 תחנת הכבאות הקרובה: {station['name']}\n"
                f"📍 מרחק: {round(km, 1)} ק\"מ\n"
                f"⏱️ זמן הגעה משוער: {round(minutes)} דקות"
            )

    # Fallback: מרחק אווירי
    nearest = stations_sorted[0]
    km = _haversine(incident_lat, incident_lon, nearest["lat"], nearest["lon"])
    return (
        f"🚒 תחנת הכבאות הקרובה: {nearest['name']}\n"
        f"📍 מרחק: ~{round(km, 1)} ק\"מ\n"
        f"⏱️ זמן הגעה משוער: ~{round(km * 2.5)} דקות"
    )
