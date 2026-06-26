"""
tools/distance_tool.py
─────────────────────────────────────────────────────────────────────────────
חישוב מרחק וזמן הגעה מתחנת הכבאות הקרובה לכתובת האירוע.
משתמש ב-Nominatim (OpenStreetMap) לגיאוקודינג + ORS למטריצת מרחקים.
Fallback: התאמה לפי שם עיר אם Geocoding נכשל.
─────────────────────────────────────────────────────────────────────────────
"""
import math
import requests
from config import ORS_API_KEY

FIRE_STATIONS = [
    {"name": "תחנת כבאות תל אביב — המלאכה",    "lat": 32.0508, "lon": 34.7719, "cities": ["תל אביב", "tel aviv"]},
    {"name": "תחנת כבאות תל אביב — יפו",        "lat": 32.0491, "lon": 34.7505, "cities": ["יפו", "jaffa"]},
    {"name": "תחנת כבאות תל אביב — צפון",       "lat": 32.1016, "lon": 34.7956, "cities": ["תל אביב צפון"]},
    {"name": "תחנת כבאות חיפה — הכרמל",         "lat": 32.8056, "lon": 34.9896, "cities": ["חיפה", "haifa", "הכרמל"]},
    {"name": "תחנת כבאות חיפה — נמל",           "lat": 32.8210, "lon": 35.0003, "cities": ["חיפה", "haifa"]},
    {"name": "תחנת כבאות ירושלים — מרכז",       "lat": 31.7767, "lon": 35.2345, "cities": ["ירושלים", "jerusalem"]},
    {"name": "תחנת כבאות באר שבע",              "lat": 31.2518, "lon": 34.7913, "cities": ["באר שבע", "beer sheva"]},
    {"name": "תחנת כבאות נתניה",                "lat": 32.3215, "lon": 34.8532, "cities": ["נתניה", "netanya"]},
    {"name": "תחנת כבאות פתח תקווה",            "lat": 32.0879, "lon": 34.8878, "cities": ["פתח תקווה", "petah tikva"]},
    {"name": "תחנת כבאות ראשון לציון",          "lat": 31.9730, "lon": 34.8073, "cities": ["ראשון לציון", "rishon lezion"]},
    {"name": "תחנת כבאות חולון",                "lat": 32.0107, "lon": 34.7796, "cities": ["חולון", "holon"]},
    {"name": "תחנת כבאות אשדוד",                "lat": 31.8044, "lon": 34.6553, "cities": ["אשדוד", "ashdod"]},
    {"name": "תחנת כבאות הרצליה",               "lat": 32.1663, "lon": 34.8438, "cities": ["הרצליה", "herzliya"]},
    {"name": "תחנת כבאות רמת גן",               "lat": 32.0680, "lon": 34.8238, "cities": ["רמת גן", "ramat gan"]},
    {"name": "תחנת כבאות טבריה",                "lat": 32.7940, "lon": 35.5310, "cities": ["טבריה", "tiberias"]},
    {"name": "תחנת כבאות אשקלון",               "lat": 31.6693, "lon": 34.5731, "cities": ["אשקלון", "ashkelon"]},
    {"name": "תחנת כבאות רחובות",               "lat": 31.8947, "lon": 34.8117, "cities": ["רחובות", "rehovot"]},
    {"name": "תחנת כבאות כפר סבא",              "lat": 32.1753, "lon": 34.9077, "cities": ["כפר סבא", "kfar saba"]},
    {"name": "תחנת כבאות בני ברק",              "lat": 32.0839, "lon": 34.8336, "cities": ["בני ברק", "bnei brak"]},
    {"name": "תחנת כבאות גבעתיים",              "lat": 32.0706, "lon": 34.8131, "cities": ["גבעתיים", "givatayim"]},
]

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_nearest_fire_station",
        "description": (
            "Calculate distance and estimated arrival time from nearest fire station to incident. "
            "Call immediately after collecting full address (street, number, city). "
            "Returns station name, distance, and driving time in minutes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "כתובת מלאה — רחוב מספר, עיר. לדוגמה: 'אבא הלל 10, רמת גן'"},
                "city": {"type": "string", "description": "שם העיר בלבד, לדוגמה: 'רמת גן'"},
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


def _geocode_nominatim(address: str) -> tuple | None:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1, "countrycodes": "il"},
            headers={"User-Agent": "DisasterGuard/1.0 (emergency-response)"},
            timeout=8,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def _get_driving_time(origin_lat, origin_lon, dest_lat, dest_lon) -> tuple | None:
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


def _station_by_city(city: str):
    city_lower = city.lower().strip()
    for s in FIRE_STATIONS:
        if any(city_lower in c.lower() or c.lower() in city_lower for c in s["cities"]):
            return s
    return None


def get_nearest_fire_station(address: str, city: str = "") -> str:
    incident_coords = _geocode_nominatim(address)

    if incident_coords:
        incident_lat, incident_lon = incident_coords
        sorted_stations = sorted(
            FIRE_STATIONS,
            key=lambda s: _haversine(incident_lat, incident_lon, s["lat"], s["lon"])
        )
        for station in sorted_stations[:3]:
            result = _get_driving_time(station["lat"], station["lon"], incident_lat, incident_lon)
            if result:
                minutes, km = result
                return (
                    f"🚒 תחנת הכבאות הקרובה: {station['name']}\n"
                    f"📍 מרחק: {round(km, 1)} ק\"מ\n"
                    f"⏱️ זמן הגעה משוער: {round(minutes)} דקות"
                )
            # Fallback aerial
            km = _haversine(incident_lat, incident_lon, station["lat"], station["lon"])
            return (
                f"🚒 תחנת הכבאות הקרובה: {station['name']}\n"
                f"📍 מרחק: ~{round(km, 1)} ק\"מ\n"
                f"⏱️ זמן הגעה משוער: ~{round(km * 2.5)} דקות"
            )

    # Fallback by city name
    station = _station_by_city(city) if city else None
    if not station and FIRE_STATIONS:
        station = FIRE_STATIONS[0]

    km = _haversine(
        station["lat"], station["lon"],
        incident_coords[0] if incident_coords else station["lat"],
        incident_coords[1] if incident_coords else station["lon"]
    ) if incident_coords else None

    est = f"~{round(km * 2.5)} דקות" if km else "5-10 דקות"
    km_str = f"~{round(km, 1)} ק\"מ" if km else "—"

    return (
        f"🚒 תחנת הכבאות הקרובה: {station['name']}\n"
        f"📍 מרחק: {km_str}\n"
        f"⏱️ זמן הגעה משוער: {est}"
    )
