"""
tools/distance_tool.py
─────────────────────────────────────────────────────────────────────────────
חישוב מרחק וזמן הגעה מתחנת הכבאות הקרובה לכתובת האירוע.
משתמש ב-OpenRouteService API (חינמי) לגיאוקודינג + מטריצת מרחקים.
─────────────────────────────────────────────────────────────────────────────
"""
import requests
from config import ORS_API_KEY

# תחנות כבאות מרכזיות בישראל עם קואורדינטות אמיתיות
FIRE_STATIONS = [
    {"name": "תחנת כבאות תל אביב — המלאכה",    "lat": 32.0508, "lon": 34.7719, "city": "תל אביב"},
    {"name": "תחנת כבאות תל אביב — יפו",        "lat": 32.0491, "lon": 34.7505, "city": "תל אביב"},
    {"name": "תחנת כבאות תל אביב — צפון",       "lat": 32.1016, "lon": 34.7956, "city": "תל אביב"},
    {"name": "תחנת כבאות חיפה — הכרמל",         "lat": 32.8056, "lon": 34.9896, "city": "חיפה"},
    {"name": "תחנת כבאות חיפה — נמל",           "lat": 32.8210, "lon": 35.0003, "city": "חיפה"},
    {"name": "תחנת כבאות ירושלים — מרכז",       "lat": 31.7767, "lon": 35.2345, "city": "ירושלים"},
    {"name": "תחנת כבאות באר שבע",              "lat": 31.2518, "lon": 34.7913, "city": "באר שבע"},
    {"name": "תחנת כבאות נתניה",                "lat": 32.3215, "lon": 34.8532, "city": "נתניה"},
    {"name": "תחנת כבאות פתח תקווה",            "lat": 32.0879, "lon": 34.8878, "city": "פתח תקווה"},
    {"name": "תחנת כבאות ראשון לציון",          "lat": 31.9730, "lon": 34.8073, "city": "ראשון לציון"},
    {"name": "תחנת כבאות חולון",                "lat": 32.0107, "lon": 34.7796, "city": "חולון"},
    {"name": "תחנת כבאות אשדוד",                "lat": 31.8044, "lon": 34.6553, "city": "אשדוד"},
    {"name": "תחנת כבאות הרצליה",               "lat": 32.1663, "lon": 34.8438, "city": "הרצליה"},
    {"name": "תחנת כבאות רמת גן",               "lat": 32.0680, "lon": 34.8238, "city": "רמת גן"},
    {"name": "תחנת כבאות טבריה",                "lat": 32.7940, "lon": 35.5310, "city": "טבריה"},
]

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_nearest_fire_station",
        "description": (
            "Calculate the distance and estimated arrival time from the nearest fire station "
            "to the incident address. Use this immediately after getting the full address "
            "to provide accurate arrival time estimates. Returns station name, distance in km, "
            "and estimated driving time in minutes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "כתובת האירוע המלאה — רחוב, מספר, עיר. לדוגמה: 'ארלוזרוב 3, תל אביב'"
                },
            },
            "required": ["address"],
        },
    },
}


# מיקומי ערים ישראליות לfallback מדויק
CITY_COORDS = {
    "תל אביב": (32.0853, 34.7818),
    "חיפה": (32.7940, 34.9896),
    "ירושלים": (31.7683, 35.2137),
    "באר שבע": (31.2518, 34.7913),
    "נתניה": (32.3215, 34.8532),
    "פתח תקווה": (32.0879, 34.8878),
    "ראשון לציון": (31.9730, 34.8073),
    "חולון": (32.0107, 34.7796),
    "אשדוד": (31.8044, 34.6553),
    "הרצליה": (32.1663, 34.8438),
    "רמת גן": (32.0680, 34.8238),
    "טבריה": (32.7940, 35.5310),
    "אשקלון": (31.6688, 34.5743),
    "רחובות": (31.8928, 34.8113),
    "בת ים": (32.0192, 34.7508),
    "בני ברק": (32.0840, 34.8340),
    "כפר סבא": (32.1794, 34.9078),
    "רמלה": (31.9285, 34.8733),
    "לוד": (31.9516, 34.8951),
    "נצרת": (32.6996, 35.3035),
    "עכו": (32.9236, 35.0680),
    "אילת": (29.5581, 34.9482),
}


def _geocode(address: str) -> tuple[float, float] | None:
    """ממיר כתובת טקסטואלית לקואורדינטות GPS."""
    try:
        resp = requests.get(
            "https://api.openrouteservice.org/geocode/search",
            params={
                "api_key": ORS_API_KEY,
                "text": address + ", Israel",
                "size": 1,
                "boundary.country": "IL",
            },
            timeout=10,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            # Fallback: חפש לפי שם עיר בלבד
            for city, coords in CITY_COORDS.items():
                if city in address:
                    return coords
            return None
        coords = features[0]["geometry"]["coordinates"]
        return coords[1], coords[0]  # lat, lon
    except Exception:
        # Fallback לפי עיר
        for city, coords in CITY_COORDS.items():
            if city in address:
                return coords
        return None


def _get_duration_km(origin_lat, origin_lon, dest_lat, dest_lon) -> tuple[float, float] | None:
    """מחשב זמן נסיעה ומרחק בין שתי נקודות."""
    try:
        resp = requests.post(
            "https://api.openrouteservice.org/v2/matrix/driving-car",
            headers={
                "Authorization": ORS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "locations": [[origin_lon, origin_lat], [dest_lon, dest_lat]],
                "metrics": ["duration", "distance"],
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        duration_sec = data["durations"][0][1]
        distance_m   = data["distances"][0][1]
        return duration_sec / 60, distance_m / 1000  # דקות, ק"מ
    except Exception:
        return None


def get_nearest_fire_station(address: str) -> str:
    coords = _geocode(address)
    if not coords:
        return f"[מרחק] לא הצלחתי לאתר את הכתובת '{address}'. הכוחות בדרך."

    incident_lat, incident_lon = coords

    # מצא 3 תחנות קרובות לפי מרחק אווירי
    import math
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    sorted_stations = sorted(
        FIRE_STATIONS,
        key=lambda s: haversine(incident_lat, incident_lon, s["lat"], s["lon"])
    )

    # חשב זמן הגעה אמיתי לתחנה הקרובה ביותר
    for station in sorted_stations[:3]:
        result = _get_duration_km(station["lat"], station["lon"], incident_lat, incident_lon)
        if result:
            minutes, km = result
            minutes_rounded = round(minutes)
            km_rounded = round(km, 1)
            return (
                f"🚒 תחנת הכבאות הקרובה: {station['name']}\n"
                f"📍 מרחק: {km_rounded} ק\"מ\n"
                f"⏱️ זמן הגעה משוער: {minutes_rounded} דקות"
            )

    # Fallback אם API נכשל
    nearest = sorted_stations[0]
    aerial_km = round(haversine(incident_lat, incident_lon, nearest["lat"], nearest["lon"]), 1)
    est_minutes = round(aerial_km * 2.5)
    return (
        f"🚒 תחנת הכבאות הקרובה: {nearest['name']}\n"
        f"📍 מרחק אווירי: ~{aerial_km} ק\"מ\n"
        f"⏱️ זמן הגעה משוער: ~{est_minutes} דקות"
    )
