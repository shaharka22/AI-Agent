"""
tools/usgs_tool.py — USGS Earthquake Hazards Program (global, live)
"""
import requests
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_exponential
from config import USGS_API

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_earthquakes_global",
        "description": (
            "Fetch recent earthquakes worldwide from USGS, optionally filtered by region "
            "(e.g. 'Philippines', 'Japan', 'Turkey'). Returns magnitude, depth, location, "
            "and a tsunami-risk flag for shallow/strong quakes. Use for earthquake and "
            "tsunami-risk questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "min_magnitude": {"type": "number", "description": "Min magnitude, default 5.0"},
                "days_back":     {"type": "integer", "description": "Days back, 1-30, default 7"},
                "region":        {"type": "string", "description": "Optional place-name filter, e.g. 'Philippines'"},
            },
            "required": [],
        },
    },
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def get_earthquakes_global(min_magnitude: float = 5.0, days_back: int = 7, region: str = "") -> str:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=min(30, max(1, days_back)))

    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime":   end.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitude,
        "orderby": "magnitude",
        "limit": 25,
    }

    try:
        resp = requests.get(USGS_API, params=params, timeout=10)
        resp.raise_for_status()
        feats = resp.json().get("features", [])
    except Exception as e:
        return f"[USGS] שגיאה: {e}"

    if region:
        feats = [f for f in feats if region.lower() in f["properties"].get("place","").lower()]

    if not feats:
        loc = f" ב{region}" if region else ""
        return f"[USGS] אין רעידות ≥M{min_magnitude}{loc} ב-{days_back} הימים האחרונים."

    lines = [f"🫨 USGS — רעידות אדמה ({len(feats)} נמצאו):"]
    for f in feats[:6]:
        p, coords = f["properties"], f["geometry"]["coordinates"]
        mag, place = p.get("mag","?"), p.get("place","?")
        depth = coords[2] if len(coords) > 2 else None
        dt = datetime.fromtimestamp(p["time"]/1000, tz=timezone.utc).strftime("%d/%m %H:%M")
        tsunami = ""
        if isinstance(mag,(int,float)) and isinstance(depth,(int,float)) and mag>=6.5 and depth<70:
            tsunami = " ⚠️ סיכון צונאמי"
        lines.append(f"  M{mag} | {place} | עומק {depth}ק\"מ | {dt}{tsunami}")
    return "\n".join(lines)
