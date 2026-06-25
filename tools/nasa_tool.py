"""
tools/nasa_tool.py — NASA EONET (Earth Observatory Natural Event Tracker)
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from config import NASA_EONET_API

ICONS = {"wildfires":"🔥","severeStorms":"⛈️","volcanoes":"🌋","floods":"🌊","drought":"🏜️"}

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_nasa_events",
        "description": (
            "Fetch active natural events from NASA EONET satellites — wildfires, severe "
            "storms, volcanoes, floods. Use for satellite-based global event monitoring."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["wildfires","severeStorms","volcanoes","floods","ALL"]},
                "days": {"type": "integer", "description": "Days back, default 7"},
            },
            "required": [],
        },
    },
}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
def get_nasa_events(category: str = "ALL", days: int = 7) -> str:
    params = {"status": "open", "days": min(30, max(1, days)), "limit": 20}
    if category != "ALL":
        params["category"] = category

    try:
        resp = requests.get(NASA_EONET_API, params=params, timeout=10)
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception as e:
        return f"[NASA EONET] שגיאה: {e}"

    if not events:
        return f"[NASA EONET] אין אירועים פעילים בקטגוריה '{category}'."

    lines = [f"🛰️ NASA EONET — לוויינים ({len(events)} אירועים):"]
    for ev in events[:8]:
        cats = ev.get("categories", [{}])
        icon = ICONS.get(cats[0].get("id",""), "🌐") if cats else "🌐"
        geo  = ev.get("geometry", [])
        date = geo[-1].get("date","")[:10] if geo else "?"
        lines.append(f"  {icon} {ev.get('title','')[:50]} | {date}")
    return "\n".join(lines)
