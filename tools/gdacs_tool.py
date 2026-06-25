"""
tools/gdacs_tool.py
─────────────────────────────────────────────────────────────────────────────
GDACS — Global Disaster Alert and Coordination System.
שיתוף פעולה של האו"ם והנציבות האירופית. מחשב רמת סיכון (ירוק/כתום/אדום)
על בסיס השפעה על בני אדם ותשתיות. זהו "החמ"ל העולמי" שמוזכר בתסריט המרצה.
─────────────────────────────────────────────────────────────────────────────
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from config import GDACS_API

ALERT_ICONS = {"Red": "🔴", "Orange": "🟠", "Green": "🟢"}
TYPE_MAP = {"EQ": "רעידת אדמה", "FL": "שיטפון", "TC": "ציקלון/הוריקן",
            "WF": "שריפת יער", "VO": "געש", "DR": "בצורת"}

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_gdacs_alerts",
        "description": (
            "Query GDACS (Global Disaster Alert and Coordination System) — a UN/EU "
            "collaboration acting as a global crisis command center. Returns disaster "
            "type, location, date, and computed risk level (Green/Orange/Red) based on "
            "human and infrastructure impact. Use for global disaster overview questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string", "enum": ["EQ","FL","TC","WF","VO","DR","ALL"]},
                "alert_level": {"type": "string", "enum": ["Red","Orange","Green","ALL"]},
            },
            "required": [],
        },
    },
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def get_gdacs_alerts(event_type: str = "ALL", alert_level: str = "ALL") -> str:
    params = {"limit": 15}
    if event_type != "ALL":
        params["eventtype"] = event_type
    if alert_level != "ALL":
        params["alertlevel"] = alert_level

    try:
        resp = requests.get(GDACS_API, params=params, timeout=10)
        resp.raise_for_status()
        feats = resp.json().get("features", []) if resp.text else []
    except Exception as e:
        return f"[GDACS] שגיאה: {e}"

    if not feats:
        return "[GDACS] אין אירועים פעילים התואמים."

    lines = [f"📡 GDACS — חמ\"ל עולמי ({len(feats)} אירועים):"]
    for f in feats[:8]:
        p     = f.get("properties", {})
        icon  = ALERT_ICONS.get(p.get("alertlevel", ""), "⚪")
        etype = TYPE_MAP.get(p.get("eventtype", ""), p.get("eventtype", ""))
        lines.append(
            f"  {icon} {etype} | {p.get('country','')} | "
            f"{p.get('name','')[:40]} | {p.get('fromdate','')[:10]}"
        )
    return "\n".join(lines)
