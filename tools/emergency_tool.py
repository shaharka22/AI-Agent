"""
tools/emergency_tool.py
─────────────────────────────────────────────────────────────────────────────
מספרי חירום ארציים וערוצי עדכונים רשמיים בישראל (כבאות והצלה, מד"א, פיקוד העורף).
מידע סטטי, לא תלוי-מיקום — לשם זה יש את כתובות תחנת הכיבוי/מד"א הספציפיות
בדאטה (Fire_Station / MDA_Address) שמוחזרות מ-search_historical_and_analyze
ומ-query_disaster_statistics.
─────────────────────────────────────────────────────────────────────────────
"""
from config import ISRAEL_EMERGENCY_HOTLINES

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_israel_emergency_contacts",
        "description": (
            "Return national Israeli emergency hotlines and official update channels: "
            "Fire & Rescue Authority (102), Magen David Adom/MDA (101), Police (100), "
            "Home Front Command (104). Use when the user asks for emergency numbers, "
            "rescue service contacts, or official alert channels in Israel."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def get_israel_emergency_contacts() -> str:
    lines = ["📞 מספרי חירום וערוצי עדכון רשמיים בישראל:"]
    for name, value in ISRAEL_EMERGENCY_HOTLINES.items():
        lines.append(f"  • {name}: {value}")
    return "\n".join(lines)
