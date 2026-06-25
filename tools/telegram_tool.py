"""
tools/telegram_tool.py
─────────────────────────────────────────────────────────────────────────────
שליחת התראת חירום לקבוצת Telegram של DisasterGuard.
מופעל כשאזרח מדווח שלא קיבל מענה ממוקד 102 או 101.
הסוכן שולח הודעה אוטומטית עם כל פרטי האירוע לקבוצת הכוננות.
─────────────────────────────────────────────────────────────────────────────
"""
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "send_emergency_alert",
        "description": (
            "Send an emergency alert to the DisasterGuard Telegram group when a citizen "
            "reports they cannot reach emergency services (102/101). Use this when the user "
            "says 'אין מענה', 'לא עונים', 'לא מצליח להתקשר', or similar. "
            "Sends all incident details automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "תפקיד המדווח — אזרח/כבאי/חובש"},
                "location": {"type": "string", "description": "מיקום האירוע"},
                "incident_type": {"type": "string", "description": "סוג האירוע — שריפה/רעידת אדמה/אחר"},
                "status": {"type": "string", "description": "תיאור קצר של המצב הנוכחי"},
            },
            "required": ["location", "incident_type"],
        },
    },
}


def send_emergency_alert(
    location: str,
    incident_type: str,
    role: str = "לא ידוע",
    status: str = "לא פורט",
) -> str:
    message = (
        f"🚨 *התראת חירום — DisasterGuard*\n\n"
        f"🔴 *סוג אירוע:* {incident_type}\n"
        f"📍 *מיקום:* {location}\n"
        f"👤 *מדווח:* {role}\n"
        f"📋 *מצב:* {status}\n\n"
        f"⚠️ המדווח לא הצליח לקבל מענה ממוקד החירום — נדרשת התערבות!"
    )

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return "✅ התראת חירום נשלחה לקבוצת הכוננות ב-Telegram."
    except Exception as e:
        return f"❌ שגיאה בשליחת ההתראה: {e}"
