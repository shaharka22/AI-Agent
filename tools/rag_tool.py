"""
tools/rag_tool.py
─────────────────────────────────────────────────────────────────────────────
מנוע RAG היסטורי + ML — הליבה הייחודית של הסוכן.

ברגע שיש קלט (מה-LLM, מה-GDACS, או משאלת משתמש חופשית) הכלי:
  1. מחפש אירועים דומים ב-corpus המאוחד (1,651 רשומות אמיתיות) — Cosine Similarity
  2. אם רלוונטי לישראל — מסווג סיכון עם K-Means ובודק אנומליית טמפ' עם Isolation Forest
  3. מחזיר תמצית קצרה + דרכי טיפול שחולצו מהיסטוריה (Action_Plan מהדאטה)
─────────────────────────────────────────────────────────────────────────────
"""
from ml.models import find_similar_events, classify_risk_israel, detect_weather_anomaly

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_historical_and_analyze",
        "description": (
            "Search the unified historical disaster database (1,651 real records: Israeli "
            "local events, global disasters, and detailed climate-anomaly events) using "
            "semantic similarity. Returns similar past events, their outcomes, and — for "
            "Israeli events — an ML risk classification and weather-anomaly check. "
            "ALWAYS use this for 'has this happened before', 'what should we do', or "
            "anomaly/deviation questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of the event/question."},
                "source_filter": {
                    "type": "string",
                    "enum": ["israel_local", "global", "climate_anomaly", "ALL"],
                    "description": "Restrict search to one corpus source, or ALL.",
                },
                "top_k": {"type": "integer", "description": "Number of results, default 3"},
            },
            "required": ["query"],
        },
    },
}


def search_historical_and_analyze(query: str, source_filter: str = "ALL", top_k: int = 3) -> str:
    filt = None if source_filter == "ALL" else source_filter
    results = find_similar_events(query, top_k=top_k, source_filter=filt)

    if not results:
        return "[RAG] לא נמצאו אירועים דומים במאגר ההיסטורי (ייתכן שהאינדקס לא נבנה — הרץ data/pipeline.py)."

    lines = [f"📚 אירועים דומים מההיסטוריה ({len(results)} נמצאו):"]
    for r in results:
        lines.append(f"\n• {r['title']} ({r['similarity_pct']}% דמיון) — {r['source']}")
        lines.append(f"  {r['text'][:180]}")

        # ML enrichment for Israeli local events
        if r["source"] in ("israel_local", "climate_anomaly"):
            raw = r.get("raw", {})
            mag    = raw.get("magnitude", 0) or 0
            fire   = raw.get("fire_dunam", 0) or 0
            temp_d = raw.get("temp_diff", 8) or 8

            try:
                risk = classify_risk_israel({
                    "magnitude": float(mag) if mag else 0,
                    "fire_dunam": float(fire) if fire else 0,
                    "pop_density": raw.get("pop_density", "אוכלוסייה מעורבת"),
                    "fire_trucks": 5, "ambulances": 5,
                })
                lines.append(f"  ML סיכון: {risk['summary']}")
            except Exception:
                pass

            try:
                anomaly = detect_weather_anomaly(
                    magnitude=float(mag) if mag else 0,
                    fire_dunam=float(fire) if fire else 0,
                    temp_diff=float(temp_d) if temp_d else 8,
                )
                lines.append(f"  אנומליה: {anomaly['summary']}")
            except Exception:
                pass

            if raw.get("action_plan"):
                lines.append(f"  תוכנית פעולה היסטורית: {raw['action_plan']}")

    return "\n".join(lines)
