"""
tools/analytics_tool.py
─────────────────────────────────────────────────────────────────────────────
שאלות עובדתיות מדויקות ("כמה אסונות היו ב-2020?", "אילו אסונות בפיליפינים?")
לא מתאימות לחיפוש סמנטי (RAG) — דרושה שאילתה מדויקת. כלי זה מריץ שאילתות
pandas ישירות על הדאטה-סטים המלאים (10,500 + 14,000 + 10,000 רשומות).
─────────────────────────────────────────────────────────────────────────────
"""
import pandas as pd
from config import GLOBAL_EVENTS_CSV, ISRAEL_EVENTS_CSV

_global_df = None
_israel_df = None


def _load():
    global _global_df, _israel_df
    if _global_df is None:
        _global_df = pd.read_csv(GLOBAL_EVENTS_CSV)
        _global_df["Date_parsed"] = pd.to_datetime(_global_df["Date"], format="%d/%m/%Y", errors="coerce")
    if _israel_df is None:
        _israel_df = pd.read_csv(ISRAEL_EVENTS_CSV)
        _israel_df["Date_parsed"] = pd.to_datetime(_israel_df["Date"], errors="coerce")
    return _global_df, _israel_df


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "query_disaster_statistics",
        "description": (
            "Run precise statistical queries on the full disaster datasets (10,500 global "
            "+ 10,000 Israeli records). Use for exact counts, filters by year/country/type, "
            "e.g. 'how many disasters in 2020', 'earthquakes in Philippines', "
            "'fires in Israel this year'. NOT for semantic/similarity questions — use "
            "search_historical_and_analyze for those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "enum": ["global", "israel"]},
                "year": {"type": "integer", "description": "Filter by year, optional"},
                "country": {"type": "string", "description": "Filter by country (global dataset only)"},
                "disaster_type": {"type": "string", "description": "Filter by disaster type in Hebrew, e.g. 'רעידת אדמה', 'שריפת יער'"},
                "city": {"type": "string", "description": "Filter by city (israel dataset only)"},
            },
            "required": ["dataset"],
        },
    },
}


def query_disaster_statistics(
    dataset: str, year: int = None, country: str = "",
    disaster_type: str = "", city: str = "",
) -> str:
    global_df, israel_df = _load()
    df = global_df if dataset == "global" else israel_df

    if year:
        df = df[df["Date_parsed"].dt.year == year]
    if country and dataset == "global":
        df = df[df["Country"].str.contains(country, na=False)]
    if city and dataset == "israel":
        df = df[df["City"].str.contains(city, na=False)]
    if disaster_type:
        df = df[df["Disaster_Type"].str.contains(disaster_type, na=False)]

    if len(df) == 0:
        return "[סטטיסטיקה] לא נמצאו רשומות תואמות."

    lines = [f"📊 נמצאו {len(df)} רשומות תואמות."]

    if dataset == "global":
        lines.append(f"לפי סוג: {df['Disaster_Type'].value_counts().head(5).to_dict()}")
        if "Fatalities" in df.columns:
            lines.append(f"סה\"כ נפגעים: {int(df['Fatalities'].sum()):,}")
        if "Alert_Level" in df.columns:
            lines.append(f"רמות התרעה: {df['Alert_Level'].value_counts().to_dict()}")
        # Sample 2 records
        for _, r in df.head(2).iterrows():
            lines.append(f"  • {r['Date']} | {r['Country']} | {r['Disaster_Type']} | {r.get('Alert_Level','')}")
    else:
        lines.append(f"לפי סוג: {df['Disaster_Type'].value_counts().to_dict()}")
        lines.append(f"לפי עיר: {df['City'].value_counts().head(5).to_dict()}")
        for _, r in df.head(2).iterrows():
            lines.append(f"  • {r['Date']} | {r['City']} | {r['Disaster_Type']} | {r.get('Action_Plan','')}")

    return "\n".join(lines)
