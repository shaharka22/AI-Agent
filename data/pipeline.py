"""
data/pipeline.py
─────────────────────────────────────────────────────────────────────────────
שלב 1: ניקוי ועיבוד מקדים (Preprocessing)
שלב 2: בניית corpus מאוחד לשימוש ב-RAG (Cosine Similarity search)
שלב 3: חישוב features ל-ML (Clustering + Anomaly Detection)

הרצה חד-פעמית לפני העלאת הסוכן:
    python -m data.pipeline

קלט (raw):
    data/raw/israel_events_filtered_10k.csv   — 10,000 אירועי ישראל (מסונן)
    data/raw/global_disasters_filtered.csv    — 10,500 אסונות גלובליים (מסונן)
    data/raw/climate_anomaly_events.csv       — 52 אירועים עם ניתוח אקלים מפורט

פלט (processed):
    data/processed/rag_corpus.json   — corpus מאוחד + embeddings מוכן ל-FAISS
    data/processed/faiss_index.bin   — אינדקס FAISS בנוי
─────────────────────────────────────────────────────────────────────────────
"""
import os
import re
import json
import numpy as np
import pandas as pd

from config import (
    ISRAEL_EVENTS_CSV, GLOBAL_EVENTS_CSV, CLIMATE_ANOMALY_CSV,
    PROCESSED_DIR, RAG_CORPUS_JSON,
)


# ─── Step 1: Load & clean Israel events ──────────────────────────────────────
def load_israel_events(sample_n: int = 600) -> pd.DataFrame:
    """
    טוען את מסד אירועי ישראל (10k שורות), מנקה ודוגם תת-קבוצה מייצגת.
    אנחנו לא מטמיעים את כל 10,000 השורות ב-FAISS (עלות embeddings מיותרת) —
    אלא דוגמים מדגם מייצג + שומרים את הסט המלא ל-pandas analytics (חיפוש מדויק).
    """
    df = pd.read_csv(ISRAEL_EVENTS_CSV)

    # ניקוי: הסרת כפילויות מדויקות
    df = df.drop_duplicates()

    # ניקוי טיפוסים
    df["Earthquake_Magnitude"] = pd.to_numeric(df["Earthquake_Magnitude"], errors="coerce")
    df["Temp_Diff"]            = pd.to_numeric(df["Temp_Diff"], errors="coerce")
    df["Available_Fire_Trucks"]= pd.to_numeric(df["Available_Fire_Trucks"], errors="coerce")
    df["Available_Ambulances"] = pd.to_numeric(df["Available_Ambulances"], errors="coerce")

    # חילוץ דונם משדה טקסט חופשי "204 דונם" → 204
    df["Fire_Extent_Dunam"] = df["Fire_Extent"].apply(_extract_dunam)

    # מילוי חסרים
    df["Earthquake_Magnitude"] = df["Earthquake_Magnitude"].fillna(0.0)
    df["Fire_Extent_Dunam"]    = df["Fire_Extent_Dunam"].fillna(0.0)

    # מדגם מייצג מאוזן — לפי Disaster_Type ו-City
    parts = []
    for _, g in df.groupby(["Disaster_Type", "City"]):
        parts.append(g.sample(min(len(g), max(1, sample_n // 10)), random_state=42))
    sampled = pd.concat(parts, ignore_index=True)
    return sampled


def _extract_dunam(text) -> float:
    if not isinstance(text, str):
        return 0.0
    m = re.search(r"(\d+)", text)
    return float(m.group(1)) if m else 0.0


# ─── Step 2: Load & clean global disasters ───────────────────────────────────
def load_global_events(sample_n: int = 400) -> pd.DataFrame:
    """טוען אסונות גלובליים מסוננים, מנקה, ודוגם מדגם מייצג לפי סוג+יבשת."""
    df = pd.read_csv(GLOBAL_EVENTS_CSV)
    df = df.drop_duplicates()

    df["Fatalities"]               = pd.to_numeric(df["Fatalities"], errors="coerce").fillna(0)
    df["Evacuated_People"]         = pd.to_numeric(df["Evacuated_People"], errors="coerce").fillna(0)
    df["Economic_Damage_Millions"] = pd.to_numeric(df["Economic_Damage_Millions"], errors="coerce").fillna(0)
    df["Affected_Area_sqkm"]       = pd.to_numeric(df["Affected_Area_sqkm"], errors="coerce").fillna(0)

    parts = []
    for _, g in df.groupby(["Disaster_Type", "Continent"]):
        parts.append(g.sample(min(len(g), max(1, sample_n // 10)), random_state=42))
    sampled = pd.concat(parts, ignore_index=True)
    return sampled


# ─── Step 3: Load climate anomaly events (small, high-detail) ───────────────
def load_climate_events() -> pd.DataFrame:
    """52 אירועים עם פירוט אקלימי מלא — נכנסים כולם ל-corpus (כבר קטן)."""
    df = pd.read_csv(CLIMATE_ANOMALY_CSV)
    df = df.drop_duplicates()
    return df


# ─── Step 4: Build unified RAG corpus ─────────────────────────────────────────
def build_corpus_text(israel_df: pd.DataFrame, global_df: pd.DataFrame,
                       climate_df: pd.DataFrame) -> list[dict]:
    """
    ממיר כל שורה לרשומת corpus אחידה:
    {id, source, title, text, type, country_or_city, lat?, lon?, raw}
    'text' הוא המחרוזת שתוטמע (embed) לחיפוש סמנטי.
    """
    corpus = []

    for i, row in israel_df.iterrows():
        text = (
            f"{row['Disaster_Type']} ב{row['City']}, {row['Specific_Location']}. "
            f"צפיפות אוכלוסין: {row['Pop_Density']}. תשתיות: {row['Infrastructure']}. "
            f"הפרש טמפ': {row['Temp_Diff']}°C. תוכנית פעולה: {row['Action_Plan']}."
        )
        corpus.append({
            "id": f"IL-{i}",
            "source": "israel_local",
            "title": f"{row['Disaster_Type']} — {row['City']}",
            "text": text,
            "type": "EQ" if row["Disaster_Type"] == "רעידת אדמה" else "WF",
            "region": row["City"],
            "date": row["Date"],
            "raw": {
                "city": row["City"], "location": row["Specific_Location"],
                "magnitude": row["Earthquake_Magnitude"], "fire_dunam": row["Fire_Extent_Dunam"],
                "pop_density": row["Pop_Density"], "infrastructure": row["Infrastructure"],
                "temp_diff": row["Temp_Diff"],
                "fire_station": row["Fire_Station"], "fire_trucks": row["Available_Fire_Trucks"],
                "mda_address": row["MDA_Address"], "ambulances": row["Available_Ambulances"],
                "evacuation_center": row["Evacuation_Center"], "action_plan": row["Action_Plan"],
            },
        })

    for i, row in global_df.iterrows():
        text = (
            f"{row['Disaster_Type']} ב{row['Country']} ({row['Continent']}). "
            f"עוצמה: {row['Magnitude']}. שטח מושפע: {row['Affected_Area_sqkm']} קמ\"ר. "
            f"נפגעים: {int(row['Fatalities'])}. מפונים: {int(row['Evacuated_People'])}. "
            f"נזק כלכלי: {row['Economic_Damage_Millions']}M$. רמת התרעה: {row['Alert_Level']}."
        )
        corpus.append({
            "id": f"GL-{i}",
            "source": "global",
            "title": f"{row['Disaster_Type']} — {row['Country']}",
            "text": text,
            "type": _map_global_type(row["Disaster_Type"]),
            "region": row["Country"],
            "date": row["Date"],
            "raw": {
                "country": row["Country"], "continent": row["Continent"],
                "magnitude": row["Magnitude"], "area_sqkm": row["Affected_Area_sqkm"],
                "fatalities": int(row["Fatalities"]), "evacuated": int(row["Evacuated_People"]),
                "economic_damage_m": row["Economic_Damage_Millions"],
                "alert_level": row["Alert_Level"],
                "intl_aid": row["International_Aid_Required"],
            },
        })

    for i, row in climate_df.iterrows():
        text = (
            f"{row['סוג אירוע']} ב{row['עיר']}, {row['מיקום מדויק']}. "
            f"היקף: {row['היקף האירוע']}. {row['אזור מאוכלס?']}. {row['אזור עם תשתיות?']}. "
            f"מזג אוויר בשגרה: {row['מזג אוויר בשגרה']}. בעת האירוע: {row['מזג אוויר בעת האירוע']}. "
            f"חריגה: {row['חריגות טמפרטורה (אירוע מול שגרה)']}."
        )
        corpus.append({
            "id": f"CL-{i}",
            "source": "climate_anomaly",
            "title": f"{row['סוג אירוע']} — {row['עיר']} (ניתוח אקלים)",
            "text": text,
            "type": "EQ" if row["סוג אירוע"] == "רעידת אדמה" else "WF",
            "region": row["עיר"],
            "date": str(row["תאריך"]),
            "raw": {
                "city": row["עיר"], "location": row["מיקום מדויק"],
                "extent": row["היקף האירוע"],
                "populated": row["אזור מאוכלס?"], "infrastructure": row["אזור עם תשתיות?"],
                "fire_station": row["כתובת מדויקת - תחנת כיבוי אש"],
                "mda_address": row["כתובת מדויקת - תחנת מד״א"],
                "normal_weather": row["מזג אוויר בשגרה"],
                "event_weather": row["מזג אוויר בעת האירוע"],
                "anomaly": row["חריגות טמפרטורה (אירוע מול שגרה)"],
            },
        })

    return corpus


def _map_global_type(hebrew_type: str) -> str:
    mapping = {
        "הוריקן": "TC", "צונאמי": "TS", "שריפת יער": "WF",
        "שיטפון": "FL", "רעידת אדמה": "EQ",
    }
    return mapping.get(hebrew_type, "OTHER")


# ─── Step 5: Embed + build FAISS index ───────────────────────────────────────
def embed_and_index(corpus: list[dict]):
    """מטמיע (embeds) את ה-corpus ובונה אינדקס FAISS. דורש GEMINI_API_KEY."""
    import time
    import faiss
    from google import genai
    from google.genai import types
    from config import GEMINI_API_KEY, EMBED_MODEL, EMBED_DIM, FAISS_INDEX_PATH

    client = genai.Client(api_key=GEMINI_API_KEY)
    texts  = [c["text"] for c in corpus]
    cfg    = types.EmbedContentConfig(output_dimensionality=EMBED_DIM)

    print(f"Embedding {len(texts)} corpus entries (batches of 25, Gemini {EMBED_MODEL})...")
    vectors = []
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for attempt in range(5):
            try:
                resp = client.models.embed_content(model=EMBED_MODEL, contents=batch, config=cfg)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  retry batch {i}: {e} (waiting {wait}s)")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to embed batch starting at {i}")
        vectors.extend([e.values for e in resp.embeddings])
        print(f"  embedded {min(i+batch_size, len(texts))}/{len(texts)}")

    vecs = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(RAG_CORPUS_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ Saved {len(corpus)} records → {RAG_CORPUS_JSON}")
    print(f"✅ Saved FAISS index → {FAISS_INDEX_PATH}")


# ─── Main pipeline ────────────────────────────────────────────────────────────
def run_pipeline(israel_sample=600, global_sample=400, build_embeddings=True):
    print("=" * 60)
    print("DisasterGuard — Data Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading & cleaning Israel events (10k → sample)...")
    israel_df = load_israel_events(sample_n=israel_sample)
    print(f"  → {len(israel_df)} sampled records")

    print("\n[2/4] Loading & cleaning global disasters (10.5k → sample)...")
    global_df = load_global_events(sample_n=global_sample)
    print(f"  → {len(global_df)} sampled records")

    print("\n[3/4] Loading climate anomaly events (52 records, full)...")
    climate_df = load_climate_events()
    print(f"  → {len(climate_df)} records")

    print("\n[4/4] Building unified RAG corpus...")
    corpus = build_corpus_text(israel_df, global_df, climate_df)
    print(f"  → {len(corpus)} total corpus entries")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if build_embeddings:
        embed_and_index(corpus)
    else:
        with open(RAG_CORPUS_JSON, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2, default=str)
        print(f"⚠️ Embeddings skipped (no API key) — saved corpus JSON only.")

    print("\n✅ Pipeline complete.")
    return corpus


if __name__ == "__main__":
    from config import GEMINI_API_KEY
    run_pipeline(build_embeddings=bool(GEMINI_API_KEY))
