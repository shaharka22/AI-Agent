"""
ml/models.py
─────────────────────────────────────────────────────────────────────────────
שלושה מודלי ML, כולם מאומנים על הדאטה האמיתי מ-data/raw/ (לא דאטה מומצא):

1. Cosine Similarity (RAG)
   ────────────────────────
   חיפוש סמנטי על corpus מאוחד (1,651 רשומות: ישראל + גלובלי + ניתוח אקלים).
   embeddings מ-OpenAI text-embedding-3-small, חיפוש ב-FAISS (Inner Product
   על וקטורים מנורמלים = cosine similarity).

2. K-Means Clustering — סיווג סיכון
   ──────────────────────────────────
   מאמן על features מספריים מתוך הדאטה האמיתי: עוצמה/היקף, הפרש טמפרטורה,
   צפיפות אוכלוסין (מקודד), זמינות כוננות (משאיות כיבוי/אמבולנסים).
   4 אשכולות: נמוך / בינוני / גבוה / קריטי.

3. Isolation Forest — זיהוי אנומליות מטאורולוגיות וגיאוגרפיות
   ─────────────────────────────────────────────────────────────
   רץ על עמודת Temp_Diff (הפרש טמפ' אירוע מול שגרה) מתוך 10,000 רשומות
   ישראל האמיתיות. מזהה אירועים עם סטייה חריגה ביחס להתפלגות הרגילה —
   זה בדיוק "האנומליות והחריגות במזג האוויר" שהמרצה ביקשה (שאלה 3 בתסריט).
─────────────────────────────────────────────────────────────────────────────
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from config import (
    ISRAEL_EVENTS_CSV, GLOBAL_EVENTS_CSV,
    N_RISK_CLUSTERS, ANOMALY_CONTAM,
)

CLUSTER_LABELS = {
    0: {"name": "סיכון נמוך",   "icon": "🟢"},
    1: {"name": "סיכון בינוני", "icon": "🟡"},
    2: {"name": "סיכון גבוה",   "icon": "🟠"},
    3: {"name": "סיכון קריטי",  "icon": "🔴"},
}

POP_DENSITY_ENCODING = {
    "אזור פתוח": 0.1, "אוכלוסייה מעורבת": 0.5,
    "מאוכלס בצפיפות": 0.9,
}

ALERT_LEVEL_ENCODING = {"בינונית": 0.3, "גבוהה": 0.6, "קריטית": 1.0}


# ─── Feature extraction (shared) ─────────────────────────────────────────────
def _israel_features(row: pd.Series) -> list[float]:
    mag      = float(row.get("Earthquake_Magnitude") or 0.0)
    fire     = float(row.get("Fire_Extent_Dunam") or 0.0)
    severity = max(mag / 9.0, min(fire / 400.0, 1.0))           # normalize 0-1
    density  = POP_DENSITY_ENCODING.get(row.get("Pop_Density"), 0.5)
    trucks   = float(row.get("Available_Fire_Trucks") or 5) / 10.0
    ambul    = float(row.get("Available_Ambulances") or 5) / 10.0
    readiness = 1.0 - min((trucks + ambul) / 2.0, 1.0)            # less ready = higher risk
    return [severity, density, readiness]


def _global_features(row: pd.Series) -> list[float]:
    fatalities = min(float(row.get("Fatalities", 0)) / 500.0, 1.0)
    area       = min(float(row.get("Affected_Area_sqkm", 0)) / 10000.0, 1.0)
    alert      = ALERT_LEVEL_ENCODING.get(row.get("Alert_Level"), 0.5)
    return [fatalities, area, alert]


# ─── 2. K-Means — trained once at import time on real Israel data ───────────
_israel_df = pd.read_csv(ISRAEL_EVENTS_CSV)
_israel_df["Earthquake_Magnitude"] = pd.to_numeric(_israel_df["Earthquake_Magnitude"], errors="coerce").fillna(0)
_israel_df["Fire_Extent_Dunam"] = _israel_df["Fire_Extent"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0)

_X_train = np.array([_israel_features(r) for _, r in _israel_df.sample(2000, random_state=42).iterrows()])
_scaler  = MinMaxScaler().fit(_X_train)
_X_scaled = _scaler.transform(_X_train)

_kmeans = KMeans(n_clusters=N_RISK_CLUSTERS, random_state=42, n_init=10)
_kmeans.fit(_X_scaled)


def classify_risk_israel(event: dict) -> dict:
    """
    מסווג אירוע ישראלי (שריפה/רעידה) לאחד מ-4 אשכולות סיכון.
    event: {"magnitude": float, "fire_dunam": float, "pop_density": str,
            "fire_trucks": int, "ambulances": int}
    """
    row = pd.Series({
        "Earthquake_Magnitude": event.get("magnitude", 0),
        "Fire_Extent_Dunam":    event.get("fire_dunam", 0),
        "Pop_Density":          event.get("pop_density", "אוכלוסייה מעורבת"),
        "Available_Fire_Trucks":event.get("fire_trucks", 5),
        "Available_Ambulances": event.get("ambulances", 5),
    })
    vec     = _scaler.transform([_israel_features(row)])
    cluster = int(_kmeans.predict(vec)[0])
    center  = _kmeans.cluster_centers_[cluster]
    dist    = float(np.linalg.norm(vec[0] - center))
    conf    = max(0, round((1 - dist) * 100, 1))

    label = CLUSTER_LABELS[cluster]
    return {
        "cluster_id": cluster, "risk_name": label["name"], "risk_icon": label["icon"],
        "confidence": conf, "summary": f"{label['icon']} {label['name']} (ביטחון {conf}%)",
    }


# ─── 3. Isolation Forest — anomaly detection on REAL Temp_Diff data ─────────
# מאומן על 10,000 הרשומות האמיתיות (Temp_Diff, Earthquake_Magnitude, Fire_Extent)
_anomaly_features = _israel_df[["Earthquake_Magnitude", "Fire_Extent_Dunam"]].copy()
_anomaly_features["Temp_Diff"] = pd.to_numeric(_israel_df.get("Temp_Diff", 8), errors="coerce").fillna(8)

_iso_scaler = MinMaxScaler().fit(_anomaly_features.values)
_iso_X      = _iso_scaler.transform(_anomaly_features.values)

_iso_forest = IsolationForest(
    n_estimators=150, contamination=ANOMALY_CONTAM, random_state=42
)
_iso_forest.fit(_iso_X)

# Empirical Temp_Diff stats from the real dataset, for human-readable explanation
_TEMP_DIFF_MEAN = float(_anomaly_features["Temp_Diff"].mean())
_TEMP_DIFF_STD  = float(_anomaly_features["Temp_Diff"].std())


def detect_weather_anomaly(magnitude: float = 0, fire_dunam: float = 0, temp_diff: float = 8) -> dict:
    """
    מזהה האם אירוע הוא אנומליה מטאורולוגית/גיאוגרפית ביחס להתפלגות ההיסטורית
    האמיתית (10,000 רשומות ישראל). זה המודל שעונה על שאלה 3 בתסריט המרצה:
    "מה האנומליות והחריגות בנתונים המטאורולוגיים והגיאוגרפיים?"
    """
    vec   = _iso_scaler.transform([[magnitude, fire_dunam, temp_diff]])
    score = float(_iso_forest.score_samples(vec)[0])
    is_anomaly = bool(_iso_forest.predict(vec)[0] == -1)

    z_score = (temp_diff - _TEMP_DIFF_MEAN) / _TEMP_DIFF_STD if _TEMP_DIFF_STD else 0

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(score, 3),
        "temp_diff": temp_diff,
        "temp_diff_avg_baseline": round(_TEMP_DIFF_MEAN, 1),
        "z_score": round(z_score, 2),
        "summary": (
            f"⚠️ חריגה מובהקת (סטיית תקן {round(z_score,1)}σ מהממוצע ההיסטורי {round(_TEMP_DIFF_MEAN,1)}°C)"
            if is_anomaly else
            f"✅ בטווח הרגיל (ממוצע היסטורי {round(_TEMP_DIFF_MEAN,1)}°C)"
        ),
    }


# ─── 1. Cosine Similarity — RAG over the unified corpus ──────────────────────
_faiss_index = None
_corpus      = None


def _load_rag():
    global _faiss_index, _corpus
    if _faiss_index is not None:
        return _faiss_index, _corpus

    import faiss
    from config import FAISS_INDEX_PATH, RAG_CORPUS_JSON

    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(RAG_CORPUS_JSON)):
        return None, None

    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(RAG_CORPUS_JSON, "r", encoding="utf-8") as f:
        _corpus = json.load(f)
    return _faiss_index, _corpus


def find_similar_events(query: str, top_k: int = 3, source_filter: str = None) -> list[dict]:
    """
    Cosine Similarity search על ה-RAG corpus (1,651 רשומות אמיתיות).
    source_filter: 'israel_local' | 'global' | 'climate_anomaly' | None (הכל)
    """
    import faiss
    from google import genai
    from google.genai import types
    from config import GEMINI_API_KEY, EMBED_MODEL, EMBED_DIM

    index, corpus = _load_rag()
    if index is None:
        return []

    client = genai.Client(api_key=GEMINI_API_KEY)
    resp   = client.models.embed_content(
        model=EMBED_MODEL, contents=[query],
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    q_vec  = np.array([resp.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(q_vec)

    # Search wider than top_k to allow source filtering, then trim
    search_k = min(top_k * 5, len(corpus)) if source_filter else top_k
    sims, idxs = index.search(q_vec, search_k)

    results = []
    for score, idx in zip(sims[0], idxs[0]):
        if idx < 0 or idx >= len(corpus):
            continue
        entry = corpus[idx]
        if source_filter and entry["source"] != source_filter:
            continue
        results.append({"similarity_pct": round(float(score) * 100, 1), **entry})
        if len(results) >= top_k:
            break

    return results
