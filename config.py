"""
config.py — הגדרות מרכזיות לכל הפרויקט
─────────────────────────────────────────────────────────────────────────────
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ─── קבצי דאטה ────────────────────────────────────────────────────────────────
ISRAEL_EVENTS_CSV            = str(RAW_DIR / "israel_events_filtered_10k.csv")
ISRAEL_EVENTS_UNFILTERED_CSV = str(RAW_DIR / "israel_events_unfiltered_15k.csv")
GLOBAL_EVENTS_CSV            = str(RAW_DIR / "global_disasters_filtered.csv")
GLOBAL_EVENTS_UNFILTERED_CSV = str(RAW_DIR / "global_disasters_full.csv")
CLIMATE_ANOMALY_CSV          = str(RAW_DIR / "climate_anomaly_events.csv")

RAG_CORPUS_JSON  = str(PROCESSED_DIR / "rag_corpus.json")
FAISS_INDEX_PATH = str(PROCESSED_DIR / "faiss_index.bin")

# ─── מפתחות API ──────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

# ─── מודלים (Gemini) ──────────────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL  = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM    = 768  # output_dimensionality מבוקש מ-gemini-embedding-001

# ─── היפר-פרמטרים ל-ML ────────────────────────────────────────────────────────
N_RISK_CLUSTERS = 4      # K-Means: נמוך / בינוני / גבוה / קריטי
ANOMALY_CONTAM  = 0.05   # Isolation Forest: שיעור חריגות צפוי

# ─── APIs חיים (אסונות טבע) ───────────────────────────────────────────────────
GDACS_API      = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
USGS_API       = "https://earthquake.usgs.gov/fdsnws/event/1/query"
NASA_EONET_API = "https://eonet.gsfc.nasa.gov/api/v3/events"

# ─── מספרי חירום ושירותי כבאות והצלה / מד"א בישראל (סטטי, ארצי) ───────────────
ISRAEL_EMERGENCY_HOTLINES = {
    "כבאות והצלה לישראל":        "102",
    "מגן דוד אדום (מד\"א)":      "101",
    "משטרת ישראל":               "100",
    "פיקוד העורף (מודיעין שוטף)": "104",
    "אתר כבאות והצלה לישראל":     "https://www.102.gov.il",
    "אתר מד\"א":                  "https://www.mda.org.il",
    "אפליקציית פיקוד העורף":      "https://www.oref.org.il",
}
