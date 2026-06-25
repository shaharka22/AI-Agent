# 🌍 DisasterGuard Agent

סוכן AI לניהול משברים וניטור אסונות טבע בזמן אמת — משלב מקורות חיים (GDACS/USGS/NASA) עם מנוע RAG היסטורי ושני מודלי ML, מאומנים על דאטה אמיתי. מנוע ה-LLM: **Google Gemini** (gemini-2.5-flash + gemini-embedding-001), נגיש בממשק צ'אט בדפדפן (Flask).

## מקורות הדאטה (כולם בתיקיית `data/raw/`)

| קובץ | תיאור | שורות |
|------|-------|-------|
| `israel_events_filtered_10k.csv` | שריפות + רעידות אדמה בישראל, מסונן | 10,000 |
| `israel_events_unfiltered_15k.csv` | אותו דאטה לפני סינון (לתיעוד תהליך הניקוי) | 15,000 |
| `global_disasters_filtered.csv` | אסונות גלובליים מסוננים (5 סוגים) | 10,500 |
| `global_disasters_full.csv` | אסונות גלובליים לפני סינון | 14,000 |
| `climate_anomaly_events.csv` | 52 אירועים עם ניתוח אקלים מפורט (חריגות טמפ') | 52 |

## ארכיטקטורה

```
dg_final/
├── main.py              # Flask Web Chat — entry point
├── templates/
│   └── chat.html          # ממשק צ'אט בדפדפן (HTML/JS, RTL)
├── agent.py              # ReAct loop — Gemini 2.5 Flash + Function Calling
├── config.py              # הגדרות מרכזיות (מפתחות, מודלים, נתיבים, APIs)
├── data/
│   ├── raw/               # 5 קבצי ה-CSV המקוריים
│   ├── processed/          # corpus מעובד + אינדקס FAISS (נבנה ע"י pipeline.py)
│   └── pipeline.py         # ניקוי + עיבוד + embeddings (Gemini) — מריצים פעם אחת
├── ml/
│   └── models.py           # Cosine Similarity (RAG) + K-Means + Isolation Forest
├── tools/
│   ├── gdacs_tool.py        # GDACS — חמ"ל עולמי (UN/EU)
│   ├── usgs_tool.py         # USGS — רעידות אדמה גלובליות חיות
│   ├── nasa_tool.py         # NASA EONET — לוויינים
│   ├── rag_tool.py          # חיפוש סמנטי + ניתוח ML משולב
│   ├── analytics_tool.py    # שאילתות סטטיסטיות מדויקות (pandas)
│   └── emergency_tool.py    # מספרי חירום וערוצי עדכון רשמיים בישראל
└── render.yaml             # פריסה אוטומטית
```

## שלושת מודלי ה-ML (לפי דרישת המחוון)

1. **Cosine Similarity** — חיפוש סמנטי על corpus מאוחד של 1,651 אירועים אמיתיים (ישראל + גלובלי + ניתוח אקלים), embeddings מ-Gemini (`gemini-embedding-001`), FAISS לחיפוש מהיר.
2. **K-Means Clustering** — מסווג אירועים ל-4 רמות סיכון (נמוך/בינוני/גבוה/קריטי), מאומן על features אמיתיים מתוך 10,000 רשומות ישראל (עוצמה, צפיפות אוכלוסין, זמינות כוננות).
3. **Isolation Forest** — מזהה אנומליות מטאורולוגיות: רץ על עמודת `Temp_Diff` האמיתית מהדאטה (הפרש טמפ' אירוע מול שגרה), מזהה סטיות חריגות מההתפלגות ההיסטורית.

---

## הפעלה — שלב אחר שלב

### 1. התקן תלויות
```bash
pip install -r requirements.txt
```

### 2. הגדר מפתחות
```bash
cp .env.example .env
# מלא: GEMINI_API_KEY
```

### 3. בנה את ה-RAG corpus (חובה לפני הפעלה ראשונה!)
```bash
python -m data.pipeline
```
זה מריץ ניקוי על כל קבצי ה-CSV, בונה corpus מאוחד (כ-1,650 רשומות), ושולח אותן ל-Gemini embeddings (`gemini-embedding-001`, batches של 25) → שומר אינדקס FAISS תחת `data/processed/`.
⚠️ דורש `GEMINI_API_KEY` תקין. תהליך מלא לוקח כ-2-3 דקות.

### 4. הפעל
```bash
python main.py
```
פתח דפדפן בכתובת `http://localhost:10000` — שם נמצא ממשק הצ'אט.

---

## פריסה ב-Render

1. **גיט:**
```bash
git init && git add . && git commit -m "DisasterGuard Agent"
git remote add origin https://github.com/YOUR_USER/disaster-guard-agent.git
git push -u origin main
```

2. **Render:** New → Web Service → Connect repo
   - Build Command: `pip install -r requirements.txt && python -m data.pipeline`
   - Start Command: `python main.py`
   - **הגדר Environment Variables:** `GEMINI_API_KEY`
   - השירות פותח פורט HTTP (Flask), כנדרש ב-Web Service של Render — אין צורך בהגדרות נוספות.

3. Deploy — ה-build command מריץ אוטומטית את pipeline.py כדי לבנות את ה-RAG index בענן.

> **שים לב:** קובץ `data/processed/faiss_index.bin` לא נכלל ב-git (ב-`.gitignore`) כי הוא נבנה אוטומטית ב-build. `rag_corpus.json` כן נכלל לעיון.

---

## בדיקה — שאלות הדגמה (מהתסריט)

1. "מה קרה בשריפות הענק באוסטרליה בשנת 2020? תציג ניתוח והמלצות"
2. "האם יש כרגע רעידות אדמה חזקות בעולם?"
3. "מה האנומליות והחריגות בנתונים המטאורולוגיים שזוהו?"
4. "אילו אסונות טבע היו בשנת 2020?"
5. "האם היו רעידות אדמה בפיליפינים לאחרונה?"
6. "מה הסיכון לצונאמי בעקבות רעידות האדמה בפיליפינים?"

---

## מגבלות ידועות

- מסד האסונות הגלובלי (`global_disasters_filtered.csv`) מכיל רק 10 מדינות (הודו, איטליה, טורקיה, יפן, ארה"ב, מקסיקו, אוסטרליה, צ'ילה, אינדונזיה, קנדה) — שאלות על מדינות אחרות (כמו פיליפינים) נופלות אוטומטית ל-USGS החי.
- ה-corpus המוטמע (embedded) הוא מדגם מאוזן של 1,650 מתוך 24,500+ הרשומות הגולמיות, לא הסט המלא — כדי לאזן עלות embeddings מול כיסוי. ניתן להגדיל ב-`data/pipeline.py` (`israel_sample`, `global_sample`).
