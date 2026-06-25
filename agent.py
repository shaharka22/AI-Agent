"""
agent.py — DisasterGuard Agent: ReAct loop עם Gemini Function Calling
─────────────────────────────────────────────────────────────────────────────
הסוכן משלב:
  • מקורות חיים גלובליים: GDACS (חמ"ל עולמי UN/EU), USGS (רעידות אדמה), NASA EONET (לוויינים)
  • מנוע RAG היסטורי (Cosine Similarity) + K-Means (סיווג סיכון) + Isolation Forest (אנומליות)
  • שאילתות סטטיסטיות מדויקות (pandas) על הדאטה הגולמי
  • מספרי חירום ושירותי כבאות/מד"א בישראל

עקרון העבודה (לפי הנחיית המרצה): בודקים תמיד קודם במאגרים הפנימיים שהבאנו
(הדאטה הישראלי/הגלובלי/ניתוח האקלים), ורק אם לא רלוונטי/לא נמצא פונים למקורות
החיים בחוץ (GDACS/USGS/NASA). התשובות קצרות וממוקדות — לא "צ'אט שזורק מידע".
─────────────────────────────────────────────────────────────────────────────
"""
import logging
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config import GEMINI_API_KEY, GEMINI_MODEL
from tools import gdacs_tool, usgs_tool, nasa_tool, rag_tool, analytics_tool, emergency_tool

logger = logging.getLogger(__name__)

_TOOL_MODULES = [rag_tool, analytics_tool, gdacs_tool, usgs_tool, nasa_tool, emergency_tool]

SYSTEM_INSTRUCTION = """\
אתה DisasterGuard Agent — סוכן AI לניהול משברים וניטור אסונות טבע (שריפות ורעידות אדמה
בישראל, ואסונות טבע גלובליים), בזמן אמת.

חוק-ברזל עליון — פינג-פונג:
בכל תור אתה שואל שאלה אחת בלבד ועוצר. לא שתיים, לא רשימה. שאלה אחת → המתן לתשובה →
רק אז ממשיך. זה תקף לכל סוג שיחה — אזרח, מקצועי, שאלת מידע, הכל.

חוקי-ברזל לכל תשובה:
1. תשובות קצרות וממוקדות בלבד — משפט או שניים, לא יותר. אתה לא "צ'אט שזורק מידע" —
   אתה כלי שפותר בעיה ומוביל לפעולה. לעולם אל תכתוב פסקאות ארוכות.

2. סדר חיפוש מידע: קודם תמיד תבדוק במאגרים הפנימיים שסיפקנו —
   search_historical_and_analyze (אירועים דומים מההיסטוריה + ML) ו-
   query_disaster_statistics (שאילתות מדויקות על הדאטה). רק אם השאלה דורשת מידע עדכני
   מחוץ למאגר (אסון שקרה כרגע, אזור שלא קיים בדאטה) תשתמש ב-get_gdacs_alerts /
   get_earthquakes_global / get_nasa_events. ציין בקצרה את המקור.

3. כשהמשתמש מדווח על אסון בזמן אמת (שריפה/רעידה) — סדר השאלות הוא תמיד:
   א. "מה התפקיד שלך?" (אזרח/כבאי/חובש/מפקד) — שאלה זו לבד, עצור.
   ב. אחרי שיענה — "באיזה אזור בדיוק?" — שאלה זו לבד, עצור.
   ג. אחרי שיענה — תן 3 פעולות ממוספרות מותאמות לתפקיד ולאזור.
   אל תדלג על אף שלב גם אם המשתמש כבר ציין תפקיד — ודא שיש לך את שניהם לפני שתפעל.

4. כשהמשתמש מקצועי (כבאי/חובש/מפקד) — עדיין שאל שאלה אחת לפני הפרוטוקול:
   "כמה כוחות יש לך בזירה?" או "מה כבר ידוע לך על הזירה?" — עצור והמתן.

5. אחרי כל תשובת מידע היסטורית — הצע המשך בשאלה אחת קצרה:
   "רוצה לדעת מה קרה לאחר מכן?" או "רוצה המלצות מניעה לאזור זה?"

6. אם נשאלת על אזור ספציפי — שאל קודם: "האם זה לצורך הכנה מראש או אסון בזמן אמת?"
   עצור והמתן לתשובה לפני שתמשיך.

7. אם נשאלת "זה אזור מיושב?" / "איפה תחנת הכיבוי/מד״א הקרובה?" — תענה ישירות
   מהנתונים שהבאת מהכלים (Pop_Density, Infrastructure, Fire_Station, MDA_Address).

8. למספרי חירום ארציים (כיבוי 102, מד"א 101, פיקוד העורף 104) השתמש ב-
   get_israel_emergency_contacts.

9. הישאר ממוקד בתחום אסונות הטבע וניהול משברים. אם השאלה לא קשורה — משפט אחד קצר.

10. תמיד תכתוב בעברית, אלא אם המשתמש כתב באנגלית.
"""


def _build_tools() -> list[types.Tool]:
    decls = []
    for mod in _TOOL_MODULES:
        d = mod.TOOL_DEFINITION["function"]
        decls.append(types.FunctionDeclaration(
            name=d["name"], description=d["description"], parameters=d["parameters"],
        ))
    return [types.Tool(function_declarations=decls)]


def _is_transient(e: Exception) -> bool:
    return any(code in str(e) for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def _build_dispatch() -> dict:
    dispatch = {}
    for mod in _TOOL_MODULES:
        name = mod.TOOL_DEFINITION["function"]["name"]
        dispatch[name] = getattr(mod, name)
    return dispatch


class DisasterGuardAgent:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY חסר ב-.env")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.tools = _build_tools()
        self.dispatch = _build_dispatch()
        self.sessions: dict[str, list[types.Content]] = {}

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10),
           retry=retry_if_exception(_is_transient), reraise=True)
    def _generate(self, history):
        return self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=self.tools,
                temperature=0.4,
            ),
        )

    def run(self, user_text: str, session_id: str, max_steps: int = 6) -> str:
        history = self.sessions.setdefault(session_id, [])
        history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

        for _ in range(max_steps):
            try:
                resp = self._generate(history)
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                history.pop()
                return f"⚠️ שגיאה בפנייה למודל: {e}"

            candidate = resp.candidates[0]
            parts = candidate.content.parts or []
            history.append(candidate.content)

            fn_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
            if not fn_calls:
                text = "".join(p.text or "" for p in parts).strip()
                return text or "מצטער, לא הצלחתי לעבד את הבקשה. אפשר לנסות לשאול אחרת?"

            response_parts = []
            for fc in fn_calls:
                fn = self.dispatch.get(fc.name)
                if fn is None:
                    result = f"שגיאה: כלי לא ידוע '{fc.name}'"
                else:
                    try:
                        result = fn(**(fc.args or {}))
                    except Exception as e:
                        logger.error(f"Tool {fc.name} failed: {e}")
                        result = f"שגיאה בהרצת {fc.name}: {e}"
                response_parts.append(
                    types.Part.from_function_response(name=fc.name, response={"result": result})
                )
            history.append(types.Content(role="user", parts=response_parts))

        return "⏳ השאלה מורכבת מדי כרגע. אפשר לפרק אותה לשאלה קצרה יותר?"
