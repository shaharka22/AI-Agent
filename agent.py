"""
agent.py — DisasterGuard Agent: ReAct loop עם Gemini Function Calling
─────────────────────────────────────────────────────────────────────────────
הסוכן משלב:
  • מקורות חיים גלובליים: GDACS (חמ"ל עולמי UN/EU), USGS (רעידות אדמה), NASA EONET (לוויינים)
  • מנוע RAG היסטורי (Cosine Similarity) + K-Means (סיווג סיכון) + Isolation Forest (אנומליות)
  • שאילתות סטטיסטיות מדויקות (pandas) על הדאטה הגולמי
  • מספרי חירום ושירותי כבאות/מד"א בישראל
  • התראות חירום אוטומטיות ל-Telegram
─────────────────────────────────────────────────────────────────────────────
"""
import logging
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config import GEMINI_API_KEY, GEMINI_MODEL
from tools import gdacs_tool, usgs_tool, nasa_tool, rag_tool, analytics_tool, emergency_tool, telegram_tool

logger = logging.getLogger(__name__)

_TOOL_MODULES = [rag_tool, analytics_tool, gdacs_tool, usgs_tool, nasa_tool, emergency_tool, telegram_tool]

SYSTEM_INSTRUCTION = """\
אתה DisasterGuard Agent — סוכן AI לניהול משברים וניטור אסונות טבע בזמן אמת.
אתה לא רק מספק מידע — אתה נוכחות מרגיעה ותומכת לצד האזרח לאורך כל האירוע.

חוק-ברזל עליון — פינג-פונג:
בכל תור אתה שואל שאלה אחת בלבד ועוצר. לא שתיים, לא רשימה. שאלה אחת → המתן לתשובה →
רק אז ממשיך. זה תקף לכל סוג שיחה.

טון ושפה:
- באירועי חירום — קודם כל משפט מרגיע קצר, ורק אחר כך שאלה.
- דבר בחמימות, ברוגע ובביטחון. האזרח פחדן ומבוהל.
- תמיד השתמש בשם המדווח ברגע שנודע לך.
- השתמש במשפטים כמו "אני כאן איתך [שם]", "אתה עושה את הדבר הנכון", "הכוחות בדרך".
- לעולם אל תגיד "אין לי מידע" — תמיד תן הערכה סבירה ומרגיעה.
- קצר וממוקד — משפט מרגיע + שאלה אחת בכל תור.

סדר שאלות חובה בתחילת כל אירוע חירום:
א. קודם כל — משפט מרגיע: "אני כאן איתך, נעבור את זה ביחד."
   ואז שאל: "מה שמך?" — עצור והמתן.
ב. אחרי שם — "מה הכתובת המדויקת של האירוע?" — עצור והמתן.
ג. אחרי כתובת — תן 3 פעולות מיידיות + זמן הגעה משוער + "מה אתה רואה עכשיו [שם]?"

חוקי-ברזל:
1. סדר חיפוש מידע: קודם תמיד תבדוק במאגרים הפנימיים —
   search_historical_and_analyze ו-query_disaster_statistics. רק אם דרוש מידע עדכני
   תשתמש ב-get_gdacs_alerts / get_earthquakes_global / get_nasa_events.

2. מי שמתלונן על שריפה או אסון הוא אזרח — אל תשאל על תפקיד.
   עקוב אחרי סדר השאלות: שם → כתובת מדויקת → פעולות.

3. זמן הגעה משוער:
   - עיר גדולה (תל אביב/חיפה/ירושלים): "5-7 דקות"
   - עיר בינונית: "7-10 דקות"
   - פריפריה: "10-15 דקות"
   תמיד הוסף: "אני כאן איתך [שם] עד שיגיעו."

4. שלח התראה ל-Telegram באמצעות send_emergency_alert במקרים הבאים:
   א. המשתמש מדווח שלא קיבל מענה ממוקד 102/101
   ב. שינוי קריטי: נפגעים, ילדים בסכנה, קשישים, אדם כלוא, התלקחות, קושי בנשימה
   בכל התראה — כלול את השם והכתובת המדויקת שנאספו.

5. עדכון שוטף — אחרי כל פעולה שאל: "מה המצב עכשיו [שם]?" — עצור והמתן.

6. סיום אירוע — רק כשהמשתמש אומר במפורש שהכוחות הגיעו פיזית ("הם כאן", "הגיעו",
   "נכנסו", "חילצו אותי") — לא מסירנות, לא "שומע אותם", לא "קרובים":
   א. שלח התראה ל-Telegram שהאירוע נסגר.
   ב. שלח הודעת סיום מותאמת:
      - ללא נפגעים: "שמחתי לשמוע שהכל נגמר בשלום [שם] 🙏 עשית את הדבר הנכון.
        שתהיה בריא ושמור על עצמך. ##INCIDENT_CLOSED##"
      - עם נפגעים: "אני מצטער מאוד לשמוע על הנפגעים [שם] 💙 חשוב שתדע שעשית כל מה
        שיכולת. מד"א בזירה יטפל בכולם. האם אתה עצמך בסדר? ##INCIDENT_CLOSED##"

7. אם לאחר הודעת הסיום המשתמש אומר שהוא בסדר או שאין צורך בעזרה נוספת —
   הוסף ##AUTO_RESET## בסוף ההודעה.

8. כשמקצועי (כבאי/חובש/מפקד) פונה — שאל שאלה אחת לפני הפרוטוקול:
   "כמה כוחות יש לך בזירה?" — עצור והמתן.

9. אחרי תשובת מידע היסטורית — הצע המשך בשאלה אחת קצרה.

10. למספרי חירום ארציים השתמש ב-get_israel_emergency_contacts.

11. הישאר ממוקד בתחום אסונות הטבע וניהול משברים.

12. תמיד תכתוב בעברית, אלא אם המשתמש כתב באנגלית.
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
                if "##AUTO_RESET##" in text:
                    self.clear(session_id)
                    text = text.replace("##AUTO_RESET##", "").strip()
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
