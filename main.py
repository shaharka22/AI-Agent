"""
main.py — DisasterGuard Agent: Web Chat
─────────────────────────────────────────────────────────────────────────────
הפעלה: python main.py
פריסה: Render → Web Service → env var: GEMINI_API_KEY
חשוב: יש להריץ פעם אחת python -m data.pipeline לפני העלאה ראשונה (בונה את אינדקס ה-RAG)

ממשק צ'אט פשוט בדפדפן (HTML + REST API) — אין תלות בטלגרם. כל לקוח מקבל
session id משלו (cookie), כדי שכמה משתמשים יוכלו לדבר עם הסוכן במקביל בלי
לערבב היסטוריית שיחה.
─────────────────────────────────────────────────────────────────────────────
"""
import logging
import os
import uuid

from flask import Flask, request, jsonify, render_template, make_response

from agent import DisasterGuardAgent

logging.basicConfig(format="%(asctime)s | %(levelname)-8s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
agent = DisasterGuardAgent()

SESSION_COOKIE = "dg_session_id"


def _get_session_id() -> tuple[str, bool]:
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        return sid, False
    return str(uuid.uuid4()), True


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    text = (data.get("message") or "").strip()
    if not text:
        return jsonify({"error": "empty message"}), 400

    sid, is_new = _get_session_id()
    try:
        reply = agent.run(text, sid)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=e)
        reply = f"⚠️ שגיאה: {e}"

    resp = make_response(jsonify({"reply": reply}))
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, max_age=60 * 60 * 24, httponly=True, samesite="Lax")
    return resp


@app.route("/api/clear", methods=["POST"])
def api_clear():
    sid, _ = _get_session_id()
    agent.clear(sid)
    return jsonify({"ok": True})


@app.route("/healthz")
def healthz():
    return "DisasterGuard Agent is running.", 200


def main():
    port = int(os.environ.get("PORT", 10000))
    logger.info("🚀 DisasterGuard Agent — starting web chat")
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
