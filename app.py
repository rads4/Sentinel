"""
Phishing Detector — Flask Application
--------------------------------------
Routes:
  GET  /              → Home page
  GET  /analyze       → URL analysis form
  POST /api/analyze   → JSON prediction endpoint
  GET  /result        → Result display (reads sessionStorage via JS)
  GET  /explain       → SHAP explainability dashboard
  GET  /research      → Research insights and model comparison
"""

import os
import json
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "phishing-detector-viva-2024")


# ---------------------------------------------------------------------------
# Lazy-load prediction module (avoids import-time model load on cold start)
# ---------------------------------------------------------------------------
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from model.predict import predict, get_training_report
        _predictor = {"predict": predict, "report": get_training_report}
    return _predictor


# ---------------------------------------------------------------------------
# Sample URLs shown on the analyze page
# ---------------------------------------------------------------------------
SAMPLE_URLS = [
    {"label": "Google",          "url": "https://google.com",                          "type": "legitimate"},
    {"label": "GitHub",          "url": "https://github.com",                          "type": "legitimate"},
    {"label": "PayPal",          "url": "https://paypal.com",                          "type": "legitimate"},
    {"label": "PayPal Phishing", "url": "http://paypal-security-login.com",            "type": "phishing"},
    {"label": "IP Address URL",  "url": "http://192.168.1.1/verify-account",           "type": "phishing"},
    {"label": "Bank Phishing",   "url": "http://secure-bankofamerica-login.com/update","type": "phishing"},
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze")
def analyze():
    prefill = request.args.get("url", "")
    return render_template("analyze.html", sample_urls=SAMPLE_URLS, prefill=prefill)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No URL provided."}), 400

    if len(url) > 2000:
        return jsonify({"error": "URL too long (max 2000 characters)."}), 400

    try:
        predictor = get_predictor()
        result = predictor["predict"](url)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    if result.get("error"):
        return jsonify({"error": result["error"]}), 422

    return jsonify(result)


@app.route("/result")
def result():
    """
    Result page — the actual result data is stored in sessionStorage
    by the frontend JS and read on page load. This keeps the URL clean
    and avoids passing large JSON through URL params.
    """
    return render_template("result.html")


@app.route("/explain")
def explain():
    return render_template("explain.html")


@app.route("/research")
def research():
    try:
        predictor = get_predictor()
        report = predictor["report"]()
    except Exception:
        report = {}
    return render_template("research.html", report=report)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    print(f"\n  Phishing Detector running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
