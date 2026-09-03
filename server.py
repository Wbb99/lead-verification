#!/usr/bin/env python3
"""
Lead Analysis API Server
Exposes the lead analyzer as a REST API with webhook support and API key auth.
"""

import os
import secrets
import logging
from functools import wraps
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from lead_analyzer import analyze_lead

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# API Key - set via environment variable or auto-generate on first run
API_KEY = os.environ.get("LEAD_API_KEY")
if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    logger.warning(f"No LEAD_API_KEY set. Generated key: {API_KEY}")
    logger.warning("Set LEAD_API_KEY in your .env file for production use.")


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not key or key != API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated


def format_explanation(result):
    """Format the explanation string as: Outcome: REASON, Answer: REASON, Spam: REASON"""
    reasoning = result.reasoning

    # Build outcome explanation
    outcome_reason = reasoning
    if ":" in reasoning:
        # Extract the main reasoning after the prefix
        outcome_reason = reasoning.split(":", 1)[1].strip()

    # Build answer explanation
    answer_map = {
        "Yes": "Human answered the call and engaged in conversation",
        "No": "No human answered - voicemail, unanswered, or spam recording",
        "Dropped": "Call reached only an IVR/automated system, no human interaction",
        "Not a Phone Call": "This is an SMS, form submission, or non-phone lead",
    }
    answer_reason = answer_map.get(result.call_answer, result.call_answer)

    # Build spam explanation
    if result.spam:
        spam_reason = "Yes - spam/scam call detected (business verification, robocall, or solicitation)"
    else:
        spam_reason = "No"

    return f"Outcome: {outcome_reason}, Answer: {answer_reason}, Spam: {spam_reason}"


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint (no auth required)."""
    return jsonify({"status": "ok", "service": "lead-analyzer"})


@app.route("/webhook", methods=["POST"])
@require_api_key
def webhook():
    """
    Webhook endpoint for lead analysis.

    Input (JSON):
        {
            "transcript": "Speaker A: Hello...",
            "phonecall": "yes" or "no"
        }

        - transcript: the call transcript or SMS text
        - phonecall: "yes" = phone call, "no" = SMS/form/non-phone (defaults to "yes")

    Output (JSON):
        {
            "outcome": "Verified",
            "answer": "Yes",
            "spam": false,
            "explanation": "Outcome: REASON, Answer: REASON, Spam: REASON"
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    transcript = data.get("transcript", "")
    phonecall = str(data.get("phonecall", "yes")).lower().strip()

    if not transcript:
        return jsonify({"error": "transcript is required"}), 400

    # If not a phone call, use "Website" source to trigger Not a Phone Call detection
    if phonecall == "no":
        source = "Website"
    else:
        source = "Google Business Profile"

    try:
        result = analyze_lead(source, transcript, use_ai=True)

        # If phonecall is "no", force answer to "Not a Phone Call"
        if phonecall == "no":
            result.call_answer = "Not a Phone Call"

        response = {
            "outcome": result.outcome,
            "answer": result.call_answer,
            "spam": result.spam,
            "explanation": format_explanation(result),
        }

        logger.info(f"Webhook: answer={result.call_answer} outcome={result.outcome} spam={result.spam}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"error": "Analysis failed", "detail": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
@require_api_key
def analyze():
    """
    Analyze a single lead (same as webhook, alternative endpoint).

    Input/Output format is identical to /webhook.
    """
    return webhook()


@app.route("/api/analyze/batch", methods=["POST"])
@require_api_key
def analyze_batch():
    """
    Analyze multiple leads in a single request.

    Input (JSON):
        {
            "leads": [
                {"transcript": "...", "phonecall": "yes"},
                {"transcript": "...", "phonecall": "no"}
            ]
        }

    Output (JSON):
        {
            "results": [
                {"outcome": "Verified", "answer": "Yes", "spam": false, "explanation": "..."},
                ...
            ]
        }
    """
    data = request.get_json()
    if not data or "leads" not in data:
        return jsonify({"error": "Request body must contain 'leads' array"}), 400

    results = []
    for i, lead in enumerate(data["leads"]):
        transcript = lead.get("transcript", "")
        phonecall = str(lead.get("phonecall", "yes")).lower().strip()

        if phonecall == "no":
            source = "Website"
        else:
            source = "Google Business Profile"

        try:
            result = analyze_lead(source, transcript, use_ai=True)
            if phonecall == "no":
                result.call_answer = "Not a Phone Call"

            results.append({
                "index": i,
                "outcome": result.outcome,
                "answer": result.call_answer,
                "spam": result.spam,
                "explanation": format_explanation(result),
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"\n{'='*50}")
    print(f"Lead Analysis API Server")
    print(f"{'='*50}")
    print(f"Running on: http://0.0.0.0:{port}")
    print(f"API Key:    {API_KEY}")
    print(f"{'='*50}")
    print(f"\nEndpoints:")
    print(f"  GET  /health            - Health check (no auth)")
    print(f"  POST /webhook           - Analyze a lead")
    print(f"  POST /api/analyze       - Analyze a lead (alias)")
    print(f"  POST /api/analyze/batch - Analyze multiple leads")
    print(f"\nAuth: X-API-Key header")
    print(f"{'='*50}\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
