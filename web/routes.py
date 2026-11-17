"""Flask routes for the CineRL web application."""

from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, jsonify, render_template, request

from .history import HistoryStore
from .recommender import Recommender


def create_blueprint(recommender: Recommender, history: HistoryStore) -> Blueprint:
    blueprint = Blueprint("cinerl", __name__)

    @blueprint.get("/")
    def index():
        return render_template("index.html")

    @blueprint.get("/insights")
    def insights():
        return render_template("insights.html")

    @blueprint.get("/history")
    def history_page():
        return render_template("history.html")

    @blueprint.get("/about")
    def about():
        return render_template("about.html")

    @blueprint.get("/api/genres")
    def genres():
        return jsonify(recommender.genres)

    @blueprint.get("/api/users")
    def users():
        return jsonify(recommender.user_ids)

    @blueprint.post("/api/recommend")
    def recommend():
        payload = request.get_json(silent=True) or {}
        try:
            result = recommender.recommend(
                int(payload["user_id"]),
                str(payload["preferred_genre"]),
                str(payload["satisfaction"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        session = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": int(payload["user_id"]),
            "preferences": result["state"],
            "selected_action": result["selected_action"],
            "selected_q_value": result["selected_q_value"],
            "alternatives": result["alternatives"],
            "recommendations": [movie["title"] for movie in result["recommendations"]],
        }
        history.add(session)
        return jsonify(result)

    @blueprint.get("/api/insights")
    def api_insights():
        entries = history.list()
        if not entries:
            return jsonify({"message": "Generate recommendations to view agent insights."}), 404
        latest = entries[0]
        return jsonify(
            {
                "state": latest["preferences"],
                "selected_action": latest["selected_action"],
                "selected_q_value": latest["selected_q_value"],
                "alternatives": latest["alternatives"],
            }
        )

    @blueprint.get("/api/history")
    def get_history():
        return jsonify(history.list())

    @blueprint.delete("/api/history")
    def delete_history():
        history.clear()
        return jsonify({"status": "cleared"})

    return blueprint
