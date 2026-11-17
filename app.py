"""CineRL Flask application entry point."""

from pathlib import Path

from flask import Flask

from web.history import HistoryStore
from web.recommender import Recommender
from web.routes import create_blueprint


ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
recommender = Recommender(ROOT / "data" / "processed")
history = HistoryStore(ROOT / "data" / "recommendation_history.json")
app.register_blueprint(create_blueprint(recommender, history))


if __name__ == "__main__":
    app.run(debug=True)
