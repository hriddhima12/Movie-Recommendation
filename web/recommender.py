"""Inference-only integration for the saved CineRL Q-table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SATISFACTION_LEVELS = ("Low", "Medium", "High")


class Recommender:
    """Load prepared data once and expose Q-learning inference helpers."""

    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir
        self.movies = pd.read_csv(processed_dir / "movies.csv")
        self.train_ratings = pd.read_csv(processed_dir / "train_ratings.csv")
        q_table = pd.read_csv(processed_dir / "q_table.csv")
        self.genres = [
            column
            for column in q_table.columns
            if column not in {"preferred_genre", "satisfaction"}
        ]
        self.q_values = {
            (row["preferred_genre"], row["satisfaction"]): {
                genre: float(row[genre]) for genre in self.genres
            }
            for _, row in q_table.iterrows()
        }
        self.user_ids = sorted(self.train_ratings["user_id"].unique().tolist())
        self.seen_by_user = (
            self.train_ratings.groupby("user_id")["movie_id"].apply(set).to_dict()
        )
        popularity = (
            self.train_ratings.groupby("movie_id")
            .agg(rating_count=("rating", "size"), average_rating=("rating", "mean"))
            .sort_values(["rating_count", "average_rating"], ascending=False)
        )
        self.popularity_order = popularity.index.tolist()
        self.average_ratings = popularity["average_rating"].to_dict()
        self.movies_by_id = self.movies.set_index("movie_id").to_dict("index")

    def recommend(
        self, user_id: int, preferred_genre: str, satisfaction: str, limit: int = 10
    ) -> dict[str, Any]:
        self._validate_inputs(user_id, preferred_genre, satisfaction)
        state = (preferred_genre, satisfaction)
        scores = self.q_values[state]
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected_genre, selected_q_value = ranked[0]
        seen = self.seen_by_user.get(user_id, set())
        recommendations = []
        for movie_id in self.popularity_order:
            if movie_id in seen or movie_id not in self.movies_by_id:
                continue
            genres = str(self.movies_by_id[movie_id]["genres"]).split("|")
            if selected_genre not in genres:
                continue
            movie = self.movies_by_id[movie_id]
            recommendations.append(
                {
                    "movie_id": int(movie_id),
                    "title": movie["title"],
                    "genres": genres,
                    "year": self._year(movie["title"]),
                    "average_rating": round(float(self.average_ratings[movie_id]), 2),
                }
            )
            if len(recommendations) == limit:
                break
        return {
            "state": {"preferred_genre": preferred_genre, "satisfaction": satisfaction},
            "selected_action": selected_genre,
            "selected_q_value": selected_q_value,
            "alternatives": [
                {"genre": genre, "q_value": value} for genre, value in ranked
            ],
            "recommendations": recommendations,
        }

    def _validate_inputs(self, user_id: int, preferred_genre: str, satisfaction: str) -> None:
        if user_id not in self.user_ids:
            raise ValueError("Unknown user ID.")
        if preferred_genre not in self.genres:
            raise ValueError("Unknown preferred genre.")
        if satisfaction not in SATISFACTION_LEVELS:
            raise ValueError("Unknown satisfaction level.")

    @staticmethod
    def _year(title: str) -> int | None:
        if len(title) >= 5 and title[-5] == "(" and title[-1] == ")":
            try:
                return int(title[-4:-1])
            except ValueError:
                return None
        return None
