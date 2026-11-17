"""Local JSON storage for recommendation sessions."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return []
            with self.path.open(encoding="utf-8") as file:
                return json.load(file)

    def add(self, session: dict[str, Any]) -> None:
        with self._lock:
            entries = self.list()
            entries.insert(0, session)
            with self.path.open("w", encoding="utf-8") as file:
                json.dump(entries[:100], file, indent=2)

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("[]", encoding="utf-8")
