"""Load and validate scoring rubrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class RubricError(Exception):
    """Raised when a rubric file is invalid."""


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str


@dataclass(frozen=True)
class Rubric:
    name: str
    criteria: tuple[Criterion, ...]
    min_score: int = 1
    max_score: int = 5

    def validate_score(self, value: int) -> int:
        if not self.min_score <= value <= self.max_score:
            raise ValueError(f"Score must be between {self.min_score} and {self.max_score}")
        return value

    @property
    def max_total(self) -> int:
        return self.max_score * len(self.criteria)


DEFAULT_RUBRIC = Rubric(
    name="General quality",
    criteria=(
        Criterion("accuracy", "Facts and code are correct, with no made-up details"),
        Criterion("completeness", "Answers every part of the prompt"),
        Criterion("clarity", "Easy to follow, well organized, right length"),
    ),
)


def load_rubric(path: str | Path | None) -> Rubric:
    """Load a rubric from a JSON file, or return the default when path is None."""
    if path is None:
        return DEFAULT_RUBRIC

    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RubricError(f"Rubric file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RubricError(f"Rubric file is not valid JSON: {exc}") from exc

    raw_criteria = data.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise RubricError("Rubric needs a non-empty 'criteria' list")

    criteria = []
    seen = set()
    for item in raw_criteria:
        name = str(item.get("name", "")).strip().lower() if isinstance(item, dict) else ""
        if not name:
            raise RubricError("Every criterion needs a 'name'")
        if name in seen:
            raise RubricError(f"Duplicate criterion: {name}")
        seen.add(name)
        criteria.append(Criterion(name, str(item.get("description", "")).strip()))

    min_score = int(data.get("min_score", 1))
    max_score = int(data.get("max_score", 5))
    if min_score >= max_score:
        raise RubricError("min_score must be lower than max_score")

    return Rubric(
        name=str(data.get("name", path.stem)),
        criteria=tuple(criteria),
        min_score=min_score,
        max_score=max_score,
    )
