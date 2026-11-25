"""The comparison record: prompt, blind-labeled responses, and scores."""

from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .providers import ModelResponse
from .rubric import Rubric


@dataclass
class Evaluation:
    scores: dict[str, int]
    note: str = ""

    @property
    def total(self) -> int:
        return sum(self.scores.values())


@dataclass
class Entry:
    label: str
    response: ModelResponse
    evaluation: Evaluation | None = None


@dataclass
class Comparison:
    prompt: str
    rubric_name: str
    entries: list[Entry]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    preferred_label: str | None = None
    preference_reason: str = ""

    @property
    def is_scored(self) -> bool:
        return all(entry.evaluation is not None for entry in self.entries)

    def entry(self, label: str) -> Entry:
        for item in self.entries:
            if item.label == label:
                return item
        raise KeyError(label)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Comparison:
        entries = []
        for raw in data["entries"]:
            evaluation = raw.get("evaluation")
            entries.append(
                Entry(
                    label=raw["label"],
                    response=ModelResponse(**raw["response"]),
                    evaluation=Evaluation(**evaluation) if evaluation else None,
                )
            )
        return cls(
            prompt=data["prompt"],
            rubric_name=data["rubric_name"],
            entries=entries,
            created_at=data["created_at"],
            preferred_label=data.get("preferred_label"),
            preference_reason=data.get("preference_reason", ""),
        )


def build_comparison(
    prompt: str,
    responses: list[ModelResponse],
    rubric: Rubric,
    rng: random.Random | None = None,
) -> Comparison:
    """Shuffle responses and label them A, B, C... so scoring is blind.

    Hiding which model wrote which answer reduces brand bias while scoring.
    """
    if len(responses) > len(string.ascii_uppercase):
        raise ValueError("Too many responses to label")
    shuffled = list(responses)
    (rng or random.Random()).shuffle(shuffled)
    entries = [
        Entry(label=string.ascii_uppercase[i], response=resp) for i, resp in enumerate(shuffled)
    ]
    return Comparison(prompt=prompt, rubric_name=rubric.name, entries=entries)


def save(comparison: Comparison, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison.to_dict(), indent=2), encoding="utf-8")
    return path


def load(path: str | Path) -> Comparison:
    return Comparison.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
