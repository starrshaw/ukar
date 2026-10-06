"""Records the router keeps. Sources see needs and facts, never a chat handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCOPES = ("personal", "public", "computed")
FRESHNESS = ("static", "daily", "live")


@dataclass
class KnowledgeNeed:
    """One missing fact. Not a request to answer the whole question."""

    id: str
    statement: str
    why: str
    scope: str
    freshness: str
    keys: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GapReport:
    question: str
    confidence: float
    can_answer_now: bool
    known: list[str]
    needs: list[KnowledgeNeed]
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["needs"] = [need.to_dict() for need in self.needs]
        return data


@dataclass
class Fact:
    need_id: str
    key: str
    value: str
    text: str
    source_id: str
    cents: float = 0.0
    privacy: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Consideration:
    source_id: str
    need_ids: list[str]
    cents: float
    privacy: str
    decision: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    """What the router did. `answer` is empty until every need is a fact."""

    question: str
    status: str
    gap: dict[str, Any] | None = None
    facts: list[Fact] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    considerations: list[Consideration] = field(default_factory=list)
    answer: str = ""
    notes: list[str] = field(default_factory=list)
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "gap": self.gap,
            "facts": [fact.to_dict() for fact in self.facts],
            "questions": list(self.questions),
            "considerations": [item.to_dict() for item in self.considerations],
            "answer": self.answer,
            "notes": list(self.notes),
            "backend": self.backend,
        }
