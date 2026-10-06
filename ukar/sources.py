"""Fact sources. `fetch` receives needs and known facts. It has no question argument."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ukar.schema import Fact, KnowledgeNeed


@dataclass(frozen=True)
class Quote:
    cents: float
    privacy: str
    calls: int = 1


class GivenFacts:
    """Facts the caller already handed in. Cost is zero. They stay on the device."""

    id = "given"
    privacy = "local"

    def cover(self, needs: Iterable[KnowledgeNeed], known: dict[str, str]) -> list[str]:
        hit: list[str] = []
        for need in needs:
            for key in need.keys:
                if key in known and str(known[key]).strip():
                    hit.append(need.id)
                    break
        return hit

    def quote(self, need_ids: list[str]) -> Quote:
        return Quote(0.0, self.privacy, 0 if not need_ids else 1)

    def fetch(self, needs: list[KnowledgeNeed], known: dict[str, str]) -> list[Fact]:
        facts: list[Fact] = []
        for need in needs:
            for key in need.keys:
                value = str(known.get(key) or "").strip()
                if not value:
                    continue
                facts.append(
                    Fact(
                        need_id=need.id,
                        key=key,
                        value=value,
                        text=f"Supplied with the question: {key} = {value}.",
                        source_id=self.id,
                        cents=0.0,
                        privacy=self.privacy,
                    )
                )
                break
        return facts


class StaticTable:
    """Unconditional local facts from rows you supply."""

    def __init__(self, source_id: str, rows: list[dict[str, str]]):
        self.id = source_id
        self.privacy = "local"
        self._by_key = {str(row["key"]): row for row in rows}

    def cover(self, needs: Iterable[KnowledgeNeed], known: dict[str, str]) -> list[str]:
        del known
        hit: list[str] = []
        for need in needs:
            if any(key in self._by_key for key in need.keys):
                hit.append(need.id)
        return hit

    def quote(self, need_ids: list[str]) -> Quote:
        return Quote(0.0, self.privacy, 1 if need_ids else 0)

    def fetch(self, needs: list[KnowledgeNeed], known: dict[str, str]) -> list[Fact]:
        del known
        facts: list[Fact] = []
        for need in needs:
            for key in need.keys:
                row = self._by_key.get(key)
                if row is None:
                    continue
                facts.append(
                    Fact(
                        need_id=need.id,
                        key=key,
                        value=str(row["value"]),
                        text=str(row["text"]),
                        source_id=self.id,
                        cents=0.0,
                        privacy=self.privacy,
                    )
                )
                break
        return facts


class LookupTable:
    """Local rows matched on one fact already in hand."""

    def __init__(
        self,
        source_id: str,
        rows: list[dict[str, str]],
        *,
        match_fact: str,
        emit_key: str,
        aliases: dict[str, str] | None = None,
    ):
        self.id = source_id
        self.privacy = "local"
        self.match_fact = match_fact
        self.emit_key = emit_key
        self._aliases = {key.lower(): value for key, value in (aliases or {}).items()}
        self._rows = {str(row["match"]).lower(): row for row in rows}

    def _token(self, known: dict[str, str]) -> str:
        raw = str(known.get(self.match_fact) or "").strip().lower()
        return self._aliases.get(raw, raw)

    def _row(self, known: dict[str, str]) -> dict[str, str] | None:
        token = self._token(known)
        if not token:
            return None
        return self._rows.get(token)

    def cover(self, needs: Iterable[KnowledgeNeed], known: dict[str, str]) -> list[str]:
        if self._row(known) is None:
            return []
        return [need.id for need in needs if self.emit_key in need.keys]

    def blocked_reason(self, need: KnowledgeNeed, known: dict[str, str]) -> str:
        if self.emit_key not in need.keys:
            return ""
        if self._row(known) is not None:
            return ""
        if not str(known.get(self.match_fact) or "").strip():
            return (
                f"{self.id} can fill {self.emit_key} once `{self.match_fact}` is known. "
                "That lookup stays on this machine."
            )
        return (
            f"{self.id} has no row for {self.match_fact}="
            f"{known.get(self.match_fact)}. Supply `{self.emit_key}` directly "
            "or add a row."
        )

    def quote(self, need_ids: list[str]) -> Quote:
        return Quote(0.0, self.privacy, 1 if need_ids else 0)

    def fetch(self, needs: list[KnowledgeNeed], known: dict[str, str]) -> list[Fact]:
        row = self._row(known)
        if row is None:
            return []
        facts: list[Fact] = []
        for need in needs:
            if self.emit_key not in need.keys:
                continue
            facts.append(
                Fact(
                    need_id=need.id,
                    key=self.emit_key,
                    value=str(row["value"]),
                    text=str(row["text"]),
                    source_id=self.id,
                    cents=0.0,
                    privacy=self.privacy,
                )
            )
        return facts
