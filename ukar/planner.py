"""Pick the local source that covers the most remaining facts.

Every source has to stay on this machine. Anything else is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ukar.schema import Consideration, Fact, KnowledgeNeed
from ukar.sources import LookupTable


@dataclass
class Budget:
    max_calls: int = 8
    calls: int = 0

    def allows(self, calls: int) -> str:
        if self.calls + calls > self.max_calls:
            return "call budget is spent"
        return ""


@dataclass
class Acquisition:
    facts: list[Fact] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    considerations: list[Consideration] = field(default_factory=list)
    known: dict[str, str] = field(default_factory=dict)


def _score(cents: float, count: int) -> float:
    return cents / max(1, count)


def _question_for(need: KnowledgeNeed, note: str) -> str:
    keys = ", ".join(need.keys)
    base = f"{need.statement} — reply with {keys}. {need.why}"
    if note:
        return f"{base} {note}"
    return base


def acquire(
    needs: list[KnowledgeNeed],
    sources: list[Any],
    known: dict[str, str],
    budget: Budget | None = None,
) -> Acquisition:
    """Bind needs to sources. Mutates a copy of `known` as facts arrive."""
    budget = budget or Budget()
    remaining = {need.id: need for need in needs}
    have = {key: str(value).strip() for key, value in known.items() if str(value).strip()}
    result = Acquisition(known=have)
    refused_off_machine: set[str] = set()
    # One pass per source pick. Eight needs and a handful of sources is enough.
    for _ in range(budget.max_calls + len(sources) + 2):
        if not remaining:
            break
        best: tuple[float, Any, list[str], float, str] | None = None
        round_notes: list[Consideration] = []
        for source in sources:
            privacy = str(getattr(source, "privacy", "local"))
            if privacy != "local":
                source_id = str(getattr(source, "id", "source"))
                if source_id not in refused_off_machine:
                    refused_off_machine.add(source_id)
                    result.considerations.append(
                        Consideration(
                            source_id=source_id,
                            need_ids=[need.id for need in remaining.values()],
                            cents=0.0,
                            privacy=privacy,
                            decision="rejected",
                            reason="UKAR only uses facts already on this machine",
                        )
                    )
                continue
            covered = [
                need_id
                for need_id in source.cover(list(remaining.values()), have)
                if need_id in remaining
            ]
            if not covered:
                continue
            quote = source.quote(covered)
            block = budget.allows(quote.calls)
            if block:
                round_notes.append(
                    Consideration(
                        source_id=source.id,
                        need_ids=covered,
                        cents=quote.cents,
                        privacy=privacy,
                        decision="rejected",
                        reason=block,
                    )
                )
                continue
            score = _score(quote.cents, len(covered))
            round_notes.append(
                Consideration(
                    source_id=source.id,
                    need_ids=list(covered),
                    cents=quote.cents,
                    privacy=privacy,
                    decision="candidate",
                    reason=f"score {score:.2f} for {len(covered)} fact(s)",
                )
            )
            if best is None or score < best[0] - 1e-9 or (
                abs(score - best[0]) < 1e-9 and len(covered) > len(best[2])
            ):
                best = (score, source, covered, quote.cents, privacy)
        if best is None:
            result.considerations.extend(round_notes)
            break
        _, source, covered, cents, privacy = best
        chosen_ids = set(covered)
        kept: list[Consideration] = []
        for note in round_notes:
            overlaps = bool(set(note.need_ids) & chosen_ids)
            if note.source_id == source.id and note.decision == "candidate" and note.need_ids == covered:
                note.decision = "chosen"
                note.reason = "cheapest source that returns these facts"
                kept.append(note)
            elif note.decision == "candidate" and overlaps:
                note.decision = "rejected"
                note.reason = "another source returns these facts for less"
                kept.append(note)
            elif note.decision == "rejected" and overlaps:
                kept.append(note)
        result.considerations.extend(kept)
        subset = [remaining[need_id] for need_id in covered]
        try:
            fetched = list(source.fetch(subset, have))
        except RuntimeError as exc:
            result.considerations.append(
                Consideration(
                    source_id=source.id,
                    need_ids=covered,
                    cents=cents,
                    privacy=privacy,
                    decision="rejected",
                    reason=str(exc),
                )
            )
            sources = [item for item in sources if item is not source]
            continue
        if not fetched:
            result.considerations.append(
                Consideration(
                    source_id=source.id,
                    need_ids=covered,
                    cents=cents,
                    privacy=privacy,
                    decision="rejected",
                    reason="the source covered the facts but returned nothing",
                )
            )
            sources = [item for item in sources if item is not source]
            continue
        budget.calls += 1
        for fact in fetched:
            result.facts.append(fact)
            have[fact.key] = fact.value
            remaining.pop(fact.need_id, None)
        result.known = dict(have)

    for need in remaining.values():
        blocker = ""
        waiting_on = ""
        for source in sources:
            if not isinstance(source, LookupTable):
                continue
            reason = source.blocked_reason(need, have)
            if not reason:
                continue
            blocker = reason
            if "once `" in reason:
                waiting_on = source.match_fact
            break
        other_asks_for_match = waiting_on and any(
            waiting_on in other.keys
            for other in remaining.values()
            if other.id != need.id
        )
        if other_asks_for_match:
            result.questions.append(f"{need.statement} waits on `{waiting_on}`. {blocker}")
            continue
        result.questions.append(_question_for(need, blocker))
    result.known = have
    return result
