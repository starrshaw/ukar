"""Run one question: gaps, then the cheapest facts, then a local answer."""

from __future__ import annotations

import re

from ukar.catalog import default_sources
from ukar.parse import GapError, parse_gap
from ukar.planner import Budget, acquire
from ukar.prompt import answer_messages, gap_messages
from ukar.schema import Fact, Trace


_HANDOFF_ANSWER = re.compile(
    r"\b(gpt-4|gpt-5|claude|gemini|bigger model|larger model|cloud model|delegate)\b",
    re.IGNORECASE,
)

# Reasoning models spend tokens before the JSON object. 420 cuts the object off.
GAP_MAX_TOKENS = 1400
GAP_RETRY_MAX_TOKENS = 2800


def _gap_was_cut_off(raw: str) -> bool:
    text = raw or ""
    return "{" in text and "}" not in text


class Router:
    def __init__(self, completer, sources: list | None = None, budget: Budget | None = None):
        self.completer = completer
        self.sources = list(sources) if sources is not None else default_sources()
        self.budget = budget or Budget()

    def run(self, question: str, facts: dict[str, str] | None = None, *, answer: bool = True) -> Trace:
        question = (question or "").strip()
        supplied = {
            str(key).strip(): str(value).strip()
            for key, value in (facts or {}).items()
            if str(key).strip() and str(value).strip()
        }
        backend = str(getattr(self.completer, "name", "") or self.completer.__class__.__name__)
        notes = ["Facts and the answer stay on this machine."]
        if not question:
            return Trace(
                question="",
                status="error",
                notes=["A question is required.", *notes],
                backend=backend,
            )
        try:
            raw_gap = self.completer.complete(
                gap_messages(question, supplied),
                json_mode=True,
                max_tokens=GAP_MAX_TOKENS,
            )
            try:
                gap = parse_gap(raw_gap, question=question)
            except GapError as first:
                # One local retry. A cut-off object gets a larger budget.
                # Still this model, still the gap schema.
                retry_tokens = GAP_RETRY_MAX_TOKENS if _gap_was_cut_off(raw_gap) else GAP_MAX_TOKENS
                raw_gap = self.completer.complete(
                    [
                        {"role": "system", "content": gap_messages(question, supplied)[0]["content"]},
                        {
                            "role": "user",
                            "content": (
                                "Your last gap report was rejected:\n"
                                f"{first}\n"
                                "Reply with one JSON object of missing facts. "
                                "Do not name another model.\n"
                                f"Question:\n{question}"
                            ),
                        },
                    ],
                    json_mode=True,
                    max_tokens=retry_tokens,
                )
                gap = parse_gap(raw_gap, question=question)
        except GapError as exc:
            return Trace(
                question=question,
                status="refused",
                notes=[str(exc), *notes],
                backend=backend,
            )
        except RuntimeError as exc:
            return Trace(
                question=question,
                status="error",
                notes=[str(exc), *notes],
                backend=backend,
            )

        trace = Trace(
            question=question,
            status="waiting_on_user",
            gap=gap.to_dict(),
            backend=backend,
            notes=notes,
        )
        if gap.can_answer_now:
            acquisition_facts = [
                Fact(
                    need_id=key,
                    key=key,
                    value=value,
                    text=f"Supplied with the question: {key} = {value}.",
                    source_id="given",
                    cents=0.0,
                    privacy="local",
                )
                for key, value in sorted(supplied.items())
            ]
            questions = []
            considerations = []
            known = supplied
        else:
            acquired = acquire(gap.needs, self.sources, supplied, self.budget)
            acquisition_facts = acquired.facts
            questions = acquired.questions
            considerations = acquired.considerations
            known = acquired.known
        trace.facts = acquisition_facts
        trace.questions = questions
        trace.considerations = considerations

        if questions and not gap.can_answer_now:
            trace.status = "waiting_on_user"
            trace.notes = [
                "Facts still missing, so the local model was not asked to guess.",
                *notes,
            ]
            return trace

        if not answer:
            trace.status = "answered"
            trace.gap = {**(trace.gap or {}), "resolved_keys": sorted(known)}
            return trace

        try:
            completion = self.completer.complete(
                answer_messages(question, acquisition_facts, known=list(gap.known)),
                json_mode=False,
            )
        except RuntimeError as exc:
            trace.status = "error"
            trace.notes = [str(exc), *notes]
            return trace
        if _HANDOFF_ANSWER.search(completion or ""):
            trace.status = "refused"
            trace.answer = ""
            trace.notes = [
                "The local completion tried to hand the question off. That text was dropped.",
                *notes,
            ]
            return trace
        trace.answer = (completion or "").strip()
        trace.status = "answered"
        trace.notes = notes
        # `known` is part of the story for callers that want to chain a second turn.
        trace.gap = {**(trace.gap or {}), "resolved_keys": sorted(known)}
        return trace
