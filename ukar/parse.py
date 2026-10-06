"""Parse a local model's gap JSON. Reject anything that is a model handoff."""

from __future__ import annotations

import json
import re
from typing import Any

from ukar.schema import FRESHNESS, SCOPES, GapReport, KnowledgeNeed


class GapError(ValueError):
    """The local completion was not a usable list of missing facts."""


# Keys that mean "send this question somewhere else."
_HANDOFF_KEYS = {
    "route",
    "route_to",
    "delegate",
    "delegate_to",
    "cloud",
    "cloud_model",
    "bigger_model",
    "fallback_model",
    "use_model",
    "model",
    "provider",
    "target_model",
}

_HANDOFF_TEXT = re.compile(
    r"\b("
    r"bigger model|larger model|smarter model|cloud model|"
    r"another model|different model|"
    r"gpt-4|gpt-5|claude|gemini|"
    r"the answer|expert opinion|just answer"
    r")\b",
    re.IGNORECASE,
)

_SLUG = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _walk_keys(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(str(key))
            found.extend(_walk_keys(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_keys(item))
    return found


def extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise GapError("The local model returned an empty gap report.")
    match = _FENCE.search(text)
    if match:
        text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise GapError("The local model did not return a JSON object.")
    try:
        loaded = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise GapError(f"Gap JSON did not parse: {exc}") from exc
    if not isinstance(loaded, dict):
        raise GapError("Gap JSON must be an object.")
    return loaded


def _as_str_list(value: Any, *, limit: int, item_max: int, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GapError(f"{label} must be a list.")
    if len(value) > limit:
        raise GapError(f"{label} has more than {limit} items.")
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        if len(text) > item_max:
            raise GapError(f"A {label} item is longer than {item_max} characters.")
        out.append(text)
    return out


def parse_gap(raw: str, *, question: str) -> GapReport:
    """Validate one gap report for this question."""
    data = extract_json_object(raw)
    leaked = sorted({key.lower() for key in _walk_keys(data)} & _HANDOFF_KEYS)
    if leaked:
        raise GapError(
            "The gap report tried to hand the question to another model "
            f"({', '.join(leaked)}). UKAR only accepts missing facts."
        )

    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise GapError("confidence must be a number from 0 to 1.") from exc
    if confidence < 0 or confidence > 1:
        raise GapError("confidence must be a number from 0 to 1.")

    if not isinstance(data.get("can_answer_now"), bool):
        raise GapError("can_answer_now must be true or false.")
    can_answer = bool(data["can_answer_now"])
    known = _as_str_list(data.get("known") or [], limit=8, item_max=200, label="known")

    raw_needs = data.get("needs")
    if raw_needs is None:
        raw_needs = []
    if not isinstance(raw_needs, list):
        raise GapError("needs must be a list.")
    if len(raw_needs) > 8:
        raise GapError("A gap report can name at most 8 missing facts.")
    if can_answer and raw_needs:
        raise GapError("can_answer_now is true, so needs must be empty.")
    if not can_answer and not raw_needs:
        raise GapError("The model is not confident and named no missing facts.")

    question_norm = re.sub(r"\s+", " ", (question or "").strip().lower())
    needs: list[KnowledgeNeed] = []
    seen: set[str] = set()
    for item in raw_needs:
        if not isinstance(item, dict):
            raise GapError("Each need must be an object.")
        need_id = str(item.get("id") or "").strip()
        if not _SLUG.match(need_id):
            raise GapError(f"Need id {need_id!r} must be a short snake_case slug.")
        if need_id in seen:
            raise GapError(f"Duplicate need id {need_id}.")
        seen.add(need_id)
        statement = str(item.get("statement") or "").strip()
        why = str(item.get("why") or "").strip()
        if not statement or len(statement) > 180:
            raise GapError(f"Need {need_id} needs a statement of 1–180 characters.")
        if not why or len(why) > 180:
            raise GapError(f"Need {need_id} needs a why of 1–180 characters.")
        scope = str(item.get("scope") or "").strip()
        freshness = str(item.get("freshness") or "").strip()
        if scope not in SCOPES:
            raise GapError(f"Need {need_id} scope must be personal, public, or computed.")
        if freshness not in FRESHNESS:
            raise GapError(f"Need {need_id} freshness must be static, daily, or live.")
        keys = _as_str_list(item.get("keys"), limit=4, item_max=40, label=f"{need_id} keys")
        if not keys or any(not _SLUG.match(key) for key in keys):
            raise GapError(f"Need {need_id} keys must be 1–4 snake_case handles.")
        blob = f"{statement} {why}"
        if _HANDOFF_TEXT.search(blob):
            raise GapError(
                f"Need {need_id} asks for a handoff or for 'the answer', not a fact."
            )
        statement_norm = re.sub(r"\s+", " ", statement.lower())
        if question_norm and statement_norm == question_norm:
            raise GapError(f"Need {need_id} repeats the question instead of naming a fact.")
        needs.append(
            KnowledgeNeed(
                id=need_id,
                statement=statement,
                why=why,
                scope=scope,
                freshness=freshness,
                keys=keys,
            )
        )

    return GapReport(
        question=question,
        confidence=confidence,
        can_answer_now=can_answer,
        known=known,
        needs=needs,
        raw=raw.strip()[:4000],
    )
