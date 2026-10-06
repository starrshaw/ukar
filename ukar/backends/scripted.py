"""Offline completer. It has no topic and does not invent facts."""

from __future__ import annotations

import json

from ukar.prompt import ANSWER_SYSTEM
from ukar.schema import Fact


GENERIC_GAP = """{
  "confidence": 0.2,
  "can_answer_now": false,
  "known": [],
  "needs": [
    {
      "id": "user_fact",
      "statement": "A value only the user knows",
      "why": "Without that value the reply would have to guess",
      "scope": "personal",
      "freshness": "static",
      "keys": ["user_fact"]
    }
  ]
}"""


def _supplied(user: str) -> list[tuple[str, str]]:
    if "Facts already supplied:" not in user:
        return []
    body = user.split("Facts already supplied:", 1)[1]
    rows: list[tuple[str, str]] = []
    for line in body.splitlines():
        piece = line.strip()
        if piece.startswith("- ") and " = " in piece:
            key, value = piece[2:].split(" = ", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                rows.append((key, value))
    return rows


def _gap(user: str) -> str:
    rows = _supplied(user)
    if not rows:
        return GENERIC_GAP
    return json.dumps(
        {
            "confidence": 0.8,
            "can_answer_now": True,
            "known": [f"{key} = {value}" for key, value in rows],
            "needs": [],
        }
    )


def _facts_from_answer_prompt(content: str) -> list[Fact]:
    facts: list[Fact] = []
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("FACT id="):
            continue
        fields: dict[str, str] = {}
        for chunk in line.split()[1:]:
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            fields[key] = value
        text = ""
        if index + 1 < len(lines) and lines[index + 1].startswith("  "):
            text = lines[index + 1].strip()
        facts.append(
            Fact(
                need_id=fields.get("id") or "fact",
                key=fields.get("key") or "",
                value=fields.get("value") or "",
                text=text,
                source_id=fields.get("source") or "",
                privacy=fields.get("privacy") or "local",
            )
        )
    return facts


class ScriptedCompleter:
    """No network and no topic. Any question uses the same gap rule."""

    name = "scripted"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_tokens: int = 700,
    ) -> str:
        del json_mode, max_tokens
        system = ""
        user = ""
        for message in messages:
            role = message.get("role")
            if role == "system":
                system = message.get("content") or ""
            elif role == "user":
                user = message.get("content") or ""
        if "PHASE: gaps" in system:
            return _gap(user)
        if system.strip().startswith("PHASE: answer") or ANSWER_SYSTEM[:12] in system:
            facts = _facts_from_answer_prompt(user)
            if not facts:
                return "No measurement was required. Nothing was invented."
            lines = ["Using only the facts that were supplied:"]
            for fact in facts:
                lines.append(f"- {fact.key} = {fact.value}")
            return "\n".join(lines)
        raise RuntimeError("Scripted completer saw a prompt it does not own.")
