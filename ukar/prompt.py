"""Prompts for the two local completions."""

from __future__ import annotations

from ukar.schema import Fact


GAP_SYSTEM = """PHASE: gaps
You are the local model answering from this machine.
You do not choose another model, a provider, or a bigger brain.
Name the missing facts that block a confident answer to the question.
Reply with one JSON object and no markdown.

Schema:
{"confidence": 0.0, "can_answer_now": false, "known": ["short fact you already trust"], "needs": [
  {"id": "user_fact", "statement": "The value only this user knows", "why": "The reply depends on that value", "scope": "personal", "freshness": "static", "keys": ["user_fact"]}
]}

Rules:
- confidence is a number from 0 to 1.
- A need is a measurement, a published figure, or a value computed from other facts.
- scope is personal (only this user knows), public (a table or fact API could hold it), or computed.
- freshness is static, daily, or live.
- keys are 1–4 short snake_case handles.
- At most 8 needs. At most 8 known strings.
- If you already trust an answer, set can_answer_now to true, put the short facts in known, and leave needs empty.
- A famous name with more than one sense is not a missing fact. Put each sense you trust in known and set can_answer_now to true.
- Name a need only for a measurement, a published figure, or a value only this user knows.
- Do not ask for "the answer", an expert, or another model.
"""


ANSWER_SYSTEM = """PHASE: answer
Answer the user's question.
FACT lines are measurements and other values acquired on this machine. Do not invent a measurement, rate, or yield that is not in a FACT line.
KNOWN lines are facts you already trust. You may use them.
If there are no FACT lines, answer from the KNOWN lines and from knowledge you already trust.
Stay on this machine. Do not name another model.
"""


def gap_messages(question: str, facts: dict[str, str]) -> list[dict[str, str]]:
    lines = [f"Question:\n{question.strip()}"]
    if facts:
        lines.append("Facts already supplied:")
        for key in sorted(facts):
            lines.append(f"- {key} = {facts[key]}")
    else:
        lines.append("Facts already supplied: none.")
    return [
        {"role": "system", "content": GAP_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def answer_messages(
    question: str,
    facts: list[Fact],
    known: list[str] | None = None,
) -> list[dict[str, str]]:
    lines = [f"Question:\n{question.strip()}", "", "KNOWN lines:"]
    trusted = [item.strip() for item in (known or []) if str(item).strip()]
    if not trusted:
        lines.append("(none)")
    for item in trusted:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("FACT lines:")
    if not facts:
        lines.append("(none)")
    for fact in facts:
        lines.append(
            f"FACT id={fact.need_id} key={fact.key} value={fact.value} "
            f"source={fact.source_id} privacy={fact.privacy}"
        )
        lines.append(f"  {fact.text}")
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
