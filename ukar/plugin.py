"""One local tool for llama.cpp and Ollama.

Those servers do not import this package. Your chat loop passes `tool_schemas()`
on the request, and when the model calls `ukar_acquire` you run `execute()`
and send the text back as the tool result.
"""

from __future__ import annotations

import json
from typing import Any

from ukar.format import format_trace
from ukar.parse import parse_gap
from ukar.planner import Budget, acquire
from ukar.router import default_sources
from ukar.schema import Trace


TOOL_NAME = "ukar_acquire"

SYSTEM_PROMPT = """You are the local model on this machine.
When a question depends on a fact you do not have, call ukar_acquire.
needs_json is one JSON object and nothing else:
{"confidence": 0.2, "can_answer_now": false, "known": [], "needs": [
  {"id": "user_fact", "statement": "The value only this user knows", "why": "The reply depends on that value", "scope": "personal", "freshness": "static", "keys": ["user_fact"]}
]}
scope is personal, public, or computed. freshness is static, daily, or live.
keys are short snake_case handles. At most 8 needs.
facts is a comma-separated list of key=value pairs you already have, or an empty string.
Each need is one missing fact and why that fact blocks a confident answer.
If the tool lists open questions, ask those too, and do not invent the missing numbers.
If it lists facts, answer from those facts. Every measurement, rate, or yield you state has to appear there.
Stay on this machine. Do not name another model.
"""


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style tool list. llama-server and Ollama both accept this shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": (
                    "Fill missing facts from local tables on this machine. "
                    "Pass the gap JSON you wrote. The result lists the facts it "
                    "filled, why any fact is still missing, and the confidence. "
                    "Answer only from those facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's question.",
                        },
                        "needs_json": {
                            "type": "string",
                            "description": "Gap JSON with confidence, can_answer_now, known, and needs.",
                        },
                        "facts": {
                            "type": "string",
                            "description": "Facts already known, as key=value pairs separated by commas.",
                        },
                    },
                    "required": ["question", "needs_json"],
                },
            },
        }
    ]


def _facts(raw: str) -> dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError("facts JSON must be an object.")
        return {str(key): str(value) for key, value in loaded.items() if str(value).strip()}
    out: dict[str, str] = {}
    for chunk in text.replace("\n", ",").split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Fact {piece!r} must look like key=value.")
        key, value = piece.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and value:
            out[key] = value
    return out


def _arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip() or "{}"
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return loaded
    raise ValueError("Tool arguments must be an object or a JSON string.")


def execute(name: str, arguments: Any) -> str:
    """Run one tool call. The only tool is ukar_acquire. It does not use the network."""
    if name != TOOL_NAME:
        return f"ERROR: unknown tool {name}."
    try:
        args = _arguments(arguments)
        question = str(args.get("question") or "").strip()
        needs_json = args.get("needs_json")
        if not isinstance(needs_json, str):
            needs_json = json.dumps(needs_json if needs_json is not None else {})
        gap = parse_gap(needs_json, question=question)
        acquired = acquire(gap.needs, default_sources(), _facts(str(args.get("facts") or "")), Budget())
        trace = Trace(
            question=question,
            status="waiting_on_user" if acquired.questions else "answered",
            gap=gap.to_dict(),
            facts=acquired.facts,
            questions=acquired.questions,
            considerations=acquired.considerations,
            answer="",
            notes=[
                "No second model was called.",
                "Answer only from the acquired facts. If questions remain, ask them.",
                "Facts and the answer stay on this machine.",
            ],
            backend="tool",
        )
        return format_trace(trace)
    except Exception as exc:
        return f"ERROR: {exc}"
