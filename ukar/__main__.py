"""CLI. `python -m ukar demo` runs one question with the offline stand-in."""

from __future__ import annotations

import argparse
import json
import sys

from ukar.backends.ollama import OllamaCompleter
from ukar.backends.openai_compat import OpenAICompatCompleter
from ukar.backends.scripted import ScriptedCompleter

DEMO_QUESTION = "What is still missing before you can answer?"
from ukar.format import format_trace
from ukar.planner import Budget
from ukar.router import Router


def _facts(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Fact {pair!r} must look like key=value.")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"Fact {pair!r} must look like key=value.")
        out[key] = value
    return out


def _completer(args: argparse.Namespace):
    try:
        if args.backend == "scripted":
            return ScriptedCompleter()
        if args.backend == "ollama":
            return OllamaCompleter(args.base_url or "http://127.0.0.1:11434", args.model or "llama3.2")
        if args.backend == "llamacpp":
            return OpenAICompatCompleter(args.base_url or "http://127.0.0.1:8080", args.model or "local")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(f"Unknown backend {args.backend}.")


def _run(args: argparse.Namespace) -> int:
    question = DEMO_QUESTION if args.command == "demo" else args.question
    router = Router(_completer(args), budget=Budget())
    trace = router.run(question, _facts(args.fact or []))
    sys.stdout.write(format_trace(trace))
    if args.json:
        sys.stdout.write(json.dumps(trace.to_dict(), indent=2) + "\n")
    if trace.status == "answered":
        return 0
    if trace.status == "waiting_on_user":
        return 0
    return 2


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Also print the trace as JSON.")
    common.add_argument(
        "--backend",
        choices=("scripted", "llamacpp", "ollama"),
        default="scripted",
        help="scripted needs no server. llamacpp is llama-server. ollama is /api/chat.",
    )
    common.add_argument("--model", default="", help="Model name passed to llama-server or Ollama.")
    common.add_argument(
        "--fact",
        action="append",
        default=[],
        help="A fact you already know, as key=value. Repeat for several.",
    )
    common.add_argument(
        "--base-url",
        default="",
        help="Loopback server origin only: 127.0.0.1, localhost, or ::1.",
    )
    return common


def main(argv: list[str] | None = None) -> int:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="ukar",
        description="Name the missing facts, acquire those facts, answer locally.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "demo",
        parents=[common],
        help="One question, scripted stand-in, no model server.",
    )
    ask = sub.add_parser("ask", parents=[common], help="Run one question.")
    ask.add_argument("question", help="The question the local model must face.")
    args = parser.parse_args(argv)
    if args.command == "demo":
        args.backend = "scripted"
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
