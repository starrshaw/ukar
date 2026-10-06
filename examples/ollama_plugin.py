"""Register UKAR as a tool on a loopback Ollama server.

Ollama does not load this file by itself. Serve a tool-capable model, then
run this script. It sends the tool schema, runs ukar_acquire when the model
calls it, and posts the result back.

    ollama serve
    ollama pull qwen3
    python examples/ollama_plugin.py "How many stools fit in the studio?" --model qwen3
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.request import Request, urlopen

from ukar.local_host import loopback_base
from ukar.format import finish_reply
from ukar.plugin import SYSTEM_PROMPT, execute, tool_schemas


def _post(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UKAR tool loop for Ollama.")
    parser.add_argument("question")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--fact", action="append", default=[], help="key=value already known.")
    args = parser.parse_args(argv)
    origin = loopback_base(args.base_url)
    user = args.question
    if args.fact:
        user += "\nFacts already known: " + ", ".join(args.fact)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    url = origin + "/api/chat"
    used = False
    for _ in range(4):
        payload = _post(
            url,
            {
                "model": args.model,
                "messages": messages,
                "tools": tool_schemas(),
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        message = payload.get("message") or {}
        calls = message.get("tool_calls") or []
        if not calls:
            text = str(message.get("content") or "").strip()
            sys.stdout.write(finish_reply(text, used=used))
            return 0
        messages.append(message)
        for call in calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            result = execute(name, function.get("arguments"))
            if not result.startswith("ERROR:"):
                used = True
            messages.append({"role": "tool", "tool_name": name, "content": result})
    sys.stdout.write(finish_reply("The model kept calling tools. Stopped after 4 rounds.", used=used))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
