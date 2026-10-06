"""Register UKAR as a tool on a loopback llama-server.

llama-server does not load this file by itself. Start the server with --jinja,
then run this script. It sends the tool schema, runs ukar_acquire when the
model calls it, and posts the result back.

    llama-server -m model.gguf --jinja --host 127.0.0.1 --port 8080
    python examples/llamacpp_plugin.py "How many stools fit in the studio?"
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


def _tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    return calls if isinstance(calls, list) else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UKAR tool loop for llama-server.")
    parser.add_argument("question")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
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
    url = origin + "/v1/chat/completions"
    used = False
    for _ in range(4):
        payload = _post(
            url,
            {
                "model": args.model,
                "messages": messages,
                "tools": tool_schemas(),
                "temperature": 0.1,
            },
        )
        message = ((payload.get("choices") or [{}])[0].get("message") or {})
        calls = _tool_calls(message)
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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or name,
                    "content": result,
                }
            )
    sys.stdout.write(finish_reply("The model kept calling tools. Stopped after 4 rounds.", used=used))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
