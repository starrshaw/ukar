"""OpenAI-compatible chat, which is what `llama-server` speaks on /v1/chat/completions."""

from __future__ import annotations

import json
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from ukar.local_host import loopback_base


Poster = Callable[[str, dict], dict]


def chat_url(base_url: str) -> str:
    text = (base_url or "").strip().rstrip("/")
    if not text:
        text = "http://127.0.0.1:8080"
    if not text.startswith("http://") and not text.startswith("https://"):
        text = "http://" + text
    if text.endswith("/v1"):
        text = text[:-3]
    return text + "/v1/chat/completions"


def build_payload(messages: list[dict[str, str]], *, model: str, max_tokens: int, json_mode: bool) -> dict:
    payload: dict = {
        "model": model or "local",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 — user named this host
            body = response.read()
    except URLError as exc:
        raise RuntimeError(f"llama.cpp chat failed at {url}: {exc}") from exc
    try:
        loaded = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"llama.cpp returned non-JSON from {url}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("llama.cpp returned a JSON value that is not an object.")
    return loaded


def message_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("llama.cpp returned no choices.")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)
    text = str(content or "").strip()
    if not text:
        raise RuntimeError("llama.cpp returned an empty completion.")
    return text


class OpenAICompatCompleter:
    """Talks to llama-server, or any other local server with the same chat route."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "local",
        *,
        timeout: float = 180.0,
        poster: Poster | None = None,
    ):
        self.base_url = loopback_base(base_url)
        self.model = model
        self.timeout = timeout
        self._poster = poster or (lambda url, payload: _post(url, payload, timeout))
        self.name = "llamacpp"
        self.last_payload: dict | None = None

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False, max_tokens: int = 700) -> str:
        payload = build_payload(messages, model=self.model, max_tokens=max_tokens, json_mode=json_mode)
        self.last_payload = payload
        loaded = self._poster(chat_url(self.base_url), payload)
        return message_text(loaded)
