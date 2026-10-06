"""Ollama /api/chat. JSON format is requested only for the gap turn."""

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
        text = "http://127.0.0.1:11434"
    if not text.startswith("http://") and not text.startswith("https://"):
        text = "http://" + text
    if text.endswith("/api/chat"):
        return text
    return text + "/api/chat"


def build_payload(messages: list[dict[str, str]], *, model: str, json_mode: bool) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    if json_mode:
        payload["format"] = "json"
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
        raise RuntimeError(f"Ollama chat failed at {url}: {exc}") from exc
    try:
        loaded = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned non-JSON from {url}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("Ollama returned a JSON value that is not an object.")
    return loaded


def message_text(payload: dict) -> str:
    message = payload.get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty completion.")
    return text


class OllamaCompleter:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.2",
        *,
        timeout: float = 180.0,
        poster: Poster | None = None,
    ):
        self.base_url = loopback_base(base_url)
        self.model = model
        self.timeout = timeout
        self._poster = poster or (lambda url, payload: _post(url, payload, timeout))
        self.name = "ollama"
        self.last_payload: dict | None = None

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False, max_tokens: int = 700) -> str:
        del max_tokens  # Ollama bounds this with options.num_predict if a caller sets it later.
        payload = build_payload(messages, model=self.model, json_mode=json_mode)
        self.last_payload = payload
        loaded = self._poster(chat_url(self.base_url), payload)
        return message_text(loaded)
