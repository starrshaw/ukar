"""UKAR only talks to a model server on this machine."""

from __future__ import annotations

from urllib.parse import urlparse


_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def loopback_base(url: str) -> str:
    """Accept a loopback origin. Refuse any other host."""
    text = (url or "").strip()
    if not text:
        raise ValueError("A local server URL is required.")
    if not text.startswith("http://") and not text.startswith("https://"):
        text = "http://" + text
    host = (urlparse(text).hostname or "").lower().strip("[]")
    if host not in _LOOPBACK:
        raise ValueError(
            "UKAR only talks to llama.cpp or Ollama on this machine "
            "(127.0.0.1, localhost, or ::1)."
        )
    return text.rstrip("/")
