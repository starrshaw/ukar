"""Local completers for llama.cpp and Ollama on this machine."""

from ukar.backends.ollama import OllamaCompleter
from ukar.backends.openai_compat import OpenAICompatCompleter
from ukar.backends.scripted import ScriptedCompleter

__all__ = [
    "OllamaCompleter",
    "OpenAICompatCompleter",
    "ScriptedCompleter",
]
