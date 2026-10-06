"""Uncertainty-aware knowledge acquisition router.

The local model does not decide whether a bigger model should answer.
It names the facts that would let it answer. Sources return those facts.
The same local model then answers.

Fact sources are local tables and facts you already have.
llama.cpp and Ollama are accepted only on this machine.
"""

from ukar.router import Router, default_sources
from ukar.schema import Fact, GapReport, KnowledgeNeed, Trace

__version__ = "0.1.0"
__all__ = [
    "Fact",
    "GapReport",
    "KnowledgeNeed",
    "Router",
    "Trace",
    "default_sources",
    "__version__",
]
