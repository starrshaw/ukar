"""Fact sources that ship with the package.

The package does not include a topic. The only built-in source is the facts
the caller already passed in. Add your own tables beside that when you have
rows that should stay on the computer.
"""

from __future__ import annotations

from ukar.sources import GivenFacts


def default_sources() -> list:
    """Facts the caller already gave. No bundled topic tables."""
    return [GivenFacts()]
