"""Plain-text trace for the CLI and for tool results."""

from __future__ import annotations

from ukar.schema import Trace


USED_LINE = "UKAR was used for this reply."


def finish_reply(text: str, *, used: bool) -> str:
    """Append the UKAR line once when the tool actually ran."""
    body = (text or "").strip() or "(empty reply)"
    if used and USED_LINE not in body:
        body = f"{body}\n\n{USED_LINE}"
    return body + "\n"


def format_trace(trace: Trace) -> str:
    lines: list[str] = [
        "UKAR",
        "The local model names missing facts. Sources return those facts. The same model answers.",
        "",
        "Question",
        f"  {trace.question or '(empty)'}",
        "",
        f"Status  {trace.status}",
        f"Backend {trace.backend or '(unset)'}",
    ]
    gap = trace.gap or {}
    if gap:
        lines.append("")
        lines.append(
            f"Confidence before acquisition  {gap.get('confidence')}"
        )
        known = gap.get("known") or []
        if known:
            lines.append("Already trusted")
            for item in known:
                lines.append(f"  - {item}")
        needs = gap.get("needs") or []
        if needs:
            lines.append("Missing facts")
            for need in needs:
                keys = ", ".join(need.get("keys") or [])
                why = str(need.get("why") or "").strip()
                reason = f" — {why}" if why else ""
                lines.append(
                    f"  {need.get('id'):<16} {need.get('scope'):<10} {need.get('statement')}{reason}  [{keys}]"
                )
    if trace.facts:
        lines.append("")
        lines.append("Acquired")
        for fact in trace.facts:
            lines.append(
                f"  {fact.key:<22} {fact.value:<8} {fact.source_id:<16} "
                f"{fact.cents:g}¢  {fact.privacy}"
            )
            lines.append(f"    {fact.text}")
    if trace.considerations:
        lines.append("")
        lines.append("How the sources were chosen")
        for item in trace.considerations:
            needs = ", ".join(item.need_ids)
            lines.append(
                f"  {item.decision:<10} {item.source_id:<16} {item.cents:g}¢  "
                f"{item.privacy:<14} {needs}"
            )
            lines.append(f"    {item.reason}")
    if trace.questions:
        lines.append("")
        lines.append("Still needed before an answer")
        for question in trace.questions:
            lines.append(f"  - {question}")
    if trace.answer:
        lines.append("")
        lines.append("Local answer")
        for line in trace.answer.splitlines():
            lines.append(f"  {line}")
    if trace.notes:
        lines.append("")
        for note in trace.notes:
            lines.append(note)
    if trace.status in {"answered", "waiting_on_user"}:
        lines.append("")
        lines.append(USED_LINE)
    return "\n".join(lines).rstrip() + "\n"
