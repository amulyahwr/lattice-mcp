from __future__ import annotations


def format_session(turns: list[dict], session_id: str, timestamp: str) -> str:
    """Convert a list of {role, content} turns into an ingest()-ready text string."""
    header = f"[Session: {session_id} | Date: {timestamp}]"
    lines = [header]
    for turn in turns:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
