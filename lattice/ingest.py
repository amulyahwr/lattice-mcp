from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from lattice.db import LatticeDB
from lattice.llm import complete
from lattice.models import Atom

_SYSTEM = """\
You are a knowledge extraction agent. Read a piece of text and extract all durable facts, \
decisions, constraints, goals, events, and preferences into discrete atoms.

Rules for each atom:
  - content    : a single self-contained statement. Do NOT reference "the text" or "the document" — \
write as a standalone fact a reader could understand without the original source.
  - kind       : a short descriptive label, e.g. fact, decision, constraint, goal, event, preference, belief
  - subject    : short canonical noun phrase identifying what the atom is about, \
e.g. "Project Alpha deadline", "auth module", "database schema", "deployment process", "API rate limit"
  - source     : where this came from — use the value from the caller's metadata if provided, \
otherwise infer: "document" for prose/notes/docs, "code" for code snippets, "conversation" for chat logs

Subject naming rules:
  - Use the most general term that still uniquely identifies the topic.
  - Use CONSISTENT subject phrasing across atoms — e.g. always "auth module" not sometimes "authentication" \
or "auth system". Consistent subjects enable supersession when the same fact is updated later.
  - valid_from / valid_until: only set if the text explicitly implies temporal bounds (e.g. "valid until end of Q2", \
"starting next Monday"). Otherwise null.
  - Resolve relative dates (e.g. "last Tuesday", "next month") to ISO 8601 (YYYY-MM-DD) using today's date.
  - Do NOT extract generic advice or universally-applicable facts that apply to everyone.

Return a JSON array of objects. Each object must have exactly these keys:
  subject, kind, source, content, valid_from, valid_until

No other keys. No markdown fences. Just raw JSON array.
"""

# ── date resolution ───────────────────────────────────────────────────────────

_RELATIVE_PATTERNS: list[tuple[re.Pattern, Any]] = [
    (re.compile(r'\b(\d+)\s+days?\s+ago\b', re.IGNORECASE),
     lambda m, ref: (ref - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")),
    (re.compile(r'\b(\d+)\s+weeks?\s+ago\b', re.IGNORECASE),
     lambda m, ref: (ref - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")),
    (re.compile(r'\b(\d+)\s+months?\s+ago\b', re.IGNORECASE),
     lambda m, ref: (ref - timedelta(days=int(m.group(1)) * 30)).strftime("%Y-%m-%d")),
    (re.compile(r'\blast\s+year\b', re.IGNORECASE),
     lambda _, ref: str(ref.year - 1)),
    (re.compile(r'\blast\s+week\b', re.IGNORECASE),
     lambda _, ref: (ref - timedelta(weeks=1)).strftime("%Y-%m-%d")),
    (re.compile(r'\blast\s+month\b', re.IGNORECASE),
     lambda _, ref: (ref - timedelta(days=30)).strftime("%Y-%m-%d")),
    (re.compile(r'\byesterday\b', re.IGNORECASE),
     lambda _, ref: (ref - timedelta(days=1)).strftime("%Y-%m-%d")),
]

_WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_PATTERN = re.compile(
    r'\blast\s+(' + '|'.join(_WEEKDAY_NAMES) + r')\b', re.IGNORECASE
)


def _resolve_dates(content: str, ref: datetime) -> str:
    """Replace relative date expressions in atom content with absolute ISO dates."""
    for pattern, resolver in _RELATIVE_PATTERNS:
        content = pattern.sub(lambda m, _r=ref, _res=resolver: _res(m, _r), content)

    def _resolve_weekday(m: re.Match) -> str:
        day_name = m.group(1).lower()
        target_dow = _WEEKDAY_NAMES.index(day_name)
        days_back = (ref.weekday() - target_dow) % 7
        if days_back == 0:
            days_back = 7
        return (ref - timedelta(days=days_back)).strftime("%Y-%m-%d")

    content = _WEEKDAY_PATTERN.sub(_resolve_weekday, content)
    return content


# ── helpers ───────────────────────────────────────────────────────────────────

def _today() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_date(val: Any) -> date | None:
    if not val:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    try:
        return date.fromisoformat(str(val)[:10])
    except ValueError:
        return None


def _extract_atoms(text: str, metadata: dict, ref: datetime) -> list[dict]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"Today's date: {ref.date().isoformat()}\n\n---\n\n{text}",
        },
    ]
    raw = complete(messages)
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())
    atoms_data: list[dict] = json.loads(raw)

    source_override = metadata.get("source")
    for a in atoms_data:
        if source_override:
            a["source"] = source_override
        # Resolve relative dates in content now that we have the ref datetime
        a["content"] = _resolve_dates(a["content"], ref)
        a["metadata"] = metadata
    return atoms_data


def _detect_supersession(db: LatticeDB, new_atom: Atom) -> str | None:
    """Return atom_id to supersede, or None.

    Uses subject registry for O(1) lookup, then asks LLM only if an existing
    active atom is found on the same subject.
    """
    # Fast path: subject registry
    existing_id = db.lookup_subject(new_atom.subject)
    if existing_id:
        try:
            existing = db.read(existing_id)
            if existing.is_superseded:
                return None
        except Exception:
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are deciding whether a new fact supersedes an existing fact. "
                    "Supersession means the new fact contradicts or replaces the old one — "
                    "not merely adds to it. Reply with the atom_id that is superseded, or null."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"New fact: {new_atom.content}\n\n"
                    f"Existing fact [{existing_id}]: {existing.content}\n\n"
                    "Reply with just the atom_id string or null."
                ),
            },
        ]
        reply = complete(messages).strip().strip('"').strip("'")
        if reply.lower() == "null" or not reply:
            return None
        return existing_id if reply == existing_id else None

    # Slow path: no registry entry, scan by subject (handles hand-edited atoms)
    existing = [a for a in db.by_subject(new_atom.subject) if not a.is_superseded]
    if not existing:
        return None

    candidates_text = "\n".join(f"[{a.atom_id}] {a.content}" for a in existing)
    messages = [
        {
            "role": "system",
            "content": (
                "You are deciding whether a new fact supersedes an existing fact. "
                "Reply with the atom_id that is superseded, or null if none."
            ),
        },
        {
            "role": "user",
            "content": (
                f"New fact: {new_atom.content}\n\n"
                f"Existing facts about '{new_atom.subject}':\n{candidates_text}\n\n"
                "Reply with just the atom_id string or null."
            ),
        },
    ]
    reply = complete(messages).strip().strip('"').strip("'")
    if reply.lower() == "null" or not reply:
        return None
    valid_ids = {a.atom_id for a in existing}
    return reply if reply in valid_ids else None


def ingest(source: str, metadata: dict | None = None, db: LatticeDB | None = None) -> dict:
    if db is None:
        db = LatticeDB()
    metadata = metadata or {}
    ref = _today()

    atoms_data = _extract_atoms(source, metadata, ref)
    created_ids: list[str] = []

    for data in atoms_data:
        atom = Atom(
            kind=data.get("kind", "fact"),
            source=data.get("source", "document"),
            subject=data["subject"],
            content=data["content"],
            valid_from=_parse_date(data.get("valid_from")),
            valid_until=_parse_date(data.get("valid_until")),
            metadata=data.get("metadata", {}),
        )

        old_id = _detect_supersession(db, atom)
        if old_id:
            db.supersede(old_id, atom)
        else:
            db.write(atom)

        # Register subject → atom_id for fast future supersession lookups
        if atom.subject:
            db.register_subject(atom.subject, atom.atom_id)

        created_ids.append(atom.atom_id)

    return {"atoms_created": len(created_ids), "atom_ids": created_ids}
