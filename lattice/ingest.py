from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from lattice.db import LatticeDB
from lattice.llm import complete
from lattice.models import Atom

_SYSTEM = """\
You are an expert knowledge extraction assistant. Your job is to decompose a piece of text into discrete, self-contained knowledge atoms.

Rules:
- Each atom captures ONE fact, belief, event, preference, or piece of information about ONE subject.
- Subjects should be concrete nouns or named entities (a person, project, concept, decision, etc.).
- kind: choose a descriptive label e.g. fact, event, preference, decision, belief, code, doc, task, goal, constraint.
- source: describe where this came from e.g. user, assistant, document, conversation, meeting, email.
- If a date appears in the text, resolve relative dates (e.g. "last Tuesday", "next month") to ISO 8601 (YYYY-MM-DD) using today's date.
- valid_from / valid_until: only set if the text explicitly implies temporal bounds. Otherwise null.
- content: a single, self-contained sentence or short paragraph. Do NOT reference "the text" or "the document" — write as a standalone fact.

Return a JSON array of objects. Each object must have these keys:
  subject, kind, source, content, valid_from, valid_until

No other keys. No markdown fences. Just raw JSON array.
"""


def _today_str() -> str:
    return date.today().isoformat()


def _parse_date(val: Any) -> date | None:
    if not val:
        return None
    if isinstance(val, date):
        return val
    try:
        return date.fromisoformat(str(val))
    except ValueError:
        return None


def _extract_atoms(text: str, metadata: dict) -> list[dict]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"Today's date: {_today_str()}\n\n---\n\n{text}",
        },
    ]
    raw = complete(messages)
    # Strip markdown fences if the model wraps anyway
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())
    atoms_data: list[dict] = json.loads(raw)
    source_override = metadata.get("source")
    for a in atoms_data:
        if source_override:
            a["source"] = source_override
        a["metadata"] = metadata
    return atoms_data


def _detect_supersession(db: LatticeDB, new_atom: Atom) -> str | None:
    """Return atom_id of an existing active atom on the same subject to supersede, or None."""
    existing = [a for a in db.by_subject(new_atom.subject) if not a.is_superseded]
    if not existing:
        return None

    candidates_text = "\n".join(
        f"[{a.atom_id}] {a.content}" for a in existing
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are deciding whether a new fact supersedes an existing fact. "
                "Supersession means the new fact contradicts or replaces the old one — not merely adds to it. "
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
    # Validate it's one of the candidates
    valid_ids = {a.atom_id for a in existing}
    return reply if reply in valid_ids else None


def ingest(source: str, metadata: dict | None = None, db: LatticeDB | None = None) -> dict:
    if db is None:
        db = LatticeDB()
    metadata = metadata or {}

    atoms_data = _extract_atoms(source, metadata)
    created_ids: list[str] = []

    for data in atoms_data:
        atom = Atom(
            kind=data.get("kind", "fact"),
            source=data.get("source", "user"),
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

        created_ids.append(atom.atom_id)

    return {"atoms_created": len(created_ids), "atom_ids": created_ids}
