from __future__ import annotations

import json
import re
from datetime import date

from lattice.db import LatticeDB
from lattice.llm import complete
from lattice.models import Atom

_SYSTEM = """\
You are a relevance filter. Given a user query and a list of knowledge atoms, return only the atom_ids that are genuinely relevant to answering the query.

Rules:
- Include an atom if it directly answers, supports, or provides useful context for the query.
- Exclude atoms that are only tangentially related or clearly off-topic.
- Preserve the ranking: most relevant first.
- Return a JSON array of atom_id strings only. No markdown fences. No explanation.
"""


def _atom_to_text(a: Atom) -> str:
    return f"[{a.atom_id}] subject={a.subject!r} kind={a.kind!r}\n{a.content}"


def select(
    query: str,
    as_of: date | None = None,
    db: LatticeDB | None = None,
    top_k: int = 20,
) -> list[dict]:
    if db is None:
        db = LatticeDB()

    candidates = db.search(query, as_of=as_of, top_k=top_k)
    if not candidates:
        return []

    candidates_text = "\n\n".join(_atom_to_text(a) for a in candidates)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"Query: {query}\n\nAtoms:\n{candidates_text}",
        },
    ]
    raw = complete(messages)
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        ranked_ids: list[str] = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to BM25 order if LLM returns garbage
        ranked_ids = [a.atom_id for a in candidates]

    id_to_atom = {a.atom_id: a for a in candidates}
    result = []
    for atom_id in ranked_ids:
        a = id_to_atom.get(atom_id)
        if a:
            result.append({
                "atom_id": a.atom_id,
                "subject": a.subject,
                "content": a.content,
                "kind": a.kind,
                "source": a.source,
                "valid_from": a.valid_from.isoformat() if a.valid_from else None,
                "valid_until": a.valid_until.isoformat() if a.valid_until else None,
            })
    return result
