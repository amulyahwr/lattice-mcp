from __future__ import annotations

import json
from datetime import date

from pydantic import BaseModel

from lattice.db import LatticeDB, _query_words
from lattice.llm import complete
from lattice.models import Atom


class _Selection(BaseModel):
    atom_ids: list[str]


_SYSTEM = """\
You are a knowledge retrieval agent. Given a query and a list of knowledge atoms, \
return the atom_ids that are relevant to answering the query.

Strategy — cover at least 3 distinct angles before concluding nothing is relevant:
  Angle 1 — Exact terms: look for atoms containing the key nouns and verbs from the query.
  Angle 2 — Synonyms and related concepts: consider alternate phrasings
             (e.g. "car" → "vehicle", "drive"; "phone" → "mobile", "device").
  Angle 3 — Topic browse: scan subjects for anything that could be related.
  Angle 4 (if still sparse) — err toward inclusion; synthesis handles filtering.

Additional rules:
- Include ALL atom_ids that directly answer, support, or provide useful context.
- For temporal questions ("after X", "before Y", "first", "last"):
    Find the anchor event and include BOTH the anchor atom and the answer atom.
- Err heavily on the side of inclusion. Synthesis handles final filtering and reasoning.
- Preserve ranking: most relevant first.

Return a JSON object with an `atom_ids` key containing an array of atom_id strings, ranked most-relevant first.
"""


def _atom_to_text(a: Atom) -> str:
    vf = a.valid_from.isoformat() if a.valid_from else "null"
    vu = a.valid_until.isoformat() if a.valid_until else "null"
    return (
        f"[{a.atom_id}] subject={a.subject!r} kind={a.kind!r} "
        f"valid_from={vf} valid_until={vu}\n{a.content}"
    )


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
    raw = complete(messages, text_format=_Selection)
    try:
        ranked_ids: list[str] = json.loads(raw)["atom_ids"]
    except (json.JSONDecodeError, KeyError):
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
