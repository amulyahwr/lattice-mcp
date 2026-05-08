from __future__ import annotations

from datetime import date

from lattice.db import AtomNotFound, LatticeDB
from lattice.models import Atom


def search_atoms(
    db: LatticeDB,
    query: str,
    as_of: date | None = None,
    top_k: int = 20,
) -> list[dict]:
    atoms = db.search(query, as_of=as_of, top_k=top_k)
    return [_atom_summary(a) for a in atoms]


def read_atom(db: LatticeDB, atom_id: str) -> dict:
    atom = db.read(atom_id)
    return _atom_full(atom)


def list_subjects(db: LatticeDB) -> list[str]:
    return db.subjects()


def list_all_atoms(db: LatticeDB) -> list[dict]:
    return [_atom_summary(a) for a in db.all()]


def _atom_summary(a: Atom) -> dict:
    return {
        "atom_id": a.atom_id,
        "subject": a.subject,
        "kind": a.kind,
        "source": a.source,
        "valid_from": a.valid_from.isoformat() if a.valid_from else None,
        "valid_until": a.valid_until.isoformat() if a.valid_until else None,
        "is_superseded": a.is_superseded,
        "content": a.content,
    }


def _atom_full(a: Atom) -> dict:
    return {
        **_atom_summary(a),
        "superseded_by": a.superseded_by,
        "supersedes": a.supersedes,
        "metadata": a.metadata,
    }
