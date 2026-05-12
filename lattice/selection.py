from __future__ import annotations

from datetime import date

from lattice.db import LatticeDB
from lattice.models import Atom


def _atom_to_dict(a: Atom) -> dict:
    return {
        "atom_id": a.atom_id,
        "subject": a.subject,
        "kind": a.kind,
        "source": a.source,
        "content": a.content,
        "valid_from": a.valid_from.isoformat() if a.valid_from else None,
        "valid_until": a.valid_until.isoformat() if a.valid_until else None,
        "is_superseded": a.is_superseded,
        "supersedes": a.supersedes,
        "superseded_by": a.superseded_by,
        # Keep flat provenance fields for product/synthesis callers.
        "ingested_at": a.ingested_at.isoformat() if a.ingested_at else None,
        "observed_at": a.observed_at.isoformat() if a.observed_at else None,
        "source_id": a.source_id,
        "source_title": a.source_title,
        "session_id": a.session_id,
        "segment_id": a.segment_id,
        "source_type": a.source_type,
        "source_span": a.source_span,
        # Mirror eval debug payload shape so select/bm25 modes differ only by
        # retrieval behavior, not metadata structure.
        "provenance": {
            "source_id": a.source_id,
            "source_title": a.source_title,
            "source_type": a.source_type,
            "session_id": a.session_id,
            "segment_id": a.segment_id,
            "source_span": a.source_span,
            "observed_at": a.observed_at.isoformat() if a.observed_at else None,
            "ingested_at": a.ingested_at.isoformat() if a.ingested_at else None,
        },
        "dedup": {
            "content_hash": a.content_hash,
            "normalized_content_hash": a.normalized_content_hash,
        },
    }


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

    return [_atom_to_dict(a) for a in candidates]
