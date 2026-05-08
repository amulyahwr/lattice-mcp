from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from rank_bm25 import BM25Okapi

from lattice.models import Atom


class AtomNotFound(Exception):
    pass


class LatticeDB:
    def __init__(self, lattice_dir: str | Path | None = None) -> None:
        path = lattice_dir or os.environ.get("LATTICE_DIR", "./lattice")
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, atom_id: str) -> Path:
        return self.dir / f"{atom_id}.md"

    # ── write ─────────────────────────────────────────────────────────────

    def write(self, atom: Atom) -> None:
        self._path(atom.atom_id).write_text(atom.to_markdown(), encoding="utf-8")

    # ── read ──────────────────────────────────────────────────────────────

    def read(self, atom_id: str) -> Atom:
        p = self._path(atom_id)
        if not p.exists():
            raise AtomNotFound(atom_id)
        return Atom.from_markdown(p.read_text(encoding="utf-8"))

    # ── list ──────────────────────────────────────────────────────────────

    def all(self) -> list[Atom]:
        atoms = []
        for p in sorted(self.dir.glob("*.md")):
            try:
                atoms.append(Atom.from_markdown(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return atoms

    def by_subject(self, subject: str) -> list[Atom]:
        return [a for a in self.all() if a.subject.lower() == subject.lower()]

    def subjects(self) -> list[str]:
        seen: set[str] = set()
        result = []
        for a in self.all():
            if a.subject not in seen:
                seen.add(a.subject)
                result.append(a.subject)
        return result

    # ── supersession ──────────────────────────────────────────────────────

    def supersede(self, old_id: str, new_atom: Atom) -> None:
        old = self.read(old_id)
        old.is_superseded = True
        old.superseded_by = new_atom.atom_id
        self.write(old)
        new_atom.supersedes = old_id
        self.write(new_atom)

    # ── search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        as_of: date | None = None,
        top_k: int = 20,
    ) -> list[Atom]:
        atoms = [a for a in self.all() if not a.is_superseded]

        if as_of is not None:
            atoms = [
                a
                for a in atoms
                if (a.valid_from is None or a.valid_from <= as_of)
                and (a.valid_until is None or a.valid_until >= as_of)
            ]

        if not atoms:
            return []

        corpus = [f"{a.subject} {a.content}" for a in atoms]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())

        ranked = sorted(zip(scores, atoms), key=lambda x: x[0], reverse=True)
        return [a for _, a in ranked[:top_k]]
