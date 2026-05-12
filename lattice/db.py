from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path

from rank_bm25 import BM25Okapi

from lattice.models import Atom

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "this", "that",
}


def _query_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOPWORDS]


def _normalized_subject(subject: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", subject.lower()))


class AtomNotFound(Exception):
    pass


class LatticeDB:
    def __init__(self, lattice_dir: str | Path | None = None) -> None:
        path = lattice_dir or os.environ.get("LATTICE_DIR", "./lattice")
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._atom_cache: dict[str, Atom] = {}
        self._subjects_cache: dict[str, str] | None = None

    @property
    def _subjects_file(self) -> Path:
        return self.dir / "subjects.json"

    # ── subject registry ──────────────────────────────────────────────────

    def _load_subjects(self) -> dict[str, str]:
        if self._subjects_cache is not None:
            return self._subjects_cache
        data = json.loads(self._subjects_file.read_text()) if self._subjects_file.exists() else {}
        self._subjects_cache = data
        return data

    def register_subject(self, subject: str, atom_id: str) -> str | None:
        """Map subject → atom_id. Returns displaced atom_id if subject already existed."""
        key = subject.lower().strip()
        subjects = self._load_subjects()
        old_id = subjects.get(key)
        subjects[key] = atom_id
        self._write_json_atomic(self._subjects_file, subjects)
        self._subjects_cache = subjects
        return old_id if old_id != atom_id else None

    def lookup_subject(self, subject: str) -> str | None:
        return self._load_subjects().get(subject.lower().strip())

    # ── path helpers ──────────────────────────────────────────────────────

    def _path(self, atom_id: str) -> Path:
        return self.dir / f"{atom_id}.md"

    def _write_json_atomic(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            Path(tmp_name).replace(path)
        finally:
            tmp = Path(tmp_name)
            if tmp.exists():
                tmp.unlink()

    # ── write ─────────────────────────────────────────────────────────────

    def write(self, atom: Atom) -> None:
        text = atom.to_markdown()
        path = self._path(atom.atom_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            Path(tmp_name).replace(path)
        finally:
            tmp = Path(tmp_name)
            if tmp.exists():
                tmp.unlink()
        self._atom_cache[atom.atom_id] = atom

    # ── read ──────────────────────────────────────────────────────────────

    def read(self, atom_id: str) -> Atom:
        if atom_id in self._atom_cache:
            return self._atom_cache[atom_id]
        p = self._path(atom_id)
        if not p.exists():
            raise AtomNotFound(atom_id)
        atom = Atom.from_markdown(p.read_text(encoding="utf-8"))
        self._atom_cache[atom_id] = atom
        return atom

    def preload(self) -> None:
        """Bulk-read all atom files into cache."""
        for p in self.dir.glob("*.md"):
            atom_id = p.stem
            if atom_id not in self._atom_cache:
                try:
                    atom = Atom.from_markdown(p.read_text(encoding="utf-8"))
                    self._atom_cache[atom_id] = atom
                except Exception:
                    pass
        _ = self._load_subjects()

    # ── list ──────────────────────────────────────────────────────────────

    def all(self) -> list[Atom]:
        atoms = []
        for p in sorted(self.dir.glob("*.md")):
            atom_id = p.stem
            if atom_id in self._atom_cache:
                atoms.append(self._atom_cache[atom_id])
            else:
                try:
                    atom = Atom.from_markdown(p.read_text(encoding="utf-8"))
                    self._atom_cache[atom_id] = atom
                    atoms.append(atom)
                except Exception:
                    pass
        return atoms

    def by_subject(self, subject: str) -> list[Atom]:
        return [a for a in self.all() if a.subject.lower() == subject.lower()]

    def find_by_normalized_hash(self, normalized_content_hash: str) -> Atom | None:
        for atom in self.all():
            if atom.normalized_content_hash == normalized_content_hash:
                return atom
        return None

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

    # ── retrieval packs ──────────────────────────────────────────────────

    def _valid_atoms(
        self,
        as_of: date | None = None,
        include_superseded: bool = False,
    ) -> list[Atom]:
        atoms = (
            self.all()
            if include_superseded
            else [a for a in self.all() if not a.is_superseded]
        )

        if as_of is not None:
            atoms = [
                a
                for a in atoms
                if (a.valid_from is None or a.valid_from <= as_of)
                and (a.valid_until is None or a.valid_until >= as_of)
            ]

        return atoms

    @staticmethod
    def _source_order_key(atom: Atom) -> tuple:
        span = atom.source_span or {}
        return (
            atom.observed_at.isoformat() if atom.observed_at else "",
            atom.source_id or "",
            atom.session_id or "",
            atom.segment_id or "",
            span.get("start", -1),
            span.get("end", -1),
            atom.atom_id,
        )

    @staticmethod
    def _add_unique(target: list[Atom], seen: set[str], atoms: list[Atom]) -> None:
        for atom in atoms:
            if atom.atom_id in seen:
                continue
            target.append(atom)
            seen.add(atom.atom_id)

    def evidence_pack(
        self,
        seed: Atom,
        as_of: date | None = None,
        nearby_window: int = 2,
        subject_limit: int = 6,
    ) -> list[Atom]:
        """Expand a BM25 seed into deterministic local evidence.

        This stays file-local: no graph sidecar required. It uses provenance
        already stored on atoms to recover source/segment context.
        """
        atoms = self._valid_atoms(as_of=as_of, include_superseded=True)
        by_id = {atom.atom_id: atom for atom in atoms}
        pack: list[Atom] = []
        seen: set[str] = set()

        seed = by_id.get(seed.atom_id, seed)
        self._add_unique(pack, seen, [seed])

        if seed.segment_id:
            same_segment = [
                atom for atom in atoms
                if atom.atom_id != seed.atom_id
                and atom.segment_id == seed.segment_id
                and (
                    (seed.source_id and atom.source_id == seed.source_id)
                    or (seed.session_id and atom.session_id == seed.session_id)
                    or (not seed.source_id and not seed.session_id)
                )
            ]
            self._add_unique(
                pack,
                seen,
                sorted(same_segment, key=self._source_order_key),
            )

        same_source = [
            atom for atom in atoms
            if atom.atom_id != seed.atom_id
            and (
                (seed.source_id and atom.source_id == seed.source_id)
                or (seed.session_id and atom.session_id == seed.session_id)
            )
        ]
        same_source_sorted = sorted(
            same_source + [seed],
            key=self._source_order_key,
        )
        seed_index = next(
            (
                i for i, atom in enumerate(same_source_sorted)
                if atom.atom_id == seed.atom_id
            ),
            -1,
        )
        if seed_index >= 0:
            start = max(0, seed_index - nearby_window)
            end = min(len(same_source_sorted), seed_index + nearby_window + 1)
            nearby = [
                atom for atom in same_source_sorted[start:end]
                if atom.atom_id != seed.atom_id
            ]
            self._add_unique(pack, seen, nearby)

        normalized = _normalized_subject(seed.subject)
        if normalized:
            same_subject = [
                atom for atom in atoms
                if atom.atom_id != seed.atom_id
                and _normalized_subject(atom.subject) == normalized
            ]
            self._add_unique(
                pack,
                seen,
                sorted(same_subject, key=self._source_order_key)[:subject_limit],
            )

        linked_ids = {
            atom_id for atom_id in (seed.supersedes, seed.superseded_by) if atom_id
        }
        linked_ids.update(
            atom.atom_id for atom in atoms
            if atom.supersedes == seed.atom_id or atom.superseded_by == seed.atom_id
        )
        linked = [
            by_id[atom_id] for atom_id in sorted(linked_ids)
            if atom_id in by_id
        ]
        self._add_unique(pack, seen, linked)

        return pack

    # ── search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        as_of: date | None = None,
        top_k: int = 20,
    ) -> list[Atom]:
        atoms = self._valid_atoms(as_of=as_of)

        if not atoms:
            return []

        words = _query_words(query)
        if not words:
            return atoms[:top_k]

        corpus = [_query_words(f"{a.subject} {a.content}") for a in atoms]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(words)

        ranked = sorted(zip(scores, atoms), key=lambda x: x[0], reverse=True)
        return [a for _, a in ranked[:top_k]]
