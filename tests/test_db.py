from datetime import date

import pytest

from lattice.db import AtomNotFound, LatticeDB
from lattice.models import Atom


def make_atom(subject="Alpha", content="Alpha does X.", **kwargs) -> Atom:
    return Atom(kind="fact", source="user", subject=subject, content=content, **kwargs)


@pytest.fixture()
def db(tmp_path):
    return LatticeDB(lattice_dir=tmp_path)


class TestWrite:
    def test_write_creates_file(self, db, tmp_path):
        a = make_atom()
        db.write(a)
        assert (tmp_path / f"{a.atom_id}.md").exists()

    def test_write_overwrites(self, db):
        a = make_atom()
        db.write(a)
        a.content = "Updated content."
        db.write(a)
        restored = db.read(a.atom_id)
        assert restored.content == "Updated content."


class TestRead:
    def test_read_returns_atom(self, db):
        a = make_atom()
        db.write(a)
        restored = db.read(a.atom_id)
        assert restored.atom_id == a.atom_id
        assert restored.content == a.content

    def test_read_missing_raises(self, db):
        with pytest.raises(AtomNotFound):
            db.read("does-not-exist")


class TestList:
    def test_all_returns_all_atoms(self, db):
        a1 = make_atom(subject="A", content="A fact.")
        a2 = make_atom(subject="B", content="B fact.")
        db.write(a1)
        db.write(a2)
        ids = {a.atom_id for a in db.all()}
        assert a1.atom_id in ids
        assert a2.atom_id in ids

    def test_all_empty(self, db):
        assert db.all() == []

    def test_by_subject_case_insensitive(self, db):
        a = make_atom(subject="Project Falcon")
        db.write(a)
        result = db.by_subject("project falcon")
        assert any(r.atom_id == a.atom_id for r in result)

    def test_by_subject_excludes_others(self, db):
        a1 = make_atom(subject="Alpha")
        a2 = make_atom(subject="Beta", content="Beta fact.")
        db.write(a1)
        db.write(a2)
        result = db.by_subject("Alpha")
        assert all(r.subject == "Alpha" for r in result)

    def test_subjects_unique(self, db):
        for i in range(3):
            db.write(make_atom(subject="Alpha", content=f"Fact {i}."))
        db.write(make_atom(subject="Beta", content="Beta fact."))
        subjects = db.subjects()
        assert subjects.count("Alpha") == 1
        assert "Beta" in subjects


class TestSupersession:
    def test_supersede_marks_old_atom(self, db):
        old = make_atom(content="Old fact.")
        db.write(old)
        new = make_atom(content="New fact.")
        db.supersede(old.atom_id, new)
        old_restored = db.read(old.atom_id)
        assert old_restored.is_superseded is True
        assert old_restored.superseded_by == new.atom_id

    def test_supersede_links_new_atom(self, db):
        old = make_atom(content="Old fact.")
        db.write(old)
        new = make_atom(content="New fact.")
        db.supersede(old.atom_id, new)
        new_restored = db.read(new.atom_id)
        assert new_restored.supersedes == old.atom_id

    def test_supersede_missing_old_raises(self, db):
        new = make_atom(content="New fact.")
        with pytest.raises(AtomNotFound):
            db.supersede("nonexistent", new)


class TestSearch:
    def test_bm25_top_result_matches_query(self, db):
        db.write(make_atom(subject="Python", content="Python is a high-level programming language."))
        db.write(make_atom(subject="Rust", content="Rust is a systems programming language focused on safety."))
        db.write(make_atom(subject="Cooking", content="Pasta should be boiled in salted water."))
        results = db.search("programming language")
        subjects = [r.subject for r in results]
        assert subjects[0] in {"Python", "Rust"}

    def test_search_excludes_superseded(self, db):
        old = make_atom(subject="API", content="The API uses REST.")
        db.write(old)
        new = make_atom(subject="API", content="The API now uses GraphQL.")
        db.supersede(old.atom_id, new)
        results = db.search("API")
        ids = [r.atom_id for r in results]
        assert old.atom_id not in ids
        assert new.atom_id in ids

    def test_search_empty_db(self, db):
        assert db.search("anything") == []

    def test_search_returns_at_most_top_k(self, db):
        for i in range(10):
            db.write(make_atom(subject=f"Topic{i}", content=f"Fact number {i} about topic."))
        results = db.search("fact topic", top_k=3)
        assert len(results) <= 3


class TestAsOf:
    def test_as_of_excludes_expired_atom(self, db):
        expired = make_atom(
            subject="Price",
            content="Price is $10.",
            valid_until=date(2023, 12, 31),
        )
        db.write(expired)
        results = db.search("price", as_of=date(2024, 6, 1))
        assert all(r.atom_id != expired.atom_id for r in results)

    def test_as_of_includes_atom_without_bounds(self, db):
        a = make_atom(subject="Price", content="Price is $10.")
        db.write(a)
        results = db.search("price", as_of=date(2024, 6, 1))
        assert any(r.atom_id == a.atom_id for r in results)

    def test_as_of_excludes_future_atom(self, db):
        future = make_atom(
            subject="Feature",
            content="Feature launches next year.",
            valid_from=date(2025, 1, 1),
        )
        db.write(future)
        results = db.search("feature", as_of=date(2024, 1, 1))
        assert all(r.atom_id != future.atom_id for r in results)

    def test_as_of_includes_atom_within_bounds(self, db):
        a = make_atom(
            subject="Offer",
            content="Special offer valid this month.",
            valid_from=date(2024, 6, 1),
            valid_until=date(2024, 6, 30),
        )
        db.write(a)
        results = db.search("offer", as_of=date(2024, 6, 15))
        assert any(r.atom_id == a.atom_id for r in results)
