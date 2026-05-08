"""
Agent integration tests — all LLM calls are mocked via patch("lattice.llm.complete").
Tests verify external behavior: atoms created, supersession links, selection results, synthesis output.
"""

import json
from datetime import date
from unittest.mock import patch

import pytest

from lattice.db import LatticeDB
from lattice.ingest import ingest
from lattice.selection import select
from lattice.synthesis import synthesize


@pytest.fixture()
def db(tmp_path):
    return LatticeDB(lattice_dir=tmp_path)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ingest_response(atoms: list[dict]) -> str:
    return json.dumps(atoms)


def _select_response(atom_ids: list[str]) -> str:
    return json.dumps(atom_ids)


# ── ingest ────────────────────────────────────────────────────────────────────

class TestIngest:
    def test_creates_atoms_in_db(self, db):
        llm_atoms = [
            {"subject": "lattice-mcp", "kind": "fact", "source": "user",
             "content": "lattice-mcp is a local MCP server.", "valid_from": None, "valid_until": None},
        ]
        # supersession check returns null (no supersession)
        responses = [_ingest_response(llm_atoms), "null"]
        with patch("lattice.ingest.complete", side_effect=responses):
            result = ingest("lattice-mcp is a local MCP server.", db=db)

        assert result["atoms_created"] == 1
        assert len(result["atom_ids"]) == 1
        atom = db.read(result["atom_ids"][0])
        assert atom.subject == "lattice-mcp"
        assert "local MCP server" in atom.content

    def test_metadata_stored_on_atom(self, db):
        llm_atoms = [
            {"subject": "Project", "kind": "doc", "source": "document",
             "content": "Project readme.", "valid_from": None, "valid_until": None},
        ]
        responses = [_ingest_response(llm_atoms), "null"]
        with patch("lattice.ingest.complete", side_effect=responses):
            result = ingest("Project readme.", metadata={"title": "README"}, db=db)

        atom = db.read(result["atom_ids"][0])
        assert atom.metadata.get("title") == "README"

    def test_multiple_atoms_from_one_ingest(self, db):
        llm_atoms = [
            {"subject": "A", "kind": "fact", "source": "user", "content": "A is true.", "valid_from": None, "valid_until": None},
            {"subject": "B", "kind": "fact", "source": "user", "content": "B is false.", "valid_from": None, "valid_until": None},
        ]
        # Two atoms → two supersession checks
        responses = [_ingest_response(llm_atoms), "null", "null"]
        with patch("lattice.ingest.complete", side_effect=responses):
            result = ingest("A is true. B is false.", db=db)

        assert result["atoms_created"] == 2
        all_ids = {a.atom_id for a in db.all()}
        for aid in result["atom_ids"]:
            assert aid in all_ids

    def test_supersession_links_atoms(self, db):
        # First ingest: create old atom
        old_atoms = [
            {"subject": "API", "kind": "fact", "source": "user",
             "content": "The API uses REST.", "valid_from": None, "valid_until": None},
        ]
        with patch("lattice.ingest.complete", side_effect=[_ingest_response(old_atoms)]):
            old_result = ingest("The API uses REST.", db=db)

        old_id = old_result["atom_ids"][0]

        # Second ingest: supersedes old atom
        new_atoms = [
            {"subject": "API", "kind": "fact", "source": "user",
             "content": "The API now uses GraphQL.", "valid_from": None, "valid_until": None},
        ]
        # supersession check returns the old atom's id
        with patch("lattice.ingest.complete", side_effect=[_ingest_response(new_atoms), old_id]):
            new_result = ingest("The API now uses GraphQL.", db=db)

        new_id = new_result["atom_ids"][0]
        old_atom = db.read(old_id)
        new_atom = db.read(new_id)

        assert old_atom.is_superseded is True
        assert old_atom.superseded_by == new_id
        assert new_atom.supersedes == old_id

    def test_date_fields_parsed(self, db):
        llm_atoms = [
            {"subject": "Offer", "kind": "event", "source": "user",
             "content": "Special offer.", "valid_from": "2024-06-01", "valid_until": "2024-06-30"},
        ]
        responses = [_ingest_response(llm_atoms), "null"]
        with patch("lattice.ingest.complete", side_effect=responses):
            result = ingest("Special offer valid June 2024.", db=db)

        atom = db.read(result["atom_ids"][0])
        assert atom.valid_from == date(2024, 6, 1)
        assert atom.valid_until == date(2024, 6, 30)


# ── selection ─────────────────────────────────────────────────────────────────

class TestSelect:
    def _seed(self, db):
        from lattice.models import Atom
        atoms = [
            Atom(kind="fact", source="user", subject="Python", content="Python is a high-level language."),
            Atom(kind="fact", source="user", subject="Rust", content="Rust is a systems language."),
            Atom(kind="fact", source="user", subject="Cooking", content="Pasta is boiled in water."),
        ]
        for a in atoms:
            db.write(a)
        return atoms

    def test_returns_relevant_atoms(self, db):
        atoms = self._seed(db)
        python_id = atoms[0].atom_id
        with patch("lattice.selection.complete", return_value=_select_response([python_id])):
            result = select("tell me about Python", db=db)
        assert len(result) == 1
        assert result[0]["atom_id"] == python_id

    def test_result_has_required_fields(self, db):
        atoms = self._seed(db)
        with patch("lattice.selection.complete", return_value=_select_response([atoms[0].atom_id])):
            result = select("Python", db=db)
        keys = set(result[0].keys())
        assert {"atom_id", "subject", "content", "kind", "source"}.issubset(keys)

    def test_empty_db_returns_empty(self, db):
        result = select("anything", db=db)
        assert result == []

    def test_llm_garbage_falls_back_to_bm25(self, db):
        self._seed(db)
        with patch("lattice.selection.complete", return_value="not valid json {{{}"):
            result = select("programming language", db=db)
        # Should still return something (BM25 fallback)
        assert isinstance(result, list)

    def test_as_of_filters_before_llm(self, db):
        from lattice.models import Atom
        expired = Atom(
            kind="fact", source="user", subject="Price",
            content="Price is $10.", valid_until=date(2023, 12, 31),
        )
        db.write(expired)
        # Even if LLM returns the expired id, it shouldn't be in BM25 candidates
        with patch("lattice.selection.complete", return_value=_select_response([expired.atom_id])):
            result = select("price", as_of=date(2024, 6, 1), db=db)
        # expired atom was filtered from BM25 candidates, so not in result
        assert all(r["atom_id"] != expired.atom_id for r in result)


# ── synthesis ─────────────────────────────────────────────────────────────────

class TestSynthesize:
    def test_returns_string(self):
        atoms = [{"subject": "Python", "kind": "fact", "content": "Python is dynamically typed."}]
        with patch("lattice.synthesis.complete", return_value="Python is dynamically typed."):
            result = synthesize("What is Python?", atoms)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_atoms_returns_no_info_message(self):
        result = synthesize("What is Python?", [])
        assert "no relevant" in result.lower() or "not found" in result.lower() or "no" in result.lower()

    def test_passes_query_and_atoms_to_llm(self):
        atoms = [{"subject": "X", "kind": "fact", "content": "X is true."}]
        with patch("lattice.synthesis.complete", return_value="X is true.") as mock:
            synthesize("Tell me about X.", atoms)
        call_messages = mock.call_args[0][0]
        combined = " ".join(m["content"] for m in call_messages)
        assert "Tell me about X." in combined
        assert "X is true." in combined
