from datetime import date

import pytest

from lattice.models import Atom


def make_atom(**kwargs) -> Atom:
    defaults = dict(kind="fact", source="user", subject="Test Subject", content="Test content.")
    return Atom(**{**defaults, **kwargs})


class TestAtomDefaults:
    def test_atom_id_generated(self):
        a = make_atom()
        assert a.atom_id
        assert len(a.atom_id) == 36  # UUID

    def test_unique_ids(self):
        a1, a2 = make_atom(), make_atom()
        assert a1.atom_id != a2.atom_id

    def test_default_temporal_null(self):
        a = make_atom()
        assert a.valid_from is None
        assert a.valid_until is None

    def test_default_supersession_false(self):
        a = make_atom()
        assert a.is_superseded is False
        assert a.superseded_by is None
        assert a.supersedes is None

    def test_default_metadata_empty(self):
        a = make_atom()
        assert a.metadata == {}

    def test_metadata_not_shared(self):
        # Default factory must not share the same dict across instances
        a1, a2 = make_atom(), make_atom()
        a1.metadata["x"] = 1
        assert "x" not in a2.metadata


class TestAtomValidation:
    def test_missing_kind_raises(self):
        with pytest.raises(Exception):
            Atom(source="user", subject="s", content="c")

    def test_missing_source_raises(self):
        with pytest.raises(Exception):
            Atom(kind="fact", subject="s", content="c")

    def test_missing_subject_raises(self):
        with pytest.raises(Exception):
            Atom(kind="fact", source="user", content="c")

    def test_missing_content_raises(self):
        with pytest.raises(Exception):
            Atom(kind="fact", source="user", subject="s")

    def test_date_fields_accept_date(self):
        a = make_atom(valid_from=date(2024, 1, 1), valid_until=date(2024, 12, 31))
        assert a.valid_from == date(2024, 1, 1)
        assert a.valid_until == date(2024, 12, 31)


class TestAtomRoundTrip:
    def test_roundtrip_minimal(self):
        a = make_atom()
        restored = Atom.from_markdown(a.to_markdown())
        assert restored.atom_id == a.atom_id
        assert restored.kind == a.kind
        assert restored.source == a.source
        assert restored.subject == a.subject
        assert restored.content == a.content
        assert restored.valid_from is None
        assert restored.valid_until is None
        assert restored.is_superseded is False
        assert restored.superseded_by is None
        assert restored.supersedes is None
        assert restored.metadata == {}

    def test_roundtrip_with_dates(self):
        a = make_atom(valid_from=date(2023, 6, 1), valid_until=date(2024, 6, 1))
        restored = Atom.from_markdown(a.to_markdown())
        assert restored.valid_from == date(2023, 6, 1)
        assert restored.valid_until == date(2024, 6, 1)

    def test_roundtrip_with_supersession(self):
        a = make_atom(is_superseded=True, superseded_by="abc", supersedes="xyz")
        restored = Atom.from_markdown(a.to_markdown())
        assert restored.is_superseded is True
        assert restored.superseded_by == "abc"
        assert restored.supersedes == "xyz"

    def test_roundtrip_with_metadata(self):
        a = make_atom(metadata={"url": "https://example.com", "author": "Alice"})
        restored = Atom.from_markdown(a.to_markdown())
        assert restored.metadata == {"url": "https://example.com", "author": "Alice"}

    def test_roundtrip_multiline_content(self):
        content = "First line.\nSecond line.\nThird line."
        a = make_atom(content=content)
        restored = Atom.from_markdown(a.to_markdown())
        assert restored.content.strip() == content

    def test_roundtrip_open_ended_kind_and_source(self):
        a = make_atom(kind="architecture-decision", source="meeting-notes-2024-01-15")
        restored = Atom.from_markdown(a.to_markdown())
        assert restored.kind == "architecture-decision"
        assert restored.source == "meeting-notes-2024-01-15"
