from __future__ import annotations

import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Optional

from pydantic import BaseModel

from lattice.db import LatticeDB
from lattice.llm import complete
from lattice.models import Atom


class _ExtractedAtom(BaseModel):
    subject: str
    kind: str
    source: str
    content: str
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


class _AtomList(BaseModel):
    atoms: list[_ExtractedAtom]


class _SupersessionResult(BaseModel):
    superseded_atom_id: str | None


_SYSTEM = """\
You are a knowledge extraction agent. Read a piece of text and extract all durable facts, \
decisions, constraints, goals, events, and preferences into discrete atoms.

Rules for each atom:
  - content    : a single self-contained statement. Do NOT reference "the text" or "the document" — \
write as a standalone fact a reader could understand without the original source.
  - kind       : a short descriptive label, e.g. fact, decision, constraint, goal, event, preference, belief
  - subject    : short canonical noun phrase identifying what the atom is about, \
e.g. "Project Alpha deadline", "auth module", "database schema", "deployment process", "API rate limit"
  - source     : where this came from — use the value from the caller's metadata if provided, \
otherwise infer: "document" for prose/notes/docs, "code" for code snippets, "conversation" for chat logs

Subject naming rules:
  - Use the most general term that still uniquely identifies the topic.
  - Use CONSISTENT subject phrasing across atoms — e.g. always "auth module" not sometimes "authentication" \
or "auth system". Consistent subjects enable supersession when the same fact is updated later.
  - valid_from / valid_until: only set if the text explicitly implies temporal bounds (e.g. "valid until end of Q2", \
"starting next Monday"). Otherwise null.
  - Resolve relative dates (e.g. "last Tuesday", "next month") to ISO 8601 (YYYY-MM-DD) using today's date.
  - Do NOT extract generic advice or universally-applicable facts that apply to everyone.

Return a JSON object with an `atoms` key containing an array of atom objects. \
Each atom must have exactly these keys: subject, kind, source, content, valid_from, valid_until.
"""

_SUPERSESSION_SYSTEM = """\
You are deciding whether a new fact supersedes an existing fact about the same subject. \
Supersession means the new fact contradicts or replaces the old one — not merely adds to it.
Return a JSON object: {"superseded_atom_id": "<atom_id>"} if superseded, or {"superseded_atom_id": null} if not.
"""

_SUPERSESSION_MULTI_SYSTEM = """\
You are deciding whether a new fact supersedes any of the existing facts listed below. \
Supersession means the new fact contradicts or replaces an old one — not merely adds to it.
Return a JSON object: {"superseded_atom_id": "<atom_id>"} for the one superseded fact, \
or {"superseded_atom_id": null} if none are superseded.
"""


@dataclass(frozen=True)
class _Segment:
    segment_id: str
    text: str
    source_type: str
    start: int
    end: int
    context: str = ""

# ── date resolution ───────────────────────────────────────────────────────────

_RELATIVE_PATTERNS: list[tuple[re.Pattern, Any]] = [
    (re.compile(r'\b(\d+)\s+days?\s+ago\b', re.IGNORECASE),
     lambda m, ref: (ref - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")),
    (re.compile(r'\b(\d+)\s+weeks?\s+ago\b', re.IGNORECASE),
     lambda m, ref: (ref - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")),
    (re.compile(r'\b(\d+)\s+months?\s+ago\b', re.IGNORECASE),
     lambda m, ref: (ref - timedelta(days=int(m.group(1)) * 30)).strftime("%Y-%m-%d")),
    (re.compile(r'\blast\s+year\b', re.IGNORECASE),
     lambda _, ref: str(ref.year - 1)),
    (re.compile(r'\blast\s+week\b', re.IGNORECASE),
     lambda _, ref: (ref - timedelta(weeks=1)).strftime("%Y-%m-%d")),
    (re.compile(r'\blast\s+month\b', re.IGNORECASE),
     lambda _, ref: (ref - timedelta(days=30)).strftime("%Y-%m-%d")),
    (re.compile(r'\byesterday\b', re.IGNORECASE),
     lambda _, ref: (ref - timedelta(days=1)).strftime("%Y-%m-%d")),
]

_WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_PATTERN = re.compile(
    r'\blast\s+(' + '|'.join(_WEEKDAY_NAMES) + r')\b', re.IGNORECASE
)


def _resolve_dates(content: str, ref: datetime) -> str:
    """Replace relative date expressions in atom content with absolute ISO dates."""
    for pattern, resolver in _RELATIVE_PATTERNS:
        content = pattern.sub(lambda m, _r=ref, _res=resolver: _res(m, _r), content)

    def _resolve_weekday(m: re.Match) -> str:
        day_name = m.group(1).lower()
        target_dow = _WEEKDAY_NAMES.index(day_name)
        days_back = (ref.weekday() - target_dow) % 7
        if days_back == 0:
            days_back = 7
        return (ref - timedelta(days=days_back)).strftime("%Y-%m-%d")

    content = _WEEKDAY_PATTERN.sub(_resolve_weekday, content)
    return content


# ── helpers ───────────────────────────────────────────────────────────────────

def _today() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_datetime(val: Any) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day, tzinfo=timezone.utc)
    text = str(val)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            d = date.fromisoformat(text[:10])
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        except ValueError:
            return None


def _parse_date(val: Any) -> date | None:
    if not val:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    try:
        return date.fromisoformat(str(val)[:10])
    except ValueError:
        return None


def _normalized_content(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _infer_source_type(text: str, metadata: dict) -> str:
    if metadata.get("source_type"):
        return str(metadata["source_type"])
    if re.search(r"^#{1,6}\s+\S", text, flags=re.MULTILINE):
        return "markdown"
    if re.search(r"^\s*(user|assistant|system|developer)\s*:", text, flags=re.IGNORECASE | re.MULTILINE):
        return "chat"
    if "```" in text or re.search(r"\b(def|class|function|import|from)\s+\w+", text):
        return "code"
    return "plain"


def _segment_plain(text: str, source_type: str, max_chars: int) -> list[_Segment]:
    if len(text) <= max_chars:
        return [_Segment("s0", text, source_type, 0, len(text))]

    segments: list[_Segment] = []
    overlap = min(300, max_chars // 5)
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            segments.append(_Segment(f"s{idx}", chunk, source_type, start, end))
            idx += 1
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return segments


def _segment_markdown(text: str, max_chars: int) -> list[_Segment]:
    headings = list(re.finditer(r"^#{1,6}\s+.+$", text, flags=re.MULTILINE))
    if len(text) <= max_chars or not headings:
        return _segment_plain(text, "markdown", max_chars)

    segments: list[_Segment] = []
    for idx, match in enumerate(headings):
        start = match.start()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        chunk = text[start:end].strip()
        heading = match.group(0).strip()
        if len(chunk) > max_chars:
            for part in _segment_plain(chunk, "markdown", max_chars):
                part_start = start + part.start
                segments.append(_Segment(f"s{len(segments)}", part.text, "markdown", part_start, start + part.end, heading))
        elif chunk:
            segments.append(_Segment(f"s{len(segments)}", chunk, "markdown", start, end, heading))
    return segments


def _segment_chat(text: str, max_chars: int) -> list[_Segment]:
    # Keep complete turns together when possible; fall back to plain windows for very long logs.
    turns = list(re.finditer(r"^\s*(user|assistant|system|developer)\s*:", text, flags=re.IGNORECASE | re.MULTILINE))
    if len(text) <= max_chars or not turns:
        return _segment_plain(text, "chat", max_chars)

    segments: list[_Segment] = []
    start = turns[0].start()
    current_start = start
    for idx, turn in enumerate(turns[1:], start=1):
        if turn.start() - current_start >= max_chars:
            chunk = text[current_start:turn.start()].strip()
            if chunk:
                segments.append(_Segment(f"s{len(segments)}", chunk, "chat", current_start, turn.start()))
            current_start = turn.start()
    chunk = text[current_start:].strip()
    if chunk:
        segments.append(_Segment(f"s{len(segments)}", chunk, "chat", current_start, len(text)))
    return segments


def _segments_for_source(source: str, metadata: dict) -> list[_Segment]:
    max_chars = int(os.environ.get("LATTICE_SEGMENT_CHARS", "12000"))
    source_type = _infer_source_type(source, metadata)
    if source_type == "markdown":
        return _segment_markdown(source, max_chars)
    if source_type == "chat":
        return _segment_chat(source, max_chars)
    return _segment_plain(source, source_type, max_chars)


def _extract_atoms(segment: _Segment, metadata: dict, ref: datetime) -> list[dict]:
    text = segment.text
    if segment.context:
        text = f"Context: {segment.context}\n\n{text}"
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"Today's date: {ref.date().isoformat()}\n\n---\n\n{text}",
        },
    ]
    raw = complete(messages, text_format=_AtomList)
    atoms_data: list[dict] = json.loads(raw)["atoms"]

    source_override = metadata.get("source")
    for a in atoms_data:
        if source_override:
            a["source"] = source_override
        a["content"] = _resolve_dates(a["content"], ref)
        a["metadata"] = metadata
        a["segment_id"] = segment.segment_id
        a["source_type"] = segment.source_type
        a["source_span"] = {"start": segment.start, "end": segment.end}
    return atoms_data


def _detect_supersession(db: LatticeDB, new_atom: Atom) -> str | None:
    # Fast path: subject registry
    existing_id = db.lookup_subject(new_atom.subject)
    if existing_id:
        try:
            existing = db.read(existing_id)
            if existing.is_superseded:
                return None
        except Exception:
            return None

        messages = [
            {"role": "system", "content": _SUPERSESSION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"New fact: {new_atom.content}\n\n"
                    f"Existing fact [{existing_id}]: {existing.content}"
                ),
            },
        ]
        raw = complete(messages, text_format=_SupersessionResult)
        superseded_id = json.loads(raw).get("superseded_atom_id")
        if not superseded_id:
            return None
        return existing_id if superseded_id == existing_id else None

    # Slow path: scan by subject (handles hand-edited atoms)
    existing = [a for a in db.by_subject(new_atom.subject) if not a.is_superseded]
    if not existing:
        return None

    candidates_text = "\n".join(f"[{a.atom_id}] {a.content}" for a in existing)
    messages = [
        {"role": "system", "content": _SUPERSESSION_MULTI_SYSTEM},
        {
            "role": "user",
            "content": (
                f"New fact: {new_atom.content}\n\n"
                f"Existing facts about '{new_atom.subject}':\n{candidates_text}"
            ),
        },
    ]
    raw = complete(messages, text_format=_SupersessionResult)
    superseded_id = json.loads(raw).get("superseded_atom_id")
    if not superseded_id:
        return None
    valid_ids = {a.atom_id for a in existing}
    return superseded_id if superseded_id in valid_ids else None


def ingest(source: str, metadata: dict | None = None, db: LatticeDB | None = None) -> dict:
    if db is None:
        db = LatticeDB()
    metadata = metadata or {}
    ref = _today()
    source_id = str(metadata.get("source_id") or uuid.uuid4())
    observed_at = _parse_datetime(
        metadata.get("observed_at") or metadata.get("date") or metadata.get("timestamp")
    )

    segments = _segments_for_source(source, metadata)
    workers = max(1, int(os.environ.get("LATTICE_INGEST_WORKERS", "1")))
    if workers == 1 or len(segments) == 1:
        nested_atoms = [_extract_atoms(segment, metadata, ref) for segment in segments]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            nested_atoms = list(pool.map(lambda s: _extract_atoms(s, metadata, ref), segments))
    atoms_data = [a for atoms in nested_atoms for a in atoms]
    created_ids: list[str] = []
    duplicate_ids: list[str] = []

    for data in atoms_data:
        content = data["content"]
        content_hash = _hash_text(content)
        normalized_hash = _hash_text(_normalized_content(content))
        duplicate = db.find_by_normalized_hash(normalized_hash)
        if duplicate is not None:
            duplicate_ids.append(duplicate.atom_id)
            continue

        atom = Atom(
            kind=data.get("kind", "fact"),
            source=data.get("source", "document"),
            subject=data["subject"],
            content=content,
            valid_from=_parse_date(data.get("valid_from")),
            valid_until=_parse_date(data.get("valid_until")),
            ingested_at=ref,
            observed_at=observed_at,
            source_id=source_id,
            source_title=metadata.get("title") or metadata.get("source_title"),
            session_id=metadata.get("session_id"),
            segment_id=data.get("segment_id"),
            source_type=data.get("source_type"),
            source_span=data.get("source_span"),
            content_hash=content_hash,
            normalized_content_hash=normalized_hash,
            metadata=data.get("metadata", {}),
        )

        old_id = _detect_supersession(db, atom)
        if old_id:
            db.supersede(old_id, atom)
        else:
            db.write(atom)

        if atom.subject:
            db.register_subject(atom.subject, atom.atom_id)

        created_ids.append(atom.atom_id)

    return {
        "atoms_created": len(created_ids),
        "atom_ids": created_ids,
        "duplicates_skipped": len(duplicate_ids),
        "duplicate_atom_ids": duplicate_ids,
        "source_id": source_id,
        "segments_processed": len(segments),
    }
