from __future__ import annotations

import re
from datetime import date, timedelta

from lattice.db import AtomNotFound, LatticeDB
from lattice.models import Atom

# ── agent tools (wrappers over LatticeDB) ────────────────────────────────────

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


# ── date arithmetic utilities ─────────────────────────────────────────────────

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "a": 1, "an": 1,
}


def _parse_offset(expression: str) -> tuple[int, str] | None:
    expr = expression.lower().strip()
    for word, val in _WORD_TO_NUM.items():
        expr = re.sub(rf"\b{word}\b", str(val), expr)
    m = re.search(
        r"(\d+)\s*(day|week|month|year)s?\s*(ago|before|earlier|from now|later|after)",
        expr,
    )
    if not m:
        return None
    n, unit, direction = int(m.group(1)), m.group(2), m.group(3)
    sign = -1 if direction in ("ago", "before", "earlier") else 1
    days = n * sign * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    return days, f"{n} {unit}(s) {'earlier' if sign < 0 else 'later'}"


def compute_date(anchor_date: str, offset_expression: str) -> str:
    """Resolve a relative time expression against an anchor date.

    anchor_date       — ISO date string, e.g. "2023-05-22"
    offset_expression — e.g. "3 weeks ago", "two months earlier", "1 day later"

    Returns the resolved date as YYYY-MM-DD.
    """
    try:
        anchor = date.fromisoformat(anchor_date[:10])
    except ValueError:
        return f"Cannot parse anchor date: {anchor_date!r}"
    parsed = _parse_offset(offset_expression)
    if parsed is None:
        return (
            f"Cannot parse offset {offset_expression!r}. "
            "Use a form like '3 weeks ago', 'two months earlier', '1 day later'."
        )
    days, label = parsed
    result = anchor + timedelta(days=days)
    return f"{result.isoformat()} ({label} from {anchor.isoformat()})"


def days_between(date1: str, date2: str) -> str:
    """Return the signed number of days between two ISO date strings (date2 − date1)."""
    try:
        d1 = date.fromisoformat(date1[:10])
        d2 = date.fromisoformat(date2[:10])
    except ValueError as e:
        return f"Cannot parse dates: {e}"
    diff = (d2 - d1).days
    if diff == 0:
        return "Same day (0 days apart)."
    earlier, later = (date1[:10], date2[:10]) if diff > 0 else (date2[:10], date1[:10])
    return f"{abs(diff)} days apart. {earlier} is earlier; {later} is later."
