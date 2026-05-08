from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import frontmatter
from pydantic import BaseModel, Field


class Atom(BaseModel):
    atom_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: str
    source: str
    subject: str
    content: str
    valid_from: date | None = None
    valid_until: date | None = None
    is_superseded: bool = False
    superseded_by: str | None = None
    supersedes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        front: dict[str, Any] = {
            "atom_id": self.atom_id,
            "kind": self.kind,
            "source": self.source,
            "subject": self.subject,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "is_superseded": self.is_superseded,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "metadata": self.metadata,
        }
        post = frontmatter.Post(self.content, **front)
        return frontmatter.dumps(post)

    @classmethod
    def from_markdown(cls, text: str) -> "Atom":
        post = frontmatter.loads(text)
        data = dict(post.metadata)
        data["content"] = post.content
        for date_field in ("valid_from", "valid_until"):
            val = data.get(date_field)
            if isinstance(val, str):
                data[date_field] = date.fromisoformat(val)
            elif val is None:
                data[date_field] = None
        return cls(**data)
