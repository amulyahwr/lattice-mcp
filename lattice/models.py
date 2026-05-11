from __future__ import annotations

import uuid
from datetime import date, datetime
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
    ingested_at: datetime | None = None
    observed_at: datetime | None = None
    source_id: str | None = None
    source_title: str | None = None
    session_id: str | None = None
    segment_id: str | None = None
    source_type: str | None = None
    source_span: dict[str, int] | None = None
    content_hash: str | None = None
    normalized_content_hash: str | None = None
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
            "provenance": {
                "ingested_at": self.ingested_at.isoformat() if self.ingested_at else None,
                "observed_at": self.observed_at.isoformat() if self.observed_at else None,
                "source_id": self.source_id,
                "source_title": self.source_title,
                "session_id": self.session_id,
                "segment_id": self.segment_id,
                "source_type": self.source_type,
                "source_span": self.source_span,
            },
            "dedup": {
                "content_hash": self.content_hash,
                "normalized_content_hash": self.normalized_content_hash,
            },
            "metadata": self.metadata,
        }
        post = frontmatter.Post(self.content, **front)
        return frontmatter.dumps(post)

    @classmethod
    def from_markdown(cls, text: str) -> "Atom":
        post = frontmatter.loads(text)
        data = dict(post.metadata)
        data["content"] = post.content

        provenance = data.pop("provenance", None) or {}
        if isinstance(provenance, dict):
            for key, field in {
                "observed_at": "observed_at",
                "ingested_at": "ingested_at",
                "source_id": "source_id",
                "source_title": "source_title",
                "session_id": "session_id",
                "segment_id": "segment_id",
                "source_type": "source_type",
                "source_span": "source_span",
            }.items():
                if field not in data:
                    data[field] = provenance.get(key)

        dedup = data.pop("dedup", None) or {}
        if isinstance(dedup, dict):
            if "content_hash" not in data:
                data["content_hash"] = dedup.get("content_hash")
            if "normalized_content_hash" not in data:
                data["normalized_content_hash"] = dedup.get("normalized_content_hash")

        for date_field in ("valid_from", "valid_until"):
            val = data.get(date_field)
            if isinstance(val, str):
                data[date_field] = date.fromisoformat(val)
            elif val is None:
                data[date_field] = None
        for datetime_field in ("ingested_at", "observed_at"):
            val = data.get(datetime_field)
            if isinstance(val, str):
                data[datetime_field] = datetime.fromisoformat(val)
            elif val is None:
                data[datetime_field] = None
        return cls(**data)
