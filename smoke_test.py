"""
End-to-end smoke test: ingest → select → answer via Ollama gemma4:e2b.
Run: LLM_PROVIDER=ollama LLM_MODEL=gemma4:e2b LATTICE_DIR=/tmp/lattice-smoke uv run python smoke_test.py
"""

import os
import shutil

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "gemma4:e2b")
os.environ.setdefault("LATTICE_DIR", "/tmp/lattice-smoke")

# Fresh dir each run
shutil.rmtree(os.environ["LATTICE_DIR"], ignore_errors=True)

from lattice.db import LatticeDB
from lattice.ingest import ingest
from lattice.selection import select
from lattice.synthesis import synthesize

db = LatticeDB()

SOURCE = """
lattice-mcp is a local-first MCP server that gives AI coding assistants persistent memory.
Text is decomposed into discrete atoms — one fact per markdown file.
Atoms support temporal validity windows via valid_from and valid_until fields.
When a new fact contradicts an existing one on the same subject, the old atom is superseded.
The server exposes three tools: lattice_ingest, lattice_select, and lattice_answer.
BM25 keyword search is combined with LLM re-ranking to retrieve relevant atoms.
"""

print("── 1. INGEST ──────────────────────────────────────────────────────")
result = ingest(SOURCE, metadata={"title": "lattice-mcp overview"}, db=db)
print(f"atoms_created : {result['atoms_created']}")
print(f"atom_ids      : {result['atom_ids']}")

print("\n── 2. SELECT ──────────────────────────────────────────────────────")
atoms = select("How does lattice handle conflicting facts?", db=db)
print(f"atoms returned: {len(atoms)}")
for a in atoms:
    print(f"  [{a['subject']}] {a['content'][:80]}...")

print("\n── 3. ANSWER ──────────────────────────────────────────────────────")
answer = synthesize("How does lattice handle conflicting facts?", atoms)
print(answer)

print("\n── 4. SUPERSESSION ────────────────────────────────────────────────")
update = "When a new fact contradicts an existing one, the old atom is marked superseded and a bidirectional link is created between old and new."
result2 = ingest(update, db=db)
print(f"atoms_created : {result2['atoms_created']}")

all_atoms = db.all()
superseded = [a for a in all_atoms if a.is_superseded]
print(f"total atoms   : {len(all_atoms)}")
print(f"superseded    : {len(superseded)}")
if superseded:
    s = superseded[0]
    print(f"  superseded  : [{s.atom_id[:8]}] {s.content[:60]}...")
    print(f"  superseded_by: {s.superseded_by[:8]}...")
