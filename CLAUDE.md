# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                        # install deps
uv run pytest                  # all tests
uv run pytest tests/test_db.py # single file
uv run pytest -k test_supersession_links_atoms  # single test
uv run lattice-mcp             # run MCP server (requires env vars)
```

Required env vars for running the server: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LATTICE_DIR`.

## Architecture

The pipeline is: **ingest → select → synthesize**, each backed by an LLM call via `lattice/llm.py`.

```
server.py          MCP stdio entrypoint. Owns one shared LatticeDB instance.
lattice/
  llm.py           Thin litellm wrapper. Reads LLM_PROVIDER/LLM_MODEL/LLM_API_KEY from env.
  models.py        Atom pydantic model + markdown serialization (python-frontmatter).
  db.py            File-based store: one .md file per atom in LATTICE_DIR. BM25 search.
                   subjects.json is a subject→atom_id index for O(1) supersession lookups.
  ingest.py        LLM extracts atoms from raw text, then checks supersession per atom.
  selection.py     BM25 pre-filter (top_k=20) → LLM re-ranks → returns atom dicts.
  synthesis.py     LLM generates prose answer from a list of atom dicts.
  tools.py         (unused placeholder)
```

### Key data flow details

**Supersession** (in `ingest.py`): when a new atom has the same subject as an existing one, an LLM call decides if it supersedes. Fast path uses `subjects.json`; slow path scans files (handles hand-edited atoms). Superseded atoms stay on disk with `is_superseded=true` and bidirectional links (`superseded_by` / `supersedes`).

**LLM calls**: all go through `lattice.llm.complete(messages)`. Tests mock this at `lattice.ingest.complete`, `lattice.selection.complete`, `lattice.synthesis.complete` — patch the module-level name, not `lattice.llm.complete`.

**Atom storage**: every atom is a `.md` file with YAML frontmatter. `LatticeDB` has an in-memory cache (`_atom_cache`). Cache is per-instance; `server.py` reuses one instance per process.

**BM25**: built fresh on each `db.search()` call from all non-superseded atoms. No persistent index.

### Test conventions

All tests mock LLM via `unittest.mock.patch`. Ingest responses mock two calls per atom: first the extraction JSON, then the supersession reply (`"null"` or an atom_id string). Use `tmp_path` fixture for isolated `LatticeDB` instances.
