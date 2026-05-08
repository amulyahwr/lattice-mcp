# Plan: lattice-mcp — General-Purpose Lattice MCP Server

## Context

Goal: implement the core pattern — ingest → atom store → select → synthesize — into a clean, general-purpose MCP server that anyone can drop into their coding assistant (Claude Code, Cursor, Cline, etc.). Named `lattice-mcp` because atoms are nodes, supersession-links are edges, and the lattice is the persistent knowledge store.

**Deployment model**: completely local. The coding assistant spawns `uvx lattice-mcp` as a subprocess on the user's machine via stdio. No cloud server, no ports. Atoms stored in the user's local `LATTICE_DIR`. API keys stay on the user's machine.

---

## Design Decisions (settled)

| Decision | Choice |
|---|---|
| Delivery | MCP server (3 tools) |
| Lattice scope | One lattice per server instance |
| Storage | Filesystem (atoms as .md files, human-readable, git-trackable) |
| LLM backend | LiteLLM — single interface for Anthropic / OpenAI / Ollama |
| Atom schema | Open-ended `kind` + `source`; `valid_from`/`valid_until` optional (null by default) |
| Lattice edges v1 | Supersession only (typed edges deferred) |
| Modality v1 | Text only; input = raw text string; file path (v2), directory (v3) roadmap |
| `as_of` filter | Optional param on `lattice_select(query, as_of=None)` — not on `lattice_answer` |
| `lattice_answer` | Kept — useful when assistant model is weak (local Ollama); power users call `lattice_select` directly |
| Distribution | PyPI + `uvx lattice-mcp`; zero-clone install |

---

## New Repo Structure: `lattice-mcp`

```
lattice-mcp/
├── lattice/
│   ├── __init__.py
│   ├── models.py       — atom schema (Pydantic, open-ended)
│   ├── db.py           — filesystem atom store / LatticeDB (adapted from db.py)
│   ├── tools.py        — agent tools: search_atoms, read_atom, list_subjects, list_all_atoms
│   ├── llm.py          — LiteLLM-based client (Anthropic / OpenAI / Ollama)
│   ├── ingest.py       — ingest agent (adapted from ingest.py)
│   ├── selection.py    — selection agent (adapted from selection.py)
│   └── synthesis.py    — synthesis agent (adapted from synthesis.py)
├── server.py           — MCP server entry point (3 tools)
├── pyproject.toml
└── README.md
```

---

## Atom Schema (open-ended)

```yaml
---
atom_id: <uuid>
kind: <string>        # open: fact, event, preference, belief, code, doc, ...
source: <string>      # open: user, assistant, document, file, url, ...
subject: <string>
valid_from: <date|null>
valid_until: <date|null>
is_superseded: <bool>
superseded_by: <atom_id|null>
supersedes: <atom_id|null>
metadata: {}          # passthrough from lattice_ingest call
---
<content>
```

Key change from current: `kind` and `source` are free-form strings. No hardcoded enum values.

---

## MCP Tools (3)

### `lattice_ingest(source: str, metadata: dict = {}) -> dict`
- `source`: raw text content (v1); file path (v2); directory path (v3)
- `metadata`: passthrough dict (title, date, url, author, etc.) — stored in atom frontmatter
- Returns: `{atoms_created: N, atom_ids: [...]}`

### `lattice_select(query: str, as_of: str = None) -> list[dict]`
- `query`: natural language question
- `as_of`: optional ISO date string — filters atoms valid at that date
- Returns: list of `{atom_id, subject, content, kind, source, valid_from, valid_until}`

### `lattice_answer(query: str, atom_ids: list[str] = [], as_of: str = None) -> str`
- `query`: natural language question
- `atom_ids`: optional — if empty, auto-runs selection first
- `as_of`: passed through to internal selection if atom_ids not provided
- Returns: synthesized answer string

---

## LLM Configuration (env vars)

```bash
LLM_PROVIDER=anthropic   # anthropic | openai | ollama
LLM_MODEL=claude-sonnet-4-6
LLM_API_KEY=sk-...
LATTICE_DIR=./lattice    # where atoms are stored (default: ./lattice)
```

`lattice/llm.py` uses **LiteLLM** as the single unifying interface across all three providers. Provider-specific SDKs are not needed — LiteLLM handles routing. Add `litellm` as a dep; drop `openai-agents` dependency.

### Claude Code `.mcp.json` example

```json
{
  "mcpServers": {
    "lattice": {
      "command": "uvx",
      "args": ["lattice-mcp"],
      "env": {
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-sonnet-4-6",
        "LLM_API_KEY": "sk-...",
        "LATTICE_DIR": "./my-lattice"
      }
    }
  }
}
```

---

## Implementation Steps

1. **Init repo** — `git init lattice-mcp`, `uv init`, add `mcp`, `litellm`, `pydantic`, `rank_bm25` as deps; publish to PyPI so `uvx lattice-mcp` works

2. **Port atom schema** — rewrite `models.py` with open-ended fields; remove LongMemEval-specific enums

3. **Port db.py** — copy LatticeDB filesystem logic, strip eval-harness coupling, adapt for open schema

4. **Port tools.py** — `search_atoms`, `read_atom`, `list_subjects`, `list_all_atoms` — adapt for new schema

5. **Write llm.py** — LiteLLM-based client; env-var-driven provider selection

6. **Port ingest.py** — adapt ingest agent; system prompt generalized (not LongMemEval-specific); date resolution kept (proven +0.14 win)

7. **Port selection.py** — adapt selection agent; BM25 scoring kept (proven improvement)

8. **Port synthesis.py** — adapt synthesis agent; generalized system prompt

9. **Write server.py** — MCP server using `mcp` Python SDK; wire 3 tools to lattice agents; read config from env vars

10. **Write README.md** — install, config, usage with Claude Code / Cursor / Cline

---

## Validation

Functional integration tests on real-world text — not LongMemEval (that benchmark lives in the research repo, not here):

1. **Ingest test**: call `lattice_ingest` with a README, verify atoms are created with correct frontmatter
2. **Supersession test**: ingest two conflicting facts about the same subject, verify second supersedes first
3. **Select test**: call `lattice_select` with a natural language query, verify relevant atoms returned
4. **as_of test**: ingest atom with `valid_until` set, call `lattice_select` with past/future `as_of`, verify filter works
5. **Answer test**: call `lattice_answer`, verify coherent synthesized response
6. **MCP wire test**: run `uvx lattice-mcp` locally, add to Claude Code `.mcp.json`, call all 3 tools from the assistant

---

## Key Source Files to Port From (this repo)

| Source | Destination |
|---|---|
| `db.py` | `lattice/db.py` |
| `tools.py` | `lattice/tools.py` |
| `ingest.py` | `lattice/ingest.py` |
| `selection.py` | `lattice/selection.py` |
| `synthesis.py` | `lattice/synthesis.py` |
| `llm_client.py` | `lattice/llm.py` (rewritten, LiteLLM-based) |
| `models.py` | `lattice/models.py` (rewritten, open-ended schema) |
