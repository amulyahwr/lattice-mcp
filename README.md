# lattice-mcp

A local-first MCP server that gives any MCP-compatible AI coding assistant persistent, structured memory.

Text you ingest is decomposed into discrete **atoms** — one fact per file, stored as human-readable markdown in a directory you control. The product direction is a local lattice: atoms connected to source, segment, subject, duplicate, and update nodes so selection can navigate memory instead of scanning a flat folder.

## Install

```bash
uvx lattice-mcp
```

No clone required. Runs as a local stdio subprocess — your API keys never leave your machine.

## Configuration

Set via environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `ollama` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model ID |
| `LLM_API_KEY` | — | API key (not required for Ollama) |
| `LATTICE_DIR` | `./lattice` | Directory where atoms are stored |

## Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "lattice": {
      "command": "uvx",
      "args": ["lattice-mcp"],
      "env": {
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-sonnet-4-6",
        "LLM_API_KEY": "sk-ant-...",
        "LATTICE_DIR": "/path/to/my-lattice"
      }
    }
  }
}
```

## Cursor / Cline

Same config — add the server under MCP settings with the same `command`, `args`, and `env`.

## Tools

### `lattice_ingest(source, metadata?)`

Decomposes raw text into atoms and stores them.

```
source    — raw text string
metadata  — optional dict (title, url, author, date, …)

→ { atoms_created: N, atom_ids: [...] }
```

For larger local knowledge bases, the roadmap is source-aware ingestion: source IDs, source spans, exact deduplication, and per-source commits so useful partial memory becomes queryable quickly.

### `lattice_select(query, as_of?)`

Returns the most relevant atoms for a natural language query.

```
query   — natural language question
as_of   — optional ISO date (YYYY-MM-DD); filters to atoms valid at that date

→ [ { atom_id, subject, content, kind, source, valid_from, valid_until }, ... ]
```

The current implementation uses BM25 + LLM filtering. The product roadmap adds an incremental graph index and committed snapshots so selection stays fast while ingest is active.

### `lattice_answer(query, atom_ids?, as_of?)`

Synthesizes a prose answer from the lattice.

```
query     — natural language question
atom_ids  — optional list of atom IDs to use; auto-selects if empty
as_of     — optional ISO date; passed to selection when atom_ids not provided

→ answer string
```

## Atom format

Atoms are stored as `.md` files with YAML frontmatter:

```markdown
---
atom_id: 3f2e1a...
kind: fact
source: user
subject: Project Alpha
valid_from: null
valid_until: null
is_superseded: false
superseded_by: null
supersedes: null
metadata: {}
---
Project Alpha targets enterprise customers and launched in Q1 2025.
```

All files are human-readable and git-trackable. You can hand-edit them.

## Evaluation

LongMemEval is used as a yardstick for long-memory pressure, not as the product target. Product priorities live in `lattice/eval/PRIORITIES.md`; benchmark-specific code stays under `lattice/eval/`.

## Development

```bash
git clone https://github.com/amulyahwr/lattice-mcp
cd lattice-mcp
uv sync
uv run pytest
```
