# PRD: lattice-mcp — General-Purpose Lattice MCP Server

## Problem Statement

Developers and knowledge workers using AI coding assistants (Claude Code, Cursor, Cline, etc.) have no persistent, structured memory layer. The assistant's context resets each session. Users must re-explain project state, decisions, and facts repeatedly. Existing solutions are either tightly coupled to a specific assistant, require cloud infrastructure, or lack temporal awareness (facts can't expire or supersede each other).

## Solution

A local-first MCP server (`lattice-mcp`) that any MCP-compatible coding assistant can use as a persistent knowledge store. The user drops `uvx lattice-mcp` into their `.mcp.json`. The assistant can then ingest text into a structured atom store, select relevant atoms for a query, and synthesize answers from those atoms. Atoms are human-readable markdown files stored locally — git-trackable, portable, no cloud dependency. Facts can supersede each other, and temporal filters (`as_of`) allow querying the lattice as it stood at a past date.

## User Stories

1. As a developer using Claude Code, I want to ingest a project README so that the assistant can answer questions about my project without me re-explaining it each session.
2. As a developer, I want to ingest raw text notes so that my decisions and context are atomized and retrievable later.
3. As a developer, I want atoms stored as local markdown files so that I can read, edit, and version-control my knowledge store in git.
4. As a developer, I want to install the MCP server with a single `uvx lattice-mcp` command so that I don't need to clone a repo or manage a Python environment.
5. As a developer, I want to configure the LLM provider via environment variables so that I can use Anthropic, OpenAI, or a local Ollama model without changing code.
6. As a developer, I want the server to run as a local stdio subprocess so that my API keys never leave my machine.
7. As a developer, I want to call `lattice_ingest` from my assistant so that raw text is automatically decomposed into discrete, retrievable atoms.
8. As a developer, I want ingested atoms to have open-ended `kind` and `source` fields so that I can store facts, events, preferences, code snippets, and documentation without being constrained to a fixed taxonomy.
9. As a developer, I want atoms to automatically detect when a new fact supersedes an older one about the same subject so that the lattice stays consistent without manual cleanup.
10. As a developer, I want to call `lattice_select` with a natural language query so that the most relevant atoms are returned ranked by relevance.
11. As a developer, I want `lattice_select` to accept an optional `as_of` date so that I can query the state of my knowledge at a past point in time.
12. As a developer, I want `lattice_select` to use a local index and graph-aware retrieval so that queries remain fast and relevant as my lattice grows.
13. As a developer, I want to call `lattice_answer` and receive a synthesized prose answer so that I don't need to manually read and synthesize multiple atoms.
14. As a developer, I want `lattice_answer` to auto-run selection if no atom IDs are provided so that I can get a direct answer from a single tool call.
15. As a developer using a weak local model (e.g. Ollama), I want to call `lattice_select` directly and synthesize the answer myself so that I can offload expensive synthesis to my local assistant.
16. As a developer, I want atoms to carry optional `valid_from` and `valid_until` dates so that time-bounded facts (e.g. a person's role, a price, a policy) are automatically excluded when querying outside their validity window.
17. As a developer, I want atoms to carry structured provenance (`source_id`, title, observed time, source span, hashes) plus passthrough metadata so that answers can be inspected, cited, and deduplicated.
18. As a developer, I want each atom to have a stable UUID so that I can reference specific atoms in `lattice_answer` calls and in supersession links.
19. As a developer, I want `lattice_ingest` to return the list of created atom IDs so that I can immediately reference or inspect the ingested atoms.
20. As a developer, I want to point `LATTICE_DIR` at any directory via env var so that I can maintain separate lattices per project.
21. As a Claude Code user, I want a working `.mcp.json` example in the README so that I can configure the server in under five minutes.
22. As a Cursor user, I want MCP server configuration instructions so that I can use lattice-mcp in my editor.
23. As a developer, I want to run functional integration tests that call all three MCP tools end-to-end so that I can verify the full pipeline before shipping.

## Implementation Decisions

### Modules

- **Atom Model** — Pydantic model with UUID atom_id, free-form string `kind` and `source`, `subject`, optional `valid_from`/`valid_until` ISO dates, supersession fields (`is_superseded`, `superseded_by`, `supersedes`), and a passthrough `metadata` dict. Serialized to/from YAML frontmatter + markdown body.

- **LatticeDB (atom store)** — Filesystem-backed store. One atom = one `.md` file in `LATTICE_DIR`. Provides: write atom, read atom by ID, list all atoms, list by subject, BM25 full-text search, and `as_of` temporal filtering. Supersession links are written atomically when a new atom supersedes an old one. BM25 index rebuilt on read (v1); persistent index is a v2 optimization.

- **Graph Index (product target)** — Local file-backed graph sidecars (`nodes.jsonl`, `edges.jsonl`, `sources.json`, `graph/manifest.json`) loaded into a NetworkX-style in-memory graph. Deterministic nodes and edges connect atoms to sources, segments, subjects, duplicate groups, and update chains. The graph has committed versions so selection can query stable snapshots while ingest continues.

- **LLM Client** — LiteLLM-based. Reads `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` from env. Exposes a single `complete(messages) -> str` interface. No provider-specific SDK imports.

- **Internal Agent Tools** — Thin wrappers over LatticeDB: `search_atoms(query, as_of)`, `read_atom(atom_id)`, `list_subjects()`, `list_all_atoms()`. Used by the ingest, selection, and synthesis agents.

- **Ingest Agent** — Takes raw text string in v1. Product direction is source-aware segmentation and provenance: source IDs, segment IDs, source spans, observed times, and content hashes. The ingest path should commit each source independently and perform cheap exact dedup before writing atoms.

- **Selection Agent** — Takes natural language query + optional `as_of`. Current implementation uses BM25 candidates plus LLM filtering. Product direction is graph-seeded selection over committed snapshots: seed from lexical search, expand through source/segment/subject/update edges, collapse duplicate/supersession groups, then use LLM tool calls for final inclusion.

- **Synthesis Agent** — Takes query + list of atom dicts. Uses LLM to produce a concise answer with uncertainty when evidence is weak or conflicting. Product output should not expose hidden chain-of-thought by default; citations/source snippets are optional when provenance is available.

- **MCP Server** — Entry point. Exposes core tools: `lattice_ingest`, `lattice_select`, `lattice_answer`. Product direction includes local status/batch-ingest helpers when MCP client concurrency supports them. Reads all configuration from env vars. Communicates via stdio (MCP protocol). No HTTP server, no ports.

### API Contracts

- `lattice_ingest(source: str, metadata: dict = {}) -> {atoms_created: int, atom_ids: list[str]}`
- `lattice_select(query: str, as_of: str | None = None) -> list[{atom_id, subject, content, kind, source, valid_from, valid_until}]`
- `lattice_answer(query: str, atom_ids: list[str] = [], as_of: str | None = None) -> str`

### Architectural Decisions

- **LiteLLM** as single LLM interface — no provider-specific SDK imports. Provider routing via env vars.
- **Filesystem storage** — atoms as `.md` files plus portable graph/index sidecars, human-readable where practical, git-trackable. No hosted DB.
- **stdio transport** — coding assistant spawns server as subprocess. No HTTP.
- **`as_of` only on select** — not on `lattice_answer` (passed through internally when atom_ids not provided).
- **Graph-first lattice** — supersession exists today; product direction adds deterministic graph edges first, then optional semantic relation enrichment.
- **Text-only ingestion v1** — `source` param is raw text string. File path and directory ingestion remain roadmap items, implemented through the same source-aware segmentation/provenance path.
- **Distribution** — PyPI + `uvx lattice-mcp`. Zero-clone install.

### Environment Variables

- `LLM_PROVIDER` — `anthropic` | `openai` | `ollama`
- `LLM_MODEL` — model ID string
- `LLM_API_KEY` — API key (not required for Ollama)
- `LATTICE_DIR` — path to atom store directory (default: `./lattice`)

## Testing Decisions

**What makes a good test:** Tests verify external behavior — what the module returns given valid inputs, what errors it raises given invalid inputs, and what side effects it produces (files written, files updated). Tests do not assert on internal implementation (prompt strings, intermediate variables, private method calls). Tests are runnable without network access where possible (mock LLM calls); tests that require LLM calls are marked as integration tests and skipped in CI unless an API key is present.

### Modules to Test

**Atom Model (unit tests)**
- Valid atom creation with all fields
- Default field values (null valid_from/valid_until, empty metadata, is_superseded=False)
- Round-trip: serialize to YAML frontmatter + body, parse back, assert equality
- Validation: missing required fields raise errors

**LatticeDB (filesystem integration tests, temp directory)**
- Write atom → file exists with correct frontmatter and body
- Read atom by ID → returns correct Atom model
- Read nonexistent ID → raises appropriate error
- List all atoms → returns all written atoms
- List by subject → returns only matching atoms
- BM25 search → top result contains query keywords
- `as_of` filter → atom with `valid_until` in the past is excluded; atom without `valid_until` is included
- Supersession write → old atom marked `is_superseded=True`, `superseded_by` set; new atom has `supersedes` set

**LLM Client (unit tests, mocked LiteLLM)**
- Anthropic provider → LiteLLM called with correct model prefix
- OpenAI provider → correct prefix
- Ollama provider → correct prefix
- Missing API key for non-Ollama provider → raises config error

**Ingest / Selection / Synthesis Agents (integration tests, real LLM or recorded fixtures)**
- Ingest: call with a short paragraph → at least one atom created, atom has valid frontmatter, atom is readable from LatticeDB
- Supersession: ingest two conflicting facts about same subject → second atom's `supersedes` field points to first
- Selection: call with query matching ingested content → ingested atom appears in top results
- `as_of` filter: ingest atom with past `valid_until` → excluded from selection with future `as_of`
- Answer: call with query → returns non-empty string

**MCP Wire Test (manual / end-to-end)**
- Run server via `uvx lattice-mcp`, add to Claude Code `.mcp.json`, call all 3 tools from assistant, verify responses.

## Out of Scope

- Hosted cloud sync
- Cloud deployment or HTTP transport
- Multi-lattice per server instance
- Web UI or atom browser
- Atom deletion / garbage collection
- Auth or multi-user access control

## Further Notes

- LongMemEval is an evaluation yardstick for long-memory pressure, not the product target. Product changes should be measured with it, but benchmark-specific hacks should stay out of core paths.
- Temporal information should be represented through provenance and validity fields where possible, not by mutating atom content solely to satisfy a benchmark.
- The `lattice_answer` tool exists primarily to serve weaker local models (Ollama) where the calling assistant cannot reliably synthesize multi-atom answers on its own. Power users on strong models will typically call `lattice_select` + synthesize inline.
- Atom files being plain markdown means users can hand-edit, delete, or version-control their knowledge store independently of the server. The server should tolerate hand-edited files gracefully.
- `uvx` install requires the package to be on PyPI. Publishing to PyPI is part of the initial milestone.
