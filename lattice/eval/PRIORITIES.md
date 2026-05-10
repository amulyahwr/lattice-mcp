# Eval Improvement Priorities

Baseline: 15% accuracy (100q, gemma4:e4b inference, qwen3.5:4b judge, longmemeval_oracle)

Failure breakdown (updated after p1 analysis):
- 20% of questions: selection returns 0 atoms despite 26+ atoms created
- 56% of questions: atoms selected but synthesis still wrong
- **100% of atoms have null valid_from** — root cause of all 26 temporal-reasoning failures
- ~10% of atoms are near-duplicates — wastes selection budget
- Multi-session selection failure rate (30%) worse than overall (20%)

Methodology: implement one priority → full 100q eval → measure delta → decide next.

| Priority | File(s) | Change | Est. ROI |
|----------|---------|--------|----------|
| P1 ✅ | `synthesis.py` | Add `thinking: str` field to `_Answer` (CoT before answer); strengthen no-info prompt rule | Done — +8.8pp |
| P2 ✅ | `ingest.py` | **Date injection**: prepend session date into atom `content` as "On {date}: ..." — 100% of p1 atoms had null valid_from, root cause of all 26 temporal failures · 5 lines · re-ingest needed | Very High — fixes root cause of 26% of questions with near-zero code |
| P3 ✅ | `selection.py` | **Question-type-aware selection prompt**: detect temporal/multi-session signals in query and inject type-specific instructions ("prioritise atoms with explicit dates", "look across distinct time periods") — prompt-only, zero infra, targets 53/100 worst failures | High — zero cost, directly targets the two worst categories |
| P4 ✅ | `selection.py` | **HyDE**: generate a hypothetical answer atom before BM25 search to bridge query↔atom vocabulary gap | High — 1 LLM call, no infra, helps all question types |
| P5 ✅ | `ingest.py`, `models.py`, `db.py` | **Question generation per atom**: generate 2-3 natural questions per atom, store in `questions` field (BM25-searchable) — reverted: 16.0% vs p1 23.8% (−7.8pp). Root cause: generated questions pollute BM25 with false-positive matches (e.g. "What does Alice drink?" causes coffee atom to match unrelated drink queries), pushing relevant atoms out of top-20 candidates | Reverted |
| P6 | `ingest.py` | **Adaptive chunking**: if `len(source)` ≤ threshold (default ~2000 chars) → single extraction call (current behavior, zero overhead); if longer → split at paragraph boundaries (`\n\n`) into ~500-word chunks → extract atoms per chunk in parallel via `ThreadPoolExecutor`; configurable via `INGEST_CHUNK_SIZE` env var. No domain assumptions — works for READMEs, meeting notes, Slack exports, code docs, any text. Dedup across chunks handled by existing supersession logic. | Medium-High — zero cost for typical short MCP ingests; improves extraction completeness on long sources by reducing attention dilution; general-purpose |
| P7 | `llm.py`, `selection.py` | Extend `complete()` with `tools/tool_choice/reasoning`; replace bulk JSON selection with `include_atom(atom_id, reason)` tool calls + zero-selection fallback (top-5 BM25 if 0 selected) | High — eliminates 20% silent 0-atom selection failures |
| P8 | `ingest.py`, `db.py` | **Atom deduplication**: before writing, BM25-score new atom against existing atoms on same subject; if similarity > threshold, merge instead of creating duplicate — eliminates ~10% near-duplicate noise that wastes selection budget | Medium-High — cleaner atom store benefits all retrieval methods |
| P9 | `selection.py` | **Multi-pass retrieval**: seed second BM25 pass with first-round atom content; second LLM selection call knows what's already found and looks for what's missing (opt-in via `SELECT_PASSES=2`) | Medium-High for multi-session (30% selection failure vs 20% overall) · needs P7 infra |
| P10 | `ingest.py` | **Session summary atom**: after ingesting each session, generate one summary atom (date + key topics + notable facts) — coarse-grained index for multi-session retrieval to land on before fine atom selection | Medium for multi-session (27q) · 1 extra LLM call per session · re-ingest needed |
| P11 | `synthesis.py`, `selection.py` | **Uncertainty-triggered re-retrieval**: when synthesis `thinking` contains "do not contain" / "no date" / "impossible to determine", extract missing fact type and run a targeted second selection pass | Medium — turns synthesis expressed uncertainty into retrieval feedback signal |
| P12 | `llm.py`, `db.py` | **Semantic embeddings**: `embed()` via ollama + hybrid BM25+cosine search; store `.emb` files at ingest — structural fix for vocabulary mismatch | Highest long-term · most effort · needs embedding model pulled |
| P13 | `ingest.py` | **Ingest tool calling**: replace bulk JSON extraction with `record_atom(...)` tool calls — per-atom reasoning, eliminates JSON parse failures | Medium — improves atom extraction quality; needs re-ingest to take effect |

## Results

| Run | Priority | Accuracy | Notes |
|-----|----------|----------|-------|
| baseline | — | 15.0% | gemma4:e4b · qwen3.5:4b judge · 100q oracle |
| p1 | Synthesis CoT | 23.8% | task-avg 25.2% · multi-session 22.2% · temporal 7.4% |
| p2 | Date injection into atom content | 19.0% | task-avg 23.1% · multi-session 3.7% · temporal 0.0% · regression vs p1 — reverted |
| p3 | Question-type-aware selection (temporal-only) | 18.0% | task-avg 20.1% · temporal 0.0% · regression vs p1 — reverted |
| p4 | HyDE expansion | skipped | BM25 + hallucinated vocab = wrong atom retrieval; HyDE suited for dense/embedding retrieval only |
| p5 | Question generation per atom | 16.0% | task-avg 18.2% · multi-session 7.4% · temporal 7.7% · regression vs p1 — reverted |
| p6 | Adaptive chunking | — | |
| p7 | Selection tool calling + fallback | — | |
| p8 | Atom deduplication | — | |
| p9 | Multi-pass retrieval | — | |
| p10 | Session summary atom | — | |
| p11 | Uncertainty re-retrieval | — | |
| p12 | Semantic embeddings | — | |
| p13 | Ingest tool calling | — | |

## Per-category tracker

| Category | baseline | p1 | p2 | p3 | p5 |
|---|---|---|---|---|---|
| **overall** | 15.0% | 23.8% | 19.0% | 18.0% | 16.0% |
| **task-avg** | — | 25.2% | 23.1% | 20.1% | 18.2% |
| single-session-user | — | 35.7% | 42.9% | 35.7% | 14.3% |
| single-session-preference | — | 0.0% | 0.0% | 0.0% | 0.0% |
| single-session-assistant | — | 54.5% | 54.5% | 36.4% | 54.5% |
| multi-session | — | 22.2% | 3.7% | 11.1% | 7.4% |
| temporal-reasoning | — | 7.4% | 0.0% | 0.0% | 7.7% |
| knowledge-update | — | 31.3% | 37.5% | 37.5% | 25.0% |
| abstention | — | 28.6% | 0.0% | 28.6% | 42.9% |
