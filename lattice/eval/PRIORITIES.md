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
| P2 | `ingest.py` | **Date injection**: prepend session date into atom `content` as "On {date}: ..." — 100% of p1 atoms had null valid_from, root cause of all 26 temporal failures · 5 lines · re-ingest needed | Very High — fixes root cause of 26% of questions with near-zero code |
| P3 | `selection.py` | **Question-type-aware selection prompt**: detect temporal/multi-session signals in query and inject type-specific instructions ("prioritise atoms with explicit dates", "look across distinct time periods") — prompt-only, zero infra, targets 53/100 worst failures | High — zero cost, directly targets the two worst categories |
| P4 | `selection.py` | **HyDE**: generate a hypothetical answer atom before BM25 search to bridge query↔atom vocabulary gap | High — 1 LLM call, no infra, helps all question types |
| P5 | `llm.py`, `selection.py` | Extend `complete()` with `tools/tool_choice/reasoning`; replace bulk JSON selection with `include_atom(atom_id, reason)` tool calls + zero-selection fallback (top-5 BM25 if 0 selected) | High — eliminates 20% silent 0-atom selection failures |
| P6 | `ingest.py`, `db.py` | **Atom deduplication**: before writing, BM25-score new atom against existing atoms on same subject; if similarity > threshold, merge instead of creating duplicate — eliminates ~10% near-duplicate noise that wastes selection budget | Medium-High — cleaner atom store benefits all retrieval methods |
| P7 | `selection.py` | **Multi-pass retrieval**: seed second BM25 pass with first-round atom content; second LLM selection call knows what's already found and looks for what's missing (opt-in via `SELECT_PASSES=2`) | Medium-High for multi-session (30% selection failure vs 20% overall) · needs P5 infra |
| P8 | `ingest.py` | **Session summary atom**: after ingesting each session, generate one summary atom (date + key topics + notable facts) — coarse-grained index for multi-session retrieval to land on before fine atom selection | Medium for multi-session (27q) · 1 extra LLM call per session · re-ingest needed |
| P9 | `synthesis.py`, `selection.py` | **Uncertainty-triggered re-retrieval**: when synthesis `thinking` contains "do not contain" / "no date" / "impossible to determine", extract missing fact type and run a targeted second selection pass | Medium — turns synthesis expressed uncertainty into retrieval feedback signal |
| P10 | `llm.py`, `db.py` | **Semantic embeddings**: `embed()` via ollama + hybrid BM25+cosine search; store `.emb` files at ingest — structural fix for vocabulary mismatch | Highest long-term · most effort · needs embedding model pulled |
| P11 | `ingest.py` | **Ingest tool calling**: replace bulk JSON extraction with `record_atom(...)` tool calls — per-atom reasoning, eliminates JSON parse failures | Medium — improves atom extraction quality; needs re-ingest to take effect |

## Results

| Run | Priority | Accuracy | Notes |
|-----|----------|----------|-------|
| baseline | — | 15.0% | gemma4:e4b · qwen3.5:4b judge · 100q oracle |
| p1 | Synthesis CoT | 23.8% | task-avg 25.2% · multi-session 22.2% · temporal 7.4% |
| p2 | Date injection into atom content | — | |
| p3 | Question-type-aware selection | — | |
| p4 | HyDE expansion | — | |
| p5 | Selection tool calling + fallback | — | |
| p6 | Atom deduplication | — | |
| p7 | Multi-pass retrieval | — | |
| p8 | Session summary atom | — | |
| p9 | Uncertainty re-retrieval | — | |
| p10 | Semantic embeddings | — | |
| p11 | Ingest tool calling | — | |
