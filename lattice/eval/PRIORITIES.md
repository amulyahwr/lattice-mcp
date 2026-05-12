# lattice-mcp Product Priorities

Goal: build a local-first MCP server for persistent, inspectable knowledge that works well with coding assistants and local models. LongMemEval is an evaluation yardstick, not the product target.

Product constraints:

- Local-only: no hosted service, no required daemon, no external database.
- Works with API models and Ollama; expensive enrichment must be optional.
- Atom files remain human-readable and git-trackable.
- Ingest can handle many local sources, but useful partial memory should commit quickly.
- Selection should be fast, graph-aware, and should not wait for active ingest or background enrichment.

## Active Product Roadmap

| Priority | File(s) | Product Change | Why It Matters |
| --- | --- | --- | --- |
| P1 ✅ | `README.md`, `PRD.md`, `CLAUDE.md`, `lattice/eval/README.md` | **Product documentation cleanup**: update docs so product architecture is local-first lattice-mcp, not benchmark-preservation. Move LongMemEval guidance under eval-only docs, remove “must preserve benchmark win” language, and document graph/provenance/status concepts as product goals. | Done — prevents future implementation drift back toward research harness assumptions. |
| P2 ⏭️ | `llm.py`, `selection.py` | **Selection tool calling + fallback**: extend `complete()` with `tools` / `tool_choice`; replace bulk JSON selection with `include_atom(atom_id, reason)` calls; if the model selects nothing, fall back to top BM25 candidates. | Skipped for now by product direction; still useful before or during graph-seeded selection. |
| P3 ✅ | `models.py`, `ingest.py`, `selection.py`, `synthesis.py` | **Source-aware ingest + provenance + exact dedup**: add `ingested_at`, optional `observed_at`, `source_id`, `source_title`, `session_id`, `segment_id`, `source_type`, `source_span`, `content_hash`, and `normalized_content_hash`. Segment by source type: Markdown headings, chat turns/windows, code/docs symbols or sections, plain text overlapped windows. Commit each source independently and skip/mark exact duplicates before write. | Done — atoms now carry provenance/hash fields, ingest segments sources with bounded workers, exact duplicates are skipped, and selection/synthesis expose provenance. |
| P3.5 ✅ | `lattice/eval/run_eval.py`, `lattice/eval/run_parallel_eval.py`, `lattice/eval/debug_viewer.py`, `lattice/eval/print_retrieval_metrics.py`, `selection.py` | **Retrieval oracle diagnostics + selected payload normalization**: use LongMemEval `answer_session_ids` to report session hit/recall/precision/MRR for BM25 candidates and selected atoms; expose `has_answer` answer-turn previews for debugging; make `select` return the same provenance/dedup/supersession payload shape as BM25 debug atoms. | Done — removes a select-vs-BM25 payload-shape confound before P4 and gives retrieval metrics that can separate recall failures from synthesis failures. |
| P4 ✅ | `selection.py`, `db.py` | **Pack retrieval over BM25 seeds**: keep BM25 as the recall path, but make selection return evidence packs instead of isolated atom picks. For each top seed atom, deterministically expand to same `segment_id`, nearby atoms in the same `source_id` / `session_id`, same normalized subject, and supersession/update neighbors when available. Flatten packs with de-duplication and stable ordering; do not use an LLM as a hard selector. | Done — improved 100q LongMemEval from P3.5 select 13.0% / BM25 15.0% to 18.0% overall and 20.19% task-avg. Avg selected atoms rose from 17.1 to 31.1; selected session hit stayed 100%, recall dipped slightly to 0.986. |
| P5 | `graph.py`, `db.py`, `ingest.py` | **Incremental heterogeneous graph index**: add a local NetworkX-style `MultiDiGraph` compute layer backed by portable sidecars: `nodes.jsonl`, `edges.jsonl`, `sources.json`, `graph/manifest.json`. Deterministic nodes/edges: `atom:<id>`, `source:<id>`, `segment:<id>`, `subject:<normalized>`, `source_contains_segment`, `segment_contains_atom`, `atom_has_subject`, `same_subject_as`, `same_hash`, `supersedes` / `updates`. Use atomic writes, graph versions, and cache reload only when the manifest changes. | Turns a flat atom folder into a real lattice while preserving local, file-based storage. |
| P6 | `selection.py`, `graph.py`, `db.py` | **Graph-seeded selection over committed snapshots**: selection reads the latest committed graph/BM25 snapshot and never waits for active ingest. BM25 seeds atoms/subjects/sources, then bounded BFS expands evidence packs through source/segment/subject/update edges. Collapse duplicate/supersession groups before synthesis; keep any LLM scoring optional and non-authoritative. | Generalizes pack retrieval onto the graph, improves relevance and latency, reduces duplicate context waste, and enables query-while-ingest. |
| P7 | `ingest.py`, `server.py`, `db.py` | **Local ingest jobs + status UX**: keep simple `lattice_ingest` for small sync inputs. Add persistent status for batch/background ingest: `job_id`, indexed/active/failed source counts, enrichment pending, `graph_version`, `last_commit_at`. If MCP client concurrency works well, expose `lattice_ingest_start` and `lattice_ingest_status`; otherwise keep status internal/debug. | Local users need to know what has been indexed and should be able to query committed memory without waiting for a large ingest to finish. |
| P8 | `graph.py`, `db.py`, `ingest.py`, `selection.py` | **Optional semantic relation enrichment**: after deterministic graph indexing, optionally add high-confidence edges: `updates`, `contradicts`, `supports`, `elaborates`, `temporally_before`. Run as explicit/background enrichment, off or cheap by default for Ollama. | Adds deeper reasoning paths without making local ingest/query latency worse. |
| P9 | `graph.py`, `selection.py` | **Topic hubs / community index**: create hub nodes from connected components first; later optional label propagation or modularity over subject/source/relation edges. Hubs store aliases, member atoms, latest `observed_at`, and centroid text. Selection falls back cleanly when hubs are stale. | Helps broad queries land on coarse concepts before atoms, while keeping hub generation eventually consistent. |
| P10 | `ingest.py` | **Ingest tool calling**: replace bulk JSON extraction with `record_atom(...)` tool calls where provider support is good; keep JSON fallback for local models/providers that handle structured JSON better. | Improves extraction reliability without forcing one LLM interface path. |
| P11 | `llm.py`, `db.py`, `selection.py` | **Optional semantic embeddings**: add `embed()` and hybrid BM25 + vector search as an optional index. Store embeddings in portable sidecars and make model choice explicit. | Useful for vocabulary mismatch, but should not be required for the local-first baseline. |
| P12 | `synthesis.py`, `server.py` | **Product answer contract**: replace eval-style hidden CoT contract with a product response shape: concise answer, optional evidence/citations, explicit uncertainty when selected atoms are weak or conflicting, and no forced answer merely because some atoms were selected. Keep raw reasoning out of normal tool output. | Makes `lattice_answer` trustworthy for real users instead of benchmark-optimized. |
| P13 | `selection.py`, `server.py` | **Selection debug/status mode**: optional debug output includes graph version, indexed source counts, BM25 ranks/scores, pack expansion paths, fallback trigger, and final include reasons. | Makes memory behavior explainable during product debugging without noisy normal answers. |
| P14 | `selection.py`, `synthesis.py`, `server.py` | **Source-grounded answer mode**: optionally return citations/snippets for selected atoms using `source_title`, `source_id`, and `source_span` when available. Normal answers stay concise. | Builds user trust in local memory and makes stored knowledge inspectable. |
| P15 | `db.py`, `server.py` | **Memory namespaces**: support project/workspace isolation via `LATTICE_NAMESPACE` or equivalent local directory layout. | Prevents cross-project memory contamination, which matters more for real users than benchmark score. |

## Removed From Active Roadmap

These remain useful experiment history, but are no longer product priorities:

- Date injection into atom content: helped some eval categories but polluted content and hurt temporal/multi-session reasoning.
- Question-type prompt patches: too benchmark-shaped and brittle.
- HyDE over BM25: hallucinated vocabulary hurt sparse retrieval.
- Generated questions per atom: polluted BM25 and caused false positives.
- Session summary atoms: LongMemEval-shaped; topic hubs and source nodes are the product-native version.
- Multi-pass retrieval and uncertainty-triggered re-retrieval: add latency and brittle control flow; pack retrieval and graph-seeded selection should solve the product problem first.

## LongMemEval Yardstick

LongMemEval is used to measure whether product changes improve retrieval and answer quality under long-memory pressure. It should not dictate architecture.

Current baseline:

- 15.0% accuracy, 100 questions
- inference: `gemma4:e4b`
- judge: `qwen3.5:4b`
- harness: `longmemeval_oracle`

Observed failure modes:

- 20% of questions: selection returns 0 atoms despite 26+ atoms created.
- 56% of questions: atoms selected but synthesis still wrong.
- 100% of atoms had null `valid_from` in early runs, exposing missing provenance/time structure.
- About 10% of atoms are near-duplicates, wasting selection budget.
- Multi-session selection failures are worse than overall failures.
- `answer_session_ids` are available across local oracle/S/M datasets and now provide session-level retrieval diagnostics; `has_answer` is available on gold sessions and is exposed for debugging, but exact turn-level metrics still require turn-span provenance.
- A pre-P4 check showed `select` and `bm25` modes could differ despite both using BM25 because selected atom payloads lacked the nested provenance/dedup/supersession fields present in BM25 debug atoms. P3.5 normalized that payload shape.

Evaluation method:

- Implement one product priority at a time.
- Run full 100-question LongMemEval.
- Compare overall score and category movement.
- Keep product-fit changes even if benchmark movement is mixed, but do not optimize architecture solely for LongMemEval.

## Experiment History

| Run | Change | Accuracy | Notes |
| --- | --- | --- | --- |
| baseline | none | 15.0% | `gemma4:e4b` inference, `qwen3.5:4b` judge, 100q oracle |
| p1 | Synthesis CoT | 23.8% | Best measured gain so far; task-avg 25.2%, multi-session 22.2%, temporal 7.4%. |
| p2 | Date injection into atom content | 19.0% | Helped some direct categories but regressed overall; content pollution. Reverted. |
| p3 | Question-type-aware selection prompt | 18.0% | Prompt-only benchmark patch regressed overall. Reverted. |
| p4 | HyDE expansion | skipped | BM25 + hallucinated vocabulary produced wrong retrieval; better suited for dense retrieval. |
| p5 | Generated questions per atom | 16.0% | Generated questions polluted BM25 with false-positive matches. Reverted. |
| p6 | Adaptive paragraph chunking | 22.0% | Helped some user-fact extraction but hurt cross-paragraph/coreference cases. Keep only as motivation for source-aware segmentation. |
| p3.5-select | Retrieval oracle + selected payload normalization | 13.0% | Verified select/BM25 payload-shape confound cleared on reused P3 lattices. Retrieval metrics matched BM25 exactly: selected hit 100%, recall 1.000, precision 1.000, MRR 1.000. Task-avg 15.03%. |
| p3.5-bm25 | BM25 ablation on normalized payload | 15.0% | Same reused P3 lattices and current diagnostics. BM25 hit 100%, recall 1.000, precision 1.000, MRR 1.000. Task-avg 17.51%. |
| p4-pack | Pack retrieval over BM25 seeds | 18.0% | Product-fit keeper. Avg selected atoms rose from 17.1 to 31.1. Selected hit 100%, recall 0.986, precision 1.000, MRR 1.000. Task-avg 20.19%. |

## Category Tracker

| Category | baseline | p1 | p2 | p3 | p5 | p6 | p3.5-select | p3.5-bm25 | p4-pack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 15.0% | 23.8% | 19.0% | 18.0% | 16.0% | 22.0% | 13.0% | 15.0% | 18.0% |
| task-avg | - | 25.2% | 23.1% | 20.1% | 18.2% | - | 15.03% | 17.51% | 20.19% |
| single-session-user | - | 35.7% | 42.9% | 35.7% | 14.3% | 42.9% | 21.4% | 42.9% | 35.7% |
| single-session-preference | - | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| single-session-assistant | - | 54.5% | 54.5% | 36.4% | 54.5% | 36.4% | 36.4% | 27.3% | 45.5% |
| multi-session | - | 22.2% | 3.7% | 11.1% | 7.4% | 14.8% | 7.4% | 3.7% | 11.1% |
| temporal-reasoning | - | 7.4% | 0.0% | 0.0% | 7.7% | 7.7% | 0.0% | 0.0% | 3.85% |
| knowledge-update | - | 31.3% | 37.5% | 37.5% | 25.0% | 37.5% | 25.0% | 31.3% | 25.0% |
| abstention | - | 28.6% | 0.0% | 28.6% | 42.9% | - | 42.9% | 57.1% | 42.9% |
