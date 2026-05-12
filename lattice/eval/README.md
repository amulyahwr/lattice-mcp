# Evaluation Harness

Runs lattice-mcp against the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark. This is a yardstick for long-memory pressure, not the product target. Benchmark-specific code lives under `lattice/eval/` and should not drive core product architecture.

## Prerequisites

- A local machine or remote box capable of running your chosen inference/judge models
- Ollama installed and running, if using local models
- LongMemEval repo cloned (for dataset + scorer scripts)
- lattice-mcp repo cloned

## Setup

### 1. Install eval dependencies

```bash
uv sync --group eval
```

### 2. Pull models

```bash
# Inference model (~18GB)
ollama pull gemma4:26b

# Judge model (~9GB) — pulled automatically at judge phase, but you can pre-pull
ollama pull qwen2.5:14b
```

### 3. Configure

```bash
cp .env.eval.example .env.eval
```

Edit `.env.eval` and set the two path variables:

```bash
DATASET=/path/to/LongMemEval/data/longmemeval_oracle.json
EVALUATE_SCRIPT=/path/to/LongMemEval/src/evaluation/evaluate_qa.py
PRINT_SCRIPT=/path/to/LongMemEval/src/evaluation/print_qa_metrics.py
```

Everything else in `.env.eval` can stay as-is for a standard run.

## Running

### Full run (inference + judge)

```bash
uv run python -m lattice.eval.run_eval
```

Runs 100 stratified questions through the lattice pipeline, then scores with the configured judge model. Prints accuracy by question type at the end.

### Inference only

```bash
uv run python -m lattice.eval.run_eval --phase inference
```

Writes `results/run1.jsonl` (hypotheses) and `results/run1.debug.jsonl` (per-question diagnostics). Safe to interrupt — re-running resumes from where it stopped.

### Selection diagnostics

Use same ingest, swap retrieval mode, then compare judged accuracy and debug files:

```bash
# Product path: BM25 candidates -> selection payload -> synthesis
uv run python -m lattice.eval.run_eval --phase inference --retrieval-mode select --priority p3-select

# BM25 ablation: BM25 candidates go straight to synthesis through the eval debug payload
uv run python -m lattice.eval.run_eval --phase inference --retrieval-mode bm25 --priority p3-bm25

# Ceiling check: all valid atoms go to synthesis
uv run python -m lattice.eval.run_eval --phase inference --retrieval-mode all --priority p3-all
```

Read result:

| Pattern | Meaning |
|---|---|
| `bm25` > `select` | Selection path differs from raw BM25; check selected atom IDs and payload shape |
| `all` > `bm25` | BM25 candidate generation weak |
| `select`/`bm25`/`all` all bad | Ingest missing facts or synthesis/reasoning weak |
| Correct atom in `bm25_candidates`, missing from `atoms_selected` | Selection or pack-expansion bug |
| Correct atom in `selected_atoms`, bad answer | Synthesis bug |

Set `--top-k 50` to test if selection improves with wider candidate pool. Debug rows include `retrieval_mode`, `top_k`, `bm25_candidates`, `selected_atoms`, `retrieval_oracle`, and `answer_oracle`.

Current `select` uses BM25 directly. Its selected atom payload is normalized to include the same provenance, dedup, and supersession fields as BM25 debug atoms, while keeping flat provenance fields for product callers. After payload normalization, `select` and `bm25` should be close; if they diverge, inspect atom IDs, ordering, and output reuse before attributing the gap to retrieval quality.

To keep ingest identical across variants, use the variant runner with `--reuse-ingest`:

```bash
uv run python -m lattice.eval.run_parallel_eval --priority p3 --reuse-ingest
```

This materializes atoms once, then runs `select`, `bm25`, and `all` against the same per-question lattice dirs.

### Retrieval oracle diagnostics

LongMemEval examples include `answer_session_ids`. The inference debug file uses those IDs to score retrieval before synthesis:

| Metric | Meaning |
|---|---|
| session hit | At least one retrieved atom came from a gold answer session |
| session recall | Fraction of gold answer sessions reached |
| session precision | Fraction of retrieved atoms whose session is gold |
| session MRR | Reciprocal rank of the first retrieved gold-session atom |

The debug row stores metrics for both `bm25_candidates` and final `selected_atoms` under `retrieval_oracle`. It also stores `answer_oracle` with `has_answer` turn counts and previews. Local oracle/S/M datasets all include `answer_session_ids`; `has_answer` appears on gold sessions and is useful debug metadata, but exact turn-level retrieval scoring needs turn-span provenance. Current atom provenance is session/segment-level.

Print retrieval metrics from an inference debug file:

```bash
uv run python lattice/eval/print_retrieval_metrics.py results/run1.debug.jsonl
```

### Judge only (inference already done)

```bash
uv run python -m lattice.eval.run_eval --phase judge
```

Starts a LiteLLM proxy pointing to `qwen2.5:14b`, runs `evaluate_qa.py` against the existing results file, then stops the proxy.

### Override config inline

```bash
uv run python -m lattice.eval.run_eval \
  --dataset /other/path/longmemeval_oracle.json \
  --out results/run2.jsonl \
  --stratify 50
```

CLI flags override `.env.eval` values.

## Output files

| File | Contents |
|---|---|
| `results/run1.jsonl` | `{question_id, hypothesis}` — input to scorer |
| `results/run1.debug.jsonl` | Per-question: sessions ingested, atoms created, atoms selected, hypothesis |
| `results/run1.jsonl.eval-results-gpt-4o` | Scorer output with per-question labels |

## Diagnosing failures

When a question is answered wrong, the debug file tells you where the pipeline broke:

| Symptom | Likely cause |
|---|---|
| `atoms_created: 0` | Ingest extraction failed (model returned malformed JSON) |
| Correct atoms exist but not in `atoms_selected` | Retrieval failure (BM25 or LLM re-rank missed them) |
| Correct atoms in `atoms_selected`, wrong answer | Synthesis failure |

When a benchmark idea improves one category but makes the product worse (for example by polluting atom content or adding latency-heavy control flow), keep it in eval notes instead of the active product roadmap.

```bash
# Quick failure breakdown
python - <<'EOF'
import json, sys
debug = [json.loads(l) for l in open("results/run1.debug.jsonl")]
print(f"Questions with 0 atoms: {sum(1 for q in debug if q['atoms_created'] == 0)}")
print(f"Questions with 0 selected: {sum(1 for q in debug if not q['atoms_selected'])}")
EOF
```

## Stratification

100 questions sampled proportionally from all 7 question types (seed=42, reproducible):

| Question type | Sample |
|---|---|
| multi-session | 27 |
| temporal-reasoning | 27 |
| knowledge-update | 16 |
| single-session-user | 14 |
| single-session-assistant | 11 |
| single-session-preference | 6 |
| abstention | 6 |

Change `STRATIFY` and `SEED` in `.env.eval` to adjust.
