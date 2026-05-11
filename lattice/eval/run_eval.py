"""
LongMemEval evaluation harness for lattice-mcp.

Usage:
    python -m lattice.eval.run_eval                             # full run (inference + judge)
    python -m lattice.eval.run_eval --phase inference           # inference only
    python -m lattice.eval.run_eval --phase judge               # judge only (needs results file)
    python -m lattice.eval.run_eval --stratify 10               # quick smoke test (10 questions)
    python -m lattice.eval.run_eval --out results/myrun.jsonl   # custom output path
    python -m lattice.eval.run_eval --keep-lattice-dirs         # keep per-question atom dirs
    python -m lattice.eval.run_eval --retrieval-mode bm25       # bypass LLM selection
    python -m lattice.eval.run_eval --retrieval-mode all        # bypass retrieval; synthesize over all atoms

    # stdout + stderr are tee'd to a log file automatically:
    python -m lattice.eval.run_eval --phase inference
    python -m lattice.eval.run_eval --phase judge
    python -m lattice.eval.run_eval --phase all
    python -m lattice.eval.run_eval --log logs/custom.log       # custom log path

    # file naming convention (auto-derived):
    #   results/<llm>_<judge>_<dataset>_<phase>.jsonl
    #   logs/run_<llm>_<judge>_<dataset>_<phase>.log
    # e.g.: results/gemma4e4b_qwen2514b_longmemeval_oracle_inference.jsonl

    # Debug viewer — browse results by question, failure mode, atoms:
    uv run --group eval streamlit run lattice/eval/debug_viewer.py

Defaults (overridable via CLI flags or env vars in .env.eval):
    dataset  : lattice/eval/data/longmemeval_oracle.json
    out      : results/<model-slug>_baseline.jsonl  (auto-derived from LLM_MODEL)
    stratify : 100 questions, stratified by question_type
    seed     : 42
    retrieval: select; set RETRIEVAL_MODE=select|bm25|all
    top_k    : 20; set TOP_K or --top-k
    keep dirs: false; set KEEP_LATTICE_EVAL_DIRS=1 or --keep-lattice-dirs

Required env vars (set in .env.eval): LLM_PROVIDER, LLM_MODEL, LLM_API_KEY (non-ollama).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from lattice.db import LatticeDB
from lattice.eval.session_formatter import format_session
from lattice.ingest import ingest
from lattice.selection import select
from lattice.synthesis import synthesize

# ── config ────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DATASET = str(_DATA_DIR / "longmemeval_oracle.json")


def _slug(s: str) -> str:
    for ch in (":", "/", ".", "-"):
        s = s.replace(ch, "")
    return s


def _results_dir(priority: str) -> str:
    return f"results/{priority}" if priority else "results"


def _logs_dir(priority: str) -> str:
    return f"logs/{priority}" if priority else "logs"


def _default_out(
    llm_model: str, dataset: str, priority: str, retrieval_mode: str = "select"
) -> str:
    """Inference output path. Always named by llm_slug only — judge slug appears in eval-results filename."""
    ds = Path(dataset).stem
    suffix = "_inference" if retrieval_mode == "select" else f"_{retrieval_mode}_inference"
    return f"{_results_dir(priority)}/{_slug(llm_model)}_{ds}{suffix}.jsonl"


def _default_log(
    llm_model: str,
    judge_model: str,
    dataset: str,
    phase: str,
    priority: str,
    retrieval_mode: str = "select",
) -> str:
    ds = Path(dataset).stem
    base = _logs_dir(priority)
    mode = "" if retrieval_mode == "select" else f"_{retrieval_mode}"
    if phase == "inference":
        name = f"run_{_slug(llm_model)}_{ds}{mode}_inference.log"
    elif phase == "judge":
        name = f"run_{_slug(judge_model)}_{ds}{mode}_judge.log"
    else:
        name = f"run_{_slug(llm_model)}_{_slug(judge_model)}_{ds}{mode}_all.log"
    return f"{base}/{name}"


class _Tee:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def fileno(self):
        return self.streams[0].fileno()


def _load_config(args: argparse.Namespace) -> dict:
    load_dotenv(".env.eval", override=False)
    model = os.environ.get("LLM_MODEL", "gemma4:e2b")
    judge = os.environ.get("JUDGE_MODEL", "qwen3.5:4b")
    dataset = args.dataset or os.environ.get("DATASET", _DEFAULT_DATASET)
    phase = args.phase
    priority = args.priority or os.environ.get("PRIORITY", "")
    retrieval_mode = args.retrieval_mode or os.environ.get("RETRIEVAL_MODE", "select")
    if retrieval_mode not in {"select", "bm25", "all"}:
        raise ValueError("RETRIEVAL_MODE must be one of: select, bm25, all")
    cfg = {
        "dataset": dataset,
        "out": args.out
        or os.environ.get("OUT", _default_out(model, dataset, priority, retrieval_mode)),
        "log": args.log
        or os.environ.get(
            "LOG", _default_log(model, judge, dataset, phase, priority, retrieval_mode)
        ),
        "priority": priority,
        "stratify": args.stratify or int(os.environ.get("STRATIFY", "100")),
        "seed": args.seed or int(os.environ.get("SEED", "42")),
        "retrieval_mode": retrieval_mode,
        "top_k": args.top_k or int(os.environ.get("TOP_K", "20")),
        "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
        "llm_model": model,
        "judge_model": judge,
        "litellm_port": int(os.environ.get("LITELLM_PORT", "4000")),
        "evaluate_script": args.evaluate_script
        or os.environ.get("EVALUATE_SCRIPT", ""),
        "print_qa_script": args.print_qa_script
        or os.environ.get("PRINT_QA_SCRIPT", ""),
        "print_retrieval_script": args.print_retrieval_script
        or os.environ.get("PRINT_RETRIEVAL_SCRIPT", ""),
        "keep_lattice_dirs": args.keep_lattice_dirs
        or os.environ.get("KEEP_LATTICE_EVAL_DIRS", "").lower() in {"1", "true", "yes"},
    }
    return cfg


# ── stratified sampling ───────────────────────────────────────────────────────


def _stratify(data: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for item in data:
        by_type.setdefault(item["question_type"], []).append(item)

    total = len(data)
    sample: list[dict] = []
    remainder: list[dict] = []

    for qtype, items in by_type.items():
        quota = round(n * len(items) / total)
        shuffled = rng.sample(items, len(items))
        sample.extend(shuffled[:quota])
        remainder.extend(shuffled[quota:])

    # top up or trim to exactly n
    shortfall = n - len(sample)
    if shortfall > 0:
        sample.extend(rng.sample(remainder, min(shortfall, len(remainder))))
    elif shortfall < 0:
        rng.shuffle(sample)
        sample = sample[:n]

    rng.shuffle(sample)
    return sample


# ── resume helpers ─────────────────────────────────────────────────────────────


def _load_done_ids(out_path: Path) -> set[str]:
    """Return question_ids that completed successfully (excludes ERROR entries so they re-run)."""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
                if not entry.get("hypothesis", "").startswith("ERROR:"):
                    done.add(entry["question_id"])
            except Exception:
                pass
    return done


# ── inference phase ───────────────────────────────────────────────────────────


def _atom_debug_dict(atom, preview_chars: int | None = None) -> dict:
    content = atom.content if preview_chars is None else atom.content[:preview_chars]
    return {
        "atom_id": atom.atom_id,
        "subject": atom.subject,
        "kind": atom.kind,
        "source": atom.source,
        "content": content,
        "valid_from": atom.valid_from.isoformat() if atom.valid_from else None,
        "valid_until": atom.valid_until.isoformat() if atom.valid_until else None,
        "is_superseded": atom.is_superseded,
        "supersedes": atom.supersedes,
        "superseded_by": atom.superseded_by,
        "provenance": {
            "source_id": atom.source_id,
            "source_title": atom.source_title,
            "source_type": atom.source_type,
            "session_id": atom.session_id,
            "segment_id": atom.segment_id,
            "source_span": atom.source_span,
            "observed_at": atom.observed_at.isoformat() if atom.observed_at else None,
            "ingested_at": atom.ingested_at.isoformat() if atom.ingested_at else None,
        },
        "dedup": {
            "content_hash": atom.content_hash,
            "normalized_content_hash": atom.normalized_content_hash,
        },
    }


def _valid_as_of(atom, as_of: date | None) -> bool:
    if atom.is_superseded:
        return False
    if as_of is None:
        return True
    return (atom.valid_from is None or atom.valid_from <= as_of) and (
        atom.valid_until is None or atom.valid_until >= as_of
    )


def _run_inference(cfg: dict) -> None:
    out_path = Path(cfg["out"])
    debug_path = out_path.with_suffix("").with_name(out_path.stem + ".debug.jsonl")
    lattice_root = out_path.with_suffix("").with_name(out_path.stem + ".lattices")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg["keep_lattice_dirs"]:
        lattice_root.mkdir(parents=True, exist_ok=True)

    os.environ["LLM_PROVIDER"] = cfg["llm_provider"]
    os.environ["LLM_MODEL"] = cfg["llm_model"]

    print(f"Loading dataset: {cfg['dataset']}")
    with open(cfg["dataset"]) as f:
        data = json.load(f)

    sample = _stratify(data, cfg["stratify"], cfg["seed"])
    done_ids = _load_done_ids(out_path)

    type_counts = Counter(q["question_type"] for q in sample)
    print(f"Stratified sample: {len(sample)} questions")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")
    print(f"Already done: {len(done_ids)} — will skip")

    n_done = n_err = 0
    atoms_created_total = atoms_selected_total = bm25_candidates_total = 0

    with (
        out_path.open("a") as out_f,
        debug_path.open("a") as dbg_f,
        tqdm(total=len(sample) - len(done_ids), unit="q") as pbar,
    ):
        for item in sample:
            qid = item["question_id"]
            if qid in done_ids:
                continue

            qtype = item["question_type"]
            pbar.set_description(qtype[:24])

            if cfg["keep_lattice_dirs"]:
                tmpdir = str(lattice_root / qid)
                shutil.rmtree(tmpdir, ignore_errors=True)
                Path(tmpdir).mkdir(parents=True, exist_ok=True)
            else:
                tmpdir = tempfile.mkdtemp(prefix="lattice-eval-")
            try:
                db = LatticeDB(lattice_dir=tmpdir)

                sessions = item.get("haystack_sessions", [])
                session_ids = item.get(
                    "haystack_session_ids", [f"s{i}" for i in range(len(sessions))]
                )
                dates = item.get("haystack_dates", ["" for _ in sessions])

                atoms_created = 0
                duplicates_skipped = 0
                ingest_results: list[dict] = []
                all_atoms: list[dict] = []

                for session, sid, ts in zip(sessions, session_ids, dates):
                    text = format_session(session, sid, ts)
                    ingest_result = ingest(
                        text,
                        metadata={"source": "conversation", "date": ts, "session_id": sid},
                        db=db,
                    )
                    atoms_created += ingest_result["atoms_created"]
                    duplicates_skipped += ingest_result.get("duplicates_skipped", 0)
                    ingest_results.append(
                        {
                            "session_id": sid,
                            "date": ts,
                            "source_id": ingest_result.get("source_id"),
                            "segments_processed": ingest_result.get("segments_processed"),
                            "atoms_created": ingest_result.get("atoms_created"),
                            "atom_ids": ingest_result.get("atom_ids", []),
                            "duplicates_skipped": ingest_result.get("duplicates_skipped", 0),
                            "duplicate_atom_ids": ingest_result.get("duplicate_atom_ids", []),
                        }
                    )

                question_date_str: str | None = item.get("question_date")
                as_of: date | None = None
                if question_date_str:
                    try:
                        as_of = date.fromisoformat(question_date_str[:10])
                    except ValueError:
                        pass

                bm25_atoms = db.search(
                    item["question"], as_of=as_of, top_k=cfg["top_k"]
                )
                bm25_candidates = [
                    _atom_debug_dict(atom, preview_chars=240) for atom in bm25_atoms
                ]

                if cfg["retrieval_mode"] == "select":
                    selected = select(
                        item["question"],
                        as_of=as_of,
                        db=db,
                        top_k=cfg["top_k"],
                    )
                elif cfg["retrieval_mode"] == "bm25":
                    selected = [_atom_debug_dict(atom) for atom in bm25_atoms]
                else:
                    selected = [
                        _atom_debug_dict(atom)
                        for atom in db.all()
                        if _valid_as_of(atom, as_of)
                    ]

                synthesis = synthesize(item["question"], selected)

                all_atoms = [
                    _atom_debug_dict(a, preview_chars=240)
                    for a in db.all()
                ]

                out_f.write(
                    json.dumps({"question_id": qid, "hypothesis": synthesis.answer})
                    + "\n"
                )
                out_f.flush()

                dbg_f.write(
                    json.dumps(
                        {
                            "question_id": qid,
                            "question_type": qtype,
                            "lattice_dir": tmpdir,
                            "lattice_dir_kept": bool(cfg["keep_lattice_dirs"]),
                            "sessions_ingested": len(sessions),
                            "retrieval_mode": cfg["retrieval_mode"],
                            "top_k": cfg["top_k"],
                            "ingest_results": ingest_results,
                            "atoms_created": atoms_created,
                            "duplicates_skipped": duplicates_skipped,
                            "atoms": all_atoms,
                            "bm25_candidates": bm25_candidates,
                            "bm25_candidate_ids": [
                                atom["atom_id"] for atom in bm25_candidates
                            ],
                            "atoms_selected": [a["atom_id"] for a in selected],
                            "selected_atoms": selected,
                            "synthesis_raw": synthesis.raw_response,
                            "hypothesis": synthesis.answer,
                        }
                    )
                    + "\n"
                )
                dbg_f.flush()

                atoms_created_total += atoms_created
                atoms_selected_total += len(selected)
                bm25_candidates_total += len(bm25_candidates)
                n_done += 1

            except Exception as exc:
                out_f.write(
                    json.dumps({"question_id": qid, "hypothesis": f"ERROR: {exc}"})
                    + "\n"
                )
                out_f.flush()
                n_err += 1

            finally:
                if not cfg["keep_lattice_dirs"]:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            pbar.update(1)

    print("\n── Inference summary ──────────────────────────────────────────")
    print(f"  Processed : {n_done}")
    print(f"  Errors    : {n_err}")
    print(f"  Retrieval : {cfg['retrieval_mode']} (top_k={cfg['top_k']})")
    if n_done:
        print(f"  Avg atoms created  : {atoms_created_total / n_done:.1f}")
        print(f"  Avg BM25 candidates: {bm25_candidates_total / n_done:.1f}")
        print(f"  Avg atoms selected : {atoms_selected_total / n_done:.1f}")
    print(f"  Results   : {out_path}")
    print(f"  Debug     : {debug_path}")
    if cfg["keep_lattice_dirs"]:
        print(f"  Lattices  : {lattice_root}")

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    requests.post(
        f"{ollama_host}/api/generate", json={"model": cfg["llm_model"], "keep_alive": 0}
    )
    print(f"Unloaded {cfg['llm_model']} from GPU.")


# ── judge phase ───────────────────────────────────────────────────────────────


def _wait_for_proxy(port: int, timeout: int = 30) -> None:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"LiteLLM proxy did not start within {timeout}s")


def _ensure_model_pulled(model: str) -> None:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model not in result.stdout:
        print(f"Pulling judge model: {model}")
        subprocess.run(["ollama", "pull", model], check=True)


def _run_judge(cfg: dict) -> None:
    out_path = Path(cfg["out"])
    if not out_path.exists():
        sys.exit(f"ERROR: Results file not found: {out_path}. Run inference first.")

    evaluate_script = cfg["evaluate_script"]
    if not evaluate_script:
        sys.exit(
            "ERROR: EVALUATE_SCRIPT not set. Add path to evaluate_qa.py in .env.eval."
        )

    judge_model = cfg["judge_model"]
    _ensure_model_pulled(judge_model)

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    env = {
        **os.environ,
        "OLLAMA_BASE_URL": f"{ollama_host}/v1",
    }

    result_file = str(out_path) + f".eval-results-{judge_model}"
    print(f"Running evaluate_qa.py with judge model: {judge_model}")
    subprocess.run(
        [sys.executable, evaluate_script, judge_model, str(out_path), cfg["dataset"]],
        env=env,
        check=True,
    )

    if Path(result_file).exists():
        print("\n── Scoring summary ────────────────────────────────────────────")
        for script, args in [
            (cfg["print_qa_script"], [result_file, cfg["dataset"]]),
            (cfg["print_retrieval_script"], [result_file]),
        ]:
            if script:
                out = subprocess.run(
                    [sys.executable, script, *args],
                    capture_output=True,
                    text=True,
                )
                if out.stdout:
                    print(out.stdout, end="")
                if out.stderr:
                    print(out.stderr, end="", file=sys.stderr)

    requests.post(
        f"{ollama_host}/api/generate",
        json={"model": judge_model, "keep_alive": 0},
    )
    print(f"Unloaded {judge_model} from GPU.")


# ── entrypoint ─────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LongMemEval harness for lattice-mcp")
    p.add_argument("--phase", choices=["inference", "judge", "all"], default="all")
    p.add_argument(
        "--priority", default="", help="Iteration label, e.g. baseline, p1, p2"
    )
    p.add_argument("--dataset", default="")
    p.add_argument("--out", default="", help="Override inference results file path")
    p.add_argument("--log", default="", help="Override log file path")
    p.add_argument("--stratify", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--retrieval-mode",
        choices=["select", "bm25", "all"],
        default="",
        help=(
            "Selection diagnostic mode: select=BM25+LLM selection, "
            "bm25=bypass LLM selection, all=bypass retrieval."
        ),
    )
    p.add_argument("--top-k", type=int, default=0, help="Candidate count for BM25/select")
    p.add_argument("--evaluate-script", default="")
    p.add_argument("--print-qa-script", default="")
    p.add_argument("--print-retrieval-script", default="")
    p.add_argument(
        "--keep-lattice-dirs",
        action="store_true",
        help="Keep per-question LatticeDB directories so atom markdown files can be inspected.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args)

    # Set up log tee: write to both stdout/stderr and the log file
    log_path = Path(cfg["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    try:
        print(f"Priority : {cfg['priority'] or '(none)'}")
        print(f"Phase    : {args.phase}")
        print(f"LLM      : {cfg['llm_model']}")
        print(f"Judge    : {cfg['judge_model']}")
        print(f"Dataset  : {Path(cfg['dataset']).stem}")
        print(f"Retrieve : {cfg['retrieval_mode']} (top_k={cfg['top_k']})")
        print(f"Out      : {cfg['out']}")
        print(f"Log      : {log_path}")
        print(f"Keep DBs : {cfg['keep_lattice_dirs']}")
        print()

        if args.phase in ("inference", "all"):
            _run_inference(cfg)

        if args.phase in ("judge", "all"):
            _run_judge(cfg)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_f.close()


if __name__ == "__main__":
    main()
