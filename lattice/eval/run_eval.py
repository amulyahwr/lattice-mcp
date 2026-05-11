"""
LongMemEval evaluation harness for lattice-mcp.

Usage:
    python -m lattice.eval.run_eval                             # full run (inference + judge)
    python -m lattice.eval.run_eval --phase inference           # inference only
    python -m lattice.eval.run_eval --phase judge               # judge only (needs results file)
    python -m lattice.eval.run_eval --stratify 10               # quick smoke test (10 questions)
    python -m lattice.eval.run_eval --out results/myrun.jsonl   # custom output path

    # capture stdout + stderr to a log file while still seeing progress:
    python -m lattice.eval.run_eval --phase inference 2>&1 | tee run_<llm>_<dataset>_<phase>.log
    python -m lattice.eval.run_eval --phase judge 2>&1 | tee run_<judge_llm>_<dataset>_<phase>.log
    python -m lattice.eval.run_eval --phase all 2>&1 | tee run_<llm>_<judge_llm>_<dataset>_<phase>.log

    # file naming convention (auto-derived):
    #   results/<llm>_<judge>_<dataset>_<phase>.jsonl
    #   run_<llm>_<judge>_<dataset>_<phase>.log
    # e.g.: results/gemma4e4b_qwen2514b_longmemeval_oracle_inference.jsonl

    # Debug viewer — browse results by question, failure mode, atoms:
    uv run streamlit run lattice/eval/debug_viewer.py

Defaults (overridable via CLI flags or env vars in .env.eval):
    dataset  : lattice/eval/data/longmemeval_oracle.json
    out      : results/<model-slug>_baseline.jsonl  (auto-derived from LLM_MODEL)
    stratify : 100 questions, stratified by question_type
    seed     : 42

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


def _default_out(llm_model: str, dataset: str, priority: str) -> str:
    """Inference output path. Always named by llm_slug only — judge slug appears in eval-results filename."""
    ds = Path(dataset).stem
    return f"{_results_dir(priority)}/{_slug(llm_model)}_{ds}_inference.jsonl"


def _default_log(llm_model: str, judge_model: str, dataset: str, phase: str, priority: str) -> str:
    ds = Path(dataset).stem
    base = _logs_dir(priority)
    if phase == "inference":
        name = f"run_{_slug(llm_model)}_{ds}_inference.log"
    elif phase == "judge":
        name = f"run_{_slug(judge_model)}_{ds}_judge.log"
    else:
        name = f"run_{_slug(llm_model)}_{_slug(judge_model)}_{ds}_all.log"
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
    model = os.environ.get("LLM_MODEL", "gemma4:26b")
    judge = os.environ.get("JUDGE_MODEL", "qwen2.5:14b")
    dataset = args.dataset or os.environ.get("DATASET", _DEFAULT_DATASET)
    phase = args.phase
    priority = args.priority or os.environ.get("PRIORITY", "")
    cfg = {
        "dataset": dataset,
        "out": args.out
        or os.environ.get("OUT", _default_out(model, dataset, priority)),
        "log": args.log
        or os.environ.get("LOG", _default_log(model, judge, dataset, phase, priority)),
        "priority": priority,
        "stratify": args.stratify or int(os.environ.get("STRATIFY", "100")),
        "seed": args.seed or int(os.environ.get("SEED", "42")),
        "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
        "llm_model": model,
        "judge_model": judge,
        "litellm_port": int(os.environ.get("LITELLM_PORT", "4000")),
        "evaluate_script": args.evaluate_script
        or os.environ.get("EVALUATE_SCRIPT", ""),
        "print_qa_script": args.print_qa_script or os.environ.get("PRINT_QA_SCRIPT", ""),
        "print_retrieval_script": args.print_retrieval_script
        or os.environ.get("PRINT_RETRIEVAL_SCRIPT", ""),
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


def _run_inference(cfg: dict) -> None:
    out_path = Path(cfg["out"])
    debug_path = out_path.with_suffix("").with_name(out_path.stem + ".debug.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    atoms_created_total = atoms_selected_total = 0

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

            tmpdir = tempfile.mkdtemp(prefix="lattice-eval-")
            try:
                db = LatticeDB(lattice_dir=tmpdir)

                sessions = item.get("haystack_sessions", [])
                session_ids = item.get(
                    "haystack_session_ids", [f"s{i}" for i in range(len(sessions))]
                )
                dates = item.get("haystack_dates", ["" for _ in sessions])

                atoms_created = 0
                all_atoms: list[dict] = []

                for session, sid, ts in zip(sessions, session_ids, dates):
                    text = format_session(session, sid, ts)
                    ingest_result = ingest(
                        text,
                        metadata={"source": "conversation", "date": ts},
                        db=db,
                    )
                    atoms_created += ingest_result["atoms_created"]

                question_date_str: str | None = item.get("question_date")
                as_of: date | None = None
                if question_date_str:
                    try:
                        as_of = date.fromisoformat(question_date_str[:10])
                    except ValueError:
                        pass

                selected = select(item["question"], as_of=as_of, db=db)
                synthesis = synthesize(item["question"], selected)

                all_atoms = [
                    {
                        "atom_id": a.atom_id,
                        "subject": a.subject,
                        "content": a.content[:120],
                    }
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
                            "sessions_ingested": len(sessions),
                            "atoms_created": atoms_created,
                            "atoms": all_atoms,
                            "atoms_selected": [a["atom_id"] for a in selected],
                            "synthesis_raw": synthesis.raw_response,
                            "hypothesis": synthesis.answer,
                        }
                    )
                    + "\n"
                )
                dbg_f.flush()

                atoms_created_total += atoms_created
                atoms_selected_total += len(selected)
                n_done += 1

            except Exception as exc:
                out_f.write(
                    json.dumps({"question_id": qid, "hypothesis": f"ERROR: {exc}"})
                    + "\n"
                )
                out_f.flush()
                n_err += 1

            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            pbar.update(1)

    print("\n── Inference summary ──────────────────────────────────────────")
    print(f"  Processed : {n_done}")
    print(f"  Errors    : {n_err}")
    if n_done:
        print(f"  Avg atoms created  : {atoms_created_total / n_done:.1f}")
        print(f"  Avg atoms selected : {atoms_selected_total / n_done:.1f}")
    print(f"  Results   : {out_path}")
    print(f"  Debug     : {debug_path}")

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    requests.post(f"{ollama_host}/api/generate", json={"model": cfg["llm_model"], "keep_alive": 0})
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
                    capture_output=True, text=True,
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
    p.add_argument("--priority", default="", help="Iteration label, e.g. baseline, p1, p2")
    p.add_argument("--dataset", default="")
    p.add_argument("--out", default="", help="Override inference results file path")
    p.add_argument("--log", default="", help="Override log file path")
    p.add_argument("--stratify", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evaluate-script", default="")
    p.add_argument("--print-qa-script", default="")
    p.add_argument("--print-retrieval-script", default="")
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
        print(f"Out      : {cfg['out']}")
        print(f"Log      : {log_path}")
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
