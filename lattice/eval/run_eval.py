"""
LongMemEval evaluation harness for lattice-mcp.

Usage:
    uv run python -m lattice.eval.run_eval                      # full run
    uv run python -m lattice.eval.run_eval --phase inference    # inference only
    uv run python -m lattice.eval.run_eval --phase judge        # judge only (inference already done)

Config via .env.eval (copy from .env.eval.example).
All .env.eval keys can be overridden as CLI flags.
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

import mlflow
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


def _default_out(model: str) -> str:
    slug = model.replace(":", "").replace("/", "_")
    return f"results/{slug}_baseline.jsonl"


def _load_config(args: argparse.Namespace) -> dict:
    load_dotenv(".env.eval", override=False)
    model = os.environ.get("LLM_MODEL", "gemma4:26b")
    cfg = {
        "dataset":      args.dataset  or os.environ.get("DATASET", _DEFAULT_DATASET),
        "out":          args.out       or os.environ.get("OUT", _default_out(model)),
        "stratify":     args.stratify  or int(os.environ.get("STRATIFY", "100")),
        "seed":         args.seed      or int(os.environ.get("SEED", "42")),
        "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
        "llm_model":    model,
        "judge_model":  os.environ.get("JUDGE_MODEL", "qwen2.5:14b"),
        "litellm_port": int(os.environ.get("LITELLM_PORT", "4000")),
        "evaluate_script": args.evaluate_script or os.environ.get("EVALUATE_SCRIPT", ""),
        "print_script":    args.print_script    or os.environ.get("PRINT_SCRIPT", ""),
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
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["question_id"])
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

    mlflow.litellm.autolog()
    mlflow.set_experiment(Path(cfg["out"]).stem)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

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
                with mlflow.start_run(
                    run_name=qid,
                    tags={
                        "question_type": qtype,
                        "model": cfg["llm_model"],
                        "provider": cfg["llm_provider"],
                    },
                ):
                    db = LatticeDB(lattice_dir=tmpdir)

                    sessions = item.get("haystack_sessions", [])
                    session_ids = item.get("haystack_session_ids", [f"s{i}" for i in range(len(sessions))])
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

                    # collect atom details for debug log
                    all_atoms = [
                        {"atom_id": a.atom_id, "subject": a.subject, "content": a.content[:120]}
                        for a in db.all()
                    ]

                    out_f.write(json.dumps({"question_id": qid, "hypothesis": synthesis.answer}) + "\n")
                    out_f.flush()

                    dbg_f.write(json.dumps({
                        "question_id": qid,
                        "question_type": qtype,
                        "sessions_ingested": len(sessions),
                        "atoms_created": atoms_created,
                        "atoms": all_atoms,
                        "atoms_selected": [a["atom_id"] for a in selected],
                        "synthesis_raw": synthesis.raw_response,
                        "hypothesis": synthesis.answer,
                    }) + "\n")
                    dbg_f.flush()

                    atoms_created_total += atoms_created
                    atoms_selected_total += len(selected)
                    n_done += 1

            except Exception as exc:
                out_f.write(json.dumps({"question_id": qid, "hypothesis": f"ERROR: {exc}"}) + "\n")
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
        sys.exit("ERROR: EVALUATE_SCRIPT not set. Add path to evaluate_qa.py in .env.eval.")

    judge_model = cfg["judge_model"]
    port = cfg["litellm_port"]

    _ensure_model_pulled(judge_model)

    print(f"Starting LiteLLM proxy → ollama/{judge_model} on port {port}")
    proxy = subprocess.Popen(
        ["litellm", "--model", f"ollama/{judge_model}", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_proxy(port)
        print("Proxy ready. Running evaluate_qa.py ...")

        env = {
            **os.environ,
            "OPENAI_BASE_URL": f"http://localhost:{port}/v1",
            "OPENAI_API_KEY": "dummy",
        }
        result_file = str(out_path) + ".eval-results-gpt-4o"
        subprocess.run(
            [sys.executable, evaluate_script, "gpt-4o", str(out_path), cfg["dataset"]],
            env=env,
            check=True,
        )

        print_script = cfg["print_script"]
        if print_script and Path(result_file).exists():
            print("\n── Scoring summary ────────────────────────────────────────────")
            subprocess.run([sys.executable, print_script, result_file])

    finally:
        proxy.terminate()
        proxy.wait()
        print("LiteLLM proxy stopped.")


# ── entrypoint ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LongMemEval harness for lattice-mcp")
    p.add_argument("--phase", choices=["inference", "judge", "all"], default="all")
    p.add_argument("--dataset", default="")
    p.add_argument("--out", default="")
    p.add_argument("--stratify", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evaluate-script", default="")
    p.add_argument("--print-script", default="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args)

    if args.phase in ("inference", "all"):
        _run_inference(cfg)

    if args.phase in ("judge", "all"):
        _run_judge(cfg)


if __name__ == "__main__":
    main()
