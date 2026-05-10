"""
Parallel LongMemEval evaluation runner.

Runs multiple independent priorities simultaneously:
  - Phase implement: parallel (API-bound, not GPU)
  - Phase inference: sequential (GPU-bound)
  - Phase judge:     sequential by default; --parallel-judge if VRAM fits 2x judge model
  - Phase merge:     winners merged to current branch, losers discarded

Usage:
    python -m lattice.eval.run_parallel_eval --priorities p2 p3 p4
    python -m lattice.eval.run_parallel_eval --priorities p2 p3 --phase inference --stratify 10
    python -m lattice.eval.run_parallel_eval --priorities p2 p3 --parallel-judge
    python -m lattice.eval.run_parallel_eval --priorities p2 --phase implement   # implement only
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent.parent
_PRIORITIES_MD = Path(__file__).parent / "PRIORITIES.md"
_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DATASET = str(_DATA_DIR / "longmemeval_oracle.json")

PRIORITY_DEPS: dict[str, list[str]] = {
    "p7": ["p5"],
}


# ── dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class PriorityResult:
    priority: str
    impl_rc: int | None = None
    inference_rc: int | None = None
    judge_rc: int | None = None
    n: int | None = None
    accuracy: float | None = None
    out_path: Path | None = None
    merged: bool = False
    conflict: bool = False


# ── slug / path helpers (mirrors run_eval.py:71-88) ───────────────────────────

def _slug(s: str) -> str:
    for ch in (":", "/", ".", "-"):
        s = s.replace(ch, "")
    return s


def _results_dir(priority: str) -> str:
    return f"results/{priority}" if priority else "results"


def _default_out(llm_model: str, dataset: str, priority: str) -> Path:
    ds = Path(dataset).stem
    return Path(f"{_results_dir(priority)}/{_slug(llm_model)}_{ds}_inference.jsonl")


def _find_python(repo_root: Path) -> str:
    venv = repo_root / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file without touching os.environ."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip().strip('"').strip("'")
    return result


def _derive_out_path(wt_path: Path, priority: str, args: argparse.Namespace) -> Path:
    env = _load_env_file(wt_path / ".env.eval")
    model = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", "gemma4:e4b"))
    dataset = args.dataset or env.get("DATASET", _DEFAULT_DATASET)
    return wt_path / _default_out(model, dataset, priority)


def _build_subprocess_env(wt_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{wt_path}:{existing}" if existing else str(wt_path)
    return env


# ── PRIORITIES.md parsing ──────────────────────────────────────────────────────

def _parse_priority_row(priority: str) -> dict[str, str] | None:
    """Extract files and description for a priority from the PRIORITIES.md table."""
    text = _PRIORITIES_MD.read_text()
    pattern = rf"^\|\s*{re.escape(priority.upper())}[^|]*\|\s*([^|]+)\|\s*([^|]+)\|"
    for line in text.splitlines():
        m = re.match(pattern, line, re.IGNORECASE)
        if m:
            return {"files": m.group(1).strip(), "change": m.group(2).strip()}
    return None


def _build_impl_prompt(priority: str) -> str:
    row = _parse_priority_row(priority)
    if not row:
        return (
            f"Implement priority {priority.upper()} as described in "
            "lattice/eval/PRIORITIES.md. Read PRIORITIES.md first, then implement the change."
        )
    return (
        f"Implement priority {priority.upper()} in this lattice-mcp repository.\n\n"
        f"Files to modify: {row['files']}\n"
        f"Change description: {row['change']}\n\n"
        "Read PRIORITIES.md and CLAUDE.md first for full context. "
        "Then implement the change described above. Make only the changes needed for this priority — "
        "no refactoring, no extra features. After implementing, run `uv run pytest` to verify "
        "nothing is broken."
    )


def parse_baseline(md_path: Path = _PRIORITIES_MD) -> float | None:
    """Read last non-empty accuracy from the Results table in PRIORITIES.md."""
    text = md_path.read_text()
    accuracies = re.findall(r"\|\s*[\w]+\s*\|\s*[^|]+\|\s*([\d.]+)%\s*\|", text)
    if not accuracies:
        return None
    return float(accuracies[-1]) / 100.0


def parse_eval_results(out_path: Path, judge_model: str) -> dict | None:
    """Parse accuracy from evaluate_qa.py output (.eval-results-{model} JSONL)."""
    for suffix in (_slug(judge_model), judge_model):
        results_file = Path(str(out_path) + f".eval-results-{suffix}")
        if results_file.exists():
            entries = [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()]
            labels = [e["autoeval_label"]["label"] for e in entries if "autoeval_label" in e]
            if labels:
                return {"n": len(labels), "accuracy": sum(labels) / len(labels), "correct": sum(labels)}
    return None


# ── worktree management ────────────────────────────────────────────────────────

def create_worktree(repo_root: Path, priority: str, base: str) -> tuple[Path, str]:
    wt_path = Path(base) / f"lattice-eval-{priority}"
    branch = f"eval/{priority}"
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(wt_path), "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    env_src = repo_root / ".env.eval"
    if env_src.exists():
        shutil.copy(env_src, wt_path / ".env.eval")
    return wt_path, branch


def remove_worktree(repo_root: Path, wt_path: Path, branch: str | None = None) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(wt_path)],
        cwd=repo_root,
        capture_output=True,
    )
    if branch:
        subprocess.run(["git", "branch", "-D", branch], cwd=repo_root, capture_output=True)


# ── implementation phase ───────────────────────────────────────────────────────

def implement_priority(wt_path: Path, priority: str) -> int:
    """Spawn Claude CLI to implement the priority non-interactively in the worktree."""
    prompt = _build_impl_prompt(priority)
    log_path = wt_path / f"logs/{priority}/implement.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        result = subprocess.run(
            ["claude", "-p", prompt, "--dangerouslySkipPermissions"],
            cwd=wt_path,
            env=dict(os.environ),
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if result.returncode == 0:
        subprocess.run(["git", "add", "-A"], cwd=wt_path, capture_output=True)
        commit = subprocess.run(
            ["git", "commit", "-m", f"eval: implement {priority}"],
            cwd=wt_path,
            capture_output=True,
        )
        # rc=1 means nothing to commit — treat as success (impl may have found nothing to change)
        if commit.returncode not in (0, 1):
            return commit.returncode

    return result.returncode


def run_all_implementations(worktrees: dict[str, tuple[Path, str]]) -> dict[str, int]:
    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(worktrees)) as pool:
        futures = {pool.submit(implement_priority, wt, p): p for p, (wt, _) in worktrees.items()}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                results[p] = fut.result()
            except Exception as exc:
                print(f"[{p}] implementation raised: {exc}")
                results[p] = 1
    return results


# ── inference / judge ──────────────────────────────────────────────────────────

def run_inference(repo_root: Path, wt_path: Path, priority: str, args: argparse.Namespace) -> tuple[int, Path]:
    out_path = _derive_out_path(wt_path, priority, args)
    log_path = wt_path / f"logs/{priority}/inference.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [_find_python(repo_root), "-m", "lattice.eval.run_eval",
           "--phase", "inference", "--priority", priority, "--out", str(out_path)]
    if args.stratify:
        cmd += ["--stratify", str(args.stratify)]
    if args.seed:
        cmd += ["--seed", str(args.seed)]
    if args.dataset:
        cmd += ["--dataset", args.dataset]

    with open(log_path, "w") as f:
        rc = subprocess.run(
            cmd, cwd=wt_path, env=_build_subprocess_env(wt_path),
            stdout=f, stderr=subprocess.STDOUT,
        ).returncode
    return rc, out_path


def run_judge(repo_root: Path, wt_path: Path, priority: str, out_path: Path, args: argparse.Namespace) -> int:
    log_path = wt_path / f"logs/{priority}/judge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [_find_python(repo_root), "-m", "lattice.eval.run_eval",
           "--phase", "judge", "--priority", priority, "--out", str(out_path)]

    with open(log_path, "w") as f:
        return subprocess.run(
            cmd, cwd=wt_path, env=_build_subprocess_env(wt_path),
            stdout=f, stderr=subprocess.STDOUT,
        ).returncode


def _run_judge_task(args_tuple: tuple) -> int:
    repo_root, wt_path, priority, out_path, args = args_tuple
    return run_judge(repo_root, wt_path, priority, out_path, args)


def run_judges_parallel(
    repo_root: Path,
    jobs: list[tuple[Path, str, Path, argparse.Namespace]],
) -> list[int]:
    tasks = [(repo_root, wt, p, out, a) for wt, p, out, a in jobs]
    rc_by_idx: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(_run_judge_task, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                rc_by_idx[idx] = fut.result()
            except Exception as exc:
                print(f"judge raised: {exc}")
                rc_by_idx[idx] = 1
    return [rc_by_idx[i] for i in range(len(tasks))]


# ── merge / discard ────────────────────────────────────────────────────────────

def apply_winner(repo_root: Path, priority: str, branch: str) -> bool:
    result = subprocess.run(
        ["git", "merge", "--no-ff", branch, "-m", f"feat: eval {priority} (accuracy improved)"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        subprocess.run(["git", "merge", "--abort"], cwd=repo_root, capture_output=True)
        return False
    return True


# ── output ─────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[PriorityResult], baseline: float | None) -> None:
    baseline_pct = f"{baseline:.1%}" if baseline is not None else "—"
    print(f"\n{'=' * 72}")
    print(f"Baseline: {baseline_pct}")
    print(f"{'Priority':<10} {'Impl':>6} {'Infer':>6} {'Judge':>6} {'N':>5} {'Accuracy':>10} {'Delta':>8}  Outcome")
    print(f"{'-' * 72}")
    for r in results:
        impl = "OK" if r.impl_rc == 0 else ("—" if r.impl_rc is None else "FAIL")
        inf = "OK" if r.inference_rc == 0 else ("—" if r.inference_rc is None else "FAIL")
        jud = "OK" if r.judge_rc == 0 else ("—" if r.judge_rc is None else "FAIL")
        acc = f"{r.accuracy:.1%}" if r.accuracy is not None else "—"
        if r.accuracy is not None and baseline is not None:
            delta = f"{(r.accuracy - baseline) * 100:+.1f}pp"
        else:
            delta = "—"
        if r.merged:
            outcome = "MERGED"
        elif r.conflict:
            outcome = "CONFLICT (manual)"
        elif r.inference_rc is not None and r.inference_rc != 0:
            outcome = "error"
        elif r.accuracy is not None and baseline is not None and r.accuracy <= baseline:
            outcome = "discarded"
        else:
            outcome = "—"
        n_str = str(r.n) if r.n is not None else "—"
        print(f"{r.priority:<10} {impl:>6} {inf:>6} {jud:>6} {n_str:>5} {acc:>10} {delta:>8}  {outcome}")
    print(f"{'=' * 72}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multiple LongMemEval priorities in parallel")
    p.add_argument("--priorities", nargs="+", required=True, metavar="P",
                   help="Priority labels, e.g. p2 p3 p4")
    p.add_argument("--phase", choices=["implement", "inference", "judge", "all"], default="all")
    p.add_argument("--parallel-judge", action="store_true",
                   help="Run judge phases concurrently (only safe if judge model fits 2x in VRAM)")
    p.add_argument("--baseline", type=float, default=0.0,
                   help="Accuracy to beat (0 = read from PRIORITIES.md results table)")
    p.add_argument("--stratify", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="")
    p.add_argument("--keep-losers", action="store_true",
                   help="Keep worktrees/branches for priorities that did not improve (debug)")
    p.add_argument("--worktree-base", default="/tmp",
                   help="Parent directory for worktrees (default: /tmp)")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    repo_root = _REPO_ROOT
    priorities = [p.lower() for p in args.priorities]

    # Dependency warnings
    for p, deps in PRIORITY_DEPS.items():
        if p in priorities:
            for dep in deps:
                if dep not in priorities:
                    print(f"WARNING: {p} depends on {dep} which is not in this run.")

    # Baseline
    baseline = args.baseline if args.baseline else (parse_baseline() or 0.0)
    print(f"Baseline accuracy : {baseline:.1%}")
    print(f"Priorities        : {', '.join(priorities)}")
    print()

    worktrees: dict[str, tuple[Path, str]] = {}
    results: dict[str, PriorityResult] = {p: PriorityResult(priority=p) for p in priorities}

    # Track live worktrees for atexit cleanup (entries removed as they are cleaned up)
    _live: list[tuple[Path, str]] = []

    def _cleanup_all() -> None:
        for wt_path, branch in list(_live):
            try:
                remove_worktree(repo_root, wt_path, branch)
            except Exception:
                pass

    atexit.register(_cleanup_all)

    # ── Phase 0: Create all worktrees ─────────────────────────────────────────
    for p in priorities:
        try:
            wt_path, branch = create_worktree(repo_root, p, args.worktree_base)
            worktrees[p] = (wt_path, branch)
            _live.append((wt_path, branch))
            print(f"[{p}] worktree ready: {wt_path}")
        except subprocess.CalledProcessError as e:
            print(f"[{p}] FAILED to create worktree: {e}")

    # ── Phase 1: Parallel implementation ──────────────────────────────────────
    if args.phase in ("implement", "all"):
        print("\n--- implement (parallel) ---")
        impl_results = run_all_implementations(
            {p: worktrees[p] for p in priorities if p in worktrees}
        )
        for p, rc in impl_results.items():
            results[p].impl_rc = rc
            print(f"[{p}] implement {'OK' if rc == 0 else f'FAILED (rc={rc})'}")

    # ── Phase 2: Sequential inference ─────────────────────────────────────────
    out_paths: dict[str, Path] = {}
    if args.phase in ("inference", "all"):
        print("\n--- inference (sequential) ---")
        for p in priorities:
            if p not in worktrees:
                continue
            if results[p].impl_rc is not None and results[p].impl_rc != 0:
                print(f"[{p}] skipping inference (impl failed)")
                continue
            wt_path, _ = worktrees[p]
            log = wt_path / f"logs/{p}/inference.log"
            print(f"[{p}] inference starting... (log: {log})")
            rc, out_path = run_inference(repo_root, wt_path, p, args)
            results[p].inference_rc = rc
            out_paths[p] = out_path
            print(f"[{p}] inference {'OK' if rc == 0 else f'FAILED (rc={rc})'}")

    # ── Phase 3: Judge ────────────────────────────────────────────────────────
    if args.phase in ("judge", "all"):
        print("\n--- judge ---")
        eligible = [
            p for p in priorities
            if p in worktrees and p in out_paths and results[p].inference_rc == 0
        ]
        if args.parallel_judge:
            jobs = [(worktrees[p][0], p, out_paths[p], args) for p in eligible]
            rcs = run_judges_parallel(repo_root, jobs)
            for p, rc in zip(eligible, rcs):
                results[p].judge_rc = rc
                print(f"[{p}] judge {'OK' if rc == 0 else f'FAILED (rc={rc})'}")
        else:
            for p in eligible:
                wt_path, _ = worktrees[p]
                log = wt_path / f"logs/{p}/judge.log"
                print(f"[{p}] judge starting... (log: {log})")
                rc = run_judge(repo_root, wt_path, p, out_paths[p], args)
                results[p].judge_rc = rc
                print(f"[{p}] judge {'OK' if rc == 0 else f'FAILED (rc={rc})'}")

    # ── Parse scores ──────────────────────────────────────────────────────────
    root_env = _load_env_file(repo_root / ".env.eval")
    judge_model = root_env.get("JUDGE_MODEL", os.environ.get("JUDGE_MODEL", "qwen3.5:4b"))
    for p, out_path in out_paths.items():
        scores = parse_eval_results(out_path, judge_model)
        if scores:
            results[p].n = scores["n"]
            results[p].accuracy = scores["accuracy"]
        results[p].out_path = out_path

    # ── Phase 4: Merge winners / discard losers ───────────────────────────────
    print("\n--- merge / discard ---")
    for p in priorities:
        if p not in worktrees:
            continue
        wt_path, branch = worktrees[p]
        r = results[p]

        is_winner = r.accuracy is not None and r.accuracy > baseline

        if is_winner:
            print(f"[{p}] WINNER ({r.accuracy:.1%} > {baseline:.1%}) — merging {branch}...")
            ok = apply_winner(repo_root, p, branch)
            if ok:
                r.merged = True
                # Remove worktree; branch stays in history via merge commit
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(wt_path)],
                    cwd=repo_root, capture_output=True,
                )
                _live[:] = [(w, b) for w, b in _live if b != branch]
                print(f"[{p}] merged.")
            else:
                r.conflict = True
                print(f"[{p}] MERGE CONFLICT — resolve manually: git merge {branch}")
        else:
            acc_str = f"{r.accuracy:.1%}" if r.accuracy is not None else "no score"
            print(f"[{p}] no improvement ({acc_str} vs {baseline:.1%}) — discarding.")
            if not args.keep_losers:
                remove_worktree(repo_root, wt_path, branch)
                _live[:] = [(w, b) for w, b in _live if b != branch]

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary_table(list(results.values()), baseline if baseline else None)


if __name__ == "__main__":
    main()
