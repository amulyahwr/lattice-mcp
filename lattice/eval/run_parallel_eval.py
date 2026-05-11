"""
Run retrieval-mode variants for LongMemEval diagnostics.

This runner is for product eval, not priority implementation:
  - no worktrees
  - no auto-merge
  - same codebase for every variant
  - isolated result dirs per variant

Usage:
    uv run python -m lattice.eval.run_parallel_eval --priority p3
    uv run python -m lattice.eval.run_parallel_eval --priority p3 --phase inference
    uv run python -m lattice.eval.run_parallel_eval --priority p3 --variants select bm25 all
    uv run python -m lattice.eval.run_parallel_eval --priority p3 --top-k 50 --stratify 100
    uv run python -m lattice.eval.run_parallel_eval --priority p3 --reuse-ingest
    uv run python -m lattice.eval.run_parallel_eval --priority p3 --parallel-inference  # API models only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DATASET = str(_DATA_DIR / "longmemeval_oracle.json")
_VARIANTS = ("select", "bm25", "all")


@dataclass
class VariantResult:
    variant: str
    priority: str
    inference_rc: int | None = None
    judge_rc: int | None = None
    n: int | None = None
    accuracy: float | None = None
    out_path: Path | None = None
    log_path: Path | None = None


def _slug(s: str) -> str:
    for ch in (":", "/", ".", "-"):
        s = s.replace(ch, "")
    return s


def _results_dir(priority: str) -> Path:
    return _REPO_ROOT / "results" / priority


def _default_out(llm_model: str, dataset: str, priority: str, variant: str) -> Path:
    ds = Path(dataset).stem
    suffix = "_inference" if variant == "select" else f"_{variant}_inference"
    return _results_dir(priority) / f"{_slug(llm_model)}_{ds}{suffix}.jsonl"


def _load_config(args: argparse.Namespace) -> dict:
    load_dotenv(_REPO_ROOT / ".env.eval", override=False)
    model = os.environ.get("LLM_MODEL", "gemma4:e2b")
    judge = os.environ.get("JUDGE_MODEL", "qwen3.5:4b")
    dataset = args.dataset or os.environ.get("DATASET", _DEFAULT_DATASET)
    return {
        "llm_model": model,
        "judge_model": judge,
        "dataset": dataset,
        "top_k": args.top_k or int(os.environ.get("TOP_K", "20")),
    }


def _variant_priority(base: str, variant: str) -> str:
    return f"{base}-{variant}" if base else variant


def _run_eval_cmd(
    phase: str,
    priority: str,
    variant: str,
    args: argparse.Namespace,
    cfg: dict,
    reuse_lattice_root: Path | None = None,
    force_keep_lattice_dirs: bool = False,
) -> tuple[int, Path, Path]:
    out_path = _default_out(cfg["llm_model"], cfg["dataset"], priority, variant)
    log_path = _REPO_ROOT / "logs" / priority / f"{phase}.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "lattice.eval.run_eval",
        "--phase",
        phase,
        "--priority",
        priority,
        "--retrieval-mode",
        variant,
        "--top-k",
        str(cfg["top_k"]),
        "--out",
        str(out_path),
        "--log",
        str(log_path),
    ]
    if args.stratify:
        cmd += ["--stratify", str(args.stratify)]
    if args.seed:
        cmd += ["--seed", str(args.seed)]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.keep_lattice_dirs or force_keep_lattice_dirs:
        cmd += ["--keep-lattice-dirs"]
    if reuse_lattice_root:
        cmd += ["--reuse-lattice-root", str(reuse_lattice_root)]

    rc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    return rc, out_path, log_path


def _lattice_root_for_out(out_path: Path) -> Path:
    return out_path.with_suffix("").with_name(out_path.stem + ".lattices")


def _run_inference_task(item: tuple[str, str, argparse.Namespace, dict]) -> VariantResult:
    variant, priority, args, cfg = item
    rc, out_path, log_path = _run_eval_cmd("inference", priority, variant, args, cfg)
    return VariantResult(
        variant=variant,
        priority=priority,
        inference_rc=rc,
        out_path=out_path,
        log_path=log_path,
    )


def _run_judge_task(item: tuple[str, str, argparse.Namespace, dict, Path]) -> VariantResult:
    variant, priority, args, cfg, out_path = item
    rc, _, log_path = _run_eval_cmd("judge", priority, variant, args, cfg)
    return VariantResult(
        variant=variant,
        priority=priority,
        judge_rc=rc,
        out_path=out_path,
        log_path=log_path,
    )


def _run_inference(
    variants: list[str], args: argparse.Namespace, cfg: dict
) -> dict[str, VariantResult]:
    jobs = [(v, _variant_priority(args.priority, v), args, cfg) for v in variants]
    results: dict[str, VariantResult] = {}

    if args.parallel_inference:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {pool.submit(_run_inference_task, job): job[0] for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                results[result.variant] = result
                _print_phase_result("inference", result.variant, result.inference_rc, result.log_path)
    else:
        for job in jobs:
            variant = job[0]
            print(f"[{variant}] inference starting...")
            result = _run_inference_task(job)
            results[variant] = result
            _print_phase_result("inference", result.variant, result.inference_rc, result.log_path)
    return results


def _run_inference_reuse_ingest(
    variants: list[str], args: argparse.Namespace, cfg: dict
) -> dict[str, VariantResult]:
    materializer = "select" if "select" in variants else variants[0]
    materializer_priority = _variant_priority(args.priority, materializer)

    print(f"[{materializer}] inference starting... (materialize lattice)")
    rc, out_path, log_path = _run_eval_cmd(
        "inference",
        materializer_priority,
        materializer,
        args,
        cfg,
        force_keep_lattice_dirs=True,
    )
    results = {
        materializer: VariantResult(
            variant=materializer,
            priority=materializer_priority,
            inference_rc=rc,
            out_path=out_path,
            log_path=log_path,
        )
    }
    _print_phase_result("inference", materializer, rc, log_path)
    if rc != 0:
        print("reuse-ingest stopped: materializer failed")
        return results

    lattice_root = _lattice_root_for_out(out_path)
    for variant in variants:
        if variant == materializer:
            continue
        priority = _variant_priority(args.priority, variant)
        print(f"[{variant}] inference starting... (reuse lattice: {lattice_root})")
        rc, out_path, log_path = _run_eval_cmd(
            "inference",
            priority,
            variant,
            args,
            cfg,
            reuse_lattice_root=lattice_root,
        )
        results[variant] = VariantResult(
            variant=variant,
            priority=priority,
            inference_rc=rc,
            out_path=out_path,
            log_path=log_path,
        )
        _print_phase_result("inference", variant, rc, log_path)
    return results


def _run_judge(
    variants: list[str],
    args: argparse.Namespace,
    cfg: dict,
    current: dict[str, VariantResult],
) -> dict[str, VariantResult]:
    jobs = []
    for variant in variants:
        priority = _variant_priority(args.priority, variant)
        out_path = current.get(variant, VariantResult(variant, priority)).out_path
        if out_path is None:
            out_path = _default_out(cfg["llm_model"], cfg["dataset"], priority, variant)
        if not out_path.exists():
            print(f"[{variant}] judge skipped, missing results: {out_path}")
            continue
        jobs.append((variant, priority, args, cfg, out_path))

    results = dict(current)
    if args.parallel_judge:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {pool.submit(_run_judge_task, job): job[0] for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                existing = results.get(result.variant, result)
                existing.judge_rc = result.judge_rc
                existing.log_path = result.log_path
                results[result.variant] = existing
                _print_phase_result("judge", result.variant, result.judge_rc, result.log_path)
    else:
        for job in jobs:
            variant = job[0]
            print(f"[{variant}] judge starting...")
            result = _run_judge_task(job)
            existing = results.get(result.variant, result)
            existing.judge_rc = result.judge_rc
            existing.log_path = result.log_path
            results[result.variant] = existing
            _print_phase_result("judge", result.variant, result.judge_rc, result.log_path)
    return results


def _print_phase_result(phase: str, variant: str, rc: int | None, log_path: Path | None) -> None:
    status = "OK" if rc == 0 else f"FAILED rc={rc}"
    print(f"[{variant}] {phase} {status} (log: {log_path})")


def _parse_eval_results(out_path: Path, judge_model: str) -> dict | None:
    candidates = [
        Path(str(out_path) + f".eval-results-{judge_model}"),
        Path(str(out_path) + f".eval-results-{_slug(judge_model)}"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        labels = [e["autoeval_label"]["label"] for e in entries if "autoeval_label" in e]
        if labels:
            return {
                "n": len(labels),
                "correct": sum(labels),
                "accuracy": sum(labels) / len(labels),
                "path": path,
            }
    return None


def _attach_scores(results: dict[str, VariantResult], cfg: dict) -> None:
    for result in results.values():
        if result.out_path is None:
            result.out_path = _default_out(
                cfg["llm_model"], cfg["dataset"], result.priority, result.variant
            )
        scores = _parse_eval_results(result.out_path, cfg["judge_model"])
        if not scores:
            continue
        result.n = scores["n"]
        result.accuracy = scores["accuracy"]


def _print_summary(results: dict[str, VariantResult]) -> None:
    print("\n" + "=" * 86)
    print(f"{'Variant':<10} {'Priority':<14} {'Infer':>7} {'Judge':>7} {'N':>5} {'Accuracy':>10}  Output")
    print("-" * 86)
    for variant in _VARIANTS:
        if variant not in results:
            continue
        r = results[variant]
        infer = "OK" if r.inference_rc == 0 else ("-" if r.inference_rc is None else "FAIL")
        judge = "OK" if r.judge_rc == 0 else ("-" if r.judge_rc is None else "FAIL")
        n = str(r.n) if r.n is not None else "-"
        acc = f"{r.accuracy:.1%}" if r.accuracy is not None else "-"
        print(f"{variant:<10} {r.priority:<14} {infer:>7} {judge:>7} {n:>5} {acc:>10}  {r.out_path}")
    print("=" * 86)

    ranked = [r for r in results.values() if r.accuracy is not None]
    if len(ranked) >= 2:
        ranked.sort(key=lambda r: r.accuracy or 0, reverse=True)
        best = ranked[0]
        print(f"\nBest: {best.variant} at {best.accuracy:.1%}")
        lookup = {r.variant: r for r in ranked}
        if "bm25" in lookup and "select" in lookup and lookup["bm25"].accuracy > lookup["select"].accuracy:
            print("Read: selector dropping useful atoms.")
        if "all" in lookup and "bm25" in lookup and lookup["all"].accuracy > lookup["bm25"].accuracy:
            print("Read: BM25 candidate generation weak.")
        if len({round(r.accuracy or 0, 4) for r in ranked}) == 1:
            print("Read: likely ingest missing facts or synthesis weak.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run retrieval variants for lattice-mcp eval")
    p.add_argument("--priority", default="p3", help="Base label. Outputs use <priority>-<variant>.")
    p.add_argument("--variants", nargs="+", choices=_VARIANTS, default=list(_VARIANTS))
    p.add_argument("--phase", choices=["inference", "judge", "all"], default="all")
    p.add_argument("--stratify", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--keep-lattice-dirs", action="store_true")
    p.add_argument(
        "--reuse-ingest",
        action="store_true",
        help=(
            "Ingest once by keeping first variant lattice dirs, then run remaining "
            "variants against same atoms."
        ),
    )
    p.add_argument(
        "--parallel-inference",
        action="store_true",
        help="Run inference variants concurrently. Use only for API models or enough separate GPU capacity.",
    )
    p.add_argument(
        "--parallel-judge",
        action="store_true",
        help="Run judges concurrently. Use only if judge model fits concurrently.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args)
    variants = list(args.variants)

    print(f"Priority : {args.priority}")
    print(f"Variants : {', '.join(variants)}")
    print(f"Phase    : {args.phase}")
    print(f"Dataset  : {Path(cfg['dataset']).stem}")
    print(f"LLM      : {cfg['llm_model']}")
    print(f"Judge    : {cfg['judge_model']}")
    print(f"Top K    : {cfg['top_k']}")
    print(f"ReuseIng : {args.reuse_ingest}")
    print()

    results: dict[str, VariantResult] = {
        v: VariantResult(v, _variant_priority(args.priority, v)) for v in variants
    }

    if args.phase in {"inference", "all"}:
        print("--- inference ---")
        if args.reuse_ingest:
            results.update(_run_inference_reuse_ingest(variants, args, cfg))
        else:
            results.update(_run_inference(variants, args, cfg))

    if args.phase in {"judge", "all"}:
        print("\n--- judge ---")
        results = _run_judge(variants, args, cfg, results)

    _attach_scores(results, cfg)
    _print_summary(results)


if __name__ == "__main__":
    main()
