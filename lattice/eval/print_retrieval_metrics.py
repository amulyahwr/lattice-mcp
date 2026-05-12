from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _metrics_for(rows: list[dict], mode: str) -> dict:
    values = []
    for row in rows:
        oracle = (row.get("retrieval_oracle") or {}).get(mode) or {}
        if oracle.get("session_hit") is None:
            continue
        values.append(oracle)

    return {
        "n": len(values),
        "hit": _mean([1.0 if v.get("session_hit") else 0.0 for v in values]),
        "recall": _mean([float(v.get("session_recall") or 0.0) for v in values]),
        "precision": _mean([float(v.get("session_precision") or 0.0) for v in values]),
        "mrr": _mean([float(v.get("session_mrr") or 0.0) for v in values]),
    }


def _print_table(title: str, groups: dict[str, list[dict]]) -> None:
    print(title)
    print(
        f"{'Group':<28} {'Mode':<9} {'N':>5} {'Hit':>8} {'Recall':>8} "
        f"{'Prec':>8} {'MRR':>8}"
    )
    print("-" * 82)
    for group, rows in groups.items():
        for mode in ("bm25", "selected"):
            metrics = _metrics_for(rows, mode)
            print(
                f"{group:<28} {mode:<9} {metrics['n']:>5} "
                f"{_fmt(metrics['hit']):>8} {_fmt(metrics['recall']):>8} "
                f"{_fmt(metrics['precision']):>8} {_fmt(metrics['mrr']):>8}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print lattice session-level retrieval metrics from debug JSONL."
    )
    parser.add_argument("debug_file", help="Path to *.debug.jsonl from run_eval.py")
    args = parser.parse_args()

    path = Path(args.debug_file)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "_abs" not in row.get("question_id", "")]

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row.get("question_type", "?")].append(row)

    print(f"File: {path}")
    _print_table("Overall", {"all": rows})
    _print_table("By question type", dict(sorted(by_type.items())))


if __name__ == "__main__":
    main()
