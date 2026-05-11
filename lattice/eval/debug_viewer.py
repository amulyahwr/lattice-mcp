"""
Lattice eval debug viewer — browse inference results by question.

Usage:
    uv run --group eval streamlit run lattice/eval/debug_viewer.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"
DATASET_PATH = Path(__file__).parent / "data" / "longmemeval_oracle.json"

st.set_page_config(page_title="Lattice Eval Debugger", layout="wide")
st.title("Lattice Eval Debugger")


# ── data loading ──────────────────────────────────────────────────────────────


@st.cache_data
def load_dataset() -> dict[str, dict]:
    if not DATASET_PATH.exists():
        return {}
    with open(DATASET_PATH) as f:
        return {d["question_id"]: d for d in json.load(f)}


@st.cache_data
def load_priority(priority: str) -> dict:
    pdir = RESULTS_DIR if priority == "(root)" else RESULTS_DIR / priority
    debug: dict[str, dict] = {}
    labels: dict[str, bool] = {}
    debug_files = sorted(pdir.glob("*.debug.jsonl"))
    label_files = sorted(pdir.glob("*.eval-results-*"))

    for path in debug_files:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    debug[d["question_id"]] = d
                except Exception:
                    pass

    for path in label_files:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    labels[d["question_id"]] = d["autoeval_label"]["label"]
                except Exception:
                    pass

    return {
        "debug": debug,
        "labels": labels,
        "debug_files": [str(p) for p in debug_files],
        "label_files": [str(p) for p in label_files],
    }


def _failure_type(d: dict, label: bool | None) -> str:
    if label is True:
        return "correct"
    if label is None:
        return "unlabelled"
    if d.get("atoms_created", 0) == 0:
        return "0 atoms created"
    if len(d.get("atoms_selected", [])) == 0:
        return "0 atoms selected"
    return "wrong synthesis"


def _text(value: Any, limit: int = 300) -> str:
    s = "" if value is None else str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _source_time(atom: dict) -> str | None:
    prov = atom.get("provenance") or {}
    return prov.get("observed_at") or atom.get("observed_at")


def _source_session(atom: dict) -> str | None:
    prov = atom.get("provenance") or {}
    return prov.get("session_id") or atom.get("session_id")


def _atom_file(lattice_dir: str | None, atom_id: str) -> Path | None:
    if not lattice_dir:
        return None
    path = Path(lattice_dir) / f"{atom_id}.md"
    return path if path.exists() else None


# ── sidebar ───────────────────────────────────────────────────────────────────

priorities = sorted(
    [
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and list(d.glob("*.debug.jsonl"))
    ],
    reverse=True,
)
if list(RESULTS_DIR.glob("*.debug.jsonl")):
    priorities.insert(0, "(root)")
if not priorities:
    st.error("No inference results found. Run `run_eval --phase inference` first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    priority = st.selectbox("Priority", priorities)
    qtype_filter = st.selectbox(
        "Question type",
        [
            "all",
            "temporal-reasoning",
            "multi-session",
            "single-session-user",
            "single-session-preference",
            "single-session-assistant",
            "knowledge-update",
            "abstention",
        ],
    )
    result_filter = st.selectbox("Result", ["all", "correct ✓", "wrong ✗"])
    failure_filter = st.selectbox(
        "Failure mode",
        ["all", "correct", "unlabelled", "0 atoms created", "0 atoms selected", "wrong synthesis"],
    )
    search = st.text_input("Search qid/question/answer/atom", "")
    min_atoms = st.number_input("Min atoms created", min_value=0, value=0, step=1)
    show_raw = st.checkbox("Show raw JSON", value=False)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ── load ──────────────────────────────────────────────────────────────────────

dataset = load_dataset()
pdata = load_priority(priority)
debug, labels = pdata["debug"], pdata["labels"]

with st.expander("Run files", expanded=False):
    st.write("Debug files")
    st.code("\n".join(pdata.get("debug_files", [])) or "(none)")
    st.write("Judge files")
    st.code("\n".join(pdata.get("label_files", [])) or "(none)")


# ── build rows ────────────────────────────────────────────────────────────────

rows = []
for qid, d in debug.items():
    ds = dataset.get(qid, {})
    label = labels.get(qid)
    failure = _failure_type(d, label)
    rows.append(
        {
            "qid": qid,
            "type": d.get("question_type", ds.get("question_type", "?")),
            "✓/✗": ("✓" if label else "✗") if label is not None else "—",
            "failure": failure,
            "atoms↑": d.get("atoms_created", 0),
            "bm25↑": len(d.get("bm25_candidate_ids", [])),
            "sel↑": len(d.get("atoms_selected", [])),
            "dupes": d.get("duplicates_skipped", 0),
            "sessions": d.get("sessions_ingested", 0),
            "mode": d.get("retrieval_mode", "select"),
            "kept": "yes" if d.get("lattice_dir_kept") else "",
            "question": ds.get("question", "")[:70],
            "answer": _text(ds.get("answer", ""), 120),
            "haystack": " ".join(
                _text(turn.get("content", ""), 120)
                for sess in ds.get("haystack_sessions", [])
                for turn in sess[:3]
            ),
        }
    )

df = pd.DataFrame(rows)

# apply filters
if qtype_filter != "all":
    df = df[df["type"] == qtype_filter]
if result_filter == "correct ✓":
    df = df[df["✓/✗"] == "✓"]
elif result_filter == "wrong ✗":
    df = df[df["✓/✗"] == "✗"]
if failure_filter != "all":
    df = df[df["failure"] == failure_filter]
if min_atoms:
    df = df[df["atoms↑"] >= min_atoms]
if search:
    needle = search.lower()
    matching_ids = []
    for qid, d in debug.items():
        ds = dataset.get(qid, {})
        atom_text = " ".join(str(a.get("content", "")) for a in d.get("atoms", []))
        haystack = " ".join(
            str(turn.get("content", ""))
            for sess in ds.get("haystack_sessions", [])
            for turn in sess
        )
        combined = " ".join(
            [
                qid,
                str(ds.get("question", "")),
                str(ds.get("answer", "")),
                str(d.get("hypothesis", "")),
                atom_text,
                haystack,
            ]
        ).lower()
        if needle in combined:
            matching_ids.append(qid)
    df = df[df["qid"].isin(matching_ids)]


# ── summary ───────────────────────────────────────────────────────────────────

labelled = df[df["✓/✗"] != "—"]
correct_n = (df["✓/✗"] == "✓").sum()
wrong_n = (df["✓/✗"] == "✗").sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Shown", len(df))
c2.metric("Accuracy", f"{correct_n / len(labelled):.1%}" if len(labelled) else "—")
c3.metric("✓ Correct", int(correct_n))
c4.metric("✗ Wrong", int(wrong_n))
c5.metric("Avg atoms↑", f"{df['atoms↑'].mean():.1f}" if len(df) else "—")
c6.metric("Avg BM25↑", f"{df['bm25↑'].mean():.1f}" if len(df) else "—")

# failure breakdown bar chart
wrong_df = df[df["✓/✗"] == "✗"]
if len(wrong_df):
    with st.expander("Overview charts", expanded=True):
        breakdown = (
            wrong_df["failure"]
            .value_counts()
            .rename_axis("mode")
            .reset_index(name="count")
        )
        st.bar_chart(breakdown.set_index("mode"))

        by_type = (
            df.assign(correct=df["✓/✗"].eq("✓"))
            .groupby("type")
            .agg(
                total=("qid", "count"),
                accuracy=("correct", "mean"),
                atoms=("atoms↑", "mean"),
                bm25=("bm25↑", "mean"),
                selected=("sel↑", "mean"),
            )
            .reset_index()
            .sort_values("accuracy")
        )
        st.markdown("**By question type**")
        st.dataframe(by_type, use_container_width=True)


# ── question table ────────────────────────────────────────────────────────────

st.subheader("Questions")

if df.empty:
    st.info("No questions match current filters.")
    st.stop()


def _style_result(val: str) -> str:
    if val == "✓":
        return "color: #4caf50; font-weight: bold"
    if val == "✗":
        return "color: #f44336; font-weight: bold"
    return "color: #888"


display_cols = [
    "qid",
    "type",
    "✓/✗",
    "failure",
    "mode",
    "atoms↑",
    "bm25↑",
    "sel↑",
    "dupes",
    "sessions",
    "kept",
    "question",
]
styled_table = df[display_cols].style.map(_style_result, subset=["✓/✗"])
st.dataframe(styled_table, use_container_width=True, height=280)


# ── question detail ───────────────────────────────────────────────────────────

st.subheader("Question detail")

qid_options = df["qid"].tolist()
selected_qid = st.selectbox(
    "Select question",
    qid_options,
    format_func=lambda qid: (
        f"{qid}  ·  {df.loc[df['qid']==qid, 'type'].values[0]}"
        f"  ·  {df.loc[df['qid']==qid, '✓/✗'].values[0]}"
        f"  ·  {df.loc[df['qid']==qid, 'failure'].values[0]}"
    ),
)

d = debug[selected_qid]
ds = dataset.get(selected_qid, {})
label = labels.get(selected_qid)
failure = _failure_type(d, label)

tab_summary, tab_atoms, tab_bm25, tab_selected, tab_sessions, tab_raw = st.tabs(
    ["Summary", "Atoms", "BM25", "Selected", "Sessions", "Raw"]
)

with tab_summary:
    col_q, col_ans = st.columns(2)
    with col_q:
        st.markdown("**Question**")
        st.info(ds.get("question", "—"))
    with col_ans:
        st.markdown("**Expected answer**")
        st.success(ds.get("answer", "—"))

    got = d.get("hypothesis", "—")
    st.markdown("**Got**")
    if label is True:
        st.success(got)
    elif label is False:
        st.error(got)
    else:
        st.info(got)

    if failure == "0 atoms created":
        st.warning(
            f"Ingest failure: 0 atoms created across {d.get('sessions_ingested', 0)} session(s)"
        )
    elif failure == "0 atoms selected":
        st.warning(
            f"Selection failure: {d.get('atoms_created', 0)} atoms created but 0 selected for retrieval"
        )
    elif failure == "wrong synthesis":
        st.warning(
            f"Synthesis failure: {len(d.get('atoms_selected', []))} atom(s) selected but answer is wrong"
        )

    meta_cols = st.columns(6)
    meta_cols[0].metric("Atoms", d.get("atoms_created", 0))
    meta_cols[1].metric("BM25", len(d.get("bm25_candidate_ids", [])))
    meta_cols[2].metric("Selected", len(d.get("atoms_selected", [])))
    meta_cols[3].metric("Duplicates", d.get("duplicates_skipped", 0))
    meta_cols[4].metric("Sessions", d.get("sessions_ingested", 0))
    meta_cols[5].metric("Mode", d.get("retrieval_mode", "select"))

    if d.get("lattice_dir"):
        st.markdown("**Lattice dir**")
        st.code(d["lattice_dir"])

    if d.get("ingest_results"):
        st.markdown("**Ingest results by session**")
        ingest_df = pd.DataFrame(d["ingest_results"])
        st.dataframe(ingest_df, use_container_width=True)

# atoms table
selected_ids = set(d.get("atoms_selected", []))
atom_rows = []
for a in d.get("atoms", []):
    atom_id = a.get("atom_id")
    atom_rows.append(
        {
            "sel": "✓" if atom_id in selected_ids else "",
            "atom_id": atom_id,
            "subject": a.get("subject"),
            "kind": a.get("kind"),
            "source": a.get("source"),
            "valid_from": a.get("valid_from"),
            "valid_until": a.get("valid_until"),
            "observed_at": _source_time(a),
            "session_id": _source_session(a),
            "segment_id": (a.get("provenance") or {}).get("segment_id"),
            "content": a.get("content"),
        }
    )

with tab_atoms:
    st.markdown(f"**Atoms created ({d.get('atoms_created', 0)})** — selected highlighted in blue")
    if atom_rows:
        atoms_df = pd.DataFrame(atom_rows)

        def _highlight_selected(row: pd.Series) -> list[str]:
            if row["sel"] == "✓":
                return ["background-color: #ABD8F5"] * len(row)
            return [""] * len(row)

        st.dataframe(
            atoms_df.style.apply(_highlight_selected, axis=1),
            use_container_width=True,
            height=min(520, 38 + len(atom_rows) * 35),
        )

        atom_lookup = {a.get("atom_id"): a for a in d.get("atoms", [])}
        atom_choice = st.selectbox(
            "Inspect atom",
            [r["atom_id"] for r in atom_rows if r["atom_id"]],
            format_func=lambda aid: f"{aid} · {atom_lookup.get(aid, {}).get('subject', '')}",
        )
        atom = atom_lookup.get(atom_choice, {})
        st.json(atom)
        path = _atom_file(d.get("lattice_dir"), atom_choice)
        if path:
            st.markdown("**Atom file**")
            st.code(str(path))
            st.code(path.read_text()[:8000], language="markdown")
    else:
        st.info("No atoms were created.")

with tab_bm25:
    bm25_rows = d.get("bm25_candidates", [])
    st.markdown(f"**BM25 candidates ({len(bm25_rows)})** — pre-LLM selector pool")
    if bm25_rows:
        bm25_df = pd.DataFrame(
            [
                {
                    "sel": "✓" if a.get("atom_id") in selected_ids else "",
                    "atom_id": a.get("atom_id"),
                    "subject": a.get("subject"),
                    "kind": a.get("kind"),
                    "valid_from": a.get("valid_from"),
                    "valid_until": a.get("valid_until"),
                    "observed_at": _source_time(a),
                    "session_id": _source_session(a),
                    "content": a.get("content"),
                }
                for a in bm25_rows
            ]
        )
        st.dataframe(bm25_df, use_container_width=True, height=420)
    else:
        st.info("No BM25 candidates in debug file.")

with tab_selected:
    if d.get("selected_atoms"):
        selected_df = pd.DataFrame(d["selected_atoms"])
        st.dataframe(selected_df, use_container_width=True, height=320)
    else:
        st.info("No selected atom details in debug file.")

    if d.get("lattice_dir_kept") and d.get("lattice_dir") and selected_ids:
        st.markdown("**Selected atom files**")
        lattice_dir = Path(d["lattice_dir"])
        for atom_id in sorted(selected_ids):
            path = lattice_dir / f"{atom_id}.md"
            st.markdown(f"`{path}`")
            if path.exists():
                st.code(path.read_text()[:6000], language="markdown")

    raw = d.get("synthesis_raw", "")
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        st.markdown("**Synthesis raw**")
        st.json(parsed)

with tab_sessions:
    sessions = ds.get("haystack_sessions", [])
    session_ids = ds.get("haystack_session_ids", [f"s{i}" for i in range(len(sessions))])
    dates = ds.get("haystack_dates", ["" for _ in sessions])

    for sess, sid, ts in zip(sessions, session_ids, dates):
        with st.expander(f"Session {sid} · {ts}", expanded=False):
            for turn in sess:
                role = turn.get("role", "?")
                content = str(turn.get("content", ""))
                st.markdown(f"**{role}**")
                st.write(content)
                st.divider()

with tab_raw:
    st.markdown("**Debug JSON**")
    st.json(d)
    if show_raw:
        st.markdown("**Dataset JSON**")
        st.json(ds)
