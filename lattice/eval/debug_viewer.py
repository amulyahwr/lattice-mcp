"""
Lattice eval debug viewer — browse inference results by question.

Usage:
    uv run streamlit run lattice/eval/debug_viewer.py
"""

from __future__ import annotations

import json
from pathlib import Path

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
    with open(DATASET_PATH) as f:
        return {d["question_id"]: d for d in json.load(f)}


@st.cache_data
def load_priority(priority: str) -> dict:
    pdir = RESULTS_DIR / priority
    debug: dict[str, dict] = {}
    labels: dict[str, bool] = {}

    for path in pdir.glob("*.debug.jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    debug[d["question_id"]] = d
                except Exception:
                    pass

    for path in pdir.glob("*.eval-results-*"):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    labels[d["question_id"]] = d["autoeval_label"]["label"]
                except Exception:
                    pass

    return {"debug": debug, "labels": labels}


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


# ── sidebar ───────────────────────────────────────────────────────────────────

priorities = sorted(
    [
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and list(d.glob("*.debug.jsonl"))
    ],
    reverse=True,
)
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
        ["all", "0 atoms created", "0 atoms selected", "wrong synthesis"],
    )


# ── load ──────────────────────────────────────────────────────────────────────

dataset = load_dataset()
pdata = load_priority(priority)
debug, labels = pdata["debug"], pdata["labels"]


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
            "sel↑": len(d.get("atoms_selected", [])),
            "sessions": d.get("sessions_ingested", 0),
            "question": ds.get("question", "")[:70],
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


# ── summary ───────────────────────────────────────────────────────────────────

labelled = df[df["✓/✗"] != "—"]
correct_n = (df["✓/✗"] == "✓").sum()
wrong_n = (df["✓/✗"] == "✗").sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Shown", len(df))
c2.metric("Accuracy", f"{correct_n / len(labelled):.1%}" if len(labelled) else "—")
c3.metric("✓ Correct", int(correct_n))
c4.metric("✗ Wrong", int(wrong_n))
c5.metric("Avg atoms↑", f"{df['atoms↑'].mean():.1f}" if len(df) else "—")

# failure breakdown bar chart
wrong_df = df[df["✓/✗"] == "✗"]
if len(wrong_df):
    with st.expander("Failure breakdown (wrong answers only)"):
        breakdown = (
            wrong_df["failure"]
            .value_counts()
            .rename_axis("mode")
            .reset_index(name="count")
        )
        st.bar_chart(breakdown.set_index("mode"))


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


display_cols = ["type", "✓/✗", "failure", "atoms↑", "sel↑", "sessions", "question"]
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

# question + expected + got
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

# failure diagnosis banner
if failure == "0 atoms created":
    st.warning(
        f"⚠️ Ingest failure: 0 atoms created across {d.get('sessions_ingested', 0)} session(s)"
    )
elif failure == "0 atoms selected":
    st.warning(
        f"⚠️ Selection failure: {d.get('atoms_created', 0)} atoms created but 0 selected for retrieval"
    )
elif failure == "wrong synthesis":
    st.warning(
        f"⚠️ Synthesis failure: {len(d.get('atoms_selected', []))} atom(s) selected"
        f" but answer is wrong — check synthesis thinking below"
    )

# atoms table
st.markdown(
    f"**Atoms created ({d.get('atoms_created', 0)})**  —  selected highlighted in blue"
)
selected_ids = set(d.get("atoms_selected", []))
atom_rows = [
    {
        "sel": "✓" if a["atom_id"] in selected_ids else "",
        "subject": a["subject"],
        "content": a["content"],
    }
    for a in d.get("atoms", [])
]

if atom_rows:
    atoms_df = pd.DataFrame(atom_rows)

    def _highlight_selected(row: pd.Series) -> list[str]:
        if row["sel"] == "✓":
            return ["background-color: #ABD8F5"] * len(row)
        return [""] * len(row)

    st.dataframe(
        atoms_df.style.apply(_highlight_selected, axis=1),
        use_container_width=True,
        height=min(420, 38 + len(atom_rows) * 35),
    )
else:
    st.info("No atoms were created.")

# synthesis thinking
raw = d.get("synthesis_raw", "")
if raw:
    try:
        thinking = json.loads(raw).get("thinking", "")
    except Exception:
        thinking = ""
    if thinking:
        with st.expander("Synthesis thinking (CoT)"):
            st.markdown(thinking)

# sessions
sessions = ds.get("haystack_sessions", [])
session_ids = ds.get("haystack_session_ids", [f"s{i}" for i in range(len(sessions))])
dates = ds.get("haystack_dates", ["" for _ in sessions])

with st.expander(f"Sessions ingested ({len(sessions)})"):
    for sess, sid, ts in zip(sessions, session_ids, dates):
        st.markdown(f"**Session `{sid}`** — `{ts}`")
        for turn in sess[:6]:
            role = turn.get("role", "?")
            content = str(turn.get("content", ""))[:300]
            prefix = "👤" if role == "user" else "🤖"
            st.markdown(f"{prefix} **{role}**: {content}")
        if len(sess) > 6:
            st.caption(f"... {len(sess) - 6} more turns")
        st.divider()
