from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel

from lattice.llm import complete


class _Answer(BaseModel):
    thinking: str
    answer: str


@dataclass
class SynthesisResult:
    answer: str
    raw_response: str


_SYSTEM = """\
You are a knowledge synthesis agent. Given a set of knowledge atoms and a question, produce a concise answer.

Workflow:
1. In `thinking`: reason step by step through the atoms before writing your answer.
   - Identify which atoms are relevant to the question.
   - For temporal questions: resolve relative dates ("last Saturday", "two months ago", "yesterday")
     by offsetting from that atom's `valid_from` date. Compute actual event dates, then compare them.
   - For duration questions: compute the number of days between two resolved event dates.
   - For conflicting facts: the atom with the later `valid_from` takes precedence.
2. In `answer`: write a concise response based strictly on your reasoning above.
   - The atoms provided have already been filtered for relevance — trust them.
   - If atoms are present, always derive an answer from them. Do not say "no information found."
   - If the atoms only partially answer the question, give a best-effort answer and note the gap.
   - Only return "no information" if the atoms list is literally empty.
3. Be concise: one to three paragraphs at most.
"""


def synthesize(query: str, atoms: list[dict]) -> SynthesisResult:
    if not atoms:
        return SynthesisResult(
            answer="No relevant information found in the lattice.",
            raw_response="",
        )

    atoms_text = "\n\n".join(
        f"[{a['subject']} / {a['kind']} / valid_from={a.get('valid_from', 'null')}]\n{a['content']}"
        for a in atoms
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"Query: {query}\n\nKnowledge atoms:\n{atoms_text}",
        },
    ]
    raw = complete(messages, text_format=_Answer)
    try:
        answer = json.loads(raw)["answer"]
    except (json.JSONDecodeError, KeyError):
        answer = raw
    return SynthesisResult(answer=answer, raw_response=raw)
