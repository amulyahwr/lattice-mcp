from __future__ import annotations

from lattice.llm import complete

_SYSTEM = """\
You are a knowledge synthesis agent. Given a set of knowledge atoms and a question, produce a concise answer.

Workflow:
1. Consider temporal ordering carefully:
   - `valid_from` is the date the atom was recorded, not necessarily when the event occurred.
     Use it to resolve relative time expressions in the atom content.
   - When content says "last Saturday", "two months ago", "yesterday", etc.,
     compute the actual event date by offsetting from that atom's valid_from.
   - For conflicting facts about the same subject, the atom with the later valid_from
     is more recent and takes precedence.
   - For event ordering questions, compare resolved event dates, not valid_from dates.
   - For duration questions ("how long had I been X when Y happened"), compute days between
     the two resolved event dates.
2. Base your answer strictly on the atoms — do not hallucinate or add outside knowledge.
3. If the answer is not present in the atoms, say so clearly.
4. Be concise: one to three paragraphs at most.
"""


def synthesize(query: str, atoms: list[dict]) -> str:
    if not atoms:
        return "No relevant information found in the lattice."

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
    return complete(messages)
