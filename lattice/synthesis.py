from __future__ import annotations

from lattice.llm import complete

_SYSTEM = """\
You are a precise, concise assistant. Answer the user's query using only the knowledge atoms provided.

Rules:
- Base your answer strictly on the atoms given. Do not hallucinate facts.
- If the atoms do not contain enough information to answer, say so clearly.
- Write in clear, flowing prose. Do not reference atom IDs or internal structure.
- Be concise: one to three paragraphs at most.
"""


def synthesize(query: str, atoms: list[dict]) -> str:
    if not atoms:
        return "No relevant information found in the lattice."

    atoms_text = "\n\n".join(
        f"[{a['subject']} / {a['kind']}]\n{a['content']}" for a in atoms
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"Query: {query}\n\nKnowledge atoms:\n{atoms_text}",
        },
    ]
    return complete(messages)
