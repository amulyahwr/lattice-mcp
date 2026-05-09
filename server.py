from __future__ import annotations

from datetime import date
from typing import Any

import mcp.server.stdio
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool

from lattice.db import LatticeDB
from lattice.ingest import ingest
from lattice.selection import select
from lattice.synthesis import synthesize

app = Server("lattice-mcp")
_db = LatticeDB()
_db.preload()


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="lattice_ingest",
            description=(
                "Decompose raw text into discrete knowledge atoms and store them in the lattice. "
                "Returns the number of atoms created and their IDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Raw text content to ingest.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional passthrough metadata (title, url, author, date, etc.).",
                        "additionalProperties": True,
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="lattice_select",
            description=(
                "Select the most relevant knowledge atoms for a natural language query. "
                "Returns a ranked list of atoms."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or topic.",
                    },
                    "as_of": {
                        "type": "string",
                        "description": "Optional ISO date (YYYY-MM-DD). Filters atoms valid at that date.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="lattice_answer",
            description=(
                "Answer a natural language query using the lattice. "
                "Optionally restrict to specific atom IDs; otherwise auto-selects relevant atoms first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question.",
                    },
                    "atom_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of atom IDs to use. If empty, auto-selects.",
                    },
                    "as_of": {
                        "type": "string",
                        "description": "Optional ISO date passed to selection when atom_ids not provided.",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "lattice_ingest":
        result = ingest(
            source=arguments["source"],
            metadata=arguments.get("metadata", {}),
            db=_db,
        )
        return [TextContent(type="text", text=str(result))]

    if name == "lattice_select":
        as_of_str: str | None = arguments.get("as_of")
        as_of = date.fromisoformat(as_of_str) if as_of_str else None
        atoms = select(query=arguments["query"], as_of=as_of, db=_db)
        import json
        return [TextContent(type="text", text=json.dumps(atoms, indent=2))]

    if name == "lattice_answer":
        import json
        as_of_str = arguments.get("as_of")
        as_of = date.fromisoformat(as_of_str) if as_of_str else None
        atom_ids: list[str] = arguments.get("atom_ids", [])

        if atom_ids:
            atoms = []
            for aid in atom_ids:
                try:
                    a = _db.read(aid)
                    atoms.append({
                        "atom_id": a.atom_id,
                        "subject": a.subject,
                        "content": a.content,
                        "kind": a.kind,
                        "source": a.source,
                        "valid_from": a.valid_from.isoformat() if a.valid_from else None,
                        "valid_until": a.valid_until.isoformat() if a.valid_until else None,
                    })
                except Exception:
                    pass
        else:
            atoms = select(query=arguments["query"], as_of=as_of, db=_db)

        result = synthesize(query=arguments["query"], atoms=atoms)
        return [TextContent(type="text", text=result.answer)]

    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    import asyncio

    async def _run() -> None:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="lattice-mcp",
                    server_version="0.1.0",
                    capabilities=app.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={},
                    ),
                ),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
