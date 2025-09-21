"""LLM-driven interface to a structured memory SQLite database.

This module demonstrates how to:
* create the memory schema described in the prompt,
* populate it with a couple of illustrative entries, and
* ask an LLM (via Ollama) to emit a structured query that is executed against the DB.

The LLM call uses JSON schema guidance so the model returns a response that the
application can validate with Pydantic before it touches the database.  The
pattern mirrors the ``FriendList`` example provided in the prompt.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Iterable

from ollama import AsyncClient
from pydantic import BaseModel, Field

DB_PATH = Path("memory.db")


# ---------------------------------------------------------------------------
# SQLite schema management
# ---------------------------------------------------------------------------

def initialize_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create the database (if needed) and seed example data.

    The schema matches the tables described in the prompt.  For the purposes of
    the demo we keep the dataset intentionally small but realistic enough for
    simple experiments.
    """

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_nodes (
            id TEXT PRIMARY KEY,
            type TEXT,
            title TEXT,
            summary TEXT,
            source TEXT,
            confidence REAL,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS node_links (
            from_id TEXT,
            to_id TEXT,
            type TEXT,
            weight REAL,
            FOREIGN KEY(from_id) REFERENCES memory_nodes(id),
            FOREIGN KEY(to_id) REFERENCES memory_nodes(id)
        );

        CREATE TABLE IF NOT EXISTS node_tags (
            node_id TEXT,
            tag TEXT,
            FOREIGN KEY(node_id) REFERENCES memory_nodes(id)
        );

        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT,
            embedding BLOB,
            FOREIGN KEY(node_id) REFERENCES memory_nodes(id)
        );

        CREATE TABLE IF NOT EXISTS cartridges (
            id TEXT PRIMARY KEY,
            name TEXT,
            version TEXT,
            author TEXT,
            description TEXT,
            license TEXT,
            created_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS cartridge_nodes (
            cartridge_id TEXT,
            node_id TEXT,
            FOREIGN KEY(cartridge_id) REFERENCES cartridges(id),
            FOREIGN KEY(node_id) REFERENCES memory_nodes(id)
        );

        CREATE TABLE IF NOT EXISTS node_logic (
            node_id TEXT,
            predicate TEXT,
            target TEXT,
            expression TEXT,
            FOREIGN KEY(node_id) REFERENCES memory_nodes(id)
        );

        CREATE TABLE IF NOT EXISTS node_content (
            node_id TEXT PRIMARY KEY,
            definition TEXT,
            process TEXT,
            formulas TEXT,
            constraints TEXT,
            FOREIGN KEY(node_id) REFERENCES memory_nodes(id)
        );

        CREATE TABLE IF NOT EXISTS examples (
            id TEXT PRIMARY KEY,
            node_id TEXT,
            scenario TEXT,
            tags TEXT,
            FOREIGN KEY(node_id) REFERENCES memory_nodes(id)
        );
        """
    )

    # Seed data only if the database is empty.
    cur.execute("SELECT COUNT(*) FROM memory_nodes")
    count = cur.fetchone()[0]
    if count:
        return conn

    cur.executemany(
        """
        INSERT INTO memory_nodes (id, type, title, summary, source, confidence, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        [
            (
                "concept-energy",
                "concept",
                "Energy",
                "Capacity to do work; conserved quantity in physics.",
                "Physics 101",
                0.9,
            ),
            (
                "fact-kinetic",
                "fact",
                "Kinetic Energy Formula",
                "Kinetic energy equals one half mass times velocity squared.",
                "Physics textbook",
                0.95,
            ),
            (
                "example-rollercoaster",
                "example",
                "Roller Coaster Energy Exchange",
                "Potential energy converts to kinetic energy on the descent.",
                "Amusement park case study",
                0.85,
            ),
        ],
    )

    cur.executemany(
        "INSERT INTO node_tags (node_id, tag) VALUES (?, ?)",
        [
            ("concept-energy", "physics"),
            ("fact-kinetic", "physics"),
            ("fact-kinetic", "formula"),
            ("example-rollercoaster", "physics"),
            ("example-rollercoaster", "real-world"),
        ],
    )

    cur.executemany(
        "INSERT INTO node_links (from_id, to_id, type, weight) VALUES (?, ?, ?, ?)",
        [
            ("concept-energy", "fact-kinetic", "child", 0.8),
            ("fact-kinetic", "example-rollercoaster", "example", 0.6),
        ],
    )

    cur.executemany(
        "INSERT INTO node_content (node_id, definition, process, formulas, constraints) VALUES (?, ?, ?, ?, ?)",
        [
            (
                "fact-kinetic",
                "The energy possessed by an object due to its motion.",
                "[\"Determine mass\", \"Measure velocity\", \"Apply formula\"]",
                "[\"KE = 0.5 * m * v^2\"]",
                "[]",
            ),
        ],
    )

    cur.executemany(
        "INSERT INTO examples (id, node_id, scenario, tags) VALUES (?, ?, ?, ?)",
        [
            (
                "ex-rollercoaster",
                "example-rollercoaster",
                "A roller coaster converts gravitational potential energy to kinetic energy as it descends the track.",
                "[\"physics\", \"mechanics\"]",
            ),
        ],
    )

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Structured query models
# ---------------------------------------------------------------------------

class NodeQuery(BaseModel):
    """Structured request produced by the LLM."""

    node_type: str | None = Field(
        default=None, description="Filter by the node's type, e.g. concept/fact/example."
    )
    tags: list[str] = Field(default_factory=list, description="Restrict results to nodes with *all* tags.")
    related_to: str | None = Field(
        default=None,
        description="Return nodes that are directly linked from the supplied node id.",
    )
    limit: int = Field(default=5, ge=1, le=25)


class NodeResult(BaseModel):
    id: str
    type: str
    title: str
    summary: str | None = None
    confidence: float | None = None
    tags: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    nodes: list[NodeResult]


# ---------------------------------------------------------------------------
# Query execution helpers
# ---------------------------------------------------------------------------

def execute_query(conn: sqlite3.Connection, query: NodeQuery) -> QueryResponse:
    """Translate the structured query into SQL and return the result set."""

    sql = [
        "SELECT n.id, n.type, n.title, n.summary, n.confidence FROM memory_nodes AS n",
    ]
    params: list[str] = []

    if query.tags:
        tag_placeholders = ",".join(["?"] * len(query.tags))
        sql.append(
            "JOIN node_tags AS t ON t.node_id = n.id"
            f" WHERE t.tag IN ({tag_placeholders})"
        )
        params.extend(query.tags)
    else:
        sql.append("WHERE 1=1")

    if query.node_type:
        sql.append("AND n.type = ?")
        params.append(query.node_type)

    if query.related_to:
        sql.append(
            "AND EXISTS (SELECT 1 FROM node_links AS l WHERE l.from_id = ? AND l.to_id = n.id)"
        )
        params.append(query.related_to)

    sql.append("GROUP BY n.id HAVING COUNT(DISTINCT t.tag) >= ?")
    sql.append("ORDER BY n.confidence DESC NULLS LAST")
    sql.append("LIMIT ?")

    required_tag_count = len(query.tags) if query.tags else 0
    params.extend([required_tag_count, query.limit])

    rows = conn.execute("\n".join(sql), params).fetchall()

    results = [
        NodeResult(
            id=row["id"],
            type=row["type"],
            title=row["title"],
            summary=row["summary"],
            confidence=row["confidence"],
            tags=list(fetch_tags(conn, row["id"])),
        )
        for row in rows
    ]

    return QueryResponse(nodes=results)


def fetch_tags(conn: sqlite3.Connection, node_id: str) -> Iterable[str]:
    cur = conn.execute("SELECT tag FROM node_tags WHERE node_id = ? ORDER BY tag", (node_id,))
    for row in cur:
        yield row[0]


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------

async def plan_query_with_llm(user_request: str) -> NodeQuery:
    """Ask the LLM to convert a natural-language request into ``NodeQuery``."""

    client = AsyncClient()
    messages = [
        {
            "role": "system",
            "content": "You translate natural language requests into JSON queries for the memory database.",
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    response = await client.chat(
        model="llama3.3:70b",
        messages=messages,
        format=NodeQuery.model_json_schema(),
        options={"temperature": 0.1},
    )

    return NodeQuery.model_validate_json(response.message.content)


async def main() -> None:
    conn = initialize_db()

    # Let the model decide which structured query to run.
    natural_language_request = (
        "Find high-confidence physics facts related to energy that have worked examples."
    )

    query = await plan_query_with_llm(natural_language_request)
    print("\n🤖 Structured query from LLM:\n", query)

    response = execute_query(conn, query)
    print("\n✅ Query results:\n", response)


if __name__ == "__main__":
    asyncio.run(main())
