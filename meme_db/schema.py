"""SQLite meme metadata schema (Milestone 3).

One row per meme. The integer `id` doubles as the meme's row in the embedding matrix /
FAISS index — the index and DB are always (re)built together by `build_index.py`, so ids
are dense `0..N-1` and align 1:1 with vector rows. Keeping them aligned is what lets
`retrieve.py` go straight from a FAISS hit (a row number) to a metadata row.

Metadata is the real moat (CLAUDE.md): the `tags` + `intensity` feed the M4 ranking
formula (`tag_match`, `intensity_match`), and the safety flags
(`copyrighted_character` / `private_person`) become the M4 hard-veto.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

# DB lives under the gitignored data/labels/ tree.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "labels" / "memes.db"


class Meme(BaseModel):
    """A single meme's metadata. `id` is the embedding row index (set at build time)."""

    id: int
    source: str = ""            # e.g. "not-lain/meme-dataset"
    source_id: str = ""         # original row id / key within the source
    image_path: str = ""        # local path under data/memes/
    text: str = ""              # caption / title / weak label text from the dataset
    tags: list[str] = Field(default_factory=list)   # auto-labeled reaction tags
    intensity: float = 0.0      # 0..1 expressive-intensity proxy
    copyrighted_character: bool = False
    private_person: bool = False


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating parent dirs) a SQLite connection with row access by name."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection, *, reset: bool = False) -> None:
    """Create the `memes` table. `reset=True` drops it first (fresh rebuilds)."""
    if reset:
        conn.execute("DROP TABLE IF EXISTS memes")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memes (
            id                    INTEGER PRIMARY KEY,
            source                TEXT NOT NULL DEFAULT '',
            source_id             TEXT NOT NULL DEFAULT '',
            image_path            TEXT NOT NULL DEFAULT '',
            text                  TEXT NOT NULL DEFAULT '',
            tags                  TEXT NOT NULL DEFAULT '[]',
            intensity             REAL NOT NULL DEFAULT 0.0,
            copyrighted_character INTEGER NOT NULL DEFAULT 0,
            private_person        INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()


def _row_to_meme(row: sqlite3.Row) -> Meme:
    return Meme(
        id=row["id"],
        source=row["source"],
        source_id=row["source_id"],
        image_path=row["image_path"],
        text=row["text"],
        tags=json.loads(row["tags"]) if row["tags"] else [],
        intensity=row["intensity"],
        copyrighted_character=bool(row["copyrighted_character"]),
        private_person=bool(row["private_person"]),
    )


def insert_memes(conn: sqlite3.Connection, memes: Iterable[Meme]) -> int:
    """Insert/replace meme rows. Returns the number written."""
    rows = [
        (
            m.id, m.source, m.source_id, m.image_path, m.text,
            json.dumps(m.tags), float(m.intensity),
            int(m.copyrighted_character), int(m.private_person),
        )
        for m in memes
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO memes
            (id, source, source_id, image_path, text, tags, intensity,
             copyrighted_character, private_person)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def fetch_all(conn: sqlite3.Connection) -> list[Meme]:
    cur = conn.execute("SELECT * FROM memes ORDER BY id")
    return [_row_to_meme(r) for r in cur.fetchall()]


def fetch_by_ids(conn: sqlite3.Connection, ids: Iterable[int]) -> dict[int, Meme]:
    """Fetch memes by id, returned as an {id: Meme} map (order-independent)."""
    ids = list(ids)
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    cur = conn.execute(f"SELECT * FROM memes WHERE id IN ({placeholders})", ids)
    return {r["id"]: _row_to_meme(r) for r in cur.fetchall()}


def count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM memes").fetchone()[0])
