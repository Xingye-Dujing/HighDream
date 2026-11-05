"""SQLite-backed history snapshots for the notebook.

The notebook used to keep a single auto-saved snapshot in ``localStorage``.
That model is replaced by an explicit history: every snapshot is written to a
small SQLite database next to the computation cache, and the user picks the one
to restore.  Nothing is saved behind the user's back.

A snapshot row stores the same JSON array produced by the page's
``getNotebookData()`` so restoring is a straight ``loadNotebookData(parsed)``.
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Same override as the computation cache: read-only deployments (Vercel) must
# point this at a writable directory (e.g. /tmp) or snapshot writes would fail.
DB_DIR = Path(os.environ.get(
    'HIGHDREAM_CACHE_DIR',
    str(Path(__file__).resolve().parent.parent / 'cache')))
DB_PATH = DB_DIR / 'snapshots.db'

# Older snapshots are pruned so the database cannot grow without bound.
MAX_SNAPSHOTS = 200
# A single notebook payload is capped: without this one request could store an
# arbitrarily large blob in the local database.
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024

_local = threading.local()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notebook_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT '',
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, 'conn') or _local.conn is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(DB_PATH))
        _local.conn.execute('PRAGMA journal_mode=WAL')
        _init_db(_local.conn)
    return _local.conn


def create_snapshot(payload: str, title: str = '') -> dict:
    """Insert one snapshot and return its metadata (without the payload).

    Raises ``ValueError`` when the payload exceeds ``MAX_SNAPSHOT_BYTES``.
    """
    if len(payload.encode('utf-8')) > MAX_SNAPSHOT_BYTES:
        raise ValueError('笔记本过大, 单个快照不能超过 2 MB')
    title = (title or '').strip()[:200]
    conn = _get_conn()
    created_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    cursor = conn.execute(
        'INSERT INTO notebook_snapshots (title, payload, created_at) VALUES (?, ?, ?)',
        (title or '', payload, created_at),
    )
    conn.commit()
    snapshot_id = cursor.lastrowid
    _prune(conn)
    return {'id': snapshot_id, 'title': title or '', 'created_at': created_at}


def _prune(conn: sqlite3.Connection) -> None:
    conn.execute(
        'DELETE FROM notebook_snapshots WHERE id NOT IN ('
        'SELECT id FROM notebook_snapshots ORDER BY id DESC LIMIT ?'
        ')',
        (MAX_SNAPSHOTS,),
    )
    conn.commit()


def list_snapshots(limit: int = 50) -> list:
    """Return recent snapshots, newest first, without their payloads."""
    conn = _get_conn()
    rows = conn.execute(
        'SELECT id, title, created_at, LENGTH(payload) FROM notebook_snapshots '
        'ORDER BY id DESC LIMIT ?',
        (max(1, int(limit)),),
    ).fetchall()
    return [
        {'id': row[0], 'title': row[1], 'created_at': row[2], 'size': row[3]}
        for row in rows
    ]


def get_snapshot(snapshot_id: int) -> Optional[dict]:
    """Return one full snapshot (including payload) or ``None``."""
    conn = _get_conn()
    row = conn.execute(
        'SELECT id, title, created_at, payload FROM notebook_snapshots WHERE id = ?',
        (int(snapshot_id),),
    ).fetchone()
    if not row:
        return None
    return {'id': row[0], 'title': row[1], 'created_at': row[2], 'payload': row[3]}


def delete_snapshot(snapshot_id: int) -> bool:
    """Delete one snapshot. Returns whether a row was removed."""
    conn = _get_conn()
    cursor = conn.execute(
        'DELETE FROM notebook_snapshots WHERE id = ?', (int(snapshot_id),)
    )
    conn.commit()
    return cursor.rowcount > 0
