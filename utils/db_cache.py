import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

DB_DIR = Path(__file__).resolve().parent.parent / 'cache'
DB_PATH = DB_DIR / 'computation.db'

_local = threading.local()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS computation_cache (
            hash_key TEXT PRIMARY KEY,
            operation_type TEXT NOT NULL,
            expression TEXT NOT NULL,
            variable TEXT DEFAULT '',
            point TEXT DEFAULT '',
            direction TEXT DEFAULT '',
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def build_hash(operation_type: str, expression: str, **params) -> str:
    normalized = json.dumps({
        'op': operation_type,
        'expr': expression.strip(),
        'var': str(params.get('variable', '')),
        'point': str(params.get('point', '')),
        'direction': str(params.get('direction', '')),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def get(hash_key: str) -> Optional[str]:
    conn = _get_conn()
    cursor = conn.execute(
        'UPDATE computation_cache SET accessed_count = accessed_count + 1, '
        'last_accessed = CURRENT_TIMESTAMP WHERE hash_key = ? RETURNING result',
        (hash_key,)
    )
    row = cursor.fetchone()
    conn.commit()
    return row[0] if row else None


def set_cache(hash_key: str, operation_type: str, expression: str,
              result: str, **params) -> None:
    conn = _get_conn()
    conn.execute('''
        INSERT OR REPLACE INTO computation_cache
            (hash_key, operation_type, expression, variable, point, direction, result)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        hash_key, operation_type, expression,
        str(params.get('variable', '')),
        str(params.get('point', '')),
        str(params.get('direction', '')),
        result,
    ))
    conn.commit()
