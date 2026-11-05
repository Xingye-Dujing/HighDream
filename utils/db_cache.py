"""SQLite-backed result cache with graceful degradation.

Every read/write is best-effort: a storage failure (e.g. Vercel's read-only
filesystem outside ``/tmp``) must never turn a successful computation into a
500.  Set ``HIGHDREAM_CACHE_DIR`` to relocate the database on such platforms.
"""
import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_DIR = Path(os.environ.get(
    'HIGHDREAM_CACHE_DIR',
    str(Path(__file__).resolve().parent.parent / 'cache')))
DB_PATH = DB_DIR / 'computation.db'

# Rows kept in the result cache (oldest ``last_accessed`` are pruned on insert).
CACHE_MAX_ROWS = 5000

_local = threading.local()


def _now() -> str:
    """Microsecond timestamp; second precision made LRU ordering arbitrary."""
    return datetime.now().isoformat(sep=' ', timespec='microseconds')


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
        conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute('PRAGMA journal_mode=WAL')
        _init_db(conn)
        _local.conn = conn
    return _local.conn


# Key schema version. Bumping this prefix invalidates every pre-existing row so
# an old key (which ignored bounds / assumptions / order / unknown params) can
# never be served as a hit for the richer v2 key.
_HASH_VERSION = 'v2'

# Canonical key name -> empty default, for every parameter known to influence a
# derivation. The key ALWAYS carries this full structure, so omitting a field
# still yields a self-consistent key. Any caller param NOT listed here is folded
# in under ``extra`` (sorted by key) instead of being silently dropped, so a
# newly introduced result-affecting field can never produce a stale cache hit.
_KNOWN_FIELDS = {
    'var': '', 'point': '', 'direction': '', 'options': {},
    'lower': '', 'upper': '', 'bounds': '', 'assumptions': '', 'order': '',
}
# Caller-facing param name -> canonical key name (``variable`` is stored as
# ``var`` to stay aligned with the DB columns).
_PARAM_ALIASES = {
    'variable': 'var',
    'point': 'point',
    'direction': 'direction',
    'options': 'options',
    'lower': 'lower',
    'upper': 'upper',
    'bounds': 'bounds',
    'assumptions': 'assumptions',
    'order': 'order',
}


def _hash_value(value):
    """Coerce a param value into a deterministic, JSON-serializable structure.

    SymPy objects (bounds, symbols, ...) are not JSON-serializable, so anything
    outside the plain scalar/container types is reduced to ``str``; dict keys are
    sorted so the result never depends on insertion order.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _hash_value(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_hash_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    return str(value)


def build_hash(operation_type: str, expression: str, **params) -> str:
    payload = {'op': operation_type, 'expr': expression.strip()}
    for name, default in _KNOWN_FIELDS.items():
        payload[name] = dict(default) if isinstance(default, dict) else default
    extras = {}
    for key in sorted(params):
        value = _hash_value(params[key])
        canonical = _PARAM_ALIASES.get(key)
        if canonical is not None:
            payload[canonical] = value
        else:
            extras[key] = value
    if extras:
        payload['extra'] = extras
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = f'{_HASH_VERSION}|{normalized}'.encode('utf-8')
    return hashlib.sha256(digest).hexdigest()


def get(hash_key: str) -> Optional[str]:
    """Return a cached result, or ``None`` on a miss or any storage failure."""
    try:
        conn = _get_conn()
        # Two statements instead of UPDATE ... RETURNING: that syntax needs
        # SQLite >= 3.35 and made every cache hit a write transaction.
        row = conn.execute(
            'SELECT result FROM computation_cache WHERE hash_key = ?',
            (hash_key,)).fetchone()
        if row is None:
            return None
        conn.execute(
            'UPDATE computation_cache SET accessed_count = accessed_count + 1, '
            'last_accessed = ? WHERE hash_key = ?',
            (_now(), hash_key))
        conn.commit()
        return row[0]
    except (sqlite3.Error, OSError):
        return None


def set_cache(hash_key: str, operation_type: str, expression: str,
              result: str, **params) -> None:
    """Store a result; silently degrade to a no-op when storage is unavailable."""
    variable = str(params.get('variable', ''))
    point = str(params.get('point', ''))
    direction = str(params.get('direction', ''))
    try:
        conn = _get_conn()
        # UPDATE-then-INSERT (instead of INSERT OR REPLACE) keeps the row's
        # accessed_count / created_at alive, so the LRU order stays meaningful.
        cursor = conn.execute(
            'UPDATE computation_cache SET operation_type = ?, expression = ?, '
            'variable = ?, point = ?, direction = ?, result = ?, '
            'last_accessed = ? WHERE hash_key = ?',
            (operation_type, expression, variable, point, direction,
             result, _now(), hash_key))
        if cursor.rowcount == 0:
            conn.execute(
                'INSERT INTO computation_cache '
                '(hash_key, operation_type, expression, variable, point, '
                'direction, result, created_at, accessed_count, last_accessed) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0, ?)',
                (hash_key, operation_type, expression, variable, point,
                 direction, result, _now()))
        # Bound the table so long-lived installs do not grow it forever.
        count = conn.execute('SELECT COUNT(*) FROM computation_cache').fetchone()[0]
        if count > CACHE_MAX_ROWS:
            conn.execute(
                'DELETE FROM computation_cache WHERE hash_key IN ('
                'SELECT hash_key FROM computation_cache '
                'ORDER BY last_accessed ASC, hash_key ASC LIMIT ?)',
                (count - CACHE_MAX_ROWS,),
            )
        conn.commit()
    except (sqlite3.Error, OSError):
        # Best-effort cache: the computation itself already succeeded.
        pass
