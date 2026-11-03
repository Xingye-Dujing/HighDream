# pylint: disable=redefined-outer-name,unused-argument
"""Computation cache store and cached-calculator contract."""

import pytest

from routes.compute_service import start_compute
from utils import db_cache


@pytest.fixture
def temp_cache_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db_cache, 'DB_DIR', tmp_path)
    monkeypatch.setattr(db_cache, 'DB_PATH', tmp_path / 'computation.db')
    db_cache._local.conn = None
    yield
    conn = getattr(db_cache._local, 'conn', None)
    if conn is not None:
        conn.close()
    db_cache._local.conn = None


def test_cache_store_roundtrip(temp_cache_db):
    key = db_cache.build_hash('diff', ' x^2 ', variable='x')
    assert key == db_cache.build_hash('diff', 'x^2', variable='x')
    assert db_cache.get(key) is None
    db_cache.set_cache(key, 'diff', 'x^2', r'\begin{align}2x\end{align}', variable='x')
    assert db_cache.get(key) == r'\begin{align}2x\end{align}'
    assert db_cache.get(key) == r'\begin{align}2x\end{align}'


def test_cached_diff_calculator_returns_same_result(temp_cache_db):
    first_ok, first = start_compute('diff', {
        'operation_type': 'diff', 'expression': 'x^2', 'variable': 'x',
        'record_cache': 'True',
    })
    assert first_ok
    second_ok, second = start_compute('diff', {
        'operation_type': 'diff', 'expression': 'x^2', 'variable': 'x',
        'use_cache': 'True',
    })
    assert second_ok
    assert first == second
