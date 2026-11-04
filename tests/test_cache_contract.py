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


def test_cache_hash_includes_calculator_options(temp_cache_db):
    plain = db_cache.build_hash('integral', 'x*exp(x)', variable='x')
    optioned = db_cache.build_hash(
        'integral', 'x*exp(x)', variable='x',
        options={'include_special_matchers': False})
    assert plain != optioned


def test_cache_separates_integral_matcher_options(temp_cache_db):
    # The two matcher sets produce different derivations for the same integrand.
    base = {'operation_type': 'integral', 'expression': 'x*exp(x)', 'variable': 'x'}
    ok, off = start_compute('integral', {
        **base, 'include_special_matchers': 'False', 'record_cache': 'True'})
    assert ok
    ok, on = start_compute('integral', {
        **base, 'include_special_matchers': 'True', 'use_cache': 'True'})
    assert ok
    # A cache key that ignored the option would have replayed the "off" derivation.
    assert on != off


def test_cache_flags_accept_real_booleans(temp_cache_db):
    # API clients may send JSON booleans; the old ``== 'True'`` comparison
    # silently ignored them (no caching, no lookup).
    ok, first = start_compute('diff', {
        'operation_type': 'diff', 'expression': 'x^3', 'variable': 'x',
        'record_cache': True,
    })
    assert ok
    ok, second = start_compute('diff', {
        'operation_type': 'diff', 'expression': 'x^3', 'variable': 'x',
        'use_cache': True,
    })
    assert ok and first == second
    conn = db_cache._get_conn()
    assert conn.execute('SELECT COUNT(*) FROM computation_cache').fetchone()[0] == 1


def test_unresolved_derivation_is_not_cached(temp_cache_db):
    # d|x|/dx is outside the rule engine: the run aborts. Recording it must be a
    # no-op so a later cache lookup cannot replay a dead end as the answer.
    ok, _ = start_compute('diff', {
        'operation_type': 'diff', 'expression': 'abs(x)', 'variable': 'x',
        'record_cache': 'True',
    })
    assert ok
    conn = db_cache._get_conn()
    assert conn.execute('SELECT COUNT(*) FROM computation_cache').fetchone()[0] == 0
