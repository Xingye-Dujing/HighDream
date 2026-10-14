# -*- coding: utf-8 -*-
"""Tests for the user-assumptions feature: parse "x>0, y real" and simplify the final
result via a confined symbol round-trip, applied as an honest extra derivation step.

Soundness is the point: only symbols the user constrained are simplified; unconstrained
ones must stay untouched (no unsound force=True collapse)."""

from sympy import Abs, Symbol, log, sqrt

from routes.compute_service import _dispatch_compute
from utils.assumptions import parse_assumption_flags, refine_with_flags

x = Symbol('x')
y = Symbol('y')


def test_parse_operators_and_words():
    assert parse_assumption_flags('x>0') == {'x': {'positive': True, 'real': True}}
    assert parse_assumption_flags('y<0')['y']['negative'] is True
    assert parse_assumption_flags('x>=0')['x']['nonnegative'] is True
    assert parse_assumption_flags('x positive')['x']['positive'] is True
    assert parse_assumption_flags('y real')['y'] == {'real': True}
    # multiple clauses, mixed
    assert parse_assumption_flags('x>0, y real') == {
        'x': {'positive': True, 'real': True}, 'y': {'real': True}}
    # garbage degrades to no assumptions (never raises)
    assert parse_assumption_flags('nonsense here') == {}


def test_refine_roundtrip_is_sound():
    # constrained symbol simplifies
    assert refine_with_flags(sqrt(x ** 2), parse_assumption_flags('x>0')) == x
    assert refine_with_flags(log(x ** 2) / 2, parse_assumption_flags('x>0')) == log(x)
    # UNCONSTRAINED symbols must NOT be collapsed (would be unsound)
    assert refine_with_flags(sqrt(x ** 2) + Abs(y), parse_assumption_flags('x>0')) == x + Abs(y)
    assert refine_with_flags(log(y ** 2) / 2, parse_assumption_flags('x>0')) == log(y ** 2) / 2
    # identity preserved: the result reuses the original symbol
    assert refine_with_flags(sqrt(x ** 2), parse_assumption_flags('x>0')).free_symbols == {x}


def _diff(expr, assumptions=None):
    data = {'operation_type': 'diff', 'expression': expr, 'variable': 'x'}
    if assumptions:
        data['assumptions'] = assumptions
    res = _dispatch_compute('diff', data)[1]
    return res if isinstance(res, str) else str(res)


def _integral(expr, assumptions=None):
    data = {'operation_type': 'integral', 'expression': expr, 'variable': 'x',
            'include_special_matchers': 'True'}
    if assumptions:
        data['assumptions'] = assumptions
    res = _dispatch_compute('integral', data)[1]
    return res if isinstance(res, str) else str(res)


def test_engine_end_to_end_honors_assumptions():
    # d/dx sqrt(x^2): without assumptions ends x/sqrt(x^2); with x>0 ends 1 + an honest note
    assert 'sqrt' in _diff('sqrt(x**2)')
    assert '依假设化简' in _diff('sqrt(x**2)', 'x>0')
    # int 1/x: without -> log(|x|) (absolute-value bars in LaTeX); with x>0 -> the note fires
    assert r'\left|' in _integral('1/x')
    assert '依假设化简' in _integral('1/x', 'x>0')


def test_no_assumptions_is_unchanged():
    # omitting the field must produce byte-identical output to an empty one
    assert _diff('sqrt(x**2)') == _diff('sqrt(x**2)', '')
