# -*- coding: utf-8 -*-
"""Tests for the LaTeX <-> operator bidirectional converter (routes/op_form)."""

from routes.compute_service import _dispatch_auto
from routes.op_form import latex_to_operator, operator_to_latex, toggle_form

CALCULUS = [
    'int(1/(x**2+1), x)',
    'int(x^2/sqrt(x^2+1), x)',
    'diff(x*sin(x), x)',
    'limit((1-1/x)^x, x, 0)',
    'limit(sin(x)/x, x, 0, +)',
]
MATRIX = [
    'det([[a,b],[c,d]])',
    'rank([[1,2],[4,2]])',
    'inverse([[a,b],[c,d]])',
    'transpose([[1,2,3],[4,a,1]])',
    'add([[1,2,3],[4,2,1]],[[0,4,5],[6,4,a]])',
    'subtract([[1,2,3],[4,2,1]],[[0,4,5],[6,4,a]])',
    'svd([[1,2],[4,2]])',
]


def _ok(expr):
    r = _dispatch_auto(expr)
    return isinstance(r, tuple) and r[0]


def test_operator_to_latex_renders_and_computes():
    for op in CALCULUS + MATRIX:
        tex = operator_to_latex(op)
        assert '\\int' in tex or '\\frac' in tex or '\\lim' in tex \
            or '\\begin{bmatrix}' in tex or '\\det' in tex or '\\mathrm' in tex, tex
        assert _ok(tex), 'latex form must dispatch: ' + tex


def test_bare_functions_round_trip_not_mangled():
    # The shared parser must read bare exp/sqrt/asin (with carets) as functions,
    # not the s*q*r*t product that parse_latex would otherwise produce.
    from utils.cache_utils import cached_sympify
    assert 'sqrt' in repr(cached_sympify('x^2/sqrt(x^2+1)')).lower()
    assert 'exp' in repr(cached_sympify('x*exp(x^2)')).lower()
    assert 'asin' in repr(cached_sympify('asin(x)^2')).lower()
    # and the wrapped LaTeX form computes
    tex = operator_to_latex('int(x^2/sqrt(x^2+1), x)')
    assert tex.startswith(r'\int') and _ok(tex)


def test_round_trip_latex_operator():
    for op in CALCULUS + MATRIX:
        rt = latex_to_operator(operator_to_latex(op))
        assert _ok(rt), 'round-tripped operator must dispatch: ' + rt


def test_limit_carries_one_sided_direction_both_ways():
    tex = operator_to_latex('limit(sin(x)/x, x, 0, +)')
    assert '0^{+}' in tex
    assert latex_to_operator(tex).endswith(',x,0,+)')


def test_unknown_expression_is_noop():
    assert operator_to_latex('sin(x)+1') == 'sin(x)+1'
    assert latex_to_operator('') == ''


def test_toggle_form_flips_both_ways():
    # operator -> latex, then latex -> operator
    tex = toggle_form('int(1/(x**2+1), x)')
    assert tex.startswith(r'\int')
    assert toggle_form(tex) == 'int(\frac{1}{x^{2} + 1},x)' or _ok(toggle_form(tex))
    # matrix operator <-> latex
    mtex = toggle_form('det([[a,b],[c,d]])')
    assert r'\det' in mtex
    assert _ok(toggle_form(mtex))  # back to operator still dispatches
