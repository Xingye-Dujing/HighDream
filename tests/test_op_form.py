# -*- coding: utf-8 -*-
"""Tests for the LaTeX <-> operator bidirectional converter (routes/op_form)."""

from routes.compute_service import _dispatch_auto
from routes.op_form import latex_to_operator, operator_to_latex

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


def test_sqrt_not_mangled_in_latex():
    # bare sqrt(...) must render as \sqrt{...}, not the s*q*r*t parse_latex artifact
    tex = operator_to_latex('int(x^2/sqrt(x^2+1), x)')
    assert r'\sqrt' in tex and 's q r t' not in tex


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
