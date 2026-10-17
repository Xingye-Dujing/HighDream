"""Tests that WYSIWYG (MathLive) LaTeX is correctly converted to SymPy."""

import pytest
from sympy import Derivative, Integral, Symbol, cos, exp, log, sin, simplify, sqrt

from utils.cache_utils import cached_sympify

x = Symbol('x')


def test_exp_from_caret_brace():
    assert cached_sympify(r'e^{x}') == exp(x)
    assert cached_sympify(r'e^{2x}') == exp(2 * x)


def test_power_brace():
    assert cached_sympify(r'x^{2}') == x ** 2


def test_ln_is_natural_log():
    assert simplify(cached_sympify(r'\ln(x)') - log(x)) == 0


def test_sqrt_and_fraction():
    assert cached_sympify(r'\sqrt{x}') == sqrt(x)
    assert cached_sympify(r'\frac{1}{x}') == 1 / x


def test_text_prose_stripped():
    assert cached_sympify(r'\text{abc}+x') == x


def test_sympy_subscript_name_not_broken():
    # a bare a_1 (no brace) stays a symbol, not routed through the LaTeX parser
    assert cached_sympify('a_1') == Symbol('a_1')


def test_derivative_and_integral_latex():
    assert cached_sympify(r'\frac{d}{dx}\left(x^2\right)') == Derivative(x ** 2, x)
    assert cached_sympify(r'\int_{0}^{1} x\, dx') == Integral(x, (x, 0, 1))


def test_unfilled_placeholder_raises_friendly():
    for s in [r'\square', r'\frac{}{}', r'\sqrt{}']:
        with pytest.raises(ValueError):
            cached_sympify(s)


def test_bare_caret_exponent_not_truncated():
    # parse_latex silently truncated an unbraced multi-token exponent (3^sin(x)->3).
    # _brace_bare_powers must keep the whole operand so the integrand survives.
    assert cached_sympify('3^sin(x)*cos(x)') == 3 ** sin(x) * cos(x)
    assert simplify(cached_sympify('x^x') - x ** x) == 0
    assert simplify(cached_sympify('x^(x/2)') - x ** (x / 2)) == 0


def test_bare_caret_integral_solves_not_truncated():
    from routes.compute_service import _dispatch_auto
    ok, res = _dispatch_auto(r'\int (3^sin(x)*cos(x))\, dx')
    assert ok
    assert '无法用初等' not in res


def test_parse_latex_muls_are_flattened_for_matching():
    # parse_latex left an exponent as an unevaluated Mul(1/2, Mul(-1, x)); substitution
    # compares with == so the non-canonical shape made this integral come back None.
    from routes.compute_service import _dispatch_auto
    from domains import IntegralCalculator
    integrand = cached_sympify('(log(x)+1)/(x^(x/2)+x^(-x/2))')
    ok, res = _dispatch_auto(r'\int ((log(x)+1)/(x^(x/2)+x^(-x/2)))\, dx')
    assert ok and '无法用初等' not in res
    steps, _ = IntegralCalculator(True).compute_list(integrand, x)
    anti = steps[-1]
    assert anti is not None and not anti.has(Integral)
    assert simplify(anti.diff(x) - integrand) == 0


def test_expr_op_accepts_wysiwyg_latex():
    # The "等价表达式" op fed MathLive LaTeX used to crash on raw sympify (\\sin, ^, \\cdot);
    # ExpressionParser.parse must go through cached_sympify so the editor's output works.
    from domains.expression_parser import ExpressionParser
    from routes.compute_service import _dispatch_compute
    results = ExpressionParser(max_depth=2).parse(r'\sin{\left(x \right)} + x^{2}')
    assert isinstance(results, list)
    ok, res = _dispatch_compute('expr', {
        'operation_type': 'expr', 'expression': r'x^{2} + 2 x + 1',
        'max_depth': '2', 'sort_strategy': 'complexity'})
    assert ok and isinstance(res, tuple) and res and res[0]
