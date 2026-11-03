"""Regression tests for the WYSIWYG input -> backend symmetry.

These cover the cases that used to fail silently or produce a wrong answer:
MathLive/LaTeX constants, operatorname functions, implicit multiplication, and
the explicit-operation paths that used to double-apply derivatives/integrals.
"""

import pytest
from sympy import Abs, I, pi

from routes.compute_service import _normalize_latex_ops, start_compute
from utils.cache_utils import cached_sympify


def test_latex_pi_is_the_constant():
    assert cached_sympify(r'\pi') == pi
    assert cached_sympify(r'2\pi') == 2 * pi
    assert cached_sympify(r'\frac{\pi}{2}') == pi / 2


def test_operatorname_functions_are_kept():
    assert str(cached_sympify(r'\operatorname{re}(z)')) == 'Re(z)'
    assert cached_sympify(r'\operatorname{abs}(z)') == Abs(cached_sympify('z'))
    assert cached_sympify(r'\left|z\right|') == Abs(cached_sympify('z'))
    assert cached_sympify(r'\operatorname{Im}(z)').has(I) is False


def test_textbook_implicit_multiplication():
    assert cached_sympify('x y') == cached_sympify('x*y')
    assert cached_sympify('2x') == cached_sympify('2*x')


def test_unfilled_template_is_rejected():
    with pytest.raises(ValueError):
        cached_sympify(r'\frac{}{}')


@pytest.mark.parametrize('expr,expected_op', [
    (r'\frac{d}{dx}(x^{2})', 'diff('),
    (r'\int_{0}^{1}(x^{2})\,dx', 'int('),
    (r'\lim_{x \to 0}(\frac{\sin x}{x})', 'limit('),
    (r'\sum_{n=1}^{3}(n)', 'sum('),
    (r'\prod_{n=1}^{3}(n)', 'prod('),
])
def test_latex_structures_normalize(expr, expected_op):
    text = _normalize_latex_ops(expr)
    assert expected_op in text


def test_explicit_diff_does_not_double_differentiate():
    ok, res = start_compute('diff', {
        'operation_type': 'diff',
        'expression': r'\frac{d}{dx}(x^{2})',
        'variable': 'x',
    })
    assert ok
    assert r'\frac{d^{2}}{d x^{2}}' not in res
    assert '2 x' in res


def test_explicit_integral_does_not_double_integrate():
    ok, res = start_compute('integral', {
        'operation_type': 'integral',
        'expression': r'\int (x^{2})\,dx',
        'variable': 'x',
    })
    assert ok
    assert r'\iint' not in res
    assert r'\frac{x^{3}}{3}' in res


def test_explicit_limit_with_latex_is_two_sided():
    ok, res = start_compute('limit', {
        'operation_type': 'limit',
        'expression': r'\lim_{x\to0}\frac{\sin x}{x}',
        'variable': 'x',
        'point': '0',
        'direction': '+',
    })
    assert ok
    assert '1' in res


def test_projection_accepts_row_vector():
    ok, res = start_compute('projection', {
        'operation_type': 'projection',
        'expression': '[[1,2]]\n[[1,0],[0,1]]',
    })
    assert ok
    assert '错误' not in res


def test_ode_accepts_diff_notation():
    ok, res = start_compute('ode', {
        'operation_type': 'ode',
        'expression': 'diff(y,x)-y',
        'variable': 'x',
    })
    assert ok
    assert '无法求解' not in res


def test_complex_i_is_imaginary_unit():
    ok, res = start_compute('complex_simplify', {
        'operation_type': 'complex_simplify',
        'expression': '(1+2i)/(3-i)',
    })
    assert ok
    assert 'i' in res
    assert 're(i)' not in res


def test_power_series_rejects_function_input():
    ok, msg = start_compute('power_series', {
        'operation_type': 'power_series',
        'expression': '1/(1-x)',
        'variable': 'x',
        'point': '0',
    })
    assert not ok
    assert '系数' in msg
