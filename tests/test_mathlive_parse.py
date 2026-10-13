"""Tests that WYSIWYG (MathLive) LaTeX is correctly converted to SymPy."""

import pytest
from sympy import Derivative, Integral, Symbol, exp, log, simplify, sqrt

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
