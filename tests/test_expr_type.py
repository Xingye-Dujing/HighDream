"""Deterministic tests for the utils.expr_type classification helpers."""

from sympy import Rational, Symbol, acos, asin, atan, cos, exp, log, sin, sqrt

from utils.expr_type import (
    has_radical, is_exp, is_inv_trig, is_log, is_poly, is_trig,
)

x = Symbol('x')


def test_is_log():
    assert is_log(log(x)) is True
    assert is_log(x ** 2) is False


def test_is_poly():
    assert is_poly(x ** 2 + 3 * x + 1, x) is True
    assert is_poly(1 / x, x) is False


def test_is_trig():
    assert is_trig(sin(x) + cos(x)) is True
    assert is_trig(x ** 2) is False


def test_is_inv_trig():
    assert is_inv_trig(asin(x)) is True
    assert is_inv_trig(acos(x)) is True
    assert is_inv_trig(atan(x)) is True
    assert is_inv_trig(sin(x)) is False


def test_is_exp():
    assert is_exp(exp(x), x) is True
    assert is_exp(2 ** x, x) is True
    assert is_exp(x ** 2, x) is False


def test_has_radical():
    # returns a SymPy boolean, so assert on truthiness rather than `is True/False`
    assert has_radical(sqrt(x), x)
    assert has_radical(x ** Rational(1, 3), x)
    assert not has_radical(x ** 2, x)
