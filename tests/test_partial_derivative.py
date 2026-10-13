"""Regression tests for partial derivatives (multi-symbol differentiation)."""

from sympy import Symbol, cos, diff, sin, simplify

from domains import DiffCalculator

x = Symbol('x')
y = Symbol('y')
t = Symbol('t')


def _final(expr, var):
    steps, _ = DiffCalculator().compute_list(expr, var)
    return steps[-1]


def test_partial_derivative_wrt_x():
    assert simplify(_final('x**2 + x*y + y**2', x) - (2 * x + y)) == 0


def test_partial_derivative_wrt_y():
    assert simplify(_final('x**2 + x*y + y**2', y) - (x + 2 * y)) == 0


def test_partial_derivative_chain_across_symbols():
    expected = y * cos(x * y) + t * x ** (t - 1)
    assert simplify(_final('sin(x*y) + x**t', x) - expected) == 0


def test_single_variable_still_works():
    assert simplify(_final('sin(x)*x', x) - diff(sin(x) * x, x)) == 0
