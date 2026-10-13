"""Deterministic tests for definite vs indefinite integral handling."""

from sympy import Rational, Symbol, pi, simplify

from domains import IntegralCalculator

x = Symbol('x')
C = Symbol('C')


def test_indefinite_result_has_constant_of_integration():
    steps, _ = IntegralCalculator().compute_list('x**2', x)
    assert steps[-1].has(C)


def test_definite_zero_to_one_of_x_squared():
    steps, _ = IntegralCalculator().compute_list('x**2', x, lower=0, upper=1)
    assert not steps[-1].has(C)
    assert simplify(steps[-1] - Rational(1, 3)) == 0


def test_definite_sin_zero_to_pi():
    steps, _ = IntegralCalculator().compute_list('sin(x)', x, lower=0, upper=pi)
    assert simplify(steps[-1] - 2) == 0


def test_definite_explanation_present():
    _, explanations = IntegralCalculator().compute_list('x**2', x, lower=0, upper=1)
    assert any(('牛顿' in e) or ('莱布尼茨' in e) for e in explanations)


def test_definite_upper_only_defaults_lower_to_zero():
    steps, _ = IntegralCalculator().compute_list('x', x, upper=2)
    assert simplify(steps[-1] - 2) == 0
