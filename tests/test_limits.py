"""Regression tests for the limit engine path (guards the sympified-input flow)."""

from sympy import Rational, S, Symbol, simplify

from domains import LimitCalculator

x = Symbol('x')


def _limit(expr, point, direction='+'):
    steps, _ = LimitCalculator().compute_list(expr, x, point, direction)
    return steps[-1]


def test_sin_over_x_at_zero():
    assert simplify(_limit('sin(x)/x', 0) - 1) == 0


def test_one_minus_cos_over_x_squared():
    assert simplify(_limit('(1-cos(x))/x**2', 0) - Rational(1, 2)) == 0


def test_polynomial_ratio_at_infinity():
    assert simplify(_limit('(2*x**2+1)/(3*x**2-5)', S('oo')) - Rational(2, 3)) == 0


def test_exponential_minus_one_over_x():
    assert simplify(_limit('(exp(x)-1)/x', 0) - 1) == 0


def test_removable_singularity_at_finite_point():
    assert simplify(_limit('(x**3-1)/(x-1)', 1) - 3) == 0


def test_third_order_trig_quotient():
    # (sin x - x)/x**3 -> -1/6; guards that mul_split must not treat (cos x - 1)
    # as the exp-standard-limit form and split into 0 * oo.
    assert simplify(_limit('(sin(x)-x)/x**3', 0) + Rational(1, 6)) == 0


def test_exp_linear_standard_limit_still_works():
    assert simplify(_limit('(exp(2*x)-1)/x', 0) - 2) == 0


def test_infinity_minus_infinity_radical():
    # sqrt(x+1) - sqrt(x) -> 0 via conjugate rationalization at infinity
    assert simplify(_limit('sqrt(x+1)-sqrt(x)', S('oo'))) == 0


def test_radical_minus_linear_conjugate():
    # sqrt(x**2 + x) - x -> 1/2 (one-radical conjugate case)
    assert simplify(_limit('sqrt(x**2+x)-x', S('oo')) - Rational(1, 2)) == 0
