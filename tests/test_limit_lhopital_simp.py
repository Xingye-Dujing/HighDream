"""Regression tests for the cost-bounded ``_simp`` used in the L'Hopital rules.

``domains/limit/rules/lhopital_rules.py`` replaced 11 ``simplify`` calls with ``_simp``
to skip the ``trigsimp``/``factor`` cost that dominated repeated L'Hopital derivatives.
The optimization is safe only because ``_simp`` is value-preserving (every branch is an
identity transform) and its cheap path never enlarges the expression. These tests lock
both that contract and the exact results of the trig-heavy notebook-limit cells that
exercise it.
"""

from sympy import E, Rational, Symbol, cos, exp, oo, simplify, sin

from domains import LimitCalculator
from domains.limit.rules.lhopital_rules import _simp

x = Symbol('x')


def _limit(expr, point, direction='+-'):
    steps, _ = LimitCalculator().compute_list(expr, x, point, direction)
    return steps[-1]


# --- _simp is value-preserving (every branch is an identity transform) --------

def test_simp_preserves_value_on_trig_ratio():
    e = (exp(-x) * cos(x * exp(-x)) - exp(x) * cos(x * exp(x))) / x ** 2
    assert simplify(_simp(e) - e) == 0


def test_simp_preserves_value_on_rational():
    e = (x ** 2 + 2 * x) / (x * (x + 2))
    assert simplify(_simp(e) - e) == 0


def test_simp_preserves_value_on_expandable_trig():
    e = sin(x + x) / (sin(x) + sin(x))
    assert simplify(_simp(e) - e) == 0


def test_simp_does_not_grow_on_cheap_cancel_branch():
    e = x ** 2 + 2 * x ** 2  # rational function: the cancel branch applies
    assert _simp(e).count_ops() <= e.count_ops()


def test_simp_falls_back_on_non_expr():
    assert _simp(3) == 3


# --- exact results of the trig-heavy notebook-limit cells (end to end) --------

def test_lhopital_cos_difference_over_cubic():
    # notebook-limit cell 4: [cos(x*e^x) - cos(x*e^-x)] / x^3 -> -2 (two-sided)
    assert simplify(_limit('(-cos(x*exp(-x))+cos(x*exp(x)))/x**3', 0) + 2) == 0


def test_lhopital_exp_sin_minus_tan():
    # notebook-limit cell 7: [e^sin x - e^tan x + ln(1+x^3)] / sin(x^2) -> 0
    assert simplify(_limit('(exp(sin(x))-exp(tan(x))+log(x**3+1))/sin(x**2)', 0)) == 0


def test_lhopital_log_rational_at_zero():
    # notebook-limit cell 1: 1 + 1/x^2 - log((x+2)/(2-x))/x^3 -> 11/12
    assert simplify(_limit('1 + 1/x**2 - log((x+2)/(2-x))/x**3', 0) - Rational(11, 12)) == 0


def test_lhopital_power_over_x_keeps_full_simplify():
    # notebook-limit cell 8: [(1+x)^(3/x) - e^3] / x -> -3 e^3 / 2 (right limit).
    # Guards that _simp still falls back to full simplify here: an over-aggressive
    # cancel-only variant wrongly returned -oo for this cell.
    got = _limit('((x+1)**(3/x) - exp(3))/x', 0, '+')
    assert simplify(got + Rational(3, 2) * E ** 3) == 0


def test_lhopital_diverges_to_infinity_honestly():
    # notebook-limit cell 9: numerator ~ 18 x^3 vs denominator ~ 350 x^5 -> +oo.
    # The engine must report the divergence, not fabricate a finite value.
    assert _limit('log(x+1)*log(3*x+1)*sin(6*x)/(5*x**3*(exp(10*x)-1)*sin(7*x))', 0) == oo
