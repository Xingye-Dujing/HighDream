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


def test_direction_spellings_are_normalized():
    """'left'/'right' and '+-' must map to the documented directions.

    Regression: 'left' raised a raw English ValueError and '+-' was treated as a
    single direction, so 1/x came back as \\tilde{\\infty} instead of "no limit".
    """
    assert _limit('1/x', 0, 'left') == -S.Infinity
    assert _limit('1/x', 0, 'right') == S.Infinity
    for spelling in ('both', '+-', 'pm'):
        steps, _ = LimitCalculator().compute_list('1/x', x, 0, spelling)
        assert steps[-1] == S.ComplexInfinity, spelling


def test_exp_linear_standard_limit_still_works():
    assert simplify(_limit('(exp(2*x)-1)/x', 0) - 2) == 0


def test_infinity_minus_infinity_radical():
    # sqrt(x+1) - sqrt(x) -> 0 via conjugate rationalization at infinity
    assert simplify(_limit('sqrt(x+1)-sqrt(x)', S('oo'))) == 0


def test_radical_minus_linear_conjugate():
    # sqrt(x**2 + x) - x -> 1/2 (one-radical conjugate case)
    assert simplify(_limit('sqrt(x**2+x)-x', S('oo')) - Rational(1, 2)) == 0


def test_two_sided_domain_restriction_note_is_kept():
    """The feasibility note must survive the single-direction run's generator reset."""
    for expr, note_side in (('sqrt(x)', '0^-'), ('sqrt(-x)', '0^+')):
        calc = LimitCalculator()
        steps, explanations = calc.compute_list(expr, x, 0, 'both')
        assert any('初始检测' in e and note_side in e for e in explanations), explanations
        assert [s for s in steps if s is not None][-1] == 0


def test_bounded_oscillation_infinity():
    # (x + sin x)/(x + cos x) -> 1 at infinity (L'Hopital invalid; divide by dominant power)
    assert simplify(_limit('(x+sin(x))/(x+cos(x))', S('oo')) - 1) == 0
