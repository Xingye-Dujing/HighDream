"""Regression tests for the equivalent-infinitesimal substitution rule.

``domains/limit/rules/important_rules.py::equivalent_infinitesimal_rule`` rewrites only
*independent multiplicative/divisive* product factors whose inner argument tends to 0
(sin u~u, tan u~u, asin/atan u~u, 1-cos u~u^2/2, e^u-1~u, ln(1+u)~u). It must NEVER fire
inside a sum, because replacing a term of a difference (e.g. sin x - x) changes the
limit — that case must fall through to L'Hopital.
"""

from sympy import Rational, Symbol, simplify

from domains import LimitCalculator
from domains.limit.rules.important_rules import equivalent_infinitesimal_matcher
from utils.cache_utils import cached_sympify

x = Symbol('x')


def _solve_and_fired(expr, point, direction='+'):
    calc = LimitCalculator()
    steps, expl = calc.compute_list(expr, x, point, direction)
    conc = [s for s in steps if s is not None]
    fired = any('等价无穷小' in e for e in expl)
    return conc[-1], fired


def test_product_of_sinfactors_substituted():
    val, fired = _solve_and_fired('tan(x)*sin(x)/x**2', 0)
    assert fired
    assert simplify(val - 1) == 0


def test_one_minus_cos_double_angle_substituted():
    # 1 - cos(2x) ~ (2x)^2/2 = 2x^2 -> ratio 2
    val, fired = _solve_and_fired('(1-cos(2*x))/x**2', 0)
    assert fired
    assert simplify(val - 2) == 0


def test_exp_and_log_product_substituted():
    val, fired = _solve_and_fired('(exp(x)-1)*log(1+2*x)/x**2', 0)
    assert fired
    assert simplify(val - 2) == 0


def test_inverse_trig_products_substituted():
    assert simplify(_solve_and_fired('atan(x)/x', 0)[0] - 1) == 0
    assert simplify(_solve_and_fired('asin(x)/x', 0)[0] - 1) == 0


def test_difference_is_NEVER_substituted():
    # sin(x) - x is a sum term, not an independent factor: substitution must not fire,
    # the -1/6 answer must come from L'Hopital, proving the pitfall is avoided.
    val, fired = _solve_and_fired('(sin(x)-x)/x**3', 0)
    assert not fired
    assert simplify(val + Rational(1, 6)) == 0


def test_matcher_declines_on_pure_sum():
    ctx = {'variable': x, 'point': 0, 'direction': '+'}
    # a lone Add is not a factor -> matcher returns None
    assert equivalent_infinitesimal_matcher(_expr('sin(x) + x'), ctx) is None


def _expr(text):
    return cached_sympify(text)
