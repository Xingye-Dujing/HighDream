"""Deterministic tests for definite vs indefinite integral handling."""

import sympy

from sympy import Integral, Rational, Symbol, latex, log, oo, pi, simplify, sqrt

from domains import IntegralCalculator
from domains.integral.rules.integral_function_rules import inverse_trig_rule
from routes.compute_service import _dispatch_compute
from utils.cache_utils import cached_sympify

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


def test_infinity_token_spellings_all_parse_to_oo():
    # users type the infinite bound many ways; all must reach SymPy's oo
    assert cached_sympify('∞') == oo
    assert cached_sympify('-∞') == -oo
    assert cached_sympify('infinity') == oo
    assert cached_sympify('负无穷') == -oo
    assert cached_sympify('正无穷大') == oo


def test_improper_integral_with_unicode_infinity_bound_returns_result():
    # the reported bug: a unicode ∞ bound failed to parse and produced no result
    ok, res = _dispatch_compute(
        'integral',
        {'operation_type': 'integral', 'expression': '1/x^2',
         'variable': 'x', 'upper': '∞', 'lower': '1'})
    assert ok
    assert '\\begin{align}' in res
    assert '&=1' in res.replace(' ', '')   # the improper integral evaluates to 1
    # no leaked literal None and no stray "1 \frac" unit coefficient
    assert 'None' not in res
    assert '1 \\frac' not in res


def test_parse_latex_drops_stray_unit_coefficient():
    # parse_latex turns 1/x^2 into Mul(1, 1/x**2) whose LaTeX shows "1 \frac{...}"
    assert latex(cached_sympify('1/x^2')) == r'\frac{1}{x^{2}}'
    # a genuine standalone 1 inside a sum is preserved
    assert latex(cached_sympify('1+1/x')) == r'1 + \frac{1}{x}'


def test_trig_substitution_backsubs_to_closed_form():
    # Keystone: the trig-substitution dummy keeps its real/positive assumptions through
    # the operation write-back, so the transformed integral is completed and back-
    # substituted instead of shipping a corrupt Integral(f, (u, inverse_of_x)).
    for expr in ['1/sqrt(1-x**2)', 'sqrt(1-x**2)', '1/sqrt(x**2+1)', 'x**2/sqrt(1-x**2)']:
        steps, _ = IntegralCalculator().compute_list(expr, x)
        fin = steps[-1]
        assert fin is not None and not fin.has(Integral), expr


def test_definite_finite_endpoint_improper_takes_limit():
    # ln(x) at 0 gives S.NaN (which exposes neither is_nan nor is_finite==False); the
    # endpoint predicate now routes non-finite values to a one-sided limit -> -1.
    steps, _ = IntegralCalculator().compute_list('ln(x)', x, lower=0, upper=1)
    assert simplify(steps[-1] - (-1)) == 0


def test_definite_periodic_denominator_does_not_hang():
    # sin+cos / 1+cos never vanish on (0, pi/2): _interior_poles must not spin over an
    # infinite Reals ImageSet; confined to the interval it terminates and solves.
    steps, _ = IntegralCalculator().compute_list(
        'sin(x)/(sin(x)+cos(x))', x, lower=0, upper=pi / 2)
    assert simplify(steps[-1] - pi / 4) == 0
    steps2, _ = IntegralCalculator().compute_list(
        'sin(x)/(1+cos(x))', x, lower=0, upper=pi / 2)
    assert simplify(steps2[-1] - log(2)) == 0


def test_sqrt_quadratic_radical_branch_matches_integrand_for_positive_x():
    # The radical substitution inverts a quadratic with two x(u) branches; the + branch
    # (valid for the assumed x>0 region) must be chosen, otherwise the antiderivative's
    # derivative is the NEGATIVE of the integrand for x>0. Check numerically at x=2.
    expr = sympy.sympify('sqrt(x**2 + 2*x)')
    steps, _ = IntegralCalculator().compute_list('sqrt(x**2+2*x)', x)
    fin = steps[-1]
    for s in (fin.free_symbols - {x}):
        fin = fin.subs(s, 0)
    pt = sympy.Float('2.0')
    h = sympy.Float('1e-5')
    d = (fin.subs(x, pt + h) - fin.subs(x, pt - h)) / (2 * h)
    assert abs(float(d) - float(expr.subs(x, pt))) < 1e-4


def test_inverse_trig_arcsec_branch_differentiates_to_integrand():
    # The 1/sqrt(x^2-1) branch used to return acos(x) (wrong sign/domain); it must now
    # return the arcsec-type ln|x+sqrt(x^2-1)| whose derivative is the integrand.
    expr = 1 / sqrt(x ** 2 - 1)
    res = inverse_trig_rule(expr, {'variable': x})
    assert res is not None and res[0].has(log)
    antideriv = res[0]
    h = sympy.Float('1e-5')
    pt = sympy.Float('2.0')
    d = (antideriv.subs(x, pt + h) - antideriv.subs(x, pt - h)) / (2 * h)
    assert abs(float(d) - float(1 / sqrt(3))) < 1e-4


def test_definite_mirror_king_constant_sum():
    # f + f(a+b-x) is a constant -> I = c(b-a)/2, shown step by step.
    steps, _ = IntegralCalculator().compute_list(
        'sin(x)/(sin(x)+cos(x))', x, lower=0, upper=pi / 2)
    assert simplify(steps[-1] - pi / 4) == 0
    steps2, _ = IntegralCalculator().compute_list(
        '1/(1+tan(x)**3)', x, lower=0, upper=pi / 2)
    assert simplify(steps2[-1] - pi / 4) == 0


def test_definite_parity_odd_zero():
    # Odd integrand on a symmetric interval -> 0 (via the mirror constant-sum = 0).
    for expr in ('x**3*cos(x)', 'x/(1+x**2)'):
        steps, _ = IntegralCalculator().compute_list(expr, x, lower=-1, upper=1)
        assert simplify(steps[-1]) == 0
