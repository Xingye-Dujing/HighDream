"""Deterministic tests for definite vs indefinite integral handling."""

from sympy import Integral, Rational, Symbol, latex, oo, pi, simplify

from domains import IntegralCalculator
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
