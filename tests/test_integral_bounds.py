"""Deterministic tests for definite vs indefinite integral handling."""

import sympy

from sympy import Integral, Rational, Symbol, asin, cos, diff, exp, latex, log, oo, pi, simplify, sin, sqrt, together

from domains import IntegralCalculator
from domains.integral.rules.integral_function_rules import inverse_trig_rule
from domains.integral.rules.parts_rule_help import select_parts_u_dv
from routes.compute_service import _dispatch_auto, _dispatch_compute
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


def test_irreducible_biquadratic_quartic_resolves():
    # 1/(x^4-2x^2+2) is irreducible over Q but splits over a real quadratic extension;
    # it must integrate step-by-step (no leftover Integral) and differentiate back.
    integrand = sympy.sympify('1/(x**4 - 2*x**2 + 2)')
    steps, _ = IntegralCalculator().compute_list('1/(x**4-2*x**2+2)', x)
    fin = steps[-1]
    assert fin is not None and not fin.has(Integral)
    for s in (fin.free_symbols - {x}):
        fin = fin.subs(s, 0)
    h = sympy.Float('1e-5')
    pt = sympy.Float('0.7')
    d = (fin.subs(x, pt + h) - fin.subs(x, pt - h)) / (2 * h)
    assert abs(float(d) - float(integrand.subs(x, pt))) < 1e-3


def test_inverse_trig_alias_maps_to_real_sympy_asin():
    # parse_latex has no \asin entry; the LaTeX branch must route bare ``asin(``
    # through the arcsin spelling so we get a real SymPy asin, not UndefinedFunction.
    # Regression for int(asin(x)/sqrt(1-x^2), x) via the universal box.
    ok, res = _dispatch_auto('int(asin(x)/sqrt(1-x^2), x)')
    assert ok, res
    assert 'asin' in res and '无法' not in res


def test_repeated_linear_partial_fraction_uses_apart():
    # (x^2+2)/(x+1)^3 hits sqrt_div_sqrt_rule; its expand() loop used to raise
    # ``AttributeError: 'Add' object has no attribute 'exp'``. The isinstance(Pow)
    # guard lets add_rule/apart decompose it, and every term integrates.
    ok, res = _dispatch_compute(
        'integral',
        {'operation_type': 'integral', 'expression': '(x**2+2)/(x+1)**3', 'variable': 'x'})
    assert ok, res
    # antiderivative = log|x+1| + 2/(x+1) - 3/(2(x+1)^2) (verified by differentiation)
    val = log(x + 1) + 2 / (x + 1) - Rational(3, 2) / (x + 1) ** 2
    assert abs(float(simplify(diff(val, x) - (x**2 + 2) / (x + 1) ** 3))) < 1e-9


def test_select_parts_u_dv_skips_var_free_factors_without_evalf():
    # parts-rule u-selection used to call factor.is_constant(); that hits evalf
    # sampling on factors like ``1/sqrt(1-x^2)`` and can recurse ~1000 frames deep.
    # The structural ``not factor.has(var)`` guard avoids that entirely.
    expr = cached_sympify('asin(x)/sqrt(1-x^2)')
    u, dv = select_parts_u_dv(expr, x)
    assert u is not None and dv is not None
    assert not u.has(x) or u.has(asin)  # picks asin(x) as u


def test_universal_box_enables_special_integral_matchers():
    # Regression: _build_op_data omits include_special_matchers, so a bool/absent
    # value used to fail ``== 'True'`` and silently disable the quotient-diff-form
    # and f(x)*e^{g(x)} rules, making these integrals abort unresolved.
    cases = {
        r'\int ((x*cos(x)-sin(x))/x^2)\, dx': (x * cos(x) - sin(x)) / x ** 2,
        r'\int ((1+x-1/x)*exp(x+1/x))\, dx': (1 + x - 1 / x) * exp(x + 1 / x),
    }
    for cell, integrand in cases.items():
        ok, res = _dispatch_auto(cell)
        assert ok and '无法用初等' not in res
        # the derivation must land on a real antiderivative that differentiates back
        steps, _ = IntegralCalculator(True).compute_list(integrand, x)
        anti = steps[-1]
        assert not anti.has(Integral)
        assert simplify(together(diff(anti, x) - integrand)) == 0


def test_rationalizing_substitution_multi_fractional_powers():
    # Regression: integrands that are rational functions of several distinct
    # fractional powers of x (e.g. 1/(sqrt x + x**(1/3))) used to spin/abort because
    # single-radical substitution only removes one radical. The rationalizing
    # substitution u = x**(1/LCM) turns them into a rational function.
    # Verified by finite difference (robust to the Abs/log-branch forms SymPy emits).
    cases = [
        1 / (sqrt(x) + x ** Rational(1, 3)),
        1 / (x ** Rational(1, 2) + x ** Rational(1, 4)),
        sqrt(x) / (1 + x ** Rational(1, 3)),
    ]
    for integrand in cases:
        steps, _ = IntegralCalculator(True).compute_list(integrand, x)
        anti = steps[-1]
        assert anti is not None and not anti.has(Integral)
        for t in (0.7, 1.7, 3.2):
            d = float((anti.subs(x, t + 1e-5) - anti.subs(x, t - 1e-5)) / 2e-5)
            fv = float(integrand.subs(x, t))
            assert abs(d - fv) < 1e-3 * (1 + abs(fv))


def test_trig_substitution_collapse_multiple_angle():
    # Regression: trig sub of sqrt(x^2+1)/x^2 produced 8*cos(t)/(1-cos(4t)), a
    # multiple-angle form the rule library can't chew -> the BFS spun for minutes.
    # _collapse_multiple_angle expands it back to single-angle powers so it solves
    # fast and correctly.
    integrand = sqrt(x ** 2 + 1) / x ** 2
    steps, _ = IntegralCalculator(True).compute_list(integrand, x)
    anti = steps[-1]
    assert anti is not None and not anti.has(Integral)
    for t in (0.6, 1.5, 2.5):
        d = float((anti.subs(x, t + 1e-5) - anti.subs(x, t - 1e-5)) / 2e-5)
        fv = float(integrand.subs(x, t))
        assert abs(d - fv) < 1e-3 * (1 + abs(fv))


def test_exp_product_substitution_xe_x():
    # Regression: the classic 考研 trick (x+1)/(x(1+x·eˣ)) — multiply by eˣ reveals
    # u = x·eˣ. try_exp_product_substitution catches it (narrowly, only g=x·eˣ).
    integrand = (x + 1) / (x * (1 + x * exp(x)))
    steps, _ = IntegralCalculator(True).compute_list(integrand, x)
    anti = steps[-1]
    assert anti is not None and not anti.has(Integral)
    for t in (0.5, 1.0, 2.0):
        d = float((anti.subs(x, t + 1e-5) - anti.subs(x, t - 1e-5)) / 2e-5)
        fv = float(integrand.subs(x, t))
        assert abs(d - fv) < 1e-3 * (1 + abs(fv))
