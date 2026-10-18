# -*- coding: utf-8 -*-
"""Regression tests for parametric indefinite integrals (a second, constant symbol
alongside the integration variable).

Before the integral engine opted into parameters, an integrand such as
``exp(-a*x)`` raised ``仅允许出现一个字母变量``. The user-facing path now sets
``_allow_parameters`` so those resolve through the rule engine. A numeric
verification guard turns any *wrong* parametric closed form into an honest UNEVAL
(``None``) instead of shipping it, so the tests below assert either a verified
antiderivative or ``None`` — never a wrong expression."""

from sympy import Symbol, exp, sin, cos, sqrt, simplify, Integral

from domains import IntegralCalculator

x = Symbol('x')
a = Symbol('a')
n = Symbol('n')


def _antiderivative(expr):
    """Integrate ``expr`` wrt x in parameter mode; return the final step (may be None)."""
    calc = IntegralCalculator(True)
    calc._allow_parameters = True
    steps, _expl = calc.compute_list(expr, x)
    return steps[-1]


def _exact(anti, expr):
    return anti is not None and not anti.has(Integral) and simplify(anti.diff(x) - expr) == 0


def _finite_diff_ok(anti, expr, aval=1.7, points=(0.31, 0.73, 1.19)):
    """Validate d(anti)/dx == expr numerically (robust to sqrt(x**2)==|x| artifacts)."""
    if anti is None or anti.has(Integral):
        return False
    deriv = anti.diff(x)
    for tv in points:
        d = complex(deriv.subs({x: tv, a: aval}).evalf())
        f = complex(expr.subs({x: tv, a: aval}).evalf())
        if d.imag or f.imag or abs(d - f) > 1e-3 * (1 + abs(f)):
            return False
    return True


def test_linear_composite_parameters():
    # f(a*x) integrates to (1/a) F(a*x) via the substitution ratio path.
    assert _exact(_antiderivative(exp(-a * x)), exp(-a * x))
    assert _exact(_antiderivative(exp(a * x)), exp(a * x))
    assert _exact(_antiderivative(sin(a * x)), sin(a * x))
    assert _exact(_antiderivative(cos(a * x)), cos(a * x))


def test_power_rule_symbolic_exponent():
    assert _exact(_antiderivative(x ** n), x ** n)


def test_polynomial_with_parameter_coefficients():
    assert _exact(_antiderivative(a * x ** 2 + a), a * x ** 2 + a)
    assert _exact(_antiderivative(x * exp(-a * x ** 2)), x * exp(-a * x ** 2))
    assert _exact(_antiderivative(a * x), a * x)


def test_trig_substitution_sqrt_a2_minus_x2():
    expr = sqrt(a ** 2 - x ** 2)
    assert _finite_diff_ok(_antiderivative(expr), expr)


def test_trig_substitution_sqrt_x2_plus_a2():
    # add_rule no longer collapses a**2/cos(u)**3 to \int 0 (it verifies the alpha/beta
    # identity), so the trig-substitution path for sqrt(x**2 + a**2) reduces correctly.
    expr = sqrt(x ** 2 + a ** 2)
    assert _finite_diff_ok(_antiderivative(expr), expr)


def test_arctangent_with_parameter():
    # The inverse-tangent matcher/rule now accept a symbolic constant term.
    expr = 1 / (x ** 2 + a ** 2)
    assert _finite_diff_ok(_antiderivative(expr), expr)


def test_unsolved_parametric_radical_is_honest_none():
    # sqrt(x**2 - a**2) has no parametric path yet; the engine returns an honest
    # UNEVAL (None) rather than a wrong expression.
    assert _antiderivative(sqrt(x ** 2 - a ** 2)) is None


def _definite(expr, lo, hi):
    """Parametric definite integral: return the final recorded value (may be None)."""
    calc = IntegralCalculator(True)
    calc._allow_parameters = True
    steps, _expl = calc.compute_list(expr, x, lo, hi)
    return steps[-1]


def test_definite_parametric_verified_correct():
    # int_0^a x*sqrt(a^2 - x^2) dx = a^3/3 (a>0). The numeric definite guard confirms
    # the closed form, so it is kept; check it matches a^3/3 at a concrete a.
    res = _definite(x * sqrt(a ** 2 - x ** 2), 0, a)
    assert res is not None
    val = complex(res.subs(a, 1.7).evalf())
    assert abs(val.real - 1.7 ** 3 / 3) < 1e-6 and abs(val.imag) < 1e-9


def test_definite_parametric_rejects_garbage():
    # int_0^a sqrt(a^2 - x^2) dx currently yields a non-finite closed form (stray zoo);
    # the numeric definite guard must turn it into an honest UNEVAL, never ship it.
    assert _definite(sqrt(a ** 2 - x ** 2), 0, a) is None
