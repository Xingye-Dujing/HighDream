"""Regression tests for the added sec/csc/cot power & product integral rules."""

from sympy import Symbol, Integral, cos, cot, csc, diff, log, sec, simplify, sin, sqrt, tan

from domains import IntegralCalculator

x = Symbol('x')
C = Symbol('C')


def _antiderivative(expr):
    steps, _ = IntegralCalculator().compute_list(expr, x)
    result = steps[-1]
    return (result - C) if result.has(C) else result


def test_sec_squared_is_tan():
    assert simplify(_antiderivative('sec(x)**2') - tan(x)) == 0


def test_plain_sec_uses_textbook_log_form():
    # ∫sec must be the textbook ln|sec+tan| (a real rule), not a doit() log-sum blob
    res = _antiderivative('sec(x)')
    assert res.has(sec(x)) and res.has(tan(x)) and res.has(log)


def test_plain_csc_csch_use_log_forms():
    assert _antiderivative('csc(x)').has(log)
    assert _antiderivative('csch(x)').has(log)


def test_csc_squared_is_neg_cot():
    assert simplify(_antiderivative('csc(x)**2') + cot(x)) == 0


def test_tan_sec_product_is_sec():
    assert simplify(_antiderivative('tan(x)*sec(x)') - sec(x)) == 0


def test_cot_csc_product_is_neg_csc():
    assert simplify(_antiderivative('cot(x)*csc(x)') + csc(x)) == 0


def test_sec_fourth_power_differentiates_back():
    assert simplify(diff(_antiderivative('sec(x)**4'), x) - simplify(sec(x) ** 4)) == 0


def test_trig_power_reduce_even_product():
    # sin^2*cos^2 -> x/8 - sin(4x)/32 (no Weierstrass blowup)
    assert simplify(_antiderivative('sin(x)**2*cos(x)**2') - (x / 8 - sin(4 * x) / 32)) == 0


def test_trig_power_reduce_product_differentiates_back():
    assert simplify(diff(_antiderivative('sin(x)**2*cos(x)**2'), x)
                    - simplify(sin(x) ** 2 * cos(x) ** 2)) == 0


def test_sqrt_one_pm_trig_half_angle():
    # Regression: sqrt(1 ± cos/sin(linear)) used to hit the radical-substitution branch
    # and return a sign-flipped (wrong) antiderivative. The half-angle rewrite fixes it.
    # Verified by finite difference on the principal branch (where |.|-dropping is valid).
    import math

    cases = {
        'sqrt(1 + cos(x))': (sqrt(1 + cos(x)), (0.5, 1.0, 2.0)),
        'sqrt(1 - cos(x))': (sqrt(1 - cos(x)), (0.5, 1.0, 2.0)),
        'sqrt(1 + cos(2*x))': (sqrt(1 + cos(2 * x)), (0.3, 0.6, 1.0)),
        'sqrt(1 + sin(x))': (sqrt(1 + sin(x)), (0.5, 1.0, 2.0)),
    }
    for expr_str, (integrand, pts) in cases.items():
        anti = _antiderivative(expr_str)
        assert not anti.has(Integral)
        for t in pts:
            d = float((anti.subs(x, t + 1e-5) - anti.subs(x, t - 1e-5)) / 2e-5)
            fv = float(integrand.subs(x, t))
            assert abs(d - fv) < 1e-3 * (1 + abs(fv)), (expr_str, t, d, fv)
