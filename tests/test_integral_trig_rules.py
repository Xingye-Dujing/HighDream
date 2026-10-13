"""Regression tests for the added sec/csc/cot power & product integral rules."""

from sympy import Symbol, cos, cot, csc, diff, sec, simplify, sin, tan

from domains import IntegralCalculator

x = Symbol('x')
C = Symbol('C')


def _antiderivative(expr):
    steps, _ = IntegralCalculator().compute_list(expr, x)
    result = steps[-1]
    return (result - C) if result.has(C) else result


def test_sec_squared_is_tan():
    assert simplify(_antiderivative('sec(x)**2') - tan(x)) == 0


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
