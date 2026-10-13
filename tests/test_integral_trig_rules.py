"""Regression tests for the added sec/csc/cot power & product integral rules."""

from sympy import Symbol, cot, csc, diff, sec, simplify, tan

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
