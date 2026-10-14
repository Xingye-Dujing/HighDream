# -*- coding: utf-8 -*-
"""Regression tests for parameterized limits (a second, constant symbol alongside x).

Before the limit calculator honored its working variable (``_allow_multi_symbols``), a
limit whose expression carried a parameter (e.g. sin(a*x)/x) raised
``仅允许出现一个字母变量``. These cases now resolve to the correct symbolic answer."""

from sympy import Symbol, exp, oo, simplify

from domains import LimitCalculator

x = Symbol('x')
a = Symbol('a')


def _limit(expr_str, point=0):
    calc = LimitCalculator()
    steps, _expl = calc.compute_list(expr_str, x, point)
    vals = [s for s in steps if s is not None]
    return vals[-1]


def test_parameterized_limits_resolve():
    assert simplify(_limit("sin(a*x)/x") - a) == 0
    assert simplify(_limit("(1-cos(a*x))/x**2") - a ** 2 / 2) == 0
    assert simplify(_limit("(exp(a*x) - 1)/x") - a) == 0


def test_parameterized_limit_exponential_form():
    # (1 + x)^(a/x) as x->0 equals e^a
    assert simplify(_limit("(1 + x)**(a/x)") - exp(a)) == 0
    # (1 + a/x)^x as x->oo equals e^a
    assert simplify(_limit("(1 + a/x)**x", point=oo) - exp(a)) == 0
