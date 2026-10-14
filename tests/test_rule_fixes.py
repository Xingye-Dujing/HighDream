# -*- coding: utf-8 -*-
"""Regression tests for correctness bugs fixed in the Maple-gap audit."""

from routes.compute_service import _dispatch_auto, _dispatch_compute


def test_quotient_rule_is_used():
    # (x+1)/(x-1) must take the quotient rule, not 9 product+chain steps.
    ok, res = _dispatch_compute('diff', {
        'operation_type': 'diff', 'expression': '(x+1)/(x-1)', 'variable': 'x'})
    assert ok
    assert '除法' in res or 'quotient' in res.lower()


def test_finite_sum_closed_form_functional():
    ok, res = _dispatch_auto('sum(k^2,k,1,100)')
    assert ok, res
    assert '338350' in res  # Faulhaber: sum k^2, 1..100


def test_finite_sum_closed_form_latex():
    ok, res = _dispatch_auto(r'\sum_{k=1}^{100} k^2')
    assert ok, res
    assert '338350' in res


def test_infinite_series_still_convergence_analysis():
    # sum(a_n, n) with no upper bound stays a convergence analysis (not a finite sum)
    ok, res = _dispatch_auto('sum(1/n^2,n)')
    assert ok, res
    assert '敛散' in res or '收敛' in res or '判别' in res


def test_radius_oscillating_coefficient_no_crash():
    # previously crashed: "Result depends on the sign of I"
    ok, _res = _dispatch_auto('radius(2**n*(1+(-1)**n),n)')
    assert ok


def test_simplify_alone_returns_rows():
    ok, res = _dispatch_auto('simplify(x+2*x+3*x)')
    assert ok
    assert isinstance(res, tuple) and len(res[0]) >= 1


def test_higher_derivative_order_engine():
    # diff(...,x,n) must run the rule engine n times feeding back (no .diff(n) black box)
    ok, res = _dispatch_compute('diff', {
        'operation_type': 'diff', 'expression': 'x^3', 'variable': 'x', 'order': '2'})
    assert ok
    assert '6 x' in res and '3 x^{2}' in res  # first then second derivative shown


def test_higher_derivative_composes():
    ok, res = _dispatch_auto('3*diff(x^3,x,2)')
    assert ok, res
    assert '\\boxed{18 x}' in res


def test_matrix_trace_value_and_compose():
    ok, res = _dispatch_auto('trace([[1,2],[3,4]])')
    assert ok and '\\boxed{5}' in res
    ok2, res2 = _dispatch_auto('2*trace([[1,2],[3,4]])')
    assert ok2 and '\\boxed{10}' in res2


def test_matrix_nullspace_colspace_rowspace():
    ok, res = _dispatch_auto('nullspace([[1,2],[2,4]])')
    assert ok, res
    assert r'-2' in res and '零空间' in res
    ok, res = _dispatch_auto('colspace([[1,2],[2,4]])')
    assert ok and '列空间' in res
    ok, res = _dispatch_auto('rowspace([[1,2],[3,4]])')
    assert ok and '行空间' in res


def test_numerical_ivp_methods():
    # y'=x+y, y(0)=1 on [0,1], n=4. RK4 is the most accurate; order euler<heun<rk4.
    ok_e, res_e = _dispatch_auto('euler(x+y,x,0,1,1,4)')
    ok_h, res_h = _dispatch_auto('heun(x+y,x,0,1,1,4)')
    ok_r, res_r = _dispatch_auto('rk4(x+y,x,0,1,1,4)')
    assert ok_e and ok_h and ok_r
    assert '欧拉法' in res_e and 'Runge-Kutta' in res_r
    # RK4 y(1) ~ 3.436 (exact 2e-2=3.4366)
    assert '3.436' in res_r


def test_ivp_table_not_scalar_composable():
    ok, res = _dispatch_auto('2*euler(x+y,x,0,1,1,4)')
    assert not ok  # a trajectory table can't be embedded in scalar arithmetic


def test_root_finding_methods():
    # root of x^2-2 near sqrt(2)=1.41421
    ok, res = _dispatch_auto('bisection(x^2-2,x,1,2,8)')
    assert ok and '1.414' in res and '根的近似' in res
    ok, res = _dispatch_auto('secant(x^2-2,x,1,2,8)')
    assert ok and '1.41421' in res
    ok, res = _dispatch_auto('fixedpoint(sqrt(x+2),x,1,10)')
    assert ok and '不动点近似' in res


def test_bisection_requires_sign_change():
    ok, res = _dispatch_auto('bisection(x^2+1,x,0,1,5)')  # always positive -> no bracket
    assert not ok and '异号' in res


def test_matrix_pinv():
    # full column rank uses (A^T A)^-1 A^T ; result shown boxed, matrix not scalar-composable
    ok, res = _dispatch_auto('pinv([[1,2],[3,4],[5,6]])')
    assert ok and '列满秩' in res and '\\boxed' in res
    ok2, _ = _dispatch_auto('2*pinv([[1,2],[3,4],[5,6]])')
    assert not ok2  # matrix result can't enter scalar arithmetic
