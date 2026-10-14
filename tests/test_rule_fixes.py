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


def test_complex_nth_roots_all_branches():
    # De Moivre: n-th roots of w, all n branches shown
    ok, res = _dispatch_auto('nthroots(1+I,2)')
    assert ok and 'n 次根' in res and '第 0 个根' in res and '第 1 个根' in res


def test_series_absolute_vs_conditional():
    # Σ(-1)^n/n^2 absolutely convergent; Σ(-1)^n/n conditionally convergent
    ok, res = _dispatch_auto('sum((-1)^{n}/n^{2}, n)')
    assert ok and '绝对收敛' in res
    ok, res = _dispatch_auto('sum((-1)^{n}/n, n)')
    assert ok and '条件收敛' in res


def test_power_series_endpoint_conditional_still_included():
    # Σ x^n/n : at x=-1 it's the alternating harmonic (conditional) -> left endpoint kept -> [-1, 1)
    from routes.compute_service import _dispatch_compute
    ok, res = _dispatch_compute('power_series', {
        'operation_type': 'power_series', 'expression': '1/n', 'variable': 'n', 'point': '0'})
    assert ok and r'[-1' in res and 'R=1' in res


def test_taylor_shows_order_remainder_term():
    ok, res = _dispatch_auto('taylor(e^x,x,0,3)')
    assert ok and 'O(' in res  # Peano remainder term displayed


def test_taylor_rejects_non_analytic_point():
    ok, res = _dispatch_auto('taylor(log(x),x,0,3)')
    assert not ok and '不能作泰勒展开' in res


def test_matrix_jordan_form():
    # defective matrix -> a non-diagonal Jordan block; diagonal case also works
    ok, res = _dispatch_auto('jordan([[4,1],[0,4]])')
    assert ok and 'J =' in res and 'P =' in res and 'Jordan' in res
    ok2, res2 = _dispatch_auto('jordan([[2,0],[0,3]])')
    assert ok2 and 'J =' in res2


def test_matrix_definiteness_sylvester():
    assert '正定' in _dispatch_auto('definite([[2,0],[0,3]])')[1]
    assert '负定' in _dispatch_auto('definite([[-1,0],[0,-2]])')[1]
    assert '不定' in _dispatch_auto('definite([[1,2],[2,1]])')[1]


def test_matrix_charpoly():
    # det(A - lambda I): rotation-like [[0,-1],[1,0]] -> lambda^2 + 1
    ok, res = _dispatch_auto('charpoly([[0,-1],[1,0]])')
    assert ok and 'p(' in res and '\\lambda' in res and '因式分解' in res


def test_gradient_multivariable():
    # grad(f) shows each partial (other symbols held constant) then the gradient vector
    ok, res = _dispatch_auto('grad(x^2*y + y*z, x)')
    assert ok and '标量场' in res and '求偏导' in res and '梯度' in res


def test_definite_integral_with_substitution_uses_backsub_form():
    # 1/x^2 integrates via substitution to -1/x (back-substituted); Newton-Leibniz on
    # that gives 1. Using the pre-back-sub (u-form) would wrongly evaluate to 0.
    ok, res = _dispatch_auto('int(1/x^2,x,1,oo)')
    assert ok, res
    assert r'- \frac{1}{x}' in res   # back-substituted antiderivative used
    assert '= 1 = 1' in res          # final value is 1, not 0


def test_definite_integral_ordinary_and_improper():
    ok, res = _dispatch_auto('int(x^2,x,0,1)')
    assert ok and '\\frac{1}{3}' in res
    ok2, res2 = _dispatch_auto('int(1/(1+x^2),x,0,oo)')
    assert ok2 and '\\pi' in res2


def test_improper_rational_integral_no_crash():
    # (x^3+x)/(x^2+1) divides exactly to the polynomial x; handle_poly must reduce it
    # instead of crashing in the irreducible-quadratic branch.
    ok, res = _dispatch_auto('int((x^3+x)/(x^2+1),x)')
    assert ok, res
    assert 'x^{2}}{2}' in res or '\\frac{x^{2}}{2}' in res
    # a mixed improper fraction: polynomial part + arctan part, must also succeed
    ok2, res2 = _dispatch_auto('int((2*x^2+3)/(x^2+1),x)')
    assert ok2 and 'atan' in res2


def test_interior_pole_definite_integral_diverges():
    # Newton-Leibniz across an interior pole is invalid. These must be reported as
    # divergent (split into improper sub-integrals), NOT a spurious finite number.
    for e in [r'int(1/x,x,-1,1)', r'int(1/x^2,x,-1,1)', r'int(1/(x-0.5),x,0,1)']:
        ok, res = _dispatch_auto(e)
        assert ok and '发散' in res, (e, res)
    # a pole exactly at 0 for 1/x^2 must not return the blind value -2
    _, res2 = _dispatch_auto(r'int(1/x^2,x,-1,1)')
    assert '= -2' not in res2 and 'NaN' in res2


def test_no_false_pole_on_smooth_definite_integral():
    # smooth integrands with no interior pole are unaffected (still finite values)
    ok, res = _dispatch_auto(r'int(1/(1+x^2),x,-1,1)')
    assert ok and '\\pi' in res and '发散' not in res
    ok2, res2 = _dispatch_auto(r'int(x^2,x,-2,2)')
    assert ok2 and '16' in res2 and '发散' not in res2
