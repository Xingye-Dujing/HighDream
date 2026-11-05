"""Regression tests for the applied-calculus / series / ODE / numerical ops."""

from routes.compute_service import start_compute


def _res(op, **kw):
    ok, res = start_compute(op, {'operation_type': op, **kw})
    assert ok, res
    return str(res).replace(' ', '')


def test_tangent_line():
    assert r'2x-1' in _res('tangent', expression='x**2', variable='x', point='1')


def test_user_iteration_counts_are_clamped():
    """API clients must not be able to request an unbounded loop count."""
    from config import HARD_MAX_ITERATIONS  # pylint: disable=import-outside-toplevel
    from domains.applied._common import to_count  # pylint: disable=import-outside-toplevel

    assert to_count('1000000000', 4) == HARD_MAX_ITERATIONS
    assert to_count(None, 4) == 4
    assert to_count('-3', 4) == 1
    body = _res('euler', expression='x', variable='x', point='0',
                yvalue='0', target='1', steps='1000000000')
    assert body


def test_arc_length():
    assert r'\sqrt{2}' in _res('arc_length', expression='x', variable='x', lower='0', upper='1')


def test_disk_volume():
    body = _res('volume_disk', expression='x', variable='x', lower='0', upper='1')
    assert r'\pi' in body


def test_taylor():
    assert r'x^{4}' in _res('taylor', expression='exp(x)', variable='x', point='0', order='4')


def test_improper_converges():
    body = _res('improper_integral', expression='exp(-x)', variable='x', lower='0', upper='oo')
    assert '收敛' in body and '$' not in body


def test_series_converges_and_sums():
    # 1/n**2 converges with closed form pi**2/6
    assert r'\pi' in _res('series_convergence', expression='1/n**2', variable='n')


def test_parametric():
    assert r'3t}{2}' in _res('parametric_diff', expression='t**2\nt**3', variable='t')


def test_ode():
    assert r'f{' in _res('ode', expression="f'(x)+f(x)-exp(x)", variable='x')


def test_ode_separable_rule():
    # y' = x*y handled by the separable rule (shows separated factors)
    body = _res('ode', expression='Derivative(f(x),x)-x*f(x)', variable='x')
    assert '分离' in body and '$' not in body


def test_ode_linear_integrating_factor():
    # y' + y = e^x handled by the linear rule (shows integrating factor)
    body = _res('ode', expression='Derivative(f(x),x)+f(x)-exp(x)', variable='x')
    assert '积分因子' in body and '$' not in body


def test_ode_exact():
    # (2xy) + (x^2+1) y' = 0 exact -> (x^2+1) y = C
    body = _res('ode', expression='2*x*f(x)+(x**2+1)*Derivative(f(x),x)', variable='x')
    assert '恰当' in body and 'x^{2}+1' in body.replace(' ', '') and '$' not in body


def test_ode_second_order_distinct_roots():
    # y'' - 5y' + 6y = 0 -> e^{2x}, e^{3x}
    body = _res('ode', expression='Derivative(f(x),(x,2))-5*Derivative(f(x),x)+6*f(x)', variable='x')
    assert '特征方程' in body and 'e^{2x}' in body.replace(' ', '') and 'e^{3x}' in body.replace(' ', '')


def test_ode_second_order_complex_roots():
    # y'' + y = 0 -> cos x, sin x
    body = _res('ode', expression='Derivative(f(x),(x,2))+f(x)', variable='x')
    assert '\\cos' in body and '\\sin' in body


def test_ode_second_order_nonhomogeneous_variation():
    # y'' + y = x -> homogeneous (cos/sin) + particular via variation of parameters
    body = _res('ode', expression='Derivative(f(x),(x,2))+f(x)-x', variable='x')
    assert '朗斯基' in body and '特解' in body and '通解' in body and '$' not in body


def test_critical_points():
    assert r'3x^{2}' in _res('critical_points', expression='x**3-3*x', variable='x')


def test_numerical_simpson_exact_for_quadratic():
    assert r'0.33333' in _res('numerical_integration', expression='x**2', variable='x',
                              lower='0', upper='1', order='4')


def test_mean_value_theorem():
    assert 'c=1' in _res('mean_value', expression='x**2', variable='x', lower='0', upper='2')


def test_radius_of_convergence():
    assert 'R=1' in _res('radius_of_convergence', expression='1/n', variable='n')


def test_complex_polar_modulus():
    assert r'\sqrt{2}' in _res('complex_polar', expression='1+I', variable='x')


def test_complex_roots():
    assert '-i' in _res('complex_roots', expression='z**2+1', variable='z')


def test_taylor_error_remainder():
    body = _res('taylor_error', expression='sin(x)', variable='x', point='0', order='3')
    assert 'sin' in body and '$' not in body


def test_series_absolute_and_integral_test_present():
    body = _res('series_convergence', expression='1/n**2', variable='n')
    assert '收敛' in body and 'p-级数' in body and '$' not in body


def test_convergence_ratio_test_factorial():
    # 1/n! converges by the ratio test (rho = 0 < 1)
    body = _res('series_convergence', expression='1/factorial(n)', variable='n')
    assert '比值判别' in body and '绝对收敛' in body and '$' not in body


def test_convergence_alternating_leibniz():
    # (-1)^n / n converges by Leibniz (nth-term 0, ratio/root inconclusive)
    body = _res('series_convergence', expression='(-1)**n/n', variable='n')
    assert '莱布尼茨' in body and '$' not in body


def test_convergence_nth_term_diverges():
    # 1/n diverges (p-series p=1)
    body = _res('series_convergence', expression='1/n', variable='n')
    assert '发散' in body and '$' not in body


def test_convergence_limit_comparison():
    # 1/(n^2+1) ~ 1/n^2 -> converges by limit comparison
    body = _res('series_convergence', expression='1/(n**2+1)', variable='n')
    assert '极限比较' in body and '收敛' in body and '$' not in body
    # 1/(n+1) ~ 1/n -> diverges
    body2 = _res('series_convergence', expression='1/(n+1)', variable='n')
    assert '极限比较' in body2 and '发散' in body2


def test_convergence_geometric():
    # 1/2**n geometric, |r|=1/2<1 -> converges (ratio test also catches it)
    body = _res('series_convergence', expression='1/2**n', variable='n')
    assert '收敛' in body and '$' not in body


def test_convergence_integral_test():
    # 1/(n log n) diverges by the integral test
    body = _res('series_convergence', expression='1/(n*log(n))', variable='n')
    assert '积分判别' in body and '发散' in body


def test_calculus_auto_detect_derivative():
    body = _res('calculus', expression=r'\frac{d}{dx}\left(x^2\right)', variable='x')
    assert '幂' in body and '$' not in body


def test_calculus_auto_detect_definite_integral():
    body = _res('calculus', expression=r'\int_{0}^{1} x^{2}\, dx', variable='x')
    assert '牛顿-莱布尼茨' in body and '1}{3}' in body.replace(' ', '')


def test_calculus_auto_detect_limit():
    body = _res('calculus', expression=r'\lim_{x \to 0} \frac{\sin(x)}{x}', variable='x')
    assert '$' not in body and len(body) > 20


def test_calculus_plain_expression_simplifies():
    body = _res('calculus', expression=r'\frac{1}{x}+\frac{1}{x}', variable='x')
    assert '\\begin{align}' in body and '$' not in body


def test_auto_matrix_det():
    body = _res('auto', expression='det([[1,2],[3,4]])', variable='x')
    assert '-2' in body and '$' not in body


def test_auto_matrix_rank():
    body = _res('auto', expression='rank([[1,2],[2,4]])', variable='x')
    assert '$' not in body and len(body) > 20


def test_auto_matrix_inverse():
    body = _res('auto', expression='inverse([[1,0],[0,2]])', variable='x')
    assert '$' not in body and len(body) > 20


def test_auto_matrix_add():
    body = _res('auto', expression='[[1,0],[0,1]]+[[2,0],[0,2]]', variable='x')
    assert '$' not in body and len(body) > 20


def test_auto_calculus_derivative():
    body = _res('auto', expression=r'\frac{d}{dx}\left(x^2\right)', variable='x')
    assert '幂' in body and '$' not in body


def test_auto_plain_simplify():
    body = _res('auto', expression='x+x', variable='x')
    assert '\\begin{align}' in body and '$' not in body


def test_auto_function_forms():
    # function-style operators route to the right engine
    assert '\\cos' in _res('auto', expression='diff(sin(x),x)', variable='x')
    assert '泰勒' in _res('auto', expression='taylor(exp(x),x,0,4)', variable='x')
    assert '积分因子' in _res('auto', expression="ode(f'(x)+f(x)-exp(x),x)", variable='x')
    assert '牛顿' in _res('auto', expression='int(x^2,x,0,1)', variable='x')


def test_auto_matrix_svd_and_symbolic():
    assert '$' not in _res('auto', expression='svd([[1,0],[0,2]])', variable='x')
    sym = _res('auto', expression='msimplify(A*B*Inverse(B))', variable='x')
    assert '$' not in sym and len(sym) > 20


def test_iterated_integral_rectangular():
    # int_0^2 int_0^1 x*y dx dy = 1, inner result F(y)=y/2
    body = _res('iterated_integral', expression='x*y', variable='x', variable2='y',
                lower='0', upper='1', lower2='0', upper2='2')
    assert r'\frac{y}{2}' in body and '$' not in body


def test_power_series_interval_of_convergence():
    # sum x^n/n : R=1, converges at left endpoint (alternating) only -> [-1, 1)
    body = _res('power_series', expression='1/n', variable='n', point='0')
    assert 'R=1' in body and r'[-1' in body and '$' not in body


def test_power_series_infinite_radius():
    # sum x^n/n! : ratio limit 0 -> R=oo, whole real line
    body = _res('power_series', expression='1/factorial(n)', variable='n', point='0')
    assert r'R=\infty' in body and '$' not in body
