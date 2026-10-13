"""Regression tests for the applied-calculus / series / ODE / numerical ops."""

from routes.compute_service import start_compute


def _res(op, **kw):
    ok, res = start_compute(op, {'operation_type': op, **kw})
    assert ok, res
    return str(res).replace(' ', '')


def test_tangent_line():
    assert r'2x-1' in _res('tangent', expression='x**2', variable='x', point='1')


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
    assert '绝对收敛' in body and '$' not in body


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

