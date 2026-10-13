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
    assert r'\pi' in body and r'V=' in body


def test_taylor():
    assert r'x^{4}' in _res('taylor', expression='exp(x)', variable='x', point='0', order='4')


def test_improper_converges():
    assert r'=1' in _res('improper_integral', expression='exp(-x)', variable='x',
                         lower='0', upper='oo')


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
