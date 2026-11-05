"""Numerical methods: definite-integration rules and IVP solvers (Euler/Heun/RK4)."""

from sympy import N, Symbol

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import (align, expr_text, get_var, render_note as note,
                                     to_count, to_num)


def numerical_integration(data):
    x = get_var(data)
    f = sympify(expr_text(data))
    a = to_num(data.get('lower'), 0)
    b = to_num(data.get('upper'), 1)
    n = to_count(data.get('order') or data.get('expr-depth'), 4)
    if not (a.is_number and b.is_number) or n <= 0:
        return False, '数值积分需要数值积分上下限'
    h = (b - a) / n
    ff = [f.subs(x, a + k * h) for k in range(n + 1)]
    trap = N((h / 2) * (ff[0] + 2 * sum(ff[1:-1]) + ff[-1]), 8)
    items = [(f, note("f(x)")), (trap, note(f"梯形法, n={n}"))]
    data['_auto_value'] = trap
    if n % 2 == 0:
        simp = N((h / 3) * (ff[0] + 4 * sum(ff[i] for i in range(1, n, 2))
                            + 2 * sum(ff[i] for i in range(2, n, 2)) + ff[-1]), 8)
        items.append((simp, note(f"辛普森法, n={n}")))
        data['_auto_value'] = simp  # prefer the higher-accuracy estimate
    return True, align(items)


def _ivp_setup(data):
    """Shared parsing for IVP solvers: y' = f(x,y), y(x0)=y0, integrate to x_end by n steps."""
    x = get_var(data)
    y = Symbol(data.get('yname') or 'y')
    f = sympify(expr_text(data))
    x0 = to_num(data.get('point'), 0)
    y0 = to_num(data.get('yvalue') or data.get('upper'), 1)
    x_end = to_num(data.get('target'), 1)
    steps = to_count(data.get('steps') or data.get('order') or data.get('expr-depth'), 4)
    if steps <= 0:
        return None
    h = (x_end - x0) / steps
    return x, y, f, x0, y0, x_end, steps, h


def solve_ivp(data, method='euler'):
    """Advance y' = f(x,y) from (x0,y0) with a fixed step, showing every step."""
    ctx = _ivp_setup(data)
    if ctx is None:
        return False, '数值 ODE 需要正的步数'
    x, y, f, x0, y0, x_end, steps, h = ctx
    xi, yi = x0, y0
    items = [(f, note("y' = f(x, y)"))]
    items.append((note(f"初值 y({x0})={y0}, 步长 h = ({x_end}-{x0})/{steps} = {N(h,6)}"), None))
    for k in range(steps):
        fx = f.subs({x: xi, y: yi})
        if method == 'euler':
            yi = yi + h * fx
        elif method == 'heun':
            fx2 = f.subs({x: xi + h, y: yi + h * fx})
            yi = yi + h / 2 * (fx + fx2)
        else:  # rk4
            k1 = fx
            k2 = f.subs({x: xi + h / 2, y: yi + h / 2 * k1})
            k3 = f.subs({x: xi + h / 2, y: yi + h / 2 * k2})
            k4 = f.subs({x: xi + h, y: yi + h * k3})
            yi = yi + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xi = xi + h
        label = {'euler': '欧拉法', 'heun': '改进欧拉(Heun)', 'rk4': '四阶 Runge-Kutta'}[method]
        items.append((N(yi, 6), note(f"第 {k+1} 步: x_{k+1} = {N(xi,6)}, {label} 得 y_{k+1}")))
    items.append((N(yi, 6), note(f"数值解 y({x_end}) ≈")))
    data['_auto_value'] = None  # a trajectory/table is not a scalar to embed in arithmetic
    return True, align(items)


def euler(data):
    return solve_ivp(data, 'euler')


def heun(data):
    return solve_ivp(data, 'heun')


def rk4(data):
    return solve_ivp(data, 'rk4')
