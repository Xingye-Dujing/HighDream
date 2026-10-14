"""Multivariable-adjacent differentiation: implicit, parametric, polar.

Parametric and polar slopes derive each coordinate's derivative with the core
DiffCalculator (real steps); polar area routes its integral through
IntegralCalculator; implicit differentiation shows the differentiated equation
then solves for y'.
"""

from sympy import Eq, Function, Symbol, cos, sin, solve, simplify, Integral

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import (align, derivative_steps, expr_text, get_var,
                                     integral_steps, render_note as note, steps_value, to_num)


def _last_value(steps):
    """The final concrete expression produced by a rule-driven ``*_steps`` row set."""
    values = [e for (e, _n) in steps if e is not None]
    return values[-1] if values else None


def implicit_diff(data):
    x = get_var(data)
    y = Function('y')(x)  # pylint: disable=not-callable
    expr = sympify(expr_text(data)).subs(Symbol('y'), y)
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    d_lhs = derivative_steps(str(eq.lhs), x)
    d_rhs = derivative_steps(str(eq.rhs), x)
    diffed = simplify(_last_value(d_lhs) - _last_value(d_rhs))
    from sympy import Derivative  # pylint: disable=import-outside-toplevel
    dy = Derivative(y, x)
    try:
        sol = solve(diffed, dy)
        result = sol[0] if sol else diffed
    except Exception:  # pylint: disable=broad-exception-caught
        result = diffed
    items = [(eq, note("方程"))]
    items.append((None, note("对等式左边求导 (链式法则, y 视为 y(x)):")))
    items += d_lhs
    items.append((None, note("对等式右边求导:")))
    items += d_rhs
    items.append((diffed, note("两边导数相减并整理, 令其为 0")))
    items.append((Eq(dy, result), note("解出 y'")))
    return True, align(items)


def parametric_diff(data):
    t = get_var(data)
    lines = expr_text(data).split('\n')
    x_str = lines[0]
    y_str = lines[1] if len(lines) > 1 else lines[0]
    x_t = sympify(x_str)
    y_t = sympify(y_str)
    dx_steps = derivative_steps(x_str, t)
    dy_steps = derivative_steps(y_str, t)
    dx_dt = _last_value(dx_steps)
    dy_dt = _last_value(dy_steps)
    items = [(Eq(Symbol('x'), x_t), note("x(t)"))] + dx_steps
    items.append((Eq(Symbol('y'), y_t), note("y(t)")))
    items += dy_steps
    dydx = simplify(dy_dt / dx_dt)
    items.append((Eq(Symbol('dy_dx'), dydx), note("dy/dx=(dy/dt)/(dx/dt)")))
    return True, align(items)


def polar_area(data):
    theta = Symbol(data.get('variable', 'theta') or 'theta')
    r = sympify(expr_text(data))
    a = to_num(data.get('lower'), 0)
    b = to_num(data.get('upper'), sympify('pi'))
    integrand = r ** 2 / 2
    items = [(r, note("半径函数")), (integrand, note("面积被积函数 (1/2)r^2")),
             (Integral(integrand, (theta, a, b)), note("建立极坐标面积积分"))]
    rows = integral_steps(integrand, theta, a, b)
    items += rows
    data['_auto_value'] = steps_value(rows)
    return True, align(items)


def polar_slope(data):
    theta = Symbol(data.get('variable', 'theta') or 'theta')
    r_str = expr_text(data)
    r = sympify(r_str)
    x_str = str(r * cos(theta))
    y_str = str(r * sin(theta))
    dx_steps = derivative_steps(x_str, theta)
    dy_steps = derivative_steps(y_str, theta)
    dx_dt = _last_value(dx_steps)
    dy_dt = _last_value(dy_steps)
    items = [(r, note("r(θ)"))]
    items += dx_steps
    items += dy_steps
    dydx = simplify(dy_dt / dx_dt)
    items.append((Eq(Symbol('slope'), dydx), note("极坐标斜率 dy/dx")))
    return True, align(items)
