"""Multivariable-adjacent differentiation: implicit, parametric, polar.

Parametric and polar slopes derive each coordinate's derivative with the core
DiffCalculator (real steps); polar area routes its integral through
IntegralCalculator; implicit differentiation shows the differentiated equation
then solves for y'.
"""

from sympy import Eq, Function, Symbol, cos, sin, solve, simplify, Integral

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import (align, derivative_steps, expr_text, get_var,
                                     integral_steps, render_note as note, to_num)


def implicit_diff(data):
    x = get_var(data)
    y = Function('y')(x)  # pylint: disable=not-callable
    expr = sympify(expr_text(data)).subs(Symbol('y'), y)
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    diffed = simplify(eq.lhs.diff(x) - eq.rhs.diff(x))
    from sympy import Derivative  # pylint: disable=import-outside-toplevel
    dy = Derivative(y, x)
    try:
        sol = solve(diffed, dy)
        result = sol[0] if sol else diffed
    except Exception:  # pylint: disable=broad-exception-caught
        result = diffed
    return True, align([
        (eq, note("方程")), (diffed, note("两边对 x 求导")),
        (Eq(dy, result), note("解出 y'")),
    ])


def parametric_diff(data):
    t = get_var(data)
    lines = expr_text(data).split('\n')
    x_str = lines[0]
    y_str = lines[1] if len(lines) > 1 else lines[0]
    x_t = sympify(x_str)
    y_t = sympify(y_str)
    items = [(Eq(Symbol('x'), x_t), note("x(t)"))] + derivative_steps(x_str, t)
    items.append((Eq(Symbol('y'), y_t), note("y(t)")))
    items += derivative_steps(y_str, t)
    dydx = simplify(y_t.diff(t) / x_t.diff(t))
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
    items += integral_steps(integrand, theta, a, b)
    return True, align(items)


def polar_slope(data):
    theta = Symbol(data.get('variable', 'theta') or 'theta')
    r_str = expr_text(data)
    r = sympify(r_str)
    x_str = str(r * cos(theta))
    y_str = str(r * sin(theta))
    items = [(r, note("r(θ)"))]
    items += derivative_steps(x_str, theta)
    items += derivative_steps(y_str, theta)
    dydx = simplify(sympify(y_str).diff(theta) / sympify(x_str).diff(theta))
    items.append((Eq(Symbol('slope'), dydx), note("极坐标斜率 dy/dx")))
    return True, align(items)
