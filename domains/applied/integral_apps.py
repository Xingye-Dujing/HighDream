"""Integral applications: arc length, solid-of-revolution volume, surface area.

Follows the core engine: the derivative f'(x) comes from DiffCalculator, the
resulting definite integral is expanded by IntegralCalculator, so both the
formula setup and the integration show complete steps.
"""

from sympy import Integral, Symbol, pi, simplify, sqrt

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import (align, derivative_steps, expr_text, get_var,
                                     integral_steps, render_note as note, to_num)


def _setup(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    a = to_num(data.get('lower'), 0)
    b = to_num(data.get('upper'), 1)
    return x, f_str, f, a, b


def arc_length(data):
    x, f_str, f, a, b = _setup(data)
    integrand = simplify(sqrt(1 + f.diff(x) ** 2))
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    items.append((integrand, note("弧长被积函数 sqrt(1+(f')^2)")))
    items.append((Integral(integrand, (x, a, b)), note("建立弧长积分")))
    items += integral_steps(integrand, x, a, b)
    return True, align(items)


def volume_disk(data):
    x, _f_str, f, a, b = _setup(data)
    integrand = pi * f ** 2
    items = [(f, note("f(x)"))]
    items.append((integrand, note("圆盘截面 (绕 x 轴)")))
    items.append((Integral(integrand, (x, a, b)), note("建立体积积分")))
    items += integral_steps(integrand, x, a, b)
    return True, align(items)


def volume_shell(data):
    x, _f_str, f, a, b = _setup(data)
    integrand = 2 * pi * x * f
    items = [(f, note("f(x)"))]
    items.append((integrand, note("柱壳 (绕 y 轴)")))
    items.append((Integral(integrand, (x, a, b)), note("建立柱壳体积积分")))
    items += integral_steps(integrand, x, a, b)
    return True, align(items)


def surface_area(data):
    x, f_str, f, a, b = _setup(data)
    integrand = simplify(2 * pi * f * sqrt(1 + f.diff(x) ** 2))
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    items.append((integrand, note("旋转曲面被积函数 2π f sqrt(1+(f')^2)")))
    items.append((Integral(integrand, (x, a, b)), note("建立面积积分")))
    items += integral_steps(integrand, x, a, b)
    return True, align(items)


def iterated_integral(data):
    """Rectangular iterated (double) integral ∫_c^d ∫_a^b f dx dy.

    The inner integral over x keeps y as a parameter, so it is evaluated with the
    definite-integral engine directly (the single-variable rule engine rejects a
    two-symbol integrand). The resulting F(y) is then expanded step-by-step by the
    core IntegralCalculator for the outer integral over y.
    """
    x = get_var(data)
    y = Symbol(data.get('variable2', 'y') or 'y')
    f = sympify(expr_text(data))
    a, b = to_num(data.get('lower'), 0), to_num(data.get('upper'), 1)
    c, d = to_num(data.get('lower2'), 0), to_num(data.get('upper2'), 1)

    inner_val = simplify(Integral(f, (x, a, b)).doit())
    items = [
        (Integral(f, (x, a, b), (y, c, d)), note("累次积分 (矩形区域)")),
        (Integral(f, (x, a, b)), note("先对 x 积分, y 视为常数")),
        (inner_val, note(r"内层结果 $F(" + str(y) + r")$")),
    ]
    items += integral_steps(inner_val, y, c, d)
    return True, align(items)
