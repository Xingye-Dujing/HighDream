"""Applications of differentiation, following the core engine (reuses DiffCalculator).

Each op shows the actual rule-driven derivative derivation, then applies the
relevant formula (tangent line, linearization, critical-point test, Newton's
iteration, mean-value theorem), so the intermediate steps are as complete as the
main calculus output.
"""

from sympy import Eq, Symbol, simplify, solve

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import align, derivative_steps, expr_text, get_var
from domains.applied._common import render_note as note, to_num


def tangent(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    a = to_num(data.get('point'), 0)
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    fp = f.diff(x)
    slope = simplify(fp.subs(x, a))
    y0 = simplify(f.subs(x, a))
    items.append((slope, note("f'(a) 即切线斜率")))
    items.append((Eq(Symbol('y'), simplify(y0 + slope * (x - a))), note("切线方程")))
    return True, align(items)


def linearization(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    a = to_num(data.get('point'), 0)
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    fp = f.diff(x)
    lin = simplify(f.subs(x, a) + fp.subs(x, a) * (x - a))
    items.append((Eq(Symbol('L'), lin), note("线性化(切线近似) L(x)")))
    return True, align(items)


def critical_points(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    fp = f.diff(x)
    try:
        crit = sorted(solve(Eq(fp, 0), x), key=lambda v: (not v.is_real, str(v)))
    except Exception:  # pylint: disable=broad-exception-caught
        crit = []
    fpp = f.diff(x, 2)
    if not crit:
        items.append((None, note("f'(x)=0 无实临界点")))
        return True, align(items)
    for c in crit:
        kind = fpp.subs(x, c)
        label = ('极小值' if kind.is_positive else '极大值' if kind.is_negative else '二阶导为零, 需进一步判定')
        items.append((Eq(x, c), note(f"临界点; 二阶导 f''={kind} => {label}")))
    return True, align(items)


def newton(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    x0 = to_num(data.get('point'), 1)
    iters = int(data.get('order') or data.get('expr-depth') or 4)
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    items.append((Eq(Symbol('x_0'), x0), note("初值")))
    cur = x0
    for k in range(1, iters + 1):
        nxt = simplify(cur - f.subs(x, cur) / f.diff(x).subs(x, cur))
        items.append((Eq(Symbol(f'x_{k}'), nxt), note("牛顿迭代 x_{k}=x_{k-1}-f/f'")))
        cur = nxt
    return True, align(items)


def mean_value(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    a = to_num(data.get('lower'), 0)
    b = to_num(data.get('upper'), 1)
    secant = simplify((f.subs(x, b) - f.subs(x, a)) / (b - a))
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    items.append((Eq(Symbol('k'), secant), note("割线斜率 (f(b)-f(a))/(b-a)")))
    try:
        cands = solve(Eq(f.diff(x), secant), x)
    except Exception:  # pylint: disable=broad-exception-caught
        cands = []
    if not cands:
        items.append((None, note("未能解出满足条件的 c")))
    for i, c in enumerate(cands):
        items.append((Eq(Symbol('c'), c), note(f"中值点 c_{i + 1} (需 a<c<b)")))
    return True, align(items)
