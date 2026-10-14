"""Applications of differentiation, following the core engine (reuses DiffCalculator).

Each op shows the actual rule-driven derivative derivation, then applies the
relevant formula (tangent line, linearization, critical-point test, Newton's
iteration, mean-value theorem), so the intermediate steps are as complete as the
main calculus output.
"""

from sympy import Eq, N, Symbol, simplify, solve

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import align, derivative_steps, derivative_value, expr_text, get_var
from domains.applied._common import render_note as note, to_num


def _as_float(value):
    """Real float of a sympy value, or None if it is not a real number."""
    try:
        c = complex(N(value))
    except (TypeError, ValueError):
        return None
    if abs(c.imag) < 1e-12:
        return c.real
    return None


def tangent(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    a = to_num(data.get('point'), 0)
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    fp = derivative_value(f_str, x, 1)
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
    fp = derivative_value(f_str, x, 1)
    lin = simplify(f.subs(x, a) + fp.subs(x, a) * (x - a))
    items.append((Eq(Symbol('L'), lin), note("线性化(切线近似) L(x)")))
    return True, align(items)


def critical_points(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    fp = derivative_value(f_str, x, 1)
    try:
        crit = sorted(solve(Eq(fp, 0), x), key=lambda v: (not v.is_real, str(v)))
    except Exception:  # pylint: disable=broad-exception-caught
        crit = []
    fpp = derivative_value(f_str, x, 2)
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
        nxt = simplify(cur - f.subs(x, cur) / derivative_value(f_str, x, 1).subs(x, cur))
        items.append((Eq(Symbol(f'x_{k}'), nxt), note("牛顿迭代 x_{k}=x_{k-1}-f/f'")))
        cur = nxt
    return True, align(items)


def bisection(data):
    x = get_var(data)
    f = sympify(expr_text(data))
    lo = to_num(data.get('lower'), 0)
    hi = to_num(data.get('upper'), 1)
    iters = int(data.get('order') or data.get('steps') or 8)
    flo, fhi = _as_float(f.subs(x, lo)), _as_float(f.subs(x, hi))
    if flo is None or fhi is None or flo * fhi > 0:
        return False, '二分法需要异号端点: f(a)·f(b) < 0'
    items = [(f, note('f(x)')),
             (None, note(f'端点 f({lo})={N(f.subs(x, lo), 6)}, f({hi})={N(f.subs(x, hi), 6)}, 异号'))]
    mid = (lo + hi) / 2
    for k in range(1, iters + 1):
        mid = (lo + hi) / 2
        fm = _as_float(f.subs(x, mid))
        items.append((Eq(Symbol(f'm_{k}'), N(mid, 6)), note(f'中点, f(m)={N(f.subs(x, mid), 6)}')))
        if fm is None:
            break
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    items.append((N(mid, 6), note('根的近似 (区间中点)')))
    return True, align(items)


def secant(data):
    x = get_var(data)
    f = sympify(expr_text(data))
    x_prev = to_num(data.get('point'), 0)
    x_cur = to_num(data.get('x1'), 1)
    iters = int(data.get('order') or data.get('steps') or 6)
    items = [(f, note('f(x)')), (None, note(f'初值 x_0={N(x_prev,6)}, x_1={N(x_cur,6)}'))]
    for k in range(1, iters + 1):
        f_prev, f_cur = _as_float(f.subs(x, x_prev)), _as_float(f.subs(x, x_cur))
        if f_prev is None or f_cur is None or abs(f_cur - f_prev) < 1e-15:
            break
        x_new = simplify(x_cur - f.subs(x, x_cur) * (x_cur - x_prev)
                         / (f.subs(x, x_cur) - f.subs(x, x_prev)))
        items.append((Eq(Symbol(f'x_{k}'), N(x_new, 6)),
                      note('割线法 x_k = x_{k-1} - f(x_{k-1})(x_{k-1}-x_{k-2})/(f(x_{k-1})-f(x_{k-2}))')))
        x_prev, x_cur = x_cur, x_new
    items.append((N(x_cur, 6), note('根的近似')))
    return True, align(items)


def fixed_point(data):
    x = get_var(data)
    g = sympify(expr_text(data))
    cur = to_num(data.get('point'), 0)
    iters = int(data.get('order') or data.get('steps') or 8)
    items = [(g, note('迭代函数 g(x)')), (Eq(Symbol('x_0'), cur), note('初值'))]
    for k in range(1, iters + 1):
        cur = simplify(g.subs(x, cur))
        items.append((Eq(Symbol(f'x_{k}'), N(cur, 6)), note('不动点迭代 x_k = g(x_{k-1})')))
    items.append((N(cur, 6), note('不动点近似')))
    return True, align(items)


def mean_value(data):
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    a = to_num(data.get('lower'), 0)
    b = to_num(data.get('upper'), 1)
    secant = simplify((f.subs(x, b) - f.subs(x, a)) / (b - a))
    data['_auto_value'] = secant
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    items.append((Eq(Symbol('k'), secant), note("割线斜率 (f(b)-f(a))/(b-a)")))
    try:
        cands = solve(Eq(derivative_value(f_str, x, 1), secant), x)
    except Exception:  # pylint: disable=broad-exception-caught
        cands = []
    if not cands:
        items.append((None, note("未能解出满足条件的 c")))
    for i, c in enumerate(cands):
        items.append((Eq(Symbol('c'), c), note(f"中值点 c_{i + 1} (需 a<c<b)")))
    return True, align(items)
