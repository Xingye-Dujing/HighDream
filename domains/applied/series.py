"""Series, Taylor expansion, convergence tests, and improper integrals.

Improper integrals reuse the IntegralCalculator rule engine; Taylor lists each
order's derivative and coefficient before assembling the polynomial; convergence
applies the standard tests (n-th term / ratio / root / absolute / integral) as
explicit steps.
"""
# pylint: disable=duplicate-code

from sympy import (Abs, Eq, Integral, Symbol, expand, factorial, latex, limit,
                   oo, simplify, summation, Sum)

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import (align, derivative_value, expr_text, get_var,
                                     integral_steps, render_note as note, to_num)
from domains.applied._convergence import convergence_rows


def taylor(data):
    x = get_var(data)
    f = sympify(expr_text(data))
    x0 = to_num(data.get('point'), 0)
    order = int(data.get('order') or data.get('expr-depth') or 6)
    # Build each order's derivative by feeding the previous result back through the
    # rule engine (never a black-box f.diff(x, k)).
    derivs = [f]
    for _k in range(1, order + 1):
        derivs.append(derivative_value(str(derivs[-1]), x, 1))
    items = [(f, note("f(x)"))]
    terms = []
    for k in range(order + 1):
        dk = derivs[k]
        coef = simplify(dk.subs(x, x0) / factorial(k))
        terms.append(coef * (x - x0) ** k)
        if k > 0:
            items.append((simplify(dk), note(f"第 {k} 阶导数 (规则引擎)")))
        items.append((coef, note(f"第 {k} 阶系数 a_{k}")))
    items.append((expand(sum(terms)), note(f"泰勒/麦克劳林多项式 (到 {order} 阶)")))
    return True, align(items)


def taylor_error(data):
    x = get_var(data)
    f = sympify(expr_text(data))
    x0 = to_num(data.get('point'), 0)
    order = int(data.get('order') or data.get('expr-depth') or 6)
    m = order + 1
    deriv = simplify(derivative_value(expr_text(data), x, m))
    xi = Symbol('xi')
    remainder = deriv.subs(x, xi) / factorial(m) * (x - x0) ** m
    return True, align([
        (f, note("f(x)")),
        (deriv, note(f"n+1 = {m} 阶导数 (规则引擎)")),
        (Eq(Symbol('R_n'), remainder), note("拉格朗日余项 (xi 介于 x0 与 x 之间)")),
    ])


def series_convergence(data):
    n = Symbol(data.get('variable', 'n') or 'n')
    a = sympify(expr_text(data))
    items, verdict = convergence_rows(a, n)
    if verdict == 'unknown':
        items.append((None, note("标准判别法未能定论 (可尝试比较/积分判别)")))
    try:
        closed = summation(a, (n, 1, oo))
        items.append((Eq(Symbol('sum_a_n'), closed), note("闭式求和(若存在), 供参考")))
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return True, align(items)


def radius_of_convergence(data):
    n = Symbol('n')
    coef = sympify(expr_text(data))
    l_val = simplify(limit(Abs(coef.subs(n, n + 1) / coef), n, oo))
    if l_val == 0:
        radius = oo
    elif l_val == oo:
        radius = 0
    else:
        radius = 1 / l_val if l_val.is_number else Symbol('invL')
    return True, align([
        (coef, note("幂级数系数")),
        (Eq(Symbol('L'), l_val), note("比值极限")),
        (Eq(Symbol('R'), radius), note("收敛半径 R=1/L")),
    ])


def improper_integral(data):
    x = get_var(data)
    f = sympify(expr_text(data))
    a = to_num(data.get('lower'), 0)
    b = to_num(data.get('upper'), oo)
    items = [(Integral(f, (x, a, b)), note("反常积分 (先求原函数, 再代入上下限)"))]
    steps = integral_steps(f, x, a, b)
    items += steps
    real = [e for (e, _n) in steps]
    if real:
        last = real[-1]
        verdict = ('收敛' if (hasattr(last, 'is_finite') and last.is_finite) else '发散(或不存在)')
        items.append((None, note(f"结论: 该反常积分{verdict}")))
    return True, align(items)


def _bound_latex(value):
    if value == oo:
        return r"\infty"
    if value == -oo:
        return r"-\infty"
    return latex(value)


def _verdict_text(value):
    if value is True:
        return "收敛"
    if value is False:
        return "发散"
    return "无法判定"


def _endpoint_converges(coef, n, radius, sign):
    """Whether Σ c_n (sign·R)^n converges, via the rule-based test engine.

    Returns True/False, or None when the tests cannot decide.
    """
    if radius in (oo, 0):
        return None
    term = coef * (sign * radius) ** n
    _rows, verdict = convergence_rows(term, n)
    return {'converge': True, 'diverge': False}.get(verdict)


def power_series(data):
    """Power series Σ c_n (x - x0)^n: radius, endpoint verdicts, interval and
    termwise differentiation / integration (radius unchanged)."""
    n = Symbol(data.get('variable', 'n') or 'n')
    coef = sympify(expr_text(data))
    x0 = to_num(data.get('point'), 0)
    x = Symbol('x')

    l_val = simplify(limit(Abs(coef.subs(n, n + 1) / coef), n, oo))
    items = [
        (Sum(coef * (x - x0) ** n, (n, 1, oo)), note("幂级数 $\\sum c_n (x-x_0)^n$")),
        (Eq(Symbol('L'), l_val), note(r"比值极限 $L=\lim_{n\to\infty}\left|\frac{c_{n+1}}{c_n}\right|$")),
    ]

    if l_val == 0:
        radius = oo
    elif l_val == oo:
        radius = 0
    elif l_val.is_number:
        radius = simplify(1 / l_val)
    else:
        radius = None

    if radius is None:
        items.append((None, note("比值极限不是具体数值, 无法确定收敛半径")))
        return True, align(items)

    items.append((Eq(Symbol('R'), radius), note("收敛半径 $R=1/L$")))

    if radius == oo:
        items.append((None, note("收敛域为全体实数 $(-\\infty,\\ +\\infty)$")))
    else:
        left = _endpoint_converges(coef, n, radius, -1)
        right = _endpoint_converges(coef, n, radius, 1)
        items.append((Sum(coef * (-radius) ** n, (n, 1, oo)),
                      note(f"左端点 $x_0-R$ 代入后{_verdict_text(left)}")))
        items.append((Sum(coef * radius ** n, (n, 1, oo)),
                      note(f"右端点 $x_0+R$ 代入后{_verdict_text(right)}")))
        lo, hi = simplify(x0 - radius), simplify(x0 + radius)
        interval = rf"({_bound_latex(lo)},\ {_bound_latex(hi)})"
        if left is True:
            interval = r"[" + interval[1:]
        if right is True:
            interval = interval[:-1] + r"]"
        items.append((None, note(rf"收敛域 ${interval}$")))

    items.append((Sum(n * coef * (x - x0) ** (n - 1), (n, 1, oo)),
                  note("逐项求导 $\\sum n c_n (x-x_0)^{n-1}$, 收敛半径不变")))
    items.append((Sum(coef * (x - x0) ** (n + 1) / (n + 1), (n, 1, oo)),
                  note("逐项积分 $\\sum \\frac{c_n}{n+1}(x-x_0)^{n+1}$, 收敛半径不变")))
    return True, align(items)
