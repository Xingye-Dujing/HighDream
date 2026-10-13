"""Applied-calculus / series / ODE / numerical operations (SymPy-backed).

Each handler returns ``(success, latex)`` where ``latex`` is a ``\\begin{align}``
block whose mathematical content is proper LaTeX (rendered by MathJax) and whose
prose labels use ``\\text{}``. No raw unicode math symbols.

Integral-based ops reuse the step-by-step ``IntegralCalculator`` so the derivation
is broken into intermediate steps like the core calculus engine.
"""

from sympy import (Abs, Add, Derivative, Eq, Function, I, Integral, N, Symbol,
                   arg, cos, dsolve, exp, expand, factorial, im, latex, limit, pi,
                   re, simplify, sin, solve, summation, sqrt, sympify, oo)


def _var(data):
    return Symbol(data.get('variable', 'x') or 'x')


def _num(value, default=None):
    if value is None or str(value).strip() == '':
        return default
    return sympify(str(value))


def _p(text):
    """Build a MathJax fragment: prose -> \\textrm{}, ``$...$`` parts -> live LaTeX."""
    parts = str(text).split('$')
    out = []
    for i, seg in enumerate(parts):
        if seg == '':
            continue
        out.append(seg if i % 2 == 1 else r"\textrm{" + seg + r"}")
    return ''.join(out)


def _align(items):
    """Build an align block. items: list of (body_or_None, note_latex)."""
    lines = []
    first = True
    for body, note in items:
        if body is not None:
            body_latex = body if isinstance(body, str) else latex(body)
            prefix = '' if first else '&='
            first = False
            line = prefix + body_latex
        else:
            line = '' if first else '&\\text{}'
            first = False
        if note:
            line += r" \quad " + note
        lines.append(line)
    return r"\begin{align}" + r"\\".join(lines) + r"\end{align}"


def _expr_text(data):
    return data.get('expression', '')


def _integral_items(integrand, x, lower=None, upper=None):
    """Reuse the step-by-step integral engine, returning [(expr, note), ...]."""
    # pylint: disable=import-outside-toplevel
    from domains import IntegralCalculator
    kwargs = {}
    if upper is not None:
        kwargs['upper'] = upper
        kwargs['lower'] = 0 if lower is None else lower
    steps, explanations = IntegralCalculator().compute_list(str(integrand), x, **kwargs)
    return [(expr, _p(reason)) for expr, reason in zip(steps, explanations)
            if reason != ''] or [(steps[-1], '')]


# ---------------------------------------------------------------- applications
def tangent_line(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('point'), 0)
    slope = simplify(f.diff(x).subs(x, a))
    y0 = simplify(f.subs(x, a))
    line = simplify(y0 + slope * (x - a))
    return True, _align([
        (f, _p("f(x)")), (f.diff(x), _p("求导")),
        (Eq(Symbol('y'), line), _p("切线方程")),
    ])


def linearization(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('point'), 0)
    lin = simplify(f.subs(x, a) + f.diff(x).subs(x, a) * (x - a))
    return True, _align([
        (f, _p("f(x)")), (Eq(Symbol('L'), lin), _p(f"在 x={a} 处的线性化(切线近似)")),
    ])


def critical_points(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    fp = f.diff(x)
    try:
        crit = sorted(solve(Eq(fp, 0), x), key=lambda v: (not v.is_real, str(v)))
    except Exception:  # pylint: disable=broad-exception-caught
        crit = []
    items = [(f, _p("f(x)")), (fp, _p("一阶导数 f'(x)"))]
    if not crit:
        items.append((None, _p("f'(x)=0 无实临界点")))
        return True, _align(items)
    fpp = f.diff(x, 2)
    for c in crit:
        kind = fpp.subs(x, c)
        label = ('极小' if kind.is_positive else '极大' if kind.is_negative else '待定(二阶=0)')
        items.append((Eq(x, c), _p(f"临界点; 二阶导判定: {label}")))
    return True, _align(items)


def newton_method(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    x0 = _num(data.get('point'), 1)
    iters = int(data.get('order') or data.get('expr-depth') or 4)
    items = [(Eq(Symbol('x_0'), x0), _p("初值"))]
    cur = x0
    for k in range(1, iters + 1):
        nxt = simplify(cur - f.subs(x, cur) / f.diff(x).subs(x, cur))
        items.append((Eq(Symbol(f'x_{k}'), nxt), _p("牛顿迭代")))
        cur = nxt
    return True, _align(items)


# ---------------------------------------------------------------- arc/volume
def _interval(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('lower'), 0)
    b = _num(data.get('upper'), 1)
    return x, f, a, b


def arc_length(data):
    x, f, a, b = _interval(data)
    fp = f.diff(x)
    integrand = simplify(sqrt(1 + fp ** 2))
    items = [(f, _p("f(x)")), (fp, _p("f'(x)")),
             (integrand, _p("弧长被积函数"))]
    items += _integral_items(integrand, x, a, b)
    return True, _align(items)


def volume_disk(data):
    x, f, a, b = _interval(data)
    integrand = pi * f ** 2
    items = [(f, _p("f(x)")), (integrand, _p("圆盘截面(绕 x 轴旋转)"))]
    items += _integral_items(integrand, x, a, b)
    return True, _align(items)


def surface_area(data):
    x, f, a, b = _interval(data)
    integrand = simplify(2 * pi * f * sqrt(1 + f.diff(x) ** 2))
    items = [(f, _p("f(x)")), (integrand, _p("旋转曲面被积函数"))]
    items += _integral_items(integrand, x, a, b)
    return True, _align(items)


# ---------------------------------------------------------------- implicit/param
def implicit_diff(data):
    x = _var(data)
    y = Function('y')(x)  # pylint: disable=not-callable
    expr = sympify(_expr_text(data)).subs(Symbol('y'), y)
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    diffed = simplify(eq.lhs.diff(x) - eq.rhs.diff(x))
    dy = Derivative(y, x)
    try:
        sol = solve(diffed, dy)
        result = sol[0] if sol else diffed
    except Exception:  # pylint: disable=broad-exception-caught
        result = diffed
    return True, _align([
        (eq, _p("两边对 x 求导")), (Eq(dy, result), _p("解出 y'")),
    ])


def parametric_diff(data):
    t = Symbol(data.get('variable', 't') or 't')
    lines = _expr_text(data).split('\n')
    x_t = sympify(lines[0])
    y_t = sympify(lines[1]) if len(lines) > 1 else sympify(lines[0])
    dydx = simplify(y_t.diff(t) / x_t.diff(t))
    return True, _align([
        (Eq(Symbol('x'), x_t), _p("x(t)")), (Eq(Symbol('y'), y_t), _p("y(t)")),
        (Eq(Symbol('y__prime'), dydx), _p("参数方程斜率")),
    ])


def polar_area(data):
    theta = Symbol(data.get('variable', 'theta') or 'theta')
    r = sympify(_expr_text(data))
    a = _num(data.get('lower'), 0)
    b = _num(data.get('upper'), pi)
    integrand = r ** 2 / 2
    items = [(r, _p("半径函数")), (integrand, _p("面积被积函数"))]
    items += _integral_items(integrand, theta, a, b)
    return True, _align(items)


def polar_slope(data):
    theta = Symbol(data.get('variable', 'theta') or 'theta')
    r = sympify(_expr_text(data))
    x_of_t = r * cos(theta)
    y_of_t = r * sin(theta)
    dydx = simplify(y_of_t.diff(theta) / x_of_t.diff(theta))
    return True, _align([
        (r, _p("r(θ)")),
        (Eq(Symbol('slope'), dydx), _p("极坐标曲线斜率 dy/dx")),
    ])


# ---------------------------------------------------------------- series/taylor
def taylor_series(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    x0 = _num(data.get('point'), 0)
    order = int(data.get('order') or data.get('expr-depth') or 6)
    items = [(f, _p("f(x)"))]
    terms = []
    for k in range(order + 1):
        dk = f if k == 0 else f.diff(x, k)
        val = simplify(dk.subs(x, x0))
        coef = simplify(val / factorial(k))
        terms.append(coef * (x - x0) ** k)
        items.append((simplify(dk), _p(f"第 {k} 阶导数")))
        items.append((coef, _p(f"第 {k} 阶系数")))
    items.append((expand(Add(*terms)), _p(f"泰勒/麦克劳林多项式 (到 {order} 阶)")))
    return True, _align(items)


def series_convergence(data):
    n = Symbol(data.get('variable', 'n') or 'n')
    a = sympify(_expr_text(data))
    items = [(a, _p("一般项"))]
    term_lim = limit(a, n, oo)
    items.append((Eq(Symbol('lim_a_n'), term_lim), _p("第 n 项极限")))
    if term_lim not in (0,):
        items.append((None, _p("第 n 项不趋于 0, 级数发散")))
        return True, _align(items)
    ratio = simplify(limit(abs(a.subs(n, n + 1) / a), n, oo))
    items.append((Eq(Symbol('L_ratio'), ratio), _p("比值判别")))
    if ratio.is_number:
        verdict = ('收敛' if ratio < 1 else '发散' if ratio > 1 else '比值判别不确定')
        items.append((None, _p(f"比值判别得 L={ratio}, 故级数{verdict}")))
    root = simplify(limit(Abs(a) ** (1 / n), n, oo))
    items.append((Eq(Symbol('rho_root'), root), _p("根值判别")))
    if root.is_number:
        r_verdict = ('收敛' if root < 1 else '发散' if root > 1 else '根值判别不确定')
        items.append((None, _p(f"根值判别得, 故级数{r_verdict}")))
    try:
        abs_sum = summation(Abs(a), (n, 1, oo))
        items.append((Eq(Symbol('sum_abs_a_n'), abs_sum), _p("绝对收敛: Σ|a_n|")))
        if abs_sum.is_finite:
            items.append((None, _p("Σ|a_n| 收敛, 故级数绝对收敛")))
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    try:
        tt = Symbol('t')
        integral_val = Integral(Abs(a).subs(n, tt), (tt, 1, oo)).doit()
        items.append((integral_val, _p("积分判别法 (积分与级数同敛散)")))
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    closed = summation(a, (n, 1, oo))
    items.append((Eq(Symbol('sum_a_n'), closed), _p("闭式求和(若存在)")))
    return True, _align(items)


def improper_integral(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('lower'), 0)
    b = _num(data.get('upper'), oo)
    items = [(Integral(f, (x, a, b)), _p("反常积分(先求原函数, 再代入上下限)"))]
    steps = _integral_items(f, x, a, b)
    items += steps
    real_steps = [e for (e, _n) in steps]
    if real_steps:
        last = real_steps[-1]
        verdict = ('收敛' if (hasattr(last, 'is_finite') and last.is_finite) else '发散(或不存在)')
        items.append((None, _p(f"结论: 该反常积分{verdict}")))
    return True, _align(items)


# ---------------------------------------------------------------- ODE
def solve_ode(data):
    x = _var(data)
    f = Function('f')(x)  # pylint: disable=not-callable
    raw = _expr_text(data)
    raw = (raw.replace("f'(x)", 'Derivative(f(x), x)')
              .replace("f''(x)", 'Derivative(f(x), (x, 2))'))
    expr = sympify(raw)
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    try:
        sol = dsolve(eq, f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'ODE 无法求解: {exc}'
    return True, _align([
        (eq, _p("微分方程")), (sol, _p("dsolve 通解")),
    ])


# ---------------------------------------------------------------- numerical
def numerical_integration(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('lower'), 0)
    b = _num(data.get('upper'), 1)
    n = int(data.get('order') or data.get('expr-depth') or 4)
    if a.is_number and b.is_number and n > 0:
        h = (b - a) / n
        ff = [f.subs(x, a + k * h) for k in range(n + 1)]
        trap = N((h / 2) * (ff[0] + 2 * sum(ff[1:-1]) + ff[-1]), 8)
        items = [(f, _p("f(x)")), (trap, _p(f"梯形法, n={n}"))]
        if n % 2 == 0:
            simp = N((h / 3) * (ff[0] + 4 * sum(ff[i] for i in range(1, n, 2))
                                + 2 * sum(ff[i] for i in range(2, n, 2)) + ff[-1]), 8)
            items.append((simp, _p(f"辛普森法, n={n}")))
        return True, _align(items)
    return False, '数值积分需要数值积分上下限'


# ---------------------------------------------------------------- complex numbers
def complex_simplify(data):
    z = sympify(_expr_text(data))
    s = simplify(expand(z, complex=True))
    return True, _align([
        (z, _p("原复数式")), (re(s), _p("实部 Re(z)")),
        (im(s), _p("虚部 Im(z)")), (s, _p("化简为 a+bi 形式")),
    ])


def complex_polar(data):
    z = sympify(_expr_text(data))
    r = simplify(Abs(z))
    theta = arg(z)
    return True, _align([
        (z, _p("复数 z")), (r, _p("模")), (theta, _p("辐角")),
        (r * exp(I * theta), _p("指数形式")),
    ])


def complex_roots(data):
    z = _var(data)
    expr = sympify(_expr_text(data))
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    try:
        sol = solve(eq, z)
    except Exception:  # pylint: disable=broad-exception-caught
        sol = []
    if not sol:
        return False, '无法在复数域求解该方程的根'
    items = [(eq, _p("求解 (复数域)"))]
    for i, s in enumerate(sol):
        items.append((Eq(z, simplify(expand(s, complex=True))), _p(f"根 {i + 1}")))
    return True, _align(items)


# ---------------------------------------------------------------- MVT / radius
def mean_value(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('lower'), 0)
    b = _num(data.get('upper'), 1)
    secant = simplify((f.subs(x, b) - f.subs(x, a)) / (b - a))
    fp = f.diff(x)
    try:
        cands = solve(Eq(fp, secant), x)
    except Exception:  # pylint: disable=broad-exception-caught
        cands = []
    items = [(f, _p("f(x)")), (Eq(Symbol('k'), secant), _p("割线斜率")),
             (fp, _p("f'(x)"))]
    if not cands:
        items.append((None, _p("未能解出满足条件的 c")))
    for i, c in enumerate(cands):
        items.append((Eq(Symbol('c'), c), _p(f"中值点 c_{i + 1} (需 a<c<b)")))
    return True, _align(items)


def radius_of_convergence(data):
    n = Symbol('n')
    coef = sympify(_expr_text(data))
    l_val = simplify(limit(Abs(coef.subs(n, n + 1) / coef), n, oo))
    if l_val == 0:
        radius = oo
    elif l_val == oo:
        radius = 0
    else:
        radius = 1 / l_val if l_val.is_number else Symbol('invL')
    return True, _align([
        (coef, _p("幂级数系数")),
        (Eq(Symbol('L'), l_val), _p("比值极限")),
        (Eq(Symbol('R'), radius), _p("收敛半径")),
    ])


def taylor_error_bound(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    x0 = _num(data.get('point'), 0)
    order = int(data.get('order') or data.get('expr-depth') or 6)
    m = order + 1
    deriv = simplify(f.diff(x, m))
    xi = Symbol('xi')
    remainder = deriv.subs(x, xi) / factorial(m) * (x - x0) ** m
    return True, _align([
        (f, _p("f(x)")),
        (deriv, _p(f"n+1 = {m} 阶导数")),
        (Eq(Symbol('R_n'), remainder), _p("拉格朗日余项 (xi 介于 x0 与 x 之间)")),
    ])


HANDLERS = {
    'tangent': tangent_line,
    'linearization': linearization,
    'critical_points': critical_points,
    'newton': newton_method,
    'arc_length': arc_length,
    'volume_disk': volume_disk,
    'surface_area': surface_area,
    'implicit_diff': implicit_diff,
    'parametric_diff': parametric_diff,
    'polar_area': polar_area,
    'polar_slope': polar_slope,
    'taylor': taylor_series,
    'series_convergence': series_convergence,
    'improper_integral': improper_integral,
    'ode': solve_ode,
    'numerical_integration': numerical_integration,
    'complex_simplify': complex_simplify,
    'complex_polar': complex_polar,
    'complex_roots': complex_roots,
    'mean_value': mean_value,
    'radius_of_convergence': radius_of_convergence,
    'taylor_error': taylor_error_bound,
}
