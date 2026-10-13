"""Applied-calculus / series / ODE / numerical operations (SymPy-backed).

These cover the topics beyond the core derivative/integral/limit engines found
in an undergraduate (Princeton-style) calculus course: implicit & parametric
differentiation, curve applications (tangent line, critical points, Newton's
method, linearization), arc length / volumes / surface area, improper integrals,
Taylor/Maclaurin series, sequence/series convergence, ordinary differential
equations, complex numbers, and numerical integration.

Each handler reads the cell payload ``data`` (expression/variable plus the
optional point/lower/upper/order fields) and returns ``(success, latex)`` where
``latex`` is a ``\\begin{align}`` block or a short rendered verdict.
"""

from sympy import (Abs, Derivative, Eq, Function, I, Integral, N, Symbol, arg,
                   cos, dsolve, exp, expand, im, latex, limit, pi, re, simplify, sin,
                   solve, summation, series, sqrt, sympify, oo)


def _var(data):
    return Symbol(data.get('variable', 'x') or 'x')


def _num(value, default=None):
    """Parse a user field into a SymPy number; '' -> default."""
    if value is None or str(value).strip() == '':
        return default
    return sympify(str(value))


def _align(rows):
    """rows: list of (body, note); body may be a SymPy object or a latex string."""
    lines = []
    for idx, (body, note) in enumerate(rows):
        text = latex(body) if not isinstance(body, str) else body
        prefix = '' if idx == 0 else '&='
        line = f"{prefix}{text}" if text.strip() else ''
        if note:
            line += rf" \quad \text{{{note}}}"
        lines.append(line)
    return r"\begin{align}" + r"\newline".join(lines) + r"\end{align}"


def _expr_text(data):
    return data.get('expression', '')


# ---------------------------------------------------------------- applications
def tangent_line(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('point'), 0)
    slope = simplify(f.diff(x).subs(x, a))
    y0 = simplify(f.subs(x, a))
    line = simplify(y0 + slope * (x - a))
    return True, _align([
        (f, ''), (f.diff(x), '求导'),
        (Eq(Symbol('y'), line), f'切线 @ {latex(a)}: y-f(a)=f\'(a)(x-a)'),
    ])


def linearization(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('point'), 0)
    lin = simplify(f.subs(x, a) + f.diff(x).subs(x, a) * (x - a))
    return True, _align([
        (f, ''), (Eq(Symbol('L'), lin), f'在 {latex(a)} 处的线性化(切线近似)'),
    ])


def critical_points(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    fp = f.diff(x)
    try:
        crit = sorted(solve(Eq(fp, 0), x), key=lambda v: (not v.is_real, str(v)))
    except Exception:  # pylint: disable=broad-exception-caught
        crit = []
    rows = [(f, ''), (fp, "一阶导数 f'(x)")]
    if not crit:
        rows.append(('f\'(x)=0 无实临界点', ''))
        return True, _align(rows)
    fpp = f.diff(x, 2)
    for c in crit:
        kind = fpp.subs(x, c)
        label = ('极小' if (kind.is_positive) else '极大' if (kind.is_negative) else '待定(二阶=0)')
        rows.append((Eq(x, c), f'临界点; 由二阶导数判定: {label}'))
    return True, _align(rows)


def newton_method(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    x0 = _num(data.get('point'), 1)
    iters = int(data.get('order') or data.get('expr-depth') or 4)
    rows = [(Eq(Symbol('x_0'), x0), '初值')]
    cur = x0
    for k in range(1, iters + 1):
        nxt = simplify(cur - f.subs(x, cur) / f.diff(x).subs(x, cur))
        rows.append((Eq(Symbol(f'x_{k}'), nxt), '牛顿迭代'))
        cur = nxt
    return True, _align(rows)


# ---------------------------------------------------------------- arc/volume
def _interval(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('lower'), 0)
    b = _num(data.get('upper'), 1)
    return x, f, a, b


def arc_length(data):
    x, f, a, b = _interval(data)
    integrand = simplify(sqrt(1 + f.diff(x) ** 2))
    value = Integral(integrand, (x, a, b)).doit()
    return True, _align([
        (f, ''), (integrand, '\\sqrt{1+(f\')^2}'),
        (Eq(Symbol('S'), value), f'弧长 on [{latex(a)},{latex(b)}]'),
    ])


def volume_disk(data):
    x, f, a, b = _interval(data)
    integrand = pi * f ** 2
    value = simplify(Integral(integrand, (x, a, b)).doit())
    return True, _align([
        (f, ''), (integrand, 'π f(x)^2 (绕 x 轴圆盘)'),
        (Eq(Symbol('V'), value), f'体积 on [{latex(a)},{latex(b)}]'),
    ])


def surface_area(data):
    x, f, a, b = _interval(data)
    integrand = 2 * pi * f * sqrt(1 + f.diff(x) ** 2)
    value = Integral(integrand, (x, a, b)).doit()
    return True, _align([
        (f, ''), (integrand, '2π f√(1+(f\')^2)'),
        (Eq(Symbol('A'), value), f'旋转曲面面积 on [{latex(a)},{latex(b)}]'),
    ])


# ---------------------------------------------------------------- implicit/param
def implicit_diff(data):
    x = _var(data)
    y = Function('y')(x)  # pylint: disable=not-callable
    # Replace the free symbol 'y' with the function y(x).
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
        (eq, '两边对 x 求导'), (Eq(dy, result), '解出 y\''),
    ])


def parametric_diff(data):
    t = Symbol(data.get('variable', 't') or 't')
    lines = _expr_text(data).split('\n')
    x_t = sympify(lines[0])
    y_t = sympify(lines[1]) if len(lines) > 1 else sympify(lines[0])
    dydx = simplify(y_t.diff(t) / x_t.diff(t))
    return True, _align([
        (Eq(Symbol('x'), x_t), ''), (Eq(Symbol('y'), y_t), ''),
        (Eq(Symbol("y'"), dydx), 'dy/dx = (dy/dt)/(dx/dt)'),
    ])


def polar_area(data):
    theta = Symbol(data.get('variable', 'theta') or 'theta')
    r = sympify(_expr_text(data))
    a = _num(data.get('lower'), 0)
    b = _num(data.get('upper'), pi)
    value = simplify(Integral(r ** 2 / 2, (theta, a, b)).doit())
    return True, _align([
        (r, ''), (Eq(Symbol('A'), value), '面积 = (1/2)∫ r^2 dθ'),
    ])


def polar_slope(data):
    theta = Symbol(data.get('variable', 'theta') or 'theta')
    r = sympify(_expr_text(data))
    x_of_t = r * cos(theta)
    y_of_t = r * sin(theta)
    dydx = simplify(y_of_t.diff(theta) / x_of_t.diff(theta))
    return True, _align([
        (r, ''), (Eq(Symbol("dy/dx"), dydx), 'r(θ) 曲线在极坐标下的斜率'),
    ])


# ---------------------------------------------------------------- series/taylor
def taylor_series(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    x0 = _num(data.get('point'), 0)
    order = int(data.get('order') or data.get('expr-depth') or 6)
    ser = series(f, x, x0, order + 1)
    return True, _align([
        (f, ''), (ser.removeO(), f'在 {latex(x0)} 处到 {order} 阶'),
        (ser, '含高阶余项 O'),
    ])


def series_convergence(data):
    n = Symbol(data.get('variable', 'n') or 'n')
    a = sympify(_expr_text(data))
    rows = [(a, '一般项 a_n')]
    term_lim = limit(a, n, oo)
    rows.append((Eq(Symbol('lim a_n'), term_lim), '第 n 项判别'))
    if term_lim not in (0,):
        rows.append(('级数发散 (第 n 项不趋于 0)', ''))
        return True, _align(rows)
    ratio = simplify(limit(abs(a.subs(n, n + 1) / a), n, oo))
    rows.append((Eq(Symbol('L=lim|a_{n+1}/a_n|'), ratio), '比值判别'))
    if ratio.is_number:
        verdict = ('收敛' if ratio < 1 else '发散' if ratio > 1 else '比值判别不确定')
        rows.append((f'比值判别: L={latex(ratio)} → {verdict}', ''))
    closed = summation(a, (n, 1, oo))
    rows.append((Eq(Symbol('Σ a_n'), closed), '求和(若可闭式)'))
    return True, _align(rows)


def improper_integral(data):
    x = _var(data)
    f = sympify(_expr_text(data))
    a = _num(data.get('lower'), 0)
    b = _num(data.get('upper'), oo)
    val = Integral(f, (x, a, b)).doit()
    verdict = '收敛' if val.is_finite else '发散'
    return True, _align([
        (Integral(f, (x, a, b)), '反常积分'),
        (val, f'{verdict} (∞ 界先求原函数再取极限)'),
    ])


# ---------------------------------------------------------------- ODE
def solve_ode(data):
    x = _var(data)
    f = Function('f')(x)  # pylint: disable=not-callable
    raw = _expr_text(data)
    # Allow a plain "f'(x)" / "f''(x)" shorthand for derivatives.
    raw = (raw.replace("f'(x)", 'Derivative(f(x), x)')
              .replace("f''(x)", 'Derivative(f(x), (x, 2))'))
    expr = sympify(raw)
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    try:
        sol = dsolve(eq, f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'ODE 无法求解: {exc}'
    return True, _align([
        (eq, '微分方程'), (sol, 'dsolve 通解'),
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
        trap = (h / 2) * (ff[0] + 2 * sum(ff[1:-1]) + ff[-1])
        if n % 2 == 0:
            simp = (h / 3) * (ff[0] + 4 * sum(ff[i] for i in range(1, n, 2))
                              + 2 * sum(ff[i] for i in range(2, n, 2)) + ff[-1])
        else:
            simp = None
        rows = [(f, ''), (N(trap, 8), f'梯形法 n={n}')]
        if simp is not None:
            rows.append((N(simp, 8), f'辛普森法 n={n}'))
        return True, _align(rows)
    return False, '数值积分需要数值积分上下限'


# ---------------------------------------------------------------- complex numbers
def complex_simplify(data):
    z = sympify(_expr_text(data))
    s = simplify(expand(z, complex=True))
    return True, _align([
        (z, '原复数式'), (re(s), '实部 Re(z)'), (im(s), '虚部 Im(z)'),
        (s, '化简为 a+bi 形式'),
    ])


def complex_polar(data):
    z = sympify(_expr_text(data))
    r = simplify(Abs(z))
    theta = arg(z)
    return True, _align([
        (z, ''), (r, '模 r=|z|'), (theta, '辐角 θ=arg(z)'),
        (r * exp(I * theta), '指数形式 r·e^{iθ}'),
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
    rows = [(eq, '求解 (复数域)')]
    for i, s in enumerate(sol):
        rows.append((Eq(z, simplify(expand(s, complex=True))), f'根 {i + 1}'))
    return True, _align(rows)


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
    rows = [(f, ''), (Eq(Symbol('k'), secant), '割线斜率 (f(b)-f(a))/(b-a)'),
            (fp, "f'(x)")]
    if not cands:
        rows.append(('未能解出满足条件的 c', ''))
    for i, c in enumerate(cands):
        rows.append((Eq(Symbol('c'), c), f'中值点 c_{i + 1} (需 a<c<b)'))
    return True, _align(rows)


def radius_of_convergence(data):
    n = Symbol('n')
    coef = sympify(_expr_text(data))
    l_val = simplify(limit(Abs(coef.subs(n, n + 1) / coef), n, oo))
    if l_val == 0:
        radius = oo
    elif l_val == oo:
        radius = 0
    else:
        radius = 1 / l_val if l_val.is_number else Symbol('1/L')
    return True, _align([
        (coef, '幂级数系数 a_n'), (l_val, 'L = lim|a_{n+1}/a_n|'),
        (Eq(Symbol('R'), radius), '收敛半径 R = 1/L'),
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
}
