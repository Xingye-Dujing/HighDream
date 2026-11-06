"""Multivariable-adjacent differentiation: implicit, parametric, polar.

Parametric and polar slopes derive each coordinate's derivative with the core
DiffCalculator (real steps); polar area routes its integral through
IntegralCalculator; implicit differentiation shows the differentiated equation
then solves for y'.
"""

from sympy import Eq, Function, Matrix, Symbol, cos, latex, sin, solve, simplify, sqrt, Integral

from utils.cache_utils import cached_sympify as sympify
from utils.expression_utils import split_top_level_pipe

from domains.applied._common import (align, derivative_steps, derivative_value, expr_text,
                                     get_var, integral_steps, render_note as note, steps_value, to_num)


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
    # DiffCalculator cannot differentiate the unknown function y(x) as an
    # elementary function; its "无法用初等函数表示" note is noise here because
    # the chain-rule term is kept and solved for y' below. Drop only that note.
    d_lhs = [(e, n) for (e, n) in d_lhs if '无法用初等函数表示' not in n]
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
    raw = expr_text(data)
    lines = [part.strip().rstrip(',').strip()
             for part in raw.split('\n') if part.strip()]
    if len(lines) < 2 and ',' in raw:
        lines = [part.strip() for part in raw.split(',', 1)]
    x_str = lines[0] if lines else ''
    y_str = lines[1] if len(lines) > 1 else ''
    if not x_str or not y_str:
        return False, '参数方程需要 x(t) 与 y(t) 两个表达式，用逗号或换行分隔'
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


def gradient(data):
    """Gradient ∇f of a scalar field: one partial derivative (rule-engine steps) per free symbol."""
    x = get_var(data)
    f_str = expr_text(data)
    f = sympify(f_str)
    syms = sorted(f.free_symbols, key=str) or [x]
    items = [(f, note("标量场 f"))]
    comps = []
    for s in syms:
        steps = derivative_steps(f_str, s)
        items.append((None, note(f"对 {s} 求偏导(其余视为常数)")))
        items += steps
        comps.append(_last_value(steps))
    if any(c is None for c in comps):
        return False, '某方向偏导未能完成求解'
    items.append((Matrix(comps), note("梯度 ∇f = (∂f/∂x_1, ..., ∂f/∂x_n)")))
    return True, align(items)


def _split_pipe(text):
    """Split a '|'-separated vector/list argument (commas would clash with the
    auto-dispatcher's top-level arg splitting)."""
    return split_top_level_pipe(str(text))


def hessian(data):
    """Hessian matrix + second-partial discriminant D = f_xx*f_yy - f_xy^2 for f(x, y).

    Each second partial is a genuine two-pass rule-engine derivation (order-1 then
    order-1 again for the mixed term), never a black-box ``.diff(2)``.
    """
    f_str = expr_text(data)
    f = sympify(f_str)
    syms = sorted(f.free_symbols, key=str)
    if len(syms) < 2:
        return False, 'Hessian 需要二元及以上函数(至少两个自变量)'
    x, y = syms[0], syms[1]
    fx = derivative_value(f, x)
    fy = derivative_value(f, y)
    fxx = derivative_value(fx, x)
    fyy = derivative_value(fy, y)
    fxy = derivative_value(fx, y)
    d = simplify(fxx * fyy - fxy ** 2)
    items = [(f, note("f"))]
    items.append((fx, note(f"$\\\\partial f/\\\\partial {x}$")))
    items.append((fy, note(f"$\\\\partial f/\\\\partial {y}$")))
    items.append((fxx, note("二阶偏导 f_xx")))
    items.append((fxy, note("混合偏导 f_xy = f_yx")))
    items.append((fyy, note("二阶偏导 f_yy")))
    items.append((Matrix([[fxx, fxy], [fxy, fyy]]), note("Hessian 矩阵 H = ∂²f/∂xᵢ∂xⱼ")))
    items.append((d, note("判别式 D = f_xx·f_yy − f_xy² (临界点处: D>0且f_xx>0极小, D>0且f_xx<0极大, D<0鞍点)")))
    data['_auto_value'] = None
    return True, align(items)


def jacobian(data):
    """Jacobian matrix and determinant of a vector map (u|v|...) over (x|y|...).

    Components and independent variables are '|'-separated (commas would be split as
    separate operator arguments). Every entry is a rule-engine partial derivative.
    """
    comps = _split_pipe(expr_text(data))
    varnames = _split_pipe(data.get('variable', '')) or []
    if not comps:
        return False, '雅可比需要至少一个分量函数'
    fexprs = [sympify(c) for c in comps]
    if not varnames:
        allsyms = set()
        for fe in fexprs:
            allsyms |= fe.free_symbols
        varnames = sorted(allsyms, key=str)
    else:
        varnames = [sympify(v) for v in varnames]
    nrows, ncols = len(fexprs), len(varnames)
    rows, entries = [], []
    for fe, cstr in zip(fexprs, comps):
        row = []
        for v in varnames:
            row.append(derivative_value(cstr, v))
        entries.append(row)
    jac = Matrix(entries)
    for i, fe in enumerate(fexprs):
        rows.append((fe, note(f"$u_{i+1}$")))
    rows.append((jac, note(f"雅可比矩阵 J = ∂(u)/∂({','.join(str(v) for v in varnames)})")))
    if nrows == ncols:
        rows.append((simplify(jac.det()), note("雅可比行列式 det(J)")))
    return True, align(rows)


def directional_derivative(data):
    """Directional derivative ∇f · û in direction (a|b|...) (normalized).

    Uses the rule-engine partials for ∇f, then forms the unit vector and the dot
    product, all shown step by step.
    """
    f_str = expr_text(data)
    f = sympify(f_str)
    syms = sorted(f.free_symbols, key=str)
    if len(syms) < 2:
        return False, '方向导数需要多元函数(至少两个自变量)'
    dir_comps = _split_pipe(data.get('direction', ''))
    if len(dir_comps) != len(syms):
        return False, '方向向量分量个数需与自变量个数一致(direction=a|b|...)'
    dir_vec = Matrix([sympify(d) for d in dir_comps])
    grad = Matrix([derivative_value(f_str, s) for s in syms])
    norm = simplify(sqrt(sum(c ** 2 for c in dir_vec)))
    if norm == 0:
        return False, '方向向量不能是零向量'
    unit = dir_vec / norm
    result = simplify((grad.T * unit)[0])
    items = [(f, note("f"))]
    items.append((grad, note("梯度 ∇f")))
    items.append((dir_vec, note("方向向量 v")))
    items.append((norm, note("|v|")))
    items.append((unit, note("单位方向 v/|v|")))
    items.append((result, note("方向导数 D_u f = ∇f · (v/|v|)")))
    return True, align(items)


def _vector_field(data):
    """Parse '|'-separated components + '|'-separated variables into (exprs, strs, syms)."""
    comps = _split_pipe(expr_text(data))
    varnames = [sympify(v) for v in _split_pipe(data.get('variable', ''))]
    cstrs = comps
    return [sympify(c) for c in comps], cstrs, varnames


def curl(data):
    """Curl ∇×F of a 3-D vector field (P|Q|R over x|y|z). Component partials are rule-derived."""
    exprs, cstrs, syms = _vector_field(data)  # pylint: disable=unbalanced-tuple-unpacking
    if len(exprs) != 3 or len(syms) != 3:
        return False, '旋度需三维向量场 curl(P|Q|R, x|y|z)'
    P, Q, R = exprs
    ps, qs, rs = cstrs  # pylint: disable=unbalanced-tuple-unpacking
    x, y, z = syms
    def d(fstr, v):
        return derivative_value(fstr, v)
    comp = [simplify(d(rs, y) - d(qs, z)), simplify(d(ps, z) - d(rs, x)), simplify(d(qs, x) - d(ps, y))]
    items = [(Matrix([P, Q, R]), note("向量场 F"))]
    items.append((None, note("∇×F = (∂R/∂y−∂Q/∂z, ∂P/∂z−∂R/∂x, ∂Q/∂x−∂P/∂y)")))
    for c, lab in zip(comp, ["∂R/∂y − ∂Q/∂z", "∂P/∂z − ∂R/∂x", "∂Q/∂x − ∂P/∂y"]):
        items.append((c, note(lab)))
    items.append((Matrix(comp), note("旋度 ∇×F")))
    return True, align(items)


def divergence(data):
    """Divergence ∇·F of a vector field (P|Q|R over x|y|z). Partials rule-derived."""
    exprs, cstrs, syms = _vector_field(data)  # pylint: disable=unbalanced-tuple-unpacking
    if len(exprs) != len(syms) or len(exprs) < 2:
        return False, '散度需分量数与变量数一致, 如 div(P|Q|R, x|y|z)'
    items = [(Matrix(exprs), note("向量场 F"))]
    terms = []
    for cstr, s in zip(cstrs, syms):
        part = derivative_value(cstr, s)
        terms.append(part)
        items.append((part, note(f"$\\partial/\\partial {s}$ 作用于对应分量")))
    div = simplify(sum(terms))
    items.append((div, note("散度 ∇·F")))
    return True, align(items)


def lagrange(data):
    """Lagrange multipliers: extrema of f subject to g = 0.

    Builds ∂f/∂xi = λ ∂g/∂xi (rule-engine partials) plus the constraint g=0, then
    solves the system for the stationary points and reports f at each."""
    parts = _split_pipe(expr_text(data))
    if len(parts) != 2:
        return False, '拉格朗日乘数需 f|g 两个表达式(lagrange(f|g, x|y))'
    fstr, gstr = parts  # pylint: disable=unbalanced-tuple-unpacking
    f, g = sympify(fstr), sympify(gstr)
    syms = [sympify(v) for v in _split_pipe(data.get('variable', ''))] or sorted((f.free_symbols | g.free_symbols), key=str)
    lam = Symbol('lambda')
    eqs = []
    items = [(f, note("目标函数 f")), (Eq(g, 0), note("约束 g = 0"))]
    for s in syms:
        dfx = derivative_value(fstr, s)
        dgx = derivative_value(gstr, s)
        eqs.append(Eq(dfx, lam * dgx))
        items.append((Eq(dfx, lam * dgx), note(f"∂f/∂{s} = λ·∂g/∂{s}")))
    eqs.append(g)
    items.append((None, note("联立求解临界点")))
    try:
        sols = solve(eqs, list(syms) + [lam], dict=True)
    except Exception:  # pylint: disable=broad-exception-caught
        return False, '拉格朗日方程组求解失败'
    if not sols:
        return False, '未求得拉格朗日临界点'
    rows = []
    for sol in sols:
        pt = {s: simplify(sol.get(s, 0)) for s in syms}
        val = simplify(f.subs(pt))
        rows.append((simplify(sol.get(lam, 0)), pt, val))
        items.append((Matrix([pt[s] for s in syms]), note(f"λ={latex(simplify(sol.get(lam, 0)))} 处, f={latex(val)}")))
    return True, align(items)
