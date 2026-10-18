"""Integral applications: arc length, solid-of-revolution volume, surface area.

Follows the core engine: the derivative f'(x) comes from DiffCalculator, the
resulting definite integral is expanded by IntegralCalculator, so both the
formula setup and the integration show complete steps.
"""

from sympy import Integral, S, Symbol, cos, diff, latex, pi, simplify, sin, sqrt

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import (align, derivative_steps, expr_text, get_var,
                                     integral_steps, render_note as note, steps_value, to_num)


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
    rows = integral_steps(integrand, x, a, b)
    items += rows
    data['_auto_value'] = steps_value(rows)
    return True, align(items)


def volume_disk(data):
    x, _f_str, f, a, b = _setup(data)
    integrand = pi * f ** 2
    items = [(f, note("f(x)"))]
    items.append((integrand, note("圆盘截面 (绕 x 轴)")))
    items.append((Integral(integrand, (x, a, b)), note("建立体积积分")))
    rows = integral_steps(integrand, x, a, b)
    items += rows
    data['_auto_value'] = steps_value(rows)
    return True, align(items)


def volume_shell(data):
    x, _f_str, f, a, b = _setup(data)
    integrand = 2 * pi * x * f
    items = [(f, note("f(x)"))]
    items.append((integrand, note("柱壳 (绕 y 轴)")))
    items.append((Integral(integrand, (x, a, b)), note("建立柱壳体积积分")))
    rows = integral_steps(integrand, x, a, b)
    items += rows
    data['_auto_value'] = steps_value(rows)
    return True, align(items)


def surface_area(data):
    x, f_str, f, a, b = _setup(data)
    integrand = simplify(2 * pi * f * sqrt(1 + f.diff(x) ** 2))
    items = [(f, note("f(x)"))] + derivative_steps(f_str, x)
    items.append((integrand, note("旋转曲面被积函数 2π f sqrt(1+(f')^2)")))
    items.append((Integral(integrand, (x, a, b)), note("建立面积积分")))
    rows = integral_steps(integrand, x, a, b)
    items += rows
    data['_auto_value'] = steps_value(rows)
    return True, align(items)


def iterated_integral(data):
    """Rectangular iterated (double) integral ∫_c^d ∫_a^b f dx dy.

    Both layers are rule-derived: the inner integral is a partial integral over x
    (every other symbol held constant) and the outer is a definite integral over y,
    each expanded step-by-step by the core IntegralCalculator.
    """
    x = get_var(data)
    y = Symbol(data.get('variable2', 'y') or 'y')
    f = sympify(expr_text(data))
    a, b = to_num(data.get('lower'), 0), to_num(data.get('upper'), 1)
    c, d = to_num(data.get('lower2'), 0), to_num(data.get('upper2'), 1)

    items = [
        (Integral(f, (x, a, b), (y, c, d)), note("累次积分 (矩形区域)")),
        (Integral(f, (x, a, b)), note("先对 x 积分, 其余字母视为常数")),
    ]
    inner = integral_steps(f, x, a, b, allow_multi=True)
    items += inner
    inner_val = simplify([e for (e, _n) in inner][-1]) if inner else f
    items.append((Integral(inner_val, (y, c, d)), note("再对 y 积分")))
    rows = integral_steps(inner_val, y, c, d)
    items += rows
    data['_auto_value'] = steps_value(rows)
    return True, align(items)


def triple_integral(data):
    """Rectangular triple integral ∫∫∫ f dz dy dx (three nested rule-derived layers).

    Inner two layers are partial integrals (other symbols held constant, like the
    iterated integral); the outer is a definite integral. Each layer is expanded
    step-by-step by the core IntegralCalculator. Bounds default to [0,1] per axis.
    """
    x = get_var(data)
    y = Symbol(data.get('variable2', 'y') or 'y')
    z = Symbol(data.get('variable3', 'z') or 'z')
    f = sympify(expr_text(data))
    a, b = to_num(data.get('lower'), 0), to_num(data.get('upper'), 1)
    c, d = to_num(data.get('lower2'), 0), to_num(data.get('upper2'), 1)
    e, g = to_num(data.get('lower3'), 0), to_num(data.get('upper3'), 1)

    items = [(Integral(f, (x, a, b), (y, c, d), (z, e, g)), note("三重累次积分(长方体区域)"))]
    # innermost: over x, y and z held constant
    items.append((Integral(f, (x, a, b)), note("先对 x 积分, y、z 视为常数")))
    i1 = integral_steps(f, x, a, b, allow_multi=True)
    items += i1
    v1 = simplify([ev for (ev, _n) in i1][-1]) if i1 else f
    # middle: over y, z held constant
    items.append((Integral(v1, (y, c, d)), note("再对 y 积分, z 视为常数")))
    i2 = integral_steps(v1, y, c, d, allow_multi=True)
    items += i2
    v2 = simplify([ev for (ev, _n) in i2][-1]) if i2 else v1
    # outer: over z, definite
    items.append((Integral(v2, (z, e, g)), note("最后对 z 积分")))
    rows = integral_steps(v2, z, e, g)
    items += rows
    data['_auto_value'] = steps_value(rows)
    return True, align(items)


def _edge_x_of_y(p1, p2, y):
    """Linear function x(y) for a non-horizontal edge through points p1,p2 (tuples),
    or None if the edge is horizontal. Returns a sympy expression in Symbol y."""
    x1, y1 = p1
    x2, y2 = p2
    if simplify(y1 - y2) == 0:
        return None
    slope = simplify((x2 - x1) / (y2 - y1))  # dx/dy
    return simplify(x1 + slope * (y - y1))


def changeorder(data):
    """Change the order of integration for a linear-bounded (polygon) region.

    Input describes a type-I region: a <= x <= b, g1(x) <= y <= g2(x) with g1, g2
    constant or affine in x (so the region is a convex polygon). The reversed type-II
    limits are derived by inverting the polygon's non-horizontal edges (splitting the
    y-range at vertex heights). The reversed integral is verified equal to the original
    (exact) before being presented; if it is not, or the bounds are not affine, it
    declines honestly rather than emitting a wrong swap.
    """
    x = get_var(data)
    y = Symbol(data.get('variable2', 'y') or 'y')
    f = sympify(expr_text(data))
    a, b = to_num(data.get('lower'), 0), to_num(data.get('upper'), 1)
    g1, g2 = sympify(str(data.get('lower2', 0))), sympify(str(data.get('upper2', 0)))

    def affine(expr, sym):
        """A function is affine (linear or constant) in sym iff its 2nd derivative is 0."""
        return simplify(diff(expr, sym, 2)) == 0
    if not (affine(g1, x) and affine(g2, x)):
        return False, '换序暂仅支持线性/常数边界(凸多边形区域)'

    corners = [(a, simplify(g1.subs(x, a))), (a, simplify(g2.subs(x, a))),
               (b, simplify(g1.subs(x, b))), (b, simplify(g2.subs(x, b)))]
    ylevs = sorted({c[1] for c in corners}, key=float)
    ylo, yhi = ylevs[0], ylevs[-1]
    if ylo == yhi:
        return False, '退化区域(面积为 0)'
    # edges as vertex pairs of the boundary cycle
    edges = [((a, simplify(g1.subs(x, a))), (b, simplify(g1.subs(x, b)))),   # bottom y=g1
             ((b, simplify(g2.subs(x, b))), (a, simplify(g2.subs(x, a)))),   # top   y=g2
             ((a, simplify(g1.subs(x, a))), (a, simplify(g2.subs(x, a)))),   # left  x=a
             ((b, simplify(g1.subs(x, b))), (b, simplify(g2.subs(x, b))))]   # right x=b
    nonhor = [e for e in edges if e[0][1] != e[1][1]]

    slabs = []
    for k in range(len(ylevs) - 1):
        y0, y1 = ylevs[k], ylevs[k + 1]
        ymid = (y0 + y1) / 2
        xs = []
        for p1, p2 in nonhor:
            xof = _edge_x_of_y(p1, p2, y)
            if xof is None:
                continue
            if simplify(p1[1] - ymid) and simplify(p2[1] - ymid) and (
                    min(p1[1], p2[1]) <= ymid <= max(p1[1], p2[1])):
                xs.append(xof)
        if not xs:
            continue
        vals = [float(xof.subs(y, ymid)) for xof in xs]
        xl, xr = xs[vals.index(min(vals))], xs[vals.index(max(vals))]
        if simplify(xl - xr) == 0:
            continue
        slabs.append((y0, y1, simplify(xl), simplify(xr)))

    # verify original vs reversed value (exact) before presenting
    orig = simplify(Integral(Integral(f, (y, g1, g2)), (x, a, b)).doit())
    rev = _rev_value(f, x, y, slabs)
    if simplify(orig - rev) != 0:
        return False, '换序结果与原式不一致, 已拒绝输出'

    items = [(Integral(Integral(f, (y, g1, g2)), (x, a, b)), note("原累次积分(先 y 后 x)"))]
    items.append((None, note("积分区域顶点: " + ", ".join(f"({latex(c[0])},{latex(c[1])})" for c in corners))))
    items.append((None, note("交换积分次序(先 x 后 y), 逐条规则求解:")))
    total = S.Zero
    for si, (y0, y1, xl, xr) in enumerate(slabs):
        if len(slabs) > 1:
            items.append((Integral(Integral(f, (x, xl, xr)), (y, y0, y1)), note(f"第 {si+1} 个 y-分条")))
        inner = integral_steps(f, x, xl, xr, allow_multi=True)  # integrate over x, y constant
        items += inner
        v1 = simplify([ev for (ev, _n) in inner][-1]) if inner else f
        rows = integral_steps(v1, y, y0, y1)                    # outer over y
        items += rows
        total += simplify(steps_value(rows))
    items.append((total, note("累加各分条 = 原值(换序前后已核验相等)")))
    data['_auto_value'] = total
    return True, align(items)


def _rev_value(f, x, y, slabs):
    """Evaluate the reversed-order integral value for verification (exact)."""
    acc = S.Zero
    for (y0, y1, xl, xr) in slabs:
        acc += simplify(Integral(Integral(f, (x, xl, xr)), (y, y0, y1)).doit())
    return acc


def _triple_axes(items, f, axes):
    """Integrate f over three nested axes (inner->outer). Inner two layers are partial
    integrals (allow_multi), the outer is definite; each shown by the core engine.
    axes = [(var, lo, hi), ...] inner first. Returns the final numeric/symbolic value."""
    cur = f
    for idx, (v, lo, hi) in enumerate(axes):
        innermost = idx == 0
        if idx == len(axes) - 1:
            items.append((Integral(cur, (v, lo, hi)), note(f"最后对 {v} 积分")))
            rows = integral_steps(cur, v, lo, hi)
        else:
            items.append((Integral(cur, (v, lo, hi)), note(f"对 {v} 积分(其余视为常数)" if not innermost
                                                           else f"先对 {v} 积分(其余变量视为常数)")))
            rows = integral_steps(cur, v, lo, hi, allow_multi=True)
        items += rows
        cur = simplify([ev for (ev, _n) in rows][-1]) if rows else cur
    return cur


def cylindrical_integral(data):
    """Triple integral in cylindrical coordinates: x=r cosθ, y=r sinθ, z=z, dV = r dz dr dθ.

    Integrand f is given in cartesian (x,y,z); it is substituted to (r,theta,z) and the
    Jacobian r is applied — the setup step that is easy to forget. Bounds (inner->outer)
    are for r, theta, z via lower/upper, lower2/upper2, lower3/upper3.
    """
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    r, theta = Symbol('r'), Symbol('theta')
    f = sympify(expr_text(data))
    rl, ru = to_num(data.get('lower'), 0), to_num(data.get('upper'), 1)
    tl, tu = to_num(data.get('lower2'), 0), to_num(data.get('upper2'), pi)
    zl, zu = to_num(data.get('lower3'), 0), to_num(data.get('upper3'), 1)
    fsub = simplify(f.subs({x: r * cos(theta), y: r * sin(theta)}))
    items = [(f, note("直角坐标被积函数 f(x,y,z)"))]
    items.append((None, note("柱坐标代换: x=r cosθ, y=r sinθ, z=z, dV = r dz dr dθ (雅可比 r)")))
    integrand = simplify(fsub * r)
    items.append((integrand, note("代入并乘雅可比后的被积式 (对 r,θ,z)")))
    val = _triple_axes(items, integrand, [(r, rl, ru), (theta, tl, tu), (z, zl, zu)])
    data['_auto_value'] = val
    return True, align(items)


def spherical_integral(data):
    """Triple integral in spherical coordinates: x=ρ sinφ cosθ, y=ρ sinφ sinθ, z=ρ cosφ,
    dV = ρ² sinφ dρ dφ dθ. Cartesian integrand substituted; Jacobian ρ² sinφ applied.

    Bounds (inner->outer) for rho, phi, theta via lower/upper, lower2/upper2, lower3/upper3.
    """
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    rho = Symbol('rho')
    phi = Symbol('phi')
    theta = Symbol('theta')
    f = sympify(expr_text(data))
    rl, ru = to_num(data.get('lower'), 0), to_num(data.get('upper'), 1)
    pl, pu = to_num(data.get('lower2'), 0), to_num(data.get('upper2'), pi)
    tl, tu = to_num(data.get('lower3'), 0), to_num(data.get('upper3'), 2 * pi)
    fsub = simplify(f.subs({x: rho * sin(phi) * cos(theta),
                            y: rho * sin(phi) * sin(theta),
                            z: rho * cos(phi)}))
    jac = rho ** 2 * sin(phi)
    items = [(f, note("直角坐标被积函数 f(x,y,z)"))]
    items.append((None, note("球坐标代换: x=ρ sinφ cosθ, y=ρ sinφ sinθ, z=ρ cosφ, dV = ρ² sinφ dρ dφ dθ")))
    integrand = simplify(fsub * jac)
    items.append((integrand, note("代入并乘雅可比后的被积式 (对 ρ,φ,θ)")))
    val = _triple_axes(items, integrand, [(rho, rl, ru), (phi, pl, pu), (theta, tl, tu)])
    data['_auto_value'] = val
    return True, align(items)
