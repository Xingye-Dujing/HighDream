"""Line and surface integrals (curve / surface).

Follows the established applied-engine pattern: every partial derivative is a
genuine rule-engine derivation (``derivative_value``), the final single or double
integral is expanded step-by-step by the core ``IntegralCalculator``
(``integral_steps``), and a numeric quadrature guard rejects grossly wrong values
so nothing incorrect is ever emitted.

Curves are parameterized: the curve is given as '|'-separated coordinate functions
of a single parameter ``t``. Surfaces are graphs ``z = g(x, y)`` over a rectangular
projection (the common undergraduate case).
"""

from sympy import Integral, Matrix, Symbol, latex, simplify, sqrt

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import (align, derivative_value, expr_text, get_var,
                                     integral_steps, render_note as note,
                                     steps_value, to_num)


def _split(text):
    """Split a '|'-separated bundle (commas would be split by the dispatcher)."""
    return [p.strip() for p in str(text).split('|') if p.strip() != '']


def _axes_sub(coords):
    """Map coordinate symbols x, y, z, ... to the curve component expressions."""
    return {Symbol(name): comp for name, comp in zip('xyzwu', coords)}


def _finite_number(value):
    """float of a finite real sympy value (via evalf), or None if complex/non-finite."""
    if value is None:
        return None
    try:
        num = complex(value.evalf())
    except Exception:  # pylint: disable=broad-exception-caught
        try:
            num = complex(float(value))
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    if num.imag != 0.0 and abs(num.imag) > 1e-9:
        return None
    result = num.real
    bounds = (-float('inf'), float('inf'))
    if not bounds[0] < result < bounds[1]:
        return None
    return result


def _simpson_1d(func, a, b, n=200):
    """Composite Simpson integral of a python callable func(x) -> float on [a, b]."""
    if n % 2:
        n += 1
    h = (b - a) / n
    total = func(a) + func(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * func(a + i * h)
    return total * h / 3


def _matches(value, numeric):
    """True iff ``value`` is finite and agrees with the numeric reference when one
    could be computed. A numeric reference that cannot be produced (e.g. an exotic
    callable) does not block; a mismatch does."""
    exact = _finite_number(value)
    if exact is None:
        return False
    if numeric is None:
        return True
    return abs(exact - numeric) <= 1e-3 * (1.0 + abs(exact))


# ---------------------------------------------------------------------------
# Line integrals
# ---------------------------------------------------------------------------

def _curve(data):
    """Parse a parameterized curve into (t, [r_i(t)], a, b)."""
    t = get_var(data)
    comps = _split(data.get('curve', '')) or _split(data.get('variable2', ''))
    if not comps:
        return None
    return t, [sympify(c) for c in comps], comps, to_num(data.get('lower'), 0), to_num(data.get('upper'), 1)


def line_integral(data):
    """Line integral of the first kind (scalar): ∫_C f ds over a parameterized curve.

    With r(t) = (x(t), y(t), ...) and ds = |r'(t)| dt, the integral becomes
    ∫ f(r(t)) |r'(t)| dt; each coordinate derivative is rule-derived and the
    resulting single definite integral is expanded step-by-step.
    """
    parsed = _curve(data)
    if not parsed:
        return False, '第一类曲线积分需参数曲线, 如 lineint(f, x(t)|y(t), t, a, b)'
    t, coords, coord_strs, a, b = parsed
    f = sympify(expr_text(data))
    items = [(f, note("标量场 f")),
             (Matrix(coords), note(f"曲线参数方程 r(t), t ∈ [{latex(a)}, {latex(b)}]"))]
    speeds = []
    for cstr in coord_strs:
        d = derivative_value(cstr, t)
        speeds.append(d)
        items.append((d, note(f"$\\\\partial r/\\\\partial {t}$ 分量(规则引擎)")))
    speed = simplify(sqrt(sum(s ** 2 for s in speeds)))
    items.append((speed, note("弧微分 ds/dt = |r'(t)|")))
    pfield = f.subs(_axes_sub(coords))
    integrand = simplify(pfield * speed)
    items.append((integrand, note("被积式 f(r(t))·|r'(t)|")))
    items.append((Integral(integrand, (t, a, b)), note("建立曲线积分")))
    rows = integral_steps(integrand, t, a, b)
    value = steps_value(rows)
    if not _matches(value, _numeric_integrand(integrand, t, a, b)):
        return False, '曲线积分未能求得可信的定积分值, 已拒绝输出'
    items += rows
    data['_auto_value'] = value
    return True, align(items)


def line_integral2(data):
    """Line integral of the second kind (work): ∫_C P dx + Q dy (+ R dz).

    Substituting the parameterized curve gives ∫ [P(r(t)) x'(t) + Q(r(t)) y'(t) + ...] dt,
    with every derivative rule-driven and the final integral shown step-by-step.
    """
    parsed = _curve(data)
    if not parsed:
        return False, '第二类曲线积分需参数曲线, 如 lineint2(P|Q|R, x(t)|y(t), t, a, b)'
    t, coords, coord_strs, a, b = parsed
    field = _split(expr_text(data))
    if len(field) != len(coords):
        return False, '向量场分量数需与曲线维数一致'
    subs = _axes_sub(coords)
    items = [(Matrix([sympify(c) for c in field]), note("向量场 F")),
             (Matrix(coords), note("曲线参数方程 r(t)"))]
    derivs = []
    for cstr in coord_strs:
        derivs.append(derivative_value(cstr, t))
    integrand_terms = []
    for fstr, d in zip(field, derivs):
        substituted = sympify(fstr).subs(subs)
        term = simplify(substituted * d)
        integrand_terms.append(term)
        items.append((term, note(f"$P_i(r(t)) \\\\cdot {latex(d)}$ 分量贡献")))
    integrand = simplify(sum(integrand_terms))
    items.append((integrand, note("被积式 Σ F_i(r(t))·r_i'(t)")))
    items.append((Integral(integrand, (t, a, b)), note("建立第二类曲线积分(功)")))
    rows = integral_steps(integrand, t, a, b)
    value = steps_value(rows)
    if not _matches(value, _numeric_integrand(integrand, t, a, b)):
        return False, '第二类曲线积分未能求得可信值, 已拒绝输出'
    items += rows
    data['_auto_value'] = value
    return True, align(items)


# ---------------------------------------------------------------------------
# Surface integrals (graph z = g(x, y) over a rectangle)
# ---------------------------------------------------------------------------

def _surface(data):
    """Parse graph-surface inputs into (x, y, g, rect) with rect = ((a, b), (c, d))."""
    x = get_var(data)
    y = Symbol(data.get('variable2', 'y') or 'y')
    g = sympify(str(data.get('surface', data.get('variable2', ''))))
    a, b = to_num(data.get('lower'), 0), to_num(data.get('upper'), 1)
    c, d = to_num(data.get('lower2'), 0), to_num(data.get('upper2'), 1)
    return x, y, g, ((a, b), (c, d))


def _double(integrand, x, y, rect):
    """Return (items, value) integrating integrand over the rectangle rect=((a,b),(c,d)),
    rule-driven: inner partial integral over x (y held constant), then definite over y."""
    (a, b), (c, d) = rect
    items = [(Integral(Integral(integrand, (x, a, b)), (y, c, d)), note("建立二重积分(矩形投影)")),
             (Integral(integrand, (x, a, b)), note("先对 x 积分, y 视为常数"))]
    inner = integral_steps(integrand, x, a, b, allow_multi=True)
    items += inner
    v1 = simplify([ev for (ev, _n) in inner][-1]) if inner else integrand
    items.append((Integral(v1, (y, c, d)), note("再对 y 积分")))
    rows = integral_steps(v1, y, c, d)
    items += rows
    return items, steps_value(rows)


def surface_integral(data):
    """Surface integral of the first kind ∫∫_S f dS for a graph z = g(x, y).

    dS = sqrt(1 + g_x^2 + g_y^2) dx dy (both partials rule-derived); the scalar
    field f is evaluated on the surface (z -> g) and integrated over the rectangle.
    """
    x, y, g, rect = _surface(data)
    if str(g) == '':
        return False, '曲面积分需给出曲面 surface = g(x,y)'
    f = sympify(expr_text(data))
    gx = derivative_value(str(g), x)
    gy = derivative_value(str(g), y)
    dS = simplify(sqrt(1 + gx ** 2 + gy ** 2))
    fsub = simplify(f.subs(Symbol('z'), g))
    integrand = simplify(fsub * dS)
    items = [(f, note("标量场 f(x,y,z)")), (g, note("曲面 z = g(x,y)"))]
    items.append((gx, note("$\\\\partial g/\\\\partial x$ (规则引擎)")))
    items.append((gy, note("$\\\\partial g/\\\\partial y$ (规则引擎)")))
    items.append((dS, note("面积元 dS = sqrt(1 + g_x² + g_y²) dx dy")))
    items.append((integrand, note("被积式 f(x,y,g)·sqrt(1+g_x²+g_y²)")))
    dbl, value = _double(integrand, x, y, rect)
    if not _matches(value, _numeric_integrand_2d(integrand, x, y, rect)):
        return False, '第一类曲面积分未能求得可信值, 已拒绝输出'
    items += dbl
    data['_auto_value'] = value
    return True, align(items)


def flux(data):
    """Flux (surface integral of the second kind) ∫∫_S F · dS for a graph z = g(x, y),
    upward orientation: the integrand becomes -P g_x - Q g_y + R, evaluated with z -> g
    and integrated over the rectangular projection.
    """
    x, y, g, rect = _surface(data)
    field = _split(expr_text(data))
    if len(field) != 3:
        return False, '通量需三维向量场 flux(P|Q|R, surface=g(x,y), ...)'
    P, Q, R = (sympify(v) for v in field)
    gx = derivative_value(str(g), x)
    gy = derivative_value(str(g), y)
    subz = {Symbol('z'): g}
    integrand = simplify(-P.subs(subz) * gx - Q.subs(subz) * gy + R.subs(subz))
    items = [(Matrix([P, Q, R]), note("向量场 F")), (g, note("曲面 z = g(x,y), 取上侧定向"))]
    items.append((gx, note("$\\\\partial g/\\\\partial x$ (规则引擎)")))
    items.append((gy, note("$\\\\partial g/\\\\partial y$ (规则引擎)")))
    items.append((integrand, note("通量被积式 -P·g_x - Q·g_y + R (代入 z = g)")))
    dbl, value = _double(integrand, x, y, rect)
    if not _matches(value, _numeric_integrand_2d(integrand, x, y, rect)):
        return False, '通量积分未能求得可信值, 已拒绝输出'
    items += dbl
    data['_auto_value'] = value
    return True, align(items)


# ---------------------------------------------------------------------------
# numeric references (verification only; the displayed work stays rule-driven)
# ---------------------------------------------------------------------------

def _num(expr, sym):
    """A safe python callable for the 1-parameter expression, or None."""
    try:
        from sympy import lambdify  # pylint: disable=import-outside-toplevel
        return lambdify(sym, expr, 'math')
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _numeric_integrand(integrand, t, a, b):
    try:
        func = _num(integrand, t)
        if func is None:
            return None
        return _simpson_1d(func, float(a), float(b))
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _numeric_integrand_2d(integrand, x, y, rect):
    (a, b), (c, d) = rect
    def column(yy):
        return _simpson_1d(lambda xx: float(integrand.subs({x: xx, y: yy})), float(a), float(b), 80)
    try:
        return _simpson_1d(column, float(c), float(d), 40)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
