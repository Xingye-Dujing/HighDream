"""Rule-based first-order ODE solving shown as explicit method steps.

Handles the two core undergraduate first-order forms — separable and linear
(integrating factor) — by classifying the equation, then carrying out each
integral through the core IntegralCalculator so the working is genuine. Returns
``handled=False`` for any other form so the caller can fall back to ``dsolve``.
"""

from sympy import (Derivative, Eq, I, Integral, Mul, Symbol, cos, diff, exp,
                   simplify, sin, solve, sqrt)

from domains.applied._common import (integral_steps, integral_value,
                                     render_note as note)


def _split_separable(phi, y):
    """If phi(x,y) factors as A(x)*B(y) with no mixed factor, return (A, B)."""
    factors = Mul.make_args(simplify(phi).factor())
    a_terms, b_terms = [], []
    for t in factors:
        has_y = y in t.free_symbols
        has_x = any(s != y for s in t.free_symbols)
        if has_y and has_x:
            return None            # a factor mixes x and y -> not separable
        (b_terms if has_y else a_terms).append(t)
    a = Mul(*a_terms) if a_terms else 1
    b = Mul(*b_terms) if b_terms else 1
    return a, b


def _separable_rows(phi, x, y, c, f):
    ab = _split_separable(phi, y)
    if ab is None:
        return None
    a, b = ab
    rows = [
        (phi, note("右端 $f(x,y)$")),
        (a, note("分离出 $A(x)$")),
        (b, note("分离出 $B(y)$")),
        (1 / b, note("左边被积函数 $1/B(y)$")),
    ]
    rows += integral_steps(1 / b, y)
    rows += integral_steps(a, x)
    ly = integral_value(1 / b, y)
    rx = integral_value(a, x)
    rows.append((Eq(ly.subs(y, f), rx + c), note("两边积分并加常数, 得隐式通解")))
    return rows


def _linear_rows(phi, x, y, c, f):
    """If phi is affine in y (y' = P(x) y + Q(x)), solve via integrating factor."""
    expr = simplify(phi)
    p = expr.coeff(y, 1)          # coefficient of y  (y' = p*y + q)
    if y in p.free_symbols:
        return None
    residual = simplify(expr - p * y)
    if y in residual.free_symbols:
        return None               # not linear in y
    q = residual
    p_std = simplify(-p)          # standard form y' + p_std(x) y = q(x)
    mu_integrand = p_std
    mu = exp(integral_value(mu_integrand, x))
    rows = [
        (p_std, note("标准形式 $y' + p(x) y = q(x)$ 中的 $p(x)$")),
        (q, note("$q(x)$")),
        (mu_integrand, note("积分因子指数 $\\int p(x)\\,dx$ 的被积式")),
    ]
    rows += integral_steps(mu_integrand, x)
    rows.append((mu, note("积分因子 $\\mu=e^{\\int p\\,dx}$")))
    rows.append((mu * q, note("右端 $\\mu\\, q(x)$")))
    rows += integral_steps(mu * q, x)
    rhs = integral_value(mu * q, x) + c
    rows.append((Eq(f, simplify(rhs / mu)), note("通解 $y=\\frac{1}{\\mu}\\int \\mu q\\,dx + \\frac{C}{\\mu}$")))
    return rows


def first_order_ode(eq, x, f):
    """Return (rows, True) if a first-order rule method applied, else (None, False)."""
    y = Symbol('y')
    p = Symbol("p")
    try:
        alg = eq.subs({f: y, Derivative(f, x): p})
        sols = solve(alg, p)
        if not sols:
            return None, False
        phi = simplify(sols[0])
        c = Symbol('C')
        rows = _separable_rows(phi, x, y, c, f)
        if rows is None:
            rows = _linear_rows(phi, x, y, c, f)
        if rows is None:
            return None, False
        return rows, True
    except Exception:  # pylint: disable=broad-exception-caught
        return None, False


def _exact_rows(alg, x, y, c, f):
    """Solve M(x,y) + N(x,y) y' = 0 when it is exact (dM/dy == dN/dx)."""
    expr = simplify(alg.lhs - alg.rhs) if isinstance(alg, Eq) else simplify(alg)
    p = Symbol('p')
    expr = expr.subs(Derivative(y, x), p)
    n = expr.coeff(p, 1)
    m = simplify(expr - n * p)
    if simplify(diff(m, y) - diff(n, x)) != 0:
        return None                 # not exact
    rows = [
        (m, note("$M(x,y)$")),
        (n, note("$N(x,y)$")),
        (simplify(diff(m, y)), note("$\\partial M/\\partial y$")),
        (simplify(diff(n, x)), note("$\\partial N/\\partial x$ (相等则为恰当方程)")),
    ]
    psi = integral_value(m, x, allow_multi=True)   # integrate M wrt x, y constant
    rows += integral_steps(m, x, allow_multi=True)
    gprime = simplify(n - diff(psi, y))
    if x in gprime.free_symbols:
        return None
    rows.append((gprime, note("由 $\\psi_y=N$ 得 $g'(y)$")))
    g = integral_value(gprime, y) if gprime != 0 else 0
    psi_final = simplify(psi + g)
    rows.append((Eq(psi_final.subs(y, f), c), note("势函数 $\\psi(x,y)=C$ 即通解")))
    return rows


def _complementary(a2, a1, a0, x, f):
    """Solve the homogeneous part; return (rows, y1, y2, yc) via the characteristic eq."""
    r = Symbol('r')
    charpoly = a2 * r ** 2 + a1 * r + a0
    disc = simplify(a1 ** 2 - 4 * a2 * a0)
    c1, c2 = Symbol('C1'), Symbol('C2')
    rows = [
        (a2, note("$a_2$")), (a1, note("$a_1$")), (a0, note("$a_0$")),
        (charpoly, note("特征方程 $a_2 r^2 + a_1 r + a_0 = 0$")),
        (disc, note("判别式 $\\Delta = a_1^2 - 4 a_2 a_0$")),
    ]
    if disc == 0:
        root = simplify(-a1 / (2 * a2))
        rows.append((root, note("重根 $r$")))
        y1, y2 = exp(root * x), x * exp(root * x)
    elif disc.is_real and disc > 0:
        r1 = simplify((-a1 - sqrt(disc)) / (2 * a2))
        r2 = simplify((-a1 + sqrt(disc)) / (2 * a2))
        rows.append((Eq(Symbol('r'), r1), note("实根 $r_1$")))
        rows.append((Eq(Symbol('r'), r2), note("实根 $r_2$")))
        y1, y2 = exp(r1 * x), exp(r2 * x)
    else:
        alpha = simplify(-a1 / (2 * a2))
        beta = simplify(sqrt(-disc) / (2 * a2))
        beta = simplify(beta.as_real_imag()[0]) if beta.has(I) else beta
        rows.append((alpha, note("复根实部 $\\alpha$")))
        rows.append((beta, note("复根虚部 $\\beta$")))
        y1, y2 = exp(alpha * x) * cos(beta * x), exp(alpha * x) * sin(beta * x)
    yc = c1 * y1 + c2 * y2
    rows.append((Eq(f, yc), note("对应齐次通解 $y_h$")))
    return rows, y1, y2, yc


def _variation_of_parameters(y1, y2, g, x):
    """Particular solution by variation of parameters; returns (rows, yp) or (None, None)."""
    yp_rows = [
        (y1, note("基本解 $y_1$")),
        (y2, note("基本解 $y_2$")),
    ]
    wronskian = simplify(y1 * y2.diff(x) - y2 * y1.diff(x))
    yp_rows.append((wronskian, note("朗斯基行列式 $W=y_1 y_2'-y_2 y_1'$")))
    u1p = simplify(-y2 * g / wronskian)
    u2p = simplify(y1 * g / wronskian)
    yp_rows.append((u1p, note("$u_1'=-y_2 g/W$")))
    yp_rows += integral_steps(u1p, x)
    yp_rows.append((u2p, note("$u_2'=y_1 g/W$")))
    yp_rows += integral_steps(u2p, x)
    u1 = integral_value(u1p, x)
    u2 = integral_value(u2p, x)
    if u1.has(Integral) or u2.has(Integral):
        return None, None            # integrals not doable by the engine -> caller falls back
    yp = simplify(u1 * y1 + u2 * y2)
    yp_rows.append((yp, note("特解 $y_p=u_1 y_1+u_2 y_2$")))
    return yp_rows, yp


def _second_order_rows(alg, x, f):
    """Solve a2 y'' + a1 y' + a0 y = g(x) with constant coefficients.

    Homogeneous (g=0) uses the characteristic equation; non-homogeneous adds a
    particular solution by variation of parameters (integrals via the engine)."""
    y = Symbol('y')
    p = Symbol('p')
    q = Symbol('q')
    expr = simplify((alg.lhs - alg.rhs) if isinstance(alg, Eq) else alg)
    expr = expr.subs({Derivative(f, (x, 2)): q, Derivative(f, x): p, f: y})
    a2 = expr.coeff(q, 1)
    rest = simplify(expr - a2 * q)
    a1 = rest.coeff(p, 1)
    rest2 = simplify(rest - a1 * p)
    a0 = rest2.coeff(y, 1)
    forcing = simplify(rest2 - a0 * y)
    if a2 == 0:
        return None                                 # not 2nd-order
    for coef in (a2, a1, a0):
        if x in coef.free_symbols or y in coef.free_symbols:
            return None                             # coefficients not constant
    rows, y1, y2, yc = _complementary(a2, a1, a0, x, f)
    if forcing == 0:
        rows.append((Eq(f, yc), note("常系数二阶齐次通解")))
        return rows
    g = simplify(-forcing / a2)                     # standard form RHS
    rows.append((g, note("非齐次项 $g(x)$ (标准形式)")))
    vp_rows, yp = _variation_of_parameters(y1, y2, g, x)
    if vp_rows is None:
        return None                                 # let the caller fall back to dsolve
    rows += vp_rows
    rows.append((Eq(f, simplify(yc + yp)), note("通解 $y=y_h+y_p$")))
    return rows


def solve_ode_rule(eq, x, f):
    """Dispatch to the applicable rule method (2nd-order, exact, separable, linear)."""
    try:
        has_second = Derivative(f, (x, 2)) in _derivs_of(eq)
        if has_second:
            rows = _second_order_rows(eq, x, f)
            if rows is not None:
                return rows, True
            return None, False
        y = Symbol('y')
        c = Symbol('C')
        alg = eq.subs({f: y, Derivative(f, x): Symbol('p')})
        rows = _exact_rows(alg, x, y, c, f)
        if rows is not None:
            return rows, True
        return first_order_ode(eq, x, f)
    except Exception:  # pylint: disable=broad-exception-caught
        return None, False


def _derivs_of(eq):
    """All Derivative objects present in the equation."""
    from sympy import preorder_traversal  # pylint: disable=import-outside-toplevel
    return [e for e in preorder_traversal(eq) if isinstance(e, Derivative)]
