"""Rule-based first-order ODE solving shown as explicit method steps.

Handles the two core undergraduate first-order forms — separable and linear
(integrating factor) — by classifying the equation, then carrying out each
integral through the core IntegralCalculator so the working is genuine. Returns
``handled=False`` for any other form so the caller can fall back to ``dsolve``.
"""

from sympy import (Add, Derivative, Eq, Function, I, Integral, Mul, Symbol, Abs,
                   cos, degree, diff, exp, factor_list, latex, log, simplify, sin, solve, sqrt)

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


def _linear_standard_rows(p_std, q_std, x, dep, c, dep_tex='y'):
    """Rows + solution for the standard linear form ``dep' + p_std(x) dep = q_std(x)``.

    Carried out by the integrating factor, with every integral produced by the core
    rule engine. Shared by the direct linear solver and Bernoulli (after substitution).
    Returns ``(rows, sol)`` or ``(None, None)`` if the integrating-factor integrals fail.
    """
    mu = exp(integral_value(p_std, x))
    if mu.has(Integral):
        return None, None
    muq = mu * q_std
    rhs_int = integral_value(muq, x)
    if rhs_int.has(Integral):
        return None, None
    rows = [
        (p_std, note("标准形式 $%s' + p(x) %s = q(x)$ 中的 $p(x)$" % (dep_tex, dep_tex))),
        (q_std, note("$q(x)$")),
        (p_std, note("积分因子指数 $\\int p(x)\\,dx$ 的被积式")),
    ]
    rows += integral_steps(p_std, x)
    rows.append((mu, note("积分因子 $\\mu=e^{\\int p\\,dx}$")))
    rows.append((muq, note("右端 $\\mu\\, q(x)$")))
    rows += integral_steps(muq, x)
    sol = simplify((rhs_int + c) / mu)
    rows.append((Eq(dep, sol), note("通解 $%s=\\frac{1}{\\mu}\\int \\mu q\\,dx + \\frac{C}{\\mu}$" % dep_tex)))
    return rows, sol


def _linear_rows(phi, x, y, c, f):
    """If phi is affine in y (y' = P(x) y + Q(x)), solve via integrating factor."""
    expr = simplify(phi)
    p = expr.coeff(y, 1)          # coefficient of y  (y' = p*y + q)
    if y in p.free_symbols:
        return None
    residual = simplify(expr - p * y)
    if y in residual.free_symbols:
        return None               # not linear in y
    p_std = simplify(-p)          # standard form y' + p_std(x) y = q(x)
    rows, _sol = _linear_standard_rows(p_std, residual, x, f, c)
    return rows


def _homogeneous_rows(phi, x, y, c, f):
    """Solve y' = phi(x, y) when phi is homogeneous of degree 0 (a function of y/x).

    Substitutes y = v x so that y' = v + x v', reducing to a separable equation in
    (v, x); every integral is carried out by the core rule engine."""
    expr = simplify(phi)
    if y not in expr.free_symbols or x not in expr.free_symbols:
        return None
    t = Symbol('t', positive=True)
    if t in simplify(expr.subs({x: t * x, y: t * y})).free_symbols:
        return None                               # not homogeneous of degree 0
    v = Symbol('v')
    h = simplify(expr.subs({y: v * x}).subs(x, 1))
    denom = simplify(h - v)
    rows = [
        (expr, note("右端 $f(x,y)$")),
        (h, note("齐次方程: 右端仅依赖 $v=y/x$, 记 $\\varphi(v)=f(1,v)$")),
    ]
    if denom == 0:
        rows.append((Eq(f, c * x), note("$\\varphi(v)=v$, 故 $v'=0$, 通解 $y=Cx$")))
        return rows
    left = simplify(1 / denom)
    lv = integral_value(left, v)
    rx = integral_value(1 / x, x)
    if lv.has(Integral):
        return None                               # the v-integral is not doable by the engine
    rows += [
        (Eq(Derivative(y, x), simplify(v + x * Derivative(v, x))),
         note("令 $y=vx$, 则 $y'=v+x\\,v'$")),
        (Eq(x * Derivative(v, x), denom), note("代入化简: $x\\,v'=\\varphi(v)-v$")),
        (left, note("分离变量: $\\frac{1}{\\varphi(v)-v}\\,dv=\\frac{1}{x}\\,dx$")),
    ]
    rows += integral_steps(left, v)
    rows += integral_steps(1 / x, x)
    rows.append((Eq(lv, rx + c), note("两边积分并加常数")))
    rows.append((Eq(lv.subs(v, y / x), rx + c), note("回代 $v=y/x$, 得隐式通解")))
    return rows


def _term_y_power(term, y):
    """Split term into (x-only coeff, exponent of y) when term is coeff*y**k, else (None, None)."""
    coeff, ypart = term.as_independent(y, as_Add=False)
    if ypart == 1:
        return coeff, 0
    base, expo = ypart.as_base_exp()
    if base != y or not expo.is_number:
        return None, None
    return coeff, expo


def _bernoulli_rows(phi, x, y, c, f):
    """Solve Bernoulli y' = A(x) y + B(x) y^n (n not 0 or 1) via u = y^{1-n}.

    The substitution linearises it in u; the resulting linear equation is solved by the
    shared integrating-factor helper, then back-substituted."""
    expr = simplify(phi)
    if not isinstance(expr, Add) or len(expr.args) != 2:
        return None
    parsed = []
    for term in expr.args:
        coeff, expo = _term_y_power(term, y)
        if coeff is None or y in coeff.free_symbols:
            return None
        parsed.append((expo, coeff))
    powers = sorted(p for p, _ in parsed)
    if 1 not in powers or len(powers) != 2:
        return None
    n = next(p for p in powers if p != 1)
    if n in (0, 1):
        return None
    a_lin = next(cf for pw, cf in parsed if pw == 1)          # A(x)
    b_pow = next(cf for pw, cf in parsed if pw == n)          # B(x)
    u = Symbol('u')
    p_std = simplify((n - 1) * a_lin)                         # u' + p_std u = q_std
    q_std = simplify((1 - n) * b_pow)
    lin_rows, u_sol = _linear_standard_rows(p_std, q_std, x, u, c, dep_tex='u')
    if lin_rows is None:
        return None
    rows = [
        (expr, note("右端 $f(x,y)$")),
        (Eq(Derivative(y, x) - a_lin * y, b_pow * y ** n),
         note("伯努利方程 $y' + P(x)\\,y = Q(x)\\,y^{n}$, 其中 $n=" + latex(n) + "$")),
        (Eq(Derivative(u, x) + p_std * u, q_std), note("令 $u=y^{1-n}$, 化为一阶线性方程")),
    ]
    rows += lin_rows
    rows.append((Eq(y ** (1 - n), u_sol), note("回代 $u=y^{1-n}$")))
    rows.append((Eq(f, simplify(u_sol ** (1 / (1 - n)))), note("伯努利方程的通解")))
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
        # A quadratic-or-higher equation in y' has several branches (e.g.
        # y'**2 - 1 = 0). This path can only display one derivation chain, so
        # silently keeping sols[0] dropped the other solutions; decline instead
        # and let the dsolve fallback list every branch.
        distinct = []
        for cand in sols:
            cand = simplify(cand)
            if not any(simplify(cand - seen) == 0 for seen in distinct):
                distinct.append(cand)
        if len(distinct) > 1:
            return None, False
        phi = distinct[0]
        c = Symbol('C')
        for maker in (_separable_rows, _homogeneous_rows, _linear_rows, _bernoulli_rows):
            rows = maker(phi, x, y, c, f)
            if rows is not None:
                return rows, True
        return None, False
    except Exception:  # pylint: disable=broad-exception-caught
        return None, False


def _exact_rows(alg, x, y, c, f):
    """Solve M(x,y) + N(x,y) y' = 0 when it is exact (dM/dy == dN/dx)."""
    expr = simplify(alg.lhs - alg.rhs) if isinstance(alg, Eq) else simplify(alg)
    p = Symbol('p')
    expr = expr.subs(Derivative(y, x), p)
    n = expr.coeff(p, 1)
    m = simplify(expr - n * p)
    # Exactness only applies to M(x,y) + N(x,y) y' = 0, i.e. linear in y'.
    # A nonlinear equation such as y'**2 - 1 = 0 has n = 0 and m depending on p;
    # the derivative of m wrt y is then trivially 0, so the test declared it
    # "exact" and integrated p as if it were a constant (a wrong answer).
    if m.has(p) or n.has(p):
        return None
    if m == 0 and n == 0:
        return None
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


def _exact_factor_rows(alg, x, y, c, f):
    """Make a non-exact M + N y' = 0 exact via an integrating factor mu(x) or mu(y).

    mu(x) exists when (My - Nx)/N depends only on x; mu(y) when (Nx - My)/M depends
    only on y. After multiplying, the exact-equation machinery solves it."""
    expr = simplify(alg.lhs - alg.rhs) if isinstance(alg, Eq) else simplify(alg)
    p = Symbol('p')
    expr = expr.subs(Derivative(y, x), p)
    n = expr.coeff(p, 1)
    m = simplify(expr - n * p)
    if n == 0 or m == 0:
        return None
    if m.has(p) or n.has(p):
        return None                               # not linear in y'
    if simplify(diff(m, y) - diff(n, x)) == 0:
        return None                               # already exact -> the plain path handles it
    my, nx = simplify(diff(m, y)), simplify(diff(n, x))
    mu = None
    kind = ''
    rx = simplify((my - nx) / n)
    if y not in rx.free_symbols:
        mu, kind = exp(integral_value(rx, x)), "只依赖 $x$: $\\mu(x)=e^{\\int (M_y-N_x)/N\\,dx}$"
    else:
        ry = simplify((nx - my) / m)
        if x not in ry.free_symbols:
            mu, kind = exp(integral_value(ry, y)), "只依赖 $y$: $\\mu(y)=e^{\\int (N_x-M_y)/M\\,dy}$"
    if mu is None or mu.has(Integral):
        return None
    sub = _exact_rows(simplify(mu * m) + simplify(mu * n) * p, x, y, c, f)
    if sub is None:
        return None
    return [(mu, note("积分因子 " + kind))] + sub


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


def _cauchy_euler_rows(alg, x, f):
    """Solve the homogeneous Cauchy-Euler equation a x^2 y'' + b x y' + c y = 0.

    Tries y = x^m, giving the indicial equation a m^2 + (b - a) m + c = 0; the three
    root cases (distinct / repeated / complex) map to the standard log-periodic basis."""
    y = Symbol('y')
    p = Symbol('p')
    q = Symbol('q')
    expr = simplify((alg.lhs - alg.rhs) if isinstance(alg, Eq) else alg)
    expr = expr.subs({Derivative(f, (x, 2)): q, Derivative(f, x): p, f: y})
    cc2 = expr.coeff(q, 1)
    rest = simplify(expr - cc2 * q)
    cc1 = rest.coeff(p, 1)
    rest2 = simplify(rest - cc1 * p)
    cc0 = rest2.coeff(y, 1)
    if simplify(rest2 - cc0 * y) != 0:
        return None                                   # has a forcing term
    a = simplify(cc2 / x ** 2)
    b = simplify(cc1 / x)
    c = simplify(cc0)
    if a == 0 or any(v.has(x) or v.has(y) for v in (a, b, c)):
        return None
    if simplify(cc2 - a * x ** 2) != 0 or simplify(cc1 - b * x) != 0 or simplify(cc0 - c) != 0:
        return None                                   # not a pure Cauchy-Euler form
    m = Symbol('m')
    charpoly = a * m ** 2 + (b - a) * m + c
    disc = simplify((b - a) ** 2 - 4 * a * c)
    c1, c2 = Symbol('C1'), Symbol('C2')
    rows = [
        (a, note("$a$")), (b, note("$b$")), (c, note("$c$")),
        (charpoly, note("欧拉方程试 $y=x^m$, 指标方程 $a m^2 + (b-a) m + c = 0$")),
        (disc, note("判别式 $\\Delta = (b-a)^2 - 4ac$")),
    ]
    if disc == 0:
        mm = simplify(-(b - a) / (2 * a))
        rows.append((mm, note("重根 $m$")))
        yc = (c1 + c2 * log(x)) * x ** mm
    elif disc.is_real and disc > 0:
        m1 = simplify((-(b - a) - sqrt(disc)) / (2 * a))
        m2 = simplify((-(b - a) + sqrt(disc)) / (2 * a))
        rows.append((m1, note("实根 $m_1$")))
        rows.append((m2, note("实根 $m_2$")))
        yc = c1 * x ** m1 + c2 * x ** m2
    else:
        alpha = simplify(-(b - a) / (2 * a))
        beta = simplify(sqrt(-disc) / (2 * a))
        beta = simplify(beta.as_real_imag()[0]) if beta.has(I) else beta
        rows.append((alpha, note("复根实部 $\\alpha$")))
        rows.append((beta, note("复根虚部 $\\beta$")))
        yc = x ** alpha * (c1 * cos(beta * log(x)) + c2 * sin(beta * log(x)))
    rows.append((Eq(f, yc), note("欧拉方程(变系数)齐次通解")))
    return rows


def _max_deriv_order(eq, x, f):
    """Highest derivative order of ``f`` wrt ``x`` appearing in the equation.

    Computed straight from ``Derivative`` arguments instead of ``.variables``:
    sympy materializes a tuple with one entry per differentiation step on demand,
    so a ten-billion order derivative would exhaust memory just asking its length.
    """
    from sympy import Integer, Tuple  # pylint: disable=import-outside-toplevel

    best = 0
    for d in _derivs_of(eq):
        if d.expr != f:
            continue
        count = 0
        for spec in d.args[1:]:
            if isinstance(spec, (tuple, Tuple)):
                if len(spec) == 2 and spec[0] == x:
                    count += int(spec[1]) if isinstance(spec[1], Integer) else 10**9
                elif len(spec) == 1 and spec[0] == x:
                    count += 1
            elif spec == x:
                count += 1
        if count > best:
            best = count
            if best > MAX_ODE_ORDER:
                return best
    return best


def _factor_basis(fi, mi, r, x):
    """Rows + solution basis from one characteristic factor ``fi`` of multiplicity ``mi``.

    Returns ``(rows, basis)`` or ``(None, None)`` if the factor is irreducible of degree >= 3
    (not solvable by the standard real/complex root table)."""
    deg = degree(fi, r)
    if deg == 0:
        return [], []
    if deg >= 3:
        return None, None
    if deg == 1:
        r0 = simplify(solve(fi, r)[0])
        rows = [(r0, note("实根 $r = " + latex(r0) + "$, 重数 " + str(mi)))]
        basis = [x ** j * exp(r0 * x) for j in range(mi)]
        return rows, basis
    a = simplify(fi.coeff(r, 2))
    b = simplify(fi.coeff(r, 1))
    c = simplify(fi.coeff(r, 0))
    disc = simplify(b ** 2 - 4 * a * c)
    if disc.is_real and disc > 0:
        r1 = simplify((-b - sqrt(disc)) / (2 * a))
        r2 = simplify((-b + sqrt(disc)) / (2 * a))
        rows = [(r1, note("实根 $r_1$")), (r2, note("实根 $r_2$"))]
        basis = []
        for j in range(mi):
            basis += [x ** j * exp(r1 * x), x ** j * exp(r2 * x)]
        return rows, basis
    alpha = simplify(-b / (2 * a))
    beta = simplify(sqrt(-disc) / (2 * a))
    beta = simplify(beta.as_real_imag()[0]) if beta.has(I) else beta
    rows = [(alpha, note("复根实部 $\\alpha$")), (beta, note("复根虚部 $\\beta$"))]
    basis = []
    for j in range(mi):
        basis += [x ** j * exp(alpha * x) * cos(beta * x),
                  x ** j * exp(alpha * x) * sin(beta * x)]
    return rows, basis


def _higher_order_homogeneous(alg, x, f, order):
    """Solve a linear constant-coefficient homogeneous ODE of any order.

    Builds the characteristic polynomial, factors it over the rationals, and maps each
    (real root / conjugate pair) with its multiplicity to the standard solution basis."""
    if order < 2:
        return None
    expr = simplify((alg.lhs - alg.rhs) if isinstance(alg, Eq) else alg)
    running = expr
    coeffs = {}
    for k in range(order, 0, -1):
        dk = Derivative(f, (x, k))
        ck = simplify(running.coeff(dk, 1))
        if x in ck.free_symbols or f in ck.free_symbols:
            return None                       # coefficient not constant
        coeffs[k] = ck
        running = simplify(running - ck * dk)
    c0 = simplify(running.coeff(f, 1))
    if x in c0.free_symbols or simplify(running - c0 * f) != 0:
        return None                           # non-homogeneous / nonlinear leftover
    if coeffs.get(order, 0) == 0:
        return None
    r = Symbol('r')
    poly_expr = simplify(c0 + sum(coeffs[k] * r ** k for k in range(1, order + 1)))
    try:
        _, facs = factor_list(poly_expr)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    rows = [(poly_expr, note("特征方程 $P(r) = 0$"))]
    basis = []
    for fi, mi in facs:
        frows, fbasis = _factor_basis(simplify(fi), mi, r, x)
        if frows is None:
            return None                       # irreducible degree >= 3 -> fall back
        rows += frows
        basis += fbasis
    if len(basis) != order:
        return None
    cs = [Symbol('C%d' % (i + 1)) for i in range(len(basis))]
    yc = sum(c * b for c, b in zip(cs, basis))
    rows.append((Eq(f, yc), note("常系数高阶齐次通解")))
    return rows


def _isolate_dependency(rows, dep):
    """Explicit solution ``dep = rhs`` from a rows list, or None if it stays implicit.

    Falls back to stripping ``Abs`` (a separable integral yields log(Abs(dep)), and the
    +/- branch is absorbed by the integration constant), so the result can be re-integrated."""
    last = rows[-1][0]
    if not isinstance(last, Eq):
        return None
    if last.lhs == dep and dep not in last.rhs.free_symbols:
        return last.rhs
    for cand in (last, last.replace(Abs, lambda a: a)):
        try:
            sols = solve(cand, dep)
        except (NotImplementedError, ValueError):  # e.g. solving Abs of a complex symbol
            continue
        for s in sols:
            if dep not in s.free_symbols:
                return s
    return None


def _reduce_missing_y(alg, x, f):
    """Reduce a second-order equation missing ``y`` (y'' = G(x, y')) to first order.

    Substitutes p = y' (so y'' = p'), solves the resulting first-order ODE for p(x) by
    reusing the whole first-order dispatcher, then integrates p once more to obtain y."""
    p = Symbol('p')
    q = Symbol('q')
    expr = simplify((alg.lhs - alg.rhs) if isinstance(alg, Eq) else alg)
    d2 = Derivative(f, (x, 2))
    d1 = Derivative(f, x)
    if d2 not in expr.atoms(Derivative):
        return None
    e2 = expr.subs({d2: q, d1: p})
    if f in e2.free_symbols:
        return None                               # y actually appears -> not this form
    sols = solve(e2, q)
    if not sols:
        return None
    G = simplify(sols[0])                         # p' = G(p, x)
    P = Function('p')(x)  # pylint: disable=not-callable
    sub_rows, handled = solve_ode_rule(Eq(Derivative(P, x), G.subs(p, P)), x, P)
    if not handled:
        return None
    p_sol = _isolate_dependency(sub_rows, P)
    if p_sol is None:
        return None                               # implicit in p -> cannot integrate for y
    c1, c2 = Symbol('C1'), Symbol('C2')
    p_sol = p_sol.subs(Symbol('C'), c1)
    y_part = integral_value(p_sol, x, allow_multi=True)
    if y_part.has(Integral):
        return None
    rows = [
        (Eq(Derivative(P, x), G.subs(p, P)),
         note("缺 $y$: 令 $p=y'$, 则 $y''=p'$, 化为关于 $p$ 的一阶方程")),
    ]
    rows += sub_rows
    rows += integral_steps(p_sol, x, allow_multi=True)
    rows.append((Eq(f, simplify(y_part + c2)), note("再积分 $y=\\int p\\,dx + C_2$, 得通解")))
    return rows


def _try(fn, *args):
    """Call a rule method, returning None (not raising) on any internal failure, so one
    form that chokes cannot prevent the other forms from being attempted."""
    try:
        return fn(*args)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


MAX_ODE_ORDER = 8


def solve_ode_rule(eq, x, f):
    """Dispatch to the applicable rule method (2nd-order, exact, separable, linear).

    The order is capped: a ten-digit derivative count would otherwise send
    ``_higher_order_homogeneous`` into billions of simplify() calls and pin a
    worker thread for the rest of the process lifetime.
    """
    try:
        order = _max_deriv_order(eq, x, f)
        if order > MAX_ODE_ORDER:
            return None, False
        if order >= 2:
            if order == 2:
                rows = _try(_second_order_rows, eq, x, f)
                if rows is not None:
                    return rows, True
                rows = _try(_cauchy_euler_rows, eq, x, f)
                if rows is not None:
                    return rows, True
            rows = _try(_higher_order_homogeneous, eq, x, f, order)
            if rows is not None:
                return rows, True
            if order == 2:
                rows = _try(_reduce_missing_y, eq, x, f)
                if rows is not None:
                    return rows, True
            return None, False
        y = Symbol('y')
        c = Symbol('C')
        alg = eq.subs({f: y, Derivative(f, x): Symbol('p')})
        rows = _try(_exact_rows, alg, x, y, c, f)
        if rows is not None:
            return rows, True
        rows = _try(_exact_factor_rows, alg, x, y, c, f)
        if rows is not None:
            return rows, True
        return first_order_ode(eq, x, f)
    except Exception:  # pylint: disable=broad-exception-caught
        return None, False


def _derivs_of(eq):
    """All Derivative objects present in the equation."""
    from sympy import preorder_traversal  # pylint: disable=import-outside-toplevel
    return [e for e in preorder_traversal(eq) if isinstance(e, Derivative)]
