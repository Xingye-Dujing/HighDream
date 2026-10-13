"""Rule-based first-order ODE solving shown as explicit method steps.

Handles the two core undergraduate first-order forms — separable and linear
(integrating factor) — by classifying the equation, then carrying out each
integral through the core IntegralCalculator so the working is genuine. Returns
``handled=False`` for any other form so the caller can fall back to ``dsolve``.
"""

from sympy import Derivative, Eq, Mul, Symbol, exp, simplify, solve

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
