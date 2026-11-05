"""Complex-number operations: a+bi simplification, polar form, solving z^n = w."""

from sympy import Abs, Eq, I, S, Symbol, arg, expand, exp, im, pi, re, root, simplify, solve

from config import HARD_MAX_ITERATIONS
from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import align, expr_text, get_var, render_note as note


def complex_simplify(data):
    z = sympify(expr_text(data))
    s = simplify(expand(z, complex=True))
    return True, align([
        (z, note("原复数式")), (re(s), note("实部 Re")),
        (im(s), note("虚部 Im")), (s, note("化为 a+bi 形式")),
    ])


def complex_polar(data):
    z = sympify(expr_text(data))
    r = simplify(Abs(z))
    theta = arg(z)
    return True, align([
        (z, note("复数 z")), (r, note("模")), (theta, note("辐角")),
        (r * exp(I * theta), note("指数形式")),
    ])


def de_moivre_power(data):
    z = sympify(expr_text(data))
    n = sympify(str(data.get('order') or data.get('point') or 2))
    powered = expand(z ** n)
    return True, align([
        (z, note("z")), (simplify(powered), note("棣莫弗/二项式展开 z^n")),
    ])


def complex_nth_roots(data):
    """All n-th roots of a complex number w via De Moivre (polar form + 2*pi*k/n)."""
    w = sympify(expr_text(data))
    try:
        n = int(str(data.get('order') or data.get('point') or 2).strip())
    except (TypeError, ValueError):
        return False, '开方阶数 n 需为正整数'
    if n <= 0:
        return False, '开方阶数 n 需为正整数'
    n = min(n, HARD_MAX_ITERATIONS)
    r = simplify(Abs(w))
    theta = arg(w)
    modulus = root(r, n) if r.is_number else r ** (S(1) / n)
    items = [(w, note("被开方数 w")), (r, note("模 |w|")), (theta, note("辐角 θ = arg(w)"))]
    items.append((note(f"n 次根: z_k = |w|^{{1/n}}·exp(i·(θ + 2πk)/n), k = 0..{n-1}"), None))
    for k in range(n):
        angle = (theta + 2 * pi * k) / n
        zk = simplify(expand(modulus * exp(I * angle), complex=True))
        items.append((Eq(Symbol('z_%d' % k), zk), note(f"第 {k} 个根 (k={k})")))
    return True, align(items)


def complex_roots(data):
    z = get_var(data)
    expr = sympify(expr_text(data))
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    try:
        sol = solve(eq, z)
    except Exception:  # pylint: disable=broad-exception-caught
        sol = []
    if not sol:
        return False, '无法在复数域求解该方程的根'
    items = [(eq, note("求解 (复数域)"))]
    for i, s in enumerate(sol):
        items.append((Eq(z, simplify(expand(s, complex=True))), note(f"根 {i + 1}")))
    return True, align(items)
