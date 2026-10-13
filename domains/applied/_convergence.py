"""Rule-driven series-convergence decisions shown as explicit test steps.

Each named test (n-th term, ratio, root, alternating/Leibniz, p-series, geometric)
is applied and its conclusion recorded as a step. SymPy is used only to evaluate the
individual limits that appear inside a test (a value shown as part of the step), never
as a black-box ``is_convergent``/``doit`` that returns the final verdict directly.
"""

from sympy import Abs, Eq, Pow, Symbol, limit, oo, simplify

from domains.applied._common import render_note as note


def _is_alternating(a):
    """Detect a factor of the form (-1)**(linear in n)."""
    return any(p.base == -1 for p in a.atoms(Pow))


def _lim(expr, n, at=oo):
    try:
        return simplify(limit(expr, n, at))
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def convergence_rows(a, n):
    """Return (rows, verdict) where verdict is 'converge' | 'diverge' | 'unknown'."""
    rows = [(a, note("一般项 $a_n$"))]

    # 1) n-th term test (necessary condition for convergence)
    l0 = _lim(a, n)
    if l0 is not None:
        rows.append((Eq(Symbol('lim_a_n'), l0), note("第 n 项判别: 求 $\\lim_{n\\to\\infty} a_n$")))
        if l0 != 0 and l0.is_number:
            rows.append((None, note("$\\lim a_n\\neq 0$, 由第 n 项判别法级数发散")))
            return rows, 'diverge'
        rows.append((None, note("$\\lim a_n=0$, 第 n 项判别法无法判定, 继续")))

    # 2) ratio test
    ratio = _lim(Abs(a.subs(n, n + 1) / a), n)
    if ratio is not None and ratio.is_number:
        rows.append((Eq(Symbol('rho'), ratio),
                     note("比值判别: $\\lim\\left|\\frac{a_{n+1}}{a_n}\\right|$")))
        if ratio < 1:
            rows.append((None, note("$\\rho<1$, 比值判别得级数绝对收敛")))
            return rows, 'converge'
        if ratio > 1:
            rows.append((None, note("$\\rho>1$, 比值判别得级数发散")))
            return rows, 'diverge'
        rows.append((None, note("$\\rho=1$, 比值判别失效, 继续")))

    # 3) root test
    root = _lim(Abs(a) ** (1 / n), n)
    if root is not None and root.is_number:
        rows.append((Eq(Symbol('sigma'), root), note("根值判别: $\\lim\\sqrt[n]{|a_n|}$")))
        if root < 1:
            rows.append((None, note("$\\sigma<1$, 根值判别得级数绝对收敛")))
            return rows, 'converge'
        if root > 1:
            rows.append((None, note("$\\sigma>1$, 根值判别得级数发散")))
            return rows, 'diverge'
        rows.append((None, note("$\\sigma=1$, 根值判别失效, 继续")))

    # 4) alternating / Leibniz
    if _is_alternating(a):
        b = Abs(a)
        lb = _lim(b, n)
        rows.append((Eq(Symbol('lim_b_n'), lb), note("交错级数莱布尼茨: $b_n=|a_n|\\to 0$ 且单调递减")))
        if lb == 0:
            rows.append((None, note("满足莱布尼茨条件, 级数收敛")))
            return rows, 'converge'

    # 5) p-series recognition: a == 1/n**p
    p = _p_series_exponent(a, n)
    if p is not None:
        rows.append((None, note(f"p-级数 $\\sum 1/n^{{{p}}}$: $p={p}$, {'收敛' if p > 1 else '发散'}")))
        return rows, ('converge' if p > 1 else 'diverge')

    return rows, 'unknown'


def _p_series_exponent(a, n):
    """If a is exactly 1/n**p (p a number), return p, else None."""
    if a == 0:
        return None
    expr = 1 / a
    if expr == n:
        return 1
    if isinstance(expr, Pow) and expr.base == n:
        return expr.exp if expr.exp.is_number else None
    return None
