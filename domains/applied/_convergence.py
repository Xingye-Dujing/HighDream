"""Rule-driven series-convergence decisions shown as explicit test steps.

Each named test (n-th term, ratio, root, alternating/Leibniz, p-series, geometric)
is applied and its conclusion recorded as a step. SymPy is used only to evaluate the
individual limits that appear inside a test (a value shown as part of the step), never
as a black-box ``is_convergent``/``doit`` that returns the final verdict directly.
"""

from sympy import Abs, Eq, Integral, Pow, Symbol, limit, oo, simplify

from domains.applied._common import integral_value, render_note as note


def _is_alternating(a):
    """Detect a factor of the form (-1)**(linear in n)."""
    return any(p.base == -1 for p in a.atoms(Pow))


def _lim(expr, n, at=oo):
    try:
        return simplify(limit(expr, n, at))
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def convergence_rows(a, n, absolute=False):
    """Return (rows, verdict) where verdict is 'converge' | 'diverge' | 'conditional'
    | 'unknown'. When ``absolute`` is True the alternating branch is skipped (used to
    test the convergence of $\\sum|a_n|$ for the absolute-vs-conditional distinction)."""
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
    if _is_alternating(a) and not absolute:
        b = Abs(a)
        lb = _lim(b, n)
        rows.append((Eq(Symbol('lim_b_n'), lb), note("交错级数莱布尼茨: $b_n=|a_n|\\to 0$ 且单调递减")))
        if lb == 0:
            # classify absolute vs conditional via the positive series $\\sum|a_n|$
            _abs_rows, abs_v = convergence_rows(b, n, absolute=True)
            if abs_v == 'converge':
                rows.append((None, note("满足莱布尼茨条件且 $\\sum|a_n|$ 收敛, 级数绝对收敛")))
                return rows, 'converge'
            if abs_v == 'diverge':
                rows.append((None, note("莱布尼茨判别收敛, 但 $\\sum|a_n|$ 发散, 故为条件收敛")))
                return rows, 'conditional'
            rows.append((None, note("满足莱布尼茨条件, 级数收敛($\\sum|a_n|$ 敛散性未定)")))
            return rows, 'converge'

    # 5) p-series recognition: a == 1/n**p
    p = _p_series_exponent(a, n)
    if p is not None:
        rows.append((None, note(f"p-级数 $\\sum 1/n^{{{p}}}$: $p={p}$, {'收敛' if p > 1 else '发散'}")))
        return rows, ('converge' if p > 1 else 'diverge')

    # 6) geometric recognition: a == r**n with constant ratio r
    geo = _geometric_ratio(a, n)
    if geo is not None:
        rows.append((Eq(Symbol('r'), geo), note("几何(等比)级数, 公比 $r$")))
        if Abs(geo) < 1:
            rows.append((None, note("$|r|<1$, 几何级数收敛")))
            return rows, 'converge'
        rows.append((None, note("$|r|\\geq 1$, 几何级数发散")))
        return rows, 'diverge'

    # 7) limit comparison against p-series (p = 1 and p = 2)
    for p_try in (1, 2):
        lc = _lim(a * n ** p_try, n)
        if lc is not None and lc.is_number and lc != 0 and lc != oo and Abs(lc) != oo:
            rows.append((Eq(Symbol('L_cmp'), lc),
                         note(f"极限比较判别: 与 $\\sum 1/n^{{{p_try}}}$ 比较, $\\lim a_n n^{{{p_try}}}$")))
            verdict = 'converge' if p_try > 1 else 'diverge'
            zh = '收敛' if p_try > 1 else '发散'
            rows.append((None, note(f"极限为正常数, 与 p={p_try} 的 p-级数同敛散, 故级数{zh}")))
            return rows, verdict

    # 8) integral test (positive terms): compare with improper integral via the engine
    x = Symbol('x')
    f = a.subs(n, x)
    try:
        anti = integral_value(f, x)
        finf = _lim(anti, x, oo)
        if finf is not None and finf.is_finite:
            rows.append((Integral(f, (x, 1, oo)), note("积分判别: $\\int_1^{\\infty} f(x)\\,dx$ 收敛")))
            return rows, 'converge'
        if finf in (oo, -oo):
            rows.append((Integral(f, (x, 1, oo)), note("积分判别: $\\int_1^{\\infty} f(x)\\,dx$ 发散")))
            return rows, 'diverge'
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    return rows, 'unknown'


def _geometric_ratio(a, n):
    """If a == r**n for a constant r, return r, else None."""
    if isinstance(a, Pow) and a.exp == n:
        return a.base if a.base.is_number or not a.base.has(n) else None
    return None


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
