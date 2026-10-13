"""Numerical integration: trapezoidal and Simpson rules over [a, b]."""

from sympy import N, sympify

from domains.applied._common import align, expr_text, get_var, render_note as note, to_num


def numerical_integration(data):
    x = get_var(data)
    f = sympify(expr_text(data))
    a = to_num(data.get('lower'), 0)
    b = to_num(data.get('upper'), 1)
    n = int(data.get('order') or data.get('expr-depth') or 4)
    if not (a.is_number and b.is_number) or n <= 0:
        return False, '数值积分需要数值积分上下限'
    h = (b - a) / n
    ff = [f.subs(x, a + k * h) for k in range(n + 1)]
    trap = N((h / 2) * (ff[0] + 2 * sum(ff[1:-1]) + ff[-1]), 8)
    items = [(f, note("f(x)")), (trap, note(f"梯形法, n={n}"))]
    if n % 2 == 0:
        simp = N((h / 3) * (ff[0] + 4 * sum(ff[i] for i in range(1, n, 2))
                            + 2 * sum(ff[i] for i in range(2, n, 2)) + ff[-1]), 8)
        items.append((simp, note(f"辛普森法, n={n}")))
    return True, align(items)
