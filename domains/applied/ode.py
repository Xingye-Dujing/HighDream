"""Ordinary differential equations.

First-order separable and linear equations are solved by the rule engine
(``_odesolver.first_order_ode``) with full method steps; any other form falls back
to SymPy ``dsolve`` (clearly labelled as a non-rule result).
"""

import re

from sympy import Eq, Expr, Function, Symbol, dsolve, simplify, solve

from utils.cache_utils import cached_sympify as sympify

from domains.applied._common import align, expr_text, get_var, render_note as note, to_num
from domains.applied._odesolver import MAX_ODE_ORDER, _max_deriv_order, solve_ode_rule


def _fit_constant(rows, x, x0, y0, f):
    """Solve the integration constant C from an initial condition on the last Eq."""
    if x0 is None or y0 is None or not rows:
        return None
    last = rows[-1][0]
    c = Symbol('C')
    if not isinstance(last, Eq) or c not in last.free_symbols:
        return None
    try:
        subs = {x: x0, f.subs(x, x0): y0}
        sol = solve(Eq(last.lhs.subs(subs), last.rhs.subs(subs)), c)
        return simplify(last.subs(c, sol[0])) if sol else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def solve_ode(data):
    x = get_var(data)
    f = Function('f')(x)  # pylint: disable=not-callable
    raw = expr_text(data).replace("f'(x)", 'Derivative(f(x), x)') \
                          .replace("f''(x)", 'Derivative(f(x), (x, 2))')
    # Also accept the two forms users naturally type / the palette inserts:
    # diff(y,x) and y'/y'' (and the LaTeX \frac{d}{dx} y, which cached_sympify
    # already turns into Derivative(y,x)).
    raw = re.sub(r'\bdiff\s*\(\s*([A-Za-z]\w*)\s*,\s*([A-Za-z]\w*)\s*\)',
                 r'Derivative(\1,\2)', raw)
    raw = re.sub(r"\b([A-Za-z]\w*)''", r'Derivative(\1, (x, 2))', raw)
    raw = re.sub(r"\b([A-Za-z]\w*)'", r'Derivative(\1, x)', raw)
    expr = sympify(raw)
    if not isinstance(expr, (Eq, Expr)):
        return False, '微分方程格式不正确：请输入方程或表达式'
    # The rule engine works with f(x); accept y as the usual dependent variable.
    if hasattr(expr, 'subs'):
        expr = expr.subs(Symbol('y'), f)
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    # Reject absurd derivative orders outright: the constant/high-order path would
    # otherwise iterate billions of simplify() calls and pin a worker thread.
    order = _max_deriv_order(eq, x, f)
    if order > MAX_ODE_ORDER:
        return False, f'微分方程阶数过高（{order} 阶），仅支持 {MAX_ODE_ORDER} 阶及以下，请检查导数阶数'

    rows, handled = solve_ode_rule(eq, x, f)
    if handled:
        items = [(eq, note("微分方程"))] + rows
        x0 = to_num(data.get('point'))
        y0 = to_num(data.get('upper'))
        particular = _fit_constant(rows, x, x0, y0, f)
        if particular is not None:
            items.append((particular, note("代入初始条件确定常数, 得特解")))
        return True, align(items)

    # Fallback: not a supported first-order rule form.
    try:
        raw_sols = dsolve(eq, f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'ODE 无法求解: {exc}'

    # dsolve returns a LIST for equations with several branches (e.g.
    # y'**2 - 1 = 0). Treating it as a single Eq crashed on ``sol.rhs`` whenever
    # an initial condition was supplied.
    solutions = raw_sols if isinstance(raw_sols, list) else [raw_sols]
    items = [(eq, note("微分方程"))]
    x0 = to_num(data.get('point'))
    y0 = to_num(data.get('upper'))
    c1 = Symbol('C1')
    fitted = False
    resolved = []
    for sol in solutions:
        if (x0 is not None and y0 is not None and isinstance(sol, Eq)
                and c1 in sol.rhs.free_symbols):
            try:
                const_sol = solve(Eq(sol.rhs.subs(x, x0), y0), c1)
                if const_sol:
                    sol = simplify(sol.subs(c1, const_sol[0]))
                    fitted = True
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        resolved.append(sol)
    if fitted:
        items.append((Eq(f.subs(x, x0), y0), note("代入初始条件求常数")))
    for sol in resolved:
        items.append((sol, note("非一阶可分离/线性型, 由 dsolve 给出结果(展示用)")))
    return True, align(items)
