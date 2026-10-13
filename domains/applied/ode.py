"""Ordinary differential equations (SymPy dsolve), with optional initial value."""

from sympy import Eq, Function, Symbol, dsolve, simplify, solve, sympify

from domains.applied._common import align, expr_text, get_var, render_note as note, to_num


def solve_ode(data):
    x = get_var(data)
    f = Function('f')(x)  # pylint: disable=not-callable
    raw = expr_text(data).replace("f'(x)", 'Derivative(f(x), x)') \
                          .replace("f''(x)", 'Derivative(f(x), (x, 2))')
    expr = sympify(raw)
    eq = expr if isinstance(expr, Eq) else Eq(expr, 0)
    try:
        sol = dsolve(eq, f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'ODE 无法求解: {exc}'

    items = [(eq, note("微分方程"))]
    x0 = to_num(data.get('point'))
    y0 = to_num(data.get('upper'))
    c1 = Symbol('C1')
    if x0 is not None and y0 is not None and c1 in sol.rhs.free_symbols:
        try:
            const_sol = solve(Eq(sol.rhs.subs(x, x0), y0), c1)
            if const_sol:
                sol = simplify(sol.subs(c1, const_sol[0]))
                items.append((Eq(f.subs(x, x0), y0), note("代入初始条件求常数")))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    items.append((sol, note("dsolve 求解 (通解或特解)")))
    return True, align(items)
