# -*- coding: utf-8 -*-
"""Regression tests for the added first-order ODE rule methods.

Each checks two things: (a) the equation is solved by the *rule* engine (``handled`` is
True, i.e. it does not fall back to ``dsolve`` — the project's no-black-box rule), and
(b) the produced general solution actually satisfies the ODE (residual is zero)."""

from sympy import Abs, Derivative, Eq, Function, N, Symbol, diff, simplify

from domains.applied._odesolver import solve_ode_rule
from domains.applied.ode import solve_ode

x = Symbol('x')
y = Symbol('y')
F = Function('f')(x)  # pylint: disable=not-callable


def _implicit_residual(eq, phi):
    """Residual of an implicit solution: y' from the level set vs the ODE right side."""
    rows, handled = solve_ode_rule(eq, x, F)
    assert handled, "fell back to dsolve"
    last = rows[-1][0]
    g = (last.lhs - last.rhs).subs(F, y).replace(Abs, lambda a: a)
    yp = simplify(-diff(g, x) / diff(g, y))
    return simplify(yp - phi)


def test_homogeneous_ode_y_over_x():
    # y' = (x + y)/(x - y) is homogeneous degree 0 (not separable/linear/exact)
    eq = Eq(Derivative(F, x), (x + F) / (x - F))
    assert _implicit_residual(eq, (x + y) / (x - y)) == 0


def test_exact_integrating_factor_ode():
    # (y^2 - 3x^2) y' + 2xy = 0 needs a mu(y) = y^-4 integrating factor
    eq = Eq((F ** 2 - 3 * x ** 2) * Derivative(F, x) + 2 * x * F, 0)
    phi = -2 * x * y / (y ** 2 - 3 * x ** 2)
    assert _implicit_residual(eq, phi) == 0


def test_bernoulli_ode():
    # y' + y = x y^3 (Bernoulli n=3), solved via u = y^(1-n)
    eq = Eq(Derivative(F, x) + F, x * F ** 3)
    rows, handled = solve_ode_rule(eq, x, F)
    assert handled, "fell back to dsolve"
    ysol = rows[-1][0].rhs
    dy = diff(ysol, x)
    resid = simplify(eq.lhs.subs({F: ysol, Derivative(F, x): dy})
                     - eq.rhs.subs({F: ysol}))
    assert resid == 0


def test_cauchy_euler_distinct_roots():
    # x^2 y'' + x y' - 4y = 0 -> indicial m^2 - 4 = 0 -> basis {x^2, x^-2}
    xp = Symbol('x', positive=True)
    fp = Function('f')(xp)  # pylint: disable=not-callable
    eq = Eq(xp ** 2 * Derivative(fp, (xp, 2)) + xp * Derivative(fp, xp) - 4 * fp, 0)
    rows, handled = solve_ode_rule(eq, xp, fp)
    assert handled, "fell back to dsolve"
    yc = rows[-1][0].rhs
    resid = simplify(eq.lhs.subs({fp: yc, Derivative(fp, xp): diff(yc, xp),
                                  Derivative(fp, (xp, 2)): diff(yc, xp, 2)}) - eq.rhs.subs({fp: yc}))
    assert resid == 0


def test_cauchy_euler_repeated_and_complex():
    xp = Symbol('x', positive=True)
    fp = Function('f')(xp)  # pylint: disable=not-callable
    # repeated root (m=2): x^2 y'' - 3x y' + 4y = 0 -> (C1 + C2 log x) x^2
    eq_r = Eq(xp ** 2 * Derivative(fp, (xp, 2)) - 3 * xp * Derivative(fp, xp) + 4 * fp, 0)
    rows, handled = solve_ode_rule(eq_r, xp, fp)
    assert handled and 'log' in str(rows[-1][0].rhs)
    # complex roots: x^2 y'' + y = 0 -> x^{1/2}(C1 cos(b log x) + C2 sin(b log x))
    eq_c = Eq(xp ** 2 * Derivative(fp, (xp, 2)) + fp, 0)
    rows_c, handled_c = solve_ode_rule(eq_c, xp, fp)
    assert handled_c and 'log' in str(rows_c[-1][0].rhs)


def _resid_2nd_or_higher(eq, order):
    """Solve and substitute back; returns the residual (must simplify to 0)."""
    rows, handled = solve_ode_rule(eq, x, F)
    assert handled, "fell back to dsolve"
    yc = rows[-1][0].rhs
    subs = {F: yc}
    for k in range(1, order + 1):
        subs[Derivative(F, (x, k))] = diff(yc, x, k)
    return simplify(eq.lhs.subs(subs) - eq.rhs.subs({F: yc}))


def test_higher_order_constant_coeff_homogeneous():
    # y''' - y' = 0 -> roots 0, 1, -1 -> span {1, e^x, e^-x}
    assert _resid_2nd_or_higher(
        Eq(Derivative(F, (x, 3)) - Derivative(F, x), 0), 3) == 0
    # triple root (r=1): (r-1)^3 -> (C1 + C2 x + C3 x^2) e^x
    assert _resid_2nd_or_higher(
        Eq(Derivative(F, (x, 3)) - 3 * Derivative(F, (x, 2)) + 3 * Derivative(F, x) - F, 0), 3) == 0
    # 4th order, two conjugate pairs: y'''' + 5 y'' + 4 y = 0
    assert _resid_2nd_or_higher(
        Eq(Derivative(F, (x, 4)) + 5 * Derivative(F, (x, 2)) + 4 * F, 0), 4) == 0
    # mixed real + complex: y''' + y'' + y' + y = 0
    assert _resid_2nd_or_higher(
        Eq(Derivative(F, (x, 3)) + Derivative(F, (x, 2)) + Derivative(F, x) + F, 0), 3) == 0


def test_applied_align_no_none_sentinel_leak():
    # Rules inject display-only integral sub-steps with the string 'None'; the applied
    # align renderer must drop them, never printing a literal "None". A Bernoulli solve
    # embeds an integrating-factor integral that triggers those sub-steps.
    ok, res = solve_ode({"expression": "f'(x)+f(x)-x*f(x)**3", "variable": "x"})
    assert ok and "\\begin{align}" in res and "None" not in res


def _numeric_residual(eq, order, points):
    """Solve via dispatch; return the max |ODE residual| over concrete (x, C1, C2)."""
    rows, handled = solve_ode_rule(eq, x, F)
    assert handled, "fell back to dsolve"
    yc = rows[-1][0].rhs.replace(Abs, lambda a: a)   # branch-local (constant absorbs sign)
    subs = {F: yc}
    for k in range(1, order + 1):
        subs[Derivative(F, (x, k))] = diff(yc, x, k)
    lhs = eq.lhs.subs(subs)
    rhs = eq.rhs.subs({F: yc})
    worst = 0.0
    for xv, c1v, c2v in points:
        d = {x: xv, Symbol('C1'): c1v, Symbol('C2'): c2v, Symbol('C'): c1v}
        worst = max(worst, abs(float(N((lhs - rhs).subs(d)))))
    return worst


def test_reduction_of_order_missing_y():
    # y'' = (y')^2 -> p' = p^2 -> y = C2 - log(C1 + x)  (reached only by the reducer)
    assert _numeric_residual(Eq(Derivative(F, (x, 2)) - Derivative(F, x) ** 2, 0), 2,
                             [(2.3, 0.5, 1.1), (4.0, -0.3, 0.0), (1.5, 2.0, -2.0)]) < 1e-6
    # y'' = y'/x (variable coefficient) -> y = C2 + k x^2
    assert _numeric_residual(Eq(Derivative(F, (x, 2)) - Derivative(F, x) / x, 0), 2,
                             [(2.3, 0.5, 1.1), (4.0, -0.3, 0.0)]) < 1e-6
