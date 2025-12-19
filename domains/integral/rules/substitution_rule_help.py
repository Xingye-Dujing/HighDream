from sympy import (
    Dummy, Eq, Expr, Integral, Pow, Rational, Symbol, Wild, acos, asin, atan,
    diff, exp, integrate, latex, log, sec, simplify, sin, solve, sqrt, tan
)

from utils import RuleFunctionReturn
from utils.latex_formatter import wrap_latex


def try_standard_substitution(expr: Expr, var: Symbol, u: Symbol) -> RuleFunctionReturn:
    """Attempt standard u-substitution for integrals of the form f(g(x)) g'(x) dx.

    Matches expressions where:
      - A unary function f(g(x)) is present (e.g., sin(x^2), log(cos(x)), exp(tan(x)))
      - The derivative g'(x) appears as a factor (up to a constant multiple)

    Returns the integrated result and explanation if successful.
    """
    # Normalize factors: handle both Mul and non-Mul (e.g., single sin(x))
    factors = list(expr.args) if expr.is_Mul else [expr]

    for factor in factors:
        # Only consider unary elementary functions: f(g(x))
        if not (factor.is_Function and len(factor.args) == 1):
            continue

        inner_expr = factor.args[0]  # g(x)
        if not inner_expr.has(var):
            continue

        gp = simplify(diff(inner_expr, var))  # g'(x)
        if gp.is_zero:
            continue

        try:
            outer_part = expr / factor  # The rest of the integrand
            if outer_part == 0:
                continue

            # Check if outer_part = k * g'(x) for some constant k
            ratio = simplify(outer_part / gp)
            if not ratio.is_constant():
                continue

            # Construct f(u): e.g., sin(g(x)) to sin(u)
            f_u = factor.func(u)

            # Integrate f(u) du
            inner_integral = simplify(integrate(f_u, u))
            if isinstance(inner_integral, Integral):
                # Integration failed or is unevaluated
                continue

            # Substitute back u = g(x)
            result = ratio * inner_integral.subs(u, inner_expr)

            var_latex = wrap_latex(var)
            explanation = (
                f"换元法: 令 $u = {latex(inner_expr)}$, $du = {latex(gp)}\\,d{var_latex}$, "
                f"原式化为 $ {latex(ratio)} \\int {latex(f_u)}\\,du = {latex(inner_integral)}$, "
                f"故积分为 ${latex(result)} + C$"
            )
            return result, explanation

        except (ZeroDivisionError, ValueError, TypeError, NotImplementedError):
            # Safely skip problematic cases (e.g., division by zero, unsupported ops)
            continue

    return None


def try_trig_substitution(expr: Expr, var: Symbol) -> RuleFunctionReturn:
    """
    Apply trigonometric substitution for integrals containing:
      - sqrt(a^2 − x^2)  to  x = a sin(theta)
      - sqrt(a^2 + x^2)  to  x = a tan(theta)
      - sqrt(x^2 − a^2)  to  x = a sec(theta)

    Uses pattern matching with Wild symbols to extract constant 'a'.

    Returns the integrated result and explanation if successful.
    """
    # Assume a > 0 and theta is real to help SymPy's simplify:
    # These assumptions are necessary for Integral to work correctly !!!
    a = Wild('a', exclude=[var], properties=[
             lambda x: x.is_positive])
    theta = Dummy('theta', real=True)

    # Helper: attempt substitution given pattern, sub_expr, d(x)/d(theta), and back-substitution
    def _apply_trig_sub(pattern, x_of_theta, theta_of_x):
        matches = expr.find(pattern)
        if not matches:
            return None

        # Take the first match (sufficient for teaching context)
        matched_sqrt = next(iter(matches))
        # Extract 'a' by solving matched_sqrt == pattern
        sol = matched_sqrt.match(pattern)
        if not sol or a not in sol:
            return None
        a_val = sol[a]

        x_sub = x_of_theta.subs(a, a_val)
        dx_dtheta = diff(x_sub, theta)

        try:
            new_expr = expr.subs(var, x_sub) * dx_dtheta
            new_expr = simplify(new_expr)
            int_theta = Integral(new_expr, theta).doit()
            if isinstance(int_theta, Integral):
                return None  # Integration failed

            result = int_theta.subs(theta, theta_of_x.subs(a, a_val))
            result = simplify(result)

            # Generate explanation
            form_str = latex(matched_sqrt)
            explanation = (
                f"三角代换：被积式含 ${form_str}$, 令 ${latex(var)} = {latex(x_sub)}$,"
                f"则 $d{latex(var)} = \\left({latex(dx_dtheta)}\\right)\\,d\\theta$, 积分化为 "
                f"$\\displaystyle \\int {latex(new_expr)}\\,d\\theta = {latex(int_theta)}$, "
                f"回代 $\\theta = {latex(theta_of_x.subs(a, a_val))}$ 得结果：${latex(result)} + C$."
            )
            return result, explanation

        except Exception:
            return None

    # Case 1: sqrt(a^2 − x^2) to x = a sin(theta)
    pattern1 = sqrt(a**2 - var**2)
    sub1 = a * sin(theta)
    back1 = asin(var / a)
    result = _apply_trig_sub(pattern1, sub1, back1)
    if result:
        return result

    # Case 2: sqrt(a^2 + x^2) to x = a tan(theta)
    pattern2 = sqrt(a**2 + var**2)
    sub2 = a * tan(theta)
    back2 = atan(var / a)
    result = _apply_trig_sub(pattern2, sub2, back2)
    if result:
        return result

    # Case 3: sqrt(x^2 − a^2) to x = a secθ=(theta)
    pattern3 = sqrt(var**2 - a**2)
    sub3 = a * sec(theta)
    # Alternative: asec(var/a), but acos(a/x) is more common in textbooks
    back3 = acos(a / var)
    result = _apply_trig_sub(pattern3, sub3, back3)
    if result:
        return result

    return None


def try_radical_substitution(expr: Expr, var: Symbol, u: Symbol) -> RuleFunctionReturn:
    """Attempt radical substitution for integrals containing nested radicals,
    e.g., sqrt(x), sqrt(x + sqrt(x)), (x + 1)^{2/3}, etc.

    Strategy:
      - Find a radical term r = (g(x))^{p/q} with 0 < p/q < 1.
      - Set u = r and g(x) = u^{q/p}, then solve for x if possible.
      - This implementation assumes the simplest case: r = x^{1/n} or r = (linear in x)^{1/n}.

    Returns the integrated result and explanation if successful.
    """
    # Collect all power expressions that are proper fractional powers of x
    candidates = []
    for atom in expr.atoms(Pow):
        base, _exp = atom.base, atom.exp
        if isinstance(_exp, Rational) and 0 < _exp < 1 and base.has(var):
            candidates.append(atom)

    # Sort by depth (simplest first) – optional but improves success rate
    candidates.sort(key=lambda r: r.count_ops())

    for rad in candidates:
        base, _exp = rad.base, rad.exp  # rad = base**exp
        q = _exp.q  # denominator
        p = _exp.p  # numerator, 0 < p < q

        # We set u = rad = base**(p/q), then base = u**(q/p)
        # To proceed, we need to express x in terms of u.
        # This is only feasible if base is linear in x (e.g., x, x+1, 2*x-3)
        if not base.is_polynomial(var) or base.as_poly(var).degree() != 1:
            continue  # Skip non-linear bases like x**2 + sqrt(x)

        # Solve base = u**(q/p) for x
        try:
            equation = Eq(base, u**(Rational(q, p)))
            sol = solve(equation, var)
            if not sol:
                continue
            x_of_u = sol[0]  # Take principal solution
        except Exception:
            continue

        try:
            dx_du = diff(x_of_u, u)
            # Substitute: every occurrence of 'var' to x(u), and rad to u
            new_expr = expr.subs(var, x_of_u)
            # Also replace any exact match of 'rad' with u (handles nested cases)
            new_expr = new_expr.subs(rad, u)
            new_expr = simplify(new_expr * dx_du)

            integral_u = simplify(integrate(new_expr, u))
            if isinstance(integral_u, Integral):
                continue  # Integration failed

            final_result = integral_u.subs(u, rad)
            final_result = simplify(final_result)

            explanation = (
                f"根式代换：令 $u = {latex(rad)}$, 则由 ${latex(base)} = u^{{{latex(Rational(q, p))}}}$ 解得 "
                f"${latex(var)} = {latex(x_of_u)}$, "
                f"且 $d{latex(var)} = {latex(dx_du)}\\,du$. 积分化为 "
                f"$\\int {latex(new_expr)}\\,du = {latex(integral_u)}$, "
                f"回代后得结果：${latex(final_result)} + C$."
            )
            return final_result, explanation

        except Exception:
            continue

    return None


def try_exp_log_substitution(expr: Expr, var: Symbol, u: Symbol) -> RuleFunctionReturn:
    """Apply substitution for integrals containing exp(x) or log(x):
    - If e^x appears: set u = e^x, then x = ln u, dx = du/u
    - If ln(x) appears: set u = ln x, then x = e^u, dx = e^u du

    This rule assumes the rest of the integrand becomes rational or elementary in u.

    Returns the integrated result and a step-by-step explanation if successful.
    """
    # Case 1: Substitution u = exp(var)
    if expr.has(exp(var)):
        # Substitute: x = ln(u), so exp(x) to u, and dx = du / u
        x_of_u = log(u)
        dx_du = diff(x_of_u, u)  # = 1/u

        # Replace all occurrences of 'var' with ln(u), and exp(var) with u
        new_expr = expr.subs(var, x_of_u)
        new_expr = new_expr.subs(exp(var), u)  # Ensure exp(x) is replaced
        new_expr = simplify(new_expr * dx_du)

        integral_u = simplify(integrate(new_expr, u))
        if isinstance(integral_u, Integral):
            pass  # Try next case
        else:
            final_result = integral_u.subs(u, exp(var))
            final_result = simplify(final_result)
            explanation = (
                f"指数代换：令 $u = e^{{{latex(var)}}}$, 则 ${latex(var)} = \\ln u$, "
                f"$d{latex(var)} = \\frac{{1}}{{u}}\\,du$。原积分化为 "
                f"$\\displaystyle \\int {latex(new_expr)}\\,du = {latex(integral_u)}$, "
                f"回代得结果：${latex(final_result)} + C$。"
            )
            return final_result, explanation

    # Case 2: Substitution u = log(var)
    if expr.has(log(var)):
        # Substitute: x = e^u, so log(x) to u, and dx = e^u du
        x_of_u = exp(u)
        dx_du = diff(x_of_u, u)  # = e^u

        # Replace all occurrences of 'var' with e^u, and log(var) with u
        new_expr = expr.subs(var, x_of_u)
        new_expr = new_expr.subs(log(var), u)
        new_expr = simplify(new_expr * dx_du)

        integral_u = simplify(integrate(new_expr, u))
        if isinstance(integral_u, Integral):
            return None
        final_result = integral_u.subs(u, log(var))
        final_result = simplify(final_result)
        explanation = (
            f"对数代换：令 $u = \\ln {latex(var)}$, 则 ${latex(var)} = e^{{u}}$, "
            f"$d{latex(var)} = e^{{u}}\\,du$. 原积分化为 "
            f"$\\displaystyle \\int {latex(new_expr)}\\,du = {latex(integral_u)}$, "
            f"回代得结果：${latex(final_result)} + C$."
        )
        return final_result, explanation

    return None
