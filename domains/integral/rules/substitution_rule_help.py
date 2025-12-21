from sympy import (
    Abs, Dummy, Eq, Expr, I, Integral, Pow, Rational, Symbol, Wild, acos, asin, atan,
    diff, latex, log, preorder_traversal, sec, simplify, sin, solve, sqrt, tan, together
)

from core import BaseStepGenerator
from utils import RuleFunctionReturn

sqrt_pow = Rational(1, 2)
minus_sqrt_pow = -Rational(1, 2)


def try_standard_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Attempt standard u-substitution for integrals of the form f(g(x)) g'(x) dx.

    Matches expressions where:
      - A unary function f(g(x)) is present (e.g., sin(x^2), log(cos(x)), exp(tan(x)))
      - The derivative g'(x) appears as a factor (up to a constant multiple)

    Returns the integrated result and explanation if successful.
    """
    # Normalize factors: handle both Mul and non-Mul (e.g., single sin(x+2))
    expr = together(expr)
    factors = list(expr.args) if expr.is_Mul else [expr]

    for factor in factors:
        if not factor.args or factor.is_constant() or factor == var:
            continue

        flag = False
        if factor.is_Pow and factor.args[1].has(var):
            flag = True  # eg. 3^g(x), etc.
        inner = factor.args[1] if flag else factor.args[0]   # g(x)

        for original_term in preorder_traversal(inner):
            if original_term == var or original_term.is_constant():
                continue

            check_list = [original_term]
            sqrt_term = sqrt(original_term)
            # Use a temporary variable with positive real assumptions to aid radical simplification
            _t = Symbol('t', real=True, positive=True)
            sqrt_term = simplify(sqrt_term.subs(var, _t)).subs(_t, var).replace(
                Abs, lambda arg: arg)
            if not sqrt_term.has(I):
                check_list.append(sqrt_term)

            for term in check_list:
                if term == var:
                    continue

                gp = simplify(diff(term, var))  # g'(x)
                if gp == 0:
                    continue

                try:
                    outer_part = expr / factor  # The rest of the integrand

                    # Check if outer_part = k * g'(x) for some constant k
                    ratio = simplify(outer_part / gp)
                    if not ratio.is_constant():
                        continue

                    # Substitute u = g(x)
                    u = step_gene.get_available_sym(var)
                    step_gene.subs_dict[u] = term
                    # Construct f(u)
                    f_u = simplify((expr/gp/ratio).subs(term, u))

                    new_expr = ratio * Integral(f_u, u)

                    ratio_latex = '' if ratio == 1 else (
                        '-' if ratio == -1 else latex(ratio))
                    gp_latex = '' if gp == 1 else (
                        '-' if gp == -1 else latex(gp))

                    explanation = (
                        f"换元法: 令 ${u.name} = {latex(term)}$, $d{u.name} = {gp_latex}\\,d{var.name}$, "
                        f"原式$ \\int {latex(expr)}\\,d{var.name}$ 化为 $ {ratio_latex} \\int {latex(f_u)}\\,d{u.name}$, "
                    )
                    return new_expr, explanation

                except (ZeroDivisionError, ValueError, TypeError, NotImplementedError) as e:
                    # Safely skip problematic cases (e.g., division by zero, unsupported ops)
                    print(f"Warning: Standard substitution failed for {e}")
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
        matched_sqrt = list(matches)[0]
        # Extract 'a' by solving matched_sqrt == pattern
        sol = matched_sqrt.match(pattern)
        if not sol or a not in sol:
            return None
        a_val = sol[a]

        x_sub = x_of_theta.subs(a, a_val)
        dx_dtheta = diff(x_sub, theta)

        try:
            new_expr = simplify((expr.subs(var, x_sub)*dx_dtheta)).replace(
                Abs, lambda arg: arg)
            int_theta = Integral(new_expr, theta).doit()
            if isinstance(int_theta, Integral):
                return None  # Integration failed

            new_subs = theta_of_x.subs(a, sqrt(a_val))
            if pattern in ((a+var**2)**minus_sqrt_pow, (var**2-a)**minus_sqrt_pow):
                result = log(Abs(sqrt(pattern.args[0]).subs(a, a_val)+var))
            elif pattern == (a+var**2)**sqrt_pow:
                result = a_val * \
                    log(Abs(sqrt(pattern.args[0]).subs(
                        a, a_val)+var))/2+var*sqrt(pattern.args[0]).subs(
                        a, a_val)/2
            elif pattern == (var**2-a)**sqrt_pow:
                result = -a_val * \
                    log(Abs(sqrt(pattern.args[0]).subs(
                        a, a_val)+var))/2+var*sqrt(pattern.args[0]).subs(
                        a, a_val)/2
            else:
                result = simplify(int_theta.subs(theta, new_subs))

            # Generate explanation
            form_str = latex(matched_sqrt)
            explanation = (
                f"三角代换：被积式含 ${form_str}$, 令 ${latex(var)} = {latex(x_sub)}$,"
                f"则 $d{latex(var)} = \\left({latex(dx_dtheta)}\\right)\\,d\\theta$, 积分化为 "
                f"$\\int {latex(new_expr)}\\,d\\theta = {latex(int_theta)}$, "
                f"回代 $\\theta = {latex(new_subs)}$ 得结果：${latex(result)} + C$."
            )
            return result, explanation

        except Exception as e:
            print(f"Error in try_trig_substitution: {e}")
            return None

    # Case 1: sqrt(a^2 − x^2) to x = a sin(theta)
    pattern1 = (a - var**2)**minus_sqrt_pow
    sub1 = sqrt(a) * sin(theta)
    back1 = asin(var / a)
    result = _apply_trig_sub(pattern1, sub1, back1)
    if result:
        return result

    pattern1_1 = (a - var**2)**sqrt_pow
    result = _apply_trig_sub(pattern1_1, sub1, back1)
    if result:
        return result

    # Case 2: sqrt(a^2 + x^2) to x = a tan(theta)
    pattern2 = (a + var**2)**minus_sqrt_pow
    sub2 = sqrt(a) * tan(theta)
    back2 = atan(var / a)
    result = _apply_trig_sub(pattern2, sub2, back2)
    if result:
        return result

    pattern2_1 = (a + var**2)**sqrt_pow
    result = _apply_trig_sub(pattern2_1, sub2, back2)
    if result:
        return result

    # Case 3: sqrt(x^2 − a^2) to x = a secθ=(theta)
    pattern3 = (var**2 - a)**minus_sqrt_pow
    sub3 = sqrt(a) * sec(theta)
    # Alternative: asec(var/a), but acos(a/x) is more common in textbooks
    back3 = acos(a / var)
    result = _apply_trig_sub(pattern3, sub3, back3)
    if result:
        return result

    pattern3_1 = (var**2 - a)**sqrt_pow
    result = _apply_trig_sub(pattern3_1, sub3, back3)
    if result:
        return result

    return None


def try_radical_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Attempt radical substitution for integrals containing nested radicals,
    e.g., sqrt(x), sqrt(x + sqrt(x)), (x + 1)^{2/3}, etc.

    Strategy:
      - Find a radical term r = (g(x))^{p/q} with -1 < p/q < 1.
      - Set u = r and g(x) = u^{q/p}, then solve for x if possible.
      - This implementation assumes the simplest case: r = x^{1/n} or r = (linear in x)^{1/n}.

    Returns the integrated result and explanation if successful.
    """
    # Collect all power expressions that are proper fractional powers of x
    candidates = []
    for atom in expr.atoms(Pow):
        base, _exp = atom.base, atom.exp
        if isinstance(_exp, Rational) and -1 < _exp < 1 and base.has(var):
            candidates.append(atom)

    # Sort by depth (simplest first) – optional but improves success rate
    candidates.sort(key=lambda r: r.count_ops())

    for rad in candidates:
        base, _exp = rad.base, rad.exp  # rad = base**exp
        q = _exp.q  # denominator
        p = _exp.p  # numerator, -1 < p < q

        # We set u = rad = base**(p/q), then base = u**(q/p)
        # To proceed, we need to express x in terms of u.
        # This is only feasible if base is linear in x (e.g., x, x+1, 2*x-3)
        if not base.is_polynomial(var) or base.as_poly(var).degree() != 1:
            continue  # Skip non-linear bases like x**2 + sqrt(x)

        # Solve base = u**(q/p) for x
        try:
            u = step_gene.get_available_sym(var)
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
            # Use a temporary variable with positive real assumptions to aid radical simplification
            _t = Symbol('t', real=True, positive=True)
            new_expr = simplify(new_expr.subs(u, _t)).subs(_t, u).replace(
                Abs, lambda arg: arg)

            final_result = Integral(new_expr, u)

            # Store substitution
            step_gene.subs_dict[u] = rad
            explanation = (
                f"根式代换：令 ${u.name} = {latex(rad)}$, 则由 ${latex(base)} = {u.name}^{{{latex(Rational(q, p))}}}$ 解得 "
                f"${var.name} = {latex(x_of_u)}$, "
                f"且 $d{var.name} = {latex(dx_du)}\\,d{u.name}$. 积分化为 "
                f"$\\int {latex(new_expr)}\\,d{u.name}$, "
            )
            return final_result, explanation

        except Exception as e:
            print(f"Error in try_radical_substitution: {e}")
            continue

    return None
