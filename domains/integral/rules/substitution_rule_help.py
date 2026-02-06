from sympy import (
    Add, Abs, Dummy, Eq, Expr, I, Integral, Pow, Rational, Symbol, Wild, acos, asin, atan,
    cancel, diff, fraction, latex, log, integrate, preorder_traversal, sec, simplify,
    sin, solve, sqrt, symbols, tan, together
)

from core import BaseStepGenerator
from utils import RuleFunctionReturn, is_elementary_expression

sqrt_pow = Rational(1, 2)
minus_sqrt_pow = -Rational(1, 2)


def try_standard_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Attempt standard u-substitution for integrals of the form f(g(x)) g'(x) dx.

    Matches expressions where:
      - A unary function f(g(x)) is present (e.g., sin(x^2), log(cos(x)), exp(tan(x)))
      - The derivative g'(x) appears as a factor (up to a constant multiple)

    Returns the integrated result and explanation if successful.
    """
    # Normalize expression into a list of factors (even for non-Mul)
    expr = together(expr)
    factors = list(expr.args) if expr.is_Mul else [expr]

    for factor in factors:
        if not factor.args or factor.is_constant() or factor.equals(var):
            continue

        # Extract internal factors for subsequent traversal
        flag = False
        # Look for unary functions like 3^g(x), etc.
        if factor.is_Pow and factor.args[1].has(var):
            flag = True
        # Look for unary functions like sin(g(x)), log(g(x)),1/g(x)^2 etc.
        inner = factor.args[1] if flag else factor.args[0]

        term_list = [factor]
        # If it's 1/den, extract den specially, e.g. x/sqrt(x^2+1) extracts sqrt(x^2+1)
        num, den = fraction(factor)
        if num.equals(1):
            term_list.append(den)
        term_list += list(preorder_traversal(inner))

        for original_term in term_list:
            # Preventing infinite loops
            if original_term.equals(var) or original_term.equals(-var) or original_term.is_constant():
                continue

            check_list = [original_term]
            # Introducing sqrt_term is to handle implicit f(x)^2 cases like x/(x**4+1), x**x*(log(x)+1)/(x**(2*x)+1)
            sqrt_term = sqrt(original_term)
            # Use a temporary variable with positive real assumptions to aid radical simplification
            _t = Dummy('t', real=True, positive=True)
            sqrt_term = simplify(sqrt_term.subs(var, _t)).subs(_t, var).replace(
                Abs, lambda arg: arg)
            if not sqrt_term.has(I) and not sqrt_term.equals(var) and not sqrt_term.equals(-var):
                check_list.append(sqrt_term)

            for term in check_list:

                gp = simplify(diff(term, var))  # g'(x)
                if gp.equals(0):  # g'(x) = 0
                    continue

                initial_ratio = simplify(expr/gp)
                # Special case: g'(x) dx, let u = g(x), g'(x) dx = 1 du
                if initial_ratio.is_constant():
                    # Substitute u = g(x)
                    u = step_gene.get_available_sym(var)
                    step_gene.subs_dict[u] = term
                    explanation = (
                        f"换元法: 令 ${u.name} = {latex(term)}$, $d{u.name} = {latex(gp)}\\,d{var.name}$, "
                        f"原式$ \\int {latex(expr)}\\,d{var.name}$ 化为 $ \\int {latex(initial_ratio)}\\,d{u.name}$"
                    )
                    return Integral(initial_ratio, u), explanation

                try:
                    # Compute the "remaining part" = expr / factor
                    outer_part = expr / factor  # The rest of the integrand

                    # Check if outer_part = k * g'(x) for some constant k
                    ratio = simplify(outer_part / gp)
                    if not ratio.is_constant():
                        continue

                    # Substitute u = g(x)
                    u = step_gene.get_available_sym(var)

                    # Construct f(u)
                    f_u = simplify((initial_ratio/ratio).subs(term, u))

                    if f_u.has(var):
                        continue

                    step_gene.subs_dict[u] = term
                    new_expr = ratio * Integral(f_u, u)

                    ratio_latex = '' if ratio.equals(1) else (
                        '-' if ratio.equals(-1) else latex(ratio))
                    gp_latex = '' if gp.equals(1) else (
                        '-' if gp.equals(-1) else latex(gp))

                    explanation = (
                        f"换元法: 令 ${u.name} = {latex(term)}$, $d{u.name} = {gp_latex}\\,d{var.name}$, "
                        f"原式$ \\int {latex(expr)}\\,d{var.name}$ 化为 $ {ratio_latex} \\int {latex(f_u)}\\,d{u.name}$"
                    )

                    return new_expr, explanation

                except (ZeroDivisionError, ValueError, TypeError, NotImplementedError):
                    # Safely skip problematic cases (e.g., division by zero, unsupported ops)
                    # print(f"Warning: Standard substitution failed for {e}")
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

    # Note: cancel() is necessary to help simplify
    # Simplify to the lowest terms to prevent matching issues with the result
    # (e.g., sqrt(x**2-8)/(x**2-8) to 1/sqrt(x**2-8))
    expr = cancel(expr)

    num, den = fraction(expr)
    if den != 1 and list(den.args):
        den_arg = list(den.args)[0]
        if num != 1 or not num.is_polynomial() or not den_arg.is_polynomial():
            return None

    # Assume a > 0 and theta is real to help SymPy's simplify:
    # These assumptions are necessary for Integral to work correctly !!!
    a = Wild('a', exclude=[var], properties=[
             lambda x: x.is_positive])
    # Theta is real and positive to help simplify
    theta = Dummy('theta', real=True, positive=True)

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
            # Note: inverse=True is necessary to handle cases like asin(sin(x)) == x
            new_expr = simplify(expr.subs(var, x_sub)*dx_dtheta, inverse=True).replace(
                Abs, lambda arg: arg)
            int_theta = simplify(integrate(new_expr, theta))
            if not is_elementary_expression(int_theta):
                return None

            new_subs = theta_of_x.subs(a, sqrt(a_val))

            # Cannot use isinstance and .has(log) to check
            if len(int_theta.args) == 2:
                if pattern in ((a+var**2)**minus_sqrt_pow, (var**2-a)**minus_sqrt_pow):
                    result = log(Abs(sqrt(pattern.args[0]).subs(a, a_val)+var))
                elif pattern == (a+var**2)**sqrt_pow:
                    result = a_val * \
                        log(Abs(sqrt(pattern.args[0]).subs(
                            a, a_val)+var))/2+var*sqrt(pattern.args[0]).subs(
                            a, a_val)/2
                elif pattern.equals((var**2-a)**sqrt_pow):
                    result = -a_val * \
                        log(Abs(sqrt(pattern.args[0]).subs(
                            a, a_val)+var))/2+var*sqrt(pattern.args[0]).subs(
                            a, a_val)/2
                else:
                    result = simplify(int_theta.subs(theta, new_subs))
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


def try_undetermined_coeffs_for_radicals(expr: Expr, var: Symbol) -> RuleFunctionReturn:
    """Attempt to solve integrals of the form (P(x))/(sqrt(ax^2+bx+c)) dx using undetermined coefficients method.

    For integrals of the form  (P(x))/(sqrt(ax^2+bx+c)) dx where P(x) is a polynomial,
    we assume the antiderivative has the form Q(x)*sqrt(ax^2+bx+c) + K ((1/sqrt(ax^2+bx+c)) dx)
    where Q(x) is a polynomial of degree deg(P)-1, and K is a constant.
    """
    # Check if expression is of the form P(x)/sqrt(ax^2+bx+c)
    num, den = fraction(expr)
    if num.equals(1) or not isinstance(den, Pow) or list(den.args)[1] != sqrt_pow:
        return None

    # Extract ax^2 + bx + c from sqrt(ax^2+bx+c)
    radicand = list(den.args)[0]

    # Check if it's a quadratic expression ax^2 + bx + c
    if not radicand.is_Add or len(radicand.args) < 2 or len(radicand.args) > 3:
        return None
    quad_poly = radicand.expand()
    if not quad_poly.is_polynomial(var) or quad_poly.as_poly(var).degree() != 2:
        return None

    # Check if numerator is a polynomial (any degree)
    if not num.is_polynomial(var):
        return None
    num_degree = num.as_poly(var).degree()
    if num_degree < 0:  # Constant numerator (degree 0)
        return None

    # Generate polynomial Q(x) of degree (num_degree - 1)
    # Q(x) = C0 + C1*x + C2*x**2 + ... + C_{m}*x**m where m = num_degree - 1
    m = num_degree - 1
    coeffs = symbols(f'C0:{m+1}')  # C0, C1, ..., Cm
    Q = sum(coeffs[i] * var**i for i in range(m+1))

    # Constant K for the integral term
    K = Symbol('K')

    # Assumed antiderivative: Q(x)*sqrt(radicand) + K * ∫(1/sqrt(radicand)) dx
    assumed_antiderivative = Q * \
        sqrt(radicand) + K * Integral(1/sqrt(radicand), var)

    # Differentiate assumed antiderivative
    diff_assumed = diff(assumed_antiderivative, var).expand().simplify()

    # Clear denominator by multiplying by sqrt(radicand)
    lhs_multiplied = (diff_assumed * sqrt(radicand)).expand()
    rhs_multiplied = num.expand()

    # Equation: lhs_multiplied - rhs_multiplied = 0
    equation_to_solve = (lhs_multiplied - rhs_multiplied).expand()

    # Collect coefficients for powers of var
    collected_eq = equation_to_solve.as_poly(var)
    if collected_eq is None:
        return None

    # Get all coefficients (should be zero)
    coeff_equations = list(collected_eq.all_coeffs())

    # Solve for all unknowns: [C0, C1, ..., Cm, K]
    unknowns = list(coeffs) + [K]
    solutions = solve(coeff_equations, unknowns, dict=True)

    if not solutions or len(solutions) == 0:
        return None

    # Use first solution
    sol = solutions[0]
    Q_solved = Q.subs(sol)
    K_val = sol[K]

    # Construct final result
    integral_sqrt_inv = Integral(1/sqrt(radicand), var)
    result = Q_solved * sqrt(radicand) + K_val * integral_sqrt_inv

    # Generate coefficient display string
    coeff_values = []
    for i, c in enumerate(coeffs):
        val = sol[c]
        coeff_values.append(f"C_{i} = {latex(val)}")

    coeff_values.append(f"K = {latex(K_val)}")

    if not coeff_values:  # Handle case where all coefficients are zero
        coeff_display = "所有系数均为零"
    else:
        coeff_display = rf",\;".join(coeff_values)

    var_latex, radicand_latex, expr_latex, result_latex = latex(
        var), latex(radicand), latex(expr), latex(result)

    explanation = (
        rf"形如$\int \frac{{P(x)}}{{\sqrt{{ax^2+bx+c}}}}dx\;$使用待定系数法:\;"
        rf"设原函数为\;$Q(x) \sqrt{{{radicand_latex}}} + K \int \frac{{1}}{{\sqrt{{{radicand_latex}}}}}dx$,\;"
        rf"其中\;$Q(x)$为\;{m}\;次多项式.\;通过求导并比较系数, 解得：\;${coeff_display}$,\;即\;"
        rf"$\int {expr_latex}\,d{var_latex} = {result_latex}$"
    )

    return result, explanation


def try_radical_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Attempt radical substitution for integrals containing nested radicals,
    e.g., sqrt(x), sqrt(x + sqrt(x)), (x + 1)^{2/3}, etc.

    Strategy:
      - Find a radical term r = (g(x))^{p/q} with -1 < p/q < 1.
      - Set u = r and g(x) = u^{q/p}, then solve for x if possible.
      - This implementation assumes the simplest case: r = x^{1/n} or r = (linear in x)^{1/n}.

    Returns the integrated result and explanation if successful.
    """
    # Exclude log(f(x)+sqrt(f(x)))
    if isinstance(expr, log) and isinstance(expr.args[0], Add):
        return None

    # Collect all power expressions that are proper fractional powers of x
    candidates = [atom for atom in expr.atoms(Pow) if isinstance(
        atom.exp, Rational) and -1 < atom.exp < 1 and atom.base.has(var)]

    # Sort by depth (simplest first) – optional but improves success rate
    candidates.sort(key=lambda r: r.count_ops())

    for rad in candidates:
        base, _exp = rad.base, rad.exp  # rad = base**exp
        q = _exp.q  # denominator
        p = _exp.p  # numerator

        # We set u = rad = base**(p/q), then base = u**(q/p)
        # To proceed, we need to express x in terms of u.
        # Solve base = u**(q/p) for x
        try:
            u = step_gene.get_available_sym(var)
            equation = Eq(base, u**(Rational(q, p)))
            sol = solve(equation, var)
            if not sol:
                continue
            x_of_u = sol[0]
            # Assume that the variable is always positive.
            # Fix 1/((2*x^2+1)*sqrt(x^2+1)) dx
            if x_of_u.has(-1) and -x_of_u in sol:
                x_of_u = -x_of_u
        except Exception:
            continue

        try:
            dx_du = diff(x_of_u, u)
            # Replace any exact match of 'rad' with u
            new_expr = expr.subs(rad, u)
            # Substitute: every occurrence of 'var' to x(u), and rad to u
            new_expr = new_expr.subs(var, x_of_u)
            if new_expr.has(var) or new_expr.has(rad):
                continue
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
                f"$\\int {latex(new_expr)}\\,d{u.name}$"
            )
            return final_result, explanation

        except Exception as e:
            print(f"Error in try_radical_substitution: {e}")
            continue

    return None
