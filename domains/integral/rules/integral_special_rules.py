from sympy import (
    Abs, Expr, I, Integer, Integral, Mul, Wild, Rational, Symbol, diff, exp, fraction,
    integrate, latex, log, powsimp, preorder_traversal, simplify, sqrt, together, sin, cos,
    tan, cot, sec, csc
)
from sympy.simplify.fu import fu

from utils import (
    MatcherFunctionReturn, RuleContext, RuleFunctionReturn,
    can_use_weierstrass, has_radical, is_elementary_expression
)
from utils.latex_formatter import wrap_latex
from domains.integral import (
    select_parts_u_dv,
    try_radical_substitution,
    try_standard_substitution,
    try_trig_substitution,
    special_add_split_exp_term,
    handle_fx_mul_exp_gx
)

sqrt_pow = Rational(1, 2)
minus_sqrt_pow = -Rational(1, 2)


def logarithmic_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply integration rule for f'(x)/f(x) which integrates to ln|f(x)| + C.

    This rule handles expressions where the integrand is the derivative of a function
    divided by that same function, which results in the natural logarithm of the
    absolute value of that function.
    """
    find = False

    var = context['variable']

    # More general approach for f'(x)/f(x)
    numerator, denominator = expr.as_numer_denom()
    f_prime = diff(denominator, var)
    ratio = simplify(numerator / f_prime)

    var_latex, f_x_latex = latex(var), latex(denominator)
    if ratio.is_constant():
        find = True

    # Also check if expression can be rearranged to this form
    if not find and expr.is_Mul:
        # Look for patterns like f'(x) * (1/f(x))
        factors = expr.args
        for factor in factors:
            if factor.is_Pow and factor.args[1] == -1:
                f_x = factor.args[0]
                f_prime = diff(f_x, var)
                # Create the rest of the multiplication
                other_factors = [f for f in factors if f != factor]
                remaining = Mul(*other_factors) if other_factors else 1

                ratio = simplify(remaining - f_prime)

                if ratio.is_constant():
                    var_latex, f_x_latex = latex(var), latex(f_x)
                    break

    if ratio == 1:
        result = ratio * log(abs(denominator))
        explaination = (
            f"对数积分法则($\\frac{{f'({var_latex})}}{{f({var_latex})}}$ 形式): $"
            f"\\int \\frac{{{latex(f_prime)}}}{{{f_x_latex}}}\\,d{var_latex} = "
            f"\\ln|{f_x_latex}| + C$"
        )
    elif ratio == -1:
        result = -log(abs(denominator))
        explaination = (
            f"对数积分法则($\\frac{{f'({var_latex})}}{{f({var_latex})}}$ 形式): $"
            f"\\int {latex(expr)}\\,d{var_latex} = "
            f"- \\int \\frac{{{latex(f_prime)}}}{{{f_x_latex}}}\\,d{var_latex} = "
            f"- \\ln|{f_x_latex}| + C$"
        )
    else:
        result = ratio * log(abs(denominator))
        explaination = (
            f"对数积分法则($\\frac{{f'({var_latex})}}{{f({var_latex})}}$ 形式): $"
            f"\\int {latex(expr)}\\,d{var_latex} = "
            f"{latex(ratio)} \\int \\frac{{{latex(f_prime)}}}{{{f_x_latex}}}\\,d{var_latex} = "
            f"{latex(ratio)} \\ln|{f_x_latex}| + C$"
        )
    return result, explaination


def parts_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply integration by parts: u dv = u v − v du.

    Uses the ILAET heuristic to select u from a product of two factors:
      I — Inverse trig     (e.g., asin(x))
      L — Logarithmic      (e.g., log(x))
      A — Algebraic        (e.g., x, x**2)
      E — Exponential      (e.g., exp(x))
      T — Trigonometric    (e.g., sin(x))

    Works with both products (multi-factor) and single-factor expressions.
    Handle single-factor expressions by setting dv = 1
    """
    var = context['variable']

    # Handle single-factor expressions by setting dv = 1
    if not isinstance(expr, Mul):
        u = expr
        dv = Integer(1)
    else:
        # For multi-factor expressions, use the ILAET heuristic
        u, dv = select_parts_u_dv(expr, var)

    du = simplify(diff(u, var))
    v = simplify(integrate(dv, var))

    if not is_elementary_expression(v):
        return None

    u_v = simplify(u * v)
    v_du = simplify(v * du)
    result = u_v - Integral(v_du, var)
    var_latex, expr_latex, dv_latex, v_du_latex = wrap_latex(
        var, expr, dv, v_du)
    return result, (
        f"分部积分法(ILAET 选择 $u={latex(u)}$, $dv={dv_latex}\\,d{var_latex}$): $"
        f"\\int {expr_latex}\\,d{var_latex} = "
        f"{latex(u_v)} - \\int {v_du_latex} \\,d{var_latex}$"
    )


def substitution_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply substitution (u-substitution) for integration.

    Tries the following strategies in order:
      1. Standard form: f(g(x)) * g'(x) to f(u) du
      2. Trigonometric substitution (e.g., sqrt(a^2−x^2), sqrt(a^2+x^2), sqrt(x^2−a^2))
      3. Radical substitution (e.g., sqrt[n]{ax + b})

    Returns a transformed integral or None if no substitution applies.
    """
    var = context['variable']
    step_gene = context['step_generator']

    # Strategy 1: Standard chain-rule pattern f(g(x)) * g'(x)
    result = try_standard_substitution(expr, var, step_gene)
    if result:
        return result

    # Strategy 2: Trigonometric substitutions for sqrt(a^2 +- x^2) etc.
    result = try_trig_substitution(expr, var)
    if result:
        return result

    # Strategy 3: Substitutions for nested radicals (e.g., (ax + b)^{1/n})
    result = try_radical_substitution(expr, var, step_gene)
    if result:
        return result

    return None


def f_x_mul_exp_g_x_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Handle f(x)^exp(g(x)) case by recognition of differential forms.

      - (1+x-1/x)*exp(x+1/x) dx
      - (1+x-1/x)*exp(x+1/x)+x dx
      - (1+x-1/x)*exp(x+1/x)*sin(x) dx
      - exp(cos(x))*(1-x*sin(x)) dx
      - exp(cos(x))*(1-x*sin(x))+x dx
    """
    # First try to split the expression into two parts: exp term and another term
    # Otherwise, anther term will Interfering with the recognition of the f(x)^exp(g(x)) structure
    split_exp_term = special_add_split_exp_term(expr, context)
    if split_exp_term:
        return split_exp_term
    # To make sure only one exp term via powsimp()
    # To make Add to Mul via together()
    expr = together(powsimp(expr))
    for arg in expr.args:
        if isinstance(arg, exp):
            another_term = simplify(expr/arg)
            return handle_fx_mul_exp_gx(expr, arg, another_term, context)
    return None


def quotient_diff_form_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Handle f(x)/g(x)^2 case by recognition of quotient differential forms.

    eg. x*sin(x)/(x+sin(x))^2"""

    var = context['variable']
    step_gene = context['step_generator']
    # Note: fu() can handle standard form simplification of trigonometric functions
    # eg. (cos(x)+1)*tan(x/2)**2 to 1-cos(x)
    # So we can use simplify(fu(expr)) to simplify trigonometric functions to a simpler form
    int_result = simplify(fu(integrate(expr, var)))
    if not isinstance(int_result, Mul):
        return None
    if not is_elementary_expression(int_result):
        return None

    int_num, int_den = fraction(int_result)
    if int_den == 1:
        return None

    int_num_diff = diff(int_num, var)
    int_den_diff = diff(int_den, var)
    need_num = int_num_diff*int_den-int_num*int_den_diff
    need_expr = simplify(need_num/int_den**2)
    if simplify(need_expr/expr).has(var):
        return None

    var_latex = latex(var)
    step_gene.add_step('None', '')
    step_gene.add_step(
        'None', f'$\\int {latex(expr)}\\,\\mathrm{{d}} {var_latex}$')
    step_gene.add_step(
        'None', f'$= \\int \\frac{{{latex(int_num_diff)}}}{{{latex(int_den)}}}\\,\\mathrm{{d}}{var_latex} - \\int \\frac{{{latex(int_num*int_den_diff)}}}{{{latex(int_den**2)}}}\\,\\mathrm{{d}}{var_latex}$')
    step_gene.add_step(
        'None', f'$= \\int \\frac{{{latex(int_num_diff)}}}{{{latex(int_den)}}}\\,\\mathrm{{d}}{var_latex} + \\int {latex(int_num)}\\,\\mathrm{{d}} \\left({latex(1/int_den)}\\right)$')
    step_gene.add_step(
        'None', f'$= \\int \\frac{{{latex(int_num_diff)}}}{{{latex(int_den)}}}\\,\\mathrm{{d}}{var_latex} + {latex(int_result)} - \\int \\frac{{{latex(int_num_diff)}}}{{{latex(int_den)}}}\\,\\mathrm{{d}}{var_latex}$')
    step_gene.add_step(
        'None', f'$= {latex(int_result)} + C$')
    step_gene.add_step('None', '')

    return int_result, (
        f'变换表达式凑商的微分形式:'
        f'$\\int {latex(expr)}\\,d{var_latex} = \\int \\frac{{{latex(need_num)}}}{{{latex(int_den**2)}}}\\,d{var_latex} ='
        f"\\int \\frac{{f'({var})g({var})-g'({var})f({var})}}{{g({var})^2}}\\,d{var_latex} = \\frac{{f({var})}}{{g({var})}},\\,"
        rf'此处\, f({var}) = {latex(int_num)},\,g({var}) = {latex(int_den)}$'
    )


def weierstrass_substitution_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply Weierstrass substitution (t = tan(x/2)) for integration.

    For expressions containing only rational functions of sin(x), cos(x), etc.,
    the substitution t = tan(x/2) transforms them to rational functions in t:
    - sin(x) = 2t/(1+t^2)
    - cos(x) = (1-t^2)/(1+t^2)
    - dx = 2dt/(1+t^2)
    """
    var = context['variable']
    step_gene = context['step_generator']

    # Get substitution variable
    t = step_gene.get_available_sym(var)
    step_gene.subs_dict[t] = tan(var/2)

    dx_dt = 2 / (1 + t**2)

    # Perform the substitution on the expression
    substituted_expr = expr.subs([
        (sin(var), 2*t / (1 + t**2)),
        (cos(var), (1 - t**2) / (1 + t**2)),
        (tan(var), 2*t / (1 - t**2)),
        (cot(var), (1 - t**2) / (2*t)),
        (sec(var), (1 + t**2) / (1 - t**2)),
        (csc(var), (1 + t**2) / (2*t))
    ])

    substituted_expr *= dx_dt

    result = Integral(simplify(substituted_expr), t)
    print(result)

    var_latex = latex(var)
    explaination = (
        f"万能代换:\\,令 ${t.name} = \\tan \\left(\\frac{{{var_latex}}}{2} \\right),\\,"
        f"则\\,d{var_latex} = \\frac{{2}}{{1+{t.name}^2}}\\,d{t.name}$"
    )

    return result, explaination


def logarithmic_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Heuristic matcher for f'(x)/f(x) integration pattern.

    Returns 'logarithmic' if the expression matches the pattern where the
    numerator is the derivative of the denominator.
    """
    # Extract numerator and denominator
    numerator, denominator = expr.as_numer_denom()

    if denominator == 1:
        return None

    var = context['variable']

    # Calculate derivative of denominator
    f_prime = diff(denominator, var)
    ratio = simplify(numerator - f_prime)

    if ratio.is_constant():
        return 'logarithmic'

    # Also check if expression can be rearranged to this form
    if expr.is_Mul:
        # Look for patterns like f'(x) * (1/f(x))
        factors = expr.args
        for factor in factors:
            if factor.is_Pow and factor.args[1] == -1:
                f_x = factor.args[0]
                f_prime = diff(f_x, var)
                # Create the rest of the multiplication
                other_factors = [f for f in factors if f != factor]
                remaining = Mul(*other_factors) if other_factors else 1
                ratio = simplify(remaining / f_prime)
                if ratio.is_constant():
                    return 'logarithmic'

    return None


def parts_matcher(_expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    return 'parts'


def substitution_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Heuristic matcher for u-substitution in integration.

    Returns 'substitution' if the expression exhibits one of:
      1. Standard pattern: f(g(x)) * g'(x) (up to constant multiple)
      2. Trigonometric substitution forms: sqrt(a^2−x^2), sqrt(a^2+x^2), sqrt(x^2−a^2)
      3. Nested radicals: (ax + b)^{p/q} with q > 1
    """
    var = context['variable']

    # Normalize expression into a list of factors (even for non-Mul)
    expr = together(expr)
    factors = list(expr.args) if expr.is_Mul else [expr]

    # Strategy 1: Standard u-substitution pattern f(g(x)) * g'(x)
    for factor in factors:
        if not factor.args or factor.is_constant() or factor == var:
            continue

        flag = False
        # Look for unary functions like 3^g(x), etc.
        if factor.is_Pow and factor.args[1].has(var):
            flag = True
        # Look for unary functions like sin(g(x)), log(g(x)),1/g(x)^2 etc.
        inner = factor.args[1] if flag else factor.args[0]

        for original_term in preorder_traversal(inner):
            if original_term == var or original_term.is_constant():
                continue
            check_list = [original_term]
            # Introducing sqrt_term is to handle implicit f(x)^2 cases like x/(x**4+1), x**x*(log(x)+1)/(x**(2*x)+1)
            sqrt_term = sqrt(original_term)
            # Use a temporary variable with positive real assumptions to aid radical simplification
            _t = Symbol('t', real=True, positive=True)
            sqrt_term = simplify(sqrt_term.subs(var, _t).subs(_t, var).replace(
                Abs, lambda arg: arg))
            if not sqrt_term.has(I):
                check_list.append(sqrt_term)
            for term in check_list:
                if term == var:
                    continue

                gp = simplify(diff(term, var))
                if gp == 0:
                    continue

                # Compute the "remaining part" = expr / factor
                outer_part = expr / factor

                # Check if outer_part is a constant multiple of g'(x)
                try:
                    ratio = simplify(outer_part / gp)
                except Exception:
                    continue

                if ratio.is_constant():
                    return 'substitution'

    # Strategy 2: Trigonometric substitution patterns

    a = Wild('a', exclude=[var], properties=[
             lambda x: x.is_positive])  # Assume a > 0
    tri_sub_patterns = [(a - var**2)**sqrt_pow,
                        (a + var**2)**sqrt_pow,
                        (var**2 - a)**sqrt_pow,
                        (a - var**2)**minus_sqrt_pow,
                        (a + var**2)**minus_sqrt_pow,
                        (var**2 - a)**minus_sqrt_pow]
    for pattern in tri_sub_patterns:
        if expr.find(pattern):
            return 'substitution'

    # Strategy 3: Radical expressions (e.g., (ax+b)^{1/n})
    if has_radical(expr, var):
        return 'substitution'

    return None


def f_x_mul_exp_g_x_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    # To make Add to Mul via together()
    expr = together(expr)
    if isinstance(expr, Mul) and expr.has(exp):
        return 'f_x_mul_exp_g_x'
    return None


def quotient_diff_form_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if isinstance(expr, Mul):
        _, den = fraction(expr)
        if den == 1:
            return None
        return 'quotient_diff_form'
    return None


def weierstrass_substitution_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:

    var = context['variable']

    if can_use_weierstrass(expr, var):
        return 'weierstrass_substitution'

    return None
