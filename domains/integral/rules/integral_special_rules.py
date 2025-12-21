from sympy import (
    Expr, Integral, Mul, Wild, Rational, diff, integrate,
    latex, log, simplify
)

from utils import (
    MatcherFunctionReturn, RuleContext, RuleFunctionReturn, has_radical
)
from utils.latex_formatter import wrap_latex
from domains.integral import (
    select_parts_u_dv,
    try_radical_substitution,
    try_standard_substitution,
    try_trig_substitution
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
        dv = 1
    else:
        # For multi-factor expressions, use the ILAET heuristic
        u, dv = select_parts_u_dv(expr, var)

    du = simplify(diff(u, var))
    v = simplify(integrate(dv, var))
    u_v = simplify(u * v)
    v_du = simplify(v * du)
    result = u_v - Integral(v_du, var)
    var_latex, expr_latex, v_du_latex = wrap_latex(
        var, expr, v_du)
    return result, (
        f"分部积分法(ILAET 选择 $u={latex(u)}$, $dv={latex(dv)}\\,d{var_latex}$): $"
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

    # Skip constant expressions
    if not expr.has(var):
        return None

    # Normalize expression into a list of factors (even for non-Mul)
    factors = list(expr.args) if expr.is_Mul else [expr]

    # Strategy 1: Standard u-substitution pattern f(g(x)) * g'(x)
    for factor in factors:
        # Look for unary functions like sin(g(x)), log(g(x)), etc.
        if factor.is_Function and len(factor.args) == 1:
            inner = factor.args[0]
            if not inner.has(var):
                continue
            gp = diff(inner, var)
            if gp.is_zero:
                continue

            # Compute the "remaining part" = expr / factor
            try:
                outer_part = expr / factor
            except ZeroDivisionError:
                continue

            if outer_part == 0:
                continue

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
