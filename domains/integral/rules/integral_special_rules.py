from sympy import (
    Dummy, Expr, Mul, Wild, diff, exp, integrate,
    latex, log, simplify, sqrt
)

from utils import (
    MatcherFunctionReturn, RuleContext, RuleFunctionReturn,
    has_radical, is_exp, is_inv_trig, is_log, is_poly, is_trig
)
from utils.latex_formatter import wrap_latex
from domains.integral import (
    select_parts_u_dv, try_exp_log_substitution,
    try_radical_substitution,
    try_standard_substitution,
    try_trig_substitution
)


def parts_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply integration by parts: u dv = u v − v du.

    Uses the LIATE heuristic to select u from a product of two factors:
      L — Logarithmic      (e.g., log(x))
      I — Inverse trig     (e.g., asin(x))
      A — Algebraic        (e.g., x, x**2)
      T — Trigonometric    (e.g., sin(x))
      E — Exponential      (e.g., exp(x))

    Only triggers for binary products (exactly two factors).
    """
    var = context['variable']
    u, dv = select_parts_u_dv(expr, var)
    du = diff(u, var)
    v = simplify(integrate(dv, var))
    result = simplify(u * v - integrate(v * du, var))
    var_latex, expr_latex, u_latex, v_latex, du_latex = wrap_latex(
        var, expr, u, v, du)
    return result, (
        f"分部积分法(LIATE 选择 $u={latex(u)}$, $dv={latex(dv)}\\,d{var_latex}$): $"
        f"\\int {expr_latex}\\,d{var_latex} = "
        f"{u_latex}\\cdot{v_latex} - \\int {v_latex}\\cdot{du_latex}\\,d{var_latex} + C$"
    )


def substitution_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply substitution (u-substitution) for integration.

    Tries the following strategies in order:
      1. Standard form: f(g(x)) * g'(x) to f(u) du
      2. Trigonometric substitution (e.g., sqrt(a^2−x^2), sqrt(a^2+x^2), sqrt(x^2−a^2))
      3. Radical substitution (e.g., sqrt[n]{ax + b})
      4. Exponential/logarithmic substitution (e.g., e^{ax}, log(ax))

    Returns a transformed integral or None if no substitution applies.
    """
    var = context['variable']
    u = Dummy('u')  # Use dummy variable to avoid symbol collision

    # Strategy 1: Standard chain-rule pattern f(g(x)) * g'(x)
    result = try_standard_substitution(expr, var, u)
    if result is not None:
        return result

    # Strategy 2: Trigonometric substitutions for sqrt(a^2 +- x^2) etc.
    result = try_trig_substitution(expr, var)
    if result is not None:
        return result

    # Strategy 3: Substitutions for nested radicals (e.g., (ax + b)^{1/n})
    result = try_radical_substitution(expr, var, u)
    if result is not None:
        return result

    # Strategy 4: Substitutions involving exp or log terms
    result = try_exp_log_substitution(expr, var, u)
    if result is not None:
        return result

    return None


def parts_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Heuristic matcher for integration by parts.

    Returns 'parts' if the expression is a product of two factors
    that likely benefit from LIATE-based integration by parts, e.g.:
      - log(x) * polynomial(x)
      - polynomial(x) * trig(x)
      - polynomial(x) * exp(x)

    This avoids triggering parts on cases like sin(x)*cos(x) or exp(x)*log(x).
    """
    var = context['variable']

    # Only consider binary products
    if not isinstance(expr, Mul) or len(expr.args) != 2:
        return None

    arg1, arg2 = expr.args

    # Classify factor type for matching using LIATE ordering:
    # L (Log), I (Inverse trig), A (Algebraic/polynomial), T (Trig), E (Exp)
    types = [
        [is_log(arg1), is_inv_trig(arg1), is_poly(
            arg1, var), is_trig(arg1), is_exp(arg1, var)],
        [is_log(arg2), is_inv_trig(arg2), is_poly(
            arg2, var), is_trig(arg2), is_exp(arg2, var)]
    ]

    # Unpack for readability following LIATE order
    (l1, i1, p1, t1, e1), (l2, i2, p2, t2, e2) = types

    # LIATE-motivated patterns:
    # 1. Any log * anything (L is highest priority -> good for u)
    if l1 or l2:
        return 'parts'

    # 2. Inverse trig * anything except log (I is second priority)
    if (i1 and not l2) or (i2 and not l1):
        return 'parts'

    # 3. Polynomial * (trig or exp) -> poly as u, trig/exp as dv
    if (p1 and (t2 or e2)) or (p2 and (t1 or e1)):
        return 'parts'

    return None


def substitution_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Heuristic matcher for u-substitution in integration.

    Returns 'substitution' if the expression exhibits one of:
      1. Standard pattern: f(g(x)) * g'(x) (up to constant multiple)
      2. Trigonometric substitution forms: sqrt(a^2−x^2), sqrt(a^2+x^2), sqrt(x^2−a^2)
      3. Nested radicals: (ax + b)^{p/q} with q > 1
      4. Pure exp(x) or log(x) terms that suggest substitution
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
    if expr.find(sqrt(a**2 - var**2)):
        return 'substitution'

    # Strategy 3: Radical expressions (e.g., (ax+b)^{1/n})
    if has_radical(expr, var):
        return 'substitution'

    # Strategy 4: Exponential or logarithmic terms in x
    if expr.has(exp(var)) or expr.has(log(var)):
        return 'substitution'

    return None
