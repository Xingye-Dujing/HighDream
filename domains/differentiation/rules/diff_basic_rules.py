from sympy import Add, Derivative, Expr, Mul, Symbol, latex

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn
from utils.latex_formatter import wrap_latex


def add_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the sum rule: d/dx (u + v + ...) = du/dx + dv/dx + ..."""
    var: Symbol = context['variable']
    var_latex = wrap_latex(var)
    terms = expr.args

    # Build derivative terms for both explanation and new expression
    derivative_terms_latex = []
    derivative_terms_expr = []

    for term in terms:
        deriv = Derivative(term, var)
        derivative_terms_expr.append(deriv)
        derivative_terms_latex.append(
            f"\\frac{{d}}{{d{var_latex}}}{wrap_latex(term)}")

    # Construct explanation
    lhs = f"\\frac{{d}}{{d{var_latex}}}{wrap_latex(expr)}"
    rhs = " + ".join(derivative_terms_latex)
    explanation = f"应用加法规则: ${lhs} = {rhs}$"

    # New expression: sum of derivatives (not yet evaluated!)
    new_expr = Add(*derivative_terms_expr)

    return new_expr, explanation


def mul_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the product rule: (uv)' = u'v + uv' and its general form"""
    var: Symbol = context['variable']
    var_latex, expr_latex = wrap_latex(var, expr)
    terms = expr.args
    n = len(terms)

    # Two-term case
    if n == 2:
        u, v = terms
        u_latex, v_latex = wrap_latex(u, v)
        # Special case one: constant * variable
        if u.is_number:
            v_diff = Derivative(v, var)
            return u * v_diff, f"应用常数乘法规则: $\\frac{{d}}{{d{var_latex}}} {expr_latex} = {wrap_latex(u)}{latex(v_diff)}$"
        # Special case two: x * f(x)
        if u == var:
            deriv_v = Derivative(v, var)
            new_expr = v + var * deriv_v
            deriv_v_latex = wrap_latex(deriv_v)
            return new_expr, (
                f"应用乘积规则: $\\frac{{d}}{{d{var_latex}}} {{{expr_latex}}} = "
                f"1 \\cdot {v_latex} + {var_latex} \\cdot {deriv_v_latex} = "
                f"{v_latex} + {var_latex} \\cdot {deriv_v_latex}$"
            )
        # General case: (uv)' = u'v + uv'
        du = Derivative(u, var)
        dv = Derivative(v, var)
        new_expr = v * du + u * dv
        du_latex, dv_latex = wrap_latex(du, dv)
        lhs = f"\\frac{{d}}{{d{var_latex}}} {{{expr_latex}}}"
        rhs = f"{v_latex} \\cdot {du_latex} + {u_latex} \\cdot {dv_latex}"
        explanation = f"应用乘积法则: ${lhs} = {rhs}$"

    # N-term case
    # Build each term of the sum: derivative of i-th factor times others
    sum_terms_expr = []
    sum_terms_latex = []
    for i in range(n):
        # Build the i-th addend
        factors_expr = []
        factors_latex = []
        deriv_term = terms[i]
        deriv_term_latex = wrap_latex(deriv_term)
        # Derivative of the i-th term
        deriv = Derivative(deriv_term, var)
        factors_expr.append(deriv)
        factors_latex.append(
            f"\\frac{{d}}{{d{var_latex}}}{deriv_term_latex}")
        # Multiply by all other terms
        for j, t in enumerate(terms):
            if j != i:
                factors_expr.append(t)
                factors_latex.append(wrap_latex(t))
        # Construct the full addend
        if len(factors_expr) == 1:
            addend_expr = factors_expr[0]
            addend_latex = factors_latex[0]
        else:
            addend_expr = Mul(*factors_expr)
            addend_latex = " \\cdot ".join(factors_latex)

        sum_terms_expr.append(addend_expr)
        sum_terms_latex.append(f"\\left({addend_latex}\\right)")

    # Final expression: sum of all addends
    new_expr = Add(*sum_terms_expr)

    # Build explanation
    lhs = f"\\frac{{d}}{{d{var_latex}}}{expr_latex}"
    rhs = " + ".join(sum_terms_latex)
    explanation = f"应用乘积法则: ${lhs} = {rhs}$"

    return new_expr, explanation


def div_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the quotient rule: d/dx(u/v) = (u' v - u v') / v^2"""
    var: Symbol = context['variable']
    u, v = expr.as_numer_denom()

    u_prime = Derivative(u, var)
    v_prime = Derivative(v, var)
    # Special case: denominator is the variable (v = x)
    if v == var:
        new_expr = (u_prime * v - u) / (v ** 2)
    else:
        new_expr = (u_prime * v - u * v_prime) / (v ** 2)

    var_latex, u_latex, v_latex = wrap_latex(var, u, v)
    v_sq_latex = f"{v_latex}^2" if not v.is_Atom else f"{v_latex}^2"

    explanation = (
        f"应用除法规则: "
        f"$\\frac{{d}}{{d{var_latex}}} \\frac{{{u_latex}}}{{{v_latex}}} = "
        f"\\frac{{"
        f"\\frac{{d}}{{d{var_latex}}}{u_latex} \\cdot {v_latex} - "
        f"{u_latex} \\cdot \\frac{{d}}{{d{var_latex}}}{v_latex}"
        f"}}{{{v_sq_latex}}}$"
    )

    return new_expr, explanation


def chain_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the chain rule to f(g(x)): d/dx f(g(x)) = f'(g(x)) * g'(x)"""
    var = context['variable']
    outer_func = expr.func
    inner_expr = expr.args[0]

    inner_derivative = Derivative(inner_expr, var)
    u = Symbol('u')
    outer_derivative = Derivative(outer_func(
        u), u).doit().subs(u, inner_expr)
    new_expr = outer_derivative * inner_derivative

    var_latex, expr_latex, inner_latex = wrap_latex(var, expr, inner_expr)
    explanation = (
        f"应用链式法则: "
        f"$\\frac{{d}}{{d{var_latex}}} {expr_latex} = "
        f"\\frac{{d}}{{d{inner_latex}}} {expr_latex} "
        f"\\cdot \\frac{{d}}{{d{var_latex}}} {inner_latex}$"
    )
    return new_expr, explanation


def add_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if isinstance(expr, Add):
        return 'add'
    return None


def mul_div_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if not isinstance(expr, Mul):
        return None

    _, den = expr.as_numer_denom()
    # If denominator is 1, it's a pure product
    return 'mul' if den == 1 or den.is_constant else 'div'


def chain_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    var = context['variable']
   # Match composite functions f(g(x)) where g(x) is not the variable itself.
    if expr.args and len(expr.args) == 1:
        arg = expr.args[0]
        if arg.has(var) and arg != var:
            return 'chain'
    return None
