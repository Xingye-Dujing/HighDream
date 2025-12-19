from sympy import Add, Expr, Integral, Mul, div, fraction, latex

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn
from utils.latex_formatter import wrap_latex


def smart_expand(expr: Expr) -> Expr:
    """Expand an expression to Add form by performing polynomial division."""
    num, den = fraction(expr)
    q, r = div(num, den)
    result = q + r/den
    if isinstance(result, Add):
        return result
    result = result.expand()
    if isinstance(result, Add):
        return result
    return None


def add_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the sum rule：(f+g) dx = f dx + g dx"""
    expr_copy = expr
    if isinstance(expr, Mul):
        expr_copy = smart_expand(expr)
    if not expr_copy:
        return None

    var = context['variable']
    var_latex, expr_latex = wrap_latex(var, expr)

    # Comparing to diff's add_rule, we use list comprehension.
    integrals = [Integral(term, var) for term in expr_copy.args]
    new_expr = Add(*integrals)

    rhs_latex = " + ".join(
        f"\\int {wrap_latex(term)}\\,d{var_latex}" for term in expr_copy.args)
    explanation = f"应用加法规则: $\\int{expr_latex}\\,d{var_latex} = {rhs_latex}$"
    return new_expr, explanation


def mul_const_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the const mul rule: c*f(x) dx = c * f(x) dx"""
    var = context['variable']

    coeff, func_part = expr.as_coeff_Mul()

    var_latex, expr_latex, func_part_latex = wrap_latex(var, expr, func_part)

    if coeff == -1:
        return -Integral(func_part, var), f"负号提出: $\\int {expr_latex}\\,d{var} = -\\int {func_part_latex}\\,d{var_latex}$"

    return coeff * Integral(func_part, var), f"常数因子提取: $\\int {expr_latex}\\,d{var} = {latex(coeff)} \\int {func_part_latex}\\,d{var_latex}$"


def add_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if isinstance(expr, Add):
        return 'add'
    if isinstance(expr, Mul) and fraction(expr)[1] != 1:
        # If is a fraction
        return 'add'
    return None


def mul_const_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if not expr.is_constant() and isinstance(expr, Mul):
        # Use as_coeff_Mul to check if there's a constant coefficient
        coeff, _ = expr.as_coeff_Mul()
        if coeff != 1:
            return 'mul_const'
    return None
