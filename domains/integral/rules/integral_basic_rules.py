from sympy import Add, Expr, Integral, Mul, latex

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn
from utils.latex_formatter import wrap_latex


def add_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the sum rule：(f+g) dx = f dx + g dx"""
    var = context['variable']
    var_latex, expr_latex = wrap_latex(var, expr)

    # Compring to diff's add_rule, we use list comprehension.
    integrals = [expr.func(Integral(term, var), evaluate=False)
                 for term in expr.args]
    new_expr = Add(*integrals)

    rhs_latex = " + ".join(
        f"\\int {wrap_latex(term)}\\,d{var_latex}" for term in expr.args)
    explanation = f"应用加法规则: $\\int{expr_latex}\\,d{var_latex} = {rhs_latex}$"
    return new_expr, explanation


def mul_const_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the const mul rule: c*f(x) dx = c * f(x) dx"""
    var = context['variable']
    factors = expr.args

    # Split the factors into two parts: const and func.
    const_factors, func_factors = [], []
    for f in factors:
        if f.has(var):
            func_factors.append(f)
        else:
            const_factors.append(f)
    const_part = Mul(*const_factors)
    func_part = Mul(*func_factors)
    var_latex, expr_latex, func_part_latex = wrap_latex(var, expr, func_part)

    if const_part == -1:
        return -Integral(func_part, var), f"负号提出: $\\int {expr_latex}\\,d{var} = -\\int {func_part_latex}\\,d{var_latex}$"
    inner_integral = Integral(func_part, var)
    return const_part * inner_integral, f"常数因子提取: $\\int {expr_latex}\\,d{var} = {latex(const_part)} \\int {func_part_latex}\\,d{var_latex}$"


def add_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if isinstance(expr, Add):
        return 'add'
    return None


def mul_const_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    if not expr.is_constant() and isinstance(expr, Mul) and any(not f.has(context['variable']) for f in expr.args):
        return 'mul_const'
    return None
