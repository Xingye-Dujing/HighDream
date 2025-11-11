from sympy import (
    Abs, Expr, Integral, Pow, acos, asin, atan, cos, cosh, cot, csc, exp,
    log, sec, sin, sinh, sqrt, tan, tanh
)

from core import RuleRegistry
from utils import Context, MatcherFunctionReturn, RuleFunctionReturn
from utils.latex_formatter import wrap_latex

_create_matcher = RuleRegistry.create_common_matcher


def _create_rule(func_name):
    return RuleRegistry.create_common_rule(Integral, func_name)


# Generate all exp and log rules using the factory function
exp_rule = _create_rule("指数")
log_rule = _create_rule("对数")
# Generate all trigonometric and hyperbolic rules using the factory function
sin_rule = _create_rule("正弦")
cos_rule = _create_rule("余弦")
tan_rule = _create_rule("正切")
sec_rule = _create_rule("正割")
csc_rule = _create_rule("余割")
cot_rule = _create_rule("余切")
sinh_rule = _create_rule("双曲正弦")
cosh_rule = _create_rule("双曲余弦")
tanh_rule = _create_rule("双曲正切")


def const_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply the const rule: c dx = c*x + C"""
    var = context['variable']
    var_latex, expr_latex = wrap_latex(var, expr)
    return expr * var, f"常数积分: $\\int {expr_latex}\\,d{var} = {var_latex} + C$"


def var_rule(_expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply the var rule: x dx = (1/2)x^2 + C"""
    var = context['variable']
    var_latex = wrap_latex(var)
    return (var**2) / 2, f"变量积分: $\\int {var_latex}\\,d{var_latex} = \\frac{{{var_latex}^2}}{2} + C$"


def pow_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply the pow rule:

    n != -1 to x^n dx = x^(n+1)/(n+1) + C;
    n = -1 to ln|x| + C
    """
    var = context['variable']
    var_latex, expr_latex = wrap_latex(var, expr)
    _, exponent = expr.as_base_exp()
    # Special case one: n = -1
    if exponent == -1:
        result = log(Abs(var))
        return result, f"特殊幂函数积分: $\\int {expr_latex}\\,d{var_latex} = \\ln|{var_latex}| + C$"
    # Special case two: n = 1
    if exponent == 1:
        result = var**2 / 2
        return result, f"基本幂函数积分: $\\int {var_latex}\\,d{var_latex} = \\frac{{{var_latex}^2}}{2} + C$"
    # General case: n != -1
    result = var**(exponent + 1) / (exponent + 1)
    exponent_latex = wrap_latex(exponent+1)
    return result, f"幂函数积分规则: $\\int {expr_latex}\\,d{var_latex} = \\frac{{{var_latex}^{{{exponent_latex}}}}}{{{exponent_latex}}} + C$"


def inverse_trig_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    mapping = {
        1/sqrt(1 - var**2): (asin(var), f"反正弦函数积分: $\\int \\frac{{1}}{{\\sqrt{{1 - {var}^2}}}}\\,d{var_latex} = \\arcsin({var}) + C$"),
        -1/sqrt(1 - var**2): (acos(var), f"反余弦函数积分: $\\int \\frac{{-1}}{{\\sqrt{{1 - {var}^2}}}}\\,d{var_latex} = \\arccos({var}) + C$"),
        1/(1 + var**2): (atan(var), f"反正切函数积分: $\\int \\frac{{1}}{{1 + {var}^2}}\\,d{var_latex} = \\arctan({var}) + C$")
    }
    for key, (res, desc) in mapping.items():
        if expr.equals(key):
            return res, desc

    return None


pow_matcher = _create_matcher(Pow)
exp_matcher = _create_matcher(exp)
log_matcher = _create_matcher(log)
sin_matcher = _create_matcher(sin)
cos_matcher = _create_matcher(cos)
tan_matcher = _create_matcher(tan)
sec_matcher = _create_matcher(sec)
csc_matcher = _create_matcher(csc)
cot_matcher = _create_matcher(cot)
sinh_matcher = _create_matcher(sinh)
cosh_matcher = _create_matcher(cosh)
tanh_matcher = _create_matcher(tanh)


def const_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if expr.is_constant():
        return 'const'
    return None


def var_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    if expr == var:
        return 'var'
    return None


def inverse_trig_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    patterns = [1/sqrt(1 - var**2), -1/sqrt(1 - var**2), 1/(1 + var**2)]
    if any(expr == p for p in patterns):
        return 'inverse_trig'
    return None
