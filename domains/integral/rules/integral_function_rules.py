from sympy import (
    Abs, Expr, Pow, acos, asin, atan, cos, cosh, cot, csc, exp,
    log, sec, sin, sinh, sqrt, tan, tanh
)

from utils import Context, MatcherFunctionReturn, RuleFunctionReturn
from utils.latex_formatter import wrap_latex


def const_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply the const rule: c dx = c*x + C"""
    var = context['variable']
    var_latex, expr_latex = wrap_latex(var, expr)
    return expr * var, f"常数积分: $\\int {expr_latex}\\,d{var} = {expr_latex} \\cdot {var_latex} + C$"


def var_rule(_expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply the var rule: x dx = (1/2)x^2 + C"""
    var = context['variable']
    var_latex = wrap_latex(var)
    return (var**2) / 2, f"变量积分: $\\int {var_latex}\\,d{var_latex} = \\frac{{{var_latex}^2}}{2} + C$"


def power_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply the pow rule:

    n != -1 -> x^n dx = x^(n+1)/(n+1) + C;
    n = -1 -> ln|x| + C
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
    return result, f"幂函数积分规则: $\\int {expr_latex}}}\\,d{var_latex} = \\frac{{{var_latex}^{{{exponent_latex}}}}}{{{exponent_latex}}} + C$"


def exp_rule(_expr: Expr, context: Context) -> RuleFunctionReturn:
    """e^x dx = e^x + C"""
    var = context['variable']
    var_latex = wrap_latex(var)
    return exp(var), f"$\\int e^{{{var}}}\\,d{var_latex} = e^{{{var}}} + C$"


def log_rule(_expr: Expr, context: Context) -> RuleFunctionReturn:
    """ln(x) dx = x * ln(x) - x + C"""
    var = context['variable']
    var_latex = wrap_latex(var)
    result = var * log(var) - var
    return result, f"自然对数积分: $\\int \\ln{var_latex}\\,d{var_latex} = {wrap_latex(result)} + C$"


def trig_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']

    rules = {
        sin: (-cos(var), f"正弦函数积分: $\\int \\sin({var})\\,d{var} = -\\cos({var}) + C$"),
        cos: (sin(var), f"余弦函数积分: $\\int \\cos({var})\\,d{var} = \\sin({var}) + C$"),
        tan: (-log(cos(var)), f"正切函数积分: $\\int \\tan({var})\\,d{var} = -\\ln|\\cos({var})| + C$"),
        sec: (log(sec(var) + tan(var)), f"正割函数积分: $\\int \\sec({var})\\,d{var} = \\ln|\\sec({var}) + \\tan({var})| + C$"),
        csc: (-log(csc(var) + cot(var)), f"余割函数积分: $\\int \\csc({var})\\,d{var} = -\\ln|\\csc({var}) + \\cot({var})| + C$"),
        cot: (log(sin(var)),
              f"余切函数积分: $\\int \\cot({var})\\,d{var} = \\ln|\\sin({var})| + C$")
    }
    return rules[expr.func]


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


def hyperbolic_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    rules = {
        sinh: (cosh(var), f"双曲正弦积分: $\\int \\sinh({var})\\,d{var_latex} = \\cosh({var}) + C$"),
        cosh: (sinh(var), f"双曲余弦积分: $\\int \\cosh({var})\\,d{var_latex} = \\sinh({var}) + C$"),
        tanh: (var - log(cosh(var)),
               f"双曲正切积分: $\\int \\tanh({var})\\,d{var_latex} = {var} - \\ln|\\cosh({var})| + C$")
    }
    return rules[expr.func]


def const_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if expr.is_constant():
        return 'const'
    return None


def var_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    if expr == var:
        return 'var'
    return None


def power_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    if isinstance(expr, Pow) and expr.base == var:
        return 'power'
    return None


def exp_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    if isinstance(expr, exp) and expr.args[0] == var:
        return 'exp'
    return None


def log_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    if isinstance(expr, log) and expr.args[0] == var:
        return 'log'
    return None


def trig_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    if expr.func in [sin, cos, tan, sec, csc, cot] and expr.args[0] == var:
        return 'trig'
    return None


def inverse_trig_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    patterns = [1/sqrt(1 - var**2), -1/sqrt(1 - var**2), 1/(1 + var**2)]
    if any(expr == p for p in patterns):
        return 'inverse_trig'
    return None


def hyperbolic_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    if expr.func in [sinh, cosh, tanh] and expr.args[0] == var:
        return 'hyperbolic'
    return None
