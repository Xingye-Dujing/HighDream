from sympy import (
    Abs, Expr, Integral, Pow, acos, asin, atan, cos, cosh, cot, csc, exp,
    latex, log, sec, sin, sinh, sqrt, tan, tanh
)

from core import RuleRegistry
from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn
from utils.latex_formatter import wrap_latex

_create_matcher = RuleRegistry.create_common_matcher


def _create_rule(func_name):
    return RuleRegistry.create_common_rule(Integral, func_name)


# Generate log rules using the factory function
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


def const_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the const rule: c dx = c*x + C"""
    var = context['variable']
    res = expr * var
    expr_latex = wrap_latex(expr)
    return res, f"常数积分: $\\int {expr_latex}\\,d{var} = {latex(res)} + C$"


def var_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the var rule: x dx = (1/2)x^2 + C"""
    var = context['variable']
    var_latex = wrap_latex(var)
    return (var**2) / 2, f"变量积分: $\\int {var_latex}\\,d{var_latex} = \\frac{{{var_latex}^2}}{2} + C$"


def pow_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
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


def exp_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the extended exponential rule for a^x where a can be any base.

    For a^x dx = a^x / ln(a) + C
    For e^x dx = e^x + C (special case)
    """
    var = context['variable']
    base, exponent = expr.as_base_exp()
    var_latex = wrap_latex(var)

    # Check if it's e^x (natural exponential)
    if base == exp(1) and exponent == var:
        result = expr
        return result, f"自然指数函数积分: $\\int e^{{{var_latex}}}\\,d{var_latex} = e^{{{var_latex}}} + C$"
    # General case: a^x where a > 0 and a != 1
    if exponent == var and base.is_positive and base.is_real and base != 1:
        result = expr / log(base)
        base_latex = wrap_latex(base)
        return result, f"指数函数积分: $\\int {base_latex}^{{{var_latex}}}\\,d{var_latex} = \\frac{{{base_latex}^{{{var_latex}}}}}{{\\ln({base_latex})}} + C$"

    return None


def inverse_trig_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
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


def inverse_tangent_linear_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the rule for 1/(x^2 + m) form: 1/(x^2 + m) dx = (1/sqrt(m))*arctan(x/sqrt(m)) + C

    Handles integrals of the form 1/(x^2 + m) where m > 0.
    """
    var = context['variable']
    var_latex = wrap_latex(var)

    # SymPy puts the constant term at the front
    m = expr.base.args[0]
    sqrt_m = sqrt(m)
    result = (1/sqrt_m) * atan(var/sqrt_m)
    m_latex = wrap_latex(m)
    sqrt_m_latex = wrap_latex(sqrt_m)

    return result, f"线性逆切函数积分: $\\int \\frac{{1}}{{{var_latex}^2 + {m_latex}}}\\,d{var_latex} = \\frac{{1}}{{{sqrt_m_latex}}} \\arctan\\left(\\frac{{{var_latex}}}{{{sqrt_m_latex}}}\\right) + C$"


pow_matcher = _create_matcher(Pow)
log_matcher = _create_matcher(log)
sin_matcher = _create_matcher(sin)
cos_matcher = _create_matcher(cos)
tan_matcher = _create_matcher(tan)
sinh_matcher = _create_matcher(sinh)
cosh_matcher = _create_matcher(cosh)
tanh_matcher = _create_matcher(tanh)


def const_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if expr.is_constant():
        return 'const'
    return None


def var_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    var = context['variable']
    if expr == var:
        return 'var'
    return None


def inverse_trig_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    var = context['variable']
    patterns = [1/sqrt(1 - var**2), -1/sqrt(1 - var**2), 1/(1 + var**2)]
    if any(expr == p for p in patterns):
        return 'inverse_trig'
    return None


def exp_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Extended matcher for exponential expressions including a^x where a can be any base."""
    var = context['variable']
    # Match direct exp(x) form
    if expr == exp(var):
        return 'exp'

    # Check if expression is of the form a^x
    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        # Match e^x or a^x where a > 0, a != 1, and exponent is the integration variable
        if (base == exp(1) and exponent == var) or \
           (exponent == var and base.is_positive and base.is_real and base != 1):
            return 'exp'

    return None


def sec_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for secant expressions including 1/cos(x).

    Matches expressions that are equivalent to sec(x).
    """
    var = context['variable']

    # Match direct sec(x) or 1/cos(x)
    if expr == sec(var) or expr == 1/cos(var):
        return 'sec'
    return None


def csc_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for cosecant expressions including 1/sin(x).

    Matches expressions that are equivalent to csc(x).
    """
    var = context['variable']

    # Match direct csc(x) or 1/sin(x)
    if expr == csc(var) or expr == 1/sin(var):
        return 'csc'
    return None


def cot_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for cotangent expressions including 1/tan(x).

    Matches expressions that are equivalent to cot(x).
    """
    var = context['variable']

    # Match direct cot(x) or 1/tan(x)
    if expr == cot(var) or expr == 1/tan(var):
        return 'cot'

    return None


def inverse_tangent_linear_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for expressions of the form 1/(x^2 + m) where m is a positive constant."""
    var = context['variable']

    # Match pattern 1/(var^2 + m) where m > 0
    if expr.is_Pow and expr.exp == -1:
        denominator = expr.base
        terms = denominator.args
        # SymPy puts the constant term at the front
        if denominator.is_Add and len(terms) == 2 and terms[1] == var**2 and terms[0] > 0:
            return 'inverse_tangent_linear'

    return None
