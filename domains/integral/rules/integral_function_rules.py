from sympy import (
    Abs, Expr, Integral, Pow, acos, asin, atan, cos, cosh, cot, csc, exp,
    latex, log, sec, sin, sinh, sqrt, tan, tanh, csch, sech, coth
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
csch_rule = _create_rule("双曲余割")
sech_rule = _create_rule("双曲正割")


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
    return (var ** 2) / 2, f"变量积分: $\\int {var_latex}\\,d{var_latex} = \\frac{{{var_latex}^2}}{2} + C$"


def pow_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the pow rule:

    n != -1 to x^n dx = x^(n+1)/(n+1) + C;
    n = -1 to ln|x| + C.
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
        result = var ** 2 / 2
        return result, f"基本幂函数积分: $\\int {var_latex}\\,d{var_latex} = \\frac{{{var_latex}^2}}{2} + C$"
    # General case: n != -1
    result = var ** (exponent + 1) / (exponent + 1)
    exponent_latex = wrap_latex(exponent + 1)
    return result, (f"幂函数积分规则: $\\int {expr_latex}\\,d{var_latex} = "
                    f"\\frac{{{var_latex}^{{{exponent_latex}}}}}{{{exponent_latex}}} + C$")


def exp_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the extended exponential rule for a^x where a can be any base.

    For a^x dx = a^x / ln(a) + C
    For e^x dx = e^x + C (special case)
    """

    var = context['variable']

    if expr.equals(exp(var)):
        var_latex = wrap_latex(var)
        return expr, f"自然指数函数积分: $\\int e^{{{var_latex}}}\\,d{var_latex} = e^{{{var_latex}}} + C$"

    base, exponent = expr.as_base_exp()

    # Check if it is e^x (natural exponential)
    # General case: a^x where a > 0 and a != 1.
    if exponent == var and base.is_positive and base.is_real and base != 1:
        var_latex = wrap_latex(var)
        result = expr / log(base)
        base_latex = wrap_latex(base)
        return result, (f"指数函数积分: $\\int {base_latex}^{{{var_latex}}}\\,d{var_latex} = "
                        f"\\frac{{{base_latex}^{{{var_latex}}}}}{{\\ln({base_latex})}} + C$")

    return None


def inverse_trig_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    var = context['variable']

    if expr.equals(1 / sqrt(1 - var ** 2)):
        var_latex = wrap_latex(var)
        return asin(
            var), f"反正弦函数积分: $\\int \\frac{{1}}{{\\sqrt{{1 - {var}^2}}}}\\,d{var_latex} = \\arcsin({var}) + C$"

    if expr.equals(1 / sqrt(var ** 2 - 1)):
        var_latex = wrap_latex(var)
        return acos(
            var), f"反余弦函数积分: $\\int \\frac{{-1}}{{\\sqrt{{1 - {var}^2}}}}\\,d{var_latex} = \\arccos({var}) + C$"

    return None


def inverse_tangent_linear_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Handles integrals of the form 1/(x^2 + m)^n."""

    var = context['variable']
    var_latex = wrap_latex(var)
    # SymPy puts the constant term at the front
    m = expr.base.args[0]
    m_latex = wrap_latex(m)
    if m > 0:
        sqrt_m = sqrt(m)
        sqrt_m_latex = "" if sqrt_m.equals(1) else wrap_latex(sqrt_m)

        if expr.exp.equals(-1):
            if m.equals(1):
                return atan(
                    var), f"反正切函数积分: $\\int \\frac{{1}}{{1 + {var}^2}}\\,d{var_latex} = \\arctan({var}) + C$"

            result = (1 / sqrt_m) * atan(var / sqrt_m)
            return result, (f"反正切函数积分: $\\int \\frac{{1}}{{{var_latex}^2 + {m_latex}}}\\,d{var_latex} = "
                            f"\\frac{{1}}{{{sqrt_m_latex}}} "
                            f"\\arctan\\left(\\frac{{{var_latex}}}{{{sqrt_m_latex}}}\\right) + C$")

        if expr.exp.equals(-2):
            step_gene = context['step_generator']
            u = step_gene.get_available_sym(var)
            step_gene.subs_dict[u] = atan(var / sqrt_m)
            return Integral(cos(u) ** 2 / (m * sqrt_m),
                            u), (f"令 ${u.name} = {latex(atan(var / sqrt_m))}$, "
                                 f"则 ${latex(var)} = {sqrt_m_latex}\\, \\tan\\left({u.name}\\right) $")

        n = -expr.exp
        res = (var / (2 * m * (n - 1) * expr.base ** (n - 1)) + ((2 * n - 3) / (2 * m * (n - 1)))
               * Integral(1 / expr.base ** (n - 1), var))
        return res, (f"使用递推公式: $\\int \\frac{{1}}{{(a x^2 + b^2)^n}} \\, dx = "
                     f"\\frac{{x}}{{2b^2(n-1)(a x^2 + b^2)^{{n-1}}}} + \\frac{{2n - 3}}{{2b^2(n-1)}} "
                     f"\\int \\frac{{1}}{{(a x^2 + b^2)^{{n-1}}}} \\, dx$")

    sqrt_m = sqrt(-m)
    if expr.exp.equals(-1):
        res = 1 / (2 * sqrt_m) * (log(abs(var - sqrt_m)) - log(abs(var + sqrt_m)))
        return res, f"反切函数积分: $\\int \\frac{{1}}{{{latex(expr.base)}}}\\,d{var_latex} = {latex(res)} + C$"
    n = -expr.exp
    res = 1 / (2 * (n - 1) * sqrt_m ** 2) * ((3 - 2 * n) * Integral(1 / expr.base
                                                                    ** (n - 1), var) - var / (expr.base ** (n - 1)))
    return res, (f"使用递推公式: $\\int \\frac{{1}}{{(x^2 - b^2)^n}} \\, dx = "
                 f"\\frac{{1}}{{2(n-1)b^2}} \\left[ (3 - 2n) \\int \\frac{{1}}{{(x^2 - b^2)^{{n-1}}}} \\, dx - "
                 f"\\frac{{x}}{{(x^2 - b^2)^{{n-1}}}} \\right]$")


def tanh_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the tanh rule: tanh(x) dx = ln(|cosh(x)|) + C"""
    var = context['variable']
    result = log(Abs(cosh(var)))
    var_latex = latex(var)
    return result, f"双曲正切函数积分: $\\int \\tanh({var_latex})\\,d{var_latex} = \\ln| \\cosh({var_latex})| + C$"


def coth_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the coth rule: coth(x) dx = ln(|sinh(x)|) + C."""
    var = context['variable']
    result = log(Abs(sinh(var)))
    var_latex = latex(var)
    return result, f"双曲余切函数积分: $\\int \\coth({var_latex})\\,d{var_latex} = \\ln| \\sinh({var_latex})| + C$"


def sin_power_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the rule for sin(x)^n."""
    var = context['variable']
    var_latex = wrap_latex(var)

    _, n = expr.as_base_exp()
    n_latex = wrap_latex(n)

    if n == 2:
        result = var / 2 - sin(2 * var) / 4
        return result, (f"正弦平方积分: $\\int \\sin^2({var_latex})\\,d{var_latex} = "
                        f"\\frac{{{var_latex}}}{2} - \\frac{{\\sin(2{var_latex})}}{4} + C$")

    # For higher powers, we can use reduction formula
    # sin^n(x) dx = -sin^(n-1)(x)cos(x)/n + (n-1)/n sin^(n-2)(x) dx
    result = (-(sin(var) ** (n - 1) * cos(var)) / n + (n - 1)
              / n * Integral(sin(var) ** (n - 2), var))
    return result, (f"正弦幂函数积分: $\\int \\sin^{{{n_latex}}}({var_latex})\\,d{var_latex} = "
                    f"-\\frac{{\\sin^{{{n_latex}-1}}({var_latex})\\cos({var_latex})}}{{{n_latex}}} + "
                    f"\\frac{{{n_latex}-1}}{{{n_latex}}} \\int \\sin^{{{n_latex}-2}}({var_latex})\\,d{var_latex}$")


def cos_power_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the rule for cos(x)^n."""
    var = context['variable']
    var_latex = wrap_latex(var)

    _, n = expr.as_base_exp()
    n_latex = wrap_latex(n)

    if n == 2:
        result = var / 2 + sin(2 * var) / 4
        return result, (f"余弦平方积分: $\\int \\cos^2({var_latex})\\,d{var_latex} = \\frac{{{var_latex}}}{2} + "
                        f"\\frac{{\\sin(2{var_latex})}}{4} + C$")

    # For higher powers, we can use reduction formula
    # cos^n(x) dx = cos^(n-1)(x)sin(x)/n + (n-1)/n cos^(n-2)(x) dx
    result = ((cos(var) ** (n - 1) * sin(var)) / n + (n - 1)
              / n * Integral(cos(var) ** (n - 2), var))
    return result, (f"余弦幂函数积分: $\\int \\cos^{{{n_latex}}}({var_latex})\\,d{var_latex} = "
                    f"\\frac{{\\cos^{{{n_latex}-1}}({var_latex})\\sin({var_latex})}}{{{n_latex}}} + "
                    f"\\frac{{{n_latex}-1}}{{{n_latex}}} \\int \\cos^{{{n_latex}-2}}({var_latex})\\,d{var_latex}$")


def tan_power_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the rule for tan(x)^n."""
    var = context['variable']
    var_latex = wrap_latex(var)

    _, n = expr.as_base_exp()
    n_latex = wrap_latex(n)

    if n == 2:
        result = tan(var) - var
        return result, f"正切平方积分: $\\int \\tan^2({var_latex})\\,d{var_latex} = \\tan({var_latex}) - {var_latex} + C$"

    # For higher powers, we can use reduction formula
    # tan^n(x) dx = tan^(n-1)(x)/(n-1) - tan^(n-2)(x) dx
    result = tan(var) ** (n - 1) / (n - 1) - Integral(tan(var) ** (n - 2), var)
    return result, (f"正切幂函数积分: $\\int \\tan^{{{n_latex}}}({var_latex})\\,d{var_latex} = "
                    f"\\frac{{\\tan^{{{n_latex}-1}}({var_latex})}}{{{n_latex}-1}} - "
                    f"\\int \\tan^{{{n_latex}-2}}({var_latex})\\,d{var_latex}$")


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


def pow_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    var = context['variable']
    if isinstance(expr, Pow) and expr.base == var and expr.exp.is_constant():
        return 'pow'
    return None


def inverse_trig_matcher(_expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    return 'inverse_trig'


def exp_matcher(_expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    return 'exp'


def sec_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for secant expressions including 1/cos(x).

    Matches expressions that are equivalent to sec(x).
    """
    var = context['variable']

    # Match direct sec(x) or 1/cos(x)
    if expr.equals(sec(var)) or expr.equals(1 / cos(var)):
        return 'sec'
    return None


def csc_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for cosecant expressions including 1/sin(x).

    Matches expressions that are equivalent to csc(x).
    """
    var = context['variable']

    # Match direct csc(x) or 1/sin(x)
    if expr.equals(csc(var)) or expr.equals(1 / sin(var)):
        return 'csc'
    return None


def cot_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for cotangent expressions including 1/tan(x).

    Matches expressions that are equivalent to cot(x).
    """
    var = context['variable']

    # Match direct cot(x) or 1/tan(x)
    if expr.equals(cot(var)) or expr.equals(1 / tan(var)):
        return 'cot'

    return None


def csch_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for hyperbolic cosecant expressions including 1/sinh(x).

    Matches expressions that are equivalent to csch(x).
    """
    var = context['variable']

    # Match direct csch(x) or 1/sinh(x)
    if expr.equals(csch(var)) or expr.equals(1 / sinh(var)):
        return 'csch'
    return None


def sech_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for hyperbolic secant expressions including 1/cosh(x).

    Matches expressions that are equivalent to sech(x).
    """
    var = context['variable']

    # Match direct sech(x) or 1/cosh(x)
    if expr.equals(sech(var)) or expr.equals(1 / cosh(var)):
        return 'sech'
    return None


def coth_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for hyperbolic cotangent expressions including 1/tanh(x).

    Matches expressions that are equivalent to coth(x).
    """
    var = context['variable']

    # Match direct coth(x) or 1/tanh(x)
    if expr.equals(coth(var)) or expr.equals(1 / tanh(var)):
        return 'coth'

    return None


def inverse_tangent_linear_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for expressions of the form 1/(x^2 + m)^n"""
    var = context['variable']

    if expr.is_Pow and expr.exp < 0 and expr.exp.is_Integer:
        terms = expr.base.args
        # SymPy puts the constant term at the front
        if expr.base.is_Add and len(terms) == 2 and terms[1].equals(var ** 2) and terms[0].is_constant():
            return 'inverse_tangent_linear'

    return None


def sin_power_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for sin(x)^n expressions."""
    var = context['variable']

    if expr.is_Pow:
        base, _exp = expr.as_base_exp()
        if base.func == sin and len(base.args) == 1 and base.args[0] == var and _exp.is_Integer and _exp > 0:
            return 'sin_power'

    return None


def cos_power_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for cos(x)^n expressions."""
    var = context['variable']

    if expr.is_Pow:
        base, _exp = expr.as_base_exp()
        if base.func == cos and len(base.args) == 1 and base.args[0] == var and _exp.is_Integer and _exp > 0:
            return 'cos_power'

    return None


def tan_power_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for tan(x)^n expressions."""
    var = context['variable']

    if expr.is_Pow:
        base, _exp = expr.as_base_exp()
        if base.func == tan and len(base.args) == 1 and base.args[0] == var and _exp.is_Integer and _exp > 0:
            return 'tan_power'

    return None
