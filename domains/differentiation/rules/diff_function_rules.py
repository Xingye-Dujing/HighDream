from sympy import (
    Derivative, Expr, Integer, Pow, acos, asin, atan, cos, cosh, cot, csc, exp,
    log, sec, sin, sinh, tan, tanh
)

from core import RuleRegistry
from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn
from utils.latex_formatter import wrap_latex

_create_matcher = RuleRegistry.create_common_matcher


def _create_rule(func_name):
    return RuleRegistry.create_common_rule(Derivative, func_name)


def const_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    return Integer(0), f"常数导数为 0 : $\\frac{{d}}{{d{var_latex}}} {wrap_latex(expr)} = 0$"


def var_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.equals(var):
        return Integer(1), f"变量导数为 1 : $\\frac{{d}}{{d{var_latex}}} {var_latex} = 1$"
    return Integer(0), f"无关变量视为常数, 导数为 0 : $\\frac{{d}}{{d{var_latex}}} {wrap_latex(expr)} = 0$"


def pow_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Differentiate power expressions u(x)**v(x) using appropriate rules:

      1. x^n (n constant)           power rule
      2. f(x)^n (n constant)        chain + power rule
      3. a^f(x) (a constant)        exponential rule
      4. f(x)^g(x)                  logarithmic differentiation
    """
    var = context['variable']
    var_latex, expr_latex = wrap_latex(var, expr)
    base, exponent = expr.base, expr.exp

    # Case 1: x^n, n constant
    if base.is_Symbol and base == var and exponent.is_constant():
        new_expr = exponent * (var ** (exponent - 1))
        new_expr_latex = wrap_latex(new_expr)

        return new_expr, (
            f"应用幂函数规则: "
            f"$\\frac{{d}}{{d{var_latex}}}\\left({expr_latex}\\right) = {new_expr_latex}$"
        )

    # Case 2: f(x)^n, n constant, f(x) != x
    if exponent.is_constant() and base.has(var) and base != var:
        # Apply chain rule: d/dx [f(x)^n] = n * f(x)^(n-1) * f'(x)
        new_expr = exponent * (base ** (exponent - 1)) * Derivative(base, var)
        new_expr_latex = wrap_latex(new_expr)

        return new_expr, (
            f"应用链式法则和幂函数规则: "
            f"$\\frac{{d}}{{d{var_latex}}} {expr_latex} = {new_expr_latex}$"
        )

    # Case 3: a^f(x), a constant > 0
    # Apply exponential rule: d/dx [a^f(x)] = a^f(x) * ln(a) * f'(x)
    if base.is_constant() and exponent.has(var):
        new_expr = expr * log(base) * Derivative(exponent, var)
        new_expr_latex = wrap_latex(new_expr)

        return new_expr, (
            f"应用指数函数规则: "
            f"$\\frac{{d}}{{d{var_latex}}}\\left({expr_latex}\\right) = {new_expr_latex}$"
        )

    # Case 4: f(x)^g(x), both depend on var
    # Apply logarithmic differentiation: d/dx [u^v] = u^v * (v' * ln(u) + v * u'/u)
    if base.has(var) and exponent.has(var):
        # Special case one: x^x
        if base == var and exponent == var:
            new_expr = expr * (log(base) + exponent / base)
        # Special case two: x^f(x)
        elif base == var:
            new_expr = expr * (
                Derivative(exponent, var) * log(base) +
                exponent / base
            )
        # Special case three: f(x)^x
        elif exponent == var:
            new_expr = expr * (
                log(base) +
                exponent * Derivative(base, var) / base
            )
        # General case: f(x)^g(x)
        else:
            new_expr = expr * (
                Derivative(exponent, var) * log(base) +
                exponent * Derivative(base, var) / base
            )
        new_expr_latex = wrap_latex(new_expr)
        return new_expr, (
            f"应用对数微分法: "
            f"$\\frac{{d}}{{d{var_latex}}} {expr_latex} = {new_expr_latex}$"
        )
    return None


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
asin_rule = _create_rule("反正弦")
acos_rule = _create_rule("反余弦")
atan_rule = _create_rule("反正切")
sinh_rule = _create_rule("双曲正弦")
cosh_rule = _create_rule("双曲余弦")
tanh_rule = _create_rule("双曲正切")


# Generate all matcher functions using the factory function
exp_matcher = _create_matcher(exp)
log_matcher = _create_matcher(log)
sin_matcher = _create_matcher(sin)
cos_matcher = _create_matcher(cos)
tan_matcher = _create_matcher(tan)
sec_matcher = _create_matcher(sec)
csc_matcher = _create_matcher(csc)
cot_matcher = _create_matcher(cot)
asin_matcher = _create_matcher(asin)
acos_matcher = _create_matcher(acos)
atan_matcher = _create_matcher(atan)
sinh_matcher = _create_matcher(sinh)
cosh_matcher = _create_matcher(cosh)
tanh_matcher = _create_matcher(tanh)


def const_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if expr.is_constant():
        return 'const'
    return None


def var_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    if expr.is_Symbol:
        return 'var'
    return None


def pow_matcher(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
    # Don't restrict to var == context['variable']
    if isinstance(expr, Pow):
        return 'pow'
    return None
