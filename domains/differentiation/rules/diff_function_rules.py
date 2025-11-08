from sympy import (
    Derivative, Expr, Integer, Pow, acos, asin, atan, cos, cosh, cot, csc, exp,
    log, sec, sin, sinh, tan, tanh
)

from utils import Context, MatcherFunctionReturn, RuleFunctionReturn
from utils.latex_formatter import wrap_latex


def const_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    return Integer(0), f"常数导数为 0 : $\\frac{{d}}{{d{wrap_latex(var)}}} {wrap_latex(expr)} = 0$"


def var_rule(_expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    return Integer(1), f"变量导数为 1 : $\\frac{{d}}{{d{var_latex}}} {var_latex} = 1$"


def pow_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
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


def exp_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return exp(var), f"应用指数函数规则: $\\frac{{d}}{{d{var_latex}}}(e^{{{var}}}) = e^{{{var}}}$"


def log_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return 1/var, f"应用自然对数规则: $\\frac{{d}}{{d{var_latex}}}\\ln({var_latex}) = \\frac{{1}}{{{var}}}$"


def sin_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return cos(var), f"应用正弦函数规则: $\\frac{{d}}{{d{var_latex}}}\\sin({var}) = \\cos({var})$"


def cos_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return -sin(var), f"应用余弦函数规则: $\\frac{{d}}{{d{var_latex}}}\\cos({var}) = -\\sin({var})$"


def tan_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return sec(var)**2, f"应用正切函数规则: $\\frac{{d}}{{d{var_latex}}}\\tan({var}) = \\sec^2({var})$"


def sec_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return sec(var)*tan(var), f"应用正割函数规则: $\\frac{{d}}{{d{var_latex}}}\\sec({var}) = \\sec({var})\\tan({var})$"


def csc_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return -csc(var)*cot(var), f"应用余割函数规则: $\\frac{{d}}{{d{var_latex}}}\\csc({var}) = -\\csc({var})\\cot({var})$"


def cot_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return -csc(var)**2, f"应用余切函数规则: $\\frac{{d}}{{d{var_latex}}}\\cot({var}) = -\\csc^2({var})$"


def asin_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return 1/((1-var**2)**0.5), f"应用反正弦函数规则: $\\frac{{d}}{{d{var_latex}}}\\arcsin({var}) = \\frac{{1}}{{\\sqrt{{1-{var}^2}}}}$"


def acos_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return -1/((1-var**2)**0.5), f"应用反余弦函数规则: $\\frac{{d}}{{d{var_latex}}}\\arccos({var}) = -\\frac{{1}}{{\\sqrt{{1-{var}^2}}}}$"


def atan_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return 1/(1+var**2), f"应用反正切函数规则: $\\frac{{d}}{{d{var_latex}}}\\arctan({var}) = \\frac{{1}}{{1+{var}^2}}$"


def sinh_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return cosh(var), f"应用双曲正弦函数规则: $\\frac{{d}}{{d{var_latex}}}\\sinh({var}) = \\cosh({var})$"


def cosh_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return sinh(var), f"应用双曲余弦函数规则: $\\frac{{d}}{{d{var_latex}}}\\cosh({var}) = \\sinh({var})$"


def tanh_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var = context['variable']
    var_latex = wrap_latex(var)
    if expr.args[0] == var:
        return 1 - tanh(var)**2, f"应用双曲正切函数规则: $\\frac{{d}}{{d{var_latex}}}\\tanh({var}) = 1 - \\tanh^2({var})$"


def const_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if expr.is_constant():
        return 'const'
    return None


def var_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    var = context['variable']
    if expr == var:
        return 'var'
    return None


def pow_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, Pow):
        return 'pow'
    return None


def exp_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, exp):
        return 'exp'
    return None


def log_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, log):
        return 'log'
    return None


def sin_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, sin):
        return 'sin'
    return None


def cos_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, cos):
        return 'cos'
    return None


def tan_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, tan):
        return 'tan'
    return None


def sec_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, sec):
        return 'sec'
    return None


def csc_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, csc):
        return 'csc'
    return None


def cot_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, cot):
        return 'cot'
    return None


def asin_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, asin):
        return 'asin'
    return None


def acos_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, acos):
        return 'acos'
    return None


def atan_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, atan):
        return 'atan'
    return None


def sinh_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, sinh):
        return 'sinh'
    return None


def cosh_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, cosh):
        return 'cosh'
    return None


def tanh_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    if isinstance(expr, tanh):
        return 'tanh'
    return None
