from typing import Optional

from sympy import (
    Abs, Expr, Integral, Pow, Rational, Wild, asin, atan, cos, cosh, cot, csc, exp,
    latex, log, sec, sin, sinh, sqrt, tan, tanh, trigsimp, csch, sech, coth
)

from core import RuleRegistry
from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn
from utils.latex_formatter import wrap_latex

_create_matcher = RuleRegistry.create_common_matcher


def _create_rule(func_name):
    return RuleRegistry.create_common_rule(Integral, func_name)


# Generate the trivial closed-form rules via the factory; the non-obvious ones
# (tan/cot/sec/csc/sech/csch/log) are hand-written below so they show the method
# and a textbook antiderivative instead of an opaque doit() blob.
log_rule = _create_rule("对数")
sin_rule = _create_rule("正弦")
cos_rule = _create_rule("余弦")
sinh_rule = _create_rule("双曲正弦")
cosh_rule = _create_rule("双曲余弦")


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
        # d/dx ln|x + sqrt(x^2-1)| = 1/sqrt(x^2-1) (the arcsec-type antiderivative).
        # acos(x) was wrong: its derivative is -1/sqrt(1-x^2) and it is undefined for |x|>1.
        result = log(Abs(var + sqrt(var ** 2 - 1)))
        return result, (f"正割型反函数积分: $\\int \\frac{{1}}{{\\sqrt{{{var}^2 - 1}}}}\\,d{var_latex} = "
                        f"\\ln\\left|{var} + \\sqrt{{{var}^2 - 1}}\\right| + C$")

    return None


def _constant_positive(m: Expr) -> Optional[bool]:
    """Decide the sign of the variable-independent constant term ``m`` in
    ``1/(x^2 + m)^n``. Returns True (arctangent branch), False (log/partial-fraction
    branch) or None when the sign cannot be determined.

    Definite for numeric ``m``. For a symbolic parameter (e.g. ``a**2``) the textbook
    convention is that the constant is positive; we confirm by probing at positive
    sample values of its free symbols. A genuine numeric ``is_positive`` always wins,
    so a negative parameter combination correctly falls to the log branch. The
    integral engine's parametric numeric-guard backstops any residual misjudgment."""
    if m.is_positive:
        return True
    if m.is_negative or m.is_zero:
        return False
    probe = m
    for sym in m.free_symbols:
        probe = probe.subs(sym, 1.7)
    if probe.is_number:
        try:
            val = float(probe.evalf())
        except (TypeError, ValueError):
            return None
        if val > 0:
            return True
        if val < 0:
            return False
    return None


def inverse_tangent_linear_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Handles integrals of the form 1/(x^2 + m)^n."""

    var = context['variable']
    var_latex = wrap_latex(var)
    # The constant term m is the base arg that does not involve the variable; SymPy's
    # arg order is not fixed (x**2 - a**2 vs x**2 + a**2), so select it by role.
    m = next(arg for arg in expr.base.args if not arg.has(var))
    m_latex = wrap_latex(m)
    m_positive = _constant_positive(m)
    if m_positive is None:
        return None
    if m_positive:
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


def tan_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """∫tan x dx = -ln|cos x| + C (= ln|sec x|)."""
    var = context['variable']
    result = -log(Abs(cos(var)))
    vl = latex(var)
    return result, f"正切积分: $\\int\\tan({vl})\\,d{vl} = -\\ln|\\cos({vl})| + C$ (因 $(\\cos)\'=-\\sin$)"


def cot_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """∫cot x dx = ln|sin x| + C."""
    var = context['variable']
    result = log(Abs(sin(var)))
    vl = latex(var)
    return result, f"余切积分: $\\int\\cot({vl})\\,d{vl} = \\ln|\\sin({vl})| + C$ (因 $(\\sin)\'=\\cos$)"


def sec_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """∫sec x dx = ln|sec x + tan x| + C (分子分母同乘 sec+tan)."""
    var = context['variable']
    result = log(Abs(sec(var) + tan(var)))
    vl = latex(var)
    return result, (
        f"正割积分: 分子分母同乘 $\\sec+\\tan$, 令 $u=\\sec({vl})+\\tan({vl})$, "
        f"$du=(\\sec\\tan+\\sec^2)d{vl}$ 恰为分子, $\\int du/u=\\ln|u|$; "
        f"$\\int\\sec({vl})\\,d{vl} = \\ln|\\sec({vl})+\\tan({vl})| + C$"
    )


def csc_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """∫csc x dx = -ln|csc x + cot x| + C."""
    var = context['variable']
    result = -log(Abs(csc(var) + cot(var)))
    vl = latex(var)
    return result, (
        f"余割积分: 同乘 $\\csc+\\cot$, 得 $\\int\\csc({vl})\\,d{vl} = "
        f"-\\ln|\\csc({vl})+\\cot({vl})| + C$"
    )


def sech_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """∫sech x dx = arctan(sinh x) + C."""
    var = context['variable']
    result = atan(sinh(var))
    vl = latex(var)
    return result, (
        f"双曲正割积分: 因 $\\frac{{d}}{{d{vl}}}\\arctan(\\sinh) = "
        f"\\frac{{\\cosh}}{{1+\\sinh^2}} = \\frac{{\\cosh}}{{\\cosh^2}} = \\operatorname{{sech}}$; "
        f"$\\int\\operatorname{{sech}}({vl})\\,d{vl} = \\arctan(\\sinh({vl})) + C$"
    )


def csch_rule(_expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """∫csch x dx = ln|tanh(x/2)| + C."""
    var = context['variable']
    result = log(Abs(tanh(var / 2)))
    vl = latex(var)
    return result, (
        f"双曲余割积分: 半角化简得 $\\int\\operatorname{{csch}}({vl})\\,d{vl} = "
        f"\\ln\\left|\\tanh\\frac{{{vl}}}{{2}}\\right| + C$"
    )


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
    # Power rule fires whenever the exponent is independent of the integration
    # variable. For single-variable input ``not expr.exp.has(var)`` is equivalent to
    # ``expr.exp.is_constant()``; the only difference is a symbolic parameter
    # exponent (x**n), which previously never matched and sent the BFS wandering
    # (hang) in parameter mode.
    if isinstance(expr, Pow) and expr.base == var and not expr.exp.has(var):
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
    """Matcher for expressions of the form 1/(x^2 + m)^n."""
    var = context['variable']

    if expr.is_Pow and expr.exp.is_Integer and expr.exp < 0:
        terms = expr.base.args
        # Exactly two Add terms: one is the pure ``var**2`` and the other a term
        # independent of ``var`` (a numeric or parameter constant such as ``a**2``).
        # SymPy's arg order is not fixed (x**2 - a**2 -> (x**2, -a**2) but
        # x**2 + a**2 -> (a**2, x**2)), so detect the roles positionally-independently.
        if expr.base.is_Add and len(terms) == 2:
            for idx in (0, 1):
                other = 1 - idx
                if terms[idx].equals(var ** 2) and not terms[other].has(var):
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


def sec_power_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for sec(x)^n (n >= 2) expressions."""
    var = context['variable']
    if expr.is_Pow:
        base, _exp = expr.as_base_exp()
        if base.func == sec and len(base.args) == 1 and base.args[0] == var \
                and _exp.is_Integer and _exp >= 2:
            return 'sec_power'
    return None


def csc_power_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for csc(x)^n (n >= 2) expressions."""
    var = context['variable']
    if expr.is_Pow:
        base, _exp = expr.as_base_exp()
        if base.func == csc and len(base.args) == 1 and base.args[0] == var \
                and _exp.is_Integer and _exp >= 2:
            return 'csc_power'
    return None


def sec_power_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Reduce sec(x)^n via the standard reduction formula (exact for n == 2)."""
    var = context['variable']
    var_latex = wrap_latex(var)
    _, n = expr.as_base_exp()
    if n == 2:
        result = tan(var)
        return result, (f"正割平方积分: $\\int \\sec^2({var_latex})\\,d{var_latex} = "
                        f"\\tan({var_latex}) + C$")
    result = (sec(var) ** (n - 2) * tan(var) / (n - 1)
              + (n - 2) / (n - 1) * Integral(sec(var) ** (n - 2), var))
    n_latex = wrap_latex(n)
    return result, (f"正割幂递推: $\\int \\sec^{{{n_latex}}} = "
                    f"\\frac{{\\sec^{{{n_latex}-2}}\\tan}}{{{n_latex}-1}} + "
                    f"\\frac{{{n_latex}-2}}{{{n_latex}-1}}\\int \\sec^{{{n_latex}-2}}$")


def csc_power_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Reduce csc(x)^n via the standard reduction formula (exact for n == 2)."""
    var = context['variable']
    var_latex = wrap_latex(var)
    _, n = expr.as_base_exp()
    if n == 2:
        result = -cot(var)
        return result, (f"余割平方积分: $\\int \\csc^2({var_latex})\\,d{var_latex} = "
                        f"-\\cot({var_latex}) + C$")
    result = (-csc(var) ** (n - 2) * cot(var) / (n - 1)
              + (n - 2) / (n - 1) * Integral(csc(var) ** (n - 2), var))
    n_latex = wrap_latex(n)
    return result, (f"余割幂递推: $\\int \\csc^{{{n_latex}}} = "
                    f"-\\frac{{\\csc^{{{n_latex}-2}}\\cot}}{{{n_latex}-1}} + "
                    f"\\frac{{{n_latex}-2}}{{{n_latex}-1}}\\int \\csc^{{{n_latex}-2}}$")


def tan_sec_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for tan(x) * sec(x)."""
    var = context['variable']
    if expr.equals(tan(var) * sec(var)):
        return 'tan_sec'
    return None


def tan_sec_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:  # pylint: disable=unused-argument
    """Integral of tan(x) * sec(x) is sec(x)."""
    var = context['variable']
    var_latex = wrap_latex(var)
    return sec(var), (f"正割正切乘积积分: $\\int \\sec({var_latex})\\tan({var_latex})\\,d{var_latex} = "
                      f"\\sec({var_latex}) + C$")


def cot_csc_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for cot(x) * csc(x)."""
    var = context['variable']
    if expr.equals(cot(var) * csc(var)):
        return 'cot_csc'
    return None


def _has_trig_power(expr: Expr, var) -> bool:
    return any(isinstance(a, Pow) and a.base.func in (sin, cos) and a.exp.is_Integer
               and a.exp > 0 and a.base.args == (var,) for a in expr.atoms(Pow))


def _fully_linearized(expr: Expr) -> bool:
    return not any(isinstance(a, Pow) and a.base.func in (sin, cos) and a.exp.is_Integer
                   and a.exp > 0 for a in expr.atoms(Pow))


def trig_power_reduce_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Match a product of sin/cos integer powers that trigsimp can reduce to a
    linear combination of multiple-angle terms (e.g. sin^2 * cos^2)."""
    var = context['variable']
    if not expr.is_Mul or not _has_trig_power(expr, var):
        return None
    reduced = trigsimp(expr)
    if reduced == expr or not _fully_linearized(reduced):
        return None
    return 'trig_power_reduce'


def trig_power_reduce_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Rewrite sin^m * cos^n into multiple-angle form so each term integrates easily."""
    var = context['variable']
    reduced = trigsimp(expr)
    return Integral(reduced, var), (
        f"降幂(倍角/积化和差): $\\int {latex(expr)}\\,d{latex(var)} = "
        f"\\int {latex(reduced)}\\,d{latex(var)}$"
    )


def _sqrt_one_pm_trig_reduced(expr: Expr, var):
    """If ``expr`` is ``sqrt(1 ± cos(a*x+b))``/``sqrt(1 ± sin(a*x+b))`` (or their
    reciprocals ``1/sqrt(...)``), return the half-angle linear form (principal branch,
    |·| dropped as CAS convention), else None.

    1+cosθ = 2cos²(θ/2), 1−cosθ = 2sin²(θ/2), 1+sinθ = (sin(θ/2)+cos(θ/2))²,
    1−sinθ = (cos(θ/2)−sin(θ/2))². Rewriting avoids the sign error / runaway the radical
    substitution hits when inverting these through acos/arcsin branches.
    """
    if not (expr.is_Pow and expr.exp in (Rational(1, 2), Rational(-1, 2))):
        return None
    base = expr.args[0]
    if not base.is_Add:
        return None
    A, B, C = Wild('A'), Wild('B'), Wild('C')
    for fn in (cos, sin):
        m = base.match(1 + A * fn(B * var + C))
        if not m:
            continue
        a, k = m.get(A), m.get(B)
        if a not in (-1, 1) or k is None or k == 0 or k.has(var):
            continue
        theta = k * var + m[C]
        if fn is cos:
            form = sqrt(2) * cos(theta / 2) if a == 1 else sqrt(2) * sin(theta / 2)
        else:
            form = sin(theta / 2) + cos(theta / 2) if a == 1 else cos(theta / 2) - sin(theta / 2)
        return form if expr.exp == Rational(1, 2) else 1 / form
    return None


def _rewrite_sqrt_one_pm_trig(expr: Expr, var):
    """Apply the half-angle rewrite to every sqrt(1 ± cos/sin(linear)) sub-expression
    of ``expr`` (not just when the whole integrand is such a sqrt), e.g. the radical in
    (sin x + cos x)/sqrt(1 + sin 2x)."""
    def _test(e):
        return _sqrt_one_pm_trig_reduced(e, var) is not None
    return expr.replace(_test, lambda e: _sqrt_one_pm_trig_reduced(e, var))


def sqrt_one_pm_trig_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Match an integrand containing sqrt(1 ± cos/sin(linear)) to rewrite via half-angle."""
    var = context['variable']
    if _rewrite_sqrt_one_pm_trig(expr, var) == expr:
        return None
    return 'sqrt_one_pm_trig'


def sqrt_one_pm_trig_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Rewrite sqrt(1 ± cos/sin(linear)) sub-expressions into a half-angle form the
    basic rules integrate (avoids the radical-substitution sign hang)."""
    var = context['variable']
    reduced = _rewrite_sqrt_one_pm_trig(expr, var)
    return Integral(reduced, var), (
        f"半角化: $1\\pm\\cos\\theta=2\\cos^2(\\theta/2)$ / $1\\pm\\sin\\theta=(\\sin(\\theta/2)\\pm\\cos(\\theta/2))^2$，"
        f"$\\int {latex(expr)}\\,d{latex(var)} = \\int {latex(reduced)}\\,d{latex(var)}$"
    )


def cot_csc_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:  # pylint: disable=unused-argument
    """Integral of cot(x) * csc(x) is -csc(x)."""
    var = context['variable']
    var_latex = wrap_latex(var)
    return -csc(var), (f"余割余切乘积积分: $\\int \\csc({var_latex})\\cot({var_latex})\\,d{var_latex} = "
                       f"-\\csc({var_latex}) + C$")
