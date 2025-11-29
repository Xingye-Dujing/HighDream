from sympy import E, Expr, Integer, Pow, exp, latex, log, sin, simplify

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn
from domains.limit import check_function_tends_to_zero


def _get_limit_args(context: RuleContext) -> tuple:
    """Special for dir_sup."""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    dir_sup = '^{+}' if direction == '+' else '^{-}'
    return var, point, direction, dir_sup


def sin_over_x_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Applies the standard limit rule: u to 0, sin(u)/u to 1.

    To expressions of the form:
        - sin(f(x))/g(x)   where f(x)/g(x) to c (constant),
        - sin(f(x))*h(x)   where f(x)*h(x) to c (constant).

    The rule is applicable only when the inner argument f(x) tends to 0 at the limit point.
    """
    var, point, _, dir_sup = _get_limit_args(context)
    ratio = None

    if expr.is_Mul:
        # Handle product form: sin(f(x))*h(x)
        sin_factor = None
        other_factor = 1
        for arg in expr.args:
            if isinstance(arg, sin):
                sin_factor = arg
            else:
                other_factor *= arg
        sin_arg = sin_factor.args[0]
        ratio = sin_arg * other_factor
        den = 1/other_factor
    else:
        # Handle quotient form: sin(f(x))/g(x)
        num, den = expr.as_numer_denom()
        if isinstance(num, sin):
            sin_arg = num.args[0]
            ratio = sin_arg / den

    # Build LaTeX explanation
    var_latexatex, t_sub_latex, point_latexatex, ratio_latexatex = latex(
        var), latex(sin_arg), latex(point), latex(ratio)

    rule_text = f"\\lim_{{x \\to {point_latexatex}{dir_sup}}} {latex(expr)} = "

    is_identity = t_sub_latex == var_latexatex and ratio == 1

    if is_identity:
        rule_text += '1'
    else:
        if t_sub_latex != var_latexatex:
            lim_expr = f"\\lim_{{t \\to 0{dir_sup}}}"
            rule_text += (
                f"{'' if ratio == 1 else ratio_latexatex} {lim_expr} \\frac{{\\sin(t)}}{{t}} = "
                f"{'1' if ratio == 1 else ratio_latexatex}"
                f"\\quad \\text{{(令 }} t = {t_sub_latex} \\text{{)}}"
            )
        else:
            lim_expr = f"\\lim_{{x \\to 0{dir_sup}}}"
            rule_text += (
                f"{'' if ratio == 1 else ratio_latexatex} {lim_expr} \\frac{{\\sin(x)}}{{x}} = "
                f"{'1' if ratio == 1 else ratio_latexatex}"
            )

    result = Integer(1) if ratio == 1 else ratio
    return result, f"重要极限: ${rule_text}$"


def one_plus_one_over_x_pow_x_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Applies the standard exponential limit: u to 0, (1+u)^(1/u) = e,

    To expressions of the form:
        - (1+f(x))^{g(x)}   where f(x) to 0 and f(x)*g(x) to c (constant),
        - equivalently, (1+1/h(x))^h(x) when h(x) yo +-oo.

    The rule is valid only if the base tends to 1 and the product f(x)*exponent
    approaches a finite constant.
    """
    var, point, _, dir_sup = _get_limit_args(context)
    var_latex, point_latex = latex(var), latex(point)

    base, _exp = expr.as_base_exp()
    # Identify f(x) such that base = 1 + f(x)
    inv_term = simplify(base - 1)
    f_expr = inv_term
    ratio = f_expr * _exp

    # Build LaTeX explanation
    if ratio == 1:
        body = f"\\lim_{{{var_latex} \\to {point_latex}{dir_sup}}} {latex(expr)} = e"
    else:
        body = (
            f"\\lim_{{{var_latex} \\to {point_latex}{dir_sup}}} {latex(expr)} = "
            f"\\lim_{{{var_latex} \\to {point_latex}{dir_sup}}} "
            f"\\left[(1 + {latex(f_expr)})^{{\\frac{{1}}{{{latex(f_expr)}}}}}\\right]^{{{latex(ratio)}}}.\\quad"
            f"\\lim_{{f(x) \\to 0{dir_sup}}} (1+f(x))^{{\\frac{{1}}{{f(x)}}}} = e,"
            f"\\text{{故原极限为 }} e^{{{latex(ratio)}}}."
        )

    c_simplified = simplify(ratio)
    result = Integer(1) if c_simplified.is_zero else E ** c_simplified
    rule_text = f"重要极限: ${body}$"
    return result, rule_text


def ln_one_plus_x_over_x_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Applies the standard logarithmic limit: u to 0, ln(1+u)/u = 1.

    To expressions of the form:
        - ln(1+f(x))/g(x)   where f(x)/g(x) to c (constant),
        - ln(1+f(x))*h(x)   where f(x)*h(x) to c (constant).

    The rule is valid only when f(x) to 0 as x approaches the limit point from the given direction.
    """
    var, point, dir_sup, _ = _get_limit_args(context)

    ratio = None

    if expr.is_Mul:
        # Handle product form: ln(1+f(x))*h(x)
        log_factor, other_factor = None, 1
        for arg in expr.args:
            if isinstance(arg, log):
                log_factor = arg
            else:
                other_factor *= arg
        f = log_factor.args[0] - 1
        ratio = simplify(f * other_factor)
    else:
        # Handle quotient form: ln(1+f(x))/g(x)
        numerator, denominator = expr.as_numer_denom()
        f = numerator.args[0] - 1
        ratio = simplify(f / denominator)

    # Build LaTeX explanation
    ratio_latex = "" if ratio == 1 else latex(ratio)
    f_latex, var_latex, point_latex = latex(f), latex(var), latex(point)
    lim_expr = f"\\lim_{{{var_latex} \\to {point_latex}{dir_sup}}}"

    result = Integer(1) if ratio == 1 else ratio
    expr_latex, result_latex = latex(expr), latex(result)

    if ratio == 1 and f == var:
        rule_text = f"{lim_expr} {expr_latex} = 1"
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} {lim_expr} \\frac{{\\ln(1+{var_latex})}}{{{var_latex}}} = {result_latex}"
        )
    else:
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} \\lim_{{t \\to 0{dir_sup}}} \\frac{{\\ln(1+t)}}{{t}} = {result_latex}"
            f" \\quad \\text{{(令 }} t = {f_latex} \\text{{)}}"
        )

    return result, f"重要极限: ${rule_text}$"


def g_over_exp_minus_one_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Applies the standard exponential limit: u to 0, u/(e^u-1) = 1.

    To expressions of the form:
        g(x)/(e^f(x)-1),
    where f(x) to 0 and the ratio g(x)/f(x) to c (a finite constant).

    This rule is valid only when the exponent tends to zero at the limit point.
    """
    var, point, _, dir_sup = _get_limit_args(context)
    num, den = expr.as_numer_denom()

    exp_part = [a for a in den.args if a.has(exp)]

    f = exp_part[0].args[0]
    ratio = simplify(num / f)

    # Build LaTeX explanation
    ratio_latex = "" if ratio == 1 else latex(ratio)
    f_latex, var_latex, point_latex = latex(f),  latex(var), latex(point)
    lim_expr = f"\\lim_{{{var_latex} \\to {point_latex}{dir_sup}}}"

    result = Integer(1) if ratio == 1 else ratio
    expr_latex, result_latex = latex(expr), latex(result)

    # Earliest special case
    if ratio == 1 and f == var:
        rule_text = f"{lim_expr} {expr_latex} = 1"
    # f(x) == x
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} {lim_expr} \\frac{{{var_latex}}}{{e^{{{var_latex}}} - 1}} = {result_latex }"
        )
    # General case
    else:
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} \\lim_{{t \\to 0{dir_sup}}} \\frac{{t}}{{e^t - 1}} = {result_latex }"
            f" \\quad \\text{{(令 }} t = {f_latex} \\text{{)}}"
        )

    return result, f"重要极限: ${rule_text}$"


def g_over_ln_one_plus_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Applies the standard logarithmic limit: u to 0, u/ln(1+u) = 1.

    To expressions of the form:
        g(x)/ln(1+f(x))},
    where f(x) to 0 and the ratio g(x)/f(x) to c (a finite constant).

    This rule is valid only when the argument of the logarithm tends to 1,
    i.e., f(x) to 0 at the limit point from the specified direction.
    """
    var, point, _, dir_sup = _get_limit_args(context)
    num, den = expr.as_numer_denom()
    f = den.args[0] - 1
    ratio = simplify(num / f)

    # Build LaTeX explanation
    expr_latex, var_latex, point_latex, f_latex = latex(
        expr), latex(var), latex(point), latex(f)
    lim_expr = f"\\lim_{{x \\to {point_latex}{dir_sup}}}"
    ratio_latex = "" if ratio == 1 else latex(ratio)

    result = Integer(1) if ratio == 1 else ratio

    rule_text = ''
    if ratio != 1 and f != var:
        lim_expr = f"\\lim_{{t \\to {point_latex}{dir_sup}}}"
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} {lim_expr} \\frac{{t}}{{\\ln(1+t)}} = "
            f"{latex(result)} "
            f"\\quad \\text{{(令 }} t = {f_latex} \\text{{)}}"
        )
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} {lim_expr} \\frac{{{var_latex}}}{{\\ln(1+{var_latex})}} = "
            f"{latex(result)}"
        )
    else:
        rule_text = f"{lim_expr} {expr_latex} = 1"

    return result, f"重要极限: ${rule_text}$"


def g_over_sin_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Applies the standard trigonometric limit: u to 0, u/sin(u) = 1.

    To expressions of the form:
        g(x)/sin(f(x)),
    where f(x) to 0 and the ratio g(x)/f(x) to c (a finite constant).

    This rule is valid only when the argument of the sine tends to 0 at the limit point.
    """
    var, point, _, dir_sup = _get_limit_args(context)
    num, den = expr.as_numer_denom()
    f = den.args[0]
    ratio = simplify(num / f)

    # Build LaTeX explanation
    expr_latex, var_latex, point_latex, f_latex = latex(
        expr), latex(var), latex(point), latex(f)
    lim_expr = f"\\lim_{{{var_latex} \\to {point_latex}{dir_sup}}}"
    ratio_latex = "" if ratio == 1 else latex(ratio)

    result = Integer(1) if ratio == 1 else ratio

    rule_text = ""
    if ratio != 1 and f != var:
        lim_expr_t = f"\\lim_{{t \\to {point_latex}{dir_sup}}}"
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} {lim_expr_t} \\frac{{t}}{{\\sin(t)}} = "
            f"{latex(result)} "
            f"\\quad \\text{{(令 }} t = {f_latex} \\text{{)}}"
        )
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} {lim_expr} \\frac{{{var_latex}}}{{\\sin({var_latex})}} = "
            f"{latex(result)}"
        )
    else:
        rule_text = f"{lim_expr} {expr_latex} = 1"

    return result, f"重要极限: ${rule_text}$"


def exp_minus_one_over_x_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Applies the standard exponential limit: u to 0, (e^u-1)/u = 1.

    To expressions of the form:
        (e^f(x)-1}/g(x) or (e^f(x)-1)*h(x),
    where f(x) to 0 and the effective ratio f(x)/g(x) or f(x)*h(x) tends to a finite constant.

    This rule is valid only when f(x) to 0 at the limit point from the specified direction.
    """
    var, point, _, dir_sup = _get_limit_args(context)

    ratio = None

    if expr.is_Mul:
        # Handle product form: (e^f(x)-1)*h(x)
        exp_factor, other_factor = None, 1
        for arg in expr.args:
            if arg.is_Add and arg.has(exp):
                exp_factor = arg
            else:
                other_factor *= arg
        # Extract e^f(x) -1
        exp_part = [a for a in exp_factor.args if a.has(exp)]
        f = exp_part[0].args[0]
        ratio = simplify(f * other_factor)
    else:
        # Handle quotient form: (e^f(x)-1)/g(x)
        numerator, denominator = expr.as_numer_denom()
        exp_part = [a for a in numerator.args if a.has(exp)]
        f = exp_part[0].args[0]
        ratio = simplify(f / denominator)

    # Build LaTeX explanation
    ratio_latex = "" if ratio == 1 else latex(ratio)
    f_latex, var_latex, point_latex = latex(f), latex(var), latex(point)
    lim_expr = f"\\lim_{{{var_latex} \\to {point_latex}{dir_sup}}}"

    result = Integer(1) if ratio == 1 else ratio
    expr_latex, result_latex = latex(expr), latex(result)

    if ratio == 1 and f == var:
        rule_text = f"{lim_expr} {expr_latex} = 1"
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} {lim_expr} \\frac{{e^{{{var_latex}}} - 1}}{{{var_latex}}} = {result_latex}"
        )
    else:
        rule_text = (
            f"{lim_expr} {expr_latex} = "
            f"{ratio_latex} \\lim_{{t \\to 0{dir_sup}}} \\frac{{e^t - 1}}{{t}} = {result_latex}"
            f" \\quad \\text{{(令 }} t = {f_latex} \\text{{)}}"
        )

    return result, f"重要极限: ${rule_text}$"


def sin_over_x_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions that can be reduced to the standard limit: u to 0, sin(u)/u = 1.

    i.e., forms like:
        sin(f(x))/g(x) or sin(f(x))*h(x),
    where f(x) to 0 and the effective ratio f(x)/g(x) or product f(x)*h(x) tends to a nonzero constant.

    This matcher only returns 'sin_over_x' if:
      1. The sine argument f(x) to 0 as x to point (from the given direction).
      2. The asymptotic coefficient (f/g or f*h) is effectively constant near the limit point.
    """
    var, point, direction, _ = _get_limit_args(context)

    # Handle product form: sin(f(x))*h(x)
    if expr.is_Mul:
        sin_factor, other_factor = None, 1
        for arg in expr.args:
            if isinstance(arg, sin):
                sin_factor = arg
            else:
                other_factor *= arg

        if sin_factor is not None:
            sin_arg = sin_factor.args[0]
            product = sin_arg * other_factor
            if not product.has(var) and check_function_tends_to_zero(sin_arg, var, point, direction):
                return 'sin_over_x'
    else:
        # Handle quotient form: sin(f(x))/g(x)
        numerator, denominator = expr.as_numer_denom()
        if isinstance(numerator, sin):
            sin_arg = numerator.args[0]
            ratio = sin_arg / denominator
            if not ratio.has(var) and check_function_tends_to_zero(sin_arg, var, point, direction):
                return 'sin_over_x'

    return None


def one_plus_one_over_x_pow_x_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions that reduce to the standard exponential limit:
        u to 0, (1+u)^(1/u) = e,
    or equivalently,
        v to oo (1+1/v)^v = e.

    This includes forms like:
        (1+f(x))^g(x},
    where f(x) to 0 and the product f(x)*g(x) to c (a finite nonzero constant).

    The matcher returns 'one_plus_one_over_x_pow_x' if:
      1. The base is of the form 1+u(x) with u(x) to 0,
      2. The exponent g(x) is such that u(x)*g(x) tends to a constant (independent of x).
    """
    var, point, direction, _ = _get_limit_args(context)
    if not isinstance(expr, Pow):
        return None
    base, _exp = expr.as_base_exp()
    inv_f = base - 1
    ratio = simplify(inv_f * _exp)
    if check_function_tends_to_zero(inv_f, var, point, direction) and not ratio.has(var):
        return 'one_plus_one_over_x_pow_x'

    return None


def ln_one_plus_x_over_x_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions that reduce to the standard logarithmic limit: u to 0, ln(1+u)/u = 1.

    This includes forms like:
        ln(1+f(x))/g(x) or ln(1+f(x))*h(x),
    where f(x) to 0 and the effective coefficient f(x)/g(x) or f(x)*h(x) tends to a finite constant.

    The matcher returns 'ln_one_plus_x_over_x' if:
      1. The logarithm argument is of the form 1+f(x) with f(x) to 0,
      2. The asymptotic ratio f(x)/denominator (after normalizing to quotient form)
         converges to a constant independent of the limit variable.
    """
    var, point, direction, _ = _get_limit_args(context)

    if expr.is_Mul:
        # Handle product form: ln(1+f(x))*h(x)
        log_factor, other_factor = None, 1
        for arg in expr.args:
            if isinstance(arg, log):
                log_factor = arg
            else:
                other_factor *= arg
        if log_factor is not None:
            f = log_factor.args[0] - 1
            product = f * other_factor
            if not product.has(var) and check_function_tends_to_zero(f, var, point, direction):
                return 'ln_one_plus_x_over_x'
    else:
        # Handle quotient form: ln(1+f(x))/g(x)
        numerator, denominator = expr.as_numer_denom()
        if isinstance(numerator, log):
            f = numerator.args[0] - 1
            ratio = f / denominator
            if not ratio.has(var) and check_function_tends_to_zero(f, var, point, direction):
                return 'ln_one_plus_x_over_x'

    return None


def exp_minus_one_over_x_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions that reduce to the standard exponential limit: u to 0, (e^u-1)/u = 1.

    This includes forms like:
        (e^f(x)-1)/g(x) or (e^f(x)-1)*h(x),
    where f(x) to 0 and the effective coefficient f(x)/g(x) or f(x)*h(x) tends to a finite constant.

    The matcher returns 'exp_minus_one_over_x' if:
      1. The expression contains a term of the form e^f(x)-1,
      2. f(x) to 0 at the limit point (from the specified direction),
      3. The asymptotic ratio f(x)/denominator (after canonical normalization)
         converges to a constant independent of the limit variable.
    """
    var, point, direction, _ = _get_limit_args(context)

    if expr.is_Mul:
        # Handle product form: (e^f(x)-1)*h(x)
        exp_factor, other_factor = None, 1
        for arg in expr.args:
            if arg.is_Add and arg.has(exp):
                exp_factor = arg
            else:
                other_factor *= arg
        if exp_factor is not None:
            exp_part = [a for a in exp_factor.args if a.has(exp)]
            if exp_part:
                f = exp_part[0].args[0]
                product = f * other_factor
                if not product.has(var) and check_function_tends_to_zero(f, var, point, direction):
                    return 'exp_minus_one_over_x'
    else:
        # Handle quotient form:
        numerator, denominator = expr.as_numer_denom()
        if numerator.is_Add and numerator.has(exp):
            exp_part = [a for a in numerator.args if a.has(exp)]
            if exp_part:
                f = exp_part[0].args[0]
                ratio = f / denominator
                if not ratio.has(var) and check_function_tends_to_zero(f, var, point, direction):
                    return 'exp_minus_one_over_x'

    return None


def g_over_sin_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions that reduce to the reciprocal of the standard sine limit: u to 0, u/sin(u) = 1,

    i.e., forms like:
        g(x)/sin(f(x)),
    where f(x) to 0 and the ratio g(x)/f(x) tends to a finite constant.

    This pattern arises when the denominator is sin(f(x)) with f(x) to 0,
    and the numerator behaves asymptotically like a constant multiple of f(x).

    The matcher returns 'g_over_sin' if:
      1. The denominator is exactly sin(f(x)),
      2. f(x) to 0 as x approaches the limit point (from the given direction),
      3. The ratio g(x) / f(x) converges to a constant independent of the limit variable.
    """
    var, point, direction, _ = _get_limit_args(context)
    num, den = expr.as_numer_denom()
    if isinstance(den, sin):
        f = den.args[0]
        ratio = simplify(num / f)
        if not ratio.has(var) and check_function_tends_to_zero(f, var, point, direction):
            return 'g_over_sin'
    return None


def g_over_ln_one_plus_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions that reduce to the reciprocal of the standard logarithmic limit: u to 0, u/ln(1+u) = 1,

    i.e., forms like:
        g(x)/ln(1+f(x)),
    where f(x) to 0 and the ratio g(x)/f(x) tends to a finite constant.

    This pattern arises when the denominator is ln(1+f(x)) with f(x) to 0,
    and the numerator behaves asymptotically like a constant multiple of f(x).

    The matcher returns 'g_over_ln_one_plus' if:
      1. The denominator is exactly log(1 + f(x)),
      2. f(x) to 0 as x approaches the limit point (from the given direction),
      3. The ratio g(x)/f(x) converges to a constant independent of the limit variable.
    """
    var, point, direction, _ = _get_limit_args(context)
    num, den = expr.as_numer_denom()
    if isinstance(den, log):
        f = den.args[0] - 1
        ratio = simplify(num / f)
        if not ratio.has(var) and check_function_tends_to_zero(f, var, point, direction):
            return 'g_over_ln_one_plus'
    return None


def g_over_exp_minus_one_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions that reduce to the reciprocal of the standard exponential limit: u to 0, u/(e^u-1) = 1,

    i.e., forms like:
        g(x)/(e^f(x)-1),
    where f(x) to 0 and the ratio g(x)/f(x) tends to a finite constant.

    This pattern arises when the denominator is exactly e^f(x) - 1 with f(x) to 0,
    and the numerator behaves asymptotically like a constant multiple of f(x).

    The matcher returns 'g_over_exp_minus_one' if:
      1. The denominator is precisely exp(f(x)) - 1 (a two-term sum with one exp and one -1),
      2. f(x) to 0 as x approaches the limit point (from the given direction),
      3. The ratio g(x)/f(x) converges to a constant independent of the limit variable.
    """
    var, point, direction, _ = _get_limit_args(context)
    num, den = expr.as_numer_denom()

    if not (den.is_Add and den.has(exp)):
        return None

    # Extract e^f(x)
    exp_part = [a for a in den.args if a.has(exp)]
    if not exp_part:
        return None

    f = exp_part[0].args[0]
    ratio = simplify(num / f)

    if not ratio.has(var) and check_function_tends_to_zero(f, var, point, direction):
        return "g_over_exp_minus_one"

    return None
