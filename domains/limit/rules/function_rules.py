from typing import Type

from sympy import (
    Expr, Limit, Pow, S, acos, asin, atan, cos, cosh, cot, csc, exp,
    latex, log, sec, sin, sinh, tan, tanh, oo
)
from sympy.functions.elementary.trigonometric import InverseTrigonometricFunction, TrigonometricFunction

from domains.limit.limit_help_func import get_limit_args
from utils import MatcherFunction, MatcherFunctionReturn, RuleContext, RuleFunction, RuleFunctionReturn


def _create_rule(func: Type[exp | log | InverseTrigonometricFunction | TrigonometricFunction],
                 func_name: str) -> RuleFunction:
    """Specially create rule function."""

    def rule_function(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
        var, point, direction = get_limit_args(context)

        arg = expr.args[0]
        expr_limit = Limit(arg, var, point, dir=direction)
        new_expr = func(expr_limit)

        return new_expr, f"应用{func_name}函数规则: ${latex(expr_limit)} = {latex(new_expr)}$"

    return rule_function


def _create_matcher(
        func: Type[exp | log | InverseTrigonometricFunction | TrigonometricFunction]) -> MatcherFunction:
    """Special create matcher function."""

    def matcher_function(expr: Expr, _context: RuleContext) -> MatcherFunctionReturn:
        # Don't restrict to var == context['variable']
        if isinstance(expr, func):
            return func.__name__.lower()
        return None

    return matcher_function


# Generate all exp and log rules using the factory function
exp_rule = _create_rule(exp, "指数")
log_rule = _create_rule(log, "对数")
# Generate all trigonometric and hyperbolic rules using the factory function
sin_rule = _create_rule(sin, "正弦")
cos_rule = _create_rule(cos, "余弦")
tan_rule = _create_rule(tan, "正切")
sec_rule = _create_rule(sec, "正割")
csc_rule = _create_rule(csc, "余割")
cot_rule = _create_rule(cot, "余切")
asin_rule = _create_rule(asin, "反正弦")
acos_rule = _create_rule(acos, "反余弦")
atan_rule = _create_rule(atan, "反正切")
sinh_rule = _create_rule(sinh, "双曲正弦")
cosh_rule = _create_rule(cosh, "双曲余弦")
tanh_rule = _create_rule(tanh, "双曲正切")


def pow_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Applies limit rules for expressions of the form `base ** exponent`.

    Handles three cases:
      1. Constant base: a**f(x) to a**lim f(x) (if a is constant).
      2. Constant exponent: f(x)**b to (lim f(x))**b (if b is constant).
      3. Variable base and exponent: Transforms f(x)**g(x) into
         exp(lim g(x)*log(f(x))), provided f(x) > 0 in a punctured neighborhood
         of the limit point (to ensure the logarithm is well-defined).

    For case 3, the function performs both symbolic and numerical checks to verify
    that the base remains positive near the limit point from the specified direction.
    """

    var, point, direction = get_limit_args(context)
    expr_limit_latex = latex(Limit(expr, var, point, dir=direction))

    base, exponent = expr.args
    base_has_var, exp_has_var = base.has(var), exponent.has(var)

    # Case 1: Constant base
    if not base_has_var:
        new_expr = base ** Limit(exponent, var, point, dir=direction)
        rule_desc = f"应用常数底数幂规则: ${expr_limit_latex} = {latex(new_expr)}$"
        return new_expr, rule_desc

    # Case 2: Constant exponent
    if not exp_has_var:
        new_expr = Limit(base, var, point, dir=direction) ** exponent
        rule_desc = f"应用常数指数幂规则: ${expr_limit_latex} = {latex(new_expr)}$"

        return new_expr, rule_desc

    # Case 3: Both base and exponent depend on var
    try:
        base_limit = Limit(base, var, point, dir=direction).doit()
        # Reject if base tends to a negative real number (log undefined in reals).
        if base_limit.is_real and base_limit < 0:
            return None
        # Reject if base tends to negative infinity.
        if base_limit == -oo:
            return None

        # For +oo, proceed (log(+oo) = +oo is acceptable in extended reals).

        # Perform a local positivity check near the limit point.
        epsilon = S(1e-8)  # Use SymPy Rational/Float for consistency
        offset = -epsilon if direction == '-' else epsilon
        near_point = point + offset

        # Evaluate base numerically at a nearby point.
        base_near_val = base.subs(var, near_point).evalf()

        # Require strict positivity to define log in a real neighborhood.
        if not (base_near_val.is_real and base_near_val > 0):
            return None

        # Safe to apply exp-log transformation.
        log_base = log(base)
        exp_argument = exponent * log_base
        new_limit = Limit(exp_argument, var, point, dir=direction)
        new_expr = exp(new_limit)

        rule_desc = f"应用指数对数变换: ${expr_limit_latex} = {latex(new_expr)}$"

        return new_expr, rule_desc
    except Exception:
        return None


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
pow_matcher = _create_matcher(Pow)
