from typing import Union

from sympy import (
    Expr, Limit, Pow, acos, asin, atan, cos, cosh, cot, csc, exp,
    latex, log, sec, sin, sinh, tan, tanh, oo
)
from sympy.functions.elementary.trigonometric import InverseTrigonometricFunction, TrigonometricFunction

from core import RuleRegistry
from utils import Context, RuleFunction, MatcherFunctionReturn, RuleFunctionReturn
from domains.limit import get_limit_args

_create_matcher = RuleRegistry.create_common_matcher


def _create_rule(func: Union[exp, log, InverseTrigonometricFunction, TrigonometricFunction], func_name: str) -> RuleFunction:
    """Special create rule function."""
    def rule_function(expr: Expr, context: Context) -> RuleFunctionReturn:
        var, point, direction, _ = get_limit_args(context)

        arg = expr.args[0]
        expr_limit = Limit(arg, var, point, dir=direction)
        new_expr = func(expr_limit)

        return new_expr, f"应用{func_name}函数规则: ${latex(expr_limit)} = {latex(new_expr)}$"

    return rule_function


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


def pow_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    var, point, direction, dir_sup = get_limit_args(context)
    expr_latex = latex(expr)
    point_latex = f"{latex(point)}{dir_sup}"
    expr_limit_latex = f"\\lim_{{{var} \\to {point_latex}}} {expr_latex}"

    base, exponent = expr.args
    # 检查底数和指数是否含变量
    base_has_var, exp_has_var = base.has(var), exponent.has(var)

    # Case 1: 底数是常数 -> a^v => a^(lim v)
    if not base_has_var:
        new_expr = base ** Limit(exponent, var, point, dir=direction)
        rule_desc = f"应用常数底数幂规则: ${expr_limit_latex} = {latex(new_expr)}$"
        return new_expr, rule_desc

    # Case 2: 指数是常数 -> u^b => (lim u)^b
    if not exp_has_var:
        new_expr = Limit(base, var, point, dir=direction) ** exponent
        rule_desc = f"应用常数指数幂规则: ${expr_limit_latex} = {latex(new_expr)}$"

        return new_expr, rule_desc

    # Case 3: 底数和指数都含变量 -> 使用 exp-log 转换
    # 此时需要确保 base > 0 在极限点附近(- 左附近/+ 右附近)成立
    try:
        # 计算底数在极限点的极限(用于初步筛选)
        base_limit = Limit(base, var, point, dir=direction).doit()
        # 初步筛选底数极限的符号(排除明显非法情况)
        # 若底数极限为负实数(非无穷), 则邻域内可能存在负数, 直接拒绝(除非指数是奇数整数, 但复杂场景暂不处理)
        if base_limit.is_real and base_limit < 0:
            return None
        # 若底数极限为正无穷, 允许(ln(+oo) 有定义)
        if base_limit == oo:
            pass  # 后续还有邻域验证处理
        # 若底数极限为负无穷, 拒绝(ln(-oo) 无定义)
        elif base_limit == -oo:
            return None
        # 关键验证 - 邻域内底数是否严格为正(根据极限方向选择邻近点)
        # 定义邻域步长(小量, 避免数值误差)
        epsilon = 1e-8
        # 根据方向选择邻近点(左极限取 point - epsilon, 右极限取 point + epsilon)
        near_point = point + (-epsilon if direction == '-' else epsilon)
        # 计算底数在邻近点的值(数值计算)
        base_near = base.subs(var, near_point).evalf()
        # 若底数在邻近点 <=0, 拒绝转换(对数无定义)
        if base_near <= 0:
            return None

        # 所有检查通过, 可以安全应用 exp-log 变换
        log_base = log(base)
        exp_argument = exponent * log_base
        new_limit = Limit(exp_argument, var, point, dir=direction)
        new_expr = exp(new_limit)

        rule_desc = (
            f"应用指数对数变换: "
            f"${expr_limit_latex} = "
            f"\\lim_{{{latex(var)} \\to {point_latex}}} e^{{({latex(exponent)}) \\cdot \\ln {latex(base)}}} = "
            f"{latex(new_expr)}$"
        )

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


def pow_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    # Don't restrict to var == context['variable']
    if isinstance(expr, Pow):
        return 'pow'
    return None
