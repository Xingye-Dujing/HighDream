# TODO 下面都有 乘法形式 和 分式形式 两种. 深入研究分式形式是否是不必要的, 即仅 乘法形式 是否就涵盖所有情况
# TODO 另外想一个好办法去提出 exp(f(x))-1, 现在写的太死，局限性很高

from typing import Any, Dict, Optional, Tuple

from sympy import E, Expr, Integer, Limit,  Pow, Symbol, exp, latex, log, oo, sin, simplify


def _get_limit_args(context: Dict[str, Any]) -> tuple:
    """获取极限参数"""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    dir_sup = '^{+}' if direction == '+' else '^{-}'
    return var, point, direction, dir_sup


def _check_function_tends_to_zero(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查函数在给定点是否趋于 0"""
    try:
        limit_val = Limit(expr, var, point, dir=direction).doit()
        return limit_val == 0
    except Exception:
        return False


def sin_over_x_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """重要极限: sin(f(x))/h(x) 或 sin(f(x))*h(x) 形式, f(x)/h(x)=常数 或 f(x)*h(x)=常数"""
    var, point, _, dir_sup = _get_limit_args(context)
    ratio = None

    if expr.is_Mul:
        # 处理形式 sin(f(x))*h(x)
        sin_factor = None
        other_factor = 1
        for arg in expr.args:
            if isinstance(arg, sin):
                sin_factor = arg
            else:
                other_factor *= arg
        sin_arg = sin_factor.args[0]
        ratio = sin_arg * other_factor
        den = 1/other_factor if other_factor != 0 else oo
    else:
        # 标准分式形式
        num, den = expr.as_numer_denom()
        if isinstance(num, sin):
            sin_arg = num.args[0]
            ratio = sin_arg / den

    t_sub = latex(sin_arg)
    var_to = latex(point)

    rule_text = f"\\lim_{{x \\to {var_to}{dir_sup}}} {latex(expr)} = "

    is_identity = t_sub == latex(var) and ratio == 1

    if is_identity:
        rule_text += '1'
    else:
        if t_sub != latex(var):
            lim_expr = f"\\lim_{{t \\to 0{dir_sup}}}"
            rule_text += (
                f"{'' if ratio == 1 else latex(ratio)} {lim_expr} \\frac{{\\sin(t)}}{{t}} = "
                f"{'1' if ratio == 1 else latex(ratio)}"
                f"\\quad \\text{{(令 }} t = {t_sub} \\text{{)}}"
            )
        else:
            lim_expr = f"\\lim_{{x \\to 0{dir_sup}}}"
            rule_text += (
                f"{'' if ratio == 1 else latex(ratio)} {lim_expr} \\frac{{\\sin(x)}}{{x}} = "
                f"{'1' if ratio == 1 else latex(ratio)}"
            )

    result = Integer(1) if ratio == 1 else ratio
    return result, f"重要极限: ${rule_text}$"


def one_plus_one_over_x_pow_x_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """重要极限: (1 + 1/f(x))^f(x) -> e 或 (1 + f(x))^(1/f(x)) -> e 的形式"""
    var, point, direction, dir_sup = _get_limit_args(context)
    var_l, point_l, dir_l = latex(var), latex(point), dir_sup

    base, _exp = expr.as_base_exp()
    # 统一为 (f(x) + 1)**h(x) 形式处理
    inv_term = simplify(base - 1)  # 即 (1 + f(x)) 里的 f(x)
    f_expr = inv_term
    g_expr = _exp
    ratio = f_expr * g_expr
    if ratio == 1:
        body = (
            f"\\lim_{{{var_l} \\to {point_l}{dir_l}}} {latex(expr)} = e"
        )
    else:
        body = (
            f"\\lim_{{{var_l} \\to {point_l}{dir_l}}} {latex(expr)} = "
            f"\\lim_{{{var_l} \\to {point_l}{dir_l}}} "
            f"\\left[(1 + {latex(f_expr)})^{{\\frac{{1}}{{{latex(f_expr)}}}}}\\right]^{{{latex(ratio)}}}.\\quad"
            f"\\lim_{{f(x) \\to 0{dir_sup}}} (1+f(x))^{{\\frac{{1}}{{f(x)}}}} = e,"
            f"\\text{{故原极限为 }} e^{{{latex(ratio)}}}."
        )

    c_simplified = simplify(ratio)
    result = Integer(1) if c_simplified.is_zero else E ** c_simplified
    rule_text = f"重要极限: ${body}$"
    return result, rule_text


def ln_one_plus_x_over_x_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """重要极限: ln(1+f(x))/g(x) 或 ln(1+f(x))*h(x)，其中 f(x)/g(x) 或 f(x)*h(x) -> 常数"""
    var, point, _, dir_sup = _get_limit_args(context)

    ratio, f = None, None

    if expr.is_Mul:
        # 处理 ln(1+f(x)) * h(x) 形式
        log_factor, other_factor = None, 1
        for arg in expr.args:
            if isinstance(arg, log):
                log_factor = arg
            else:
                other_factor *= arg
        if log_factor is None:
            return None
        f = log_factor.args[0] - 1
        ratio = simplify(f * other_factor)
    else:
        # 处理 ln(1+f(x)) / g(x) 形式
        numerator, denominator = expr.as_numer_denom()
        if not isinstance(numerator, log):
            return None
        f = numerator.args[0] - 1
        ratio = simplify(f / denominator)

    if f is None:
        return None

    f_l = latex(f)
    ratio_l = "" if ratio == 1 else latex(ratio)
    var_l, point_l, dir_sup_l = latex(var), latex(point), dir_sup
    lim_expr = f"\\lim_{{{var_l} \\to {point_l}{dir_sup_l}}}"

    result = Integer(1) if ratio == 1 else ratio

    if ratio == 1 and f == var:
        rule_text = f"{lim_expr} {latex(expr)} = 1"
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} {lim_expr} \\frac{{\\ln(1+{var_l})}}{{{var_l}}} = {latex(result)}"
        )
    else:
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} \\lim_{{t \\to 0{dir_sup}}} \\frac{{\\ln(1+t)}}{{t}} = {latex(result)}"
            f" \\quad \\text{{(令 }} t = {f_l} \\text{{)}}"
        )

    return result, f"重要极限: ${rule_text}$"


def sin_over_x_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 sin(f(x))/h(x) 或 sin(f(x))*h(x) 形式, f(x)/h(x) = 常数 或 f(x)*h(x) = 常数"""
    var, point, direction, _ = _get_limit_args(context)

    # 处理形式 sin(f(x))*h(x)
    if expr.is_Mul:
        sin_factor = None
        other_factor = 1
        for arg in expr.args:
            if isinstance(arg, sin):
                sin_factor = arg
            else:
                other_factor *= arg

        if sin_factor is not None:
            sin_arg = sin_factor.args[0]
            # 检查是否满足 sin(f(x))*g(x) 且 f(x)*g(x) = 常数
            product = sin_arg * other_factor
            if not product.has(var) and _check_function_tends_to_zero(sin_arg, var, point, direction):
                return 'sin_over_x'

    # 处理标准分式形式
    numerator, denominator = expr.as_numer_denom()
    if isinstance(numerator, sin):
        sin_arg = numerator.args[0]
        ratio = sin_arg / denominator
        if not ratio.has(var) and _check_function_tends_to_zero(sin_arg, var, point, direction):
            return 'sin_over_x'

    return None


def one_plus_one_over_x_pow_x_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 (1 + 1/f(x))**h(x)  或 (1 + f(x))**(1/h(x)) 形式, 其中 f(x) -> oo 或 f(x) -> 0 且 f(x)/h(x) = 常数"""
    var, point, direction, _ = _get_limit_args(context)
    if not isinstance(expr, Pow):
        return None
    base, _exp = expr.as_base_exp()
    # 统一为 (f(x) + 1)**h(x) 形式处理
    inv_f = base - 1
    ratio = simplify(inv_f * _exp)
    if _check_function_tends_to_zero(inv_f, var, point, direction) and not ratio.has(var):
        return 'one_plus_one_over_x_pow_x'

    return None


def ln_one_plus_x_over_x_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 ln(1+f(x))/g(x) 或 ln(1+f(x))*h(x) 形式，且 f(x)/g(x) 或 f(x)*h(x) = 常数"""
    var, point, direction, _ = _get_limit_args(context)

    if expr.is_Mul:
        # 乘法形式
        log_factor, other_factor = None, 1
        for arg in expr.args:
            if isinstance(arg, log):
                log_factor = arg
            else:
                other_factor *= arg
        if log_factor is not None:
            f = log_factor.args[0] - 1
            product = f * other_factor
            if not product.has(var) and _check_function_tends_to_zero(f, var, point, direction):
                return 'ln_one_plus_x_over_x'
    else:
        # 分式形式
        numerator, denominator = expr.as_numer_denom()
        if isinstance(numerator, log):
            f = numerator.args[0] - 1
            ratio = f / denominator
            if not ratio.has(var) and _check_function_tends_to_zero(f, var, point, direction):
                return 'ln_one_plus_x_over_x'

    return None


def exp_minus_one_over_x_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 (e^{f(x)} - 1)/g(x) 或 (e^{f(x)} - 1)*h(x) 形式，且 f(x)/g(x) 或 f(x)*h(x) = 常数"""
    var, point, direction, _ = _get_limit_args(context)

    if expr.is_Mul:
        # 乘法形式
        exp_factor, other_factor = None, 1
        for arg in expr.args:
            if arg.is_Add and arg.has(exp):
                exp_factor = arg
            else:
                other_factor *= arg
        if exp_factor is not None:
            exp_part = [a for a in exp_factor.args if a.has(exp)]
            if exp_part:
                try:
                    # 提取公共常数, 凑重要极限
                    f = exp_part[0].args[1].args[0]
                    if exp_part[0].args[0].has(var):
                        return None
                except:
                    f = exp_part[0].args[0]
                product = f * other_factor
                if not product.has(var) and _check_function_tends_to_zero(f, var, point, direction):
                    return 'exp_minus_one_over_x'
    else:
        # 分式形式
        numerator, denominator = expr.as_numer_denom()
        if numerator.is_Add and numerator.has(exp):
            exp_part = [a for a in numerator.args if a.has(exp)]
            if exp_part:
                f = exp_part[0].args[0]
                ratio = f / denominator
                if not ratio.has(var) and _check_function_tends_to_zero(f, var, point, direction):
                    return 'exp_minus_one_over_x'

    return None


def exp_minus_one_over_x_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """重要极限: (e^{f(x)} - 1)/g(x) 或 (e^{f(x)} - 1)*h(x)，其中 f(x)/g(x) 或 f(x)*h(x) -> 常数"""
    var, point, _, dir_sup = _get_limit_args(context)

    ratio, f = None, None

    if expr.is_Mul:
        # 处理 (e^{f(x)} - 1) * h(x)
        exp_factor, other_factor = None, 1
        for arg in expr.args:
            if arg.is_Add and arg.has(exp):
                exp_factor = arg
            else:
                other_factor *= arg
        if exp_factor is None:
            return None
        # 提取 e^{f(x)} - 1
        exp_part = [a for a in exp_factor.args if a.has(exp)]
        if not exp_part:
            return None
        try:
            # 提取公共常数, 凑重要极限
            f = exp_part[0].args[1].args[0]
            if exp_part[0].args[0].has(var):
                return None
            ratio = simplify(f * other_factor * exp_part[0].args[0])
        except:
            f = exp_part[0].args[0]
            ratio = simplify(f * other_factor)
    else:
        # 处理 (e^{f(x)} - 1) / g(x)
        numerator, denominator = expr.as_numer_denom()
        if not (numerator.is_Add and numerator.has(exp)):
            return None
        exp_part = [a for a in numerator.args if a.has(exp)]
        if not exp_part:
            return None
        f = exp_part[0].args[0]
        ratio = simplify(f / denominator)

    if f is None:
        return None

    f_l = latex(f)
    ratio_l = "" if ratio == 1 else latex(ratio)
    var_l, point_l, dir_sup_l = latex(var), latex(point), dir_sup
    lim_expr = f"\\lim_{{{var_l} \\to {point_l}{dir_sup_l}}}"

    result = Integer(1) if ratio == 1 else ratio

    if ratio == 1 and f == var:
        rule_text = f"{lim_expr} {latex(expr)} = 1"
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} {lim_expr} \\frac{{e^{{{var_l}}} - 1}}{{{var_l}}} = {latex(result)}"
        )
    else:
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} \\lim_{{t \\to 0{dir_sup}}} \\frac{{e^t - 1}}{{t}} = {latex(result)}"
            f" \\quad \\text{{(令 }} t = {f_l} \\text{{)}}"
        )

    return result, f"重要极限: ${rule_text}$"


# 倒数重要极限规则
def g_over_sin_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """重要极限: g(x)/sin(f(x))，其中 g(x)/f(x) -> 常数"""
    var, point, _, dir_sup = _get_limit_args(context)
    num, den = expr.as_numer_denom()
    if not isinstance(den, sin):
        return None
    f = den.args[0]
    ratio = simplify(num / f)

    var_l, point_l, f_l = latex(var), latex(point), latex(f)
    lim_expr = f"\\lim_{{{var_l} \\to {point_l}{dir_sup}}}"
    ratio_l = "" if ratio == 1 else latex(ratio)

    result = Integer(1) if ratio == 1 else ratio

    rule_text = ""
    if ratio != 1 and f != var:
        lim_expr_t = f"\\lim_{{t \\to {point_l}{dir_sup}}}"
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} {lim_expr_t} \\frac{{t}}{{\\sin(t)}} = "
            f"{latex(result)} "
            f"\\quad \\text{{(令 }} t = {f_l} \\text{{)}}"
        )
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} {lim_expr} \\frac{{{var_l}}}{{\\sin({var_l})}} = "
            f"{latex(result)}"
        )
    else:
        rule_text = f"{lim_expr} {latex(expr)} = 1"

    return result, f"重要极限: ${rule_text}$"


def g_over_sin_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    var, point, direction, _ = _get_limit_args(context)
    num, den = expr.as_numer_denom()
    if isinstance(den, sin):
        f = den.args[0]
        ratio = simplify(num / f)
        if not ratio.has(var) and _check_function_tends_to_zero(f, var, point, direction):
            return 'g_over_sin'
    return None


def g_over_ln_one_plus_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """重要极限: g(x)/ln(1+f(x))，其中 g(x)/f(x) -> 常数"""
    var, point, _, dir_sup = _get_limit_args(context)
    num, den = expr.as_numer_denom()
    if not isinstance(den, log):
        return None
    f = den.args[0] - 1
    ratio = simplify(num / f)

    var_l, point_l, f_l = latex(var), latex(point), latex(f)
    lim_expr = f"\\lim_{{x \\to {point_l}{dir_sup}}}"
    ratio_l = "" if ratio == 1 else latex(ratio)

    result = Integer(1) if ratio == 1 else ratio

    rule_text = ''
    if ratio != 1 and f != var:
        lim_expr = f"\\lim_{{t \\to {point_l}{dir_sup}}}"
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} {lim_expr} \\frac{{t}}{{\\ln(1+t)}} = "
            f"{latex(result)} "
            f"\\quad \\text{{(令 }} t = {f_l} \\text{{)}}"
        )
    elif ratio != 1 and f == var:
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} {lim_expr} \\frac{{{var_l}}}{{\\ln(1+{var_l})}} = "
            f"{latex(result)}"
        )
    else:
        rule_text = f"{lim_expr} {latex(expr)} = 1"

    return result, f"重要极限: ${rule_text}$"


def g_over_ln_one_plus_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    var, point, direction, _ = _get_limit_args(context)
    num, den = expr.as_numer_denom()
    if isinstance(den, log):
        f = den.args[0] - 1
        ratio = simplify(num / f)
        if not ratio.has(var) and _check_function_tends_to_zero(f, var, point, direction):
            return 'g_over_ln_one_plus'
    return None


def g_over_exp_minus_one_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 g(x)/(e^{f(x)} - 1) 形式，且 g(x)/f(x) = 常数"""
    var, point, direction, _ = _get_limit_args(context)
    num, den = expr.as_numer_denom()

    if not (den.is_Add and den.has(exp)):
        return None

    # 提取 e^{f(x)} 项
    exp_part = [a for a in den.args if a.has(exp)]
    if not exp_part:
        return None

    try:
        # 提取公共常数, 凑重要极限
        f = exp_part[0].args[1].args[0]
    except:
        f = exp_part[0].args[0]
    ratio = simplify(num / f)

    if not ratio.has(var) and _check_function_tends_to_zero(f, var, point, direction):
        return "g_over_exp_minus_one"

    return None


def g_over_exp_minus_one_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """重要极限: g(x)/(e^{f(x)} - 1)，其中 g(x)/f(x) -> 常数"""
    var, point, _, dir_sup = _get_limit_args(context)
    num, den = expr.as_numer_denom()

    if not (den.is_Add and den.has(exp)):
        return None

    exp_part = [a for a in den.args if a.has(exp)]
    if not exp_part:
        return None

    try:
        # 提取公共常数, 凑重要极限
        f = exp_part[0].args[1].args[0]
        ratio = simplify(num / f * 1/exp_part[0].args[0])
    except:
        f = exp_part[0].args[0]
        ratio = simplify(num / f)

    f_l = latex(f)
    ratio_l = "" if ratio == 1 else latex(ratio)
    var_l, point_l, dir_sup_l = latex(var), latex(point), dir_sup
    lim_expr = f"\\lim_{{{var_l} \\to {point_l}{dir_sup_l}}}"

    result = Integer(1) if ratio == 1 else ratio

    # 规则文本
    if ratio == 1 and f == var:
        # 最简单的情况
        rule_text = f"{lim_expr} {latex(expr)} = 1"
    elif ratio != 1 and f == var:
        # f(x) = x 的情况
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} {lim_expr} \\frac{{{var_l}}}{{e^{{{var_l}}} - 1}} = {latex(result)}"
        )
    else:
        # 一般换元情况
        rule_text = (
            f"{lim_expr} {latex(expr)} = "
            f"{ratio_l} \\lim_{{t \\to 0{dir_sup}}} \\frac{{t}}{{e^t - 1}} = {latex(result)}"
            f" \\quad \\text{{(令 }} t = {f_l} \\text{{)}}"
        )

    return result, f"重要极限: ${rule_text}$"
