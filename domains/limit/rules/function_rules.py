from typing import Any, Dict, Tuple

from sympy import (
    Expr, Limit, Pow, acos, asin, atan, cos, cosh, cot, csc, exp,
    latex, log, sec, sin, sinh, tan, tanh, oo
)


def _preprocess_expression(expr: Expr) -> Expr:
    """ln(1/f(x)) = -ln(f(x))"""
    expr = expr.replace(
        lambda e: e.is_Function and e.func == log and len(
            e.args) == 1 and e.args[0].is_Pow and e.args[0].args[1] == -1,
        lambda e: -log(e.args[0].args[0])
    )
    return expr


def pow_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"
    base, exponent = expr.args

    # 检查底数和指数是否含变量
    base_has_var = base.has(var)
    exp_has_var = exponent.has(var)

    # Case 1: 底数是常数 -> a^v => a^(lim v)
    if not base_has_var:
        new_expr = base ** Limit(exponent, var, point, dir=direction)
        rule_desc = (
            f"应用常数底数幂规则: "
            f"$\\lim_{{{var} \\to {target}}} \\left({latex(base)}\\right)^{{{latex(exponent)}}} = "
            f"{latex(base)}^{{\\lim_{{{var} \\to {target}}} {latex(exponent)}}}$"
        )
        return new_expr, rule_desc

    # Case 2: 指数是常数 -> u^b => (lim u)^b
    if not exp_has_var:
        new_expr = Limit(base, var, point, dir=direction) ** exponent
        rule_desc = (
            f"应用常数指数幂规则: "
            f"$\\lim_{{{var} \\to {target}}} \\left({latex(base)}\\right)^{{{latex(exponent)}}} = "
            f"\\left(\\lim_{{{var} \\to {target}}} {latex(base)}\\right)^{{{latex(exponent)}}}$"
        )
        return new_expr, rule_desc

    # Case 3: 底数和指数都含变量 -> 使用 exp-log 转换
    # 此时需要确保 base > 0 在极限点附近(-左附近/+右附近)成立
    try:
        # 计算底数在极限点的极限（用于初步筛选）
        base_limit = Limit(base, var, point, dir=direction)
        # 初步筛选底数极限的符号（排除明显非法情况）
        # 若底数极限为负实数（非无穷），则邻域内可能存在负数，直接拒绝（除非指数是奇数整数，但复杂场景暂不处理）
        if base_limit.is_real and base_limit < 0:
            return None
        # 若底数极限为正无穷，允许（ln(+oo) 有定义）
        if base_limit == oo:
            pass  # 允许，后续邻域验证会处理
        # 若底数极限为负无穷，拒绝（ln(-oo) 无定义）
        elif base_limit == -oo:
            return None
        # 关键验证 - 邻域内底数是否严格为正（根据极限方向选择邻近点）
        # 定义邻域步长（小量，避免数值误差）
        epsilon = 1e-8
        # 根据方向选择邻近点（左极限取 point - epsilon，右极限取 point + epsilon）
        if direction == '-':
            near_point = point - epsilon
        else:
            near_point = point + epsilon
        # 计算底数在邻近点的值（数值计算）
        base_near = base.subs(var, near_point).evalf()
        # 若底数在邻近点 <=0，拒绝转换（对数无定义）
        if base_near <= 0:
            return None

        # 所有检查通过，可以安全应用 exp-log 变换
        log_base = log(base)
        log_base = _preprocess_expression(log_base)
        exp_argument = exponent * log_base
        new_limit_inside = Limit(exp_argument, var, point, dir=direction)
        new_expr = exp(new_limit_inside)

        base_latex = latex(base)
        exp_latex = latex(exponent)
        var_latex = latex(var)

        rule_desc = (
            f"应用指数对数变换: "
            f"$\\lim_{{{var_latex} \\to {target}}} \\left(\\left({base_latex}\\right)^{{{exp_latex}}}\\right) = "
            f"\\lim_{{{var_latex} \\to {target}}} e^{{({exp_latex}) \\cdot \\ln {base_latex}}} = "
            f"e^{{\\lim_{{{var_latex} \\to {target}}} \\left(({exp_latex}) \\cdot \\ln {base_latex}\\right)}}$"
        )

        return new_expr, rule_desc
    except Exception as e:
        return None


def exp_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    if expr.func == exp:
        new_expr = exp(Limit(arg, var, point, dir=direction))
        return new_expr, (
            f"应用指数函数规则: "
            f"$\\lim_{{{var} \\to {target}}} e^{{{latex(arg)}}} = "
            f"e^{{\\lim_{{{var} \\to {target}}} {latex(arg)}}}$"
        )
    return None


def log_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = log(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用对数函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\ln\\left({latex(arg)}\\right) = "
        f"\\ln\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def sin_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = sin(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用正弦函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\sin\\left({latex(arg)}\\right) = "
        f"\\sin\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right) \\right)$"
    )


def cos_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = cos(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用余弦函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\cos\\left({latex(arg)}\\right) = "
        f"\\cos\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def tan_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = tan(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用正切函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\tan\\left({latex(arg)}\\right) = "
        f"\\tan\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def sec_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = sec(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用正割函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\sec\\left({latex(arg)}\\right) = "
        f"\\sec\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def csc_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = csc(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用余割函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\csc\\left({latex(arg)}\\right) = "
        f"\\csc\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def cot_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = cot(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用余切函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\cot\\left({latex(arg)}\\right) = "
        f"\\cot\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def asin_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = asin(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用反正弦函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\arcsin\\left({latex(arg)}\\right) = "
        f"\\arcsin\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def acos_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = acos(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用反余弦函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\arccos\\left({latex(arg)}\\right) = "
        f"\\arccos\\left(\\lim_{{{var} \\to {target}}} {latex(arg)}\\right)$"
    )


def atan_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = atan(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用反正切函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\arctan\\left({latex(arg)}\\right) = "
        f"\\arctan\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def sinh_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = sinh(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用双曲正弦函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\sinh\\left({latex(arg)}\\right) = "
        f"\\sinh\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def cosh_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = cosh(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用双曲余弦函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\cosh\\left({latex(arg)}\\right) = "
        f"\\cosh\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


def tanh_rule(expr: Expr, context: Dict[str, Any]):
    var = context['variable']
    point = context['point']
    direction = context.get('direction', '+')
    target = f"{{{latex(point)}}}{'^+' if direction == '+' else '^-'}"

    arg = expr.args[0]

    new_expr = tanh(Limit(arg, var, point, dir=direction))
    return new_expr, (
        f"应用双曲正切函数规则: "
        f"$\\lim_{{{var} \\to {target}}} \\tanh\\left({latex(arg)}\\right) = "
        f"\\tanh\\left(\\lim_{{{var} \\to {target}}} \\left({latex(arg)}\\right)\\right)$"
    )


# 匹配器函数
def pow_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, Pow):
        return 'pow'
    return None


def exp_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, exp):
        return 'exp'
    return None


def log_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, log):
        return 'log'
    return None


def sin_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, sin):
        return 'sin'
    return None


def cos_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, cos):
        return 'cos'
    return None


def tan_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, tan):
        return 'tan'
    return None


def sec_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, sec):
        return 'sec'
    return None


def csc_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, csc):
        return 'csc'
    return None


def cot_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, cot):
        return 'cot'
    return None


def asin_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, asin):
        return 'asin'
    return None


def acos_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, acos):
        return 'acos'
    return None


def atan_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, atan):
        return 'atan'
    return None


def sinh_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, sinh):
        return 'sinh'
    return None


def cosh_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, cosh):
        return 'cosh'
    return None


def tanh_matcher(expr: Expr, _context: Dict[str, Any]):
    if isinstance(expr, tanh):
        return 'tanh'
    return None
