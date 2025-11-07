# TODO f-string 里都是单 $ 包裹

from typing import Any, Dict, Optional, Tuple
from sympy import (
    Expr, Limit, Pow, Symbol, diff, latex, oo, series, together, simplify, S, log, Mul, Pow, exp
)


def _limit_or_series(expr: Expr, var: Symbol, point: Any, direction: str) -> Any:
    """尝试求极限，失败则用级数首项近似"""
    try:
        return Limit(expr, var, point, dir=direction).doit()
    except:
        try:
            s = series(expr, var, point, n=1).removeO()
            return s if not s.has(S.Infinity, -oo, S.NaN) else oo
        except:
            return None


def _is_infinite(val: Any) -> bool:
    """判断值是否为无穷大"""
    return val in [oo, -oo]


def _is_zero(val: Any) -> bool:
    """判断值是否为零（含符号零）"""
    return val == 0 or (hasattr(val, 'is_zero') and val.is_zero)


def _extract_num_den(expr: Expr) -> Tuple[Expr, Expr]:
    """提取分子分母"""
    # 使用 together 确保表达式为有理式形式
    combined = together(expr)
    if hasattr(combined, 'as_numer_denom'):
        num, den = combined.as_numer_denom()
        return num, den


def _get_type(a: Any, b: Any) -> str:
    """根据分子分母极限值判断不定型类型"""
    if _is_zero(a) and _is_zero(b):
        return r'0/0'
    if _is_infinite(a) and _is_infinite(b):
        return r"\infty/\infty"
    return "no_match"


def _get_indeterminate_type(numerator: Expr, denominator: Expr, var: Symbol, point: Any, direction: str) -> str:
    """判断分式是否为 0/0 或 oo/oo 型"""
    if point in [oo, -oo]:
        t = Symbol('t', positive=True)
        dir_t = '+' if point == oo else '-'
        num_t = numerator.subs(var, 1/t)
        den_t = denominator.subs(var, 1/t)
        a = _limit_or_series(num_t, t, 0, dir_t)
        b = _limit_or_series(den_t, t, 0, dir_t)
    else:
        a = _limit_or_series(numerator, var, point, direction)
        b = _limit_or_series(denominator, var, point, direction)
    return _get_type(a, b)


def _choose_best_conversion(f: Expr, g: Expr, var: Symbol, point: Any, direction: str) -> str:
    """
    选择最优的转换方式：0/0 或 oo/oo

    选择策略：
    1. 优先选择导数更简单的形式
    2. 避免产生更复杂的表达式
    3. 考虑后续求导的难度
    """
    # 计算两种转换方式的导数复杂度
    zero_zero_complexity = _estimate_derivative_complexity(f, 1/g, var)
    inf_inf_complexity = _estimate_derivative_complexity(g, 1/f, var)
    # 选择复杂度较低的形式
    if zero_zero_complexity <= inf_inf_complexity:
        return "zero_over_zero"
    else:
        return "inf_over_inf"


def _estimate_derivative_complexity(numerator: Expr, denominator: Expr, var: Symbol) -> int:
    """
    估计求导后的表达式复杂度

    返回一个整数值表示复杂度（值越小越简单）
    """
    # 计算分子和分母的导数
    num_diff = diff(numerator, var)
    den_diff = diff(denominator, var)
    result = simplify(num_diff / den_diff)

    # 使用多种指标估计复杂度
    complexity = 0

    # 1. 表达式长度（节点数）
    complexity += _count_nodes(result)

    # 2. 特殊函数的数量（如三角函数、指数函数等）
    complexity += 2 * _count_special_functions(result)

    # 3. 分式结构的惩罚（如果导数产生分式，增加复杂度）
    if _has_fraction(result):
        complexity += 5

    # 4. 乘积结构的惩罚
    complexity += _count_products(result)

    # 5. 幂结构的惩罚
    complexity += 3 * _count_powers(result)

    return complexity


def _count_nodes(expr: Expr) -> int:
    """计算表达式的节点数"""
    return len(expr.args) + sum(_count_nodes(arg) for arg in expr.args) if expr.args else 1


def _count_special_functions(expr: Expr) -> int:
    """计算特殊函数的数量"""
    if expr.is_Function:
        func_name = str(expr.func)
        special_funcs = {'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                         'asin', 'acos', 'atan', 'acot',
                         'sinh', 'cosh', 'tanh', 'coth',
                         'exp', 'log', 'ln'}
        return 1 if any(name in func_name.lower() for name in special_funcs) else 0
    return sum(_count_special_functions(arg) for arg in expr.args) if expr.args else 0


def _has_fraction(expr: Expr) -> bool:
    """检查表达式是否包含分式"""
    if expr.is_Pow and expr.exp.is_negative:
        return True
    if expr.is_Mul:
        return any(_has_fraction(arg) for arg in expr.args)
    return any(_has_fraction(arg) for arg in expr.args) if expr.args else False


def _count_products(expr: Expr) -> int:
    """计算乘积结构的数量"""
    if expr.is_Mul:
        return len(expr.args) - 1 + sum(_count_products(arg) for arg in expr.args)
    return sum(_count_products(arg) for arg in expr.args) if expr.args else 0


def _count_powers(expr: Expr) -> int:
    """计算幂运算的数量及复杂度"""
    if expr.is_Pow:
        base, _exp = expr.args
        # 如果底数或指数含变量，复杂度更高
        if base.has(Symbol) or _exp.has(Symbol):
            return 2 + _count_powers(base) + _count_powers(_exp)
        return 1 + _count_powers(base) + _count_powers(_exp)
    return sum(_count_powers(arg) for arg in expr.args) if expr.args else 0


def lhopital_direct_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')
    num, den = _extract_num_den(expr)
    if den == 1:
        return None
    typ = _get_indeterminate_type(num, den, var, point, direction)
    if typ not in ["0/0", r"\infty/\infty"]:
        return None
    try:
        num_deriv = diff(num, var)
        den_deriv = diff(den, var)
        new_expr = simplify(num_deriv/den_deriv)
        typ = r'\frac{0}{0}' if type == '0/0' else r'\frac{\infty}{\infty}'
        explanation = (
            f"原式为 ${typ}$ 型不定式，应用洛必达法则："
            f"对分子 ${latex(num)}$ 和分母 ${latex(den)}$ 关于 ${latex(var)}$ 分别求导，得到：\n\n"
            f"$\n\\frac{{d}}{{d{latex(var)}}} \\left( {latex(num)} \\right) = {latex(num_deriv)},\\quad"
            f"\\frac{{d}}{{d{latex(var)}}} \\left( {latex(den)} \\right) = {latex(den_deriv)}\n$\n\n"
            f"因此极限转化为：\n$\n\\lim_{{{latex(var)} \\to {latex(point)}^{{{direction}}}}} {latex(new_expr)}\n$"
        )
        return Limit(new_expr, var, point, dir=direction), explanation
    except Exception as e:
        print(e)
        return None


def lhopital_zero_times_inf_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """处理 0*oo 情况, 动态选择转换为 0/0 或 oo/oo 形式，并应用洛必达法则"""
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')

    # 提取两个因子
    f, g = expr.args

    # 确定哪个因子趋于 0，哪个趋于 oo
    lim_f = _limit_or_series(f, var, point, direction)
    lim_g = _limit_or_series(g, var, point, direction)

    if not ((_is_zero(lim_f) and _is_infinite(lim_g)) or
            (_is_infinite(lim_f) and _is_zero(lim_g))):
        return None

    # 动态选择最优转换方式
    conversion_type = _choose_best_conversion(f, g, var, point, direction)

    if conversion_type == "zero_over_zero":
        # 转换为 f/(1/g) = 0/0 形式
        numerator = f
        denominator = 1/g
        conversion_explanation = (
            f"原式为 $0 \\cdot \\infty$ 型不定式，转换为 $\\frac{{0}}{{0}}$ 型：\n"
            f"${latex(expr)} = \\frac{{{latex(f)}}}{{{latex(1/g)}}}$\n\n"
        )
    else:  # conversion_type == "inf_over_inf"
        # 转换为 g/(1/f) = oo/oo 形式
        numerator = g
        denominator = 1/f
        conversion_explanation = (
            f"原式为 $0 \\cdot \\infty$ 型不定式，转换为 $\\frac{{\\infty}}{{\\infty}}$ 型：\n"
            f"${latex(expr)} = \\frac{{{latex(g)}}}{{{latex(1/f)}}}$\n\n"
        )

    # 对分子和分母分别求导
    numerator_diff = diff(numerator, var)
    denominator_diff = diff(denominator, var)

    # 构建求导后的极限表达式
    diff_expr = numerator_diff / denominator_diff
    diff_limit = Limit(diff_expr, var, point, direction)

    # 添加求导步骤的说明
    explanation = conversion_explanation + (
        f"应用洛必达法则，分子分母分别求导：\n"
        f"$\\frac{{d}}{{d{var}}} ({latex(numerator)}) = {latex(numerator_diff)}$\\quad"
        f"$\\frac{{d}}{{d{var}}} ({latex(denominator)}) = {latex(denominator_diff)}$\\quad"
        f"得到新极限：$\\lim_{{{var} \\to {latex(point)}}} ({latex(diff_expr)})$"
    )

    return diff_limit, explanation


def lhopital_inf_minus_inf_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """处理 oo - oo 情况, 转换为 0/0 或 oo/oo 形式"""
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')

    if not expr.is_Add or len(expr.args) != 2:
        return None

    f, g = expr.args
    lim_f = _limit_or_series(f, var, point, direction)
    lim_g = _limit_or_series(g, var, point, direction)
    if not (_is_infinite(lim_f) and _is_infinite(lim_g)):
        return None

    # 转换策略：f - g = (1/g - 1/f) / (1/(f*g)), 转换为 0/0 型
    # 注意：这个转换要求 f 和 g 在极限点附近不为零（通常成立, 因为趋于无穷）
    try:
        numerator = 1/g - 1/f
        denominator = 1/(f * g)

        conversion_explanation = (
            f"原式为 $\\infty - \\infty$ 型不定式，通过代数变形转换为 $\\frac{{0}}{{0}}$ 或 $\\frac{{\\infty}}{{\\infty}}$ 型：\n"
            f"${latex(expr)} = \\frac{{{latex(numerator)}}}{{{latex(denominator)}}}$\n\n"
        )

        # 对分子分母分别求导
        numerator_diff = diff(numerator, var)
        denominator_diff = diff(denominator, var)

        # 构建求导后的极限表达式
        diff_expr = numerator_diff / denominator_diff
        diff_limit = Limit(diff_expr, var, point, direction)
        explanation = conversion_explanation + (
            f"应用洛必达法则，分子分母分别求导：\n"
            f"$\\frac{{d}}{{d{var}}} \\left({latex(numerator)}\\right) = {latex(numerator_diff)}$\\quad"
            f"$\\frac{{d}}{{d{var}}} \\left({latex(denominator)}\\right) = {latex(denominator_diff)}$\\quad"
            f"得到新极限：$\\lim_{{{var} \\to {latex(point)}}} \\left({latex(diff_expr)}\\right)$"
        )

        return diff_limit, explanation

    except Exception:
        return None


def lhopital_power_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')
    base, exp_arg = expr.args

    lim_base = _limit_or_series(base, var, point, direction)
    lim_exp = _limit_or_series(exp_arg, var, point, direction)

    # 根据极限值确定具体类型
    if _is_zero(lim_base) and _is_zero(lim_exp):
        typ = "0^0"
    elif _is_infinite(lim_base) and _is_zero(lim_exp):
        typ = r"\infty^0"
    else:  # 1**oo 型
        typ = "1^\\infty"

    # 使用上 e 等价变换：f(x)^g(x) = e^(g(x)*ln(f(x)))
    transformed_expr = exp_arg * log(base)
    exp_expr = exp(transformed_expr)  # e^(g(x)*ln(f(x)))
    limit_exp_expr = Limit(exp_expr, var, point, direction)

    explanation = (
        f"原式为 ${typ}$ 型不定式，使用指数变换：\n"
        f"${latex(expr)} = e^{{{latex(transformed_expr)}}}$\n\n"
        f"因此，$\\lim_{{{latex(var)} \\to {latex(point)}^{{{direction}}}}} {latex(expr)} "
        f"= {{{latex(limit_exp_expr)}}}$\n\n"
    )
    return limit_exp_expr, explanation


def lhopital_direct_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """直接分式形式 (0/0 ,oo/oo)"""
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')
    num, den = _extract_num_den(expr)
    if num == expr and den == 1:
        return None
    typ = _get_indeterminate_type(num, den, var, point, direction)
    return 'lhopital_direct' if typ in ["0/0", r"\infty/\infty"] else None


def lhopital_zero_times_inf_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """检查表达式是否为 0*oo 不定型"""
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')

    # 只处理两个项的乘法
    if not isinstance(expr, Mul) or len(expr.args) != 2:
        return None
    f, g = expr.args

    # 检查两个因子的极限
    lim_f = _limit_or_series(f, var, point, direction)
    lim_g = _limit_or_series(g, var, point, direction)
    # 检查是否为 0*oo 形式
    if (_is_zero(lim_f) and _is_infinite(lim_g)) or (_is_infinite(lim_f) and _is_zero(lim_g)):
        return 'lhopital_zero_times_inf'

    return None


def lhopital_inf_minus_inf_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """oo-oo型"""
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')
    if not expr.is_Add or len(expr.args) != 2:
        return None
    f, g = expr.args
    lim_f = _limit_or_series(f, var, point, direction)
    lim_g = _limit_or_series(g, var, point, direction)
    return 'lhopital_inf_minus_inf' if _is_infinite(lim_f) and _is_infinite(lim_g) else None


def lhopital_power_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')
    if not isinstance(expr, Pow):
        return None

    base, exp = expr.args
    lim_base = _limit_or_series(base, var, point, direction)
    lim_exp = _limit_or_series(exp, var, point, direction)
    if lim_base is None or lim_exp is None:
        return None
    # 处理符号无穷和符号零
    if lim_base == S.Zero or (_is_zero(lim_base) and not _is_infinite(lim_base)):
        base_zero = True
    else:
        base_zero = False
    if lim_exp == S.Zero or (_is_zero(lim_exp) and not _is_infinite(lim_exp)):
        exp_zero = True
    else:
        exp_zero = False
    base_inf = _is_infinite(lim_base)
    exp_inf = _is_infinite(lim_exp)
    # 检查是否为 1 (考虑符号计算的精度)
    try:
        if hasattr(lim_base, 'evalf'):
            base_val = lim_base.evalf()
            near_one = abs(base_val - 1) < 1e-10
        else:
            near_one = abs(lim_base - 1) < 1e-10
    except:
        near_one = False
    # 0**0 型
    if base_zero and exp_zero:
        return 'lhopital_power'
    # oo**0 型
    if base_inf and exp_zero:
        return 'lhopital_power'
    # 1**oo 型
    if near_one and exp_inf:
        return 'lhopital_power'
    return None
