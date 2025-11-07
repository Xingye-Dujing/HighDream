from typing import Optional, Tuple, Dict, Any, List
from sympy import Symbol, series, limit, oo, exp, log, Pow, Add, Mul, S, Expr, latex, preorder_traversal, Function, Wild, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, E, I, pi, factorial, sqrt, symbols, simplify, expand, fraction, Poly


def is_power_minus_one_form(expr: Expr, var: Symbol) -> Optional[Tuple[Expr, Expr, Expr]]:
    """
    检查表达式是否是 x^n * (f(x)^g(x) - 1) 形式
    返回 (多项式部分, 幂指函数部分, 底数, 指数) 或 None
    """
    # 检查是否是乘法形式
    if not expr.is_Mul:
        # 检查是否是 (f(x)^g(x) - 1) 形式, n==0  时
        if expr.is_Add and len(expr.args) == 2:
            has_minus_one = False
            power_expr = None
            for term in expr.args:
                if term == -1 or term.is_Number and term == -1:
                    has_minus_one = True
                elif term.is_Pow and term.base.has(var) and term.exp.has(var):
                    power_expr = term
            if has_minus_one and power_expr is not None:
                base, exponent = power_expr.args
                return S.One, power_expr, base, exponent
        return None

    # 寻找多项式部分和幂指函数减一部分
    poly_part = None
    power_minus_one = None
    power_expr = None

    for arg in expr.args:
        # 检查是否是多项式部分 (x^n 形式)
        if arg.is_Pow and arg.base == var and (arg.exp.is_Integer or arg.exp.is_Number) and arg.exp > 0:
            poly_part = arg
        # 检查是否是 (f(x)^g(x) - 1) 形式
        elif arg.is_Add and len(arg.args) == 2:
            has_minus_one = False
            for term in arg.args:
                if term == -1 or term.is_Number and term == -1:
                    has_minus_one = True
                elif term.is_Pow and term.base.has(var) and term.exp.has(var):
                    power_expr = term
            if has_minus_one and power_expr is not None:
                power_minus_one = arg
        # 检查是否是常数因子
        elif arg.is_Number:
            if poly_part is None:
                poly_part = arg
            else:
                poly_part = poly_part * arg

    if poly_part is None or power_minus_one is None:
        return None

    # 验证表达式结构
    if expr != poly_part * power_minus_one:
        return None

    # 提取幂指函数的底数和指数
    base, exponent = power_expr.args

    return poly_part, power_expr, base, exponent


def find_taylor_candidates(expr: Expr, var: Symbol, point: Any) -> List[Tuple[Expr, Expr, Expr, Expr]]:
    """在表达式中查找所有可能的泰勒展开候选"""
    candidates = []

    # 遍历所有子表达式
    # preorder_traversal(expr) 会按前序遍历的顺序生成 expr 中的所有子表达式（包括 expr 自身）
    for subexpr in preorder_traversal(expr):
        # 检查是否是 x^n * (f(x)^g(x) - 1) 形式
        result = is_power_minus_one_form(subexpr, var)
        if result:
            poly_part, power_expr, base, exponent = result
            candidates.append((subexpr, poly_part, base, exponent))

    return candidates


def get_expression_order(expr: Expr, var: Symbol) -> int:
    """
    确定表达式的主导项阶数
    使用级数展开的方法
    """
    try:
        # 尝试展开为级数
        series_expr = series(expr, var, 0, 6).removeO()

        if series_expr == 0:
            return 0

        # 如果是加法表达式，找到最低次项
        if series_expr.is_Add:
            min_order = float('inf')
            for term in series_expr.args:
                term_order = get_simple_term_order(term, var)
                min_order = min(min_order, term_order)
            return min_order if min_order != float('inf') else 0
        else:
            return get_simple_term_order(series_expr, var)

    except:
        # 如果无法确定，返回默认值
        return 0


def get_simple_term_order(term: Expr, var: Symbol) -> int:
    """
    简单获取单项式的阶数
    """
    if term.is_constant():
        return 0
    elif term == var:
        return 1
    elif term.is_Pow and term.base == var:
        return term.exp
    elif term.is_Mul:
        order = 0
        for factor in term.args:
            if factor == var:
                order += 1
            elif factor.is_Pow and factor.base == var:
                order += factor.exp
            elif factor.has(var):
                # 递归处理
                order += get_simple_term_order(factor, var)
        return order
    else:
        # 尝试使用 as_coeff_exponent
        try:
            coeff, exp = term.as_coeff_exponent(var)
            return exp
        except:
            return 0


def taylor_expansion_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """
    使用泰勒展开求解极限
    适用于表达式中的任意部分
    动态决定展开阶数
    """
    var, point, direction, step_gen = context['variable'], context['point'], context.get(
        'direction', '+'), context.get('step_gen')

    # 只处理 x->oo 的情况
    if point not in [oo, -oo]:
        return None
    # 查找所有可能的泰勒展开候选
    candidates = find_taylor_candidates(expr, var, point)
    if not candidates:
        return None
    # 选择第一个候选进行处理
    candidate, poly_part, base, exponent = candidates[0]
    # 变量代换 t = 1/x
    t = Symbol('t', positive=True)
    # 将底数和指数表示为 t 的函数
    base_t = simplify(base.subs(var, 1/t))
    exponent_t = simplify(exponent.subs(var, 1/t))
    # 检查 base_t 是否趋近于 1
    base_limit = limit(base_t, t, 0, '+')
    if base_limit != 1:
        return None
    # 动态确定所需的展开阶数
    # 分析多项式部分的阶数
    poly_t = poly_part.subs(var, 1/t)
    # 确定多项式的主导项阶数
    poly_order = get_expression_order(poly_t, t)
    # 对于指数部分，我们需要展开到足够高的阶数以确保准确性
    # 通常需要展开到多项式阶数 + 2 阶
    required_order = max(4, abs(poly_order) +
                         2) if poly_order < 0 else max(4, poly_order + 2)
    # 计算 ln(base_t) 的泰勒展开
    try:
        ln_base_series = series(log(base_t), t, 0, required_order + 1)
    except:
        return None
    # 计算 exponent_t * ln_base_series 的泰勒展开
    exponent_ln_series = (exponent_t * ln_base_series).expand()
    # 计算 e**(exponent_ln_series) 的泰勒展开
    try:
        exp_series = series(exp(exponent_ln_series), t, 0, required_order + 2)
    except:
        return None
    # 计算幂指函数减一的泰勒展开
    power_minus_one_series = exp_series - 1
    # 计算整个表达式的泰勒展开
    full_series = (poly_t * power_minus_one_series).simplify()
    # 取极限
    try:
        result = limit(full_series, t, 0, '+')
    except:
        return None

    sub_candidate_latex = latex(simplify(candidate.subs(var, 1/t)))
    explanation = (
        f"令 $t = \\frac{{1}}{{{latex(var)}}}$，则当 $x \\to {latex(point)}$ 时 $t \\to 0^+$,"
        f"代入得：${latex(candidate)} = {sub_candidate_latex}.$"

        f"将 ${latex(base_t**exponent_t)}$ 化为指数形式：$e^{{{latex(exponent_t)} \\cdot \\ln({latex(base_t)})}}$."
        f"由于 $\\lim_{{t \\to 0^+}} {latex(base_t)} = 1$, 可展开:"
        # 显示主要项，省略高阶项
        f"$\\ln({latex(base_t)}) = {latex(ln_base_series.removeO())} + O(t^{required_order+1})$,"

        # 显示主要项
        f"指数部分：${latex(exponent_t)} \\cdot \\ln({latex(base_t)}) = {latex(exponent_ln_series.removeO())} + O(t^{required_order+1})$,"
        # 显示主要项
        f"指数展开：$e^{{{latex(exponent_ln_series.removeO())}}} = {latex(exp_series.removeO())} + O(t^{required_order+2})$,\n"

        f"故 ${latex(base**exponent)} = {latex(base_t**exponent_t)} = e^{{{latex(exponent_t)} \\cdot \\ln({latex(base_t)})}}$"
        # 显示主要项
        f" = ${latex(exp_series.removeO())} + O(t^{required_order+2})$."

        f"则有 ${latex(candidate)} = {sub_candidate_latex} = {latex(full_series.removeO())} + O(t^{required_order+abs(poly_order)+1})$.\n"  # 显示主要项
        f"取极限得：$\\lim_{{t \\to 0^+}} {latex(full_series)} = {latex(result)}$"
        f" (展开到 {required_order} 阶以确保精度)"
    )

    # 用结果替换原表达式中的候选部分
    new_expr = expr.subs(candidate, result)
    return new_expr, explanation


def equivalent_infinitesimal_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """
    使用等价无穷小求解极限
    适用于表达式中的任意部分
    """
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')

    # 只处理 x->oo的情况
    if point not in [oo, -oo]:
        return None

    # 查找所有可能的等价无穷小候选
    candidates = find_taylor_candidates(expr, var, point)
    if not candidates:
        return None

    # 选择第一个候选进行处理
    candidate, poly_part, base, exponent = candidates[0]

    # 变量代换 t = 1/x
    t = Symbol('t', positive=True)

    # 根据极限方向确定 t 的趋近方向
    if point == oo:
        t_direction = '+'
    else:  # point == -oo
        t_direction = '-'

    # 将底数和指数表示为 t 的函数
    base_t = base.subs(var, 1/t)
    exponent_t = exponent.subs(var, 1/t)

    # 检查等价无穷小的使用条件
    try:
        # 条件1: base_t → 1
        base_limit = limit(base_t, t, 0, t_direction)
        if base_limit != 1:
            return None

        # 条件2: exponent_t * (base_t - 1) → 0
        # 这是更直接的条件，确保等价无穷小适用
        equiv_condition = limit(exponent_t * (base_t - 1), t, 0, t_direction)
        if equiv_condition == oo or equiv_condition == -oo:
            return None

    except:
        return None

    # 判断是否是 f(x)^g(x) - 1 形式
    is_minus_one_form = False
    if candidate.is_Add:
        # 检查是否包含 -1 项
        for term in candidate.args:
            if term == -1 or term.is_Number and term == -1:
                is_minus_one_form = True
                break

    # 使用等价无穷小：
    # f(x)^g(x) - 1 = e^(g(x)ln(f(x))) - 1 ~ g(x)ln(f(x)) ~ g(x)(f(x) - 1)
    # 因为 ln(f(x)) ~ f(x) - 1 当 f(x) → 1

    if is_minus_one_form:
        # f(x)^g(x) - 1 ~ g(x)(f(x) - 1)
        power_minus_one_equiv = exponent_t * (base_t - 1)
    else:
        # 如果是 f(x)^g(x) 形式，需要先转换为 f(x)^g(x) - 1 + 1
        # 但通常我们处理的是 f(x)^g(x) - 1 的形式
        return None

    # 计算整个表达式的等价无穷小
    poly_t = poly_part.subs(var, 1/t)
    full_equiv = (poly_t * power_minus_one_equiv).simplify()

    # 取极限
    try:
        result = limit(full_equiv, t, 0, t_direction)
    except:
        return None

    # 生成解释
    candidate_latex = latex(candidate)

    # 使用变量来避免 f-string 中的反斜杠
    infinity_symbol = r'\infty' if point == oo else r'-\infty'
    t_arrow = f"t \\to 0^{t_direction}"

    explanation_lines = [
        f"在表达式 ${latex(expr)}$ 中，子表达式 ${candidate_latex}$ 适合使用等价无穷小法：",
        "",
        f"令 $t = \\frac{{1}}{{{latex(var)}}}$，当 ${latex(var)} \\to {infinity_symbol}$ 时 ${t_arrow}$。",
        "",
        "验证使用条件：",
        f"1. $\\lim_{{{t_arrow}}} {latex(base_t)} = 1$",
        f"2. $\\lim_{{{t_arrow}}} \\left[{latex(exponent_t)} \\cdot ({latex(base_t)} - 1)\\right] = 0$",
        "",
        "使用等价无穷小：",
        f"1. $\\ln(1 + u) \\sim u$ 当 $u \\to 0$（这里 $u = {latex(base_t - 1)}$）",
        f"2. $e^v - 1 \\sim v$ 当 $v \\to 0$（这里 $v = {latex(exponent_t * log(base_t))}$）",
        "",
        "因此：",
        f"${candidate_latex} \\sim {latex(power_minus_one_equiv)}$",
        "",
        "原表达式：",
        f"${latex(poly_t)} \\cdot {latex(power_minus_one_equiv)} = {latex(full_equiv)}$",
        "",
        "取极限得：",
        f"$\\lim_{{{t_arrow}}} {latex(full_equiv)} = {latex(result)}$"
    ]

    explanation = "\n".join(explanation_lines)

    # 用结果替换原表达式中的候选部分
    new_expr = expr.subs(candidate, result)
    return new_expr, explanation


def taylor_or_equivalent_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配适合使用泰勒展开或等价无穷小的表达式(只要表达式的某一部分适用即可)"""
    var, point, direction = context['variable'], context['point'], context.get(
        'direction', '+')

    # 只处理 x -> oo 的情况
    if point not in [oo, -oo]:
        return None

    # 查找所有可能的候选
    candidates = find_taylor_candidates(expr, var, point)
    if not candidates:
        return None

    return 'taylor_or_equivalent'
