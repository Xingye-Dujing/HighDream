from typing import Any, Dict
from sympy import (
    Add, Expr, Limit, Mul, Symbol, limit, log, Pow, asin,
    acos, Interval, solveset, oo, S, sympify, zoo
)


def detect_feasible_directions(expr: Expr, var: Symbol, point: str) -> Dict[str, bool]:
    """
    检测在给定极限点处, 左极限和右极限是否具备基本的可计算性(主要基于定义域).
    Returns:
        Dict[str, bool]: 字典, 包含 'left' 和 'right' 键, 值为布尔值.
                         True 表示该方向可能可计算(满足基本定义域要求),
                         False 表示该方向很可能不可计算(违反基本定义域要求).
    """
    # 初始化结果，默认两个方向都可行
    feasible = {'left': True, 'right': True}

    # 如果极限点是无穷大，则只有单侧（或视为双侧相同），无需复杂检测
    if point in (oo, -oo, S.Infinity, S.NegativeInfinity):
        # 对于无穷大，我们通常不区分严格的左/右，或者认为只有一个方向有意义
        # 这里简单返回都可行，因为检测无穷附近的定义域比较复杂且不常见
        return feasible

    # 尝试将 point 转换为 SymPy 对象
    point = sympify(point)

    # 核心思路：检查表达式在极限点 var=point 附近，左右邻域内是否有定义。
    # 我们通过分析表达式中关键子表达式（对数、根号、反三角）的定义域约束来实现。

    # 收集所有需要检查定义域的子表达式及其约束条件
    constraints = []

    # 1. 检查对数函数 log(f(var)) 或 ln(f(var))
    for log_expr in expr.atoms(log):
        # 获取对数的真数部分
        arg_expr = log_expr.args[0]
        # 约束：真数 > 0
        constraints.append((arg_expr > 0, f"对数真数 {arg_expr} 必须大于0"))
    # 2. 检查偶次根号 sqrt(f(var)) 或 f(var)**(1/n) 其中 n 为偶数
    #    注意：sympy 的 sqrt 是特殊的，**(1/2) 也是
    for pow_expr in expr.atoms(Pow):
        base, exponent = pow_expr.args
        # 判断是否为偶次根号: 指数是正的有理数且分母为偶数
        # 或者是 sqrt 函数 (sympy 的 sqrt 是 Pow 的特例，指数为 1/2)
        if exponent.is_Rational and exponent > 0:
            if exponent.q % 2 == 0:  # 分母是偶数，即偶次根
                constraints.append(
                    (base >= 0, f"偶次根号 {pow_expr} 的底数 {base} 必须非负"))
    # 3. 检查反三角函数 asin(f(var)), acos(f(var))
    for asin_expr in expr.atoms(asin):
        arg_expr = asin_expr.args[0]
        constraints.append(
            (arg_expr >= -1, f"asin 参数 {arg_expr} 必须 >= -1"))
        constraints.append((arg_expr <= 1, f"asin 参数 {arg_expr} 必须 <= 1"))
    for acos_expr in expr.atoms(acos):
        arg_expr = acos_expr.args[0]
        constraints.append(
            (arg_expr >= -1, f"acos 参数 {arg_expr} 必须 >= -1"))
        constraints.append((arg_expr <= 1, f"acos 参数 {arg_expr} 必须 <= 1"))
    # 4. 可以扩展检查其他有定义域限制的函数，例如 1/sqrt(f(var)) 隐含 f(var)>0
    #    但上述检查通常已覆盖主要情况
    # 如果没有约束，所有方向都可行
    if not constraints:
        return feasible
    # 对于每个约束，检查在极限点左侧和右侧的邻域内是否可能满足
    for constraint_expr, _ in constraints:
        # 我们关心的是在 point 的左侧 (var < point) 和右侧 (var > point) 是否存在满足约束的点
        # 使用 solveset 或直接分析来判断
        # 方法1: 尝试求解约束在 point 附近的解集
        # 定义左侧邻域 (开区间) 和右侧邻域 (开区间)
        left_interval = Interval.open(
            point - 1e-6, point) if point != -oo else Interval(-oo, point)
        right_interval = Interval.open(
            point, point + 1e-6) if point != oo else Interval(point, oo)
        # 检查约束在左侧邻域是否有解
        try:
            left_solution_set = solveset(
                constraint_expr, var, domain=left_interval)
            has_left_solution = not left_solution_set.is_empty
        except (NotImplementedError, ValueError):
            # 如果求解失败，保守地假设该方向可能可行，或者标记为需要进一步检查
            has_left_solution = True  # 保守策略：假设可行，让后续计算去处理
            # 或者可以设置为 False 来更严格地排除，但这可能导致误报
            # has_left_solution = False
        # 检查约束在右侧邻域是否有解
        try:
            right_solution_set = solveset(
                constraint_expr, var, domain=right_interval)
            has_right_solution = not right_solution_set.is_empty
        except (NotImplementedError, ValueError):
            has_right_solution = True  # 保守策略

        # 如果左侧邻域无解，则左极限很可能不可计算
        if not has_left_solution:
            feasible['left'] = False
            # print(f"左极限检测失败: {reason} 在 x->{point}- 时无法满足")
        # 如果右侧邻域无解，则右极限很可能不可计算
        if not has_right_solution:
            feasible['right'] = False
            # print(f"右极限检测失败: {reason} 在 x->{point}+ 时无法满足")
        # 如果某个方向已经被标记为 False，可以提前跳出内层循环，但外层循环继续检查其他约束
        if not feasible['left'] and not feasible['right']:
            break  # 所有方向都不可行，无需检查剩余约束

    return feasible


def get_limit_args(context: Dict[str, Any]) -> tuple:
    """获取极限参数"""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    return var, point, direction


def check_limit_exists_oo(lim_val: Expr) -> bool:
    """
    检查极限是否存在且为有限值(含无穷)
    """
    return lim_val.is_finite or lim_val == oo or lim_val == -oo


def check_function_tends_to_zero(f: Expr, var: Symbol, point: Any, direction: str) -> bool:
    try:
        lim = limit(f, var, point, dir=direction)
        return lim == 0
    except Exception:
        return False


def check_limit_exists(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """
    检查极限是否存在且为有限值
    """
    try:
        lim = Limit(expr, var, point, dir=direction).doit()
        return lim.is_finite
    except Exception:
        return False


def is_indeterminate_form(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """
    检查表达式在给定点是否为不定式
    不定式包括: 0/0, oo/oo, 0·oo, oo-oo, 1^oo, 0^0, oo^0
    """
    try:
        # 尝试计算极限值
        _lim_val = limit(expr, var, point, dir=direction)
        # 检查常见的不定式形式
        if isinstance(expr, Mul):
            factors = expr.as_ordered_factors()
            # 检查 0·oo 形式
            zero_found = False
            inf_found = False
            for factor in factors:
                factor_lim = limit(factor, var, point, dir=direction)
                if factor_lim == 0:
                    zero_found = True
                elif factor_lim in (oo, -oo):
                    inf_found = True
            if zero_found and inf_found:
                return True
        elif isinstance(expr, Add):
            terms = expr.as_ordered_terms()
            # 检查 oo-oo 形式
            pos_inf_found = False
            neg_inf_found = False
            for term in terms:
                term_lim = limit(term, var, point, dir=direction)
                if term_lim == oo:
                    pos_inf_found = True
                elif term_lim == -oo:
                    neg_inf_found = True
            if pos_inf_found and neg_inf_found:
                return True
        elif isinstance(expr, Pow):
            base, exponent = expr.base, expr.exp
            base_lim = limit(base, var, point, dir=direction)
            exp_lim = limit(exponent, var, point, dir=direction)

            # 检查 1^oo, 0^0, oo^0 形式
            if (base_lim == 1 and exp_lim in (oo, -oo)) or \
               (base_lim == 0 and exp_lim == 0) or \
               (base_lim in (oo, -oo) and exp_lim == 0):
                return True

        return False

    except Exception:
        return True


def check_combination_indeterminate(part1: Expr, part2: Expr, var: Symbol, point: Any, direction: str, operation: str) -> bool:
    """
    检查两个部分组合后是否会产生不定式
    operation: 'mul' 或 'add'
    """
    try:
        if operation == 'mul':
            # 检查乘法组合：0·oo 或 oo·0
            lim1 = limit(part1, var, point, dir=direction)
            lim2 = limit(part2, var, point, dir=direction)

            if (lim1 == 0 and lim2 in (oo, -oo)) or (lim1 in (oo, -oo) and lim2 == 0):
                return True

        elif operation == 'add':
            # 检查加法组合：oo - oo
            lim1 = limit(part1, var, point, dir=direction)
            lim2 = limit(part2, var, point, dir=direction)

            if (lim1 == oo and lim2 == -oo) or (lim1 == -oo and lim2 == oo):
                return True

        return False
    except Exception:
        return False


def is_infinite(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查表达式极限是否为无穷"""
    try:
        lim = limit(expr, var, point, dir=direction)
        return lim in (oo, -oo)
    except Exception:
        return False


def is_zero(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查表达式极限是否为 0"""
    try:
        lim = limit(expr, var, point, dir=direction)
        return lim == 0
    except Exception:
        return False


def is_constant(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查表达式的极限是否为常数(即数值而非无穷或符号)"""
    try:
        lim = limit(expr, var, point, dir=direction)
        # 检查是否为数值常数(非无穷, 非符号)
        return lim.is_real and not lim.has(oo, -oo, zoo)
    except Exception:
        return False
