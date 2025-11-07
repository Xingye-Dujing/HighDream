from typing import Any, Dict, Optional, Tuple, List
from sympy import (
    Expr, Symbol, Dummy, limit, series, O, sympify, expand, sqrt, Limit,
    sin, cos, tan, exp, log, asin, acos, atan, sinh, cosh, tanh, simplify,
    latex, S, Add, Mul, Pow, Integer, Rational, factorial, I, pi, E
)

def _get_limit_args(context: Dict[str, Any]) -> tuple:
    """获取极限参数"""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    dir_sup = '^{+}' if direction == '+' else '^{-}'
    return var, point, direction, dir_sup

def _is_zero_limit(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查表达式在给定点是否趋于 0"""
    try:
        lim_val = limit(expr, var, point, dir=direction)
        return lim_val == 0
    except:
        return False

def _is_infinite_limit(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查表达式在给定点是否趋于无穷"""
    try:
        lim_val = limit(expr, var, point, dir=direction)
        return lim_val in (S.Infinity, -S.Infinity)
    except:
        return False

def _is_safe_to_expand(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """
    检查表达式在给定点是否适合进行泰勒展开
    排除以下情况：
    - 包含 sin(1/x), cos(1/x), tan(1/x) 等在 0 点震荡发散的函数
    - 在展开点不连续或不可导
    - 含有未定义行为（如分母为 0, 对数负值等）
    """
    if point == 0:
        # 检查是否包含 sin(1/x), cos(1/x) 等形式
        for func in [sin, cos, tan, sinh, cosh, tanh, asin, acos, atan]:
            for subexpr in expr.atoms(func):
                if not subexpr.args:
                    continue
                arg = subexpr.args[0]
                # 检查参数是否包含 1/var 或类似发散形式
                if arg.has(1/var) or arg.has(var**(-1)) or arg.has(Pow(var, -1)):
                    return False
                # 检查参数在 point 处是否趋于无穷
                try:
                    arg_limit = limit(arg, var, point, dir=direction)
                    if arg_limit in (S.Infinity, -S.Infinity, S.ComplexInfinity):
                        return False
                except:
                    pass  # 无法计算极限，保守起见视为不安全
    # 检查函数在展开点是否定义
    try:
        val = expr.subs(var, point)
        if val.has(S.NaN, S.ComplexInfinity, S.Infinity, -S.Infinity):
            return False
    except:
        return False
    # 检查前几阶导数是否存在（保守检查1阶）
    try:
        deriv = expr.diff(var)
        deriv_val = deriv.subs(var, point)
        if deriv_val.has(S.NaN, S.ComplexInfinity):
            return False
    except:
        return False
    return True

def _determine_expansion_order(expr: Expr, var: Symbol, point: Any, direction: str, max_order: int = 10) -> int:
    """
    动态确定泰勒展开的阶数
    返回能够消除不定式的最小阶数
    """
    # 对于 0/0 型，我们需要找到分子分母的最低非零阶
    if expr.is_Mul or expr.is_Add:
        # 对于加减乘除表达式，递归检查各部分
        orders = []
        for arg in expr.args:
            if arg.has(var):
                orders.append(_determine_expansion_order(arg, var, point, direction, max_order))
        return max(orders) if orders else 1
    # 对于基本函数，根据类型确定展开阶数
    if expr.func in [sin, cos, tan, sinh, cosh, tanh]:
        return 2  # 至少展开到二阶
    elif expr.func in [exp, log]:
        return 3  # 至少展开到三阶
    elif expr.func in [asin, acos, atan]:
        return 3  # 至少展开到三阶
    elif isinstance(expr, Pow):
        base, exponent = expr.as_base_exp()
        if _is_zero_limit(base - 1, var, point, direction):
            return 3  # (1+x)^a 形式至少展开到三阶
        return 2
    return 1  # 默认展开到一阶

def binomial_coefficient(alpha, k):
    """计算广义二项式系数 C(α, k) = α(α-1)...(α-k+1)/k!"""
    if k < 0:
        return S.Zero
    if k == 0:
        return S.One
    result = S.One
    for i in range(k):
        result *= (alpha - i)
    return result / factorial(k)

def _taylor_expand_with_steps(expr: Expr, var: Symbol, point: Any, direction, order: int) -> Tuple[Expr, str]:
    """
    对表达式进行泰勒展开, 并返回展开式和步骤说明.
    """
    steps = []
    expanded = S.Zero
    
    if expr.is_Add:
        # 对加法表达式逐项展开
        term_expansions = []
        term_step_descs = []
        for term in expr.args:
            term_expanded, term_steps = _taylor_expand_with_steps(term, var, point, direction, order)
            term_expansions.append(term_expanded)
            term_step_descs.append(term_steps)
        expanded = Add(*term_expansions)
        return expanded, " + ".join(filter(None, term_step_descs))
    
    elif expr.is_Mul:
        # 对乘法表达式逐因子展开
        factor_expansions = []
        factor_step_descs = []
        for factor in expr.args:
            factor_expanded, factor_step = _taylor_expand_with_steps(factor, var, point, direction, order)
            factor_expansions.append(factor_expanded)
            factor_step_descs.append(factor_step)
        # 乘法展开需要特殊处理，避免过于复杂
        expanded = Mul(*factor_expansions)
        return expanded, " × ".join(filter(None, factor_step_descs))
    
    elif isinstance(expr, Pow):
        base, exponent = expr.as_base_exp()
        # 处理 (1+f(x))^g(x) 形式的幂
        if _is_zero_limit(base - 1, var, point, direction):
            f = base - 1
            f_expanded, f_steps = _taylor_expand_with_steps(f, var, point, direction, order)
            # 使用二项式定理展开
            if exponent.is_number:
                # 数值指数情况
                expanded = S.One
                step_parts = [f"1"]
                for k in range(1, order + 1):
                    binom_coeff = binomial_coefficient(exponent, k)
                    term = binom_coeff * (f_expanded)**k
                    expanded += term
                    if not term.is_zero:
                        step_parts.append(f"{latex(binom_coeff)} \\cdot ({f_steps})^{{{k}}}")
                step_desc = f"(1 + {f_steps})^{{{latex(exponent)}}} = {' + '.join(step_parts)}"
                return expanded, step_desc
            else:
                # 函数指数情况，使用指数对数变换
                log_expanded, log_steps = _taylor_expand_with_steps(log(base), var, point, direction, order)
                exponent_expanded, exponent_steps = _taylor_expand_with_steps(exponent, var, point, direction, order)
                expanded = exp(exponent_expanded * log_expanded)
                step_desc = f"e^{{{exponent_steps} \\cdot \\ln({base})}} = e^{{{exponent_steps} \\cdot {log_steps}}}"
                return expanded, step_desc
        # 处理其他幂函数
        base_expanded, base_steps = _taylor_expand_with_steps(base, var, point, direction, order)
        if exponent.is_number:
            expanded = base_expanded**exponent
            step_desc = f"({base_steps})^{{{latex(exponent)}}}"
            return expanded, step_desc
        else:
            exponent_expanded, exponent_steps = _taylor_expand_with_steps(exponent, var, point, direction, order)
            try:
                expanded = exp(exponent_expanded * log(base_expanded))
            except:
                expanded = expr  # fallback
            step_desc = f"e^{{{exponent_steps} \\cdot \\ln({base_steps})}}"
            return expanded, step_desc
    
    elif expr.func in [sin, cos, tan, exp, log, asin, acos, atan, sinh, cosh, tanh]:
        # 基本函数的泰勒展开
        arg = expr.args[0]
        arg_expanded, arg_steps = _taylor_expand_with_steps(arg, var, point, direction, order)
        
        # 计算函数在 0 点的泰勒级数（若 point 不为 0, 需平移）
        x0 = Dummy('x0')
        try:
            if point == 0:
                func_series = series(expr.func(x0), x0, 0, order+1).removeO()
            else:
                # 平移展开点
                func_series = series(expr.func(x0), x0, point, order+1).removeO()
        except:
            return expr, f"[展开失败：{expr.func.__name__} 在 {point} 处展开失败]"
        
        # 将参数代入级数
        try:
            expanded = func_series.subs(x0, arg_expanded)
        except:
            expanded = expr
        
        # 生成详细步骤说明
        func_name = expr.func.__name__
        
        if func_name == 'exp':
            # 特别处理 exp 函数，显示完整的泰勒级数
            exp_series_terms = []
            for i in range(order+1):
                term = (arg_expanded**i) / factorial(i)
                if not term.is_zero:
                    if i == 0:
                        exp_series_terms.append("1")
                    elif i == 1:
                        exp_series_terms.append(f"{arg_steps}")
                    else:
                        exp_series_terms.append(f"\\frac{{{arg_steps}^{{{i}}}}}{{{i}!}}")
            
            step_desc = f"e^{{{arg_steps}}} = {' + '.join(exp_series_terms)} + o({var}^{order})"
            return expanded, step_desc
            
        elif func_name == 'sin':
            sin_series_terms = []
            for i in range(0, order+1, 2):
                if i % 4 == 0:  # 正号项
                    if i == 0:
                        sin_series_terms.append(f"{arg_steps}")
                    else:
                        sin_series_terms.append(f"\\frac{{{arg_steps}^{{{i+1}}}}}{{{i+1}!}}")
                else:  # 负号项
                    sin_series_terms.append(f"-\\frac{{{arg_steps}^{{{i+1}}}}}{{{i+1}!}}")
            
            step_desc = f"\\sin({arg_steps}) = {' + '.join(sin_series_terms)} + o({var}^{order})"
            return expanded, step_desc
            
        elif func_name == 'cos':
            cos_series_terms = ["1"]
            for i in range(1, order+1, 2):
                if i % 4 == 1:  # 负号项
                    cos_series_terms.append(f"-\\frac{{{arg_steps}^{{{i+1}}}}}{{{i+1}!}}")
                else:  # 正号项
                    cos_series_terms.append(f"\\frac{{{arg_steps}^{{{i+1}}}}}{{{i+1}!}}")
            
            step_desc = f"\\cos({arg_steps}) = {' + '.join(cos_series_terms)} + o({var}^{order})"
            return expanded, step_desc
            
        else:
            # 其他函数使用通用描述
            step_desc = f"{func_name}({arg_steps}) 在{point}点的泰勒展开: {latex(func_series)}"
            return expanded, step_desc
    
    else:
        # 默认情况：直接使用 SymPy 的 series 函数
        try:
            expanded = series(expr, var, point, order+1).removeO()
            step_desc = f"{latex(expr)} 在 {latex(point)} 点的泰勒展开: {latex(expanded)}"
            return expanded, step_desc
        except:
            # 如果无法展开，返回原表达式
            return expr, f"[无法展开：{latex(expr)} 在 {latex(point)} 处]"

def taylor_quotient_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """
    使用泰勒展开处理 0/0 型极限.
    对分子和分母分别进行泰勒展开, 然后相除
    """
    var, point, direction, dir_sup = _get_limit_args(context)
    # 提取分子和分母
    numerator, denominator = expr.as_numer_denom()
    # 确定展开阶数
    order = _determine_expansion_order(numerator, var, point, direction)
    order = max(order, _determine_expansion_order(denominator, var, point, direction))
    # 对分子和分母分别进行泰勒展开
    num_expanded, num_steps = _taylor_expand_with_steps(numerator, var, point, direction, order)
    den_expanded, den_steps = _taylor_expand_with_steps(denominator, var, point, direction, order)
    # 构建新的表达式
    try:
        new_expr = simplify(num_expanded / den_expanded)
    except:
        new_expr = expr
    # 生成说明
    explanation = (
        f"原式为 $\\frac{{0}}{{0}}$ 型不定式，应用泰勒展开法：\n\n"
        f"分子 ${latex(numerator)}$ 在 ${latex(var)} = {latex(point)}$ 处的泰勒展开：\n"
        f"${num_steps}$\n\n"
        f"分母 ${latex(denominator)}$ 在 ${latex(var)} = {latex(point)}$ 处的泰勒展开：\n"
        f"${den_steps}$\n\n"
        f"因此，原极限可转化为：\n"
        f"$\\lim_{{{var} \\to {latex(point)}{dir_sup}}} \\frac{{{latex(num_expanded)}}}{{{latex(den_expanded)}}}$"
    )
    return Limit(new_expr, var, point, direction), explanation

def taylor_substitution_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """
    使用变量替换结合泰勒展开处理极限,
    特别适用于极限点不是 0 的情况
    """
    var, point, direction, dir_sup = _get_limit_args(context)
    # 创建新变量 t = x - a，将极限点移动到 0
    t = Dummy('t')
    substitution = {var: t + point}
    new_expr = expr.subs(substitution)
    # 对新表达式应用泰勒展开
    new_context = context.copy()
    new_context['variable'] = t
    new_context['point'] = 0
    result, explanation = taylor_quotient_rule(new_expr, new_context)
    # 替换回原变量
    result = result.subs(t, var - point)
    # 更新说明
    new_explanation = (
        f"通过变量替换 $t = {latex(var)} - {latex(point)}$，将极限点移动到0：\n"
        f"${latex(expr)} = {latex(new_expr)}$\n\n"
        f"{explanation}"
    )
    return result, new_explanation

def taylor_infinity_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """处理极限点为无穷的情况, 通过变量替换 t = 1/x"""
    var, point, direction, dir_sup = _get_limit_args(context)
    # 创建新变量 t = 1/x
    t = Dummy('t')
    if point == S.Infinity:
        substitution = {var: 1/t}
        new_point = 0
        new_direction = '+' if direction == '+' else '-'
    else:  # point == -S.Infinity
        substitution = {var: -1/t}
        new_point = 0
        new_direction = '-' if direction == '+' else '+'
    new_expr = expr.subs(substitution)
    new_context = context.copy()
    new_context['variable'] = t
    new_context['point'] = new_point
    new_context['direction'] = new_direction
    result, explanation = taylor_quotient_rule(new_expr, new_context)
    new_explanation = (
        f"通过变量替换 $t = \\frac{{1}}{{{latex(var)}}}$, 将无穷极限转化为有限极限: \n"
        f"${latex(expr)} = {latex(new_expr)}$\n\n"
        f"{explanation}"
    )
    return result, new_explanation

def taylor_composition_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """
    处理复合函数的泰勒展开,
    例如：sin(sin(x)), exp(sin(x))等.
    """
    var, point, direction, dir_sup = _get_limit_args(context)
    # 确定展开阶数
    order = _determine_expansion_order(expr, var, point, direction)
    
    # 特别处理 exp(f(x)) 的情况，提供详细步骤
    if expr.func == exp and len(expr.args) == 1:
        inner_expr = expr.args[0]
        
        # 第一步：对内层函数进行泰勒展开
        inner_expanded, inner_steps = _taylor_expand_with_steps(inner_expr, var, point, direction, order)
        
        # 第二步：对 exp 函数进行泰勒展开
        exp_expanded, exp_steps = _taylor_expand_with_steps(expr, var, point, direction, order)
        exp_expanded = simplify(exp_expanded)
        
        # 生成详细步骤说明
        explanation = (
            f"应用复合函数的泰勒展开，详细步骤如下：\n\n"
            f"第一步：对内层函数 ${latex(inner_expr)}$ 在 ${latex(var)} = {latex(point)}$ 处进行泰勒展开\n"
            f"${inner_steps}$\n\n"
            f"第二步：将展开结果代入指数函数 $e^u$ 的泰勒级数\n"
            f"${exp_steps}$\n\n"
            f"因此，原极限可转化为：\n"
            f"$\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {latex(exp_expanded)}$"
        )
        
        return Limit(exp_expanded, var, point, direction), explanation
    
    # 一般情况
    expanded, steps = _taylor_expand_with_steps(expr, var, point, direction, order)
    # 生成说明
    explanation = (
        f"应用复合函数的泰勒展开：\n"
        f"${latex(expr)} = {steps}$\n\n"
        f"因此，原极限可转化为：\n"
        f"$\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {latex(expanded)}$"
    )
    return Limit(expanded, var, point, direction), explanation

def taylor_quotient_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 0/0 型极限，适合用泰勒展开处理"""
    var, point, direction, _ = _get_limit_args(context)
    # 检查是否为分式
    numerator, denominator = expr.as_numer_denom()
    if denominator == 1:
        return None
    # 检查是否为 0/0 型
    if not (_is_zero_limit(numerator, var, point, direction) and
            _is_zero_limit(denominator, var, point, direction)):
        return None
    # 分子和分母都必须是可展开的
    if not (_is_safe_to_expand(numerator, var, point, direction) and
            _is_safe_to_expand(denominator, var, point, direction)):
        return None
    return 'taylor_quotient'

def taylor_substitution_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配需要变量替换的极限(极限点不是 0)"""
    var, point, direction, _ = _get_limit_args(context)
    # 检查极限点是否为 0 或无穷
    if point == 0 or point in (S.Infinity, -S.Infinity):
        return None
    # 检查表达式是否包含复杂函数
    has_complex_func = any(
        expr.has(func) for func in [sin, cos, tan, exp, log, asin, acos, atan, sinh, cosh, tanh]
    )
    if not has_complex_func:
        return None
    # 替换后(t = x - a)在 t = 0 处必须是可展开的
    t = Dummy('t')
    substitution = {var: t + point}
    new_expr = expr.subs(substitution)
    if not _is_safe_to_expand(new_expr, t, 0, direction):
        return None
    return 'taylor_substitution'

def taylor_infinity_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配极限点为无穷的情况"""
    var, point, direction, _ = _get_limit_args(context)
    if point not in (S.Infinity, -S.Infinity):
        return None
    # 替换后 (t=1/x 或 t=-1/x) 在 t = 0+ 或 t = 0- 处必须是可展开的
    t = Dummy('t')
    if point == S.Infinity:
        substitution = {var: 1/t}
        new_direction = '+' if direction == '+' else '-'
    else:
        substitution = {var: -1/t}
        new_direction = '-' if direction == '+' else '+'
    new_expr = expr.subs(substitution)
    if not _is_safe_to_expand(new_expr, t, 0, new_direction):
        return None
    return 'taylor_infinity'

def taylor_composition_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配复合函数的情况"""
    if not expr.is_Function:
        return None
    # 检查是否为复合函数
    arg = expr.args[0]
    if not arg.has(context['variable']):
        return None
    # 检查内函数是否趋于 0
    if not _is_zero_limit(arg, context['variable'], context['point'], context.get('direction', '+')):
        return None
    # 外层函数和内层函数在相关点都必须是可展开的
    var, point, direction, _ = _get_limit_args(context)
    # 内层函数 arg 在 point 处是否可展开
    if not _is_safe_to_expand(arg, var, point, direction):
        return None
    # 检查外层函数 f(u) 在 u=0 处是否可展开 (因为 arg -> 0)
    u = Dummy('u')
    outer_func = expr.func(u)
    if not _is_safe_to_expand(outer_func, u, 0, '+'):
        return None
    return 'taylor_composition'