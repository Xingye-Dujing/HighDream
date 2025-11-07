from typing import Optional, Tuple, Dict, Any, List
from sympy import (
    Symbol, series, limit, oo, exp, log, Pow, Add, Mul, S, Expr, latex, 
    preorder_traversal, Function, sin, tan, asin, atan, sinh, tanh, asinh, 
    atanh, E, I, pi, factorial, sqrt, symbols, simplify, expand, fraction, 
    Poly, Wild, sympify, together, cos, acos, cot, sec, csc, acot, asec, acsc, 
    cosh, coth, sech, csch, acosh, acoth, asech, acsch, Order, PoleError,
    nsimplify
)


def _substitute_to_zero(expr: Expr, var: Symbol, point: Any) -> Tuple[Expr, Symbol, str]:
    """将极限点转换为 0, 返回新表达式、新变量和方向"""
    if point == oo:
        t = Symbol('t', positive=True)
        return expr.subs(var, 1/t), t, '+'
    elif point == -oo:
        t = Symbol('t', positive=True)
        return expr.subs(var, -1/t), t, '+'
    elif point == 0:
        t = Symbol('t')
        return expr.subs(var, t), var, '+'
    else:
        t = Symbol('t')
        return expr.subs(var, point + t), t, '0'


def _restore_limit_point(expr: Expr, new_var: Symbol, original_var: Symbol, point: Any) -> Expr:
    """将表达式从转换后的变量恢复为原始变量"""
    if point == oo:
        return expr.subs(new_var, 1/original_var)
    elif point == -oo:
        return expr.subs(new_var, -1/original_var)
    elif point == 0:
        return expr
    else:
        return expr.subs(new_var, original_var - point)


def _get_series_order(expr: Expr, var: Symbol, point: Any, direction: str) -> int:
    """动态确定泰勒展开所需的阶数"""
    complexity = _count_nodes(expr)
    
    # 检查是否为分式形式
    try:
        num, den = fraction(expr)
        if num.has(var) and den.has(var):
            complexity += 15
    except:
        pass
    
    # 检查是否有复合函数
    composite_count = len(_find_composite_functions(expr))
    complexity += composite_count * 5
    
    # 检查是否有幂指函数
    exponential_count = len(_find_exponential_forms(expr))
    complexity += exponential_count * 8
    
    # 根据复杂度确定阶数
    if complexity < 15:
        return 4
    elif complexity < 30:
        return 6
    elif complexity < 50:
        return 8
    else:
        return 10


def _count_nodes(expr: Expr) -> int:
    """计算表达式的节点数"""
    if not expr.args:
        return 1
    return 1 + sum(_count_nodes(arg) for arg in expr.args)


def _is_exponential_form(expr: Expr) -> bool:
    """检查表达式是否为指数形式 f(x)^g(x)"""
    return (isinstance(expr, Pow) and 
            expr.base.has(Symbol) and 
            expr.exp.has(Symbol) and
            not any(arg.is_constant() for arg in expr.args))


def _find_exponential_forms(expr: Expr) -> List[Expr]:
    """在表达式中查找所有指数形式的子表达式"""
    return [subexpr for subexpr in preorder_traversal(expr) if _is_exponential_form(subexpr)]


def _find_minus_one_forms(expr: Expr) -> List[Expr]:
    """在表达式中查找所有形如 f(x)^g(x) - 1 的子表达式"""
    minus_one_forms = []
    for subexpr in preorder_traversal(expr):
        if (isinstance(subexpr, Add) and 
            any(term == -1 for term in subexpr.args) and
            len(subexpr.args) == 2):
            other_term = next(t for t in subexpr.args if t != -1)
            if _is_exponential_form(other_term):
                minus_one_forms.append(subexpr)
    return minus_one_forms


def _find_composite_functions(expr: Expr) -> List[Expr]:
    """在表达式中查找所有复合函数形式的子表达式"""
    composite_functions = []
    for subexpr in preorder_traversal(expr):
        if (isinstance(subexpr, Function) and 
            any(arg.has(Symbol) for arg in subexpr.args) and
            not subexpr.is_constant()):
            composite_functions.append(subexpr)
        elif (isinstance(subexpr, (Mul, Add, Pow)) and 
              any(isinstance(arg, Function) and not arg.is_constant() for arg in subexpr.args)):
            composite_functions.append(subexpr)
    return composite_functions


def _get_safe_series(expr: Expr, var: Symbol, point: int, order: int) -> Optional[Expr]:
    """安全地进行泰勒展开，避免错误"""
    try:
        # 尝试直接展开
        series_result = series(expr, var, point, order)
        return series_result
    except Exception as e:
        try:
            # 如果直接展开失败，尝试简化后再展开
            simplified_expr = simplify(expr)
            series_result = series(simplified_expr, var, point, order)
            return series_result
        except:
            return None


def taylor_exponential_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """使用泰勒展开求解幂指函数极限"""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    
    exponential_forms = _find_exponential_forms(expr)
    minus_one_forms = _find_minus_one_forms(expr)
    
    if not exponential_forms and not minus_one_forms:
        return None
    
    # 选择最合适的候选表达式
    candidate = None
    is_minus_one = False
    
    if minus_one_forms:
        candidate = minus_one_forms[0]
        is_minus_one = True
        power_expr = next(term for term in candidate.args if term != -1)
        base, exponent = power_expr.args
    elif exponential_forms:
        candidate = exponential_forms[0]
        base, exponent = candidate.args
    else:
        return None
    
    new_expr, t, t_direction = _substitute_to_zero(expr, var, point)
    
    # 构建替换后的基础表达式
    if point in [oo, -oo]:
        base_t = base.subs(var, 1/t)
        exponent_t = exponent.subs(var, 1/t)
    else:
        base_t = base.subs(var, point + t)
        exponent_t = exponent.subs(var, point + t)
    
    # 验证基础是否趋近于1
    try:
        base_limit = limit(base_t, t, 0, t_direction)
        if base_limit != 1:
            return None
    except:
        return None
    
    order = _get_series_order(new_expr, t, 0, t_direction)
    
    try:
        # 对 ln(base) 进行泰勒展开
        ln_base_series = _get_safe_series(log(base_t), t, 0, order)
        if ln_base_series is None:
            return None
        
        # 计算指数部分
        exponent_ln_series = (exponent_t * ln_base_series).expand()
        
        # 对指数函数进行泰勒展开
        exp_series = _get_safe_series(exp(exponent_ln_series), t, 0, order)
        if exp_series is None:
            return None
        
        # 根据是否是减1形式调整结果
        if is_minus_one:
            result_series = exp_series - 1
        else:
            result_series = exp_series
        
        # 计算极限
        result = limit(result_series, t, 0, t_direction)
        
    except Exception as e:
        return None
    
    # 构建详细的解释
    if point in [oo, -oo]:
        t_sub = f"t = \\frac{{1}}{{{latex(var)}}}"
    else:
        t_sub = f"t = {latex(var)} - {latex(point)}"
    
    typ = f"{latex(base)}^{{{latex(exponent)}}}" + (" - 1" if is_minus_one else "")
    
    explanation = (
        f"## 幂指函数极限求解（泰勒展开法）\n\n"
        f"**原表达式**: $\\lim_{{{latex(var)} \\to {latex(point)}}} {latex(expr)}$\n\n"
        f"**步骤1**: 变量替换\n"
        f"令 ${t_sub}$，则当 ${latex(var)} \\to {latex(point)}$ 时 $t \\to 0^{t_direction}$\n\n"
        f"**步骤2**: 表达式转换\n"
        f"转换后表达式: ${latex(new_expr)}$\n\n"
        f"**步骤3**: 识别幂指函数形式\n"
        f"识别到幂指函数: ${typ}$\n\n"
        f"**步骤4**: 泰勒展开\n"
        f"对 $\\ln({latex(base_t)})$ 进行 {order} 阶泰勒展开:\n"
        f"$\\ln({latex(simplify(base_t))}) = {latex(ln_base_series)}$\n\n"
        f"计算指数部分:\n"
        f"${latex(exponent_t)} \\cdot \\ln({latex(simplify(base_t))}) = {latex(exponent_ln_series)}$\n\n"
        f"对指数函数进行泰勒展开:\n"
        f"$e^{{{latex(exponent_ln_series)}}} = {latex(exp_series)}$\n\n"
        f"**步骤5**: 计算极限\n"
        f"最终极限: $\\lim_{{t \\to 0^{t_direction}}} {latex(result_series)} = {latex(result)}$\n\n"
        f"**最终结果**: $\\lim_{{{latex(var)} \\to {latex(point)}}} {latex(expr)} = {latex(result)}$"
    )
    
    return result, explanation


def taylor_exponential_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配适合使用泰勒展开的幂指函数表达式"""
    var, point = context['variable'], context['point']
    
    exponential_forms = _find_exponential_forms(expr)
    minus_one_forms = _find_minus_one_forms(expr)
    
    for candidate in exponential_forms + minus_one_forms:
        if isinstance(candidate, Add) and -1 in candidate.args:
            power_expr = next(term for term in candidate.args if term != -1)
            base, exponent = power_expr.args
        else:
            base, exponent = candidate.args
        
        new_expr, t, t_direction = _substitute_to_zero(expr, var, point)
        
        if point in [oo, -oo]:
            base_t = base.subs(var, 1/t)
        else:
            base_t = base.subs(var, point + t)
        
        try:
            base_limit = limit(base_t, t, 0, t_direction)
            if base_limit == 1:
                return 'taylor_exponential'
        except:
            continue
    
    return None


def taylor_composite_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """使用泰勒展开求解复合函数极限"""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    
    composite_functions = _find_composite_functions(expr)
    if not composite_functions:
        return None
    
    # 选择最复杂的复合函数进行处理
    candidate = max(composite_functions, key=lambda x: _count_nodes(x))
    
    new_expr, t, t_direction = _substitute_to_zero(expr, var, point)
    
    if point in [oo, -oo]:
        candidate_t = candidate.subs(var, 1/t)
    else:
        candidate_t = candidate.subs(var, point + t)
    
    order = _get_series_order(new_expr, t, 0, t_direction)
    
    try:
        series_expansion = _get_safe_series(candidate_t, t, 0, order)
        if series_expansion is None:
            return None
        
        result = limit(series_expansion, t, 0, t_direction)
        
    except Exception as e:
        return None
    
    # 构建详细的解释
    if point in [oo, -oo]:
        t_sub = f"t = \\frac{{1}}{{{latex(var)}}}"
    else:
        t_sub = f"t = {latex(var)} - {latex(point)}"
    
    explanation = (
        f"变量替换: \\quad"
        f"令 ${t_sub}$, 则当 ${latex(var)} \\to {latex(point)}^{direction}$ 时 $t \\to 0^{t_direction}$ \\quad"
        f"转换后表达式: ${latex(new_expr)}$\\quad"
        f"识别到复合函数: ${latex(candidate)}$\\quad"
        f"对 ${latex(candidate_t)}$ 进行 {order} 阶泰勒展开:"
        f"${latex(candidate_t)} = {latex(series_expansion)}$ \\quad"
        f"最终极限: $\\lim_{{t \\to 0^{t_direction}}} {latex(series_expansion)} = {latex(result)}$ \\quad"
        f"最终结果: $\\lim_{{{latex(var)} \\to {latex(point)}}} {latex(expr)} = {latex(result)}$"
    )
    
    return result, explanation


def taylor_composite_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配适合使用泰勒展开的复合函数表达式"""
    var, point = context['variable'], context['point']
    composite_functions = _find_composite_functions(expr)
    
    if composite_functions:
        new_expr, t, t_direction = _substitute_to_zero(expr, var, point)
        try:
            # 尝试进行低阶展开来验证可行性
            test_series = _get_safe_series(new_expr, t, 0, 3)
            if test_series is not None:
                return 'taylor_composite'
        except:
            pass
    return None


def equivalent_infinitesimal_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """使用等价无穷小替换求解极限"""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    
    new_expr, t, t_direction = _substitute_to_zero(expr, var, point)
    
    # 定义常见的等价无穷小替换对
    equivalents = [
        (sin(t), t), (tan(t), t), (asin(t), t), (atan(t), t),
        (sinh(t), t), (tanh(t), t), (asinh(t), t), (atanh(t), t),
        (exp(t) - 1, t), (log(1 + t), t),
        (1 - cos(t), t**2/2), ((1 + t)**S('a') - 1, S('a')*t)
    ]
    
    simplified_expr = new_expr
    replacements = []
    
    # 应用等价无穷小替换
    for old, new in equivalents:
        # 检查是否存在可以替换的模式
        if simplified_expr.has(old.func):
            # 创建一个模式来匹配 old
            pattern = old
            # 尝试替换
            new_simplified = simplified_expr.replace(old, new)
            if new_simplified != simplified_expr:
                simplified_expr = new_simplified
                replacements.append((latex(old), latex(new)))
    
    if simplified_expr == new_expr:
        return None
    
    try:
        result = limit(simplified_expr, t, 0, t_direction)
    except:
        return None
    
    # 构建详细的解释
    if point in [oo, -oo]:
        t_sub = f"t = \\frac{{1}}{{{latex(var)}}}"
    else:
        t_sub = f"t = {latex(var)} - {latex(point)}"
    
    replacement_text = "\n".join([f"- ${old} \\sim {new}$" for old, new in replacements])
    
    explanation = (
        f"## 极限求解（等价无穷小替换法）\n\n"
        f"**原表达式**: $\\lim_{{{latex(var)} \\to {latex(point)}}} {latex(expr)}$\n\n"
        f"**步骤1**: 变量替换\n"
        f"令 ${t_sub}$，则当 ${latex(var)} \\to {latex(point)}$ 时 $t \\to 0^{t_direction}$\n\n"
        f"**步骤2**: 表达式转换\n"
        f"转换后表达式: ${latex(new_expr)}$\n\n"
        f"**步骤3**: 应用等价无穷小替换\n"
        f"应用以下等价关系:\n{replacement_text}\n\n"
        f"**步骤4**: 简化表达式\n"
        f"简化后表达式: ${latex(simplified_expr)}$\n\n"
        f"**步骤5**: 计算极限\n"
        f"最终极限: $\\lim_{{t \\to 0^{t_direction}}} {latex(simplified_expr)} = {latex(result)}$\n\n"
        f"**最终结果**: $\\lim_{{{latex(var)} \\to {latex(point)}}} {latex(expr)} = {latex(result)}$"
    )
    
    return result, explanation


def equivalent_infinitesimal_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配适合使用等价无穷小替换的表达式"""
    var, point = context['variable'], context['point']
    new_expr, t, t_direction = _substitute_to_zero(expr, var, point)
    
    common_forms = [
        sin(t), tan(t), asin(t), atan(t), sinh(t), tanh(t), asinh(t), atanh(t),
        exp(t) - 1, log(1 + t), 1 - cos(t), (1 + t)**S('a') - 1
    ]
    
    for form in common_forms:
        if new_expr.has(form.func):
            return 'equivalent_infinitesimal'
    return None


def taylor_rational_rule(expr: Expr, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """使用泰勒展开求解分式极限"""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    
    try:
        numerator, denominator = fraction(expr)
    except:
        return None
    
    new_expr, t, t_direction = _substitute_to_zero(expr, var, point)
    new_num, new_den = fraction(new_expr)
    
    try:
        num_limit = limit(new_num, t, 0, t_direction)
        den_limit = limit(new_den, t, 0, t_direction)
        # 只处理 0/0 或 ∞/∞ 型未定式
        if not ((num_limit == 0 and den_limit == 0) or 
                (abs(num_limit) == oo and abs(den_limit) == oo)):
            return None
    except:
        return None
    
    order = _get_series_order(new_expr, t, 0, t_direction)
    
    try:
        num_series = _get_safe_series(new_num, t, 0, order)
        den_series = _get_safe_series(new_den, t, 0, order)
        
        if num_series is None or den_series is None:
            return None
        
        result = limit(num_series / den_series, t, 0, t_direction)
        
    except Exception as e:
        return None
    
    # 构建详细的解释
    if point in [oo, -oo]:
        t_sub = f"t = \\frac{{1}}{{{latex(var)}}}"
    else:
        t_sub = f"t = {latex(var)} - {latex(point)}"
    
    explanation = (
        f"## 分式极限求解（泰勒展开法）\n\n"
        f"**原表达式**: $\\lim_{{{latex(var)} \\to {latex(point)}}} {latex(expr)}$\n\n"
        f"**步骤1**: 变量替换\n"
        f"令 ${t_sub}$，则当 ${latex(var)} \\to {latex(point)}$ 时 $t \\to 0^{t_direction}$\n\n"
        f"**步骤2**: 表达式转换\n"
        f"转换后表达式: ${latex(new_expr)}$\n\n"
        f"**步骤3**: 识别未定式类型\n"
        f"分子极限: ${latex(num_limit)}$, 分母极限: ${latex(den_limit)}$\n"
        f"识别为 {'0/0' if num_limit == 0 else '∞/∞'} 型未定式\n\n"
        f"**步骤4**: 泰勒展开\n"
        f"分子 {order} 阶泰勒展开:\n"
        f"${latex(new_num)} = {latex(num_series)} + O(t^{order})$\n\n"
        f"分母 {order} 阶泰勒展开:\n"
        f"${latex(new_den)} = {latex(den_series)} + O(t^{order})$\n\n"
        f"**步骤5**: 计算分式极限\n"
        f"$\\lim_{{t \\to 0^{t_direction}}} \\frac{{{latex(num_series)}}}{{{latex(den_series)}}} = {latex(result)}$\n\n"
        f"**最终结果**: $\\lim_{{{latex(var)} \\to {latex(point)}}} {latex(expr)} = {latex(result)}$"
    )
    
    return result, explanation


def taylor_rational_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配适合使用泰勒展开的分式表达式"""
    var, point = context['variable'], context['point']
    
    try:
        numerator, denominator = fraction(expr)
    except:
        return None
    
    new_expr, t, t_direction = _substitute_to_zero(expr, var, point)
    new_num, new_den = fraction(new_expr)
    
    try:
        num_limit = limit(new_num, t, 0, t_direction)
        den_limit = limit(new_den, t, 0, t_direction)
        if (num_limit == 0 and den_limit == 0) or (abs(num_limit) == oo and abs(den_limit) == oo):
            return 'taylor_rational'
    except:
        pass
    return None