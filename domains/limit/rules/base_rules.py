# TODO 判断是否添加括号 / 获取极限参数的函数可以考虑放在工具库里, 共所有领域使用
# TODO 另外想一个好办法去提出 exp(f(x))-1, 现在写的太死，局限性很高

from typing import Any, Dict, Optional

from sympy import (
    Add, Expr, Limit, Mul, Pow, latex, Symbol,
    limit, nan, zoo, oo, UnevaluatedExpr, S, exp,
    Number, Integer, AccumBounds, simplify, Rational,
    sin, log
)


def _add_parentheses_if_needed(term):
    """为需要括号的表达式添加括号"""
    term_str = latex(term)
    # 数值负数需要括号
    if isinstance(term, Number) and term < 0:
        return f"\\left({term_str}\\right)"
    # 加法表达式需要括号（除非是单个项）
    if isinstance(term, Add) and len(term.args) > 1:
        return f"\\left({term_str}\\right)"
    # 乘法表达式：如果包含需要括号的项，则整个表达式需要括号
    if isinstance(term, Mul):
        for arg in term.args:
            if isinstance(arg, Add) or (isinstance(arg, Number) and arg < 0):
                return f"\\left({term_str}\\right)"
    return term_str


def _get_limit_args(context: Dict[str, Any]) -> tuple:
    """获取极限参数"""
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    dir_sup = '^{+}' if direction == '+' else '^{-}'
    return var, point, direction, dir_sup


def _check_limit_exists_oo(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """
    检查极限是否存在且为有限值(含无穷)
    """
    try:
        lim = Limit(expr, var, point, dir=direction).doit()
        return lim.is_finite or lim == oo or lim == -oo
    except:
        return False


def _check_function_tends_to_zero(f: Expr, var: Symbol, point: Any, direction: str) -> bool:
    try:
        lim = limit(f, var, point, dir=direction)
        return lim == 0
    except Exception:
        return False


def _check_limit_exists(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """
    检查极限是否存在且为有限值
    """
    try:
        lim = Limit(expr, var, point, dir=direction).doit()
        return lim.is_finite
    except:
        return False


def _is_indeterminate_form(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
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

    except:
        return True


def _check_combination_indeterminate(part1: Expr, part2: Expr, var: Symbol, point: Any, direction: str, operation: str) -> bool:
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
    except:
        return False


def _is_infinite(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查表达式极限是否为无穷"""
    try:
        lim = limit(expr, var, point, dir=direction)
        return lim in (oo, -oo)
    except:
        return False


def _is_zero(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查表达式极限是否为 0"""
    try:
        lim = limit(expr, var, point, dir=direction)
        return lim == 0
    except:
        return False


def _is_constant(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """检查表达式的极限是否为常数(即数值而非无穷或符号)"""
    try:
        lim = limit(expr, var, point, dir=direction)
        # 检查是否为数值常数（非无穷、非符号）
        return lim.is_real and not lim.has(oo, -oo, zoo)
    except:
        return False


def direct_substitution_rule(expr: Expr, context: Dict[str, Any]):
    """直接代入规则"""
    var, point, direction, dir_sup = _get_limit_args(context)

    result = limit(expr, var, point, dir=direction)
    # 判断是否跳过中间步骤（不展示代入过程）
    skip_intermediate = (
        expr == var
        or expr.is_number
        or result.is_infinite
    )
    expr_latex = _add_parentheses_if_needed(expr)
    point_latex = latex(point)
    result_latex = latex(result)
    limit_var = f"{var} \\to {point_latex}{dir_sup}"
    rule_body = f"\\lim_{{{limit_var}}} {expr_latex}"

    if skip_intermediate:
        full_rule = f"{rule_body} = {result_latex}"
    else:
        expr_subbed = expr.subs(var, UnevaluatedExpr(point))
        subbed_latex = _add_parentheses_if_needed(expr_subbed)
        full_rule = f"{rule_body} = {subbed_latex} = {result_latex}"

    return result, f"直接代入: ${full_rule}$"


def mul_split_rule(expr: Expr, context: Dict[str, Any]):
    """乘法拆分规则, 支持多个重要极限的提出"""
    var, point, direction, dir_sup = _get_limit_args(context)
    factors = expr.as_ordered_factors()

    # 提出 sin(f(x))/f(x) 或 f(x)/sin(f(x)), f(x)->0
    for i, factor in enumerate(factors):
        num, den = factor.as_numer_denom()
        # sin(f(x)) 位于分母的情况
        if isinstance(den, sin) and den.has(var) and _check_function_tends_to_zero(den, var, point, direction):
            sin_arg = den.args[0]
            sin_over_x = sin_arg / den
            rest_factors = factors[:i] + [1/sin_arg] + factors[i+1:]
            rest_part = Mul(*rest_factors)

            new_expr = Limit(sin_over_x, var, point, dir=direction) * \
                Limit(rest_part, var, point, dir=direction)

            rule_text = (
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\left({latex(expr)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\left({latex(sin_over_x)} \\cdot {_add_parentheses_if_needed(rest_part)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {latex(sin_over_x)} "
                f"\\cdot "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {_add_parentheses_if_needed(rest_part)}"
            )
            return new_expr, f"应用重要极限的乘法拆分规则: ${rule_text}$"

        if isinstance(num, sin) and num.has(var) and _check_function_tends_to_zero(num, var, point, direction):
            sin_arg = num.args[0]
            sin_over_x = num / sin_arg
            rest_factors = factors[:i] + [sin_arg] + factors[i+1:]
            rest_part = Mul(*rest_factors)

            new_expr = Limit(sin_over_x, var, point, dir=direction) * \
                Limit(rest_part, var, point, dir=direction)

            rule_text = (
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\left({latex(expr)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\left({latex(sin_over_x)} \\cdot {_add_parentheses_if_needed(rest_part)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {latex(sin_over_x)} "
                f"\\cdot "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {_add_parentheses_if_needed(rest_part)}"
            )
            return new_expr, f"应用重要极限的乘法拆分规则: ${rule_text}$"

    # 检测 ln(1+f(x))/f(x)
    for i, factor in enumerate(factors):
        numerator, den = factor.as_numer_denom()
        if isinstance(numerator, log):
            f = numerator.args[0] - 1
            if f.has(var) and _check_function_tends_to_zero(f, var, point, direction):
                rest_factors = factors[:i] + [f] + factors[i+1:]
                rest_part = Mul(*rest_factors)

                new_expr = Limit(factor / f, var, point, dir=direction) * \
                    Limit(rest_part, var, point, dir=direction)

                rule_text = (
                    f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                    f"\\left({latex(expr)}\\right) = "
                    f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                    f"\\left(\\frac{{\\ln(1+{latex(f)})}}{{{latex(f)}}} \\cdot {latex(rest_part)}\\right) = "
                    f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                    f"\\frac{{\\ln(1+{latex(f)})}}{{{latex(f)}}} "
                    f"\\cdot "
                    f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {latex(rest_part)}"
                )

                return new_expr, f"应用重要极限的乘法拆分规则: ${rule_text}$"
        # 位于分母
        if isinstance(den, log):
            f = den.args[0] - 1
            if f.has(var) and _check_function_tends_to_zero(f, var, point, direction):
                rest_factors = factors[:i] + [1/f] + factors[i+1:]
                rest_part = Mul(*rest_factors)

                new_expr = Limit(f * factor, var, point, dir=direction) * \
                    Limit(rest_part, var, point, dir=direction)

                rule_text = (
                    f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                    f"\\left({latex(expr)}\\right) = "
                    f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                    f"\\left(\\frac{{{latex(f)}}}{{\\ln(1+{latex(f)})}} \\cdot {latex(rest_part)}\\right) = "
                    f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                    f"\\frac{{{latex(f)}}}{{\\ln(1+{latex(f)})}}"
                    f"\\cdot "
                    f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {latex(rest_part)}"
                )

                return new_expr, f"应用重要极限的乘法拆分规则: ${rule_text}$"

    # 提出 (e^f(x) - 1)/f(x), f(x)->0
    for i, factor in enumerate(factors):
        numerator, den = factor.as_numer_denom()
        f = None
        if isinstance(numerator, Add) and len(numerator.args) == 2:
            try:
                # 通过捕获错误来判断是否有系数
                numerator.args[1].args[1]
                const_1 = numerator.args[1].args[0]
                const_2 = numerator.args[0]
                if const_1.has(var) or const_2.has(var):
                    continue
                # 提取公共常数, 凑重要极限
                if const_1 / const_2 == -1 and isinstance(numerator.args[1].args[1], exp):
                    f = numerator.args[1].args[1].args[0]
                    if not _check_function_tends_to_zero(f, var, point, direction):
                        continue
            except:
                try:
                    f = numerator.args[1].args[0]
                    other = numerator.args[0]
                    if other.has(var):
                        continue
                    if not f.has(var) and _check_function_tends_to_zero(f, var, point, direction) and not isinstance(numerator.args[1], exp):
                        continue
                except:
                    continue
            if not f:
                continue
            rest_factors = factors[:i] + [f] + factors[i+1:]
            rest_part = Mul(*rest_factors)

            new_expr = Limit(factor / f, var, point, dir=direction) * \
                Limit(rest_part, var, point, dir=direction)

            rule_text = (
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\left({latex(expr)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\left(\\frac{{{latex(numerator)}}}{{{latex(f)}}} \\cdot {_add_parentheses_if_needed(rest_part)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\frac{{{latex(numerator)}}}{{{latex(f)}}} "
                f"\\cdot "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {_add_parentheses_if_needed(rest_part)}"
            )
            return new_expr, f"应用重要极限的乘法拆分规则: ${rule_text}$"
        # 位于分母
        if isinstance(den, Add) and len(den.args) == 2:
            try:
                # 通过捕获错误来判断是否有系数
                den.args[1].args[1]
                const = den.args[1].args[0]
                if const.has(var):
                    continue
                # 提取公共常数, 凑重要极限
                if const / den.args[0] == -1 and isinstance(den.args[1].args[1], exp):
                    f = den.args[1].args[1].args[0]
                    if not _check_function_tends_to_zero(f, var, point, direction):
                        continue
            except:
                try:
                    f = den.args[1].args[0]
                    if not f.has(var) and _check_function_tends_to_zero(f, var, point, direction) or not isinstance(den.args[1], exp):
                        continue
                except:
                    continue
            if not f:
                continue
            rest_factors = factors[:i] + [1/f] + factors[i+1:]
            rest_part = Mul(*rest_factors)

            new_expr = Limit(f * factor, var, point, dir=direction) * \
                Limit(rest_part, var, point, dir=direction)

            rule_text = (
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}}"
                f"\\left({latex(expr)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}}"
                f"\\left(\\frac{{{latex(f)}}}{{{latex(den)}}} \\cdot {_add_parentheses_if_needed(rest_part)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}}"
                f"\\frac{{{latex(f)}}}{{{latex(den)}}}"
                f"\\cdot "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {_add_parentheses_if_needed(rest_part)}"
            )
            return new_expr, f"应用重要极限的乘法拆分规则: ${rule_text}$"

    # 提出 (f(x) + 1)**h(x), f(x) -> 0
    for i, factor in enumerate(factors):
        if not isinstance(factor, Pow):
            continue
        base, _exp = factor.as_base_exp()
        # 统一为 (f(x) + 1)**h(x) 形式处理
        inv_f = base - 1
        ratio = simplify(inv_f * _exp)
        if _check_function_tends_to_zero(inv_f, var, point, direction) and not ratio.has(var):
            rest_factors = factors[:i] + factors[i+1:]
            rest_part = Mul(*rest_factors)

            new_expr = Limit(factor, var, point, dir=direction) * \
                Limit(rest_part, var, point, dir=direction)

            rule_text = (
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\left({latex(expr)}\\right) = "
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} {latex(factor)}"
                f"\\cdot \\lim_{{{var} \\to {latex(point)}{dir_sup}}} {latex(rest_part)}"
            )
            return new_expr, f"应用重要极限的乘法拆分规则: ${rule_text}$"

    # 常规乘法拆分
    for i, factor in enumerate(factors):
        first_part = factor
        rest_factors = factors[:i] + factors[i+1:]
        if not rest_factors:
            continue
        rest_part = Mul(*rest_factors)

        if (_check_limit_exists(first_part, var, point, direction) and
            _check_limit_exists(rest_part, var, point, direction) and
                not _check_combination_indeterminate(first_part, rest_part, var, point, direction, 'mul')):

            first_limit = Limit(first_part, var, point, dir=direction)
            new_expr = first_limit * \
                Limit(rest_part, var, point, dir=direction)

            rule_text = (
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"\\left({latex(first_part)} \\cdot {latex(rest_part)}\\right) = "
                f"{latex(first_limit)} \\cdot \\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
                f"{latex(rest_part)}"
            )

            return new_expr, f"应用乘法拆分规则: ${rule_text}$"

    return None, None


def add_split_rule(expr: Expr, context: Dict[str, Any]):
    """加法拆分规则: 每次拆出一项, 前提是该项和剩余部分的极限都存在且拆分后不会产生不定式"""
    var, point, direction, dir_sup = _get_limit_args(context)
    terms = expr.as_ordered_terms()

    for i, _ in enumerate(terms):
        first_part = Add(*terms[:i+1])
        rest_terms = terms[i+1:] if i+1 < len(terms) else []
        rest_part = Add(*rest_terms) if rest_terms else S.Zero

        # 检查两个极限是否存在且拆分后不会产生不定式
        if (_check_limit_exists(first_part, var, point, direction) and
            _check_limit_exists(rest_part, var, point, direction) and
                not _check_combination_indeterminate(first_part, rest_part, var, point, direction, 'add')):

            # 计算第一个部分的极限
            first_limit = Limit(first_part, var, point, dir=direction)

            new_expr = first_limit + \
                Limit(rest_part, var, point, dir=direction)
            first_latex = _add_parentheses_if_needed(first_part)
            rest_latex = _add_parentheses_if_needed(rest_part)
            first_limit_latex = latex(first_limit)

            rule_text = (
                f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}}"
                f"\\left({first_latex} + {rest_latex}\\right) = "
                f"{first_limit_latex} + \\lim_{{{var} \\to {latex(point)}{dir_sup}}} {rest_latex}"
            )

            return new_expr, f"应用加法拆分规则: ${rule_text}$"

    return None, None


def div_split_rule(expr: Expr, context: Dict[str, Any]):
    """除法拆分规则: 分式的极限等于分子极限除以分母极限, 前提是分母极限不为零且拆分后不会产生不定式"""
    var, point, direction, dir_sup = _get_limit_args(context)

    # 获取分子和分母
    numerator, denominator = expr.as_numer_denom()

    # 构造新的表达式: Limit(numerator) / Limit(denominator)
    num_limit_expr = Limit(numerator, var, point, dir=direction)
    den_limit_expr = Limit(denominator, var, point, dir=direction)
    new_expr = num_limit_expr / den_limit_expr

    num_latex = _add_parentheses_if_needed(numerator)
    den_latex = _add_parentheses_if_needed(denominator)
    num_limit_latex = latex(num_limit_expr)
    den_limit_latex = latex(den_limit_expr)

    rule_text = (
        f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} "
        f"\\frac{{{num_latex}}}{{{den_latex}}} = "
        f"\\frac{{{num_limit_latex}}}{{{den_limit_latex}}}"
    )

    return new_expr, f"应用除法拆分规则: ${rule_text}$"


def _check_mul_split(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """
    检查表达式是否为乘法且存在一个因子，使得:
    该因子的极限存在(有限值或无穷);
    剩下的乘积的极限也存在(有限值或无穷);
    且拆分后不会产生不定式.
    同时检测 sin(x)/x 以及其他重要极限提出情况。
    """
    if not isinstance(expr, Mul):
        return False
    factors = expr.as_ordered_factors()
    # 检测 sin(f(x))/f(x), f(x)->0
    for factor in factors:
        num, den = factor.as_numer_denom()
        # 既检查分子也检查分母
        for part in (num, den):
            if isinstance(part, sin) and factor.has(var) and _check_function_tends_to_zero(part.args[0], var, point, direction):
                return True

    # 检测 ln(1+f(x)), f(x)->0
    for factor in factors:
        num, den = factor.as_numer_denom()
        # 既检查分子也检查分母
        for part in (num, den):
            if isinstance(part, log):
                f = part.args[0] - 1
                if f.has(var) and _check_function_tends_to_zero(f, var, point, direction):
                    return True

    # 检测 (e^f(x) - 1), f(x)->0
    for factor in factors:
        num, den = factor.as_numer_denom()
        # 既检查分子也检查分母
        for part in (num, den):
            if isinstance(part, Add) and len(part.args) == 2:
                try:
                    # 通过捕获错误来判断是否有系数
                    part.args[1].args[1]
                    const = part.args[1].args[0]
                    if const.has(var):
                        continue
                    # 提取公共常数, 凑重要极限
                    if const / part.args[0] == -1 and isinstance(part.args[1].args[1], exp):
                        f = part.args[1].args[1].args[0]
                        if _check_function_tends_to_zero(f, var, point, direction):
                            return True
                except:
                    # sympify 保证常数在前
                    if part.args[0] == -1 and isinstance(part.args[1], exp):
                        f = part.args[1].args[0]
                        if f.has(var) and _check_function_tends_to_zero(f, var, point, direction):
                            return True

    # 检测 (f(x) + 1)**h(x), f(x) -> 0
    for factor in factors:
        if not isinstance(factor, Pow):
            continue
        base, _exp = factor.as_base_exp()
        # 统一为 (f(x) + 1)**h(x) 形式处理
        inv_f = base - 1
        ratio = simplify(inv_f * _exp)
        if _check_function_tends_to_zero(inv_f, var, point, direction) and not ratio.has(var):
            return True

    # 常规乘法拆分检测
    for i, factor in enumerate(factors):
        first_part = factor
        rest_factors = factors[:i] + factors[i+1:]
        if not rest_factors:
            continue
        rest_part = Mul(*rest_factors)

        if (_check_limit_exists(first_part, var, point, direction) and
            _check_limit_exists(rest_part, var, point, direction) and
                not _check_combination_indeterminate(first_part, rest_part, var, point, direction, 'mul')):
            return True

    return False


def _check_add_split(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """
    检查表达式是否为加法且存在一项, 使得:
    该项的极限存在(有限值或无穷); 剩下的和的极限也存在(有限值或无穷)
    且拆分后不会产生不定式
    """
    if not isinstance(expr, Add):
        return False
    terms = expr.as_ordered_terms()
    for i, term in enumerate(terms):
        # 提出第 i 项
        first_part = term
        rest_terms = terms[:i] + terms[i+1:]
        rest_part = Add(*rest_terms) if rest_terms else S.Zero

        # 检查两个极限是否存在且拆分后不会产生不定式
        if (_check_limit_exists(first_part, var, point, direction) and
            _check_limit_exists(rest_part, var, point, direction) and
                not _check_combination_indeterminate(first_part, rest_part, var, point, direction, 'add')):
            return True
    return False


def _check_div_split(expr: Expr, var: Symbol, point: Any, direction: str) -> bool:
    """
    检查表达式是否为分式, 且满足:
    1. 分子极限存在(有限或无穷)
    2. 分母极限存在且不为零
    3. 不产生不定式 (0/0, oo/oo)
    """
    if not isinstance(expr, Mul) or not any(isinstance(arg, Pow) and arg.exp == -1 for arg in expr.args):
        return False
    numerator, denominator = expr.as_numer_denom()
    if denominator == S.One:
        return False
    # 检查分子极限是否存在
    if not _check_limit_exists_oo(numerator, var, point, direction):
        return False
    # 检查分母极限是否存在且不为零
    denom_limit = limit(denominator, var, point, dir=direction).doit()
    if not _check_limit_exists_oo(denominator, var, point, direction) or denom_limit == 0:
        return False
    # 检查是否为不定式
    num_limit = limit(numerator, var, point, dir=direction).doit()
    if (num_limit == 0 and denom_limit == 0) or \
            (num_limit in (oo, -oo) and denom_limit in (oo, -oo)):
        return False
    return True


def direct_substitution_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """
    检查是否可以应用直接代入法.
    如果代入后为不定式(如 0/0, oo/oo, 0·oo, oo-oo, 1^oo, 0^0, oo^0), 则返回 None.
    否则, 若极限存在（有限或确定无穷），返回 'direct_substitution'.
    """
    var, point, direction, _ = _get_limit_args(context)
    try:
        # 尝试直接代入表达式整体
        substituted_value = expr.subs(var, point)
        # 代入值后出现复数无穷的情况(比如 1/x(x->0)  时就会出现)
        if substituted_value.has(zoo):
            return 'direct_substitution'
        if substituted_value is nan:
            return None  # 显式 nan, 不能代入
        # 特殊情况：表达式本身就是变量或常数
        if expr.is_number or expr == var:
            return 'direct_substitution'
        # 排除不定式: 任一一部分均不可以是不定式
        factors = expr.as_ordered_factors()
        for factor in factors:
            if _is_indeterminate_form(factor, var, point, direction):
                return None
        # 检查极限是否存在
        lim_val = limit(expr, var, point, dir=direction)
        if lim_val.is_finite or lim_val in (oo, -oo):
            return 'direct_substitution'
        return None
    except Exception:
        return None


def mul_split_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """检查表达式是否为乘法, 且可以拆分为两个部分, 每部分极限存在（有限或无穷）且不会产生不定式"""
    var, point, direction, _ = _get_limit_args(context)
    if _check_mul_split(expr, var, point, direction):
        return 'mul_split'
    return None


def add_split_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """检查表达式是否为加法, 且可以拆分为两个部分, 每部分极限存在（有限或无穷）且不会产生不定式"""
    var, point, direction, _ = _get_limit_args(context)
    if _check_add_split(expr, var, point, direction):
        return 'add_split'
    return None


def div_split_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """检查表达式是否为分式且可以应用除法极限法则"""
    var, point, direction, _ = _get_limit_args(context)
    if _check_div_split(expr, var, point, direction):
        return 'div_split'
    return None


def const_inf_add_rule(expr: Expr, context: Dict[str, Any]):
    """处理 趋于常数(有界) +- 趋于无穷（所有无穷项符号一致）的情况"""
    var, point, direction, dir_sup = _get_limit_args(context)
    terms = expr.as_ordered_terms()

    inf_sign = None
    for term in terms:
        # 用第一个无穷记录正负号
        if _is_infinite(term, var, point, direction) and inf_sign is None:
            lim_val = limit(term, var, point, dir=direction)
            inf_sign = 1 if lim_val == oo else -1 if lim_val == -oo else None

    result = oo * inf_sign
    result_latex = latex(result)

    rule_text = f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} \\left({latex(expr)}\\right) = {result_latex}"
    explanation = f"应用\,趋于常数(有界)+-趋于无穷\, 规则(所有无穷项同号): ${rule_text}$"
    return result, explanation


def const_inf_add_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 趋于常数(有界) +- 趋于无穷（所有无穷项符号一致）的情况"""
    if not isinstance(expr, Add):
        return None
    var, point, direction, _ = _get_limit_args(context)
    terms = expr.as_ordered_terms()

    const_terms = []
    inf_terms = []
    inf_sign = None

    for term in terms:
        if _is_constant(term, var, point, direction):
            const_terms.append(term)
        elif _is_infinite(term, var, point, direction):
            inf_terms.append(term)
            lim_val = limit(term, var, point, dir=direction)
            term_sign = 1 if lim_val == oo else -1 if lim_val == -oo else None
            if term_sign is None:
                return None  # 非标准无穷，不匹配
            if inf_sign is None:
                inf_sign = term_sign
            elif inf_sign != term_sign:
                return None  # 符号冲突，不匹配
        else:
            return None  # 存在其他类型项，不匹配

    # 必须至少有一个无穷项，常数项可有可无
    if len(inf_terms) >= 1:
        return 'const_inf_add'

    return None


def const_inf_mul_rule(expr: Expr, context: Dict[str, Any]):
    """处理 (趋于非零常数) × (趋于无穷) 的情况"""
    var, point, direction, dir_sup = _get_limit_args(context)
    factors = expr.as_ordered_factors()

    const_factors = []
    inf_factors = []

    for factor in factors:
        if _is_constant(factor, var, point, direction):
            const_factors.append(factor)
        elif _is_infinite(factor, var, point, direction):
            inf_factors.append(factor)

    # 计算所有非零常数因子的乘积极限
    const_product = 1
    for f in const_factors:
        const_product *= limit(f, var, point, dir=direction)
    # 计算所有无穷因子的符号乘积
    inf_sign_product = 1
    for f in inf_factors:
        lim_val = limit(f, var, point, dir=direction)
        if lim_val == oo:
            inf_sign_product *= 1
        elif lim_val == -oo:
            inf_sign_product *= -1

    total_sign = 1 if (const_product * inf_sign_product) > 0 else -1
    result = oo * total_sign

    rule_text = f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} \\left({latex(expr)}\\right)= {latex(result)}"
    return result, f"应用\,趋于非零常数(有界)(可无)$\cdot$趋于无穷\,规则: ${rule_text}$"


def const_inf_mul_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 趋于非零常数(可无) × 趋于无穷 的情况"""
    if not isinstance(expr, Mul):
        return None
    numerator, denominator = expr.as_numer_denom()
    # 必须是纯粹的乘法, 不可是那种可以视作乘法的分式
    if denominator != 1:
        return None
    var, point, direction, _ = _get_limit_args(context)
    factors = expr.as_ordered_factors()

    has_inf = False
    for factor in factors:
        if _is_infinite(factor, var, point, direction):
            has_inf = True
        elif _is_constant(factor, var, point, direction):
            lim_val = limit(factor, var, point, dir=direction)
            # 如有振荡, 不可应用
            if isinstance(lim_val, AccumBounds):
                return None
            if lim_val == 0:
                return None

    if not has_inf:
        return None

    return 'const_inf_mul'


def const_inf_div_rule(expr: Expr, context: Dict[str, Any]):
    """处理 趋于常数(有界) / 趋于无穷 的情况"""
    var, point, direction, dir_sup = _get_limit_args(context)
    numerator, denominator = expr.as_numer_denom()

    const_val = limit(numerator, var, point, dir=direction)
    const_latex = latex(const_val)
    denom_latex = latex(denominator)

    rule_text = f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} \\left({latex(expr)}\\right) = 0"
    return Integer(0), f"应用\,趋于常数(有界)/趋于无穷\,规则: ${rule_text}$"


def const_inf_div_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 趋于常数(有界) / 趋于无穷 的情况"""
    var, point, direction, _ = _get_limit_args(context)
    numerator, denominator = expr.as_numer_denom()
    if denominator == 1:
        return None

    if _is_constant(numerator, var, point, direction) and _is_infinite(denominator, var, point, direction):
        return 'const_inf_div'
    return None


def const_zero_div_rule(expr: Expr, context: Dict[str, Any]):
    """处理 趋于常数(有界)(不为 0) / 趋于 0 的情况"""
    var, point, direction, dir_sup = _get_limit_args(context)
    numerator, denominator = expr.as_numer_denom()

    num_lim = limit(numerator, var, point, dir=direction).doit()
    sign = 1 if direction == '+' else -1
    # sympy 会把负号提给分子
    sign *= 1 if num_lim > 0 else -1
    result = oo * sign

    rule_text = (
        f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} \\left({latex(expr)}\\right) = {latex(result)}"
    )
    return result, f"应用\,趋于非零常数(有界)/趋于0\,规则(可能需要通分再观察): ${rule_text}$"


def const_zero_div_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配 趋于常数(有界)(不为 0) / 趋于 0 的情况"""
    var, point, direction, _ = _get_limit_args(context)
    numerator, denominator = expr.as_numer_denom()
    if denominator == 1:
        return None

    if _is_constant(numerator, var, point, direction):
        num_lim = limit(numerator, var, point, dir=direction)
        if num_lim != 0 and _is_zero(denominator, var, point, direction):
            return 'const_zero_div'
    return None


def small_o_add_rule(expr: Expr, context: Dict[str, Any]):
    """处理多个无穷小相加减的情况: 极限为 0"""
    var, point, direction, dir_sup = _get_limit_args(context)

    rule_text = f"\\lim_{{{var} \\to {latex(point)}{dir_sup}}} \\left({latex(expr)}\\right) = 0"
    return Integer(0), f"应用\,多个趋于\,0\,相加减\,规则: ${rule_text}$"


def small_o_add_matcher(expr: Expr, context: Dict[str, Any]) -> Optional[str]:
    """匹配多个无穷小相加减的情况: 所有项极限均为 0"""
    if not isinstance(expr, Add):
        return None

    var, point, direction, _ = _get_limit_args(context)
    terms = expr.as_ordered_terms()
    # 至少要有两项(否则就是单个无穷小，应由直接代入处理)
    if len(terms) < 2:
        return None
    for term in terms:
        if not _is_zero(term, var, point, direction):
            return None  # 存在非无穷小项，不匹配

    return 'small_o_add'


def has_sqrt(expr):
    # 判断表达式中是否存在平方根
    return any(isinstance(arg, Pow) and arg.exp == Rational(1, 2) for arg in expr.atoms(Pow))


def conjugate_rationalize_matcher(expr, context):
    """
    匹配形如 (sqrt(A) - sqrt(B)) / something 的情况
    """
    try:
        num, _ = expr.as_numer_denom()
    except Exception:
        return None
    if num.is_Add and len(num.args) == 2:
        a, b = num.args
        # 判断是否存在根号（幂为1/2）
        has_a = any(isinstance(arg, Pow) and arg.exp == Rational(1, 2)
                    for arg in a.atoms(Pow))
        has_b = any(isinstance(arg, Pow) and arg.exp == Rational(1, 2)
                    for arg in b.atoms(Pow))
        if has_a or has_b:
            return 'conjugate_rationalize'
    return None


def conjugate_rationalize_rule(expr, context):
    """
    对分子为 sqrt(A)-sqrt(B) 的分式进行有理化
    """
    var, point, direction, _ = _get_limit_args(context)
    num, den = expr.as_numer_denom()

    a, b = num.args
    conj = a - b  # 共轭
    new_num = simplify(a**2 - b**2)
    new_den = simplify(den * conj)
    new_expr = simplify(new_num / new_den)
    new_limit = Limit(new_expr, var, point, dir=direction)

    explanation = (
        f"$分子含有根号差，乘以共轭\,{latex(conj)}\,进行有理化:"
        f"\\lim_{{{var} \\to {latex(point)}{direction}}}{latex(expr)}="
        f"\\lim_{{{var} \\to {latex(point)}{direction}}}"
        f"\\frac{{({latex(num)})({latex(conj)})}}{{{latex(den)}({latex(conj)})}}="
        f"\\lim_{{{var} \\to {latex(point)}{direction}}}"
        f"\\frac{{{latex(new_num)}}}{{{latex(new_den)}}}$"
    )

    return new_limit, explanation
