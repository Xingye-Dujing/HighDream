from typing import Any, Dict, Tuple
from sympy import (
    Expr, Symbol, factorial, sin, cos, exp, log,
    latex, Add, Mul, Pow, series,
)


def const_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """常数函数的泰勒展开规则"""
    point = context['point']
    order = context['order']
    return expr, f"常数函数的泰勒展开是其自身: ${latex(expr)}$"


def var_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """变量的泰勒展开规则"""
    var = context['variable']
    point = context['point']
    order = context['order']

    if expr == var:
        # 变量在点point处的泰勒展开: (x - point)
        taylor_expr = (var - point)
        return taylor_expr, f"变量在点${latex(point)}$处的泰勒展开: ${latex(var)} - {latex(point)}$"
    return None, ""


def sin_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """正弦函数的泰勒展开规则"""
    var = context['variable']
    point = context['point']
    order = context['order']

    if isinstance(expr, sin) and expr.args[0] == var:
        # sin(x)在0处的泰勒展开: x - x^3/3! + x^5/5! - ...
        terms = []
        explanation_terms = []
        for n in range(0, order + 1):
            if n % 2 == 1:  # 只保留奇数项
                sign = (-1)**((n-1)//2)
                term = sign * var**n / factorial(n)
                terms.append(term)
                if n == 1:
                    explanation_terms.append(f"{latex(var)}")
                else:
                    explanation_terms.append(
                        f"\\frac{{{(-1)**((n-1)//2)}}}{{{n}!}}{latex(var)}^{{{n}}}")

        taylor_expr = Add(*terms)
        explanation = " + ".join(explanation_terms)
        return taylor_expr, f"正弦函数在0处的泰勒展开: $\\sin({latex(var)}) = {explanation} + O({latex(var)}^{{{order+1}}})$"
    return None, ""


def cos_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """余弦函数的泰勒展开规则"""
    var = context['variable']
    point = context['point']
    order = context['order']

    if isinstance(expr, cos) and expr.args[0] == var:
        # cos(x)在0处的泰勒展开: 1 - x^2/2! + x^4/4! - ...
        terms = []
        explanation_terms = []
        for n in range(0, order + 1):
            if n % 2 == 0:  # 只保留偶数项
                sign = (-1)**(n//2)
                term = sign * var**n / factorial(n)
                terms.append(term)
                if n == 0:
                    explanation_terms.append("1")
                else:
                    explanation_terms.append(
                        f"\\frac{{{(-1)**(n//2)}}}{{{n}!}}{latex(var)}^{{{n}}}")

        taylor_expr = Add(*terms)
        explanation = " + ".join(explanation_terms)
        return taylor_expr, f"余弦函数在0处的泰勒展开: $\\cos({latex(var)}) = {explanation} + O({latex(var)}^{{{order+1}}})$"
    return None, ""


def exp_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """指数函数的泰勒展开规则"""
    var = context['variable']
    point = context['point']
    order = context['order']

    if isinstance(expr, exp) and expr.args[0] == var:
        # exp(x)在0处的泰勒展开: 1 + x + x^2/2! + x^3/3! + ...
        terms = []
        explanation_terms = []
        for n in range(0, order + 1):
            term = var**n / factorial(n)
            terms.append(term)
            if n == 0:
                explanation_terms.append("1")
            elif n == 1:
                explanation_terms.append(f"{latex(var)}")
            else:
                explanation_terms.append(
                    f"\\frac{{{latex(var)}^{{{n}}}}}{{{n}!}}")

        taylor_expr = Add(*terms)
        explanation = " + ".join(explanation_terms)
        return taylor_expr, f"指数函数在0处的泰勒展开: $e^{{{latex(var)}}} = {explanation} + O({latex(var)}^{{{order+1}}})$"
    return None, ""


def log_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """对数函数的泰勒展开规则"""
    var = context['variable']
    point = context['point']
    order = context['order']

    if isinstance(expr, log) and expr.args[0] == var and point == 1:
        # ln(x)在1处的泰勒展开: (x-1) - (x-1)^2/2 + (x-1)^3/3 - ...
        terms = []
        explanation_terms = []
        for n in range(1, order + 1):
            sign = (-1)**(n+1)
            term = sign * (var - 1)**n / n
            terms.append(term)
            if n == 1:
                explanation_terms.append(f"({latex(var)} - 1)")
            else:
                explanation_terms.append(
                    f"\\frac{{{(-1)**(n+1)}}}{{{n}}}({latex(var)} - 1)^{{{n}}}")

        taylor_expr = Add(*terms)
        explanation = " + ".join(explanation_terms)
        return taylor_expr, f"自然对数在1处的泰勒展开: $\\ln({latex(var)}) = {explanation} + O(({latex(var)}-1)^{{{order+1}}})$"
    return None, ""


def add_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """加法泰勒展开规则"""
    var = context['variable']
    point = context['point']
    order = context['order']

    if isinstance(expr, Add):
        # 和的泰勒展开等于泰勒展开的和
        taylor_terms = []
        explanations = []
        for term in expr.args:
            # 递归计算每个项的泰勒展开
            term_taylor, term_explanation = _apply_taylor_rule(term, context)
            taylor_terms.append(term_taylor)
            explanations.append(f"({term_explanation})")

        taylor_expr = Add(*taylor_terms)
        explanation = " + ".join(
            [f"泰勒展开({latex(term)})" for term in expr.args])
        return taylor_expr, f"应用加法规则: 泰勒展开({latex(expr)}) = {explanation}"
    return None, ""


def mul_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """乘法泰勒展开规则"""
    if isinstance(expr, Mul):
        # 乘积的泰勒展开需要将各个因子的泰勒展开相乘，并截断高阶项
        factors = expr.args
        if len(factors) == 2:
            u, v = factors
            # 递归计算 u 和 v 的泰勒展开
            u_taylor, u_explanation = _apply_taylor_rule(u, context)
            v_taylor, v_explanation = _apply_taylor_rule(v, context)

            # 相乘并截断高阶项
            product = u_taylor * v_taylor
            # 移除高阶项 (O(x^n))
            product = product.removeO() if hasattr(product, 'removeO') else product

            return product, f"应用乘法规则: 泰勒展开$({latex(u)} \\cdot {latex(v)}) = $泰勒展开$({latex(u)}) \\cdot $泰勒展开$({latex(v)})$"
    return None, ""


def pow_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """幂函数泰勒展开规则"""
    var = context['variable']
    point = context['point']
    order = context['order']

    if isinstance(expr, Pow):
        base, exponent = expr.args
        if base == var and exponent.is_constant():
            # x^n 在0处的泰勒展开就是x^n本身
            return expr, f"幂函数的泰勒展开: ${latex(expr)}$"
        elif exponent == -1 and base == var:
            # 1/x 在1处的泰勒展开
            if point == 1:
                # 1/x = 1 - (x-1) + (x-1)^2 - (x-1)^3 + ...
                terms = []
                for n in range(0, order + 1):
                    term = (-1)**n * (var - 1)**n
                    terms.append(term)

                taylor_expr = Add(*terms)
                return taylor_expr, f"倒数函数在1处的泰勒展开: $\\frac{{1}}{{{latex(var)}}} = 1 - ({latex(var)}-1) + ({latex(var)}-1)^2 - \\cdots$"
    return None, ""


def composite_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    """复合函数泰勒展开规则"""
    var = context['variable']
    point = context['point']
    order = context['order']

    # 检查是否是复合函数 f(g(x))
    if hasattr(expr, 'func') and len(expr.args) == 1:
        inner_expr = expr.args[0]
        if inner_expr != var and inner_expr.has(var):
            # 这是一个复合函数
            outer_func = expr.func
            inner_point = inner_expr.subs(var, point)

            # 先计算内函数的泰勒展开
            inner_context = context.copy()
            inner_context['order'] = order
            inner_taylor, inner_explanation = _apply_taylor_rule(
                inner_expr, inner_context)

            # 计算外函数在内函数展开点处的泰勒展开
            outer_context = {
                'variable': Symbol('u'),
                'point': inner_point,
                'order': order
            }
            outer_taylor, outer_explanation = _apply_taylor_rule(
                outer_func(Symbol('u')), outer_context)

            # 将内函数的泰勒展开代入外函数的泰勒展开
            composite_taylor = outer_taylor.subs(Symbol('u'), inner_taylor)
            # 展开并移除高阶项
            composite_taylor = composite_taylor.expand().removeO() if hasattr(
                composite_taylor, 'removeO') else composite_taylor.expand()

            return composite_taylor, f"应用复合函数规则: 先计算内函数泰勒展开，再代入外函数泰勒展开"
    return None, ""


def const_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """常数匹配器"""
    if expr.is_constant():
        return 'const'
    return None


def var_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """变量匹配器"""
    var = context['variable']
    if expr == var:
        return 'var'
    return None


def sin_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """正弦函数匹配器"""
    if isinstance(expr, sin):
        return 'sin'
    return None


def cos_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """余弦函数匹配器"""
    if isinstance(expr, cos):
        return 'cos'
    return None


def exp_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """指数函数匹配器"""
    if isinstance(expr, exp):
        return 'exp'
    return None


def log_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """对数函数匹配器"""
    if isinstance(expr, log):
        return 'log'
    return None


def add_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """加法匹配器"""
    if isinstance(expr, Add):
        return 'add'
    return None


def mul_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """乘法匹配器"""
    if isinstance(expr, Mul):
        return 'mul'
    return None


def pow_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """幂函数匹配器"""
    if isinstance(expr, Pow):
        return 'pow'
    return None


def composite_matcher(expr: Expr, context: Dict[str, Any]) -> str:
    """复合函数匹配器"""
    if hasattr(expr, 'func') and len(expr.args) == 1:
        inner_expr = expr.args[0]
        var = context['variable']
        if inner_expr != var and inner_expr.has(var):
            return 'composite'
    return None


# 辅助函数
def _apply_taylor_rule(expr: Expr, context: Dict[str, Any]) -> Tuple[Expr, str]:
    var = context['variable']
    point = context['point']
    order = context['order']
    taylor_series = series(expr, var, point, order + 1).removeO()
    return taylor_series, f"使用 SymPy 计算泰勒展开: ${latex(taylor_series)} + O(({latex(var)}-{latex(point)})^{{{order+1}}})$"
