from typing import Any, Dict, Tuple, Optional
from sympy import Determinant, Expr, latex, Matrix


def vandermonde_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """范德蒙德行列式规则"""
    matrix = det_expr.args[0]
    n = matrix.rows

    # 提取第二行作为基元素
    base_elements = matrix.row(1)

    # 范德蒙德行列式公式: ∏_{1≤i<j≤n} (a_j - a_i)
    result = 1
    for i in range(n):
        for j in range(i + 1, n):
            result *= (base_elements[j] - base_elements[i])

    explanation = rf"范德蒙德行列式: $\prod_{{1 \leq i \lt j \leq {n}}} (a_j - a_i)$"
    return result, explanation


def circulant_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """循环矩阵规则"""
    matrix = det_expr.args[0]
    n = matrix.rows

    # 检查是否为循环矩阵
    first_row = matrix.row(0)
    for i in range(1, n):
        expected_row = [first_row[(j - i) % n] for j in range(n)]
        actual_row = matrix.row(i)
        if expected_row != actual_row:
            return None

    # 循环矩阵的行列式公式涉及复数根，这里简化处理
    if n == 3:
        a, b, c = first_row
        result = a**3 + b**3 + c**3 - 3*a*b*c
        explanation = f"3阶循环矩阵行列式: $a^3 + b^3 + c^3 - 3abc$"
        return result, explanation

    return None


def symmetric_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """对称矩阵的特殊处理"""
    matrix = det_expr.args[0]

    if not matrix.is_symmetric():
        return None

    n = matrix.rows

    # 对于2x2对称矩阵
    if n == 2:
        a, b, c = matrix[0, 0], matrix[0, 1], matrix[1, 1]
        result = a*c - b**2
        explanation = f"2 阶对称矩阵: $ac - b^2 = {latex(result)}$"
        return result, explanation

    # 检查是否可以对角化或分块
    return None


def vandermonde_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    matrix = det_expr.args[0]
    n = matrix.rows

    # 检查是否为范德蒙德形式: 第 i 行为 [1, a_i, a_i^2, ..., a_i^(n-1)]
    # 至少要有两行, 但两行时我们直接使用对角线去求，不需要范德蒙德的结论
    if n <= 2:
        return None

    # 检查第一行是否全为 1
    first_row = matrix.row(0)
    if not all(element == 1 for element in first_row):
        return None

    # 提取第二行作为基元素
    base_elements = matrix.row(1)

    # 棜查后续行是否为基元素的幂
    for i in range(2, n):
        expected_row = []
        for elem in base_elements:
            expected_row.append(elem**i)

        actual_row = matrix.row(i)

        # 一定要先使用 list 转换成相同的数据类型再比较
        if list(expected_row) != list(actual_row):
            return None

    return 'vandermonde'


def circulant_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    matrix = det_expr.args[0]
    n = matrix.rows

    first_row = matrix.row(0)
    for i in range(1, n):
        expected_row = [first_row[(j - i) % n] for j in range(n)]
        actual_row = matrix.row(i)
        if expected_row != actual_row:
            return None
    return 'circulant'


def symmetric_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    matrix = det_expr.args[0]
    if matrix.is_symmetric():
        return 'symmetric'
    return None
