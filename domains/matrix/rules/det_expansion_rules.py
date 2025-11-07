from typing import Any, Dict, Tuple, Optional, List
from sympy import Matrix, Determinant, Expr, Mul, Add, Pow


def laplace_expansion_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """拉普拉斯展开规则, 智能地选择展开行/列"""
    matrix = det_expr.args[0]
    n = matrix.rows

    # 如果矩阵很小, 直接计算
    if n <= 2:
        result = matrix.det()
        return result, rf"$直接计算\,{n} \times {n}\,矩阵的行列式$"

    # 寻找最优展开行/列：考虑零元素数量和元素复杂度
    best_score = -1
    best_index = 0
    best_type = 'row'  # 'row' or 'col'

    # 评估每行的得分
    for i in range(n):
        row = matrix.row(i)
        zero_count = sum(1 for elem in row if elem == 0)
        # 计算复杂度：复杂表达式（Mul, Add, Pow）的数量
        complexity = sum(1 for elem in row if isinstance(
            elem, (Mul, Add, Pow)) and elem != 0)
        score = zero_count * 2 - complexity  # 零元素权重更高

        if score > best_score:
            best_score = score
            best_index = i
            best_type = 'row'

    # 评估每列的得分
    for j in range(n):
        col = matrix.col(j)
        zero_count = sum(1 for elem in col if elem == 0)
        complexity = sum(1 for elem in col if isinstance(
            elem, (Mul, Add, Pow)) and elem != 0)
        score = zero_count * 2 - complexity

        if score > best_score:
            best_score = score
            best_index = j
            best_type = 'col'

    # 执行展开
    if best_type == 'row':
        expansion_terms = []
        row_zero_count = sum(1 for elem in matrix.row(best_index) if elem == 0)
        for j in range(n):
            element = matrix[best_index, j]
            if element == 0:
                continue

            minor_matrix = matrix.minor_submatrix(best_index, j)
            cofactor = (-1) ** (best_index + j) * \
                element * Determinant(minor_matrix)
            expansion_terms.append(cofactor)

        result = Add(*expansion_terms)
        explanation = f"$按第\,{best_index+1}\,行展开(该行有\,{row_zero_count}\,个零元素)$"

    else:  # best_type == 'col'
        expansion_terms = []
        col_zero_count = sum(1 for elem in matrix.col(best_index) if elem == 0)
        for i in range(n):
            element = matrix[i, best_index]
            if element == 0:
                continue

            minor_matrix = matrix.minor_submatrix(i, best_index)
            cofactor = (-1) ** (i + best_index) * \
                element * Determinant(minor_matrix)
            expansion_terms.append(cofactor)

        result = Add(*expansion_terms)
        explanation = f"$按第\,{best_index+1}\,列展开(该列有\,{col_zero_count}\,个零元素)$"

    return result, explanation


def laplace_expansion_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配需要拉普拉斯展开的情况"""
    return 'laplace_expansion'
