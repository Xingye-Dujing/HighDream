from sympy import Add, Determinant, Matrix, Mul, Pow

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn


def laplace_expansion_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Laplace expansion rule for determinant evaluation.

    Performs cofactor expansion along the row or column that maximizes a heuristic
    score based on the number of zero entries (which eliminate terms) and the
    symbolic complexity of nonzero entries. Zero elements are weighted more heavily
    than complex expressions (e.g., sums, products, powers), encouraging expansion
    along sparse and simple lines.

    For matrices of size <= 2x2, the determinant is computed directly.
    """
    n = matrix.rows

    # Direct computation for small matrices
    if n <= 2:
        result = matrix.det()
        return result, rf"直接计算 ${n} \times {n}$ 矩阵"

    best_score = -float('inf')
    best_index = 0
    best_type = 'row'  # 'row' or 'col'

    # Evaluate rows
    for i in range(n):
        row = matrix.row(i)
        zero_count = sum(1 for elem in row if elem == 0)
        complexity = sum(
            1 for elem in row
            if elem != 0 and isinstance(elem, (Add, Mul, Pow))
        )
        score = 2 * zero_count - complexity
        if score > best_score:
            best_score = score
            best_index = i
            best_type = 'row'

    # Evaluate columns
    for j in range(n):
        col = matrix.col(j)
        zero_count = sum(1 for elem in col if elem == 0)
        complexity = sum(
            1 for elem in col
            if elem != 0 and isinstance(elem, (Add, Mul, Pow))
        )
        score = 2 * zero_count - complexity
        if score > best_score:
            best_score = score
            best_index = j
            best_type = 'col'

    # Perform expansion
    if best_type == 'row':
        expansion_terms = []
        row_zero_count = sum(1 for elem in matrix.row(best_index) if elem == 0)
        for j in range(n):
            element = matrix[best_index, j]
            if element == 0:
                continue
            minor = matrix.minor_submatrix(best_index, j)
            cofactor = (-1) ** (best_index + j) * element * Determinant(minor)
            expansion_terms.append(cofactor)

        result = Add(*expansion_terms)
        explanation = rf"$沿第 {best_index + 1} 行展开 (包含 {row_zero_count} 个零元素)$"

    else:  # column expansion
        expansion_terms = []
        col_zero_count = sum(1 for elem in matrix.col(best_index) if elem == 0)
        for i in range(n):
            element = matrix[i, best_index]
            if element == 0:
                continue
            minor = matrix.minor_submatrix(i, best_index)
            cofactor = (-1) ** (i + best_index) * element * Determinant(minor)
            expansion_terms.append(cofactor)

        result = Add(*expansion_terms)
        explanation = rf"$沿第 {best_index + 1} 列展开 (包含 {col_zero_count} 个零元素)$"

    return result, explanation


def laplace_expansion_matcher(_matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matcher for Laplace expansion applicability.

    This matcher assumes that Laplace expansion is always a fallback strategy
    for symbolic determinant evaluation when no simpler rule applies (e.g.,
    triangular form, duplicate rows, etc.). It unconditionally signals that
    the rule may be applied.
    """
    return 'laplace_expansion'
