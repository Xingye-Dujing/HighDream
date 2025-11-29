from sympy import I, Matrix, exp, latex, pi, prod, simplify

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn


def vandermonde_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Apply the formula to a Vandermonde determinant."""
    n = matrix.rows

    # In a standard Vandermonde matrix,
    #   row 0: [1, 1, ..., 1]
    #   row 1: [x_0 x_1, ..., x_{n-1}]
    # So we take row 1 as the base vector of nodes.
    base_elements = matrix.row(1)

    result = 1
    for i in range(n):
        for j in range(i + 1, n):
            result *= (base_elements[j] - base_elements[i])

    explanation = rf"范德蒙德行列式: $\prod_{{1 \leq i \lt j \leq {n}}} (a_j - a_i)$"
    return result, explanation


def circulant_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Apply the formula to a circulant determinant.

    A circulant determinant is fully determined by its first row [c_0, c_1, ..., c_{n-1}],
    with each subsequent row being a right cyclic shift of the row above.

    The determinant of an n x n circulant determinant is given by:

        det(C) = prod_{k=0}^{n-1} ( c_0 + c_1*omega_k + c_2*omega_k^2 ... c_{n-1}*omega_k^{n-1} )

    where omega_k = e^{2pi*i*k/n} are the n-th roots of unity.
    """

    n = matrix.rows

    # 3x3 with explicit formula
    if n == 3:
        a, b, c = matrix.row(0)
        result = a**3 + b**3 + c**3 - 3*a*b*c
        explanation = f"3 阶循环矩阵行列式: $a^3 + b^3 + c^3 - 3abc$"
    # Generic case
    else:
        first_row = list(matrix.row(0))
        terms = []
        for k in range(n):
            omega = exp(2*pi*I*k/n)
            term = sum(first_row[j] * omega**j for j in range(n))
            terms.append(term)
        result = simplify(prod(terms))
        explanation = (
            f"{n}$\\times${n} 循环矩阵行列式: "
            f"$\\prod_{{k=0}}^{{n-1}} (c_0 + c_1\\omega_k + c_2\\omega_k^2 + ... + c_{{n-1}}\\omega_k^{{n-1}})$."
        )

    return result, explanation


def symmetric_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Apply the formula to 2x2 symmetric determinant.

    A symmetric determinant satisfies A = A^T. For a 2x2 symmetric determinant:

        A = [[a, b], [b, c]]

    the determinant is given by the closed-form expression:

        det(A) = ac - b^2.

    This rule only applies to 2x2 symmetric determinant.
    """
    n = matrix.rows

    # For a 2x2 symmetric determinant
    if n == 2:
        a, b, c = matrix[0, 0], matrix[0, 1], matrix[1, 1]
        result = a*c - b**2
        explanation = f"2 阶对称矩阵: $ac - b^2 = {latex(result)}$"
        return result, explanation

    return None


def vandermonde_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Match whether a determinant is a Vandermonde determinant.

    This matcher verifies that:
      - The order at least 3x3 (smaller cases are handled by simpler rules).
      - Row 0 is all ones.
      - For each row i (starting from 0), the entries equal base[j]**i,
        where base = [x_0, ..., x_{n-1}] is taken from row 1.

    Note: This function assumes standard Vandermonde layout with powers increasing downward.
    It does not support transposed or generalized variants.
    """

    n = matrix.rows
    if n <= 2:
        # Rule system handles 1x1 and 2x2 via direct computation;
        # Vandermonde formula is overkill for n <= 2.
        return None

    # Check first row: must be all 1s
    first_row = matrix.row(0)
    if not all(elem == 1 for elem in first_row):
        return None

    # Extract base elements from second row (row index 1)
    base_elements = list(matrix.row(1))

    # Validate remaining rows: row i should be [x_j**i for x_j in base_elements]
    for i in range(2, n):
        expected_row = [elem**i for elem in base_elements]
        actual_row = list(matrix.row(i))

        for exp_val, act_val in zip(expected_row, actual_row):
            if exp_val != act_val:
                return None

    return 'vandermonde'


def circulant_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Match whether a determinant a circulant determinant.

    A circulant determinant is fully determined by its first row [c_0, c_1, ..., c_{n-1}],
    with each subsequent row being a right cyclic shift of the row above. Equivalently,
    the element at position (i, j) satisfies:

        A_{i,j} = c_{(j - i) mod n}

    This matcher verifies that every row i equals the first row cyclically shifted right by i positions.
    """
    n = matrix.rows

    first_row = list(matrix.row(0))

    # Check each subsequent row
    for i in range(1, n):
        # Expected: right cyclic shift by i to equivalent to taking
        # element at column j from first_row[(j - i) mod n]
        expected_row = [first_row[(j - i) % n] for j in range(n)]
        actual_row = list(matrix.row(i))

        for exp_elem, act_elem in zip(expected_row, actual_row):
            if exp_elem != act_elem:
                return None

    return 'circulant'


def symmetric_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Match whether a determinant is symmetric (i.e., equal to its transpose).

    This matcher uses SymPy's built-in is_symmetric() method.
    """

    if matrix.is_symmetric():
        return 'symmetric'
    return None
