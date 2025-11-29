from sympy import Determinant, Integer, Matrix, Mul, gcd, latex

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn


def is_upper_triangular(matrix: Matrix):
    """Check whether a matrix is upper triangular.

    A matrix is upper triangular if all entries below the main diagonal are zero.
    """

    rows, _ = matrix.shape
    for i in range(rows):
        for j in range(i):
            if matrix[i, j] != 0:
                return False
    return True


def is_lower_triangular(matrix: Matrix):
    """Check whether a matrix is lower triangular.

    A matrix is lower triangular if all entries above the main diagonal are zero.
    """

    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            if matrix[i, j] != 0:
                return False
    return True


def zero_row_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Zero-row rule for determinant evaluation.

    If the input matrix contains any row consisting entirely of zeros,
    its determinant is zero.
    """

    rows = matrix.tolist()
    for i, row in enumerate(rows):
        if all(element == 0 for element in row):
            return Integer(0), f"第 {i+1} 行为零行, 行列式为 0"
    return None


def zero_column_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Zero-col rule for determinant evaluation.

    If the input matrix contains any xol consisting entirely of zeros,
    its determinant is zero.
    """

    # Check columns by inspecting the transpose of the matrix.
    cols = matrix.T.tolist()
    for i, col in enumerate(cols):
        if all(element == 0 for element in col):
            return Integer(0), f"第 {i+1} 列为零列, 行列式为 0"
    return None


def duplicate_row_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Duplicate-row rule for determinant evaluation.

    If the input matrix contains two identical rows, its determinant is zero.
    This function compares all pairs of rows and returns zero with an
    explanatory message if a duplicate pair is detected.
    """

    rows = matrix.tolist()
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i] == rows[j]:
                return Integer(0), f"第 {i+1} 行和第 {j+1} 行相同, 行列式为 0"
    return None


def duplicate_column_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Duplicate-col rule for determinant evaluation.

    If the input matrix contains two identical cols, its determinant is zero.
    This function compares all pairs of cols and returns zero with an
    explanatory message if a duplicate pair is detected.
    """

    # Check columns by inspecting the transpose of the matrix.
    cols = matrix.T.tolist()
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            if cols[i] == cols[j]:
                return Integer(0), f"第 {i+1} 列和第 {j+1} 列相同, 行列式为 0"
    return None


def diagonal_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Diagonal rule for determinant evaluation.

    For a diagonal matrix, the determinant is equal to the product of its
    diagonal entries.
    """
    diag_elements = [matrix[i, i] for i in range(matrix.rows)]
    product = Mul(*diag_elements)
    return product, f"对角矩阵的行列式等于对角线元素的乘积: ${latex(product)}$"


def triangular_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Triangular rule for determinant evaluation.

    The determinant of a triangular matrix (either upper or lower) is equal to
    the product of its diagonal entries. This function checks whether the input
    matrix is upper or lower triangular and, if so, returns the product of the
    diagonal elements along with an explanatory message.
    """

    if is_upper_triangular(matrix):
        diag_elements = [matrix[i, i] for i in range(matrix.rows)]
        product = Mul(*diag_elements)
        return product, f"上三角矩阵的行列式等于对角线元素的乘积: ${latex(product)}$"
    if is_lower_triangular(matrix):
        diag_elements = [matrix[i, i] for i in range(matrix.rows)]
        product = Mul(*diag_elements)
        return product, f"下三角矩阵的行列式等于对角线元素的乘积: ${latex(product)}$"
    return None


def scalar_multiple_row_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Scalar multiple row rule for determinant simplification.

    If a row of the matrix has a nontrivial common factor among its nonzero
    entries, that factor can be factored out of the determinant. This function
    scans each row for such a common scalar factor (greater than 1 in absolute
    value or non-unit in symbolic terms), constructs a new matrix with that row
    divided by the factor, and returns the scaled determinant expression.

    Note: This rule does not compute the full determinant-it returns an
    equivalent expression common_factor*det(new_matrix), which may be
    further processed by other rules.
    """
    n = matrix.rows

    for i in range(n):
        row = matrix.row(i)
        non_zero_elements = [elem for elem in row if elem != 0]
        if len(non_zero_elements) <= 1:
            continue

        try:
            # Compute GCD of all nonzero elements in the row
            common_factor = non_zero_elements[0]
            for elem in non_zero_elements[1:]:
                common_factor = gcd(common_factor, elem)

            # Skip if the GCD is a unit (e.g., ±1 or symbolic unit)
            if common_factor in (1, -1):
                continue

            # Construct new matrix with the i-th row divided by the common factor
            new_rows = []
            for row_idx in range(n):
                if row_idx == i:
                    new_row = [
                        elem / common_factor for elem in matrix.row(row_idx)]
                else:
                    new_row = list(matrix.row(row_idx))
                new_rows.append(new_row)

            new_matrix = Matrix(new_rows)
            result_expr = common_factor * Determinant(new_matrix)
            return result_expr, f"$提取公因子 {latex(common_factor)} 从行 {i + 1}$"

        except Exception:
            pass

    return None


def scalar_multiple_column_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Scalar multiple column rule for determinant simplification.

    If a column of the matrix has a nontrivial common factor among its nonzero
    entries, that factor can be factored out of the determinant. This function
    scans each column for such a common scalar factor (greater than 1 in absolute
    value or non-unit in symbolic terms), constructs a new matrix with that column
    divided by the factor, and returns an expression representing
    common_factor*det(new_matrix).
    """
    n = matrix.cols

    for j in range(n):
        col = matrix.col(j)
        non_zero_elements = [elem for elem in col if elem != 0]
        if len(non_zero_elements) <= 1:
            continue

        try:
            # Compute GCD of all nonzero elements in the column
            common_factor = non_zero_elements[0]
            for elem in non_zero_elements[1:]:
                common_factor = gcd(common_factor, elem)

            # Skip if the GCD is a unit (e.g., ±1)
            if common_factor in (1, -1):
                continue

            # Build new matrix with the j-th column divided by the common factor
            new_rows = []
            for i in range(matrix.rows):
                new_row = []
                for k in range(matrix.cols):
                    if k == j:
                        new_row.append(matrix[i, k] / common_factor)
                    else:
                        new_row.append(matrix[i, k])
                new_rows.append(new_row)

            new_matrix = Matrix(new_rows)
            result_expr = common_factor * Determinant(new_matrix)
            return result_expr, f"$提取公因子 {latex(common_factor)} 从列 {j + 1}$"

        except Exception:
            continue

    return None


def linear_combination_rule(matrix: Matrix, _context: RuleContext) -> RuleFunctionReturn:
    """Linear combination rule for determinant-preserving simplification.

    This rule searches for an elementary row or column operation of the form
    R_i <- R_i + c*R_j or C_i <- C_i + c*C_j that maximizes the number of
    newly introduced zero entries in the matrix. Such operations do not change
    the value of the determinant and are useful for simplifying subsequent
    evaluation (e.g., cofactor expansion).

    The function evaluates all valid pairwise combinations and selects the
    transformation that yields the greatest increase in zero elements.
    """
    n = matrix.rows

    best_transform = None
    best_zero_gain = -1
    transform_type = None  # 'row' or 'column'

    # Row operations
    for target_row in range(n):
        current_zeros = sum(1 for elem in matrix.row(target_row) if elem == 0)

        for source_row in range(n):
            if target_row == source_row:
                continue

            for pivot_col in range(n):
                a_target = matrix[target_row, pivot_col]
                a_source = matrix[source_row, pivot_col]
                if a_target != 0 and a_source != 0:
                    try:
                        factor = -a_target / a_source
                    except (ZeroDivisionError, TypeError):
                        continue

                    # Simulate the transformed row
                    new_zeros = 0
                    for col in range(n):
                        new_val = matrix[target_row, col] + \
                            factor * matrix[source_row, col]
                        if new_val == 0:
                            new_zeros += 1

                    zero_gain = new_zeros - current_zeros
                    if zero_gain > best_zero_gain:
                        best_zero_gain = zero_gain
                        best_transform = (
                            target_row, source_row, pivot_col, factor)
                        transform_type = 'row'

    # Column operations
    for target_col in range(n):
        current_zeros = sum(1 for elem in matrix.col(target_col) if elem == 0)

        for source_col in range(n):
            if target_col == source_col:
                continue

            for pivot_row in range(n):
                a_target = matrix[pivot_row, target_col]
                a_source = matrix[pivot_row, source_col]
                if a_target != 0 and a_source != 0:
                    try:
                        factor = -a_target / a_source
                    except (ZeroDivisionError, TypeError):
                        continue

                    # Simulate the transformed column
                    new_zeros = 0
                    for row in range(n):
                        new_val = matrix[row, target_col] + \
                            factor * matrix[row, source_col]
                        if new_val == 0:
                            new_zeros += 1

                    zero_gain = new_zeros - current_zeros
                    if zero_gain > best_zero_gain:
                        best_zero_gain = zero_gain
                        best_transform = (
                            target_col, source_col, pivot_row, factor)
                        transform_type = 'column'

    # Apply best transformation
    if transform_type == 'row':
        target_row, source_row, _, factor = best_transform
        new_rows = []
        for i in range(n):
            if i == target_row:
                new_row = [matrix[i, j] + factor *
                           matrix[source_row, j] for j in range(n)]
            else:
                new_row = list(matrix.row(i))
            new_rows.append(new_row)
        new_matrix = Determinant(Matrix(new_rows))
        explanation = (
            rf"$R_{{{target_row + 1}}} + ({latex(factor)}) \cdot R_{{{source_row + 1}}} "
            rf"\to R_{{{target_row + 1}}}$ (创造 {best_zero_gain} 个零元素)"
        )
        return new_matrix, explanation

    # column
    target_col, source_col, _, factor = best_transform
    new_rows = []
    for i in range(n):
        new_row = []
        for j in range(n):
            if j == target_col:
                new_val = matrix[i, j] + factor * matrix[i, source_col]
                new_row.append(new_val)
            else:
                new_row.append(matrix[i, j])
        new_rows.append(new_row)
    new_matrix = Determinant(Matrix(new_rows))
    explanation = (
        rf"$C_{{{target_col + 1}}} + ({latex(factor)}) \cdot C_{{{source_col + 1}}} "
        rf"\to C_{{{target_col + 1}}}$ (创造 {best_zero_gain} 个零元素)"
    )
    return new_matrix, explanation


def zero_row_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches matrix containing at least one all-zero row."""

    for row in matrix.tolist():
        if all(element == 0 for element in row):
            return 'zero_row'
    return None


def zero_column_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches matrix containing at least one all-zero col."""

    cols = matrix.T.tolist()
    for col in cols:
        if all(element == 0 for element in col):
            return 'zero_column'
    return None


def duplicate_row_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches matrix that contain two identical rows."""

    rows = matrix.tolist()
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i] == rows[j]:
                return 'duplicate_row'
    return None


def duplicate_column_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches matrix that contain two identical cols."""

    cols = matrix.T.tolist()
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            if cols[i] == cols[j]:
                return 'duplicate_column'
    return None


def diagonal_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches diagonal matrix."""

    if matrix.is_diagonal():
        return 'diagonal'
    return None


def triangular_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches upper or lower triangular matrix."""

    if is_upper_triangular(matrix) or is_lower_triangular(matrix):
        return 'triangular'
    return None


def scalar_multiple_row_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches matrices where at least one row has a nontrivial common factor.

    Specifically, checks each row for a greatest common divisor (GCD) of its
    nonzero elements that is not a unit (i.e., != +-1).
    """

    n = matrix.rows
    for i in range(n):
        row = matrix.row(i)
        non_zero_elements = [elem for elem in row if elem != 0]
        if len(non_zero_elements) > 1:
            try:
                common_factor = non_zero_elements[0]
                for elem in non_zero_elements[1:]:
                    common_factor = gcd(common_factor, elem)
                if common_factor not in (1, -1):
                    return 'scalar_multiple_row'
            except Exception:
                continue
    return None


def scalar_multiple_column_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches matrices where at least one column has a nontrivial common factor.

    Checks each column for a GCD of its nonzero elements that is not a unit.
    """
    n = matrix.cols
    for j in range(n):
        col = matrix.col(j)
        non_zero_elements = [elem for elem in col if elem != 0]
        if len(non_zero_elements) > 1:
            try:
                common_factor = non_zero_elements[0]
                for elem in non_zero_elements[1:]:
                    common_factor = gcd(common_factor, elem)
                if common_factor not in (1, -1):
                    return 'scalar_multiple_column'
            except Exception:
                continue
    return None


def linear_combination_matcher(matrix: Matrix, _context: RuleContext) -> MatcherFunctionReturn:
    """Matches matrix that can benefit from a determinant-preserving linear combination
    (elementary row or column operation) to introduce new zero entries.

    This matcher avoids applying the rule to small matrix (<= 2x2), where direct
    evaluation is more efficient, and skips matrix already sparse enough that
    further simplification is unlikely to help. It then checks whether any valid
    row or column operation of the form R_i <- R_i + c*R_j or C_i <- C_i + c*C_j
    can create at least one new zero (i.e., in a position that was previously nonzero).
    """

    n = matrix.rows

    # Skip small matrix - direct computation is preferable
    if n <= 2:
        return None

    total_elements = n * n
    zero_count = sum(1 for i in range(n)
                     for j in range(n) if matrix[i, j] == 0)
    zero_ratio = zero_count / total_elements

    # If matrix is already > 50% zeros, likely not worth transforming
    if zero_ratio > 0.5:
        return None

    # Avoid over-sparse rows/columns (e.g., > 60% zeros) - diminishing returns
    sparsity_threshold = 0.6
    for i in range(n):
        if sum(1 for j in range(n) if matrix[i, j] == 0) / n > sparsity_threshold:
            return None
    for j in range(n):
        if sum(1 for i in range(n) if matrix[i, j] == 0) / n > sparsity_threshold:
            return None

    # Check for beneficial row operations
    for target_row in range(n):
        for source_row in range(n):
            if target_row == source_row:
                continue
            for pivot_col in range(n):
                a_target = matrix[target_row, pivot_col]
                a_source = matrix[source_row, pivot_col]
                if a_target != 0 and a_source != 0:
                    try:
                        factor = -a_target / a_source
                    except (ZeroDivisionError, TypeError):
                        continue

                    # Look for at least one new zero created in a previously nonzero position
                    for col in range(n):
                        if matrix[target_row, col] != 0:
                            new_val = matrix[target_row, col] + \
                                factor * matrix[source_row, col]
                            if new_val == 0:
                                return 'linear_combination'

    # Check for beneficial column operations
    for target_col in range(n):
        for source_col in range(n):
            if target_col == source_col:
                continue
            for pivot_row in range(n):
                a_target = matrix[pivot_row, target_col]
                a_source = matrix[pivot_row, source_col]
                if a_target != 0 and a_source != 0:
                    try:
                        factor = -a_target / a_source
                    except (ZeroDivisionError, TypeError):
                        continue

                    # Look for at least one new zero in a previously nonzero position
                    for row in range(n):
                        if matrix[row, target_col] != 0:
                            new_val = matrix[row, target_col] + \
                                factor * matrix[row, source_col]
                            if new_val == 0:
                                return 'linear_combination'

    return None
