from sympy import Expr, Matrix, latex


def _swap_rows_rule(matrix: Matrix, row1: int, row2: int) -> tuple[Matrix, str]:
    """Perform elementary row swap: R_{row1+1} <-> R_{row2+1}."""
    new = matrix.copy()
    new.row_swap(row1, row2)
    explanation = rf"$\mathrm{{R}}_{{{row1 + 1}}} \leftrightarrow \mathrm{{R}}_{{{row2 + 1}}}$"
    return new, explanation


def _scale_row_rule(matrix: Matrix, row: int, factor: Expr) -> tuple[Matrix, str]:
    """Scale a row by a nonzero scalar: R_{row+1} <- factor*R_{row+1}."""
    new = matrix.copy()
    new.row_op(row, lambda v, _: factor * v)
    explanation = rf"$({latex(factor)}) \cdot \mathrm{{R}}_{{{row + 1}}} \to \mathrm{{R}}_{{{row + 1}}}$"
    return new, explanation


def _add_rows_rule(matrix: Matrix, target_row: int, source_row: int, factor: Expr) -> tuple[Matrix, str]:
    """Add a multiple of one row to another: R_{target+1} <- R_{target+1}+factor*R_{source+1}."""

    new = matrix.copy()
    new.row_op(target_row, lambda v, j: v + factor * matrix[source_row, j])
    explanation = (
        rf"$\mathrm{{R}}_{{{target_row + 1}}} + ({latex(factor)}) \cdot "
        rf"\mathrm{{R}}_{{{source_row + 1}}} \to \mathrm{{R}}_{{{target_row + 1}}}$"
    )
    return new, explanation


def apply_swap_rule(matrix: Matrix, pivot_row: int, other_row: int) -> tuple[Matrix, str]:
    """Swap the pivot row with another row."""

    return _swap_rows_rule(matrix, pivot_row, other_row)


def apply_scale_rule(matrix: Matrix, pivot_row: int, col: int) -> tuple[Matrix, str]:
    """Scale the pivot row so that the pivot element becomes 1.

    If the pivot is already 1 or 0, no operation is performed (returns original matrix
    and empty string).
    """
    val = matrix[pivot_row, col]
    if val in (0, 1):
        return matrix, ""
    scale_factor = 1 / val
    return _scale_row_rule(matrix, pivot_row, scale_factor)


def apply_elimination_rule(matrix: Matrix, pivot_row: int, col: int, target_row: int) -> tuple[Matrix, str]:
    """Eliminate the entry in target_row, col using the pivot at (pivot_row, col).

    Applies: R_target <- R_target−(a_target/a_pivot)*R_pivot,
    which zeroes out the target entry.
    """

    factor = -matrix[target_row, col]
    return _add_rows_rule(matrix, target_row, pivot_row, factor)
