from sympy import latex


# 定义三个基本行变换
def _swap_rows_rule(matrix, row1, row2):
    new = matrix.copy()
    new.row_swap(row1, row2)
    return new, f"$\\mathrm{{R}}_{{{row1+1}}} \\leftrightarrow \\mathrm{{R}}_{{{row2+1}}}$"


def _scale_row_rule(matrix, row, factor):
    new = matrix.copy()
    new.row_op(row, lambda v, j: factor * v)
    return new, f"$({latex(factor)}) \\times \\mathrm{{R}}_{{{row+1}}} \\to \\mathrm{{R}}_{{{row+1}}}$"


def _add_rows_rule(matrix, target_row, source_row, factor):
    new = matrix.copy()
    # R_target <- R_target + factor * R_source
    new.row_op(target_row, lambda v, j: v + factor * matrix[source_row, j])
    return new, f"$\\mathrm{{R}}_{{{target_row+1}}} + ({latex(factor)}) \\times \\mathrm{{R}}_{{{source_row+1}}} \\to \\mathrm{{R}}_{{{target_row+1}}}$"


def apply_swap_rule(matrix, pivot_row, col):
    """交换两行"""
    return _swap_rows_rule(matrix, pivot_row, col)


def apply_scale_rule(matrix, pivot_row, col):
    """缩放, 使主元为 1"""
    val = matrix[pivot_row, col]
    if val == 1 or val == 0:
        return matrix, ""

    scale_factor = 1 / val
    return _scale_row_rule(matrix, pivot_row, scale_factor)


def apply_elimination_rule(matrix, pivot_row, col, target_row):
    """使用主元 (pivot_row, col) 消去 target_row 在 col 处的元素"""
    factor = -matrix[target_row, col]
    return _add_rows_rule(matrix, target_row, pivot_row, factor)
