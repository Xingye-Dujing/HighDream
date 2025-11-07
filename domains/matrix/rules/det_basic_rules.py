from typing import Any, Dict, Tuple, Optional
from sympy import Determinant, Expr, latex, Mul, Matrix, gcd, Integer


def zero_row_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """零行规则：如果矩阵有全零行，则行列式为0"""
    matrix = det_expr.args[0]
    rows = matrix.tolist()
    for i, row in enumerate(rows):
        if all(element == 0 for element in row):
            return Integer(0), f"第{i+1}行为零行, 行列式为0"
    return None


def zero_column_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """零列规则：如果矩阵有全零列，则行列式为0"""
    matrix = det_expr.args[0]
    cols = matrix.T.tolist()  # 转置后检查列
    for i, col in enumerate(cols):
        if all(element == 0 for element in col):
            return Integer(0), f"第{i+1}列为零列, 行列式为0"
    return None


def duplicate_row_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """重复行规则：如果矩阵有两行相同，则行列式为0"""
    matrix = det_expr.args[0]
    rows = matrix.tolist()
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i] == rows[j]:
                return Integer(0), f"第{i+1}行和第{j+1}行相同, 行列式为0"
    return None


def duplicate_column_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """重复列规则：如果矩阵有两列相同，则行列式为0"""
    matrix = det_expr.args[0]
    cols = matrix.T.tolist()  # 转置后检查列
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            if cols[i] == cols[j]:
                return Integer(0), f"第{i+1}列和第{j+1}列相同, 行列式为0"
    return None


def diagonal_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """对角矩阵规则：对角矩阵的行列式等于对角线元素的乘积"""
    matrix = det_expr.args[0]
    if matrix.is_diagonal():
        diag_elements = [matrix[i, i] for i in range(matrix.rows)]
        product = Mul(*diag_elements)
        return product, f"对角矩阵的行列式等于对角线元素的乘积: ${latex(product)}$"
    return None


def triangular_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """三角矩阵规则：三角矩阵的行列式等于对角线元素的乘积"""
    matrix = det_expr.args[0]
    if is_upper_triangular(matrix):
        diag_elements = [matrix[i, i] for i in range(matrix.rows)]
        product = Mul(*diag_elements)
        return product, f"上三角矩阵的行列式等于对角线元素的乘积: ${latex(product)}$"
    elif is_lower_triangular(matrix):
        diag_elements = [matrix[i, i] for i in range(matrix.rows)]
        product = Mul(*diag_elements)
        return product, f"下三角矩阵的行列式等于对角线元素的乘积: ${latex(product)}$"
    return None


def scalar_multiple_row_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """行标量乘法规则：如果某行有公因子，可以提取出来"""
    matrix = det_expr.args[0]
    n = matrix.rows

    for i in range(n):
        row = matrix.row(i)
        # 检查是否所有非零元素有公因子
        non_zero_elements = [elem for elem in row if elem != 0]
        if len(non_zero_elements) <= 1:
            continue

        # 尝试找到公因子
        try:
            common_factor = gcd(non_zero_elements)
            if common_factor != 1:
                # 提取公因子 - 创建新矩阵而不是修改原矩阵
                new_matrix_elements = []
                for row_idx in range(n):
                    if row_idx == i:
                        # 当前行除以公因子
                        new_row = [
                            elem / common_factor for elem in matrix.row(row_idx)]
                    else:
                        # 其他行保持不变
                        new_row = matrix.row(row_idx)
                    new_matrix_elements.append(new_row)

                new_matrix = Matrix(new_matrix_elements)
                result = common_factor * Determinant(new_matrix)
                return result, f"$从第\\,{i+1}\\,行提取公因子\\,{latex(common_factor)}$"
        except:
            continue

    return None


def scalar_multiple_column_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """列标量乘法规则：如果某列有公因子，可以提取出来"""
    matrix = det_expr.args[0]
    n = matrix.cols

    for j in range(n):
        col = matrix.col(j)
        # 检查是否所有非零元素有公因子
        non_zero_elements = [elem for elem in col if elem != 0]
        if len(non_zero_elements) <= 1:
            continue

        # 尝试找到公因子
        try:
            common_factor = gcd(non_zero_elements)
            if common_factor != 1:
                # 提取公因子 - 创建新矩阵而不是修改原矩阵
                new_matrix_elements = []
                for row_idx in range(n):
                    new_row = []
                    for col_idx in range(n):
                        if col_idx == j:
                            # 当前列除以公因子
                            new_row.append(
                                matrix[row_idx, col_idx] / common_factor)
                        else:
                            # 其他列保持不变
                            new_row.append(matrix[row_idx, col_idx])
                    new_matrix_elements.append(new_row)

                new_matrix = Matrix(new_matrix_elements)
                result = common_factor * Determinant(new_matrix)
                return result, f"$从第\\,{j+1}\\,列提取公因子\\,{latex(common_factor)}$"
        except:
            continue

    return None


def linear_combination_rule(det_expr: Determinant, context: Dict[str, Any]) -> Optional[Tuple[Expr, str]]:
    """线性组合规则：通过行变换或列变换创造更多零元素, 选择创造零元素最多的变换"""
    matrix = det_expr.args[0]
    n = matrix.rows

    best_transform = None
    best_zero_gain = -1
    transform_type = None  # 'row' 或 'column'

    # 检查行变换
    for target_row in range(n):
        current_zeros = sum(1 for elem in matrix.row(target_row) if elem == 0)

        for source_row in range(n):
            if target_row == source_row:
                continue

            for pivot_col in range(n):
                if matrix[target_row, pivot_col] != 0 and matrix[source_row, pivot_col] != 0:
                    factor = -matrix[target_row, pivot_col] / \
                        matrix[source_row, pivot_col]

                    # 模拟行变换结果
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

    # 检查列变换
    for target_col in range(n):
        current_zeros = sum(1 for elem in matrix.col(target_col) if elem == 0)

        for source_col in range(n):
            if target_col == source_col:
                continue

            for pivot_row in range(n):
                if matrix[pivot_row, target_col] != 0 and matrix[pivot_row, source_col] != 0:
                    factor = -matrix[pivot_row, target_col] / \
                        matrix[pivot_row, source_col]

                    # 模拟列变换结果
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

    # 应用最佳变换
    if best_transform and best_zero_gain > 0:
        if transform_type == 'row':
            target_row, source_row, pivot_col, factor = best_transform

            new_matrix_elements = []
            for row_idx in range(n):
                if row_idx == target_row:
                    new_row = [matrix[target_row, j] + factor * matrix[source_row, j]
                               for j in range(n)]
                else:
                    new_row = matrix.row(row_idx)
                new_matrix_elements.append(new_row)

            new_matrix = Matrix(new_matrix_elements)
            explanation = (rf"$R_{{{target_row+1}}} + ({latex(factor)}) \times "
                           rf"R_{{{source_row+1}}} \to R_{{{target_row+1}}}"
                           rf"\,(创造\,{best_zero_gain}\,个零元素)$")
            return Determinant(new_matrix), explanation

        else:
            target_col, source_col, pivot_row, factor = best_transform

            new_matrix_elements = []
            for row_idx in range(n):
                new_row = []
                for col_idx in range(n):
                    if col_idx == target_col:
                        new_val = matrix[row_idx, target_col] + \
                            factor * matrix[row_idx, source_col]
                        new_row.append(new_val)
                    else:
                        new_row.append(matrix[row_idx, col_idx])
                new_matrix_elements.append(new_row)

            new_matrix = Matrix(new_matrix_elements)
            explanation = (rf"$C_{{{target_col+1}}} + ({latex(factor)}) \times "
                           rf"C_{{{source_col+1}}} \to C_{{{target_col+1}}}"
                           rf"\,(创造\,{best_zero_gain}\,个零元素)$")
            return Determinant(new_matrix), explanation

    return None


def zero_row_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配零行"""
    matrix = det_expr.args[0]
    rows = matrix.tolist()
    for row in rows:
        if all(element == 0 for element in row):
            return 'zero_row'
    return None


def zero_column_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配零列"""
    matrix = det_expr.args[0]
    cols = matrix.T.tolist()
    for col in cols:
        if all(element == 0 for element in col):
            return 'zero_column'
    return None


def duplicate_row_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配重复行"""
    matrix = det_expr.args[0]
    rows = matrix.tolist()
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i] == rows[j]:
                return 'duplicate_row'
    return None


def duplicate_column_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配重复列"""
    matrix = det_expr.args[0]
    cols = matrix.T.tolist()
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            if cols[i] == cols[j]:
                return 'duplicate_column'
    return None


def diagonal_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配对角矩阵"""
    matrix = det_expr.args[0]
    if matrix.is_diagonal():
        return 'diagonal'
    return None


def is_upper_triangular(matrix):
    """检查矩阵是否为上三角矩阵"""
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(i):  # 检查下三角部分（不包括对角线）
            if matrix[i, j] != 0:
                return False
    return True


def is_lower_triangular(matrix):
    """检查矩阵是否为下三角矩阵"""
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(i + 1, cols):  # 检查上三角部分(不包括对角线)
            if matrix[i, j] != 0:
                return False
    return True


def triangular_matcher(det_expr, context):
    matrix = Matrix(det_expr.args[0])

    # 使用自定义的三角矩阵检查函数
    if is_upper_triangular(matrix) or is_lower_triangular(matrix):
        return 'triangular'
    return None


def scalar_multiple_row_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配可以提取行公因子的情况"""
    matrix = det_expr.args[0]
    n = matrix.rows

    for i in range(n):
        row = matrix.row(i)
        non_zero_elements = [elem for elem in row if elem != 0]
        if len(non_zero_elements) > 1:
            try:
                common_factor = gcd(non_zero_elements)
                if common_factor != 1:
                    return 'scalar_multiple_row'
            except:
                continue
    return None


def scalar_multiple_column_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配可以提取列公因子的情况"""
    matrix = det_expr.args[0]
    n = matrix.cols

    for j in range(n):
        col = matrix.col(j)
        non_zero_elements = [elem for elem in col if elem != 0]
        if len(non_zero_elements) > 1:
            try:
                common_factor = gcd(non_zero_elements)
                if common_factor != 1:
                    return 'scalar_multiple_column'
            except:
                continue
    return None


def linear_combination_matcher(det_expr: Determinant, context: Dict[str, Any]) -> Optional[str]:
    """匹配适合线性组合的情况: 检查是否能通过行变换或列变换显著增加零元素数量"""
    matrix = det_expr.args[0]
    n = matrix.rows

    # 对于小矩阵(2x2), 直接计算通常更高效
    if n <= 2:
        return None

    # 检查当前矩阵的零元素比例
    total_elements = n * n
    zero_count = sum(1 for i in range(n)
                     for j in range(n) if matrix[i, j] == 0)
    zero_ratio = zero_count / total_elements

    # 如果零元素比例已经很高(>50%), 可能不需要线性组合
    if zero_ratio > 0.5:
        return None

    # 检查是否有任一行或任一列的零元素比例过高
    threshold = 0.6  # 设定阈值, 例如 0.6

    # 检查每一行的零元素比例
    for i in range(n):
        row_zero_count = sum(1 for j in range(n) if matrix[i, j] == 0)
        if row_zero_count / n > threshold:
            return None

    # 检查每一列的零元素比例
    for j in range(n):
        col_zero_count = sum(1 for i in range(n) if matrix[i, j] == 0)
        if col_zero_count / n > threshold:
            return None

    # 检查是否存在能创造至少 1 个零元素的变换
    # 行变换检查
    for target_row in range(n):
        for source_row in range(n):
            if target_row == source_row:
                continue

            for pivot_col in range(n):
                if matrix[target_row, pivot_col] != 0 and matrix[source_row, pivot_col] != 0:
                    factor = -matrix[target_row, pivot_col] / \
                        matrix[source_row, pivot_col]

                    # 快速检查是否能创造新零元素
                    for col in range(n):
                        new_val = matrix[target_row, col] + \
                            factor * matrix[source_row, col]
                        if new_val == 0 and matrix[target_row, col] != 0:
                            return 'linear_combination'

    # 列变换检查
    for target_col in range(n):
        for source_col in range(n):
            if target_col == source_col:
                continue

            for pivot_row in range(n):
                if matrix[pivot_row, target_col] != 0 and matrix[pivot_row, source_col] != 0:
                    factor = -matrix[pivot_row, target_col] / \
                        matrix[pivot_row, source_col]

                    # 快速检查是否能创造新零元素
                    for row in range(n):
                        new_val = matrix[row, target_col] + \
                            factor * matrix[row, source_col]
                        if new_val == 0 and matrix[row, target_col] != 0:
                            return 'linear_combination'

    return None
