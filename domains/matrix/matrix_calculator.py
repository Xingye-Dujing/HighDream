from typing import List, Tuple, Union
from sympy import Matrix, latex, sympify
# from IPython.display import Math, display


class MatrixCalculator:
    def __init__(self) -> None:
        """Initialize the MatrixCalculator with empty history and results storage. """
        self.matrix_history = []
        self.latex_results = []
        self.record_all_steps = True

    def parse_matrix_expression(self, matrix_str: str) -> Matrix:
        """Parse a matrix expression string into a sympy Matrix.

        Args:
            matrix_str (str): Matrix expression in format like '[[1,2],[3,4]]'

        Returns:
            Matrix: A sympy Matrix object

        Raises:
            ValueError: If the matrix expression cannot be parsed
        """
        try:
            return Matrix(sympify(matrix_str))
        except Exception as e:
            raise ValueError(f"无法解析矩阵表达式: {matrix_str} \n 错误: {e}") from e

    def apply_operation(self, matrix: Matrix, operation: str) -> Tuple[Matrix, str]:
        """Apply a single matrix operation and return the result with description.

        Args:
            matrix (Matrix): The input matrix to operate on
            operation (str): Operation command string

        Returns:
            tuple: (result, description) where result is the operation result
                   and description explains what was done

        Raises:
            ValueError: If operation is invalid or missing required parameters
        """
        op_parts = operation.strip().split()
        op_type = op_parts[0].lower()

        try:
            if op_type in ('transpose', '转置'):
                result = matrix.T
                description = f"转置矩阵"
                return result, description

            if op_type in ('inverse', '求逆'):
                result = matrix.inv()
                description = f"求逆矩阵"
                return result, description

            if op_type in ('rref', '化为行最简形'):
                result, _ = matrix.rref()
                description = f"化为行最简形"
                return result, description

            if op_type in ('multiply', '乘'):
                if len(op_parts) < 2:
                    raise ValueError("乘法操作需要指定矩阵")
                other_matrix = self.parse_matrix_expression(
                    ' '.join(op_parts[1:]))
                result = matrix * other_matrix
                description = f"矩阵乘法"
                return result, description

            if op_type in ('add', '加'):
                if len(op_parts) < 2:
                    raise ValueError("加法操作需要指定矩阵")
                other_matrix = self.parse_matrix_expression(
                    ' '.join(op_parts[1:]))
                result = matrix + other_matrix
                description = f"矩阵加法"
                return result, description

            if op_type in ('subtract', '减'):
                if len(op_parts) < 2:
                    raise ValueError("减法操作需要指定矩阵")
                other_matrix = self.parse_matrix_expression(
                    ' '.join(op_parts[1:]))
                result = matrix - other_matrix
                description = f"矩阵减法"
                return result, description

            if op_type in ('scale', '缩放'):
                if len(op_parts) < 2:
                    raise ValueError("缩放操作需要指定标量")
                scalar = sympify(op_parts[1])
                result = scalar * matrix
                description = f"矩阵缩放(乘以 ${latex(scalar)}$)"
                return result, description

            if op_type in ('power', '幂'):
                if len(op_parts) < 2:
                    raise ValueError("幂操作需要指定指数")
                exponent = int(op_parts[1])
                result = matrix ** exponent
                description = f"矩阵的 {exponent} 次幂"
                return result, description

            if op_type in ('swap_rows', '交换行'):
                if len(op_parts) < 3:
                    raise ValueError("交换行需要指定两个行索引")
                i, j = int(op_parts[1]), int(op_parts[2])
                result = matrix.copy()
                result.row_swap(i, j)
                description = f"交换行 {i} 和行 {j}"
                return result, description

            if op_type in ('swap_cols', '交换列'):
                if len(op_parts) < 3:
                    raise ValueError("交换列需要指定两个列索引")
                i, j = int(op_parts[1]), int(op_parts[2])
                result = matrix.copy()
                result.col_swap(i, j)
                description = f"交换列 {i} 和列 {j}"
                return result, description

            if op_type in ('add_rows', '行相加'):
                if len(op_parts) < 4:
                    raise ValueError("行相加需要指定源行、目标行和系数")
                src, dest, coeff = int(op_parts[1]), int(
                    op_parts[2]), sympify(op_parts[3])
                result = matrix.copy()
                result.row_op(dest, lambda v, i: v + coeff * matrix[src, i])
                description = f"将 {coeff} 乘以行 {src} 加到行 {dest}"
                return result, description

            if op_type in ('add_cols', '列相加'):
                if len(op_parts) < 4:
                    raise ValueError("列相加需要指定源列、目标列和系数")
                src, dest, coeff = int(op_parts[1]), int(
                    op_parts[2]), sympify(op_parts[3])
                result = matrix.copy()
                result.col_op(dest, lambda v, i: v + coeff * matrix[i, src])
                description = f"将 {coeff} 乘以列 {src} 加到列 {dest}"
                return result, description

            if op_type in ('scale_row', '缩放行'):
                if len(op_parts) < 3:
                    raise ValueError("缩放行需要指定行索引和系数")
                row, coeff = int(op_parts[1]), sympify(op_parts[2])
                result = matrix.copy()
                result.row_op(row, lambda v: coeff * v)
                description = f"将行 {row} 乘以 ${latex(coeff)}$"
                return result, description

            if op_type in ('scale_col', '缩放列'):
                if len(op_parts) < 3:
                    raise ValueError("缩放列需要指定列索引和系数")
                col, coeff = int(op_parts[1]), sympify(op_parts[2])
                result = matrix.copy()
                result.col_op(col, lambda v: coeff * v)
                description = f"将列 {col} 乘以 ${latex(coeff)}$"
                return result, description

            if op_type in ('insert_row', '插入行'):
                if len(op_parts) < 3:
                    raise ValueError("插入行需要指定位置和行元素")
                pos = int(op_parts[1])
                row_elements = [sympify(x) for x in op_parts[2:]]
                result = matrix.row_insert(pos, Matrix([row_elements]))
                description = f"在位置 {pos} 插入新行"
                return result, description

            if op_type in ('insert_col', '插入列'):
                if len(op_parts) < 3:
                    raise ValueError("插入列需要指定位置和列元素")
                pos = int(op_parts[1])
                col_elements = [sympify(x) for x in op_parts[2:]]
                result = matrix.col_insert(pos, Matrix(col_elements))
                description = f"在位置 {pos} 插入新列"
                return result, description

            if op_type in ('remove_row', '删除行'):
                if len(op_parts) < 2:
                    raise ValueError("删除行需要指定行索引")
                row = int(op_parts[1])
                rows_to_keep = [i for i in range(matrix.rows) if i != row]
                result = matrix.extract(rows_to_keep, list(range(matrix.cols)))
                description = f"删除行 {row}"
                return result, description

            if op_type in ('remove_col', '删除列'):
                if len(op_parts) < 2:
                    raise ValueError("删除列需要指定列索引")
                col = int(op_parts[1])
                cols_to_keep = [i for i in range(matrix.cols) if i != col]
                result = matrix.extract(list(range(matrix.rows)), cols_to_keep)
                description = f"删除列 {col}"
                return result, description

            if op_type in ('submatrix', '子矩阵'):
                if len(op_parts) < 5:
                    raise ValueError("提取子矩阵需要指定起始行、结束行、起始列、结束列")
                r1, r2, c1, c2 = map(int, op_parts[1:5])
                result = matrix[r1:r2+1, c1:c2+1]
                description = f"提取子矩阵 $[{r1}:{r2},{c1}:{c2}]$"
                return result, description

            raise ValueError(f"未知操作: {op_type}")

        except Exception as e:
            raise ValueError(f"操作 '{operation}' 执行失败: {e}") from e

    def calculate(self, matrix_expr: str, operations: List[str], record_all_steps: bool = False) -> Union[str, List[str]]:
        """Main calculation function that applies a series of operations to a matrix.

        Args:
            matrix_expr (str): Initial matrix expression string
            operations (List[str]): List of operation commands to apply
            record_all_steps (bool): Whether to record all intermediate steps

        Returns:
            Union[str, List[str]]: Either final result LaTeX or list of all steps LaTeX
        """
        # Reset history records
        self.matrix_history = []
        self.latex_results = []
        self.record_all_steps = record_all_steps

        try:
            # Parse initial matrix
            current_matrix = self.parse_matrix_expression(matrix_expr)
            self.matrix_history.append(current_matrix)
            self.latex_results.append(latex(current_matrix))

            # Apply all operations
            for operation in operations:
                result, desc = self.apply_operation(
                    current_matrix, operation)

                # Update current matrix (if result is a matrix)
                if isinstance(result, Matrix):
                    current_matrix = result
                    self.matrix_history.append(current_matrix)

                # Record result
                latex_str = latex(result)
                if record_all_steps:
                    latex_str += f"\\quad \\text{{{desc}}}"
                    self.latex_results.append(latex_str)
                else:
                    # Only keep the last result
                    self.latex_results = [latex_str]

            return self.latex_results[-1]

        except Exception as e:
            return f"计算错误: {e}"

    def get_steps_latex(self) -> List[str]:
        """Generate LaTeX formatted steps for display.

        Returns:
            List[str]: LaTeX formatted string containing all steps or just the final result
        """
        if not self.record_all_steps:
            latex_str = "\\begin{align}"
            latex_str += self.latex_results[0]
            latex_str += "\\end{align}"
            return latex_str

        latex_str = "\\begin{align}"
        for i, step in enumerate(self.latex_results):
            if i == 0:
                step_str = f"& \\text{{初始矩阵:}} {step}"
            else:
                step_str = f"& \\text{{步骤 {i} :}} {step}"
            latex_str += step_str
            # New line and skip one line
            latex_str += r"\\\\"
        latex_str += "\\end{align}"
        return latex_str


# def demo():
#     calculator = MatrixCalculator()

#     # Demo 1: Basic matrix operations

#     display(Math("\\text{示例 1: 基本矩阵操作}"))
#     matrix_expr = "[[1, 2], [3, 4]]"
#     operations = ["transpose", "inverse"]
#     calculator.calculate(
#         matrix_expr, operations, record_all_steps=True)
#     display(Math(calculator.get_steps_latex()))

#     # Demo 2: Symbolic matrix

#     display(Math("\\text{示例 2: 符号矩阵}"))
#     matrix_expr = "[[a, b], [c, d]]"
#     operations = ["inverse", "transpose"]
#     calculator.calculate(
#         matrix_expr, operations, record_all_steps=True)
#     display(Math(calculator.get_steps_latex()))

#     # Demo 3: Row operations

#     display(Math("\\text{示例 3: 行操作}"))
#     matrix_expr = "[[1,2,3],[4,5,6],[7,8,9]]"
#     operations = ["swap_rows 0 1", "add_rows 0 1 2", "scale_row 2 3"]
#     calculator.calculate(
#         matrix_expr, operations, record_all_steps=True)
#     display(Math(calculator.get_steps_latex()))

#     display(Math("\\text{示例 3: 行操作}"))
#     matrix_expr = "[[1,2,3],[4,5,6],[7,8,9]]"
#     calculator.calculate(
#         matrix_expr, operations, record_all_steps=True)
#     display(Math(calculator.get_steps_latex()))


# if __name__ == "__main__":
#     demo()
