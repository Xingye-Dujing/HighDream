from typing import List, Union, Any
from sympy import Matrix, sympify, latex
from IPython.display import display, Math


class MatrixCalculator:
    def __init__(self):
        self.matrix_history = []
        self.latex_results = []
        self.record_all_steps = True

    def parse_matrix_expression(self, matrix_str: str) -> Matrix:
        """
        解析矩阵表达式字符串为sympy矩阵
        支持格式: '[[1,2],[3,4]]'
        """
        try:
            return Matrix(sympify(matrix_str))
        except Exception as e:
            raise ValueError(f"无法解析矩阵表达式: {matrix_str} \n 错误: {e}") from e

    def apply_operation(self, matrix: Matrix, operation: str) -> Any:
        """
        应用单个矩阵操作
        返回操作结果和操作描述
        """
        op_parts = operation.strip().split()
        op_type = op_parts[0].lower()

        try:
            if op_type == 'transpose' or op_type == '转置':
                result = matrix.T
                description = f"转置矩阵"
                return result, description

            elif op_type == 'inverse' or op_type == '求逆':
                result = matrix.inv()
                description = f"求逆矩阵"
                return result, description

            elif op_type == 'rref' or op_type == '化为行最简形':
                result, _ = matrix.rref()
                description = f"化为行最简形"
                return result, description

            elif op_type == 'multiply' or op_type == '乘':
                if len(op_parts) < 2:
                    raise ValueError("乘法操作需要指定矩阵")
                other_matrix = self.parse_matrix_expression(
                    ' '.join(op_parts[1:]))
                result = matrix * other_matrix
                description = f"矩阵乘法"
                return result, description

            elif op_type == 'add' or op_type == '加':
                if len(op_parts) < 2:
                    raise ValueError("加法操作需要指定矩阵")
                other_matrix = self.parse_matrix_expression(
                    ' '.join(op_parts[1:]))
                result = matrix + other_matrix
                description = f"矩阵加法"
                return result, description

            elif op_type == 'subtract' or op_type == '减':
                if len(op_parts) < 2:
                    raise ValueError("减法操作需要指定矩阵")
                other_matrix = self.parse_matrix_expression(
                    ' '.join(op_parts[1:]))
                result = matrix - other_matrix
                description = f"矩阵减法"
                return result, description

            elif op_type == 'scale' or op_type == '缩放':
                if len(op_parts) < 2:
                    raise ValueError("缩放操作需要指定标量")
                scalar = sympify(op_parts[1])
                result = scalar * matrix
                description = f"矩阵缩放(乘以 ${latex(scalar)}$)"
                return result, description

            elif op_type == 'power' or op_type == '幂':
                if len(op_parts) < 2:
                    raise ValueError("幂操作需要指定指数")
                exponent = int(op_parts[1])
                result = matrix ** exponent
                description = f"矩阵的 {exponent} 次幂"
                return result, description

            elif op_type == 'swap_rows' or op_type == '交换行':
                if len(op_parts) < 3:
                    raise ValueError("交换行需要指定两个行索引")
                i, j = int(op_parts[1]), int(op_parts[2])
                result = matrix.copy()
                result.row_swap(i, j)
                description = f"交换行 {i} 和行 {j}"
                return result, description

            elif op_type == 'swap_cols' or op_type == '交换列':
                if len(op_parts) < 3:
                    raise ValueError("交换列需要指定两个列索引")
                i, j = int(op_parts[1]), int(op_parts[2])
                result = matrix.copy()
                result.col_swap(i, j)
                description = f"交换列 {i} 和列 {j}"
                return result, description

            elif op_type == 'add_rows' or op_type == '行相加':
                if len(op_parts) < 4:
                    raise ValueError("行相加需要指定源行、目标行和系数")
                src, dest, coeff = int(op_parts[1]), int(
                    op_parts[2]), sympify(op_parts[3])
                result = matrix.copy()
                result.row_op(dest, lambda v, i: v + coeff * matrix[src, i])
                description = f"将 {coeff} 乘以行 {src} 加到行 {dest}"
                return result, description

            elif op_type == 'add_cols' or op_type == '列相加':
                if len(op_parts) < 4:
                    raise ValueError("列相加需要指定源列、目标列和系数")
                src, dest, coeff = int(op_parts[1]), int(
                    op_parts[2]), sympify(op_parts[3])
                result = matrix.copy()
                result.col_op(dest, lambda v, i: v + coeff * matrix[i, src])
                description = f"将 {coeff} 乘以列 {src} 加到列 {dest}"
                return result, description

            elif op_type == 'scale_row' or op_type == '缩放行':
                if len(op_parts) < 3:
                    raise ValueError("缩放行需要指定行索引和系数")
                row, coeff = int(op_parts[1]), sympify(op_parts[2])
                result = matrix.copy()
                result.row_op(row, lambda v, i: coeff * v)
                description = f"将行 {row} 乘以 ${latex(coeff)}$"
                return result, description

            elif op_type == 'scale_col' or op_type == '缩放列':
                if len(op_parts) < 3:
                    raise ValueError("缩放列需要指定列索引和系数")
                col, coeff = int(op_parts[1]), sympify(op_parts[2])
                result = matrix.copy()
                result.col_op(col, lambda v, i: coeff * v)
                description = f"将列 {col} 乘以 ${latex(coeff)}$"
                return result, description

            elif op_type == 'insert_row' or op_type == '插入行':
                if len(op_parts) < 3:
                    raise ValueError("插入行需要指定位置和行元素")
                pos = int(op_parts[1])
                row_elements = [sympify(x) for x in op_parts[2:]]
                result = matrix.row_insert(pos, Matrix([row_elements]))
                description = f"在位置 {pos} 插入新行"
                return result, description

            elif op_type == 'insert_col' or op_type == '插入列':
                if len(op_parts) < 3:
                    raise ValueError("插入列需要指定位置和列元素")
                pos = int(op_parts[1])
                col_elements = [sympify(x) for x in op_parts[2:]]
                result = matrix.col_insert(pos, Matrix(col_elements))
                description = f"在位置 {pos} 插入新列"
                return result, description

            elif op_type == 'remove_row' or op_type == '删除行':
                if len(op_parts) < 2:
                    raise ValueError("删除行需要指定行索引")
                row = int(op_parts[1])
                rows_to_keep = [i for i in range(matrix.rows) if i != row]
                result = matrix.extract(rows_to_keep, list(range(matrix.cols)))
                description = f"删除行 {row}"
                return result, description

            elif op_type == 'remove_col' or op_type == '删除列':
                if len(op_parts) < 2:
                    raise ValueError("删除列需要指定列索引")
                col = int(op_parts[1])
                cols_to_keep = [i for i in range(matrix.cols) if i != col]
                result = matrix.extract(list(range(matrix.rows)), cols_to_keep)
                description = f"删除列 {col}"
                return result, description

            elif op_type == 'submatrix' or op_type == '子矩阵':
                if len(op_parts) < 5:
                    raise ValueError("提取子矩阵需要指定起始行、结束行、起始列、结束列")
                r1, r2, c1, c2 = map(int, op_parts[1:5])
                result = matrix[r1:r2+1, c1:c2+1]
                description = f"提取子矩阵 $[{r1}:{r2},{c1}:{c2}]$"
                return result, description

            else:
                raise ValueError(f"未知操作: {op_type}")

        except Exception as e:
            raise ValueError(f"操作 '{operation}' 执行失败: {e}") from e

    def calculate(self, matrix_expr: str, operations: List[str], record_all_steps: bool = False) -> Union[str, List[str]]:
        """
        主计算函数

        参数:
        - matrix_expr: 矩阵表达式字符串
        - operations: 操作列表
        - record_all_steps: 是否记录所有步骤

        返回:
        - 如果record_all_steps为True: 返回所有步骤的LaTeX列表
        - 如果record_all_steps为False: 返回最终结果的LaTeX字符串
        """
        # 重置历史记录
        self.matrix_history = []
        self.latex_results = []
        self.record_all_steps = record_all_steps

        try:
            # 解析初始矩阵
            current_matrix = self.parse_matrix_expression(matrix_expr)
            self.matrix_history.append(current_matrix)
            self.latex_results.append(latex(current_matrix))

            # 应用所有操作
            for operation in operations:
                result, desc = self.apply_operation(
                    current_matrix, operation)

                # 更新当前矩阵(如果结果是矩阵)
                if isinstance(result, Matrix):
                    current_matrix = result
                    self.matrix_history.append(current_matrix)

                # 记录结果
                latex_str = latex(result)
                if record_all_steps:
                    latex_str += f"\\quad \\text{{{desc}}}"
                    self.latex_results.append(latex_str)
                else:
                    # 只保留最后一个结果
                    self.latex_results = [latex_str]

            return self.latex_results[-1]

        except Exception as e:
            return f"计算错误: {e}"

    def get_steps_latex(self) -> List[str]:
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
            # 换行并空一行
            latex_str += r"\\\\"
        latex_str += "\\end{align}"
        return latex_str


def demo():
    calculator = MatrixCalculator()

    # 示例1: 基本操作
    display(Math("\\text{示例 1: 基本矩阵操作}"))
    matrix_expr = "[[1, 2], [3, 4]]"
    operations = ["transpose", "inverse"]
    calculator.calculate(
        matrix_expr, operations, record_all_steps=True)
    display(Math(calculator.get_steps_latex()))

    # 示例2: 符号矩阵
    display(Math("\\text{示例 2: 符号矩阵}"))
    matrix_expr = "[[a, b], [c, d]]"
    operations = ["inverse", "transpose"]
    calculator.calculate(
        matrix_expr, operations, record_all_steps=True)
    display(Math(calculator.get_steps_latex()))

    # 示例3: 行操作
    display(Math("\\text{示例 3: 行操作}"))
    matrix_expr = "[[1,2,3],[4,5,6],[7,8,9]]"
    operations = ["swap_rows 0 1", "add_rows 0 1 2", "scale_row 2 3"]
    calculator.calculate(
        matrix_expr, operations, record_all_steps=True)
    display(Math(calculator.get_steps_latex()))

    display(Math("\\text{示例 3: 行操作}"))
    matrix_expr = "[[1,2,3],[4,5,6],[7,8,9]]"
    calculator.calculate(
        matrix_expr, operations, record_all_steps=True)
    display(Math(calculator.get_steps_latex()))


if __name__ == "__main__":
    demo()
