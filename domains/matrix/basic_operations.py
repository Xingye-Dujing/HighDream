from sympy import Matrix, sympify, latex, zeros, symbols
from IPython.display import display, Math

from core import CommonMatrixCalculator


class BasicOperations(CommonMatrixCalculator):

    def matrix_addition(self, A_input, B_input, show_steps=True, is_clear=True):
        """
        矩阵加法 A + B
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(A_input)
        B = self.parse_matrix_input(B_input)

        if A.shape != B.shape:
            self.add_step(f"矩阵维度不匹配: $A{A.shape} + B{B.shape}$")
            return

        if show_steps:
            self.add_step("矩阵加法: $A + B$")
            self.add_matrix(A, "A")
            self.add_matrix(B, "B")

            # 显示维度信息
            self.step_generator.add_step(
                f"\\text{{维度: }} {A.rows} \\times {A.cols}")

            # 显示逐元素相加过程
            self.add_step("逐元素相加过程")
            result = zeros(A.rows, A.cols)

            for i in range(A.rows):
                row_steps = []
                for j in range(A.cols):
                    part_1 = f"{latex(A[i, j])} + {latex(B[i, j])}"
                    part_2 = latex(A[i, j] + B[i, j])
                    part = part_1 if part_1 == part_2 else f"{part_1} = {part_2}"
                    step = f"a_{{{i+1}{j+1}}} + b_{{{i+1}{j+1}}} = {part}"
                    row_steps.append(step)
                    result[i, j] = A[i, j] + B[i, j]

                # 显示当前行的计算过程
                self.step_generator.add_step(r",\quad ".join(row_steps))

        # 最终结果
        result = A + B
        if show_steps:
            self.add_step("最终结果")
            self.add_matrix(result, "A + B")

        return result

    def matrix_subtraction(self, A_input, B_input, show_steps=True, is_clear=True):
        """
        矩阵减法 A - B
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(A_input)
        B = self.parse_matrix_input(B_input)

        if A.shape != B.shape:
            self.add_step(f"矩阵维度不匹配: $A{A.shape} - B{B.shape}$")
            return

        if show_steps:
            self.add_step("矩阵减法: A $-$ B")
            self.add_matrix(A, "A")
            self.add_matrix(B, "B")

            # 显示维度信息
            self.step_generator.add_step(
                f"\\text{{维度: }} {A.rows} \\times {A.cols}")

            # 显示逐元素相减过程
            self.add_step("逐元素相减过程")
            result = zeros(A.rows, A.cols)

            for i in range(A.rows):
                row_steps = []
                for j in range(A.cols):
                    part_1 = f"{latex(A[i, j])} - {latex(B[i, j])}"
                    part_2 = latex(A[i, j] - B[i, j])
                    part = part_1 if part_1 == part_2 else f"{part_1} = {part_2}"
                    step = f"a_{{{i+1}{j+1}}} - b_{{{i+1}{j+1}}} = {part}"
                    row_steps.append(step)
                    result[i, j] = A[i, j] - B[i, j]

                # 显示当前行的计算过程
                self.step_generator.add_step(r",\quad ".join(row_steps))

        # 最终结果
        result = A - B
        if show_steps:
            self.add_step("最终结果")
            self.add_matrix(result, "A - B")

        return result

    def matrix_multiplication(self, A_input, B_input, show_steps=True, is_clear=True):
        """
        矩阵乘法 A * B
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(A_input)
        B = self.parse_matrix_input(B_input)

        if A.cols != B.rows:
            self.add_step(f"矩阵维度不兼容: $A{A.shape} \\times B{B.shape}$")
            return

        if show_steps:
            self.add_step("矩阵乘法: A $\\times$ B")
            self.add_matrix(A, "A")
            self.add_matrix(B, "B")

            # 显示维度信息
            self.step_generator.add_step(
                f"\\text{{维度: }} A: {A.rows} \\times {A.cols}, \\quad B: {B.rows} \\times {B.cols}")
            self.step_generator.add_step(
                f"\\text{{结果维度: }} {A.rows} \\times {B.cols}")

            # 显示乘法过程
            self.add_step("乘法过程 (行 × 列)")
            result = zeros(A.rows, B.cols)

            for i in range(A.rows):
                for j in range(B.cols):
                    # 计算当前元素
                    steps = []
                    total = 0

                    for k in range(A.cols):
                        product = A[i, k] * B[k, j]
                        step = f"a_{{{i+1}{k+1}}} \\cdot b_{{{k+1}{j+1}}} = {latex(A[i,k])} \\cdot {latex(B[k,j])} = {latex(product)}"
                        steps.append(step)
                        total += product

                    # 显示当前元素的计算过程
                    self.step_generator.add_step(f"c_{{{i+1}{j+1}}} = " + " + ".join(
                        [f"a_{{{i+1}{k+1}}} \\cdot b_{{{k+1}{j+1}}}" for k in range(A.cols)]))
                    part_1 = f"= " + \
                        " + ".join([latex(A[i, k] * B[k, j])
                                   for k in range(A.cols)])
                    part_2 = f"= {latex(total)}"
                    self.step_generator.add_step(part_1)
                    if part_1 != part_2:
                        self.step_generator.add_step(part_2)

                    result[i, j] = total

        # 最终结果
        result = A * B
        if show_steps:
            self.add_step("最终结果")
            self.add_matrix(result, "A \\times B")

        return result

    def scalar_multiplication(self, scalar_input, matrix_input, show_steps=True, is_clear=True):
        """
        标量乘法 k * A
        """
        if is_clear:
            self.step_generator.clear()

        scalar = sympify(scalar_input)
        if not scalar.is_number:
            self.add_step(f"标量输入错误: {scalar_input}")
            return
        matrix = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_step("标量乘法: $k \\times A$")
            self.step_generator.add_step(f"k = {latex(scalar)}")
            self.add_matrix(matrix, "A")

            # 显示维度信息
            self.step_generator.add_step(
                f"\\text{{维度: }} {matrix.rows} \\times {matrix.cols}")

            # 显示逐元素相乘过程
            self.add_step("逐元素相乘过程")
            result = zeros(matrix.rows, matrix.cols)

            for i in range(matrix.rows):
                row_steps = []
                for j in range(matrix.cols):
                    step = f"k \\cdot a_{{{i+1}{j+1}}} = {latex(scalar)} \\cdot {latex(matrix[i,j])} = {latex(scalar * matrix[i, j])}"
                    row_steps.append(step)
                    result[i, j] = scalar * matrix[i, j]

                # 显示当前行的计算过程
                self.step_generator.add_step(r",\quad ".join(row_steps))

        # 最终结果
        result = scalar * matrix
        if show_steps:
            self.add_step("最终结果")
            self.add_matrix(result, f"{latex(scalar)} \\times A")

        return result

    def transpose(self, matrix_input, show_steps=True, is_clear=True):
        """
        矩阵转置 A^T
        """
        if is_clear:
            self.step_generator.clear()

        matrix = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_step("矩阵转置: $A^T$")
            self.add_matrix(matrix, "A")

            # 显示维度信息
            self.step_generator.add_step(
                f"\\text{{原矩阵维度: }} {matrix.rows} \\times {matrix.cols}")
            self.step_generator.add_step(
                f"\\text{{转置矩阵维度: }} {matrix.cols} \\times {matrix.rows}")

            # 显示转置过程
            self.add_step("转置过程(行变列，列变行)")

            for i in range(matrix.rows):
                for j in range(matrix.cols):
                    self.step_generator.add_step(
                        f"a^T_{{{j+1}{i+1}}} = a_{{{i+1}{j+1}}} = {latex(matrix[i,j])}")

        # 最终结果
        result = matrix.T
        if show_steps:
            self.add_step("最终结果")
            self.add_matrix(result, "A^T")

        return result

    def dot_product(self, vectorA_input, vectorB_input, show_steps=True, is_clear=True):
        """
        向量点积 A dot B
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(vectorA_input)
        B = self.parse_matrix_input(vectorB_input)

        # 确保是向量
        if not (A.rows == 1 or A.cols == 1) or not (B.rows == 1 or B.cols == 1):
            self.add_step("输入必须是向量")
            return

        # 转换为列向量形式
        if A.rows == 1:
            A = A.T
        if B.rows == 1:
            B = B.T

        if A.rows != B.rows:
            self.add_step(f"向量维度不匹配: A{A.shape} · B{B.shape}")
            return

        if show_steps:
            self.add_step("向量点积: A · B")
            self.add_vector(A, "A")
            self.add_vector(B, "B")

            # 显示维度信息
            self.step_generator.add_step(f"\\text{{向量长度: }} {A.rows}")

            # 显示点积计算过程
            self.add_step("点积计算过程")
            steps = ''
            total = 0

            for i in range(A.rows):
                step = f"a_{{{i+1}}} \\cdot b_{{{i+1}}}"
                total += A[i, 0] * B[i, 0]

                if i == 0:
                    steps += f"A \\cdot B = {step}"
                else:
                    steps += (f"+ {step}")

            self.step_generator.add_step(steps)
            self.step_generator.add_step(
                f"= " + " + ".join([latex(A[i, 0] * B[i, 0]) for i in range(A.rows)]))
            self.step_generator.add_step(f"= {latex(total)}")

        # 最终结果
        result = A.dot(B)
        if show_steps:
            self.add_step("最终结果")
            self.step_generator.add_step(f"A \\cdot B = {latex(result)}")

        return result

    def cross_product(self, vectorA_input, vectorB_input, show_steps=True, is_clear=True):
        """
        三维向量叉积 A × B
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(vectorA_input)
        B = self.parse_matrix_input(vectorB_input)

        # 确保是三维向量
        if A.shape != (3, 1) and A.shape != (1, 3):
            self.add_step("叉积只适用于三维向量")
            return
        if B.shape != (3, 1) and B.shape != (1, 3):
            self.add_step("叉积只适用于三维向量")
            return

        # 转换为列向量
        if A.rows == 1:
            A = A.T
        if B.rows == 1:
            B = B.T

        if show_steps:
            self.add_step("向量叉积: A × B")
            self.add_vector(A, "A")
            self.add_vector(B, "B")

            # 显示叉积公式
            self.add_step("叉积公式")
            self.step_generator.add_step(
                r"A \times B = \begin{vmatrix} \boldsymbol{i} & \boldsymbol{j} & \boldsymbol{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix}")

            # 显示计算过程
            self.add_step("计算过程")

            # i 分量
            i_component = A[1, 0]*B[2, 0] - A[2, 0]*B[1, 0]
            self.step_generator.add_step(
                f"i_{{}} = (a_2 \\cdot b_3 - a_3 \\cdot b_2) = ({latex(A[1,0])} \\cdot {latex(B[2,0])} - {latex(A[2,0])} \\cdot {latex(B[1,0])}) = {latex(i_component)}")

            # j 分量
            j_component = A[2, 0]*B[0, 0] - A[0, 0]*B[2, 0]
            self.step_generator.add_step(
                f"j_{{}} = (a_3 \\cdot b_1 - a_1 \\cdot b_3) = ({latex(A[2,0])} \\cdot {latex(B[0,0])} - {latex(A[0,0])} \\cdot {latex(B[2,0])}) = {latex(j_component)}")

            # k 分量
            k_component = A[0, 0]*B[1, 0] - A[1, 0]*B[0, 0]
            self.step_generator.add_step(
                f"k_{{}} = (a_1 \\cdot b_2 - a_2 \\cdot b_1) = ({latex(A[0,0])} \\cdot {latex(B[1,0])} - {latex(A[1,0])} \\cdot {latex(B[0,0])}) = {latex(k_component)}")

        # 最终结果
        result = A.cross(B)
        if show_steps:
            self.add_step("最终结果")
            self.add_vector(result, "A \\times B")

        return result

    def compute(self, expression, operation):
        expression = expression.split('\n')
        if operation in ['add', '+']:
            self.matrix_addition(expression[0], expression[1])
        elif operation in ['subtract', '-']:
            self.matrix_subtraction(expression[0], expression[1])
        elif operation in ['multiply', '*']:
            self.matrix_multiplication(expression[0], expression[1])
        elif operation in ['scalar_multiply', 'scalar_mul']:
            self.scalar_multiplication(expression[0], expression[1])
        elif operation in ['dot_product', 'dot']:
            self.dot_product(expression[0], expression[1])
        elif operation in ['cross_product', 'cross']:
            self.cross_product(expression[0], expression[1])
        elif operation in ['transpose', 'T']:
            self.transpose(expression[0])
        return self.get_steps_latex()


def demo():
    """演示所有基础运算"""
    ops = BasicOperations()

    # 示例矩阵
    A_str = '[[1,2,3],[4,5,6],[7,8,9]]'
    B_str = '[[9,8,7],[6,5,4],[3,2,1]]'
    C_str = '[[1,0,2],[-1,3,1]]'
    D_str = '[[3,1],[2,1],[1,0]]'

    # 示例向量
    v1_str = '[1,2,3]'
    v2_str = '[4,5,6]'

    # 1. 矩阵加法
    ops.step_generator.add_step(r"\textbf{1. 矩阵加法}")
    try:
        ops.matrix_addition(A_str, B_str)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    # 2. 矩阵减法
    ops.step_generator.clear()
    ops.step_generator.add_step(r"\textbf{2. 矩阵减法}")
    try:
        ops.matrix_subtraction(A_str, B_str)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    # 3. 矩阵乘法
    ops.step_generator.clear()
    ops.step_generator.add_step(r"\textbf{3. 矩阵乘法}")
    try:
        ops.matrix_multiplication(C_str, D_str)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    # 4. 标量乘法
    ops.step_generator.clear()
    ops.step_generator.add_step(r"\textbf{4. 标量乘法}")
    try:
        ops.scalar_multiplication(2, A_str)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    # 5. 矩阵转置
    ops.step_generator.clear()
    ops.step_generator.add_step(r"\textbf{5. 矩阵转置}")
    try:
        ops.transpose(C_str)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    # 6. 向量点积
    ops.step_generator.clear()
    ops.step_generator.add_step(r"\textbf{6. 向量点积}")
    try:
        ops.dot_product(v1_str, v2_str)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    # 7. 向量叉积
    ops.step_generator.clear()
    ops.step_generator.add_step(r"\textbf{7. 向量叉积}")
    try:
        ops.cross_product(v1_str, v2_str)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))


def demo_symbolic():
    """演示符号运算"""
    ops = BasicOperations()

    a, b, c, d = symbols('a b c d')

    A_sym = Matrix([[a, b], [c, 3]])
    B_sym = Matrix([[1, b], [d, 4]])

    ops.step_generator.add_step(r"\textbf{符号矩阵运算}")

    # 符号矩阵加法
    ops.step_generator.add_step(r"\textbf{符号矩阵加法}")
    try:
        ops.matrix_addition(A_sym, B_sym)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    ops.step_generator.add_step(r"\textbf{符号矩阵减法}")
    try:
        ops.matrix_subtraction(A_sym, B_sym)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    ops.step_generator.add_step(r"\textbf{符号矩阵标量乘法}")
    try:
        ops.scalar_multiplication(2, A_sym)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    # 符号矩阵乘法
    ops.step_generator.add_step(r"\textbf{符号矩阵乘法}")
    try:
        ops.matrix_multiplication(
            A_sym, B_sym)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))

    ops.step_generator.add_step(r"\textbf{符号矩阵转置}")
    try:
        ops.transpose(A_sym)
        display(Math(ops.get_steps_latex()))
    except Exception as e:
        ops.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(ops.get_steps_latex()))


if __name__ == "__main__":
    # 运行数值演示
    demo()
    # 运行符号演示
    demo_symbolic()
