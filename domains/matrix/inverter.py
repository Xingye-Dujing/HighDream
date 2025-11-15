from sympy import latex, zeros, eye
from IPython.display import display, Math

from core import CommonMatrixCalculator


class Inverter(CommonMatrixCalculator):

    def is_square(self, matrix):
        """检查是否为方阵"""
        return matrix.rows == matrix.cols

    def is_invertible(self, matrix, show_steps=True, is_clear=True):
        """检查矩阵是否可逆"""
        if is_clear:
            self.step_generator.clear()

        if isinstance(matrix, str):
            matrix = self.parse_matrix_input(matrix)

        if not self.is_square(matrix):
            if show_steps:
                self.step_generator.add_step(r"\text{矩阵不是方阵, 不可逆}")
            return False

        det = matrix.det()
        if show_steps:
            self.add_step("检查可逆性:")
            self.add_matrix(matrix, "A")
            self.step_generator.add_step(f"\\det(A) = {latex(det)}")

        if det == 0:
            if show_steps:
                self.step_generator.add_step(r"\text{矩阵行列式为 0, 不可逆}")
            return False
        else:
            if show_steps:
                self.step_generator.add_step(r"\text{矩阵行列式不为 0, 可逆}")
            return True

    def check_special_matrix(self, matrix):
        """检查特殊矩阵类型"""
        n = matrix.rows

        # 检查是否为单位矩阵
        if matrix == eye(n):
            return "identity"

        # 检查是否为对角矩阵
        is_diagonal = True
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] != 0:
                    is_diagonal = False
                    break
            if not is_diagonal:
                break
        if is_diagonal:
            return "diagonal"

        # 检查是否为置换矩阵(每行每列只有一个1, 其余为 0)
        is_permutation = True
        for i in range(n):
            row_ones = 0
            col_ones = 0
            for j in range(n):
                if matrix[i, j] not in [0, 1]:
                    is_permutation = False
                    break
                if matrix[i, j] == 1:
                    row_ones += 1
                if matrix[j, i] == 1:
                    col_ones += 1
            if row_ones != 1 or col_ones != 1:
                is_permutation = False
                break
        if is_permutation:
            return "permutation"

        # 检查是否为三角矩阵
        is_upper_triangular = True
        is_lower_triangular = True
        for i in range(n):
            for j in range(n):
                if i > j and matrix[i, j] != 0:
                    is_upper_triangular = False
                if i < j and matrix[i, j] != 0:
                    is_lower_triangular = False
        if is_upper_triangular:
            return "upper_triangular"
        if is_lower_triangular:
            return "lower_triangular"

        return "general"

    def inverse_by_augmented(self, matrix_input, show_steps=True, simplify_result=True, is_clear=True):
        """方法一：增广矩阵法求逆"""
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法一: 增广矩阵法}")
            self.add_matrix(A, "A")

        # 检查是否可逆
        if not self.is_invertible(A, show_steps, is_clear=False):
            return None

        n = A.rows

        # 创建增广矩阵 [A | I]
        augmented = A.row_join(eye(n))
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|I]")

        # 高斯-约当消元
        for i in range(n):
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 处理第 {i+1} 列}}")

            # 寻找主元
            pivot_row = i
            for r in range(i, n):
                if augmented[r, i] != 0:
                    pivot_row = r
                    break

            # 行交换(如果需要)
            if pivot_row != i:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{行交换: }} R_{i+1} \\leftrightarrow  R_{pivot_row+1}")
                augmented.row_swap(i, pivot_row)
                if show_steps:
                    self.add_matrix(augmented, f"[A|I]")

            # 归一化主元行
            pivot = augmented[i, i]
            if pivot != 1:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{归一化: }} R_{i+1} \\times {latex(1/pivot)}")
                augmented[i, :] = augmented[i, :] / pivot
                if show_steps:
                    self.add_matrix(augmented, f"[A|I]")

            # 消元其他行
            for j in range(n):
                if j != i and augmented[j, i] != 0:
                    factor = augmented[j, i]
                    if show_steps:
                        self.step_generator.add_step(
                            f"\\text{{消元: }} R_{j+1} - {latex(factor)} \\times R_{i+1}"
                        )
                    augmented[j, :] = augmented[j, :] - \
                        factor * augmented[i, :]
                    if show_steps:
                        self.add_matrix(augmented, f"[A|I]")

        # 提取逆矩阵
        A_inv = augmented[:, n:]

        # 化简结果
        if simplify_result:
            A_inv_simplified = self.simplify_matrix(A_inv)
        else:
            A_inv_simplified = A_inv

        if show_steps:
            self.add_step("最终结果:")
            if simplify_result and A_inv != A_inv_simplified:
                self.add_matrix(A_inv, "A^{-1}_{\\text{未化简}}")
                self.add_matrix(A_inv_simplified, "A^{-1}")
            else:
                self.add_matrix(A_inv_simplified, "A^{-1}")

            # 验证
            self.add_step("验证:")
            identity_check = A * A_inv_simplified
            identity_check_simplified = self.simplify_matrix(identity_check)
            self.add_matrix(identity_check_simplified, "A \\times A^{-1}")
            if identity_check_simplified == eye(n):
                self.step_generator.add_step(
                    r"\text{验证通过: } A \times A^{-1} = I")
            else:
                self.step_generator.add_step(r"\text{验证失败}")

        return A_inv_simplified

    def inverse_by_adjugate(self, matrix_input, show_steps=True, simplify_result=True, is_clear=True):
        """方法二：伴随矩阵法求逆"""
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法二: 伴随矩阵法}")
            self.add_matrix(A, "A")

        # 检查是否可逆
        if not self.is_invertible(A, show_steps, is_clear=False):
            return None

        n = A.rows
        det_A = A.det()

        if show_steps:
            self.add_step("步骤 1: 计算行列式")
            self.step_generator.add_step(f"\\det(A) = {latex(det_A)}")

        # 计算余子式矩阵
        if show_steps:
            self.add_step("步骤 2: 计算余子式矩阵")

        cofactor_matrix = zeros(n)
        for i in range(n):
            for j in range(n):
                # 计算代数余子式
                minor = A.minor_submatrix(i, j)
                cofactor = (-1)**(i+j) * minor.det()
                cofactor_matrix[i, j] = cofactor

                if show_steps:
                    self.step_generator.add_step(
                        f"C_{{{i+1}{j+1}}} = (-1)^{{{i+j+2}}} \\cdot \\det(M_{{{i+1}{j+1}}}) = {latex(cofactor)}"
                    )
                    self.add_matrix(minor, f"M_{{{i+1}{j+1}}}")

        if show_steps:
            self.add_matrix(cofactor_matrix, "C")

        # 计算伴随矩阵(余子式矩阵的转置)
        adjugate = cofactor_matrix.T
        if show_steps:
            self.add_step("步骤 3: 计算伴随矩阵(余子式矩阵的转置)")
            self.add_matrix(adjugate, "\\text{adj}(A)=C^T")

        # 计算逆矩阵
        A_inv = adjugate / det_A

        # 化简结果
        if simplify_result:
            A_inv_simplified = self.simplify_matrix(A_inv)
        else:
            A_inv_simplified = A_inv

        if show_steps:
            self.add_step("步骤 4: 计算逆矩阵")
            self.step_generator.add_step(
                r"A^{-1} = \frac{1}{\det(A)} \cdot \text{adj}(A)")
            self.step_generator.add_step(
                f"A^{{-1}} = \\frac{{1}}{{{latex(det_A)}}} \\cdot {latex(adjugate)}")

            if simplify_result and A_inv != A_inv_simplified:
                self.add_matrix(A_inv, "A^{-1}_{\\text{未化简}}")
                self.add_matrix(A_inv_simplified, "A^{-1}")
            else:
                self.add_matrix(A_inv_simplified, "A^{-1}")

            # 验证
            self.add_step("验证:")
            identity_check = A * A_inv_simplified
            identity_check_simplified = self.simplify_matrix(identity_check)
            self.add_matrix(identity_check_simplified, "A \\times A^{-1}")
            if identity_check_simplified == eye(n):
                self.step_generator.add_step(
                    r"\text{验证通过: } A \times A^{-1} = I")
            else:
                self.step_generator.add_step(r"\text{验证失败}")

        return A_inv_simplified

    def inverse_by_lu_decomposition(self, matrix_input, show_steps=True, simplify_result=True, is_clear=True):
        """方法三: LU 分解法求逆"""
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法三: LU 分解法}")
            self.add_matrix(A, "A")

        # 检查是否可逆
        if not self.is_invertible(A, show_steps, is_clear=False):
            return None

        # 进行 LU 分解(使用 Doolittle 方法)
        n = A.rows

        L = eye(n)
        U = zeros(n)

        for i in range(n):
            # 计算 U 的第 i 行
            for j in range(i, n):
                sum_val = sum(L[i, k] * U[k, j] for k in range(i))
                U[i, j] = A[i, j] - sum_val

            # 计算 L 的第 i 列
            for j in range(i+1, n):
                sum_val = sum(L[j, k] * U[k, i] for k in range(i))
                if U[i, i] == 0:
                    self.step_generator.add_step(r"\textbf{不能进行 LU 分解}")
                    return None
                L[j, i] = (A[j, i] - sum_val) / U[i, i]

        if show_steps:
            self.add_step("步骤 1: 进行 LU 分解")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            # 验证 LU 分解
            self.add_step("验证 LU 分解:")
            LU_product = L * U
            self.add_matrix(LU_product, "L \\times U")
            if LU_product == A:
                self.step_generator.add_step(r"\text{LU 分解正确}")
            else:
                self.step_generator.add_step(r"\text{LU 分解错误}")

        # 解方程组 L * Y = I 和 U * X = Y 来求逆
        if show_steps:
            self.add_step("步骤 2: 解方程组求逆")
            self.step_generator.add_step(
                r"\text{解: } LY = I \quad \text{和} \quad UX = Y")
            self.step_generator.add_step(r"\text{其中 } X = A^{-1}")

        A_inv = zeros(n)

        # 对每一列求解
        for col in range(n):
            if show_steps:
                self.step_generator.add_step(f"\\text{{求解第 {col+1} 列}}")

            # 前代法解 L * y = e_col
            y = zeros(n, 1)
            e = zeros(n, 1)
            e[col] = 1

            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{(1) 前代法求解 }} L \\cdot \\boldsymbol{{y_{{{col+1}}}}} = \\boldsymbol{{e_{{{col+1}}}}}")
                self.add_matrix(e, f"\\boldsymbol{{e_{{{col+1}}}}}")

            for i in range(n):
                sum_val = sum(L[i, j] * y[j] for j in range(i))
                y[i] = (e[i] - sum_val) / L[i, i]

                if show_steps:
                    # 显示前代法每一步的计算过程
                    if i == 0:
                        self.step_generator.add_step(
                            f"y_{{{i+1}}} = \\frac{{e_{{{i+1}}}}}{{L_{{{i+1}{i+1}}}}} = \\frac{{{latex(e[i])}}}{{{latex(L[i, i])}}} = {latex(y[i])}")
                    else:
                        sum_terms = " + ".join(
                            [f"L_{{{i+1}{j+1}}} \\cdot y_{{{j+1}}}" for j in range(i)])
                        self.step_generator.add_step(
                            f"y_{{{i+1}}} = \\frac{{e_{{{i+1}}} - ({sum_terms})}}{{L_{{{i+1}{i+1}}}}} = \\frac{{{latex(e[i])} - ({latex(sum_val)})}}{{{latex(L[i, i])}}} = {latex(y[i])}")

            if show_steps:
                self.add_matrix(y, f"\\boldsymbol{{y_{{{col+1}}}}}")

            # 回代法解 U * x = y
            x = zeros(n, 1)
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{(2) 回代法求解 }} U \\cdot \\boldsymbol{{x_{{{col+1}}}}} = \\boldsymbol{{y_{{{col+1}}}}}")

            for i in range(n-1, -1, -1):
                sum_val = sum(U[i, j] * x[j] for j in range(i+1, n))
                x[i] = (y[i] - sum_val) / U[i, i]

                if show_steps:
                    if i == n-1:
                        self.step_generator.add_step(
                            f"x_{{{i+1}}} = \\frac{{y_{{{i+1}}}}}{{U_{{{i+1}{i+1}}}}} = \\frac{{{latex(y[i])}}}{{{latex(U[i, i])}}} = {latex(x[i])}")
                    else:
                        sum_terms = " + ".join(
                            [f"U_{{{i+1}{j+1}}} \\cdot x_{{{j+1}}}" for j in range(i+1, n)])
                        self.step_generator.add_step(
                            f"x_{{{i+1}}} = \\frac{{y_{{{i+1}}} - ({sum_terms})}}{{U_{{{i+1}{i+1}}}}} = \\frac{{{latex(y[i])} - ({latex(sum_val)})}}{{{latex(U[i, i])}}} = {latex(x[i])}")

            if show_steps:
                self.add_matrix(x, f"\\boldsymbol{{x_{{{col+1}}}}}")
                self.step_generator.add_step(f"\\text{{第 {col+1} 列求解完成}}")

            # 将解存入逆矩阵
            for i in range(n):
                A_inv[i, col] = x[i]

        # 化简结果
        if simplify_result:
            A_inv_simplified = self.simplify_matrix(A_inv)
        else:
            A_inv_simplified = A_inv

        if show_steps:
            self.add_step("最终结果:")
            if simplify_result and A_inv != A_inv_simplified:
                self.add_matrix(A_inv, "A^{-1}_{\\text{未化简}}")
                self.add_matrix(A_inv_simplified, "A^{-1}")
            else:
                self.add_matrix(A_inv_simplified, "A^{-1}")

            # 验证
            self.add_step("验证:")
            identity_check = A * A_inv_simplified
            identity_check_simplified = self.simplify_matrix(identity_check)
            self.add_matrix(identity_check_simplified, "A \\times A^{-1}")
            if identity_check_simplified == eye(n):
                self.step_generator.add_step(
                    r"\text{验证通过: } A \times A^{-1} = I")
            else:
                self.step_generator.add_step(r"\text{验证失败}")

        return A_inv_simplified

    def inverse_by_gauss_jordan(self, matrix_input, show_steps=True, simplify_result=True, is_clear=True):
        """方法四：高斯-约当消元法(直接求逆)"""
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法四: 高斯-约当消元法}")
            self.add_matrix(A, "A")

        # 检查是否可逆
        if not self.is_invertible(A, show_steps, is_clear=False):
            return None

        n = A.rows
        A_inv = eye(n)
        A_work = A.copy()

        if show_steps:
            self.add_step("初始状态:")
            self.add_matrix(A_work, "A")
            self.add_matrix(A_inv, "A^{-1}")

        for col in range(n):
            if show_steps:
                self.step_generator.add_step(f"\\text{{处理第 {col+1} 列}}")

            # 寻找主元
            pivot_row = col
            for r in range(col, n):
                if A_work[r, col] != 0:
                    pivot_row = r
                    break

            # 行交换
            if pivot_row != col:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{行交换: }} R_{{{col+1}}} \\leftrightarrow R_{{{pivot_row+1}}}")
                A_work.row_swap(col, pivot_row)
                A_inv.row_swap(col, pivot_row)
                if show_steps:
                    self.add_matrix(A_work, "A")
                    self.add_matrix(A_inv, "A^{-1}")

            # 归一化主元行
            pivot = A_work[col, col]
            if pivot != 1:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{归一化: }} \\frac{{1}}{{{latex(pivot)}}} \\cdot R_{{{col+1}}} \\rightarrow R_{{{col+1}}} ")
                A_work[col, :] = A_work[col, :] / pivot
                A_inv[col, :] = A_inv[col, :] / pivot
                if show_steps:
                    self.add_matrix(A_work, "A")
                    self.add_matrix(A_inv, "A^{-1}")

            # 消元其他行
            for row in range(n):
                if row != col and A_work[row, col] != 0:
                    factor = A_work[row, col]
                    if show_steps:
                        self.step_generator.add_step(
                            f"\\text{{消元: }} R_{{{row+1}}} - {latex(factor)} \\cdot R_{{{col+1}}} \\rightarrow R_{{{row+1}}}"
                        )
                    A_work[row, :] = A_work[row, :] - factor * A_work[col, :]
                    A_inv[row, :] = A_inv[row, :] - factor * A_inv[col, :]
                    if show_steps:
                        self.add_matrix(A_work, "A")
                        self.add_matrix(A_inv, "A^{-1}")

        # 化简结果
        if simplify_result:
            A_inv_simplified = self.simplify_matrix(A_inv)
        else:
            A_inv_simplified = A_inv

        if show_steps:
            self.add_step("最终结果:")
            if simplify_result and A_inv != A_inv_simplified:
                self.add_matrix(A_inv, "A^{-1}_{\\text{未化简}}")
                self.add_matrix(A_inv_simplified, "A^{-1}")
            else:
                self.add_matrix(A_inv_simplified, "A^{-1}")

            # 验证
            self.add_step("验证:")
            identity_check = A * A_inv_simplified
            identity_check_simplified = self.simplify_matrix(identity_check)
            self.add_matrix(identity_check_simplified, "A \\times A^{-1}")
            if identity_check_simplified == eye(n):
                self.step_generator.add_step(
                    r"\text{验证通过: } A \times A^{-1} = I")
            else:
                self.step_generator.add_step(r"\text{验证失败}")

        return A_inv_simplified

    def inverse_special_matrices(self, matrix_input, show_steps=True, simplify_result=True, is_clear=True):
        """特殊矩阵求逆方法"""
        if is_clear:
            self.step_generator.clear()

        english_to_chinese = {
            "identity": "单位矩阵",
            "diagonal": "对角矩阵",
            "permutation": "置换矩阵",
            "upper_triangular": "上三角矩阵",
            "lower_triangular": "下三角矩阵"
        }
        A = self.parse_matrix_input(matrix_input)
        matrix_type = self.check_special_matrix(A)

        if show_steps:
            self.step_generator.add_step(r"\textbf{特殊矩阵求逆}")
            self.add_matrix(A, "A")
            self.step_generator.add_step(
                f"\\text{{矩阵类型: }} {english_to_chinese[matrix_type]}")

        n = A.rows
        A_inv = None

        if matrix_type == "identity":
            if show_steps:
                self.step_generator.add_step(r"\text{单位矩阵的逆等于自身}")
            A_inv = eye(n)

        elif matrix_type == "diagonal":
            if show_steps:
                self.step_generator.add_step(r"\text{对角矩阵的逆是对角线元素的倒数}")
            A_inv = zeros(n)
            for i in range(n):
                A_inv[i, i] = 1 / A[i, i]
                if show_steps:
                    self.step_generator.add_step(
                        f"a_{{{i+1}{i+1}}}^{{-1}} = \\frac{{1}}{{{latex(A[i,i])}}} = {latex(A_inv[i,i])}")

        elif matrix_type == "permutation":
            if show_steps:
                self.step_generator.add_step(r"\text{置换矩阵的逆等于其转置}")
            A_inv = A.T
            if show_steps:
                self.add_matrix(A_inv, "A^T")

        elif matrix_type == "upper_triangular":
            if show_steps:
                self.step_generator.add_step(r"\text{上三角矩阵的逆(通过回代法求解)}")
            # 使用回代法求上三角矩阵的逆
            A_inv = zeros(n)
            for col in range(n):
                e = zeros(n, 1)
                e[col] = 1
                x = zeros(n, 1)

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{回代法求解 }} A \\cdot \\boldsymbol{{x_{{{col+1}}}}} = \\boldsymbol{{e_{{{col+1}}}}}")
                    self.add_matrix(e, f"\\boldsymbol{{e_{{{col+1}}}}}")

                # 回代过程
                for i in range(n-1, -1, -1):
                    sum_val = sum(A[i, j] * x[j] for j in range(i+1, n))
                    x[i] = (e[i] - sum_val) / A[i, i]

                    if show_steps:
                        if i == n-1:
                            self.step_generator.add_step(
                                f"x_{{{i+1}}} = \\frac{{e_{{{i+1}}}}}{{A_{{{i+1}{i+1}}}}} = \\frac{{{latex(e[i])}}}{{{latex(A[i, i])}}} = {latex(x[i])}")
                        else:
                            sum_terms = " + ".join(
                                [f"A_{{{i+1}{j+1}}} \\cdot x_{{{j+1}}}" for j in range(i+1, n)])
                            self.step_generator.add_step(
                                f"x_{{{i+1}}} = \\frac{{e_{{{i+1}}} - ({sum_terms})}}{{A_{{{i+1}{i+1}}}}} = \\frac{{{latex(e[i])} - ({latex(sum_val)})}}{{{latex(A[i, i])}}} = {latex(x[i])}")

                if show_steps:
                    self.add_matrix(x, f"\\boldsymbol{{x_{{{col+1}}}}}")
                    self.step_generator.add_step(f"\\text{{第 {col+1} 列求解完成}}")

                for i in range(n):
                    A_inv[i, col] = x[i]

        elif matrix_type == "lower_triangular":
            if show_steps:
                self.step_generator.add_step(r"\text{下三角矩阵的逆(通过前代法求解)}")
            # 使用前代法求下三角矩阵的逆
            A_inv = zeros(n)
            for col in range(n):
                e = zeros(n, 1)
                e[col] = 1
                x = zeros(n, 1)

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{前代法求解 }} A \\cdot \\boldsymbol{{x_{{{col+1}}}}} = \\boldsymbol{{e_{{{col+1}}}}}")
                    self.add_matrix(e, f"\\boldsymbol{{e_{{{col+1}}}}}")

                # 前代过程
                for i in range(n):
                    sum_val = sum(A[i, j] * x[j] for j in range(i))
                    x[i] = (e[i] - sum_val) / A[i, i]

                    if show_steps:
                        if i == 0:
                            self.step_generator.add_step(
                                f"x_{{{i+1}}} = \\frac{{e_{{{i+1}}}}}{{A_{{{i+1}{i+1}}}}} = \\frac{{{latex(e[i])}}}{{{latex(A[i, i])}}} = {latex(x[i])}")
                        else:
                            sum_terms = " + ".join(
                                [f"A_{{{i+1}{j+1}}} \\cdot x_{{{j+1}}}" for j in range(i)])
                            self.step_generator.add_step(
                                f"x_{{{i+1}}} = \\frac{{e_{{{i+1}}} - ({sum_terms})}}{{A_{{{i+1}{i+1}}}}} = \\frac{{{latex(e[i])} - ({latex(sum_val)})}}{{{latex(A[i, i])}}} = {latex(x[i])}")

                if show_steps:
                    self.add_matrix(x, f"\\boldsymbol{{x_{{{col+1}}}}}")
                    self.step_generator.add_step(f"\\text{{第 {col+1} 列求解完成}}")

                for i in range(n):
                    A_inv[i, col] = x[i]

        # 化简结果
        if simplify_result and A_inv is not None:
            A_inv_simplified = self.simplify_matrix(A_inv)
        else:
            A_inv_simplified = A_inv

        if A_inv_simplified is not None and show_steps:
            self.add_step("最终结果:")
            if simplify_result and A_inv != A_inv_simplified:
                self.add_matrix(A_inv, "A^{-1}_{\\text{未化简}}")
                self.add_matrix(A_inv_simplified, "A^{-1}")
            else:
                self.add_matrix(A_inv_simplified, "A^{-1}")

            # 验证
            self.add_step("验证:")
            identity_check = A * A_inv_simplified
            identity_check_simplified = self.simplify_matrix(identity_check)
            self.add_matrix(identity_check_simplified, "A \\times A^{-1}")
            if identity_check_simplified == eye(n):
                self.step_generator.add_step(
                    r"\text{验证通过: } A \times A^{-1} = I")
            else:
                self.step_generator.add_step(r"\text{验证失败}")

        return A_inv_simplified

    def auto_matrix_inverse(self, matrix_input, show_steps=True, simplify_result=True, is_clear=True):
        """自动选择求逆方法"""
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{自动矩阵求逆}")
            self.add_matrix(A, "A")

        # 检查特殊矩阵
        matrix_type = self.check_special_matrix(A)
        if matrix_type != "general":
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{检测到特殊矩阵: {matrix_type}, 使用特殊方法求逆}}")
            return self.inverse_special_matrices(matrix_input, show_steps, simplify_result, is_clear=False)

        # 对于一般矩阵, 提供多种方法
        if show_steps:
            self.step_generator.add_step(r"\text{检测到一般矩阵, 提供多种求逆方法}")

        results = {}

        # 方法1: 增广矩阵法
        try:
            results["augmented"] = self.inverse_by_augmented(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{增广矩阵法失败: {str(e)}}}")

        # 方法2: 伴随矩阵法
        try:
            results["adjugate"] = self.inverse_by_adjugate(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{伴随矩阵法失败: {str(e)}}}")

        # 方法3: LU分解法
        try:
            results["lu"] = self.inverse_by_lu_decomposition(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{LU 分解法失败: {str(e)}}}")

        # 方法4: 高斯-约当消元法
        try:
            results["gauss_jordan"] = self.inverse_by_gauss_jordan(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{高斯-约当消元法失败: {str(e)}}}")

        # 检查所有结果是否一致
        if show_steps and len(results) > 1:
            self.add_step("方法一致性检查:")
            methods = list(results.keys())
            consistent = True
            for i in range(len(methods)-1):
                if results[methods[i]] is None:
                    continue
                if results[methods[i+1]] is None:
                    continue
                if results[methods[i]] != results[methods[i+1]]:
                    consistent = False
                    break

            if consistent:
                self.step_generator.add_step(r"\text{所有方法结果一致}")
            else:
                self.step_generator.add_step(r"\text{警告: 不同方法结果不一致}")

        # 返回第一个成功的结果
        for _, result in results.items():
            if result is not None:
                return result

        return None


# 演示函数
def demo_basic_inverse():
    """演示基本矩阵求逆"""
    inverter = Inverter()

    # 可逆矩阵示例
    A1 = '[[0,1,1],[4,3,3],[8,7,9]]'
    A2 = '[[1,2,3],[0,1,4],[5,6,0]]'
    A3 = '[[1,1],[2,3]]'

    inverter.step_generator.add_step(r"\textbf{基本矩阵求逆演示}")

    test_matrices = [A1, A2, A3]

    for i, matrix in enumerate(test_matrices, 1):
        inverter.step_generator.add_step(f"\\textbf{{示例 {i}}}")
        try:
            inverter.auto_matrix_inverse(matrix)
            display(Math(inverter.get_steps_latex()))
        except Exception as e:
            inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(inverter.get_steps_latex()))


def demo_special_matrices():
    """演示特殊矩阵求逆"""
    inverter = Inverter()

    # 特殊矩阵示例
    identity = '[[1,0,0],[0,1,0],[0,0,1]]'
    diagonal = '[[2,0,0],[0,3,0],[0,0,5]]'
    permutation = '[[0,1,0],[0,0,1],[1,0,0]]'
    upper_triangular = '[[1,2,3],[0,4,5],[0,0,6]]'
    lower_triangular = '[[1,0,0],[2,3,0],[4,5,6]]'

    inverter.step_generator.add_step(r"\textbf{特殊矩阵求逆演示}")

    special_cases = [
        ("单位矩阵", identity),
        ("对角矩阵", diagonal),
        ("置换矩阵", permutation),
        ("上三角矩阵", upper_triangular),
        ("下三角矩阵", lower_triangular)
    ]

    for name, matrix in special_cases:
        inverter.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            inverter.inverse_special_matrices(matrix)
            display(Math(inverter.get_steps_latex()))
        except Exception as e:
            inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(inverter.get_steps_latex()))


def demo_singular_matrix():
    """演示奇异矩阵情况"""
    inverter = Inverter()

    # 奇异矩阵示例
    singular1 = '[[1,2,3],[4,5,6],[7,8,9]]'  # 行线性相关
    singular2 = '[[1,1],[1,1]]'  # 两行相同

    inverter.step_generator.add_step(r"\textbf{奇异矩阵演示}")

    singular_matrices = [singular1, singular2]

    for i, matrix in enumerate(singular_matrices, 1):
        inverter.step_generator.add_step(f"\\textbf{{奇异矩阵示例 {i}}}")
        try:
            inverter.is_invertible(matrix)
            display(Math(inverter.get_steps_latex()))
        except Exception as e:
            inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(inverter.get_steps_latex()))


def demo_symbolic_matrix():
    """演示符号矩阵求逆"""
    inverter = Inverter()

    # 符号矩阵
    symbolic_2x2 = '[[a,b],[c,d]]'
    symbolic_3x3 = '[[a,b,c],[d,e,f],[g,h,i]]'

    inverter.step_generator.add_step(r"\textbf{符号矩阵求逆演示}")
    inverter.step_generator.add_step(r"\textbf{假设所有符号表达式不为 0, 可作分母}")

    inverter.step_generator.add_step(r"\textbf{2×2 符号矩阵}")
    try:
        inverter.auto_matrix_inverse(symbolic_2x2)
        display(Math(inverter.get_steps_latex()))
    except Exception as e:
        inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(inverter.get_steps_latex()))

    inverter.step_generator.add_step(r"\textbf{3×3 符号矩阵}")
    try:
        inverter.auto_matrix_inverse(symbolic_3x3)
        display(Math(inverter.get_steps_latex()))
    except Exception as e:
        inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(inverter.get_steps_latex()))


if __name__ == "__main__":
    # 运行演示
    demo_basic_inverse()
    demo_special_matrices()
    demo_singular_matrix()
    demo_symbolic_matrix()
