from sympy import Matrix, eye, latex, zeros
# from IPython.display import Math, display

from core import CommonMatrixCalculator


class Inverter(CommonMatrixCalculator):
    """Matrix inversion calculator with multiple methods.

    This class provides various approaches to compute matrix inverses including
    augmented matrix method, adjugate method, LU decomposition method, and
    Gauss-Jordan elimination method. It also handles special matrix types.
    """

    def is_square(self, matrix: Matrix) -> bool:
        """Check if a matrix is square."""
        return matrix.rows == matrix.cols

    def is_invertible(self, matrix: Matrix, show_steps: bool = True, is_clear: bool = True) -> bool:
        """Check if a matrix is invertible by computing its determinant.

        A matrix is invertible if and only if its determinant is non-zero.
        """
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

        if show_steps:
            self.step_generator.add_step(r"\text{矩阵行列式不为 0, 可逆}")
        return True

    def check_special_matrix(self, matrix: Matrix) -> str:
        """Identify special types of matrices that have optimized inversion methods.


        Returns:
            str: Type of special matrix ('identity', 'diagonal', 'permutation',
                 'upper_triangular', 'lower_triangular', or 'general')
        """
        n = matrix.rows

        # Check if it's an identity matrix
        if matrix == eye(n):
            return "identity"

        # Check if it's a diagonal matrix
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

        # Check if it's a permutation matrix (only one 1 per row/column, rest 0)
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

        # Check if it's a triangular matrix
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

    def inverse_by_augmented(self, matrix_input: str, show_steps: bool = True, simplify_result: bool = True, is_clear: bool = True) -> Matrix:
        """Compute matrix inverse using the augmented matrix method.

        This method creates an augmented matrix [A|I] and applies Gauss-Jordan
        elimination to transform it into [I|A^-1].

        Returns:
            Matrix: The inverse of the input matrix, or None if not invertible
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法一: 增广矩阵法}")
            self.add_matrix(A, "A")

        # Check if matrix is invertible
        if not self.is_invertible(A, show_steps, is_clear=False):
            return None

        n = A.rows

        # Create augmented matrix [A | I]
        augmented = A.row_join(eye(n))
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|I]")

        # Gauss-Jordan elimination
        for i in range(n):
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 处理第 {i+1} 列}}")

            # Find pivot element
            pivot_row = i
            for r in range(i, n):
                if augmented[r, i] != 0:
                    pivot_row = r
                    break

            # Row swap if needed
            if pivot_row != i:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{行交换: }} R_{i+1} \\leftrightarrow  R_{pivot_row+1}")
                augmented.row_swap(i, pivot_row)
                if show_steps:
                    self.add_matrix(augmented, f"[A|I]")

            # Normalize pivot row
            pivot = augmented[i, i]
            if pivot != 1:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{归一化: }} R_{i+1} \\times {latex(1/pivot)}")
                augmented[i, :] = augmented[i, :] / pivot
                if show_steps:
                    self.add_matrix(augmented, f"[A|I]")

            # Eliminate other rows
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

        # Extract inverse matrix from right half of augmented matrix
        A_inv = augmented[:, n:]

        # Simplify result if requested
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

            # Verification
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

    def inverse_by_adjugate(self, matrix_input: str, show_steps: bool = True, simplify_result: bool = True, is_clear: bool = True) -> Matrix:
        """
        Compute matrix inverse using the adjugate method.

        This method uses the formula: A^-1 = (1/det(A)) * adj(A)
        where adj(A) is the adjugate matrix (transpose of cofactor matrix).

        Returns:
            Matrix: The inverse of the input matrix, or None if not invertible
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法二: 伴随矩阵法}")
            self.add_matrix(A, "A")

        # Check if matrix is invertible
        if not self.is_invertible(A, show_steps, is_clear=False):
            return None

        n = A.rows
        det_A = A.det()

        if show_steps:
            self.add_step("步骤 1: 计算行列式")
            self.step_generator.add_step(f"\\det(A) = {latex(det_A)}")

        # Calculate cofactor matrix
        if show_steps:
            self.add_step("步骤 2: 计算余子式矩阵")

        cofactor_matrix = zeros(n)
        for i in range(n):
            for j in range(n):
                # Calculate algebraic cofactor
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

        # Calculate adjugate matrix (transpose of cofactor matrix)
        adjugate = cofactor_matrix.T
        if show_steps:
            self.add_step("步骤 3: 计算伴随矩阵(余子式矩阵的转置)")
            self.add_matrix(adjugate, "\\text{adj}(A)=C^T")

        # Calculate inverse matrix
        A_inv = adjugate / det_A

        # Simplify result if requested
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

            # Verification
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

    def inverse_by_lu_decomposition(self, matrix_input: str, show_steps: bool = True, simplify_result: bool = True, is_clear: bool = True) -> Matrix:
        """Compute matrix inverse using LU decomposition method.

        This method decomposes A into L*U and then solves AX = I column by column,
        where X will be A^-1.

        Returns:
            Matrix: The inverse of the input matrix, or None if not invertible
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法三: LU 分解法}")
            self.add_matrix(A, "A")

        # Check if matrix is invertible
        if not self.is_invertible(A, show_steps, is_clear=False):
            return None

        # Perform LU decomposition using Doolittle method
        n = A.rows

        L = eye(n)
        U = zeros(n)

        for i in range(n):
            # Calculate U's i-th row
            for j in range(i, n):
                sum_val = sum(L[i, k] * U[k, j] for k in range(i))
                U[i, j] = A[i, j] - sum_val

            # Calculate L's i-th column
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

            # Verify LU decomposition
            self.add_step("验证 LU 分解:")
            LU_product = L * U
            self.add_matrix(LU_product, "L \\times U")
            if LU_product == A:
                self.step_generator.add_step(r"\text{LU 分解正确}")
            else:
                self.step_generator.add_step(r"\text{LU 分解错误}")

        # Solve equation systems L * Y = I and U * X = Y to find inverse
        if show_steps:
            self.add_step("步骤 2: 解方程组求逆")
            self.step_generator.add_step(
                r"\text{解: } LY = I \quad \text{和} \quad UX = Y")
            self.step_generator.add_step(r"\text{其中 } X = A^{-1}")

        A_inv = zeros(n)

        # Solve for each column
        for col in range(n):
            if show_steps:
                self.step_generator.add_step(f"\\text{{求解第 {col+1} 列}}")

            # Forward substitution to solve L * y = e_col
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
                    # Show forward substitution steps
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

            # Backward substitution to solve U * x = y
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

            # Store solution in inverse matrix
            for i in range(n):
                A_inv[i, col] = x[i]

        # Simplify result if requested
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

            # Verification
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

    def inverse_by_gauss_jordan(self, matrix_input: str, show_steps: bool = True, simplify_result: bool = True, is_clear: bool = True) -> Matrix:
        """Compute matrix inverse using Gauss-Jordan elimination method.

        Directly applies Gauss-Jordan elimination on matrix A while simultaneously
        applying the same operations on an identity matrix to produce A^-1.

        Returns:
            Matrix: The inverse of the input matrix, or None if not invertible
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法四: 高斯-约当消元法}")
            self.add_matrix(A, "A")

        # Check if matrix is invertible
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

            # Find pivot element
            pivot_row = col
            for r in range(col, n):
                if A_work[r, col] != 0:
                    pivot_row = r
                    break

            # Row swap
            if pivot_row != col:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{行交换: }} R_{{{col+1}}} \\leftrightarrow R_{{{pivot_row+1}}}")
                A_work.row_swap(col, pivot_row)
                A_inv.row_swap(col, pivot_row)
                if show_steps:
                    self.add_matrix(A_work, "A")
                    self.add_matrix(A_inv, "A^{-1}")

            # Normalize pivot row
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

            # Eliminate other rows
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

        # Simplify result if requested
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

            # Verification
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

    def inverse_special_matrices(self, matrix_input: str, show_steps: bool = True, simplify_result: bool = True, is_clear: bool = True) -> Matrix:
        """Compute inverse of special matrices using optimized methods.

        Handles specific matrix types like identity, diagonal, permutation,
        triangular matrices with specialized algorithms for better efficiency.

        Returns:
            Matrix: The inverse of the input matrix, or None if not invertible
        """
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
            # Use backward substitution to find inverse of upper triangular matrix
            A_inv = zeros(n)
            for col in range(n):
                e = zeros(n, 1)
                e[col] = 1
                x = zeros(n, 1)

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{回代法求解 }} A \\cdot \\boldsymbol{{x_{{{col+1}}}}} = \\boldsymbol{{e_{{{col+1}}}}}")
                    self.add_matrix(e, f"\\boldsymbol{{e_{{{col+1}}}}}")

                # Backward substitution process
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
            # Use forward substitution to find inverse of lower triangular matrix
            A_inv = zeros(n)
            for col in range(n):
                e = zeros(n, 1)
                e[col] = 1
                x = zeros(n, 1)

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{前代法求解 }} A \\cdot \\boldsymbol{{x_{{{col+1}}}}} = \\boldsymbol{{e_{{{col+1}}}}}")
                    self.add_matrix(e, f"\\boldsymbol{{e_{{{col+1}}}}}")

                # Forward substitution process
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

        # Simplify result if requested
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

            # Verification
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

    def auto_matrix_inverse(self, matrix_input: str, show_steps: bool = True, simplify_result: bool = True, is_clear: bool = True) -> Matrix:
        """
        Automatically select the best method to compute matrix inverse.

        This function analyzes the matrix type and chooses the most appropriate
        inversion method. For special matrices, optimized methods are used.
        For general matrices, multiple methods are attempted.

        Returns:
            Matrix: The inverse of the input matrix, or None if not invertible
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{自动矩阵求逆}")
            self.add_matrix(A, "A")

        # Check for special matrix types
        matrix_type = self.check_special_matrix(A)
        if matrix_type != "general":
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{检测到特殊矩阵: {matrix_type}, 使用特殊方法求逆}}")
            return self.inverse_special_matrices(matrix_input, show_steps, simplify_result, is_clear=False)

        # For general matrices, provide multiple methods
        if show_steps:
            self.step_generator.add_step(r"\text{检测到一般矩阵, 提供多种求逆方法}")

        results = {}

        # Method 1: Augmented matrix method
        try:
            results["augmented"] = self.inverse_by_augmented(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{增广矩阵法失败: {str(e)}}}")

        # Method 2: Adjugate method
        try:
            results["adjugate"] = self.inverse_by_adjugate(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{伴随矩阵法失败: {str(e)}}}")

        # Method 3: LU decomposition method
        try:
            results["lu"] = self.inverse_by_lu_decomposition(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{LU 分解法失败: {str(e)}}}")

        # Method 4: Gauss-Jordan elimination method
        try:
            results["gauss_jordan"] = self.inverse_by_gauss_jordan(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{高斯-约当消元法失败: {str(e)}}}")

        # Check consistency of results
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

        # Return first successful result
        for _, result in results.items():
            if result is not None:
                return result

        return None


# # Demo functions
# def demo_basic_inverse():
#     """
#     Demonstrate basic matrix inversion with regular matrices.

#     Shows how to use the auto_matrix_inverse method with various examples.
#     """
#     inverter = Inverter()

#     # Invertible matrix examples
#     A1 = '[[0,1,1],[4,3,3],[8,7,9]]'
#     A2 = '[[1,2,3],[0,1,4],[5,6,0]]'
#     A3 = '[[1,1],[2,3]]'

#     inverter.step_generator.add_step(r"\textbf{基本矩阵求逆演示}")

#     test_matrices = [A1, A2, A3]

#     for i, matrix in enumerate(test_matrices, 1):
#         inverter.step_generator.add_step(f"\\textbf{{示例 {i}}}")
#         try:
#             inverter.auto_matrix_inverse(matrix)
#             display(Math(inverter.get_steps_latex()))
#         except Exception as e:
#             inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(inverter.get_steps_latex()))


# def demo_special_matrices():
#     """
#     Demonstrate matrix inversion with special matrix types.

#     Shows optimized inversion methods for identity, diagonal, permutation,
#     and triangular matrices.
#     """
#     inverter = Inverter()

#     # Special matrix examples
#     identity = '[[1,0,0],[0,1,0],[0,0,1]]'
#     diagonal = '[[2,0,0],[0,3,0],[0,0,5]]'
#     permutation = '[[0,1,0],[0,0,1],[1,0,0]]'
#     upper_triangular = '[[1,2,3],[0,4,5],[0,0,6]]'
#     lower_triangular = '[[1,0,0],[2,3,0],[4,5,6]]'

#     inverter.step_generator.add_step(r"\textbf{特殊矩阵求逆演示}")

#     special_cases = [
#         ("单位矩阵", identity),
#         ("对角矩阵", diagonal),
#         ("置换矩阵", permutation),
#         ("上三角矩阵", upper_triangular),
#         ("下三角矩阵", lower_triangular)
#     ]

#     for name, matrix in special_cases:
#         inverter.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             inverter.inverse_special_matrices(matrix)
#             display(Math(inverter.get_steps_latex()))
#         except Exception as e:
#             inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(inverter.get_steps_latex()))


# def demo_singular_matrix():
#     """
#     Demonstrate handling of singular (non-invertible) matrices.

#     Shows how the system detects and handles matrices with zero determinant.
#     """
#     inverter = Inverter()

#     # Singular matrix examples
#     singular1 = '[[1,2,3],[4,5,6],[7,8,9]]'  # Linearly dependent rows
#     singular2 = '[[1,1],[1,1]]'  # Identical rows

#     inverter.step_generator.add_step(r"\textbf{奇异矩阵演示}")

#     singular_matrices = [singular1, singular2]

#     for i, matrix in enumerate(singular_matrices, 1):
#         inverter.step_generator.add_step(f"\\textbf{{奇异矩阵示例 {i}}}")
#         try:
#             inverter.is_invertible(matrix)
#             display(Math(inverter.get_steps_latex()))
#         except Exception as e:
#             inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(inverter.get_steps_latex()))


# def demo_symbolic_matrix():
#     """
#     Demonstrate matrix inversion with symbolic expressions.

#     Shows how the system handles matrices containing variables.
#     """
#     inverter = Inverter()

#     # Symbolic matrices
#     symbolic_2x2 = '[[a,b],[c,d]]'
#     symbolic_3x3 = '[[a,b,c],[d,e,f],[g,h,i]]'

#     inverter.step_generator.add_step(r"\textbf{符号矩阵求逆演示}")
#     inverter.step_generator.add_step(r"\textbf{假设所有符号表达式不为 0, 可作分母}")

#     inverter.step_generator.add_step(r"\textbf{2×2 符号矩阵}")
#     try:
#         inverter.auto_matrix_inverse(symbolic_2x2)
#         display(Math(inverter.get_steps_latex()))
#     except Exception as e:
#         inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#         display(Math(inverter.get_steps_latex()))

#     inverter.step_generator.add_step(r"\textbf{3×3 符号矩阵}")
#     try:
#         inverter.auto_matrix_inverse(symbolic_3x3)
#         display(Math(inverter.get_steps_latex()))
#     except Exception as e:
#         inverter.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#         display(Math(inverter.get_steps_latex()))


# if __name__ == "__main__":
#     # Run demonstrations
#     demo_basic_inverse()
#     demo_special_matrices()
#     demo_singular_matrix()
#     demo_symbolic_matrix()
