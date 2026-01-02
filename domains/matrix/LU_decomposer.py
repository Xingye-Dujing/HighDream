from typing import Tuple
from sympy import Matrix, Symbol, eye, latex, zeros
# from sympy import symbols
# from IPython.display import Math, display

from core import CommonMatrixCalculator


class LUDecomposition(CommonMatrixCalculator):
    """LU Decomposition Calculator

    This class provides multiple methods for performing LU decomposition on matrices,
    including Gaussian elimination approach, Doolittle method, Crout method, and PLU
    decomposition with partial pivoting.
    """

    def is_square(self, matrix: Matrix) -> bool:
        """Check if the matrix is square."""
        return matrix.rows == matrix.cols

    def lu_decomposition_gaussian(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix]:
        """Perform LU decomposition using Gaussian elimination approach.

        This method performs LU decomposition by applying Gaussian elimination process
        to obtain U while recording elimination coefficients to construct L.

        Args:
            matrix_input: Matrix input (string representation or sympy.Matrix)
            show_steps (bool): Whether to record and show calculation steps

        Returns:
            tuple: (L, U) matrices, or None if decomposition fails

        Raises:
            ValueError: If input matrix is not square
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{方法一: 高斯消元法}")

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("LU 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")

        # Initialize L and U matrices
        L = eye(n)
        U = A.copy()

        if show_steps:
            self.add_step("初始化:")
            self.add_matrix(L, "L_0")
            self.add_matrix(U, "U_0")

        # Gaussian elimination process
        for k in range(n-1):  # Pivot column
            if show_steps:
                self.step_generator.add_step(f"\\text{{第 {k+1} 步消元:}}")

            # Check if pivot element is zero
            if U[k, k] == 0:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\textbf{{警告: 主元 }} u_{{{k+1}{k+1}}} \\textbf{{ = 0, 可能需要行交换或进行 PLU 分解}}")
                return None

            for i in range(k+1, n):  # Row to eliminate
                # Calculate elimination coefficient
                factor = U[i, k] / U[k, k]
                L[i, k] = factor

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{计算消元系数: }} l_{{{i+1}{k+1}}} = \\frac{{u_{{{i+1}{k+1}}}^{{({k})}}}}{{u_{{{k+1}{k+1}}}^{{({k})}}}} = " +
                        f"\\frac{{{latex(U[i,k])}}}{{{latex(U[k,k])}}} = {latex(factor)}"
                    )

                # Perform row operation - show calculation process for each element
                if show_steps:
                    self.step_generator.add_step(f"\\text{{更新第 {i+1} 行:}}")

                for j in range(k, n):
                    old_value = U[i, j]
                    new_value = U[i, j] - factor * U[k, j]
                    U[i, j] = new_value

                    if show_steps:
                        # Show detailed calculation process for each element
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}}^{{({k+1})}} = u_{{{i+1}{j+1}}}^{{({k})}} - l_{{{i+1}{k+1}}} \\cdot u_{{{k+1}{j+1}}}^{{({k})}} = " +
                            f"{latex(old_value)} - {latex(factor)} \\cdot {latex(U[k,j])} = {latex(new_value)}"
                        )

            if show_steps:
                self.add_step(f"第 {k+1} 步消元后:")
                self.add_matrix(L, f"L_{{{k+1}}}")
                self.add_matrix(U, f"U_{{{k+1}}}")

        # Final verification of decomposition result
        if show_steps:
            self.add_step("最终结果:")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            self.add_step("最终验证:")
            self.step_generator.add_step(r"\text{验证: } L \times U = A")
            L_times_U = L * U
            self.add_matrix(L_times_U, "L \\times U")
            self.add_matrix(A, "A")

            if L_times_U == A:
                self.step_generator.add_step("分解正确")
            else:
                self.step_generator.add_step("分解错误")

        return L, U

    def lu_decomposition_doolittle(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix]:
        """Perform LU decomposition using Doolittle direct decomposition method.

        This method directly calculates elements of L and U using matrix multiplication rules.

        Args:
            matrix_input: Matrix input (string representation or sympy.Matrix)
            show_steps (bool): Whether to record and show calculation steps

        Returns:
            tuple: (L, U) matrices, or None if decomposition fails

        Raises:
            ValueError: If input matrix is not square
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{方法二: 假设形式, 待定系数法}")

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("LU 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")
            self.step_generator.add_step(
                r"\text{假设: } L = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ l_{21} & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ l_{n1} & l_{n2} & \cdots & 1 \end{bmatrix}, \quad U = \begin{bmatrix} u_{11} & u_{12} & \cdots & u_{1n} \\ 0 & u_{22} & \cdots & u_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & u_{nn} \end{bmatrix}")

        # Initialize L and U matrices
        L = eye(n)
        U = zeros(n)

        if show_steps:
            # Create initial symbolic matrices
            L_symbolic = eye(n)
            U_symbolic = zeros(n)

            for i in range(n):
                for j in range(n):
                    if i > j:  # Lower triangular part of L (below diagonal)
                        L_symbolic[i, j] = Symbol(f'l_{{{i+1}{j+1}}}')
                    elif i < j:  # Upper triangular part of U (above diagonal)
                        U_symbolic[i, j] = Symbol(f'u_{{{i+1}{j+1}}}')
                    elif i == j:  # Diagonal elements
                        U_symbolic[i, j] = Symbol(f'u_{{{i+1}{i+1}}}')

            self.add_step("初始化 L 和 U:")
            self.add_matrix(L_symbolic, "L_0")
            self.add_matrix(U_symbolic, "U_0")

        # Doolittle algorithm
        for i in range(n):
            if show_steps and i < n-1:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 计算 L 的第 {i+1} 列和 U 的第 {i+1} 行}}")
            elif show_steps:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 计算 U 的第 {i+1} 行}}")

            # Calculate row i of U
            for j in range(i, n):
                sum_val = 0
                sum_terms = []
                for k in range(i):
                    product = L[i, k] * U[k, j]
                    sum_val += product
                    sum_terms.append(
                        f"l_{{{i+1}{k+1}}} \\cdot u_{{{k+1}{j+1}}}")

                U[i, j] = A[i, j] - sum_val

                if show_steps:
                    if sum_terms:
                        sum_expr = " + ".join(sum_terms)
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = a_{{{i+1}{j+1}}} - ({sum_expr}) = " +
                            f"{latex(A[i,j])} - ({latex(sum_val)}) = {latex(U[i,j])}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = a_{{{i+1}{j+1}}} = {latex(A[i,j])}")

            if U[i, i] == 0 and i < n-1:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\textbf{{警告: }} u_{{{i+1}{i+1}}} \\textbf{{= 0, 可能需要行交换或进行 PLU 分解}}")
                return None

            # Calculate column i of L (starting from row i+1)
            for j in range(i+1, n):
                sum_val = 0
                sum_terms = []
                for k in range(i):
                    product = L[j, k] * U[k, i]
                    sum_val += product
                    sum_terms.append(
                        f"l_{{{j+1}{k+1}}} \\cdot u_{{{k+1}{i+1}}}")

                L[j, i] = (A[j, i] - sum_val) / U[i, i]

                if show_steps:
                    if sum_terms:
                        sum_expr = " + ".join(sum_terms)
                        self.step_generator.add_step(
                            f"l_{{{j+1}{i+1}}} = \\frac{{a_{{{j+1}{i+1}}} - ({sum_expr})}}{{u_{{{i+1}{i+1}}}}} = " +
                            f"\\frac{{{latex(A[j,i])} - ({latex(sum_val)})}}{{{latex(U[i,i])}}} = {latex(L[j,i])}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"l_{{{j+1}{i+1}}} = \\frac{{a_{{{j+1}{i+1}}}}}{{u_{{{i+1}{i+1}}}}} = " +
                            f"\\frac{{{latex(A[j,i])}}}{{{latex(U[i,i])}}} = {latex(L[j,i])}"
                        )

            if show_steps and i < n-1:
                # Create display matrices for current step - based entirely on actual calculated values
                L_display = eye(n)
                U_display = zeros(n)

                # Fill in calculated values
                for r in range(n):
                    for c in range(n):
                        if r <= i or c <= i:  # Calculated parts of L
                            L_display[r, c] = L[r, c]
                        else:  # Uncalculated parts of L
                            if r > c:
                                L_display[r, c] = Symbol(f'l_{{{r+1}{c+1}}}')

                        if c <= i or r <= i:  # Calculated parts of U
                            U_display[r, c] = U[r, c]
                        else:  # Uncalculated parts of U
                            if r <= c:
                                U_display[r, c] = Symbol(f'u_{{{r+1}{c+1}}}')

                self.add_step(f"第 {i+1} 步后的 L 和 U:")
                self.add_matrix(L_display, f"L_{{{i+1}}}")
                self.add_matrix(U_display, f"U_{{{i+1}}}")

        # Verify decomposition result
        if show_steps:
            self.add_step("最终结果:")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            self.step_generator.add_step(r"\text{验证: } L \times U = A")
            L_times_U = L * U
            self.add_matrix(L_times_U, "L \\times U")
            self.add_matrix(A, "A")

            if L_times_U == A:
                self.step_generator.add_step(r"\text{分解正确}")
            else:
                self.step_generator.add_step(r"\text{分解错误}")

        return L, U

    def lu_decomposition_crout(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix]:
        """Perform LU decomposition using Crout decomposition method.

        In this method, diagonal elements of L are 1, and diagonal elements of U need to be calculated.

        Args:
            matrix_input: Matrix input (string representation or sympy.Matrix)
            show_steps (bool): Whether to record and show calculation steps

        Returns:
            tuple: (L, U) matrices, or None if decomposition fails

        Raises:
            ValueError: If input matrix is not square
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{方法三: 另一种假设形式法}")

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("LU 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")
            self.step_generator.add_step(
                r"\text{假设: } L = \begin{bmatrix} l_{11} & 0 & \cdots & 0 \\ l_{21} & l_{22} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ l_{n1} & l_{n2} & \cdots & l_{nn} \end{bmatrix}, \quad U = \begin{bmatrix} 1 & u_{12} & \cdots & u_{1n} \\ 0 & 1 & \cdots & u_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}")

        # Initialize L and U matrices
        L = zeros(n)
        U = eye(n)

        if show_steps:
            # Create initial symbolic matrices
            L_symbolic = zeros(n)
            U_symbolic = eye(n)

            for i in range(n):
                for j in range(n):
                    # Lower triangular part of L (including diagonal)
                    if i >= j:
                        L_symbolic[i, j] = Symbol(f'l_{{{i+1}{j+1}}}')
                    elif i < j:  # Upper triangular part of U (above diagonal)
                        U_symbolic[i, j] = Symbol(f'u_{{{i+1}{j+1}}}')

            self.add_step("初始化 L 和 U:")
            self.add_matrix(L_symbolic, "L_0")
            self.add_matrix(U_symbolic, "U_0")

        # Crout algorithm
        for i in range(n):
            if show_steps and i < n-1:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 计算 L 的第 {i+1} 列和 U 的第 {i+1} 行}}")
            else:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 计算 L 的第 {i+1} 列}}")

            # Calculate column i of L
            for j in range(i, n):
                sum_val = 0
                sum_terms = []
                for k in range(i):
                    product = L[j, k] * U[k, i]
                    sum_val += product
                    sum_terms.append(
                        f"l_{{{j+1}{k+1}}} \\cdot u_{{{k+1}{i+1}}}")

                L[j, i] = A[j, i] - sum_val

                if show_steps:
                    if sum_terms:
                        sum_expr = " + ".join(sum_terms)
                        self.step_generator.add_step(
                            f"l_{{{j+1}{i+1}}} = a_{{{j+1}{i+1}}} - ({sum_expr}) = " +
                            f"{latex(A[j,i])} - ({latex(sum_val)}) = {latex(L[j,i])}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"l_{{{j+1}{i+1}}} = a_{{{j+1}{i+1}}} = {latex(A[j,i])}")

            if L[i, i] == 0:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\textbf{{警告: }} l_{{{i+1}{i+1}}} \\textbf{{= 0, 可能需要行交换或进行 PLU 分解}}")
                return None

            # Calculate row i of U (starting from column i+1)
            for j in range(i+1, n):
                sum_val = 0
                sum_terms = []
                for k in range(i):
                    product = L[i, k] * U[k, j]
                    sum_val += product
                    sum_terms.append(
                        f"l_{{{i+1}{k+1}}} \\cdot u_{{{k+1}{j+1}}}")

                U[i, j] = (A[i, j] - sum_val) / L[i, i]

                if show_steps:
                    if sum_terms:
                        sum_expr = " + ".join(sum_terms)
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = \\frac{{a_{{{i+1}{j+1}}} - ({sum_expr})}}{{l_{{{i+1}{i+1}}}}} = " +
                            f"\\frac{{{latex(A[i,j])} - ({latex(sum_val)})}}{{{latex(L[i,i])}}} = {latex(U[i,j])}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = \\frac{{a_{{{i+1}{j+1}}}}}{{l_{{{i+1}{i+1}}}}} = " +
                            f"\\frac{{{latex(A[i,j])}}}{{{latex(L[i,i])}}} = {latex(U[i,j])}"
                        )

            if show_steps and i < n-1:
                # Create display matrices for current step - fix: ensure calculated values are shown as numbers
                L_display = zeros(n)
                U_display = eye(n)  # U diagonal is always 1

                # Fill L matrix: show numbers for calculated parts, show symbols for uncalculated parts
                for r in range(n):
                    for c in range(n):
                        if r >= c:  # Lower triangular part of L
                            if c <= i:  # Calculated columns (column 0 to i)
                                L_display[r, c] = L[r, c]  # Show number
                            else:  # Uncalculated columns
                                L_display[r, c] = Symbol(f'l_{{{r+1}{c+1}}}')

                        # U matrix: show numbers for calculated parts, show symbols for uncalculated parts
                        # Upper triangular part of U (excluding diagonal)
                        if r < c:
                            # Calculated rows (row 0 to i)
                            if r <= i and c <= n:
                                U_display[r, c] = U[r, c]  # Show number
                            else:  # Uncalculated rows
                                U_display[r, c] = Symbol(f'u_{{{r+1}{c+1}}}')

                self.add_step(f"第 {i+1} 步后的 L 和 U:")
                self.add_matrix(L_display, f"L_{{{i+1}}}")
                self.add_matrix(U_display, f"U_{{{i+1}}}")

        # Verify decomposition result
        if show_steps:
            self.add_step("最终结果:")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            self.step_generator.add_step(r"\text{验证: } L \times U = A")
            L_times_U = L * U
            self.add_matrix(L_times_U, "L \\times U")
            self.add_matrix(A, "A")

            if L_times_U == A:
                self.step_generator.add_step(r"\text{分解正确}")
            else:
                self.step_generator.add_step(r"\text{分解错误}")

        return L, U

    def plu_decomposition(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix, Matrix]:
        """Perform PLU decomposition with partial pivoting.

        Returns P, L, U such that PA = LU where P is the permutation matrix.

        Args:
            matrix_input: Matrix input (string representation or sympy.Matrix)
            show_steps (bool): Whether to record and show calculation steps

        Returns:
            tuple: (P, L, U) matrices, or None if decomposition fails

        Raises:
            ValueError: If input matrix is not square
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{PLU 分解 (带部分主元选择)}")

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("PLU 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")

        # Initialize matrices
        P = eye(n)  # Permutation matrix
        L = eye(n)  # Unit lower triangular matrix
        U = A.copy()  # Upper triangular matrix

        if show_steps:
            self.add_step("初始化:")
            self.add_matrix(P, "P_0")
            self.add_matrix(L, "L_0")
            self.add_matrix(U, "U_0")

        # Record row exchanges
        pivot_history = []

        # Gaussian elimination process (with partial pivoting)
        for k in range(n-1):
            if show_steps:
                self.step_generator.add_step(f"\\text{{第 {k+1} 步消元:}}")

            # Find pivot
            pivot_row = k
            max_val = abs(U[k, k])

            for i in range(k+1, n):
                if abs(U[i, k]) > max_val:
                    max_val = abs(U[i, k])
                    pivot_row = i

            # Row exchange if needed
            if pivot_row != k:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{行交换: 将第 {k+1} 行与第 {pivot_row+1} 行交换, 因为 }} |{latex(U[pivot_row, k])}| > |{latex(U[k, k])}|"
                    )

                # Exchange rows in U
                U.row_swap(k, pivot_row)

                # Exchange rows in L (only exchange already calculated parts)
                for j in range(k):
                    L[k, j], L[pivot_row, j] = L[pivot_row, j], L[k, j]

                # Exchange rows in P
                P.row_swap(k, pivot_row)

                pivot_history.append((k, pivot_row))

                if show_steps:
                    self.add_step(f"行交换后:")
                    self.add_matrix(P, f"P_{{{k+1}}}")
                    self.add_matrix(L, f"L_{{{k+1}}}")
                    self.add_matrix(U, f"U_{{{k+1}}}")

            # Check if pivot is zero
            if U[k, k] == 0:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\textbf{{警告: 主元为 0, 矩阵可能奇异}}")
                return None

            # Elimination process
            for i in range(k+1, n):
                # Calculate elimination coefficient
                factor = U[i, k] / U[k, k]
                L[i, k] = factor

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{计算消元系数: }} l_{{{i+1}{k+1}}} = \\frac{{u_{{{i+1}{k+1}}}}}{{u_{{{k+1}{k+1}}}}} = " +
                        f"\\frac{{{latex(U[i,k])}}}{{{latex(U[k,k])}}} = {latex(factor)}"
                    )

                # Update row i of U
                if show_steps:
                    self.step_generator.add_step(f"\\text{{更新第 {i+1} 行:}}")

                for j in range(k, n):
                    old_value = U[i, j]
                    new_value = U[i, j] - factor * U[k, j]
                    U[i, j] = new_value

                    if show_steps and j == k:  # Only show detailed calculation for first element
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = u_{{{i+1}{j+1}}} - l_{{{i+1}{k+1}}} \\cdot u_{{{k+1}{j+1}}} = " +
                            f"{latex(old_value)} - {latex(factor)} \\cdot {latex(U[k,j])} = {latex(new_value)}"
                        )

            if show_steps:
                self.add_step(f"第 {k+1} 步消元后:")
                self.add_matrix(L, f"L_{{{k+1}}}")
                self.add_matrix(U, f"U_{{{k+1}}}")

        # Final result
        if show_steps:
            self.add_step("最终结果:")
            self.add_matrix(P, "P")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            # Verify decomposition
            self.step_generator.add_step(r"\text{验证: }")
            P_times_A = P * A
            L_times_U = L * U
            self.add_matrix(P_times_A, "P \\times A")
            self.add_matrix(L_times_U, "L \\times U")

            if P_times_A == L_times_U:
                self.step_generator.add_step(r"\text{分解正确: } PA = LU")
            else:
                self.step_generator.add_step(r"\text{分解错误}")

            # Show row exchange history
            if pivot_history:
                self.add_step("行交换历史:")
                for i, (old_row, new_row) in enumerate(pivot_history):
                    self.step_generator.add_step(
                        f"\\text{{步骤 {i+1}: 行 {old_row+1}}} \\leftrightarrow 行 \\text{{{new_row+1}}}")

        return P, L, U

    def check_lu_conditions(self, matrix_input: str, show_steps: bool = True) -> bool:
        """Check if matrix satisfies conditions for LU decomposition.

        Args:
            matrix_input: Matrix input (string representation or sympy.Matrix)
            show_steps (bool): Whether to record and show calculation steps

        Returns:
            bool: True if matrix satisfies LU decomposition conditions, False otherwise
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{检查LU分解条件}")

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_matrix(A, "A")

        n = A.rows
        conditions_met = True

        # Check if matrix is square
        if not self.is_square(A):
            if show_steps:
                self.step_generator.add_step(r"\text{矩阵不是方阵, 无法进行LU分解}")
            return False

        if show_steps:
            self.step_generator.add_step(r"\text{矩阵是方阵}")

        # Check if leading principal minors are all non-zero
        for k in range(1, n+1):
            submatrix = A[:k, :k]
            det = submatrix.det()

            if show_steps:
                self.add_matrix(submatrix, f"A_{{{k}}}")
                self.step_generator.add_step(
                    f"\\text{{主子式 }} \\det(A_{{{k}}}) = {latex(det)}")

            if det == 0:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{第 {k} 阶顺序主子式为 0, 可能需要行交换}}")
                conditions_met = False
            else:
                if show_steps:
                    self.step_generator.add_step(f"\\text{{第 {k} 阶顺序主子式不为 0}}")

        if conditions_met:
            if show_steps:
                self.step_generator.add_step(rf"\text{{矩阵满足 LU 分解条件(不需要行交换)}}")
        else:
            if show_steps:
                self.step_generator.add_step(
                    r"\text{矩阵不满足 LU 分解条件, 可能需要行交换或使用 PLU 分解}")

        return conditions_met

    def auto_lu_decomposition(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix, Matrix]:
        """Automatically select LU or PLU decomposition.

        Automatically determines whether row exchanges are needed based on matrix conditions.

        Args:
            matrix_input: Matrix input (string representation or sympy.Matrix)
            show_steps (bool): Whether to record and show calculation steps

        Returns:
            tuple: Decomposition result matrices
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{自动 LU/PLU 分解}")

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_matrix(A, "A")

        # Check conditions
        can_do_lu = self.check_lu_conditions(matrix_input, show_steps=False)

        if can_do_lu:
            if show_steps:
                self.step_generator.add_step(r"\text{矩阵满足 LU 分解条件，使用标准 LU 分解}")
            return self.lu_decomposition_doolittle(matrix_input, show_steps)

        if show_steps:
            self.step_generator.add_step(r"\text{矩阵不满足 LU 分解条件，使用 PLU 分解}")
        return self.plu_decomposition(matrix_input, show_steps)


# def demo():
#     """Demonstrate LU decomposition"""
#     lu = LUDecomposition()

#     # Example matrices
#     A1 = '[[2,1,1],[4,3,3],[8,7,9]]'
#     A2 = '[[1,2,3],[2,5,7],[3,7,10]]'
#     A3 = '[[2,4,6],[1,3,7],[1,1,1]]'

#     # Gaussian elimination method
#     try:
#         lu.lu_decomposition_gaussian(A1)
#         display(Math(lu.get_steps_latex()))
#     except Exception as e:
#         lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#         display(Math(lu.get_steps_latex()))

#     # Doolittle method
#     try:
#         lu.lu_decomposition_doolittle(A2)
#         display(Math(lu.get_steps_latex()))
#     except Exception as e:
#         lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#         display(Math(lu.get_steps_latex()))

#     # Crout method
#     try:
#         lu.lu_decomposition_crout(A3)
#         display(Math(lu.get_steps_latex()))
#     except Exception as e:
#         lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#         display(Math(lu.get_steps_latex()))


# def demo_plu():
#     """Demonstrate PLU decomposition"""
#     lu = LUDecomposition()

#     # Matrices requiring row exchanges
#     A_need_pivot_1 = '[[0,1,1],[1,1,1],[2,3,4]]'
#     A_need_pivot_2 = '[[1,2,3],[4,5,6],[7,8,9]]'
#     A_need_pivot_3 = '[[0,1,1],[4,3,3],[8,7,9]]'

#     lu.step_generator.add_step(r"\textbf{PLU 分解演示}")

#     cases = [A_need_pivot_1, A_need_pivot_2, A_need_pivot_3]

#     for matrix in cases:
#         try:
#             lu.plu_decomposition(matrix)
#             display(Math(lu.get_steps_latex()))
#         except Exception as e:
#             lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(lu.get_steps_latex()))


# def demo_auto():
#     """Demonstrate automatic selection of decomposition method"""
#     lu = LUDecomposition()

#     # Test matrices
#     test_matrices = [
#         ("可 LU 分解的矩阵", '[[2,1,1],[4,3,3],[8,7,9]]'),
#         ("需要 PLU 的矩阵", '[[0,1,1],[1,1,1],[2,3,4]]'),
#         ("对角占优矩阵", '[[3,1,1],[1,4,2],[1,1,5]]')
#     ]

#     lu.step_generator.add_step(r"\textbf{自动 LU/PLU 分解演示}")

#     for name, matrix in test_matrices:
#         lu.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             lu.auto_lu_decomposition(matrix)
#             display(Math(lu.get_steps_latex()))
#         except Exception as e:
#             lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(lu.get_steps_latex()))


# def demo_symbolic():
#     """Demonstrate LU decomposition of symbolic matrices"""
#     lu = LUDecomposition()

#     a, b, c, d, e, f, g, h, i = symbols('a b c d e f g h i')

#     A_sym = Matrix([[a, b, c], [d, e, f], [g, h, i]])

#     lu.step_generator.add_step(r"\textbf{符号矩阵 LU 分解}")
#     lu.step_generator.add_step(r"\textbf{假设所有符号表达式不为 0, 可作主元}")
#     lu.add_matrix(A_sym, "A")

#     try:
#         lu.lu_decomposition_gaussian(A_sym)
#         display(Math(lu.get_steps_latex()))
#     except Exception as ex:
#         lu.step_generator.add_step(f"\\text{{错误: }} {str(ex)}")
#         display(Math(lu.get_steps_latex()))

#     B_sym = Matrix([[a, b, c], [4, 5, 6], [g, h, i]])
#     lu.lu_decomposition_gaussian(B_sym)
#     display(Math(lu.get_steps_latex()))


# def demo_special_cases():
#     """Demonstrate special cases"""
#     lu = LUDecomposition()

#     # Diagonal matrix
#     diag = '[[2,0,0],[0,3,0],[0,0,5]]'

#     # Triangular matrix
#     triangular = '[[1,2,3],[0,4,5],[0,0,6]]'

#     # Potentially undecomposable matrix
#     problematic = '[[0,1],[1,0]]'

#     lu.step_generator.add_step(r"\textbf{特殊情况演示}")

#     cases = [
#         ("对角矩阵", diag),
#         ("上三角矩阵", triangular),
#         ("可能无法分解的矩阵", problematic)
#     ]

#     for name, matrix in cases:
#         display(Math(f"\\textbf{{{name}}}"))
#         try:
#             lu.lu_decomposition_crout(matrix)
#             display(Math(lu.get_steps_latex()))
#         except Exception as e:
#             lu.step_generator.add_step(f"\\text{{分解失败: }} {str(e)}")
#             display(Math(lu.get_steps_latex()))


# if __name__ == "__main__":
#     # Run numerical demonstrations
#     demo()
#     # Run PLU demonstration
#     demo_plu()
#     # Run automatic decomposition demonstration
#     demo_auto()
#     # Run special cases demonstration
#     demo_special_cases()
#     # Run symbolic demonstration
#     demo_symbolic()
