from typing import List
from sympy import Matrix, eye, latex, symbols, zeros
# from IPython.display import Math, display

from core import CommonMatrixCalculator


class LinearSystemSolver(CommonMatrixCalculator):
    """A class for solving linear systems of equations using various methods.

    This class provides multiple approaches to solve linear systems including
    Gaussian elimination, Gauss-Jordan elimination, matrix inversion,
    LU decomposition, and Cramer's rule, along with handling special cases
    such as underdetermined, overdetermined, and singular systems.
    """

    # def display_steps(self) -> None:
    #     """Display all solution steps in mathematical notation."""
    #     display(Math(self.step_generator.get_steps_latex()))

    def display_system(self, A: Matrix, b: Matrix, variables: List[str] = None) -> None:
        """Display the linear system Ax = b in equation form.

        Parameters:
            A: Coefficient matrix
            b: Constant vector
            variables: List of variable names (optional)
        """
        if variables is None:
            n = A.cols
            variables = [f'x_{i+1}' for i in range(n)]

        equations = []
        for i in range(A.rows):
            equation_terms = []
            for j in range(A.cols):
                if A[i, j] != 0:
                    if A[i, j] == 1:
                        equation_terms.append(f"{variables[j]}")
                    elif A[i, j] == -1:
                        equation_terms.append(f"-{variables[j]}")
                    else:
                        equation_terms.append(
                            f"{latex(A[i, j])}{variables[j]}")

            if equation_terms:
                equation_str = " + ".join(equation_terms).replace("+ -", "- ")
                equations.append(f"{equation_str} = {latex(b[i])}")
            else:
                equations.append(f"0 = {latex(b[i])}")

        for eq in equations:
            self.add_equation(eq)

    def is_square(self, matrix: Matrix) -> bool:
        """Check if a matrix is square."""
        return matrix.rows == matrix.cols

    def check_system_type(self, A: Matrix, b: Matrix, show_steps: bool = True) -> str:
        """Analyze and classify the type of linear system.

        Parameters:
            A: Coefficient matrix
            b: Constant vector
            show_steps: Whether to display analysis steps

        Returns:
            str: System type classification
        """
        A = self.parse_matrix_input(A)
        b = self.parse_vector_input(b)
        m, n = A.rows, A.cols

        if show_steps:
            self.add_step("系统分析:")
            self.add_step(f"系数矩阵 A: {m} \\times {n}")
            self.add_matrix(A, "A")
            self.add_step(f"常数向量 \\boldsymbol{{b}}: {m} \\times 1")
            self.add_vector(b, "\\boldsymbol{b}")

        # Check if it's a square system
        if m == n:
            det_A = A.det()
            if show_steps:
                self.add_step(f"\\det(A) = {latex(det_A)}")

            if det_A != 0:
                if show_steps:
                    self.add_step("系统有唯一解")
                return "unique_solution"

            # Check ranks to determine if there are infinite solutions or no solution
            rank_A = A.rank()
            augmented = A.row_join(b)
            rank_augmented = augmented.rank()

            if show_steps:
                self.add_step(f"\\text{{rank}}(A) = {rank_A}")
                self.add_step(f"\\text{{rank}}([A|b]) = {rank_augmented}")

            if rank_A == rank_augmented:
                if show_steps:
                    self.add_step("系统有无穷多解")
                return "singular_infinite"

            if show_steps:
                self.add_step("系统无解")
            return "singular_no_solution"

        if m < n:
            if show_steps:
                self.add_step("欠定系统: 方程数少于未知数, 通常有无穷多解")
            return "underdetermined"

        # Check if overdetermined system has a solution
        rank_A = A.rank()
        augmented = A.row_join(b)
        rank_augmented = augmented.rank()

        if show_steps:
            self.add_step(f"\\text{{rank}}(A) = {rank_A}")
            self.add_step(f"\\text{{rank}}([A|b]) = {rank_augmented}")

        if rank_A == rank_augmented:
            if show_steps:
                self.add_step("超定系统有精确解, 使用标准方法")
            return "overdetermined_exact"

        if show_steps:
            self.add_step("超定系统无精确解, 使用最小二乘法")
        return "overdetermined"

    def check_special_matrix(self, matrix: Matrix) -> str:
        """Identify special types of matrices.

        Parameters:
            matrix: Matrix to analyze

        Returns:
            str: Type of special matrix or "general" if none
        """
        n = matrix.rows
        if n > matrix.cols:
            return "general"

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

    def solve_singular_system(self, A_input: str, b_input: str, show_steps: bool = True) -> Matrix:
        """Solve singular systems (square matrices with determinant 0).

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            show_steps: Whether to display solution steps

        Returns:
            Solution vector or None if no solution exists
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        if show_steps:
            self.add_step("\\textbf{奇异系统求解: 方阵但行列式为0}")
            self.display_system(A, b)

        n = A.rows

        # Create augmented matrix
        augmented = A.row_join(b)
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        # Gaussian-Jordan elimination to get reduced row echelon form
        rref_matrix, _ = augmented.rref()

        if show_steps:
            self.add_step("简化行阶梯形:")
            self.add_matrix(rref_matrix, "[A|\\boldsymbol{b}]_{rref}")

        # Check if solution exists
        has_solution = True
        for i in range(n):
            # If coefficient part is all zeros but constant term is non-zero, no solution
            if all(rref_matrix[i, j] == 0 for j in range(n)) and rref_matrix[i, n] != 0:
                has_solution = False
                if show_steps:
                    self.add_step(
                        f"第 {i+1} 行: 0 = {latex(rref_matrix[i, n])} ≠ 0，系统无解")
                break

        if not has_solution:
            if show_steps:
                self.add_step("系统无解")
            return None

        # Identify pivot columns and free variable columns
        pivot_cols = []
        free_cols = []

        for j in range(n):
            is_pivot = False
            for i in range(n):
                if rref_matrix[i, j] == 1 and all(rref_matrix[i, k] == 0 for k in range(j)):
                    pivot_cols.append(j)
                    is_pivot = True
                    break
            if not is_pivot:
                free_cols.append(j)

        if show_steps:
            self.add_step(f"主元列: {[c+1 for c in pivot_cols]}")
            self.add_step(f"自由变量列: {[c+1 for c in free_cols]}")

        # If all variables are pivot variables, there's a unique solution
        # (though theoretically singular matrices shouldn't have unique solutions,
        # but numerical computation may have errors)
        if len(free_cols) == 0:
            if show_steps:
                self.add_step("系统有唯一解")
            x = zeros(n, 1)
            for i in range(n):
                x[i] = rref_matrix[i, n]

            x_simplified = self.simplify_matrix(x)

            if show_steps:
                self.add_step("解:")
                self.add_vector(x_simplified, "\\boldsymbol{x}")
            return x_simplified

        # Create free variable symbols
        free_vars = symbols(f't_1:{len(free_cols)+1}')

        # Build solution vector
        x = zeros(n, 1)

        # Establish equations for each pivot variable
        pivot_rows = []
        for i in range(n):
            if any(rref_matrix[i, j] != 0 for j in range(n)):
                pivot_rows.append(i)

        for i, row_idx in enumerate(pivot_rows):
            pivot_col = pivot_cols[i]
            x[pivot_col] = rref_matrix[row_idx, n]  # Particular solution

            # Subtract contribution from free variables
            for j, free_col in enumerate(free_cols):
                x[pivot_col] -= rref_matrix[row_idx, free_col] * free_vars[j]

        # Set free variables
        for j, free_col in enumerate(free_cols):
            x[free_col] = free_vars[j]

        if show_steps:
            self.add_step("通解:")
            self.add_equation(
                "\\boldsymbol{x} = \\boldsymbol{x_p} + \\sum t \\boldsymbol{h}")

            # Show particular solution
            x_particular = zeros(n, 1)
            for i in range(n):
                if i in pivot_cols:
                    idx = pivot_cols.index(i)
                    x_particular[i] = rref_matrix[pivot_rows[idx], n]
                else:
                    x_particular[i] = 0

            self.add_vector(x_particular, "\\boldsymbol{x_p}")

            # Show homogeneous solutions
            if free_cols:
                for j, free_col in enumerate(free_cols):
                    x_homogeneous = zeros(n, 1)
                    x_homogeneous[free_col] = 1
                    for i, pivot_col in enumerate(pivot_cols):
                        x_homogeneous[pivot_col] = - \
                            rref_matrix[pivot_rows[i], free_col]
                    self.add_vector(
                        x_homogeneous, f"\\boldsymbol{{h_{j+1}}}")

        if show_steps:
            self.add_vector(x, "\\boldsymbol{x}")

        return x

    def solve_underdetermined_system(self, A_input: str, b_input: str, show_steps: bool = True) -> Matrix:
        """Solve underdetermined systems (introducing free variables).

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            show_steps: Whether to display solution steps

        Returns:
            Solution vector with free variables
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        if show_steps:
            self.add_step("\\textbf{欠定系统求解: 引入自由变量}")
            self.display_system(A, b)

        m, n = A.rows, A.cols

        # Create augmented matrix
        augmented = A.row_join(b)
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        # Gaussian-Jordan elimination to get reduced row echelon form
        rref_matrix, _ = augmented.rref()

        if show_steps:
            self.add_step("简化行阶梯形:")
            self.add_matrix(rref_matrix, "[A|\\boldsymbol{b}]_{rref}")

        # Check if solution exists
        for i in range(m):
            if all(rref_matrix[i, j] == 0 for j in range(n)) and rref_matrix[i, n] != 0:
                if show_steps:
                    self.add_step("系统无解")
                return None

        # Identify pivot columns and free variable columns
        pivot_cols = []
        free_cols = []

        for j in range(n):
            is_pivot = False
            for i in range(m):
                if rref_matrix[i, j] == 1 and all(rref_matrix[i, k] == 0 for k in range(j)):
                    pivot_cols.append(j)
                    is_pivot = True
                    break
            if not is_pivot:
                free_cols.append(j)

        if show_steps:
            self.add_step(f"主元列: {[c+1 for c in pivot_cols]}")
            self.add_step(f"自由变量列: {[c+1 for c in free_cols]}")

        # Create free variable symbols
        free_vars = symbols(f't_1:{len(free_cols)+1}')

        # Build solution vector
        x = zeros(n, 1)

        for i, pivot_col in enumerate(pivot_cols):
            x[pivot_col] = rref_matrix[i, n]  # Particular solution part

            # Subtract contribution from free variables
            for j, free_col in enumerate(free_cols):
                x[pivot_col] -= rref_matrix[i, free_col] * free_vars[j]

        for j, free_col in enumerate(free_cols):
            x[free_col] = free_vars[j]

        if show_steps:
            self.add_step("通解:")
            self.add_equation(
                "\\boldsymbol{x} = \\boldsymbol{x_p} + \\sum t \\boldsymbol{h}")

            # Show particular and homogeneous solutions
            x_particular = zeros(n, 1)
            for i in range(n):
                if i in pivot_cols:
                    x_particular[i] = rref_matrix[pivot_cols.index(i), n]
                else:
                    x_particular[i] = 0

            self.add_vector(x_particular, "\\boldsymbol{x_p}")

            if free_cols:
                for j, free_col in enumerate(free_cols):
                    x_homogeneous = zeros(n, 1)
                    x_homogeneous[free_col] = 1
                    for i, pivot_col in enumerate(pivot_cols):
                        x_homogeneous[pivot_col] = -rref_matrix[i, free_col]
                    self.add_vector(
                        x_homogeneous, f"\\boldsymbol{{h_{j+1}}}")

        if show_steps:
            self.add_vector(x, "\\boldsymbol{x}")

        return x

    def solve_overdetermined_system(self, A_input: str, b_input: str, show_steps: bool = True, simplify_result: bool = True) -> Matrix:
        """Solve overdetermined systems (least squares method).

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            show_steps: Whether to display solution steps
            simplify_result: Whether to simplify the result

        Returns:
            Least squares solution or None if failed
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        if show_steps:
            self.add_step("\\textbf{{超定系统求解: 最小二乘法}}")
            self.display_system(A, b)
            self.add_step(
                "求解正规方程: A^TA\\boldsymbol{{x}} = A^T\\boldsymbol{{b}}")

        # Calculate A^T A and A^T b
        A_T = A.T
        ATA = A_T * A
        ATb = A_T * b

        if show_steps:
            self.add_matrix(A_T, "A^T")
            self.add_matrix(ATA, "A^TA")
            self.add_vector(ATb, "A^T\\boldsymbol{{b}}")

        # Check if A^T A is invertible
        if ATA.det() == 0:
            if show_steps:
                self.add_step("警告: A^TA 不可逆，最小二乘解不唯一")
            return None

        # Solve normal equations
        try:
            x = ATA.inv() * ATb

            if simplify_result:
                x_simplified = self.simplify_matrix(x)
            else:
                x_simplified = x

            if show_steps:
                self.add_step("最小二乘解:")
                self.add_vector(x_simplified, "\\boldsymbol{x}")

                # Calculate residuals
                residual = A * x_simplified - b
                residual_norm = (residual.T * residual)[0]

                self.add_step("残差分析:")
                self.add_vector(
                    residual, "\\boldsymbol{r} = A\\boldsymbol{x} - \\boldsymbol{b}")
                self.add_step(
                    f"\\|\\boldsymbol{{r}}\\|^2 = {latex(residual_norm)}")

            return x_simplified

        except Exception as e:
            if show_steps:
                self.add_step(f"求解失败: {str(e)}")
            return None

    def solve_by_gaussian_elimination(self, A_input: str, b_input: str, show_steps: bool = True) -> Matrix:
        """Method 1: Gaussian elimination.

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            show_steps: Whether to display solution steps

        Returns:
            Solution vector
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # Check system type
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            return self.solve_underdetermined_system(A, b, show_steps)
        if system_type == "overdetermined":
            self.add_step("警告: 超定系统，高斯消元法可能无解")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{{方法一: 高斯消元法}}")
            self.display_system(A, b)

        m, n = A.rows, A.cols

        # Create augmented matrix
        augmented = A.row_join(b)
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|\\boldsymbol{{b}}]")

        # Gaussian elimination
        for i in range(min(m, n)):
            if show_steps:
                self.add_step(f"第 {i+1} 步: 处理第 {i+1} 列")

            # Find pivot
            pivot_row = i
            for r in range(i, m):
                if augmented[r, i] != 0:
                    pivot_row = r
                    break

            # Row swap (if needed)
            if pivot_row != i:
                if show_steps:
                    self.add_step(
                        f"行交换: R_{i+1} \\leftrightarrow  R_{pivot_row+1}")
                augmented.row_swap(i, pivot_row)
                if show_steps:
                    self.add_matrix(augmented, "[A|\\boldsymbol{{b}}]")

            # If pivot is 0, skip this column
            if augmented[i, i] == 0:
                if show_steps:
                    self.add_step(f"第 {i+1} 列主元为 0，跳过")
                continue

            # Normalize pivot row
            pivot = augmented[i, i]
            if pivot != 1:
                if show_steps:
                    self.add_step(f"归一化: R_{i+1} \\times {latex(1/pivot)}")
                augmented[i, :] = augmented[i, :] / pivot
                if show_steps:
                    self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

            # Eliminate other rows
            for j in range(i+1, m):
                if augmented[j, i] != 0:
                    factor = augmented[j, i]
                    if show_steps:
                        self.add_step(
                            f"消元: R_{j+1} - {latex(factor)} \\times R_{i+1}")
                    augmented[j, :] = augmented[j, :] - \
                        factor * augmented[i, :]
                    if show_steps:
                        self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        if show_steps:
            self.add_step("行阶梯形:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]_{ref}")

        # Back substitution
        x = zeros(n, 1)
        for i in range(min(m, n)-1, -1, -1):
            if augmented[i, i] == 0:
                continue

            sum_val = sum(augmented[i, j] * x[j] for j in range(i+1, n))
            x[i] = (augmented[i, n] - sum_val) / augmented[i, i]

            if show_steps:
                if i == min(m, n)-1:
                    self.add_step(
                        f"x_{{{i+1}}} = \\frac{{b_{{{i+1}}}}}{{A_{{{i+1}{i+1}}}}} = \\frac{{{latex(augmented[i, n])}}}{{{latex(augmented[i, i])}}} = {latex(x[i])}")
                else:
                    sum_terms = " + ".join(
                        [f"A_{{{i+1}{j+1}}} \\cdot x_{{{j+1}}}" for j in range(i+1, n)])
                    self.add_step(
                        f"x_{{{i+1}}} = \\frac{{b_{{{i+1}}} - ({sum_terms})}}{{A_{{{i+1}{i+1}}}}} = \\frac{{{latex(augmented[i, n])} - ({latex(sum_val)})}}{{{latex(augmented[i, i])}}} = {latex(x[i])}")

        # Simplify results
        x_simplified = self.simplify_matrix(x)

        if show_steps:
            self.add_step("最终解:")
            self.add_vector(x_simplified, "\\boldsymbol{x}")

            # Verification
            self.add_step("验证:")
            b_check = A * x_simplified
            b_check_simplified = self.simplify_matrix(b_check)
            self.add_vector(b_check_simplified, "A \\times \\boldsymbol{x}")
            self.add_vector(b, "\\boldsymbol{b}")
            if b_check_simplified == b:
                self.add_step(
                    "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
            else:
                self.add_step("验证失败")

        return x_simplified

    def solve_by_gauss_jordan(self, A_input: str, b_input: str, show_steps: bool = True) -> Matrix:
        """Method 2: Gauss-Jordan elimination.

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            show_steps: Whether to display solution steps

        Returns:
            Solution vector
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # Check system type
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            return self.solve_underdetermined_system(A, b, show_steps)
        if system_type == "overdetermined":
            self.add_step("警告: 超定系统，高斯-约当消元法可能无解")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法二: 高斯-约当消元法}")
            self.display_system(A, b)

        m, n = A.rows, A.cols

        # Create augmented matrix
        augmented = A.row_join(b)
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        # Gauss-Jordan elimination
        for i in range(min(m, n)):
            if show_steps:
                self.add_step(f"第 {i+1} 步: 处理第 {i+1} 列")

            # Find pivot
            pivot_row = i
            for r in range(i, m):
                if augmented[r, i] != 0:
                    pivot_row = r
                    break

            # Row swap (if needed)
            if pivot_row != i:
                if show_steps:
                    self.add_step(
                        f"行交换: R_{i+1} \\leftrightarrow  R_{pivot_row+1}")
                augmented.row_swap(i, pivot_row)
                if show_steps:
                    self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

            # If pivot is 0, skip this column
            if augmented[i, i] == 0:
                if show_steps:
                    self.add_step(f"第 {i+1} 列主元为 0, 跳过")
                continue

            # Normalize pivot row
            pivot = augmented[i, i]
            if pivot != 1:
                if show_steps:
                    self.add_step(f"归一化: R_{i+1} \\times {latex(1/pivot)}")
                augmented[i, :] = augmented[i, :] / pivot
                if show_steps:
                    self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

            # Eliminate other rows
            for j in range(m):
                if j != i and augmented[j, i] != 0:
                    factor = augmented[j, i]
                    if show_steps:
                        self.add_step(
                            f"消元: R_{j+1} - {latex(factor)} \\times R_{i+1}")
                    augmented[j, :] = augmented[j, :] - \
                        factor * augmented[i, :]
                    if show_steps:
                        self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        if show_steps:
            self.add_step("简化行阶梯形:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]_{rref}")

        # Extract solution
        x = zeros(n, 1)
        for i in range(min(m, n)):
            if augmented[i, i] != 0:
                x[i] = augmented[i, n]

        # Simplify results
        x_simplified = self.simplify_matrix(x)

        if show_steps:
            self.add_step("最终解:")
            self.add_vector(x_simplified, "\\boldsymbol{x}")

            # Verification
            self.add_step("验证:")
            b_check = self.simplify_matrix(A * x_simplified)
            self.add_vector(b_check, "A \\times \\boldsymbol{x}")
            self.add_vector(b, "\\boldsymbol{b}")
            if b.equals(b_check):
                self.add_step(
                    "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
            else:
                self.add_step("验证失败")

        return x_simplified

    def solve_by_matrix_inverse(self, A_input: str, b_input: str, show_steps: bool = True) -> Matrix:
        """Method 3: Matrix inversion method.

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            show_steps: Whether to display solution steps

        Returns:
            Solution vector or None if matrix is not invertible
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # Check system type
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            self.add_step("警告: 欠定系统，矩阵求逆法不适用")
            return self.solve_underdetermined_system(A, b, show_steps)
        if system_type == "overdetermined":
            self.add_step("警告: 超定系统，矩阵求逆法不适用")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法三: 矩阵求逆法}")
            self.display_system(A, b)

        # Check if invertible
        if not self.is_square(A):
            if show_steps:
                self.add_step("矩阵不是方阵, 不能使用求逆法")
            return None

        det_A = A.det()
        if det_A == 0:
            if show_steps:
                self.add_step("矩阵行列式为 0, 不可逆")
            return None

        if show_steps:
            self.add_step("计算逆矩阵:")
            self.add_step(f"\\det(A) = {latex(det_A)}")

        # Calculate inverse matrix
        try:
            A_inv = A.inv()
            A_inv_simplified = self.simplify_matrix(A_inv)

            if show_steps:
                self.add_matrix(A_inv_simplified, "A^{-1}")

            # Calculate solution
            x = A_inv_simplified * b
            x_simplified = self.simplify_matrix(x)

            if show_steps:
                self.add_step("计算解:")
                self.add_equation(
                    "\\boldsymbol{x} = A^{-1} \\cdot \\boldsymbol{b}")
                self.add_vector(x_simplified, "\\boldsymbol{x}")

                # Verification
                self.add_step("验证:")
                b_check = A * x_simplified
                b_check_simplified = self.simplify_matrix(b_check)
                self.add_vector(b_check_simplified,
                                "A \\times \\boldsymbol{x}")
                self.add_vector(b, "\\boldsymbol{b}")
                if b_check_simplified == b:
                    self.add_step(
                        "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
                else:
                    self.add_step("验证失败")

            return x_simplified

        except Exception as e:
            if show_steps:
                self.add_step(f"求逆失败: {str(e)}")
            return None

    def solve_by_lu_decomposition(self, A_input: str, b_input: str, show_steps: bool = True) -> Matrix:
        """Method 4: LU decomposition method.

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            show_steps: Whether to display solution steps

        Returns:
            Solution vector or None if LU decomposition fails
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # Check system type
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            self.add_step("警告: 欠定系统，LU 分解法不适用")
            return self.solve_underdetermined_system(A, b, show_steps)
        if system_type == "overdetermined":
            self.add_step("警告: 超定系统，LU 分解法不适用")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法四: LU 分解法}")
            self.display_system(A, b)

        if not self.is_square(A):
            if show_steps:
                self.add_step("矩阵不是方阵, 不能使用 LU 分解法")
            return None

        n = A.rows

        # Perform LU decomposition (using Doolittle method)
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
                    if show_steps:
                        self.add_step("不能进行 LU 分解")
                    return None
                L[j, i] = (A[j, i] - sum_val) / U[i, i]

        if show_steps:
            self.add_step("LU 分解:")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            # Verify LU decomposition
            self.add_step("验证 LU 分解:")
            LU_product = L * U
            self.add_matrix(LU_product, "L \\times U")
            if LU_product == A:
                self.add_step("LU 分解正确")
            else:
                self.add_step("LU 分解错误")

        # Solve system L * y = b and U * x = y
        if show_steps:
            self.add_step("解方程组:")
            self.add_equation(
                "解: L \\cdot \\boldsymbol{y} = \\boldsymbol{b} \\quad \\text{和} \\quad U \\cdot \\boldsymbol{x} = \\boldsymbol{y}")

        # Forward substitution to solve L * y = b
        y = zeros(n, 1)
        if show_steps:
            self.add_step(
                "(1) 前代法求解 L \\cdot \\boldsymbol{y} = \\boldsymbol{b}")

        for i in range(n):
            sum_val = sum(L[i, j] * y[j] for j in range(i))
            y[i] = (b[i] - sum_val) / L[i, i]

            if show_steps:
                if i == 0:
                    self.add_step(
                        f"y_{{{i+1}}} = \\frac{{b_{{{i+1}}}}}{{L_{{{i+1}{i+1}}}}} = \\frac{{{latex(b[i])}}}{{{latex(L[i, i])}}} = {latex(y[i])}")
                else:
                    sum_terms = " + ".join(
                        [f"L_{{{i+1}{j+1}}} \\cdot y_{{{j+1}}}" for j in range(i)])
                    self.add_step(
                        f"y_{{{i+1}}} = \\frac{{b_{{{i+1}}} - ({sum_terms})}}{{L_{{{i+1}{i+1}}}}} = \\frac{{{latex(b[i])} - ({latex(sum_val)})}}{{{latex(L[i, i])}}} = {latex(y[i])}")

        if show_steps:
            self.add_vector(y, "\\boldsymbol{y}")

        # Back substitution to solve U * x = y
        x = zeros(n, 1)
        if show_steps:
            self.add_step(
                "(2) 回代法求解 U \\cdot \\boldsymbol{x} = \\boldsymbol{y}")

        for i in range(n-1, -1, -1):
            sum_val = sum(U[i, j] * x[j] for j in range(i+1, n))
            x[i] = (y[i] - sum_val) / U[i, i]

            if show_steps:
                if i == n-1:
                    self.add_step(
                        f"x_{{{i+1}}} = \\frac{{y_{{{i+1}}}}}{{U_{{{i+1}{i+1}}}}} = \\frac{{{latex(y[i])}}}{{{latex(U[i, i])}}} = {latex(x[i])}")
                else:
                    sum_terms = " + ".join(
                        [f"U_{{{i+1}{j+1}}} \\cdot x_{{{j+1}}}" for j in range(i+1, n)])
                    self.add_step(
                        f"x_{{{i+1}}} = \\frac{{y_{{{i+1}}} - ({sum_terms})}}{{U_{{{i+1}{i+1}}}}} = \\frac{{{latex(y[i])} - ({latex(sum_val)})}}{{{latex(U[i, i])}}} = {latex(x[i])}")

        x_simplified = self.simplify_matrix(x)

        if show_steps:
            self.add_step("最终解:")
            self.add_vector(x_simplified, "\\boldsymbol{x}")

            # Verification
            self.add_step("验证:")
            b_check = A * x_simplified
            b_check_simplified = self.simplify_matrix(b_check)
            self.add_vector(b_check_simplified, "A \\times \\boldsymbol{x}")
            self.add_vector(b, "\\boldsymbol{b}")
            if b_check_simplified == b:
                self.add_step(
                    "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
            else:
                self.add_step("验证失败")

        return x_simplified

    def solve_by_cramers_rule(self, A_input: str, b_input: str, show_steps: bool = True) -> Matrix:
        """Method 5: Cramer's rule.

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            show_steps: Whether to display solution steps

        Returns:
            Solution vector or None if Cramer's rule cannot be applied
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # Check system type
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            self.add_step("警告: 欠定系统，克莱姆法则不适用")
            return self.solve_underdetermined_system(A, b, show_steps)
        if system_type == "overdetermined":
            self.add_step("警告: 超定系统，克莱姆法则不适用")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法五: 克莱姆法则}")
            self.display_system(A, b)

        if not self.is_square(A):
            if show_steps:
                self.add_step("矩阵不是方阵, 不能使用克莱姆法则")
            return None

        n = A.rows

        # Calculate determinant of coefficient matrix
        det_A = A.det()
        if show_steps:
            self.add_step(f"\\det(A) = {latex(det_A)}")

        if det_A == 0:
            if show_steps:
                self.add_step("行列式为 0, 克莱姆法则不适用")
            return None

        # Solve using Cramer's rule
        x = zeros(n, 1)

        for i in range(n):
            # Create replacement matrix
            A_i = A.copy()
            for j in range(n):
                A_i[j, i] = b[j]

            det_A_i = A_i.det()

            if show_steps:
                self.add_step(f"计算 x_{{{i+1}}}:")
                self.add_matrix(A_i, f"A_{{{i+1}}}")
                self.add_step(f"\\det(A_{{{i+1}}}) = {latex(det_A_i)}")
                self.add_step(
                    f"x_{{{i+1}}} = \\frac{{\\det(A_{{{i+1}}})}}{{\\det(A)}} = \\frac{{{latex(det_A_i)}}}{{{latex(det_A)}}}")

            x[i] = det_A_i / det_A

        x_simplified = self.simplify_matrix(x)

        if show_steps:
            self.add_step("最终解:")
            self.add_vector(x_simplified, "\\boldsymbol{x}")

            # Verification
            self.add_step("验证:")
            b_check = A * x_simplified
            b_check_simplified = self.simplify_matrix(b_check)
            self.add_vector(b_check_simplified, "A \\times \\boldsymbol{x}")
            self.add_vector(b, "\\boldsymbol{b}")
            if b_check_simplified == b:
                self.add_step(
                    "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
            else:
                self.add_step("验证失败")

        return x_simplified

    def solve(self, A_input: str, b_input: str, method: str = 'auto', show_steps: bool = True) -> Matrix:
        """Main solver function.

        Parameters:
            A_input: Coefficient matrix
            b_input: Constant vector
            method: Solution method ('auto', 'gaussian', 'gauss_jordan',
                   'inverse', 'lu', 'cramer', 'underdetermined',
                   'overdetermined', 'singular')
            show_steps: Whether to display solution steps

        Returns:
            Solution vector or None if no solution exists
        """
        self.step_generator.clear()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # Automatically select method
        if method == 'auto':
            system_type = self.check_system_type(A, b, show_steps)

            if system_type == "unique_solution":
                if A.rows <= 3:
                    method = 'gauss_jordan'
                else:
                    method = 'lu'
            elif system_type == "underdetermined":
                method = 'underdetermined'
            elif system_type == "overdetermined":
                method = 'overdetermined'
            elif system_type == "singular_infinite":
                method = 'singular'
            elif system_type == "singular_no_solution":
                return None
            else:
                method = 'gauss_jordan'

        # Solve according to selected method
        if method == 'gaussian':
            return self.solve_by_gaussian_elimination(A, b, show_steps)
        if method == 'gauss_jordan':
            return self.solve_by_gauss_jordan(A, b, show_steps)
        if method == 'inverse':
            return self.solve_by_matrix_inverse(A, b, show_steps)
        if method == 'lu':
            return self.solve_by_lu_decomposition(A, b, show_steps)
        if method == 'cramer':
            return self.solve_by_cramers_rule(A, b, show_steps)
        if method == 'underdetermined':
            return self.solve_underdetermined_system(A, b, show_steps)
        if method == 'overdetermined':
            return self.solve_overdetermined_system(A, b, show_steps)
        if method == 'singular':
            return self.solve_singular_system(A, b, show_steps)

        raise ValueError(f"Unknown solving method: {method}")


# # Demo functions
# def demo_underdetermined_systems():
#     """Demonstrate solving underdetermined systems."""
#     solver = LinearSystemSolver()

#     # Underdetermined system examples
#     under_A1 = '[[1,2,3],[4,5,6]]'  # 2 equations, 3 unknowns
#     under_b1 = '[7,8]'

#     under_A2 = '[[1,1,1,1],[0,1,1,1]]'  # 2 equations, 4 unknowns
#     under_b2 = '[5,3]'

#     solver.add_step("\\textbf{欠定线性方程组求解演示}")

#     under_systems = [(under_A1, under_b1), (under_A2, under_b2)]

#     for i, (A, b) in enumerate(under_systems, 1):
#         solver.add_step(f"\\textbf{{欠定系统示例 {i}}}")
#         try:
#             solver.solve_underdetermined_system(A, b)
#             solver.display_steps()
#             solver.step_generator.clear()
#         except Exception as e:
#             solver.add_step(f"\\text{{错误: }} {str(e)}")
#             solver.display_steps()
#             solver.step_generator.clear()


# def demo_overdetermined_systems():
#     """Demonstrate solving overdetermined systems."""
#     solver = LinearSystemSolver()

#     # Overdetermined system examples
#     over_A1 = '[[1,2],[3,4],[5,6],[2,6]]'
#     over_b1 = '[7,8,9,3]'

#     over_A2 = '[[1,1],[2,1],[3,1]]'
#     over_b2 = '[3,4,5]'

#     solver.add_step("\\textbf{超定线性方程组求解演示}")

#     over_systems = [(over_A1, over_b1), (over_A2, over_b2)]

#     for i, (A, b) in enumerate(over_systems, 1):
#         solver.add_step(f"\\textbf{{超定系统示例 {i}}}")
#         try:
#             result = solver.solve_overdetermined_system(A, b)
#             if result is not None:
#                 solver.add_vector(result, "\\boldsymbol{x}")
#             solver.display_steps()
#             solver.step_generator.clear()
#         except Exception as e:
#             solver.add_step(f"\\text{{错误: }} {str(e)}")
#             solver.display_steps()
#             solver.step_generator.clear()


# def demo_basic_systems():
#     """Demonstrate solving basic linear systems."""
#     solver = LinearSystemSolver()

#     # Solvable system examples
#     A1 = '[[2,1],[1,3]]'
#     b1 = '[5,10]'

#     A2 = '[[1,2,3],[0,1,4],[5,6,0]]'
#     b2 = '[14,7,8]'

#     A3 = '[[1,1],[2,3]]'
#     b3 = '[5,13]'

#     solver.add_step("\\textbf{基本线性方程组求解演示}")

#     test_systems = [(A1, b1), (A2, b2), (A3, b3)]

#     for i, (A, b) in enumerate(test_systems, 1):
#         solver.add_step(f"\\textbf{{示例 {i}}}")
#         try:
#             solver.solve(A, b, method='auto', show_steps=True)
#             solver.display_steps()
#             solver.step_generator.clear()
#         except Exception as e:
#             solver.add_step(f"\\text{{错误: }} {str(e)}")
#             solver.display_steps()
#             solver.step_generator.clear()


# def demo_special_matrices():
#     """Demonstrate solving systems with special matrices."""
#     solver = LinearSystemSolver()

#     # Special matrix examples
#     identity_A = '[[1,0,0],[0,1,0],[0,0,1]]'
#     diagonal_A = '[[2,0,0],[0,3,0],[0,0,5]]'
#     permutation_A = '[[0,1,0],[0,0,1],[1,0,0]]'
#     upper_triangular_A = '[[1,2,3],[0,4,5],[0,0,6]]'
#     lower_triangular_A = '[[1,0,0],[2,3,0],[4,5,6]]'

#     b = '[1,2,3]'

#     solver.add_step("\\textbf{特殊矩阵系统求解演示}")

#     special_cases = [
#         ("单位矩阵", identity_A, b),
#         ("对角矩阵", diagonal_A, b),
#         ("置换矩阵", permutation_A, b),
#         ("上三角矩阵", upper_triangular_A, b),
#         ("下三角矩阵", lower_triangular_A, b)
#     ]

#     for name, A, b in special_cases:
#         solver.add_step(f"\\textbf{{{name}}}")
#         try:
#             # Use Gaussian elimination to demonstrate solving with special matrices
#             solver.solve_by_gaussian_elimination(A, b, show_steps=True)
#             solver.display_steps()
#             solver.step_generator.clear()
#         except Exception as e:
#             solver.add_step(f"\\text{{错误: }} {str(e)}")
#             solver.display_steps()
#             solver.step_generator.clear()


# def demo_singular_systems():
#     """Demonstrate solving singular systems."""
#     solver = LinearSystemSolver()

#     # Singular system examples
#     # Row linearly dependent, infinite solutions
#     singular_A1 = '[[1,2,3],[4,5,6],[7,8,9]]'
#     singular_b1 = '[1,2,3]'

#     singular_A2 = '[[1,1],[1,1]]'  # Identical rows, infinite solutions
#     singular_b2 = '[2,2]'

#     # Identical rows but different constants, no solution
#     singular_A3 = '[[1,1],[1,1]]'
#     singular_b3 = '[2,3]'

#     singular_A4 = '[[1,2],[2,4]]'  # Second row is multiple of first
#     singular_b4 = '[1,2]'  # Has solution

#     singular_A5 = '[[1,2],[2,4]]'  # Second row is multiple of first
#     singular_b5 = '[1,3]'  # No solution

#     solver.add_step("\\textbf{{奇异系统演示}}")

#     singular_systems = [
#         ("无穷多解示例1", singular_A1, singular_b1),
#         ("无穷多解示例2", singular_A2, singular_b2),
#         ("无解示例1", singular_A3, singular_b3),
#         ("无穷多解示例3", singular_A4, singular_b4),
#         ("无解示例2", singular_A5, singular_b5)
#     ]

#     for name, A, b in singular_systems:
#         solver.add_step(f"\\textbf{{{name}}}")
#         try:
#             solver.solve(A, b, method='auto', show_steps=True)
#             solver.display_steps()
#             solver.step_generator.clear()
#         except Exception as e:
#             solver.add_step(f"\\text{{错误: }} {str(e)}")
#             solver.display_steps()
#             solver.step_generator.clear()


# def demo_symbolic_systems():
#     """Demonstrate solving symbolic systems."""
#     solver = LinearSystemSolver()

#     # Symbolic systems
#     symbolic_A_2x2 = '[[a,b],[c,d]]'
#     symbolic_b_2x2 = '[p,q]'

#     symbolic_A_3x3 = '[[a,b,c],[d,e,f],[g,h,i]]'
#     symbolic_b_3x3 = '[p,q,r]'

#     solver.add_step("\\textbf{{符号系统求解演示}}")
#     solver.add_step("\\textbf{假设所有符号表达式不为 0, 可作分母}")

#     solver.add_step("\\textbf{2×2 符号系统}")
#     try:
#         solver.solve(symbolic_A_2x2, symbolic_b_2x2,
#                      method='auto', show_steps=True)
#         solver.display_steps()
#         solver.step_generator.clear()
#     except Exception as e:
#         solver.add_step(f"\\text{{错误: }} {str(e)}")
#         solver.display_steps()
#         solver.step_generator.clear()

#     solver.add_step("\\textbf{3×3 符号系统}")
#     try:
#         solver.solve(symbolic_A_3x3, symbolic_b_3x3,
#                      method='auto', show_steps=True)
#         solver.display_steps()
#         solver.step_generator.clear()
#     except Exception as e:
#         solver.add_step(f"\\text{{错误: }} {str(e)}")
#         solver.display_steps()
#         solver.step_generator.clear()


# def demo_all_methods():
#     """Demonstrate all solving methods."""
#     solver = LinearSystemSolver()

#     # Test system
#     A = '[[2,1],[1,3]]'
#     b = '[5,10]'

#     methods = [
#         ('gaussian', '高斯消元法'),
#         ('gauss_jordan', '高斯-约当消元法'),
#         ('inverse', '矩阵求逆法'),
#         ('lu', 'LU分解法'),
#         ('cramer', '克莱姆法则')
#     ]

#     solver.add_step("\\textbf{{所有求解方法演示}}")

#     for method_key, method_name in methods:
#         solver.add_step(f"\\textbf{{{method_name}}}")
#         try:
#             solver.solve(A, b, method=method_key, show_steps=True)
#             solver.display_steps()
#             solver.step_generator.clear()
#         except Exception as e:
#             solver.add_step(f"\\text{{错误: }} {str(e)}")
#             solver.display_steps()
#             solver.step_generator.clear()


# def demo_auto_solve():
#     """Demonstrate automatic solving functionality."""
#     solver = LinearSystemSolver()

#     # Various types of systems
#     systems = [
#         ("唯一解系统", '[[2,1],[1,3]]', '[5,10]'),
#         ("欠定系统", '[[1,2,3],[4,5,6]]', '[7,8]'),
#         ("超定系统", '[[1,2],[3,4],[5,6]]', '[7,8,9]'),
#         ("上三角系统", '[[1,2,3],[0,4,5],[0,0,6]]', '[1,2,3]'),
#         ("对角系统", '[[2,0,0],[0,3,0],[0,0,5]]', '[4,6,10]')
#     ]

#     solver.add_step("\\textbf{{自动求解功能演示}}")

#     for name, A, b in systems:
#         solver.add_step(f"\\textbf{{{name}}}")
#         try:
#             solver.solve(A, b, method='auto', show_steps=True)
#             solver.display_steps()
#             solver.step_generator.clear()
#         except Exception as e:
#             solver.add_step(f"\\text{{错误: }} {str(e)}")
#             solver.display_steps()
#             solver.step_generator.clear()


# if __name__ == "__main__":
#     # Run all demos
#     demo_basic_systems()
#     demo_special_matrices()
#     demo_singular_systems()
#     demo_symbolic_systems()
#     demo_underdetermined_systems()
#     demo_overdetermined_systems()
#     demo_all_methods()
#     demo_auto_solve()
