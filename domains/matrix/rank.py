from itertools import combinations
from sympy import eye, latex, simplify, zeros
# from IPython.display import Math, display

from core import CommonMatrixCalculator


class Rank(CommonMatrixCalculator):
    """Class for computing matrix rank using various methods."""

    def rank_by_row_echelon(self, matrix_input: str, show_steps: bool = True, simplify_result: bool = True, is_clear: bool = True) -> int:
        """Method 1: Compute rank using row echelon form.

        Parameters:
            matrix_input: Matrix input data
            show_steps (bool): Whether to show computation steps
            simplify_result (bool): Whether to simplify the result
            is_clear (bool): Whether to clear previous steps

        Returns:
            int: The rank of the matrix
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法一: 行阶梯形法}")
            self.add_matrix(A, "A")
            self.add_equation(r"\text{原理: 通过初等行变换化为行阶梯形，非零行数即为秩}")

        m, n = A.rows, A.cols
        rank = 0
        current_matrix = A.copy()

        if show_steps:
            self.add_step("初等行变换过程")

        # Gaussian elimination to row echelon form
        pivot_row = 0
        pivot_col = 0

        while pivot_row < m and pivot_col < n:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{处理第 {pivot_row+1} 行, 处理第 {pivot_col+1} 列:}}")

            # Find pivot element
            pivot_found = False
            for r in range(pivot_row, m):
                if current_matrix[r, pivot_col] != 0:
                    pivot_found = True
                    if r != pivot_row:
                        if show_steps:
                            self.step_generator.add_step(
                                f"\\text{{行交换: }} R_{pivot_row+1} \\leftrightarrow R_{r+1}")
                        current_matrix.row_swap(pivot_row, r)
                        if show_steps:
                            self.add_matrix(current_matrix, "A")
                    break

            if pivot_found:
                rank += 1
                pivot = current_matrix[pivot_row, pivot_col]

                # Normalize pivot row (optional, for easier calculation)
                if pivot != 1 and show_steps:
                    self.step_generator.add_step(
                        f"\\text{{归一化: }} R_{pivot_row+1} \\times {latex(1/pivot)}")
                    current_matrix[pivot_row,
                                   :] = current_matrix[pivot_row, :] / pivot
                    if show_steps:
                        self.add_matrix(current_matrix, "A")

                # Eliminate elements below pivot
                for r in range(pivot_row + 1, m):
                    if current_matrix[r, pivot_col] != 0:
                        factor = current_matrix[r, pivot_col]
                        if show_steps:
                            self.step_generator.add_step(
                                f"\\text{{消元: }} R_{r+1} - {latex(factor)} \\times R_{pivot_row+1}")
                        current_matrix[r, :] = current_matrix[r,
                                                              :] - factor * current_matrix[pivot_row, :]
                        if show_steps:
                            self.add_matrix(current_matrix, "A")

                pivot_row += 1

            else:
                self.step_generator.add_step(
                    f"\\text{{第 {pivot_col+1} 列没有主元}}")

            pivot_col += 1

        # Simplify result
        if simplify_result:
            current_matrix_simplified = self.simplify_matrix(current_matrix)
        else:
            current_matrix_simplified = current_matrix

        if show_steps:
            self.add_step("最终行阶梯形矩阵")
            self.add_matrix(current_matrix_simplified, "A_{ref}")

            # Count non-zero rows
            non_zero_rows = 0
            for i in range(m):
                if any(current_matrix_simplified[i, j] != 0 for j in range(n)):
                    non_zero_rows += 1

            self.step_generator.add_step(f"\\text{{非零行数: }} {non_zero_rows}")
            self.step_generator.add_step(
                f"\\text{{矩阵的秩: }} \\operatorname{{rank}}(A) = {rank}")

            # Explanation of result
            if rank == min(m, n):
                self.step_generator.add_step(r"\text{说明: 这是满秩矩阵}")
            elif rank == 0:
                self.step_generator.add_step(r"\text{说明: 这是零矩阵}")
            else:
                self.step_generator.add_step(
                    f"\\text{{说明: 秩亏缺 {min(m, n) - rank}}}")

        return rank

    def rank_by_determinants(self, matrix_input: str, show_steps: bool = True, is_clear: bool = True) -> int:
        """Method 2: Compute rank using determinants of submatrices.

        Parameters:
            matrix_input: Matrix input data
            show_steps (bool): Whether to show computation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            int: The rank of the matrix
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法二: 子式法}")
            self.add_matrix(A, "A")
            self.add_equation(r"\text{原理: 矩阵的秩是其最高阶非零子式的阶数}")

        m, n = A.rows, A.cols
        rank = 0
        max_possible_rank = min(m, n)

        if show_steps:
            self.add_step("计算过程")
            self.step_generator.add_step(
                f"\\text{{最大可能秩: }} \\min({m}, {n}) = {max_possible_rank}")

        # Check minors starting from maximum possible order
        for k in range(max_possible_rank, 0, -1):
            if show_steps:
                self.step_generator.add_step(f"\\text{{检查 {k} 阶子式}}")

            found_nonzero_minor = False
            nonzero_minor_example = None
            checked_count = 0

            # Generate all possible k×k submatrices
            row_combinations = list(combinations(range(m), k))
            col_combinations = list(combinations(range(n), k))

            for rows in row_combinations:
                for cols in col_combinations:
                    checked_count += 1

                    # Extract submatrix
                    submatrix = zeros(k, k)
                    for i, row in enumerate(rows):
                        for j, col in enumerate(cols):
                            submatrix[i, j] = A[row, col]

                    det = submatrix.det()
                    simplified_det = simplify(det)

                    # Show details of first few minors
                    if show_steps:
                        # Display row and column indices (starting from 1)
                        row_str = ','.join(str(r+1) for r in rows)
                        col_str = ','.join(str(c+1) for c in cols)

                        # Display minor matrix
                        self.step_generator.add_step(
                            f"\\text{{选取第 }}{{{row_str}}}\\text{{ 行和第 }}{{{col_str}}}\\text{{ 列}}")

                        submatrix_latex = r"\begin{bmatrix}"
                        for i in range(k):
                            row_elements = []
                            for j in range(k):
                                row_elements.append(
                                    latex(simplify(submatrix[i, j])))
                            submatrix_latex += " & ".join(row_elements)
                            if i < k-1:
                                submatrix_latex += r" \\ "
                        submatrix_latex += r"\end{bmatrix}"

                        self.step_generator.add_step(
                            f"\\text{{子式矩阵: }} {submatrix_latex}")

                        # Display determinant calculation
                        det_status = "\\neq 0" if simplified_det != 0 else ""
                        det_latex = latex(
                            simplified_det) if simplified_det != 0 else "0"
                        self.step_generator.add_step(
                            f"\\det = {det_latex} {det_status}")

                    # If a non-zero minor is found, record it and break all loops
                    if simplified_det != 0:
                        nonzero_minor_example = (
                            rows, cols, simplified_det, submatrix)
                        found_nonzero_minor = True
                        break  # Break inner loop

                if found_nonzero_minor:
                    break  # Break outer loop

            if show_steps and not found_nonzero_minor:
                self.step_generator.add_step(
                    f"\\text{{共检查了 {checked_count} 个 {k} 阶子式}}")

            if found_nonzero_minor:
                rank = k
                if show_steps and nonzero_minor_example:
                    rows, cols, det_value, submatrix = nonzero_minor_example
                    row_str = ','.join(str(r+1) for r in rows)
                    col_str = ','.join(str(c+1) for c in cols)

                    self.step_generator.add_step(f"\\textbf{{找到非零 {k} 阶子式:}}")
                    self.step_generator.add_step(
                        f"\\text{{选取: 第 }}{{{row_str}}}\\text{{ 行, 第 }}{{{col_str}}}\\text{{ 列}}")

                    # Display minor matrix
                    submatrix_latex = r"\begin{bmatrix}"
                    for i in range(k):
                        row_elements = []
                        for j in range(k):
                            row_elements.append(
                                latex(simplify(submatrix[i, j])))
                        submatrix_latex += " & ".join(row_elements)
                        if i < k-1:
                            submatrix_latex += r" \\ "
                    submatrix_latex += r"\end{bmatrix}"

                    self.step_generator.add_step(
                        f"\\text{{子式矩阵: }} {submatrix_latex}")
                    self.step_generator.add_step(
                        f"\\det = {latex(det_value)} \\neq 0")

                # End function immediately after finding a non-zero minor
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{最高阶非零子式的阶数: }} {rank}")
                    self.step_generator.add_step(
                        f"\\text{{矩阵的秩: }} \\operatorname{{rank}}(A) = {rank}")
                return rank

            if show_steps:
                self.step_generator.add_step(f"\\text{{所有 {k} 阶子式都为零}}")

        # If all minors are zero, rank is 0
        if rank == 0 and all(A[i, j] == 0 for i in range(m) for j in range(n)):
            rank = 0
            if show_steps:
                self.step_generator.add_step(r"\text{零矩阵, 所有元素都为0}")
        elif rank == 0:
            rank = 1 if any(A[i, j] != 0 for i in range(m)
                            for j in range(n)) else 0
            if show_steps and rank == 1:
                self.step_generator.add_step(
                    r"\text{存在非零元素, 但所有 2 阶及以上子式都为 0}")

        if show_steps:
            self.step_generator.add_step(f"\\text{{最高阶非零子式的阶数: }} {rank}")
            self.step_generator.add_step(
                f"\\text{{矩阵的秩: }} \\operatorname{{rank}}(A) = {rank}")

        return rank

    def rank_by_eigenvalues(self, matrix_input: str, show_steps: bool = True, is_clear: bool = True) -> int:
        """Method 3: Compute rank using eigenvalues (for square matrices only).

        Parameters:
            matrix_input: Matrix input data
            show_steps (bool): Whether to show computation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            int or None: The rank of the matrix, or None if not applicable
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法三: 特征值法}")
            self.add_matrix(A, "A")

        if A.rows != A.cols:
            if show_steps:
                self.step_generator.add_step(r"\text{不是方阵, 无法使用此方法}")
            return None

        try:
            # Calculate eigenvalues
            eigenvalues = A.eigenvals()

            if show_steps:
                self.add_equation(r"\text{原理: 对于方阵, 秩 = 非零特征值个数}")
                self.step_generator.add_step(rf"\text{{特征值: }}" + rf",\,".join(
                    [f"\\lambda_{{{i+1}}} = {latex(val)}" for i, val in enumerate(eigenvalues)]))

            # Count number of non-zero eigenvalues
            nonzero_eigenvalues = sum(
                1 for eig in eigenvalues if simplify(eig) != 0)
            rank = nonzero_eigenvalues

            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{非零特征值个数: }} {nonzero_eigenvalues}")
                self.step_generator.add_step(
                    f"\\text{{矩阵的秩: }} \\operatorname{{rank}}(A) = {rank}")

                # Special case explanations
                if rank == A.rows:
                    self.step_generator.add_step(r"\text{说明: 所有特征值非零，矩阵满秩且可逆}")
                elif rank == 0:
                    self.step_generator.add_step(r"\text{说明: 所有特征值为零，矩阵为幂零矩阵}")

            return rank

        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{特征值计算失败: {str(e)}}}")
            # Fallback to row echelon method
            return self.rank_by_row_echelon(A, show_steps=False)

    def rank_by_row_reduction(self, matrix_input: str, show_steps: bool = True, is_clear: bool = True) -> int:
        """Method 4: Compute rank using reduced row echelon form.

        Parameters:
            matrix_input: Matrix input data
            show_steps (bool): Whether to show computation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            int: The rank of the matrix
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法四: 行简化阶梯形法}")
            self.add_matrix(A, "A")
            self.add_equation(r"\text{原理: 行简化阶梯形的主元列数就是矩阵的秩}")

        # Compute reduced row echelon form
        rref_matrix, pivot_columns = A.rref()

        if show_steps:
            self.add_step("行简化阶梯形矩阵")
            self.add_matrix(rref_matrix, "A_{rref}")

        rank = len(pivot_columns)

        if show_steps:
            self.step_generator.add_step(
                f"\\text{{主元列位置: }} {[c+1 for c in pivot_columns]}")
            self.step_generator.add_step(f"\\text{{主元个数: }} {rank}")
            self.step_generator.add_step(
                f"\\text{{矩阵的秩: }} \\operatorname{{rank}}(A) = {rank}")

            # Explain significance of pivots
            if rank > 0:
                self.step_generator.add_step(r"\text{说明: 主元列对应的列向量构成列空间的一组基}")

        return rank

    def check_special_cases(self, matrix_input: str, show_steps: bool = True) -> int:
        """Check special cases for matrix rank.

        Parameters:
            matrix_input: Matrix input data
            show_steps (bool): Whether to show computation steps

        Returns:
            int or None: The rank if a special case is detected, otherwise None
        """
        A = self.parse_matrix_input(matrix_input)
        m, n = A.rows, A.cols

        if show_steps:
            self.add_step("特殊情况检查")

        # Check for zero matrix
        if all(A[i, j] == 0 for i in range(m) for j in range(n)):
            if show_steps:
                self.step_generator.add_step(r"\text{零矩阵: 所有元素为零}")
                self.step_generator.add_step(r"\operatorname{rank}(A) = 0")
            return 0

        # Check for identity matrix
        if m == n and A == eye(m):
            if show_steps:
                self.step_generator.add_step(r"\text{单位矩阵: 主对角线为 1, 其余为 0}")
                self.step_generator.add_step(
                    f"\\operatorname{{rank}}(I) = {m}")
                self.step_generator.add_step(r"\text{说明: 单位矩阵总是满秩的}")
            return m

        # Check for diagonal matrix
        is_diagonal = True
        for i in range(m):
            for j in range(n):
                if i != j and A[i, j] != 0:
                    is_diagonal = False
                    break
            if not is_diagonal:
                break

        if is_diagonal:
            nonzero_diag = sum(1 for i in range(min(m, n)) if A[i, i] != 0)
            if show_steps:
                self.step_generator.add_step(r"\text{对角矩阵: 非零元素只在主对角线上}")
                self.step_generator.add_step(
                    f"\\operatorname{{rank}}(A) = {nonzero_diag}")
                self.step_generator.add_step(r"\text{说明: 秩等于非零对角线元素的个数}")
            return nonzero_diag

        det_A = A.det()
        # Check for full-rank square matrix
        if m == n and det_A != 0:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{满秩方阵: }} \\det(A) = {latex(det_A)} \\neq 0")
                self.step_generator.add_step(
                    f"\\operatorname{{rank}}(A) = {m}")
                self.step_generator.add_step(r"\text{说明: 行列式非零意味着矩阵可逆，因此满秩}")
            return m

        return None

    def auto_matrix_rank(self, matrix_input: str, show_steps: bool = True, simplify_result: bool = True) -> int:
        """Automatically select method to compute matrix rank.

        Parameters:
            matrix_input: Matrix input data
            show_steps (bool): Whether to show computation steps
            simplify_result (bool): Whether to simplify the result

        Returns:
            int or None: The computed rank, or None if failed
        """
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{自动矩阵求秩}")
            self.add_matrix(A, "A")

        # First check special cases
        if A.rows == A.cols:
            special_rank = self.check_special_cases(matrix_input, show_steps)
            if special_rank is not None:
                return special_rank

        if show_steps:
            self.step_generator.add_step(r"\text{检测到一般矩阵, 使用多种方法求秩}")

        results = {}

        # Method 1: Row echelon form (most reliable)
        try:
            results["row_echelon"] = self.rank_by_row_echelon(
                matrix_input, show_steps, simplify_result, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{行阶梯形法失败: {str(e)}}}")

        # Method 2: Determinant method (suitable for small matrices)
        if A.rows <= 5 and A.cols <= 5:  # Limit matrix size to avoid combinatorial explosion
            try:
                results["determinants"] = self.rank_by_determinants(
                    matrix_input, show_steps, is_clear=False)
            except Exception as e:
                if show_steps:
                    self.step_generator.add_step(f"\\text{{子式法失败: {str(e)}}}")

        # Method 3: Eigenvalue method
        try:
            results["eigenvalues"] = self.rank_by_eigenvalues(
                matrix_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{特征值法失败: {str(e)}}}")

        # Method 4: Reduced row echelon form method
        try:
            results["rref"] = self.rank_by_row_reduction(
                matrix_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{行简化阶梯形法失败: {str(e)}}}")

        # Check consistency of results
        if show_steps and len(results) > 1:
            self.add_step("方法一致性检查")
            methods = list(results.keys())
            consistent = True
            first_result = results[methods[0]]
            for method in methods[1:]:
                if results[method] is None:
                    continue
                if results[method] != first_result:
                    consistent = False
                    break

            if consistent:
                self.step_generator.add_step(r"\text{所有方法结果一致}")
            else:
                self.step_generator.add_step(r"\textbf{警告: 不同方法结果不一致}")

        # Return most reliable result (row echelon form preferred)
        if "row_echelon" in results:
            return results["row_echelon"]
        if "rref" in results:
            return results["rref"]
        if results:
            return next(iter(results.values()))

        return None


# # Demo functions
# def demo_basic_rank():
#     """Demonstrate basic matrix rank computation."""
#     rank_calculator = Rank()

#     # Examples of matrices with different ranks
#     full_rank = '[[1,2,3],[4,5,6],[7,8,10]]'  # Full rank
#     rank_2 = '[[1,2,3],[4,5,6],[7,8,9]]'     # Rank 2
#     rank_1 = '[[1,2,3],[2,4,6],[3,6,9]]'     # Rank 1
#     zero_matrix = '[[0,0,0],[0,0,0],[0,0,0]]'  # Zero matrix

#     rank_calculator.step_generator.add_step(r"\textbf{基本矩阵求秩演示}")

#     test_matrices = [
#         ("满秩矩阵", full_rank),
#         ("秩为 2 的矩阵", rank_2),
#         ("秩为 1 的矩阵", rank_1),
#         ("零矩阵", zero_matrix)
#     ]

#     for name, matrix in test_matrices:
#         rank_calculator.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             rank_calculator.auto_matrix_rank(matrix)
#             display(Math(rank_calculator.get_steps_latex()))
#         except Exception as e:
#             rank_calculator.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(rank_calculator.get_steps_latex()))
#         rank_calculator.step_generator.add_step("\\" + "\\")


# def demo_rectangular_matrices():
#     """Demonstrate rectangular matrix rank computation."""
#     rank_calculator = Rank()

#     # Rectangular matrix examples
#     tall_full_rank = '[[1,2],[3,4],[5,6]]'      # Tall matrix, full rank
#     wide_full_rank = '[[1,2,3,4],[5,6,7,8]]'   # Wide matrix, full rank
#     tall_rank_deficient = '[[1,2],[2,4],[3,6]]'  # Tall matrix, rank deficient
#     # Wide matrix, rank deficient
#     wide_rank_deficient = '[[1,2,3,4],[2,4,6,8]]'

#     rank_calculator.step_generator.add_step(r"\textbf{矩形矩阵求秩演示}")

#     rectangular_matrices = [
#         ("高满秩矩阵", tall_full_rank),
#         ("宽满秩矩阵", wide_full_rank),
#         ("高秩不足矩阵", tall_rank_deficient),
#         ("宽秩不足矩阵", wide_rank_deficient)
#     ]

#     for name, matrix in rectangular_matrices:
#         rank_calculator.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             rank_calculator.auto_matrix_rank(matrix)
#             display(Math(rank_calculator.get_steps_latex()))
#         except Exception as e:
#             rank_calculator.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(rank_calculator.get_steps_latex()))
#         rank_calculator.step_generator.add_step("\\" + "\\")


# def demo_special_matrices():
#     """Demonstrate special matrix rank computation."""
#     rank_calculator = Rank()

#     # Special matrix examples
#     identity = '[[1,0,0],[0,1,0],[0,0,1]]'      # Identity matrix
#     diagonal_full = '[[2,0,0],[0,3,0],[0,0,5]]'  # Full rank diagonal matrix
#     diagonal_rank2 = '[[2,0,0],[0,0,0],[0,0,5]]'  # Diagonal matrix with rank 2
#     permutation = '[[0,1,0],[0,0,1],[1,0,0]]'   # Permutation matrix

#     rank_calculator.step_generator.add_step(r"\textbf{特殊矩阵求秩演示}")

#     special_matrices = [
#         ("单位矩阵", identity),
#         ("满秩对角矩阵", diagonal_full),
#         ("秩为 2 的对角矩阵", diagonal_rank2),
#         ("置换矩阵", permutation)
#     ]

#     for name, matrix in special_matrices:
#         rank_calculator.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             rank_calculator.auto_matrix_rank(matrix)
#             display(Math(rank_calculator.get_steps_latex()))
#         except Exception as e:
#             rank_calculator.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(rank_calculator.get_steps_latex()))
#         rank_calculator.step_generator.add_step("\\" + "\\")


# def demo_symbolic_matrix():
#     """Demonstrate symbolic matrix rank computation."""
#     rank_calculator = Rank()

#     # Symbolic matrices
#     symbolic_2x2 = '[[a,b],[c,d]]'
#     symbolic_3x3 = '[[a,b,c],[d,e,f],[g,h,i]]'
#     symbolic_rank1 = '[[a,b],[2*a,2*b]]'
#     symbolic_rank2 = '[[a,b,c],[2*a,2*b,2*c], [c,d,e]]'

#     rank_calculator.step_generator.add_step(r"\textbf{符号矩阵求秩演示}")
#     rank_calculator.step_generator.add_step(r"\textbf{假设所有符号表达式不为 0, 可作主元}")

#     symbolic_matrices = [
#         ("2×2 符号矩阵", symbolic_2x2),
#         ("3×3 符号矩阵", symbolic_3x3),
#         ("秩为 2 的符号矩阵", symbolic_rank2),
#         ("秩为 1 的符号矩阵", symbolic_rank1)
#     ]

#     for name, matrix in symbolic_matrices:
#         rank_calculator.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             rank_calculator.auto_matrix_rank(matrix)
#             display(Math(rank_calculator.get_steps_latex()))
#         except Exception as e:
#             rank_calculator.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(rank_calculator.get_steps_latex()))
#         rank_calculator.step_generator.add_step("\\" + "\\")


# if __name__ == "__main__":
#     demo_basic_rank()
#     demo_rectangular_matrices()
#     demo_special_matrices()
#     demo_symbolic_matrix()
