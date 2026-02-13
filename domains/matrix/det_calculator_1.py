from typing import List, Tuple

from sympy import Expr, Matrix, latex, simplify, sympify

from domains.matrix import RefStepGenerator


class DeterminantCalculator:
    """A class for calculating matrix determinants using various methods with step-by-step solutions.

    This calculator provides multiple approaches to compute determinants including:
    - Laplace expansion.
    - Sarrus rule (for 3*3 matrices).
    - LU decomposition.
    - Gaussian elimination.
    - Property-based simplification.
    """

    def __init__(self) -> None:
        """Initialize the determinant calculator with a step generator."""
        self.step_generator = RefStepGenerator()

    def _reset(self) -> None:
        """Reset the step generator to start fresh."""
        self.step_generator.reset()

    @staticmethod
    def _parse_matrix_input(matrix_input: str) -> Matrix:
        """Parse the input string into a SymPy Matrix."""
        try:
            M = Matrix(sympify(matrix_input))
            if M.shape[0] != M.shape[1]:  # The Matrix must be square
                raise ValueError("矩阵必须为方阵")
            return M
        except Exception as e:
            raise ValueError("无法解析矩阵输入, 请使用格式如 '[[1,2],[3,4]]'") from e

    def _add_section_title(self, title: str) -> None:
        """Add a section title with surrounding blank lines for better readability."""
        # New line
        self.step_generator.add_step("", "")
        self.step_generator.add_step(title, "")
        self.step_generator.add_step("", "")

    @staticmethod
    def _best_expand_axis(M: Matrix) -> Tuple[str, int]:
        """Find the best axis (row/column) to expand along based on the number of non-zero elements.

        Returns:
            Tuple: A tuple containing ("row"/"col", index) of the best expansion axis.
        """
        n = M.shape[0]
        min_nonzeros = n + 1
        best = ("row", 0)
        for i in range(n):
            cnt = sum(1 for x in M.row(i) if simplify(x) != 0)
            if cnt < min_nonzeros:
                min_nonzeros = cnt
                best = ("row", i)

            cnt = sum(1 for x in M.col(i) if simplify(x) != 0)
            if cnt < min_nonzeros:
                min_nonzeros = cnt
                best = ("col", i)
        return best

    def _method_laplace(self, A: Matrix) -> Expr:
        """Calculate determinant using Laplace expansion along the best row/column.

        Returns:
            The calculated determinant value
        """
        # Handle direct cases for 1*1 and 2*2 matrices
        n = A.shape[0]
        original_n = n
        if n == 1:
            val = simplify(A[0, 0])
            self.step_generator.add_step(val, rf"$|1×1\,行列式|=元素本身$")
            return val
        if n == 2:
            res = simplify(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
            self.step_generator.add_step(res, rf"$2×2\,行列式$")
            return res

        # Construct expression latex (join terms with + and replace '+-' with '-')
        def expr_to_latex(terms):
            parts = []
            for t in terms:
                s_ = term_to_latex(t)
                if s_:
                    parts.append(s_)
            if not parts:
                return "0"
            expr = "+".join(parts)
            expr = expr.replace("+-", "-")
            expr = expr.replace("--", "+")
            return expr

        # active_terms stores current expression terms list
        # Term is (coeff, obj), representing coeff * det(obj)
        active_terms = [(1, A)]

        # Main loop: find first term to expand and replace it, record before/after expressions.
        while True:
            idx_to_expand = None
            # Find first term that needs expansion
            for idx, (coeff, obj) in enumerate(active_terms):
                if isinstance(obj, Matrix) and obj.shape[0] > 1 and simplify(coeff) != 0:
                    idx_to_expand = idx
                    break

            # No matrix terms need expansion, exit
            if idx_to_expand is None:
                break

            coeff, M = active_terms[idx_to_expand]
            axis, ind = self._best_expand_axis(M)
            expand_desc = rf"按第\,{ind + 1}\,{'行' if axis == 'row' else '列'}展开"

            n = M.shape[0]
            new_terms = []
            for j in range(n):
                if axis == "row":
                    a = simplify(M[ind, j])
                    if a == 0:
                        continue
                    s = (-1) ** (ind + j)
                    minor = M.minor_submatrix(ind, j)
                else:
                    a = simplify(M[j, ind])
                    if a == 0:
                        continue
                    s = (-1) ** (j + ind)
                    minor = M.minor_submatrix(j, ind)

                new_coeff = simplify(coeff * s * a)

                # If a minor is 1*1 or 2*2, calculate directly to replace with the scalar term.
                if minor.shape[0] == 1:
                    scalar = simplify(minor[0, 0])
                    new_terms.append((new_coeff, scalar))
                # For 3*3 matrices, still show minor expansion
                elif minor.shape[0] == 2 and original_n != 3:
                    det2 = simplify(
                        minor[0, 0] * minor[1, 1] - minor[0, 1] * minor[1, 0])
                    new_terms.append((new_coeff, det2))
                else:
                    new_terms.append((new_coeff, minor))

            # Replace the idx_to_expand term in active_terms with new_terms
            active_terms = active_terms[:idx_to_expand] + new_terms + active_terms[idx_to_expand + 1:]
            new_expr = expr_to_latex(active_terms)

            # Record this step's equality chain
            self.step_generator.add_step(
                rf"$\;\Rightarrow\;{new_expr}$", expand_desc)
            self.step_generator.add_step("", "")

        final_sum = simplify(sum(simplify(c * o) for c, o in active_terms))  # type: ignore
        self.step_generator.add_step(
            rf'$\;\Rightarrow\;{latex(final_sum)}$', "最终结果")

        return final_sum

    # Sarrus rule (3x3)
    def _method_sarrus(self, A: Matrix) -> Expr | None:
        """Calculate determinant using Sarrus rule for 3x3 matrices.

        Returns:
            The determinant value or None if matrix is not 3x3
        """
        if A.shape != (3, 3):
            return None
        a, b, c = A[0, 0], A[0, 1], A[0, 2]
        d, e, f = A[1, 0], A[1, 1], A[1, 2]
        g, h, i = A[2, 0], A[2, 1], A[2, 2]
        pos = simplify(a * e * i + b * f * g + c * d * h)
        neg = simplify(c * e * g + a * f * h + b * d * i)
        res = simplify(pos - neg)
        self.step_generator.add_step(
            pos,
            rf"$主和\,=\,{latex(a)} \times {latex(e)} \times {latex(i)}+{latex(b)} \times {latex(f)}"
            rf"\times {latex(g)}+{latex(c)} \times {latex(d)} \times {latex(h)}$")
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            neg,
            rf"$副和\,=\,{latex(c)} \times {latex(e)} \times {latex(g)}+{latex(a)} \times {latex(f)}"
            rf"\times {latex(h)}+{latex(b)} \times {latex(d)} \times {latex(i)}$")
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            res, rf"$结果\,=\,主和 -\,副和\,=\,{latex(res)}$")
        return res

    def _method_lu(self, A: Matrix) -> Expr | None:
        """Calculate determinant using LU decomposition method.

        Returns:
            The determinant value or None if LU decomposition fails
        """
        try:
            L, U, perm = A.LUdecomposition()
        except Exception:
            self.step_generator.add_step(A.copy(), "LU 分解失败或不可用")
            return None

        # Construct permutation matrix P
        P = Matrix.eye(A.rows)
        for (i, j) in perm:
            P.row_swap(i, j)
        # Calculate the number of permutations (row swaps)
        swap_count = len(perm)
        # Each row swap flips the determinant sign
        det_P = (-1) ** swap_count

        # Calculate final determinant: |A| = |P^-1| * |L| * |U| = |P| * |L| * |U|
        det_val = simplify(det_P * L.det() * U.det())

        # Record steps
        if perm:
            # With row swaps
            self.step_generator.add_step(
                P.copy(), rf"$P(置换矩阵),行交换次数={swap_count}.\;|P| = (-1)^{{{swap_count}}} = {det_P}$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(L.copy(), "$L(下三角矩阵)$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(U.copy(), "$U(上三角矩阵)$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                det_val,
                rf"$|A|=|P| \cdot |L| \cdot |U|={det_P} \times {latex(L.det())}"
                rf"\times {latex(U.det())}={latex(det_val)}$"
            )
        else:
            # No row swaps
            self.step_generator.add_step(L.copy(), "$L(下三角矩阵)$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(U.copy(), "$U(上三角矩阵)$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                det_val,
                f"$|A|=|L| \\cdot |U|={latex(L.det())} \\times {latex(U.det())}={latex(det_val)}$"
            )

        return det_val

    # Properties method (quick detection of proportional rows/triangular etc.)
    def _method_properties(self, A: Matrix) -> Expr | int | None:
        """Check for special matrix properties that simplify determinant calculation.

        This method checks for:
        - Zero rows/columns.
        - Identical or proportional rows/columns.
        - Diagonal/upper/lower triangular matrices.

        Returns:
            Determinant value if special property found, otherwise None
        """
        n = A.rows
        m = A.cols
        # Zero row
        for i in range(n):
            if all(simplify(A[i, j]) == 0 for j in range(m)):
                self.step_generator.add_step(
                    0, rf"$第\,{i + 1}\,行全为\,0$")
                return 0
        # Zero column
        for j in range(m):
            if all(simplify(A[i, j]) == 0 for i in range(n)):
                self.step_generator.add_step(
                    0, rf"$第\,{j + 1}\,列全为\,0$")
                return 0

        # Identical or proportional rows
        for i in range(n):
            for j in range(i + 1, n):
                if all(simplify(A[i, k] - A[j, k]) == 0 for k in range(A.cols)):
                    self.step_generator.add_step(
                        0, rf"$第\,{i + 1}\,行\,=\,第\,{j + 1}\,行$")
                    return 0
                # Proportional (avoid division by zero)
                try:
                    ratios = []
                    valid = True
                    for k in range(A.cols):
                        if A[i, k] == 0 and A[j, k] == 0:
                            continue
                        if A[i, k] == 0:
                            valid = False
                            break
                        ratios.append(simplify(A[j, k] / A[i, k]))
                    if valid and ratios and all(r == ratios[0] for r in ratios):
                        self.step_generator.add_step(
                            0, rf"$第\,{j + 1}\,行\,=\,({latex(ratios[0])}) \\cdot 第\,{i + 1}\,行$")
                        return 0
                except Exception:
                    pass

        # Identical or proportional columns
        for i in range(m):
            for j in range(i + 1, m):
                if all(simplify(A[k, i] - A[k, j]) == 0 for k in range(A.rows)):
                    self.step_generator.add_step(
                        0, rf"$第\,{i + 1}\,列\,=\,第\,{j + 1}\,列$")
                    return 0
                # Proportional (avoid division by zero)
                try:
                    ratios = []
                    valid = True
                    for k in range(A.rows):
                        if A[k, i] == 0 and A[k, j] == 0:
                            continue
                        if A[k, i] == 0:
                            valid = False
                            break
                        ratios.append(simplify(A[k, j] / A[k, i]))
                    if valid and ratios and all(r == ratios[0] for r in ratios):
                        self.step_generator.add_step(
                            0, rf"$第\,{j + 1}\,列\,=\,({latex(ratios[0])}) \\cdot 第\,{i + 1}\,列$")
                        return 0
                except Exception:
                    pass

        # Identity / diagonal / triangular
        is_diag = all((i == j) or (A[i, j] == 0)
                      for i in range(n) for j in range(n))
        is_upper = all((i <= j) or (A[i, j] == 0)
                       for i in range(n) for j in range(n))
        is_lower = all((i >= j) or (A[i, j] == 0)
                       for i in range(n) for j in range(n))
        if is_diag or is_upper or is_lower:
            val = simplify(A.det())
            self.step_generator.add_step(
                val, rf"$|对角/三角矩阵|\,=\,对角线元素乘积\,=\,{latex(val)}$")
            return val

        return None

    # Row operations (Gaussian elimination method)
    def _method_gaussian_elimination(self, matrix: Matrix) -> Expr:
        """Calculate determinant using Gaussian elimination to convert to upper triangular form.

        Returns:
            The calculated determinant value
        """
        M = matrix.copy()
        rows = M.rows
        sign = 1

        for pivot_row in range(rows):
            pivot_elem = M[pivot_row, pivot_row]
            if pivot_elem == 0:
                found = False
                for swap_row in range(pivot_row + 1, rows):
                    if M[swap_row, pivot_row] != 0:
                        M.row_swap(pivot_row, swap_row)
                        self.step_generator.add_step("", "")
                        self.step_generator.add_step(
                            M.copy(), rf"$交换第\,{pivot_row + 1}\,行和\,{swap_row + 1}\,行(正负号取反一次)$")
                        sign *= -1
                        found = True
                        pivot_elem = M[pivot_row, pivot_row]
                        break
                if not found:
                    # No pivot available for this column, continue to next column
                    self.step_generator.add_step("", "")
                    self.step_generator.add_step(
                        M.copy(), rf"$第\,{pivot_row + 1}\,列无可用主元, 跳过(此时已可得出行列式的值为\,0)$")
                    continue

            # Assuming all symbolic expressions (denominators) are non-zero, continue elimination
            for r in range(pivot_row + 1, rows):
                if simplify(M[r, pivot_row]) == 0:
                    continue
                factor_ = simplify(M[r, pivot_row] / pivot_elem)
                # R_r <- R_r - factor * R_pivot
                M.row_op(r, lambda v, j, pr=pivot_row, factor=factor_: simplify(v - factor * M[pr, j]))
                self.step_generator.add_step("", "")

                if factor_ == 1:
                    self.step_generator.add_step(
                        M.copy(), f"$R_{{{r + 1}}} - R_{{{pivot_row + 1}}} \\to R_{{{r + 1}}}$")
                elif factor_ == -1:
                    self.step_generator.add_step(
                        M.copy(), f"$R_{{{r + 1}}} + R_{{{pivot_row + 1}}} \\to R_{{{r + 1}}}$")
                elif factor_ < 0:
                    self.step_generator.add_step(
                        M.copy(),
                        f"$R_{{{r + 1}}} + \\left({latex(-factor_)}\\right) \\cdot R_{{{pivot_row + 1}}}"
                        rf"\\to R_{{{r + 1}}}$")
                else:
                    self.step_generator.add_step(
                        M.copy(),
                        f"$R_{{{r + 1}}} - \\left({latex(factor_)}\\right) \\cdot R_{{{pivot_row + 1}}}"
                        rf"\\to R_{{{r + 1}}}$")

        # Elimination complete, calculate diagonal product multiplied by sign
        diag_prod = 1
        for i in range(rows):
            diag_prod *= M[i, i]
        det_val = simplify(sign * diag_prod)  # type: ignore
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            det_val, rf"$已化为上三角矩阵:\;|A|\,=\,{'' if sign == 1 else '-'}对角线元素乘积$")
        return det_val

    def compute_list(self, matrix_input: str) -> Tuple[List[Expr], List[str]]:
        """Compute determinant using multiple methods and return step-by-step results.

        Returns:
            tuple: Steps and explanations for each computation step
        """
        self._reset()
        A = self._parse_matrix_input(matrix_input)
        self.step_generator.add_step(A.copy(), "输入矩阵 $A$")
        method_count = 1

        # Properties method
        self._add_section_title(rf"$法\,{method_count}:\,特殊结构检测$")
        res = self._method_properties(A)
        if res is None:
            self.step_generator.add_step(f"$\\Rightarrow 无特殊结构$", "")
        method_count += 1

        # Laplace's expansion method (if matrix isn't too large)
        n = A.shape[0]
        if n <= 6:
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                rf"$法\,{method_count}:\;$拉普拉斯展开法", "")
            if n > 3:
                self.step_generator.add_step("", "")
                self.step_generator.add_step(
                    rf"$3 \times 3\,行列式不显示展开的余子式, 直接出结果, 以减少中间过程$")
            self.step_generator.add_step("", "")
            self._method_laplace(A)
            method_count += 1
        else:
            self.step_generator.add_step(
                "", rf"矩阵维度较大, 展开过于复杂, 跳过完整$\,$拉普拉斯展开(可自行启用)")

        # LU/PLU decomposition
        self._add_section_title(rf"$法\,{method_count}:\;$LU/PLU$\,$分解法")
        self._method_lu(A)
        method_count += 1

        # Diagonal rule (3x3)
        if n == 3:
            self._add_section_title(rf"$法\,{method_count}:\;$对角线法则(3×3)")
            self._method_sarrus(A)
            method_count += 1

        # Row operations method (Gaussian elimination)
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            rf"$法\,{method_count}:\,行变换法(高斯消元法)$", rf"$假设所有符号表达式不为\,0$")
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            rf"在计算行列式时, 即使分类讨论最后也会统一成该假设下的结果", rf"$所以此处并不需要进行是否为\,0\,的讨论$")
        self._method_gaussian_elimination(A)
        method_count += 1

        steps, explanations = self.step_generator.get_steps()
        return steps, explanations

    def compute_latex(self, matrix_input: str) -> str:
        """Compute the determinant and return result in LaTeX format.

        Returns:
            str: LaTeX formatted solution steps
        """
        steps, explanations = self.compute_list(matrix_input)
        return self.step_generator.get_latex(steps, explanations)
