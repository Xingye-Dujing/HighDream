from typing import List, Tuple
from sympy import Expr, I, Matrix, Symbol, eye, latex, simplify, solve, sqrt, symbols, zeros
# from sympy sin, cos
# from IPython.display import Math, display

from core import CommonMatrixCalculator


class SVDSolver(CommonMatrixCalculator):
    """A class for computing Singular Value Decomposition (SVD) of matrices.

    This class provides methods to compute the SVD of a matrix, which decomposes
    a matrix A into the product U * Sigma * V^T, where U and V are orthogonal
    matrices and Sigma is a diagonal matrix containing the singular values.
    """

    def compute_eigenpairs(self, matrix: Matrix) -> List[Tuple[Expr, Matrix]]:
        """Compute eigenvalues and eigenvectors of a real symmetric matrix.

        Returns sorted eigenpairs (eigenvalue, eigenvector) list.

        Parameters
        ----------
        matrix : sympy.Matrix
            The input matrix (must be square and symmetric)

        Returns
        -------
        list
            List of tuples (eigenvalue, normalized eigenvector)

        Raises
        ------
        ValueError
            If matrix is not square or not symmetric
        """
        try:
            # For real symmetric matrices, use more stable method
            if matrix.rows != matrix.cols:
                raise ValueError("Matrix must be square")

            # Check if matrix is symmetric
            if simplify(matrix - matrix.T) != zeros(matrix.rows, matrix.cols):
                raise ValueError("Matrix must be symmetric")

            # Use SymPy's eigenvalue decomposition
            eigenpairs = []

            # Calculate characteristic polynomial
            lambda_sym = symbols('lambda')
            char_poly = (matrix - lambda_sym * eye(matrix.rows)).det()
            eigenvalues = solve(char_poly, lambda_sym)

            # Compute eigenspace for each eigenvalue
            for eig in eigenvalues:
                # Compute null space (eigenspace)
                eig_matrix = matrix - eig * eye(matrix.rows)
                nullspace = eig_matrix.nullspace()

                # Normalize each eigenvector
                for vec in nullspace:
                    if vec.norm() != 0:
                        unit_vec = vec / vec.norm()
                        eigenpairs.append((eig, unit_vec))

            # Sort by eigenvalue magnitude (descending), but keep original order for symbolic values
            try:
                eigenpairs.sort(key=lambda x: x[0], reverse=True)
            except TypeError:
                # Cannot compare if contains symbolic expressions
                self.step_generator.add_step(
                    r"\textbf{Warning: Cannot accurately sort eigenvalues with symbols, assuming earlier ones are larger}")

            return eigenpairs

        except Exception as e:
            raise ValueError(f"Eigenvalue computation error: {str(e)}") from e

    def gram_schmidt(self, vectors: List[Matrix]) -> List[Matrix]:
        """Perform Gram-Schmidt orthogonalization process.

        Parameters
        ----------
        vectors : list
            List of vectors to orthogonalize

        Returns
        -------
        list
            List of orthonormalized vectors
        """
        ortho_vectors = []
        for v in vectors:
            w = v.copy()
            for u in ortho_vectors:
                # Check for zero denominator
                u_norm_sq = u.dot(u)
                if u_norm_sq != 0:
                    projection = (v.dot(u) / u_norm_sq) * u
                    w -= projection
            # Check for zero vector
            if w.norm() != 0:
                ortho_vectors.append(w / w.norm())
        return ortho_vectors

    def complete_orthogonal_basis(self, existing_vectors: List[Matrix], target_dim: int) -> List[Matrix]:
        """Complete an orthogonal basis to reach target dimension.

        Parameters
        ----------
        existing_vectors : list
            List of existing orthogonal vectors
        target_dim : int
            Target dimension for the completed basis

        Returns
        -------
        list
            List of orthogonal vectors spanning the full space
        """
        current_dim = len(existing_vectors)
        if current_dim >= target_dim:
            return existing_vectors

        # Create standard basis vectors
        standard_basis = [zeros(target_dim, 1) for _ in range(target_dim)]
        for i in range(target_dim):
            standard_basis[i][i] = 1

        # Use improved Gram-Schmidt process
        ortho_vectors = existing_vectors.copy()

        for basis_vec in standard_basis:
            if len(ortho_vectors) >= target_dim:
                break

            w = basis_vec.copy()
            for u in ortho_vectors:
                u_norm_sq = u.dot(u)
                if u_norm_sq != 0:
                    projection = (w.dot(u) / u_norm_sq) * u
                    w -= projection

            w_norm = w.norm()
            if w_norm != 0:
                ortho_vectors.append(w / w_norm)

        return ortho_vectors

    def compute_svd(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix, Matrix]:
        """Compute the Singular Value Decomposition of a matrix.

        Computes the decomposition A = U * Sigma * V^T where U and V are
        orthogonal matrices and Sigma is a diagonal matrix of singular values.

        Parameters
        ----------
        matrix_input : str or sympy.Matrix
            Input matrix as string representation or sympy Matrix
        show_steps : bool, optional
            Whether to show detailed computation steps (default is True)

        Returns
        -------
        tuple
            Tuple of (U, Sigma, V) matrices representing the SVD decomposition,
            or (None, None, None) if computation fails
        """
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{奇异值分解求解器}")
            self.add_matrix(A, "A")
            self.add_equation(r"\text{原理: } A = U \Sigma V^T")
            self.add_equation(r"\text{其中 } \Sigma \text{ 是对角矩阵，对角线元素是奇异值}")
            self.add_equation(r"\text{奇异值是 } A^TA \text{ 的特征值的平方根}")

        m, n = A.rows, A.cols

        # Step 1: Compute A^T A
        if show_steps:
            self.add_step("计算 $A^T A$")

        A_T = A.T
        if show_steps:
            self.add_matrix(A_T, "A^T")

        ATA = A_T * A
        if show_steps:
            self.add_matrix(ATA, "A^T A")
            self.add_equation(r"\text{注意: } A^TA \text{ 是实对称矩阵，特征值都是实数}")

        # Step 2: Compute eigenvalues and eigenvectors of A^T A
        if show_steps:
            self.add_step("计算 $A^T A$ 的特征值和特征向量")

        try:
            eigenpairs = self.compute_eigenpairs(ATA)

            if show_steps:
                eig_list = rf',\;'.join([latex(eig) for eig, _ in eigenpairs])
                self.step_generator.add_step(rf"A^T A\;的特征值:\;{eig_list}")

                for i, (eig, vec) in enumerate(eigenpairs):
                    self.step_generator.add_step(
                        rf"\text{{特征值 }} {latex(eig)}: \; \boldsymbol{{v}}_{{{i+1}}} = {latex(vec)}")

        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{特征值计算错误: }} {str(e)}")
            return None, None, None

        # Step 3: Compute singular values
        if show_steps:
            self.add_step("计算奇异值")
            self.add_equation(r"\text{奇异值 } \sigma_i = \sqrt{\lambda_i}")

        singular_values = []
        for eig, _ in eigenpairs:
            if eig.has(Symbol):
                self.step_generator.add_step(f"\\textbf{{特征值含符号, 无法再进行操作}}")
                return None, None, None
            if eig >= 0:  # Only take square root of non-negative eigenvalues
                sigma = sqrt(eig)
                singular_values.append(sigma)
                if show_steps:
                    self.step_generator.add_step(
                        f"\\sigma_{{{len(singular_values)}}} = \\sqrt{{{latex(eig)}}} = {latex(sigma)}")
            else:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{忽略负特征值: }} {latex(eig)}")

        # Singular values are already sorted in descending order
        if show_steps:
            sorted_sigmas = rf',\;'.join([f'\\sigma_{{{i+1}}} = {latex(sigma)}'
                                          for i, sigma in enumerate(singular_values)])
            self.step_generator.add_step(
                f"\\text{{排序后的奇异值: }} {sorted_sigmas}")

        if show_steps:
            self.add_step("构造 $\\Sigma$ 矩阵")

        Sigma = zeros(m, n)
        min_dim = min(m, n)
        for i in range(min_dim):
            if i < len(singular_values):
                Sigma[i, i] = singular_values[i]

        if show_steps:
            self.add_matrix(Sigma, "\\Sigma")
            self.add_equation(r"\Sigma \text{ 是对角矩阵，对角线元素是奇异值}")

        # Step 5: Compute V matrix (right singular vectors)
        if show_steps:
            self.add_step("计算 V 矩阵")
            self.add_equation(r"\text{V 的列向量是 } A^TA \text{ 的特征向量}")

        # Construct V matrix
        V = zeros(n, n)
        for i, (_, vec) in enumerate(eigenpairs):
            if i < n:
                for j in range(n):
                    V[j, i] = vec[j]

        # If insufficient eigenvectors, complete orthogonal basis
        if len(eigenpairs) < n:
            existing_vectors = [V[:, i] for i in range(len(eigenpairs))]
            complete_basis = self.complete_orthogonal_basis(
                existing_vectors, n)
            for i in range(len(eigenpairs), n):
                for j in range(n):
                    V[j, i] = complete_basis[i][j]

        if show_steps:
            self.add_matrix(V, "V")
            self.add_equation(r"\text{V 是正交矩阵: } V^T V = I")

        # Step 6: Compute U matrix (left singular vectors)
        if show_steps:
            self.add_step("计算 U 矩阵")
            self.add_equation(
                r"\text{U 的列向量通过 } u_i = \frac{1}{\sigma_i} A v_i \text{ 计算}")

        U = zeros(m, m)

        # Compute left singular vectors for non-zero singular values
        computed_u_vectors = []
        for i, sigma in enumerate(singular_values):
            if i < min(m, n) and sigma != 0:
                v_i = V[:, i]
                u_i = (1/sigma) * A * v_i
                for j in range(m):
                    U[j, i] = u_i[j]
                computed_u_vectors.append(u_i)
            elif sigma == 0 and show_steps:
                self.step_generator.add_step(
                    f"\\text{{跳过零奇异值 }} \\sigma_{{{i+1}}}")

        # If U columns are insufficient, complete orthogonal basis
        if len(computed_u_vectors) < m:
            if show_steps:
                self.step_generator.add_step(r"\text{补充 U 的正交基}")

            complete_basis = self.complete_orthogonal_basis(
                computed_u_vectors, m)

            for i in range(len(computed_u_vectors), m):
                u_vec = complete_basis[i]
                for j in range(m):
                    U[j, i] = u_vec[j]

        if show_steps:
            self.add_matrix(U, "U")
            self.add_equation(r"\text{U 是正交矩阵: } U^T U = I")

        # Step 7: Verify decomposition
        if show_steps:
            self.add_step("验证分解")
            self.add_equation(r"\text{验证: } A = U \Sigma V^T")

        reconstructed_A = U * Sigma * V.T
        if show_steps:
            self.add_matrix(reconstructed_A, "U \\Sigma V^T")

        diff = simplify(A - reconstructed_A)
        if all(diff[i, j] == 0 for i in range(m) for j in range(n)):
            if show_steps:
                self.step_generator.add_step(r"\text{验证成功: } A = U \Sigma V^T")
        else:
            if show_steps:
                self.step_generator.add_step(r"\text{验证失败}")
                self.step_generator.add_step(f"\\text{{误差矩阵: }}{latex(diff)}")

        # Display summary
        if show_steps:
            self.display_svd_summary(A, U, Sigma, V, singular_values)

        return U, Sigma, V

    def display_svd_summary(self, A: Matrix, U: Matrix, Sigma: Matrix, V: Matrix, singular_values):
        """Display a summary of the SVD decomposition.

        Parameters
        ----------
        A (Matrix): Original matrix
        U (Matrix): Left singular vectors matrix
        Sigma (Matrix): Singular values matrix
        V (Matrix): Right singular vectors matrix
        singular_values (List): List of computed singular values
        """
        self.add_step("奇异值分解总结")

        self.step_generator.add_step(r"\textbf{原矩阵:}")
        self.add_matrix(A, "A")

        self.step_generator.add_step(r"\textbf{左奇异向量矩阵:}")
        self.add_matrix(U, "U")

        self.step_generator.add_step(r"\textbf{奇异值矩阵:}")
        self.add_matrix(Sigma, "\\Sigma")

        self.step_generator.add_step(r"\textbf{右奇异向量矩阵:}")
        self.add_matrix(V, "V")

        self.step_generator.add_step(r"\textbf{奇异值:}")
        for i, sigma in enumerate(singular_values):
            self.step_generator.add_step(f"\\sigma_{{{i+1}}} = {latex(sigma)}")

        self.step_generator.add_step(r"\textbf{验证:}")
        self.add_equation(r"A = U \Sigma V^T")
        reconstructed = U * Sigma * V.T
        if simplify(A - reconstructed) == zeros(A.rows, A.cols):
            self.step_generator.add_step(r"\text{分解正确}")
        else:
            self.step_generator.add_step(r"\textbf{分解有误}")

    def compute_singular_values_only(self, matrix_input: str, show_steps: bool = True) -> List[Expr]:
        """Compute only the singular values without full SVD.

        Parameters
        ----------
        matrix_input : str or sympy.Matrix
            Input matrix as string representation or sympy Matrix
        show_steps : bool, optional
            Whether to show detailed computation steps (default is True)

        Returns
        -------
        list
            List of singular values
        """
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{奇异值计算}")
            self.add_matrix(A, "A")

        # Compute A^T A
        ATA = A.T * A

        if show_steps:
            self.add_step("计算 $A^T A$")
            self.add_matrix(ATA, "A^T A")

        # Compute eigenvalues
        lambda_sym = symbols('lambda')
        char_poly = (ATA - lambda_sym * eye(ATA.rows)).det()
        eigenvalues = solve(char_poly, lambda_sym)

        if show_steps:
            self.add_step("计算 $A^T A$ 的特征值")
            eig_list = rf',\;'.join([latex(eig) for eig in eigenvalues])
            self.step_generator.add_step(f"\\text{{特征值: }} {eig_list}")

        # Compute singular values
        singular_values = []
        for eig in eigenvalues:
            if eig.has(I):  # Complex eigenvalue
                sigma = abs(eig)
                singular_values.append(sigma)
            elif eig >= 0:  # Non-negative real eigenvalue
                sigma = sqrt(eig)
                singular_values.append(sigma)
        if show_steps:
            self.add_step("计算奇异值")
            sigma_list = rf',\;'.join([f'\\sigma_{{{i+1}}} = {latex(sigma)}'
                                       for i, sigma in enumerate(singular_values)])
            self.step_generator.add_step(f"\\text{{奇异值: }} {sigma_list}")

        return singular_values


# def demo_svd_basic():
#     """Demonstrate basic SVD computation."""
#     svd_solver = SVDSolver()

#     svd_solver.step_generator.add_step(r"\textbf{基本 SVD 计算演示}")

#     matrices = [
#         ("2×2 矩阵", "[[3,1],[1,3]]"),
#         ("3×2 矩阵", "[[1,0],[2,0],[3,0]]"),
#         ("2×3 矩阵", "[[-1,1,0],[0,-1,1]]"),
#         ("对称矩阵", "[[2,1],[1,2]]"),
#         ("正交矩阵", "[[1,0],[0,1]]")
#     ]

#     for name, matrix in matrices:
#         svd_solver.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             svd_solver.compute_svd(matrix)
#             display(Math(svd_solver.get_steps_latex()))
#         except Exception as e:
#             svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(svd_solver.get_steps_latex()))
#         svd_solver.step_generator.add_step("\\" * 2)


# def demo_singular_values_only():
#     """Demonstrate computation of singular values only."""
#     svd_solver = SVDSolver()

#     svd_solver.step_generator.add_step(r"\textbf{仅计算奇异值演示}")

#     matrices = [
#         ("简单矩阵", "[[1,0],[0,2]]"),
#         ("全 1 矩阵", "[[1,1],[1,1]]"),
#         ("秩 1 矩阵", "[[1,2],[2,4]]"),
#         ("零矩阵", "[[0,0],[0,0]]")
#     ]

#     for name, matrix in matrices:
#         svd_solver.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             singular_values = svd_solver.compute_singular_values_only(matrix)
#             svd_solver.step_generator.add_step(
#                 f"\\text{{奇异值: }} {', '.join([latex(s) for s in singular_values])}")
#             display(Math(svd_solver.get_steps_latex()))
#         except Exception as e:
#             svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(svd_solver.get_steps_latex()))
#         svd_solver.step_generator.add_step("\\" * 2)


# def demo_svd_applications():
#     """Demonstrate applications of SVD."""
#     svd_solver = SVDSolver()

#     svd_solver.step_generator.add_step(r"\textbf{SVD 应用演示}")

#     matrices = [
#         ("图像压缩示例", "[[255,255,0,0],[255,255,0,0],[0,0,128,128],[0,0,128,128]]"),
#         ("数据矩阵", "[[1,2,1],[2,4,2],[1,2,1]]"),
#     ]

#     for name, matrix in matrices:
#         svd_solver.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             U, Sigma, V = svd_solver.compute_svd(matrix)

#             # Show low-rank approximation
#             svd_solver.step_generator.add_step(r"\text{低秩近似演示}")
#             singular_values = [Sigma[i, i] for i in range(
#                 min(Sigma.rows, Sigma.cols)) if Sigma[i, i] != 0]
#             if len(singular_values) > 1:
#                 # Approximate using first k singular values
#                 k = len(singular_values) - 1
#                 Sigma_approx = zeros(Sigma.rows, Sigma.cols)
#                 for i in range(k):
#                     Sigma_approx[i, i] = singular_values[i]

#                 A_approx = U * Sigma_approx * V.T
#                 svd_solver.step_generator.add_step(
#                     f"\\text{{使用前 {k} 个奇异值的近似}}")
#                 svd_solver.add_matrix(A_approx, "A_{approx}")

#             display(Math(svd_solver.get_steps_latex()))

#         except Exception as e:
#             svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(svd_solver.get_steps_latex()))
#         svd_solver.step_generator.add_step("\\" * 2)


# def demo_zero_singular_values():
#     """Demonstrate handling of zero singular values."""
#     svd_solver = SVDSolver()

#     svd_solver.step_generator.add_step(r"\textbf{零奇异值情况演示}")

#     # Test matrices with zero singular values
#     matrices = [
#         ("秩亏矩阵", "[[1,1,1],[1,1,1],[1,1,1]]"),
#         ("零矩阵", "[[0,0,0],[0,0,0]]"),
#         ("线性相关列", "[[1,2,3],[2,4,6],[3,6,9]]"),
#         ("不满秩矩阵", "[[1,0,0],[0,0,0],[0,0,0]]")
#     ]

#     for name, matrix in matrices:
#         svd_solver.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             _, Sigma, _ = svd_solver.compute_svd(matrix)
#             # Display singular values
#             singular_values = [Sigma[i, i]
#                                for i in range(min(Sigma.rows, Sigma.cols))]
#             zero_sv_count = sum(1 for sv in singular_values if sv == 0)
#             svd_solver.step_generator.add_step(
#                 f"\\text{{零奇异值数量: }} {zero_sv_count}")
#             display(Math(svd_solver.get_steps_latex()))
#         except Exception as e:
#             svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(svd_solver.get_steps_latex()))
#         svd_solver.step_generator.add_step("\\" * 2)


# def demo_symbolic_svd():
#     """Demonstrate SVD with symbolic elements."""
#     svd_solver = SVDSolver()

#     display(Math(r"\textbf{符号奇异值分解演示}"))
#     display(Math(
#         r"\textbf{基本不能用, 仅能分解很特殊的情况:}"))
#     display(Math(
#         r"\text{A 的元素含符号, 但 A 的奇异值($A^T A$ 的特征值) 为数值时可用}"))

#     display(Math(r"\textbf{例: 示例特殊在正弦余弦平方和为 1, 奇异值为数值}"))

#     theta = symbols('theta')
#     rotation_like = Matrix([
#         [cos(theta), -sin(theta)],
#         [sin(theta), cos(theta)]
#     ])

#     try:
#         svd_solver.compute_svd(rotation_like)
#         display(Math(svd_solver.get_steps_latex()))
#     except Exception as e:
#         svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#         display(Math(svd_solver.get_steps_latex()))

#     svd_solver.step_generator.add_step("\\" + "\\")

#     theta = symbols('theta')
#     rotation_like = Matrix([
#         [3*cos(theta/2), -3*sin(theta/2)],
#         [3*sin(theta/2), 3*cos(theta/2)]
#     ])

#     try:
#         svd_solver.compute_svd(rotation_like)
#         display(Math(svd_solver.get_steps_latex()))
#     except Exception as e:
#         svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#         display(Math(svd_solver.get_steps_latex()))


# def demo_hard_svd():
#     """Demonstrate SVD on a challenging matrix."""
#     svd_solver = SVDSolver()
#     display(Math(r"\textbf{困难奇异值分解演示}"))
#     demo_hard_matrix = '[[1, 2, 3],[4, 5, 6],[7, 8, 9]]'
#     svd_solver.compute_svd(demo_hard_matrix)
#     display(Math(svd_solver.get_steps_latex()))


# if __name__ == "__main__":
#     demo_svd_basic()
#     demo_singular_values_only()
#     demo_svd_applications()
#     demo_zero_singular_values()
#     demo_symbolic_svd()
#     demo_hard_svd()
