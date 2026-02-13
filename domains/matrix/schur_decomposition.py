from typing import Dict, List, Tuple

from sympy import Expr, Matrix, eye, latex, simplify, sqrt, symbols, zeros

from core import CommonMatrixCalculator


# from IPython.display import Math, display


class SchurDecomposition(CommonMatrixCalculator):
    """A class for performing Schur decomposition on matrices.

    This class provides various methods to compute the Schur decomposition of different
    types of matrices including general, Hermitian, and normal matrices.
    """

    @staticmethod
    def is_square(matrix: Matrix) -> bool:
        """Check if the matrix is square."""
        return matrix.rows == matrix.cols

    @staticmethod
    def is_hermitian(matrix: Matrix) -> bool:
        """Check if the matrix is Hermitian.

        A matrix is Hermitian if it equals its conjugate transpose.
        """
        return matrix == matrix.conjugate().transpose()

    @staticmethod
    def is_normal(matrix: Matrix) -> bool:
        """Check if the matrix is normal.

        A matrix is normal if it commutes with its conjugate transpose.
        """
        A_H = matrix.conjugate().transpose()
        return matrix * A_H == A_H * matrix

    def compute_eigenvalues(self, matrix: Matrix, show_steps: bool = True) -> Dict:
        """Compute eigenvalues with detailed steps.

        Parameters
        ----------
        matrix : Matrix
            The matrix to compute eigenvalues for.
        show_steps : bool, optional
            Whether to show computation steps (default is True).

        Returns
        -------
        dict
            Dictionary with eigenvalues as keys, and their multiplicities as values.
        """
        if show_steps:
            self.add_step("计算特征多项式")

        # Calculate characteristic polynomial
        n = matrix.rows
        lambda_symbol = symbols('lambda')
        char_poly_matrix = matrix - lambda_symbol * eye(n)

        if show_steps:
            self.add_matrix(char_poly_matrix, "A - \\lambda I")

        char_poly = char_poly_matrix.det()

        if show_steps:
            self.step_generator.add_step(
                f"\\text{{特征多项式: }} \\det(A - \\lambda I) = {latex(char_poly)}")
            self.step_generator.add_step(
                f"\\text{{化简: }} {latex(simplify(char_poly))}")

        # Solve for eigenvalues
        if show_steps:
            self.add_step("求解特征方程")

        eigenvalues = matrix.eigenvals()

        if show_steps:
            eigen_eq = " = 0".join([f"({latex(simplify(char_poly))})"])
            self.step_generator.add_step(f"\\text{{特征方程: }} {eigen_eq}")

            for eigenval, multiplicity in eigenvalues.items():
                self.step_generator.add_step(
                    f"\\lambda = {latex(eigenval)}, \\quad \\text{{重数: }} {multiplicity}")

        return eigenvalues

    def compute_eigenvector(self, matrix: Matrix, eigenvalue: Expr, show_steps: bool = True) -> List[Matrix]:
        """Compute eigenvectors with detailed steps.

        Parameters
        ----------
        matrix : Matrix
            The matrix to compute eigenvectors for.
        eigenvalue : scalar
            The eigenvalue to compute corresponding eigenvectors.
        show_steps : bool, optional
            Whether to show computation steps (default is True)."""
        if show_steps:
            self.add_step(f"计算特征值 {latex(eigenvalue)} 对应的特征向量")

        n = matrix.rows
        # Construct eigen system
        eigen_system = matrix - eigenvalue * eye(n)

        if show_steps:
            self.add_matrix(eigen_system, f"A - {latex(eigenvalue)}I")

        # Solve null space
        nullspace = eigen_system.nullspace()

        if show_steps:
            if nullspace:
                self.step_generator.add_step(
                    f"\\text{{找到 {len(nullspace)} 个线性无关的特征向量}}")
                for i, vec in enumerate(nullspace):
                    self.add_vector(vec, f"\\boldsymbol{{v_{i + 1}}}")
            else:
                self.step_generator.add_step(f"\\text{{警告: 未找到特征向量}}")

        return nullspace

    def gram_schmidt(self, vectors: List[Matrix], show_steps: bool = True) -> List[Matrix]:
        """Perform the Gram-Schmidt orthogonalization process.

        Parameters
        ----------
        vectors : list
            List of vectors to orthogonalize.
        show_steps : bool, optional
            Whether to show computation steps (default is True)."""
        if show_steps:
            self.add_step("Gram-Schmidt 正交化")

        orthogonal_vectors = []

        for i, v in enumerate(vectors):
            if show_steps:
                self.add_step(f"处理向量 $\\boldsymbol{{v_{i + 1}}}$")
                self.add_vector(v, f"\\boldsymbol{{v_{i + 1}^{{(0)}}}}")

            # Start orthogonalization
            u = v.copy()

            for j in range(i):
                # Calculate projection
                projection = (v.dot(orthogonal_vectors[j]) /
                              orthogonal_vectors[j].dot(orthogonal_vectors[j])) * orthogonal_vectors[j]

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{减去到 }} \\boldsymbol{{u_{j + 1}}} \\text{{ 的投影: }} " +
                        f"\\frac{{\\boldsymbol{{v_{{{i + 1}}}^{{(0)}}}} "
                        f"\\cdot \\boldsymbol{{u_{{{j + 1}}}}}}}{{\\boldsymbol{{u_{{{j + 1}}}}} "
                        f"\\cdot \\boldsymbol{{u_{{{j + 1}}}}}}} \\boldsymbol{{u_{{{j + 1}}}}} = " +
                        f"\\frac{{{latex(v.dot(orthogonal_vectors[j]))}}}"
                        f"{{{latex(orthogonal_vectors[j].dot(orthogonal_vectors[j]))}}} "
                        f"{latex(orthogonal_vectors[j])} = {latex(projection)}"
                    )

                u = u - projection

                if show_steps:
                    self.add_vector(
                        u, f"\\boldsymbol{{v_{{{i + 1}}}^{{({j + 1})}}}}")

            # Normalize
            norm = sqrt(u.dot(u))
            if norm != 0:
                u_normalized = u / norm

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{归一化: }} \\boldsymbol{{u_{i + 1}}} = \\frac{{\\boldsymbol{{v_{i + 1}^"
                        f"{{({i})}}}}}}{{\\sqrt{{\\boldsymbol{{v_{i + 1}^{{({i})}}}} \\cdot "
                        f"\\boldsymbol{{v_{i + 1}^{{({i})}}}}}}}} = " +
                        f"\\frac{{{latex(u)}}}{{{latex(norm)}}} = {latex(u_normalized)}"
                    )
            else:
                if show_steps:
                    self.step_generator.add_step(f"\\text{{零向量, 跳过}}")
                continue

            orthogonal_vectors.append(u_normalized)

        return orthogonal_vectors

    def schur_decomposition_iterative(self, matrix_input: str | Matrix, show_steps: bool = True,
                                      max_iterations: int = 100) -> Tuple[Matrix, Matrix]:
        """Iterative method for Schur decomposition (applicable to general matrices).

        Parameters
        ----------
        matrix_input : Matrix or str
            The matrix to decompose.
        show_steps : bool, optional
            Whether to show computation steps (default is True).
        max_iterations : int, optional
            Maximum number of iterations (default is 100)."""
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError(
                "Schur decomposition only applies to square matrices")

        n = A.rows

        if show_steps:
            self.step_generator.add_step(r"\text{Schur 分解 - 迭代法}")
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")

        # Initialize with A and identity matrix
        T = A.copy()
        Q = eye(n)

        if show_steps:
            self.add_step("初始化")
            self.add_matrix(Q, "Q_0")
            self.add_matrix(T, "T_0")

        for iteration in range(min(n - 1, max_iterations)):
            if show_steps:
                self.add_step(f"迭代 {iteration + 1}")

            # Select bottom-right submatrix for QR decomposition
            submatrix_size = n - iteration
            sub_T = T[iteration:, iteration:]

            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{处理子矩阵(大小 ${submatrix_size} \\times {submatrix_size})$}}")
                self.add_matrix(
                    sub_T, f"T_{iteration}[{iteration}:, {iteration}:]")

            # QR decomposition
            Q_sub, R_sub = sub_T.QRdecomposition()

            if show_steps:
                self.add_step("子矩阵的 QR 分解")
                self.add_matrix(Q_sub, f"Q_{iteration}^{{sub}}")
                self.add_matrix(R_sub, f"R_{iteration}^{{sub}}")

                # Verify QR decomposition
                Q_times_R = Q_sub * R_sub
                self.add_matrix(
                    Q_times_R, f"Q_{iteration}^{{sub}} R_{iteration}^{{sub}}")
                if Q_times_R == sub_T:
                    self.step_generator.add_step("\\text{QR 分解验证正确}")
                else:
                    self.step_generator.add_step("\\text{QR 分解验证错误}")

            # Update T matrix
            # Construct full Q rotation matrix
            Q_full = eye(n)
            for i in range(submatrix_size):
                for j in range(submatrix_size):
                    Q_full[iteration + i, iteration + j] = Q_sub[i, j]

            # Update: T = Q^H * T * Q
            T_new = Q_full.transpose().conjugate() * T * Q_full

            if show_steps:
                self.add_step("更新 T 矩阵")
                self.add_equation(
                    f"T_{iteration + 1} = Q_{iteration}^H T_{iteration} Q_{iteration}")
                self.add_matrix(T_new, f"T_{iteration + 1}")

            # Update accumulated Q matrix
            Q = Q * Q_full

            if show_steps:
                self.add_step("更新 Q 矩阵")
                self.add_equation(
                    f"Q_{iteration + 1} = Q_{iteration} Q_{iteration}^{{rot}}")
                self.add_matrix(Q, f"Q_{iteration + 1}")

            T = T_new

            # Check convergence (whether lower triangular part approaches 0)
            convergence = True
            for i in range(iteration + 2, n):
                for j in range(iteration + 1):
                    if abs(T[i, j]) > 1e-10:  # Convergence threshold
                        convergence = False
                        break
                if not convergence:
                    break

            if convergence and show_steps:
                self.step_generator.add_step(
                    f"\\text{{在第 {iteration + 1} 次迭代后收敛}}")
                break

        # Final verification
        if show_steps:
            self.add_step("最终结果验证")
            self.add_matrix(Q, "Q")
            self.add_matrix(T, "T")

            Q_H = Q.transpose().conjugate()
            Q_T_Q = Q_H * Q
            reconstruction = Q * T * Q_H

            self.add_step("正交性验证")
            self.add_matrix(Q_T_Q, "Q^H Q")
            if Q_T_Q == eye(n):
                self.step_generator.add_step("\\text{Q 是酉矩阵}")

            self.add_step("重构验证")
            self.add_matrix(reconstruction, "Q T Q^H")
            self.add_matrix(A, "A")

            if reconstruction == A:
                self.step_generator.add_step("\\text{Schur 分解正确}")

        return Q, T

    def schur_decomposition_direct(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix]:
        """Direct method for Schur decomposition (applicable to diagonalizable matrices)."""
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError(
                "Schur decomposition only applies to square matrices")

        n = A.rows

        if show_steps:
            self.step_generator.add_step("Schur 分解 - 直接法")
            self.add_matrix(A, "A")

        # Step 1: Compute eigenvalues and eigenvectors
        eigenvalues = self.compute_eigenvalues(A, show_steps)

        if show_steps:
            self.add_step("计算所有特征向量")

        all_eigenvectors = []
        for eigenval in eigenvalues.keys():
            eigenvectors = self.compute_eigenvector(A, eigenval, show_steps)
            all_eigenvectors.extend(eigenvectors)

        # Check if we have enough eigenvectors
        if len(all_eigenvectors) < n:
            if show_steps:
                self.step_generator.add_step("\\text{警告: 矩阵不可对角化, 使用迭代法}")
            return self.schur_decomposition_iterative(A, show_steps)

        # Step 2: Orthogonalize eigenvectors
        orthogonal_basis = self.gram_schmidt(all_eigenvectors, show_steps)

        if len(orthogonal_basis) < n:
            if show_steps:
                self.step_generator.add_step("\\text{警告: 无法找到完整的正交基, 使用迭代法}")
            return self.schur_decomposition_iterative(A, show_steps)

        # Construct Q matrix
        Q = zeros(n)
        for i in range(n):
            for j in range(n):
                Q[j, i] = orthogonal_basis[i][j]

        if show_steps:
            self.add_step("构造酉矩阵 Q")
            self.add_matrix(Q, "Q")

        # Compute T matrix: T = Q^H A Q
        Q_H = Q.transpose().conjugate()
        T = Q_H * A * Q

        if show_steps:
            self.add_step("计算上三角矩阵 T")
            self.add_equation("T = Q^H A Q")
            self.add_matrix(T, "T")

        # Verification
        if show_steps:
            self.add_step("验证")
            reconstruction = Q * T * Q_H
            self.add_matrix(reconstruction, "Q T Q^H")
            self.add_matrix(A, "A")

            if reconstruction == A:
                self.step_generator.add_step("\\text{Schur分解正确}")
            else:
                self.step_generator.add_step("\\text{分解存在误差}")

        return Q, T

    def schur_decomposition_hermitian(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix]:
        """Special handling for Hermitian matrices (results in the diagonal matrix)."""

        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if not self.is_hermitian(A):
            raise ValueError("This method only applies to Hermitian matrices")

        if show_steps:
            self.step_generator.add_step(r"\text{Hermitian 矩阵的 Schur 分解}")
            self.add_matrix(A, "A")
            self.step_generator.add_step(
                "\\text{Hermitia 矩阵的 Schur 分解就是特征值分解, T 是对角矩阵}")

        n = A.rows

        # Compute eigenvalues and eigenvectors
        eigenvalues = self.compute_eigenvalues(A, show_steps)

        all_eigenvectors = []
        for eigenval in eigenvalues.keys():
            eigenvectors = self.compute_eigenvector(A, eigenval, show_steps)
            all_eigenvectors.extend(eigenvectors)

        # Orthogonalize
        orthogonal_basis = self.gram_schmidt(all_eigenvectors, show_steps)

        # Construct Q and T
        Q = zeros(n)
        T = zeros(n)

        for i in range(n):
            for j in range(n):
                Q[j, i] = orthogonal_basis[i][j]
            T[i, i] = list(eigenvalues.keys())[i]  # Eigenvalues on diagonal

        if show_steps:
            self.add_matrix(Q, "Q")
            self.add_matrix(T, "T")

        return Q, T

    def auto_schur_decomposition(self, matrix_input: str, show_steps: bool = True) -> Tuple[Matrix, Matrix]:
        """Automatically select the appropriate Schur decomposition method."""

        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step("自动 Schur 分解")
            self.add_matrix(A, "A")

        # Check the matrix type
        is_hermitian = self.is_hermitian(A)
        is_normal = self.is_normal(A)

        if show_steps:
            self.step_generator.add_step(
                f"\\text{{Hermitian 矩阵: }} {'是' if is_hermitian else '否'}")
            self.step_generator.add_step(
                f"\\text{{正规矩阵: }} {'是' if is_normal else '否'}")

        if is_hermitian:
            if show_steps:
                self.step_generator.add_step(
                    "\\text{检测到 Hermitian 矩阵, 使用特殊方法}")
            return self.schur_decomposition_hermitian(matrix_input, show_steps)
        if is_normal:
            if show_steps:
                self.step_generator.add_step("\\text{检测到正规矩阵, 尝试直接法}")
            try:
                return self.schur_decomposition_direct(matrix_input, show_steps)
            except Exception:
                if show_steps:
                    self.step_generator.add_step("\\text{直接法失败, 使用迭代法}")
                return self.schur_decomposition_iterative(matrix_input, show_steps)

        if show_steps:
            self.step_generator.add_step("\\text{一般矩阵, 使用迭代法}")
        return self.schur_decomposition_iterative(matrix_input, show_steps)

# def demo_schur():
#     """Demonstrate Schur decomposition on various matrices."""

#     schur = SchurDecomposition()

#     # Example matrix
#     schur.step_generator.add_step(r"\text{Hermitian 矩阵示例}")
#     A_hermitian = '[[2,1],[1,2]]'
#     schur.auto_schur_decomposition(A_hermitian)
#     display(Math(schur.get_steps_latex()))

#     schur.step_generator.add_step("\\" + "\\")

#     schur.step_generator.add_step(r"\text{正规矩阵示例}")
#     A_normal = '[[0,-1],[1,0]]'  # Rotation matrix
#     schur.auto_schur_decomposition(A_normal)
#     display(Math(schur.get_steps_latex()))

#     schur.step_generator.add_step("\\" + "\\")

#     schur.step_generator.add_step(r"\text{一般矩阵示例}")
#     A_general = '[[2,1,0],[0,2,1],[0,0,3]]'
#     schur.auto_schur_decomposition(A_general)
#     display(Math(schur.get_steps_latex()))

#     schur.step_generator.add_step("\\" + "\\")


# def demo_special_cases():
#     """Demonstrate special cases of Schur decomposition."""

#     schur = SchurDecomposition()

#     schur.step_generator.add_step(r"\text{对角矩阵}")
#     A_diag = '[[1,0,0],[0,2,0],[0,0,3]]'
#     schur.auto_schur_decomposition(A_diag)
#     display(Math(schur.get_steps_latex()))

#     schur.step_generator.add_step("\\" + "\\")

#     schur.step_generator.add_step(r"\text{三角矩阵}")
#     A_tri = '[[1,2,3],[0,4,5],[0,0,6]]'
#     schur.auto_schur_decomposition(A_tri)
#     display(Math(schur.get_steps_latex()))

#     schur.step_generator.add_step("\\" + "\\")

#     schur.step_generator.add_step(r"\text{复数矩阵}")
#     A_complex = '[[1,I],[-I,1]]'
#     schur.auto_schur_decomposition(A_complex)
#     display(Math(schur.get_steps_latex()))

#     schur.step_generator.add_step("\\" + "\\")


# def demo_convergence():
#     """Demonstrate convergence properties."""

#     schur = SchurDecomposition()

#     schur.step_generator.add_step(r"\text{收敛性测试}")
#     A_slow = '[[2,1,1],[1,3,1],[1,1,4]]'
#     _, T = schur.schur_decomposition_iterative(A_slow)

#     schur.step_generator.add_step(r"\text{最终上三角矩阵 T: }")
#     schur.add_matrix(T, "T")

#     # Check upper triangular property
#     n = T.rows
#     is_upper_triangular = True
#     for i in range(n):
#         for j in range(i):
#             if abs(T[i, j]) > 1e-10:
#                 is_upper_triangular = False
#                 break
#         if not is_upper_triangular:
#             break

#     if is_upper_triangular:
#         schur.step_generator.add_step(r"\text{成功收敛到上三角矩阵}")
#     else:
#         schur.step_generator.add_step(r"\text{未完全收敛到上三角矩阵}")

#     display(Math(schur.get_steps_latex()))


# if __name__ == "__main__":
#     # Run demonstrations
#     demo_schur()
#     demo_special_cases()
#     demo_convergence()
