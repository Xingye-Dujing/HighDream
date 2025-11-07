from sympy import Matrix, sympify, latex, zeros, eye, symbols, simplify, sqrt
from IPython.display import display, Math
from domains.matrix import CommonStepGenerator


class SchurDecomposition:

    def __init__(self):
        self.step_generator = CommonStepGenerator()

    def add_step(self, title):
        """显示步骤标题"""
        self.step_generator.add_step(
            f"\\text{{{title}}}")

    def add_matrix(self, matrix, name="M"):
        """显示矩阵"""
        self.step_generator.add_step(f"{name} = {latex(matrix)}")

    def add_vector(self, vector, name="v"):
        """显示向量"""
        self.step_generator.add_step(f"{name} = {latex(vector)}")

    def add_equation(self, equation):
        """显示方程"""
        self.step_generator.add_step(equation)

    def get_steps_latex(self):
        return self.step_generator.get_steps_latex()

    def parse_matrix_input(self, matrix_input):
        """解析矩阵输入"""
        try:
            if isinstance(matrix_input, str):
                matrix = Matrix(sympify(matrix_input))
            else:
                matrix = matrix_input
            return matrix
        except:
            raise ValueError(f"无法解析矩阵输入: {matrix_input}")

    def is_square(self, matrix):
        """检查是否为方阵"""
        return matrix.rows == matrix.cols

    def is_hermitian(self, matrix):
        """检查矩阵是否为Hermitian矩阵"""
        return matrix == matrix.conjugate().transpose()

    def is_normal(self, matrix):
        """检查矩阵是否为正规矩阵"""
        A_H = matrix.conjugate().transpose()
        return matrix * A_H == A_H * matrix

    def compute_eigenvalues(self, matrix, show_steps=True):
        """计算特征值并显示详细过程"""
        if show_steps:
            self.add_step("计算特征多项式")

        # 计算特征多项式
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

        # 求解特征值
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

    def compute_eigenvector(self, matrix, eigenvalue, show_steps=True):
        """计算特征向量并显示详细过程"""
        if show_steps:
            self.add_step(f"计算特征值 {latex(eigenvalue)} 对应的特征向量")

        n = matrix.rows
        # 构造特征系统
        eigen_system = matrix - eigenvalue * eye(n)

        if show_steps:
            self.add_matrix(eigen_system, f"A - {latex(eigenvalue)}I")

        # 求解零空间
        nullspace = eigen_system.nullspace()

        if show_steps:
            if nullspace:
                self.step_generator.add_step(
                    f"\\text{{找到 {len(nullspace)} 个线性无关的特征向量}}")
                for i, vec in enumerate(nullspace):
                    self.add_vector(vec, f"\\boldsymbol{{v_{i+1}}}")
            else:
                self.step_generator.add_step(f"\\text{{警告: 未找到特征向量}}")

        return nullspace

    def gram_schmidt(self, vectors, show_steps=True):
        """Gram-Schmidt 正交化过程"""
        if show_steps:
            self.add_step("Gram-Schmidt 正交化")

        orthogonal_vectors = []

        for i, v in enumerate(vectors):
            if show_steps:
                self.add_step(f"处理向量 $\\boldsymbol{{v_{i+1}}}$")
                self.add_vector(v, f"\\boldsymbol{{v_{i+1}^{{(0)}}}}")

            # 开始正交化
            u = v.copy()

            for j in range(i):
                # 计算投影
                projection = (v.dot(orthogonal_vectors[j]) /
                              orthogonal_vectors[j].dot(orthogonal_vectors[j])) * orthogonal_vectors[j]

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{减去到 }} \\boldsymbol{{u_{j+1}}} \\text{{ 的投影: }} " +
                        f"\\frac{{\\boldsymbol{{v_{{{i+1}}}^{{(0)}}}} \\cdot \\boldsymbol{{u_{{{j+1}}}}}}}{{\\boldsymbol{{u_{{{j+1}}}}} \\cdot \\boldsymbol{{u_{{{j+1}}}}}}} \\boldsymbol{{u_{{{j+1}}}}} = " +
                        f"\\frac{{{latex(v.dot(orthogonal_vectors[j]))}}}{{{latex(orthogonal_vectors[j].dot(orthogonal_vectors[j]))}}} {latex(orthogonal_vectors[j])} = {latex(projection)}"
                    )

                u = u - projection

                if show_steps:
                    self.add_vector(
                        u, f"\\boldsymbol{{v_{{{i+1}}}^{{({j+1})}}}}")

            # 归一化
            norm = sqrt(u.dot(u))
            if norm != 0:
                u_normalized = u / norm

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{归一化: }} \\boldsymbol{{u_{i+1}}} = \\frac{{\\boldsymbol{{v_{i+1}^{{({i})}}}}}}{{\\sqrt{{\\boldsymbol{{v_{i+1}^{{({i})}}}} \\cdot \\boldsymbol{{v_{i+1}^{{({i})}}}}}}}} = " +
                        f"\\frac{{{latex(u)}}}{{{latex(norm)}}} = {latex(u_normalized)}"
                    )
            else:
                u_normalized = u
                if show_steps:
                    self.step_generator.add_step(f"\\text{{零向量, 跳过}}")
                continue

            orthogonal_vectors.append(u_normalized)

        return orthogonal_vectors

    def schur_decomposition_iterative(self, matrix_input, show_steps=True, max_iterations=100):
        """迭代法进行 Schur 分解(适用于一般矩阵)"""
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("Schur 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.step_generator.add_step(r"\text{Schur 分解 - 迭代法}")
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")

        # 初始化为A和单位矩阵
        T = A.copy()
        Q = eye(n)

        if show_steps:
            self.add_step("初始化")
            self.add_matrix(Q, "Q_0")
            self.add_matrix(T, "T_0")

        for iteration in range(min(n-1, max_iterations)):
            if show_steps:
                self.add_step(f"迭代 {iteration + 1}")

            # 选择右下角子矩阵进行 QR 分解
            submatrix_size = n - iteration
            sub_T = T[iteration:, iteration:]

            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{处理子矩阵(大小 ${submatrix_size} \\times {submatrix_size})$}}")
                self.add_matrix(
                    sub_T, f"T_{iteration}[{iteration}:, {iteration}:]")

            # QR 分解
            Q_sub, R_sub = sub_T.QRdecomposition()

            if show_steps:
                self.add_step("子矩阵的 QR 分解")
                self.add_matrix(Q_sub, f"Q_{iteration}^{{sub}}")
                self.add_matrix(R_sub, f"R_{iteration}^{{sub}}")

                # 验证 QR 分解
                Q_times_R = Q_sub * R_sub
                self.add_matrix(
                    Q_times_R, f"Q_{iteration}^{{sub}} R_{iteration}^{{sub}}")
                if Q_times_R == sub_T:
                    self.step_generator.add_step("\\text{QR 分解验证正确}")
                else:
                    self.step_generator.add_step("\\text{QR 分解验证错误}")

            # 更新 T 矩阵
            # 构造完整的 Q 旋转矩阵
            Q_full = eye(n)
            for i in range(submatrix_size):
                for j in range(submatrix_size):
                    Q_full[iteration+i, iteration+j] = Q_sub[i, j]

            # 更新: T = Q^H * T * Q
            T_new = Q_full.transpose().conjugate() * T * Q_full

            if show_steps:
                self.add_step("更新 T 矩阵")
                self.add_equation(
                    f"T_{iteration+1} = Q_{iteration}^H T_{iteration} Q_{iteration}")
                self.add_matrix(T_new, f"T_{iteration+1}")

            # 更新累积的 Q 矩阵
            Q = Q * Q_full

            if show_steps:
                self.add_step("更新 Q 矩阵")
                self.add_equation(
                    f"Q_{iteration+1} = Q_{iteration} Q_{iteration}^{{rot}}")
                self.add_matrix(Q, f"Q_{iteration+1}")

            T = T_new

            # 检查收敛性(下三角部分是否接近 0)
            convergence = True
            for i in range(iteration+2, n):
                for j in range(iteration+1):
                    if abs(T[i, j]) > 1e-10:  # 收敛阈值
                        convergence = False
                        break
                if not convergence:
                    break

            if convergence and show_steps:
                self.step_generator.add_step(
                    f"\\text{{在第 {iteration+1} 次迭代后收敛}}")
                break

        # 最终验证
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

    def schur_decomposition_direct(self, matrix_input, show_steps=True):
        """直接法进行 Schur 分解(适用于可对角化矩阵)"""
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("Schur 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.step_generator.add_step("Schur 分解 - 直接法")
            self.add_matrix(A, "A")

        # 步骤1: 计算特征值和特征向量
        eigenvalues = self.compute_eigenvalues(A, show_steps)

        if show_steps:
            self.add_step("计算所有特征向量")

        all_eigenvectors = []
        for eigenval in eigenvalues.keys():
            eigenvectors = self.compute_eigenvector(A, eigenval, show_steps)
            all_eigenvectors.extend(eigenvectors)

        # 检查是否有足够的特征向量
        if len(all_eigenvectors) < n:
            if show_steps:
                self.step_generator.add_step("\\text{警告: 矩阵不可对角化, 使用迭代法}")
            return self.schur_decomposition_iterative(A, show_steps)

        # 步骤2: 正交化特征向量
        orthogonal_basis = self.gram_schmidt(all_eigenvectors, show_steps)

        if len(orthogonal_basis) < n:
            if show_steps:
                self.step_generator.add_step("\\text{警告: 无法找到完整的正交基, 使用迭代法}")
            return self.schur_decomposition_iterative(A, show_steps)

        # 构造Q矩阵
        Q = zeros(n)
        for i in range(n):
            for j in range(n):
                Q[j, i] = orthogonal_basis[i][j]

        if show_steps:
            self.add_step("构造酉矩阵 Q")
            self.add_matrix(Q, "Q")

        # 计算T矩阵: T = Q^H A Q
        Q_H = Q.transpose().conjugate()
        T = Q_H * A * Q

        if show_steps:
            self.add_step("计算上三角矩阵 T")
            self.add_equation("T = Q^H A Q")
            self.add_matrix(T, "T")

        # 验证
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

    def schur_decomposition_hermitian(self, matrix_input, show_steps=True):
        """Hermitian矩阵的特殊处理(得到对角矩阵)"""
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if not self.is_hermitian(A):
            raise ValueError("此方法只适用于 Hermitian 矩阵")

        if show_steps:
            self.step_generator.add_step(r"\text{Hermitian 矩阵的 Schur 分解}")
            self.add_matrix(A, "A")
            self.step_generator.add_step(
                "\\text{Hermitia 矩阵的 Schur 分解就是特征值分解, T 是对角矩阵}")

        n = A.rows

        # 计算特征值和特征向量
        eigenvalues = self.compute_eigenvalues(A, show_steps)

        all_eigenvectors = []
        for eigenval in eigenvalues.keys():
            eigenvectors = self.compute_eigenvector(A, eigenval, show_steps)
            all_eigenvectors.extend(eigenvectors)

        # 正交化
        orthogonal_basis = self.gram_schmidt(all_eigenvectors, show_steps)

        # 构造 Q 和 T
        Q = zeros(n)
        T = zeros(n)

        for i in range(n):
            for j in range(n):
                Q[j, i] = orthogonal_basis[i][j]
            T[i, i] = list(eigenvalues.keys())[i]  # 特征值在对角线上

        if show_steps:
            self.add_matrix(Q, "Q")
            self.add_matrix(T, "T")

        return Q, T

    def auto_schur_decomposition(self, matrix_input, show_steps=True):
        """自动选择 Schur 分解方法"""
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step("自动 Schur 分解")
            self.add_matrix(A, "A")

        # 检查矩阵类型
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
        elif is_normal:
            if show_steps:
                self.step_generator.add_step("\\text{检测到正规矩阵, 尝试直接法}")
            try:
                return self.schur_decomposition_direct(matrix_input, show_steps)
            except:
                if show_steps:
                    self.step_generator.add_step("\\text{直接法失败, 使用迭代法}")
                return self.schur_decomposition_iterative(matrix_input, show_steps)
        else:
            if show_steps:
                self.step_generator.add_step("\\text{一般矩阵, 使用迭代法}")
            return self.schur_decomposition_iterative(matrix_input, show_steps)


def demo_schur():
    """演示 Schur 分解"""
    schur = SchurDecomposition()

    # 示例矩阵
    schur.step_generator.add_step(r"\text{Hermitian 矩阵示例}")
    A_hermitian = '[[2,1],[1,2]]'
    schur.auto_schur_decomposition(A_hermitian)
    display(Math(schur.get_steps_latex()))

    schur.step_generator.add_step("\\" + "\\")

    schur.step_generator.add_step(r"\text{正规矩阵示例}")
    A_normal = '[[0,-1],[1,0]]'  # 旋转矩阵
    schur.auto_schur_decomposition(A_normal)
    display(Math(schur.get_steps_latex()))

    schur.step_generator.add_step("\\" + "\\")

    schur.step_generator.add_step(r"\text{一般矩阵示例}")
    A_general = '[[2,1,0],[0,2,1],[0,0,3]]'
    schur.auto_schur_decomposition(A_general)
    display(Math(schur.get_steps_latex()))

    schur.step_generator.add_step("\\" + "\\")


def demo_special_cases():
    """演示特殊情况"""
    schur = SchurDecomposition()

    schur.step_generator.add_step(r"\text{对角矩阵}")
    A_diag = '[[1,0,0],[0,2,0],[0,0,3]]'
    schur.auto_schur_decomposition(A_diag)
    display(Math(schur.get_steps_latex()))

    schur.step_generator.add_step("\\" + "\\")

    schur.step_generator.add_step(r"\text{三角矩阵}")
    A_tri = '[[1,2,3],[0,4,5],[0,0,6]]'
    schur.auto_schur_decomposition(A_tri)
    display(Math(schur.get_steps_latex()))

    schur.step_generator.add_step("\\" + "\\")

    schur.step_generator.add_step(r"\text{复数矩阵}")
    A_complex = '[[1,I],[-I,1]]'
    schur.auto_schur_decomposition(A_complex)
    display(Math(schur.get_steps_latex()))

    schur.step_generator.add_step("\\" + "\\")


def demo_convergence():
    """演示收敛性"""
    schur = SchurDecomposition()

    schur.step_generator.add_step(r"\text{收敛性测试}")
    A_slow = '[[2,1,1],[1,3,1],[1,1,4]]'
    Q, T = schur.schur_decomposition_iterative(A_slow)

    schur.step_generator.add_step(r"\text{最终上三角矩阵 T: }")
    schur.add_matrix(T, "T")

    # 检查上三角性质
    n = T.rows
    is_upper_triangular = True
    for i in range(n):
        for j in range(i):
            if abs(T[i, j]) > 1e-10:
                is_upper_triangular = False
                break
        if not is_upper_triangular:
            break

    if is_upper_triangular:
        schur.step_generator.add_step(r"\text{成功收敛到上三角矩阵}")
    else:
        schur.step_generator.add_step(r"\text{未完全收敛到上三角矩阵}")

    display(Math(schur.get_steps_latex()))


if __name__ == "__main__":
    # 运行演示
    demo_schur()
    demo_special_cases()
    demo_convergence()
