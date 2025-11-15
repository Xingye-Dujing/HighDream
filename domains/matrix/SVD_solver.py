from sympy import Matrix, latex, zeros, eye, simplify, symbols, solve, sqrt, sin, cos, Symbol, I
from IPython.display import display, Math

from core import CommonMatrixCalculator


class SVDSolver(CommonMatrixCalculator):

    def compute_eigenpairs(self, matrix):
        """
        计算实对称矩阵的特征值和特征向量
        返回排序后的(特征值, 特征向量)列表
        """
        try:
            # 对于实对称矩阵，使用更稳定的方法
            if matrix.rows != matrix.cols:
                raise ValueError("矩阵必须是方阵")

            # 检查矩阵是否对称
            if simplify(matrix - matrix.T) != zeros(matrix.rows, matrix.cols):
                raise ValueError("矩阵必须是对称的")

            # 使用 sympy 的特征值分解
            eigenpairs = []

            # 计算特征多项式
            lambda_sym = symbols('lambda')
            char_poly = (matrix - lambda_sym * eye(matrix.rows)).det()
            eigenvalues = solve(char_poly, lambda_sym)

            # 对每个特征值计算特征空间
            for eig in eigenvalues:
                # 计算零空间(特征空间)
                eig_matrix = matrix - eig * eye(matrix.rows)
                nullspace = eig_matrix.nullspace()

                # 对每个特征向量进行单位化
                for vec in nullspace:
                    if vec.norm() != 0:
                        unit_vec = vec / vec.norm()
                        eigenpairs.append((eig, unit_vec))

            # 按特征值大小排序(降序), 但如果包含符号则按添加顺序
            try:
                eigenpairs.sort(key=lambda x: x[0], reverse=True)
            except TypeError:
                # 如果不能比较(例如符号表达式), 则按原先顺序
                self.step_generator.add_step(
                    r"\textbf{警告: 无法对含符号的特征值进行准确的大小排序, 以下假设在前的大于在后的}")

            return eigenpairs

        except Exception as e:
            raise ValueError(f"特征值计算错误: {str(e)}") from e

    def gram_schmidt(self, vectors):
        """
        Gram-Schmidt 正交化过程
        """
        ortho_vectors = []
        for v in vectors:
            w = v.copy()
            for u in ortho_vectors:
                # 检查分母是否为零
                u_norm_sq = u.dot(u)
                if u_norm_sq != 0:
                    projection = (v.dot(u) / u_norm_sq) * u
                    w -= projection
            # 检查向量是否为零向量
            if w.norm() != 0:
                ortho_vectors.append(w / w.norm())
        return ortho_vectors

    def complete_orthogonal_basis(self, existing_vectors, target_dim):
        """
        补充正交基到目标维度
        """
        current_dim = len(existing_vectors)
        if current_dim >= target_dim:
            return existing_vectors

        # 创建标准基向量
        standard_basis = [zeros(target_dim, 1) for _ in range(target_dim)]
        for i in range(target_dim):
            standard_basis[i][i] = 1

        # 使用改进的 Gram-Schmidt 过程
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

    def compute_svd(self, matrix_input, show_steps=True):
        """
        计算矩阵的奇异值分解
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

        # 步骤1: 计算 A^T A
        if show_steps:
            self.add_step("计算 $A^T A$")

        A_T = A.T
        if show_steps:
            self.add_matrix(A_T, "A^T")

        ATA = A_T * A
        if show_steps:
            self.add_matrix(ATA, "A^T A")
            self.add_equation(r"\text{注意: } A^TA \text{ 是实对称矩阵，特征值都是实数}")

        # 步骤2: 计算 A^T A 的特征值和特征向量
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

        # 步骤3: 计算奇异值
        if show_steps:
            self.add_step("计算奇异值")
            self.add_equation(r"\text{奇异值 } \sigma_i = \sqrt{\lambda_i}")

        singular_values = []
        for eig, _ in eigenpairs:
            if eig.has(Symbol):
                self.step_generator.add_step(f"\\textbf{{特征值含符号, 无法再进行操作}}")
                return None, None, None
            if eig >= 0:  # 只取非负特征值的平方根
                sigma = sqrt(eig)
                singular_values.append(sigma)
                if show_steps:
                    self.step_generator.add_step(
                        f"\\sigma_{{{len(singular_values)}}} = \\sqrt{{{latex(eig)}}} = {latex(sigma)}")
            else:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{忽略负特征值: }} {latex(eig)}")

        # 奇异值已经按从大到小排序
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

        # 步骤5: 计算 V 矩阵(右奇异向量)
        if show_steps:
            self.add_step("计算 V 矩阵")
            self.add_equation(r"\text{V 的列向量是 } A^TA \text{ 的特征向量}")

        # 构造 V 矩阵
        V = zeros(n, n)
        for i, (_, vec) in enumerate(eigenpairs):
            if i < n:
                for j in range(n):
                    V[j, i] = vec[j]

        # 如果特征向量数量不足, 补充正交基
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

        # 步骤6: 计算 U 矩阵(左奇异向量)
        if show_steps:
            self.add_step("计算 U 矩阵")
            self.add_equation(
                r"\text{U 的列向量通过 } u_i = \frac{1}{\sigma_i} A v_i \text{ 计算}")

        U = zeros(m, m)

        # 计算非零奇异值对应的左奇异向量
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

        # 如果 U 的列向量不足，补充正交基
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

        # 步骤7: 验证分解
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

        # 显示总结
        if show_steps:
            self.display_svd_summary(A, U, Sigma, V, singular_values)

        return U, Sigma, V

    def display_svd_summary(self, A, U, Sigma, V, singular_values):
        """
        显示 SVD 分解的总结
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

    def compute_singular_values_only(self, matrix_input, show_steps=True):
        """
        仅计算奇异值
        """
        self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{奇异值计算}")
            self.add_matrix(A, "A")

        # 计算 A^T A
        ATA = A.T * A

        if show_steps:
            self.add_step("计算 $A^T A$")
            self.add_matrix(ATA, "A^T A")

        # 计算特征值
        lambda_sym = symbols('lambda')
        char_poly = (ATA - lambda_sym * eye(ATA.rows)).det()
        eigenvalues = solve(char_poly, lambda_sym)

        if show_steps:
            self.add_step("计算 $A^T A$ 的特征值")
            eig_list = rf',\;'.join([latex(eig) for eig in eigenvalues])
            self.step_generator.add_step(f"\\text{{特征值: }} {eig_list}")

        # 计算奇异值
        singular_values = []
        for eig in eigenvalues:
            if eig.has(I):  # 复数特征值
                sigma = abs(eig)
                singular_values.append(sigma)
            elif eig >= 0:  # 非负实数特征值
                sigma = sqrt(eig)
                singular_values.append(sigma)
        if show_steps:
            self.add_step("计算奇异值")
            sigma_list = rf',\;'.join([f'\\sigma_{{{i+1}}} = {latex(sigma)}'
                                       for i, sigma in enumerate(singular_values)])
            self.step_generator.add_step(f"\\text{{奇异值: }} {sigma_list}")

        return singular_values


def demo_svd_basic():
    """演示基本的 SVD 计算"""
    svd_solver = SVDSolver()

    svd_solver.step_generator.add_step(r"\textbf{基本 SVD 计算演示}")

    matrices = [
        ("2×2 矩阵", "[[3,1],[1,3]]"),
        ("3×2 矩阵", "[[1,0],[2,0],[3,0]]"),
        ("2×3 矩阵", "[[-1,1,0],[0,-1,1]]"),
        ("对称矩阵", "[[2,1],[1,2]]"),
        ("正交矩阵", "[[1,0],[0,1]]")
    ]

    for name, matrix in matrices:
        svd_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            svd_solver.compute_svd(matrix)
            display(Math(svd_solver.get_steps_latex()))
        except Exception as e:
            svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(svd_solver.get_steps_latex()))
        svd_solver.step_generator.add_step("\\" * 2)


def demo_singular_values_only():
    """演示仅计算奇异值"""
    svd_solver = SVDSolver()

    svd_solver.step_generator.add_step(r"\textbf{仅计算奇异值演示}")

    matrices = [
        ("简单矩阵", "[[1,0],[0,2]]"),
        ("全 1 矩阵", "[[1,1],[1,1]]"),
        ("秩 1 矩阵", "[[1,2],[2,4]]"),
        ("零矩阵", "[[0,0],[0,0]]")
    ]

    for name, matrix in matrices:
        svd_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            singular_values = svd_solver.compute_singular_values_only(matrix)
            svd_solver.step_generator.add_step(
                f"\\text{{奇异值: }} {', '.join([latex(s) for s in singular_values])}")
            display(Math(svd_solver.get_steps_latex()))
        except Exception as e:
            svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(svd_solver.get_steps_latex()))
        svd_solver.step_generator.add_step("\\" * 2)


def demo_svd_applications():
    """演示 SVD 的应用"""
    svd_solver = SVDSolver()

    svd_solver.step_generator.add_step(r"\textbf{SVD 应用演示}")

    matrices = [
        ("图像压缩示例", "[[255,255,0,0],[255,255,0,0],[0,0,128,128],[0,0,128,128]]"),
        ("数据矩阵", "[[1,2,1],[2,4,2],[1,2,1]]"),
    ]

    for name, matrix in matrices:
        svd_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            U, Sigma, V = svd_solver.compute_svd(matrix)

            # 显示低秩近似
            svd_solver.step_generator.add_step(r"\text{低秩近似演示}")
            singular_values = [Sigma[i, i] for i in range(
                min(Sigma.rows, Sigma.cols)) if Sigma[i, i] != 0]
            if len(singular_values) > 1:
                # 使用前 k 个奇异值进行近似
                k = len(singular_values) - 1
                Sigma_approx = zeros(Sigma.rows, Sigma.cols)
                for i in range(k):
                    Sigma_approx[i, i] = singular_values[i]

                A_approx = U * Sigma_approx * V.T
                svd_solver.step_generator.add_step(
                    f"\\text{{使用前 {k} 个奇异值的近似}}")
                svd_solver.add_matrix(A_approx, "A_{approx}")

            display(Math(svd_solver.get_steps_latex()))

        except Exception as e:
            svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(svd_solver.get_steps_latex()))
        svd_solver.step_generator.add_step("\\" * 2)


def demo_zero_singular_values():
    """演示零奇异值的情况"""
    svd_solver = SVDSolver()

    svd_solver.step_generator.add_step(r"\textbf{零奇异值情况演示}")

    # 测试包含零奇异值的矩阵
    matrices = [
        ("秩亏矩阵", "[[1,1,1],[1,1,1],[1,1,1]]"),
        ("零矩阵", "[[0,0,0],[0,0,0]]"),
        ("线性相关列", "[[1,2,3],[2,4,6],[3,6,9]]"),
        ("不满秩矩阵", "[[1,0,0],[0,0,0],[0,0,0]]")
    ]

    for name, matrix in matrices:
        svd_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            _, Sigma, _ = svd_solver.compute_svd(matrix)
            # 显示奇异值
            singular_values = [Sigma[i, i]
                               for i in range(min(Sigma.rows, Sigma.cols))]
            zero_sv_count = sum(1 for sv in singular_values if sv == 0)
            svd_solver.step_generator.add_step(
                f"\\text{{零奇异值数量: }} {zero_sv_count}")
            display(Math(svd_solver.get_steps_latex()))
        except Exception as e:
            svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(svd_solver.get_steps_latex()))
        svd_solver.step_generator.add_step("\\" * 2)


def demo_symbolic_svd():
    """演示包含符号元素的奇异值分解"""
    svd_solver = SVDSolver()

    display(Math(r"\textbf{符号奇异值分解演示}"))
    display(Math(
        r"\textbf{基本不能用, 仅能分解很特殊的情况:}"))
    display(Math(
        r"\text{A 的元素含符号, 但 A 的奇异值($A^T A$ 的特征值) 为数值时可用}"))

    display(Math(r"\textbf{例: 示例特殊在正弦余弦平方和为 1, 奇异值为数值}"))

    theta = symbols('theta')
    rotation_like = Matrix([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])

    try:
        svd_solver.compute_svd(rotation_like)
        display(Math(svd_solver.get_steps_latex()))
    except Exception as e:
        svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(svd_solver.get_steps_latex()))

    svd_solver.step_generator.add_step("\\" + "\\")

    theta = symbols('theta')
    rotation_like = Matrix([
        [3*cos(theta/2), -3*sin(theta/2)],
        [3*sin(theta/2), 3*cos(theta/2)]
    ])

    try:
        svd_solver.compute_svd(rotation_like)
        display(Math(svd_solver.get_steps_latex()))
    except Exception as e:
        svd_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(svd_solver.get_steps_latex()))


def demo_hard_svd():
    svd_solver = SVDSolver()
    display(Math(r"\textbf{困难奇异值分解演示}"))
    demo_hard_matrix = '[[1, 2, 3],[4, 5, 6],[7, 8, 9]]'
    svd_solver.compute_svd(demo_hard_matrix)
    display(Math(svd_solver.get_steps_latex()))


if __name__ == "__main__":
    demo_svd_basic()
    demo_singular_values_only()
    demo_svd_applications()
    demo_zero_singular_values()
    demo_symbolic_svd()
    demo_hard_svd()
