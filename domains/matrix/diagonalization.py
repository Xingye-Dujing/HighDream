from sympy import Matrix, latex, eye, symbols, simplify, diag, factor, I
from IPython.display import display, Math

from core import CommonMatrixCalculator


class Diagonalization(CommonMatrixCalculator):

    def is_square(self, matrix):
        """检查是否为方阵"""
        return matrix.rows == matrix.cols

    def check_diagonalizable_conditions(self, matrix_input, show_steps=True, is_clear=True):
        """
        检查矩阵是否满足对角化条件
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("对角化只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_step("检查矩阵是否为方阵")
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")
            self.step_generator.add_step(r"\text{矩阵是方阵，可以进行对角化分析}")

        # 计算特征多项式
        if show_steps:
            self.add_step("计算特征多项式")

        x = symbols('lambda')
        char_poly = A.charpoly(x)

        if show_steps:
            self.step_generator.add_step(
                f"\\text{{特征多项式: }} p(\\lambda) = {latex(char_poly.as_expr())}")
            self.step_generator.add_step(
                f"\\text{{特征多项式可写为: }} p(\\lambda) = {latex(factor(char_poly.as_expr()))}")

        # 计算特征值
        if show_steps:
            self.add_step("计算特征值")

        eigenvalues = A.eigenvals()

        if show_steps:
            eigen_display = []
            for eigenval, multiplicity in eigenvalues.items():
                eigen_display.append(
                    f"{latex(eigenval)} \\text{{ (重数: {multiplicity})}}")
            self.step_generator.add_step(
                f"\\text{{特征值: }} \\lambda = " + ", ".join(eigen_display))

        # 检查每个特征值的代数重数和几何重数
        if show_steps:
            self.add_step("检查每个特征值的代数重数和几何重数")

        diagonalizable = True
        conditions = []

        for eigenval, algebraic_multiplicity in eigenvalues.items():
            # 计算几何重数(特征空间的维数)
            eigen_space = A - eigenval * eye(n)
            geometric_multiplicity = n - eigen_space.rank()

            conditions.append({
                'eigenvalue': eigenval,
                'algebraic': algebraic_multiplicity,
                'geometric': geometric_multiplicity,
                'satisfied': algebraic_multiplicity == geometric_multiplicity
            })

            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{特征值 }} \\lambda = {latex(eigenval)}: "
                    f"\\text{{代数重数}} = {algebraic_multiplicity}, "
                    f"\\text{{几何重数}} = {geometric_multiplicity}"
                )

                if algebraic_multiplicity == geometric_multiplicity:
                    self.step_generator.add_step(
                        f"\\text{{满足条件: 代数重数 = 几何重数}}")
                else:
                    self.step_generator.add_step(
                        f"\\text{{不满足条件: 代数重数 $\\neq$ 几何重数}}")
                    diagonalizable = False

        # 检查特征向量总数
        total_geometric = sum(cond['geometric'] for cond in conditions)
        has_enough_eigenvectors = total_geometric == n

        if show_steps:
            self.add_step("检查线性无关特征向量的总数")
            self.step_generator.add_step(
                f"\\text{{线性无关特征向量的总数: }} {total_geometric}")
            self.step_generator.add_step(f"\\text{{矩阵的阶数: }} {n}")

            if has_enough_eigenvectors:
                self.step_generator.add_step(f"\\text{{足够多的线性无关特征向量}}")
            else:
                self.step_generator.add_step(f"\\text{{线性无关特征向量不足}}")
                diagonalizable = False

            if diagonalizable:
                self.step_generator.add_step(r"\textbf{矩阵可对角化}")
            else:
                self.step_generator.add_step(r"\textbf{矩阵不可对角化}")

        return diagonalizable, eigenvalues, conditions

    def compute_eigenvectors(self, matrix_input, show_steps=True, is_clear=True):
        """
        计算特征向量
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_step("计算特征向量")

        eigenvectors_dict = {}

        # 获取特征值和特征向量
        eigenvects = A.eigenvects()

        for eigenval, _, eigenvectors in eigenvects:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{特征值 }} \\lambda = {latex(eigenval)}:")

            eigenvectors_dict[eigenval] = []

            for i, eigenvector in enumerate(eigenvectors):
                # 规范化特征向量
                eigenvector_norm = eigenvector / eigenvector.norm()

                eigenvectors_dict[eigenval].append({
                    'raw': eigenvector,
                    'normalized': eigenvector_norm
                })

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{特征向量 {i+1}: }} \\boldsymbol{{v}}_{{{i+1}}} = {latex(eigenvector)}"
                    )
                    self.step_generator.add_step(
                        f"\\text{{单位特征向量: }} \\hat{{\\boldsymbol{{v}}}}_{{{i+1}}} = {latex(eigenvector_norm)}"
                    )

        return eigenvectors_dict

    def diagonalize_matrix(self, matrix_input, show_steps=True, normalize=True, is_clear=True):
        """
        对角化矩阵
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_step("矩阵对角化分解")
            self.add_matrix(A, "A")

        # 检查对角化条件
        diagonalizable, eigenvalues, _ = self.check_diagonalizable_conditions(
            matrix_input, show_steps, is_clear=False
        )

        if not diagonalizable:
            raise ValueError("矩阵不可对角化")

        # 计算特征向量
        eigenvectors_dict = self.compute_eigenvectors(
            matrix_input, show_steps, is_clear=False)

        if show_steps:
            self.add_step("构造变换矩阵 P 和对角矩阵 D")

        # 构造变换矩阵 P 和对角矩阵 D
        P_columns = []
        D_diag = []

        for eigenval in eigenvalues.keys():
            eigenvectors = eigenvectors_dict[eigenval]
            for eigenvec_info in eigenvectors:
                if normalize:
                    P_columns.append(eigenvec_info['normalized'])
                else:
                    P_columns.append(eigenvec_info['raw'])
                D_diag.append(eigenval)

        P = Matrix.hstack(*P_columns)
        D = diag(*D_diag)

        if show_steps:
            self.add_matrix(P, "P")
            self.add_matrix(D, "D")

            # 显示特征值在D中的排列
            self.add_step("对角矩阵 D 中特征值的排列")
            eigenval_positions = []
            pos = 1
            for eigenval, multiplicity in eigenvalues.items():
                for _ in range(multiplicity):
                    eigenval_positions.append(
                        f"D_{{{pos}{pos}}} = {latex(eigenval)}")
                    pos += 1
            self.step_generator.add_step(", ".join(eigenval_positions))

        # 计算 P 的逆矩阵
        if show_steps:
            self.add_step("计算变换矩阵的逆矩阵 $P^{-1}$")

        try:
            P_inv = P.inv()
            if show_steps:
                self.add_matrix(P_inv, "P^{-1}")
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(
                    r"\text{警告: 矩阵 P 不可逆, 这可能意味着特征向量线性相关}")
            raise ValueError("特征向量矩阵不可逆") from e

        # 验证分解
        if show_steps:
            self.add_step("验证对角化分解")
            self.add_equation(r"\text{验证: } A = P D P^{-1}")

            # 计算 P D P^{-1}
            PDP_inv = P * D * P_inv
            self.add_matrix(PDP_inv, "P D P^{-1}")
            self.add_matrix(A, "A")

            # 简化比较
            PDP_inv_simplified = simplify(PDP_inv)
            if PDP_inv_simplified == A:
                self.step_generator.add_step(r"\text{对角化分解正确}")
            else:
                self.step_generator.add_step(r"\textbf{验证失败, 对角化分解可能有误}")

        return P, D, P_inv

    def diagonalize_symmetric(self, matrix_input, show_steps=True, is_clear=True):
        """
        对称矩阵的特殊对角化(正交对角化)
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_step("对称矩阵的正交对角化")
            self.add_matrix(A, "A")

        # 检查对称性
        if A != A.T:
            if show_steps:
                self.step_generator.add_step(r"\text{警告: 矩阵不是对称矩阵, 但继续尝试对角化}")
            return self.diagonalize_matrix(matrix_input, show_steps, normalize=True, is_clear=False)

        if show_steps:
            self.step_generator.add_step(r"\text{矩阵是对称矩阵}")
            self.step_generator.add_step(r"\text{对称矩阵的性质:}")
            self.step_generator.add_step(r"\text{1. 所有特征值都是实数}")
            self.step_generator.add_step(r"\text{2. 存在正交矩阵 Q 使得 } A = Q D Q^T")

        # 进行标准对角化
        P, D, _ = self.diagonalize_matrix(
            matrix_input, show_steps, normalize=True, is_clear=False)

        # 对于对称矩阵, P 应该是正交矩阵
        Q = P
        Q_T = Q.T

        if show_steps:
            self.add_step("对称矩阵的正交对角化结果")
            self.add_matrix(Q, "Q")
            self.add_matrix(D, "D")
            self.add_matrix(Q_T, "Q^T")

            # 验证正交性
            QQT = Q * Q_T
            QTQ = Q_T * Q

            self.add_step("验证正交性")
            self.add_matrix(QQT, "Q Q^T")
            self.add_matrix(QTQ, "Q^T Q")

            if QQT == eye(A.rows) and QTQ == eye(A.rows):
                self.step_generator.add_step(r"\text{Q 是正交矩阵}")

            # 验证分解
            self.add_step("验证正交对角化")
            self.add_equation(r"\text{验证: } A = Q D Q^T")
            QDQ_T = Q * D * Q_T
            self.add_matrix(QDQ_T, "Q D Q^T")
            self.add_matrix(A, "A")

            if simplify(QDQ_T) == A:
                self.step_generator.add_step(r"\text{正交对角化正确}")

        return Q, D, Q_T

    def diagonalize_complex(self, matrix_input, show_steps=True, is_clear=True):
        """
        复矩阵的对角化
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_step("复矩阵对角化")
            self.add_matrix(A, "A")
            self.step_generator.add_step(r"\text{注意: 允许特征值和特征向量为复数}")

        # 检查是否有复数元素
        has_complex = any(val.has(I) for val in A)

        if not has_complex and show_steps:
            self.step_generator.add_step(r"\text{矩阵元素均为实数，但特征值/向量可能为复数}")

        return self.diagonalize_matrix(matrix_input, show_steps, normalize=True, is_clear=False)

    def auto_diagonalization(self, matrix_input, show_steps=True, is_clear=True):
        """
        自动选择对角化方法
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_step("自动对角化分析")
            self.add_matrix(A, "A")

        # 检查是否可对角化
        diagonalizable, eigenvalues, _ = self.check_diagonalizable_conditions(
            matrix_input, show_steps=False, is_clear=False
        )

        # 检查是否为对称矩阵
        is_symmetric = A == A.T

        # 检查是否有复数
        has_complex = any(val.has(I) for val in A) or any(
            eigenval.has(I) for eigenval in eigenvalues.keys()
        )

        if show_steps:
            method_info = []
            if diagonalizable:
                method_info.append("可对角化")
            if is_symmetric:
                method_info.append("对称矩阵")
            if has_complex:
                method_info.append("包含复数")
            self.step_generator.add_step(
                f"\\text{{矩阵特性: }} {', '.join(method_info)}")

        # 选择方法
        if diagonalizable:
            self.add_step("自动选择分解方法")
            if is_symmetric:
                if show_steps:
                    self.step_generator.add_step(r"\text{选择: 正交对角化}")
                return self.diagonalize_symmetric(matrix_input, show_steps, is_clear=False)
            elif has_complex:
                if show_steps:
                    self.step_generator.add_step(r"\text{选择: 复矩阵对角化}")
                return self.diagonalize_complex(matrix_input, show_steps, is_clear=False)
            else:
                if show_steps:
                    self.step_generator.add_step(r"\text{选择: 标准对角化}")
                return self.diagonalize_matrix(matrix_input, show_steps, is_clear=False)
        else:
            raise ValueError("矩阵不可对角化")


# 演示函数
def demo_diagonalization():
    """演示对角化分解"""
    diag_1 = Diagonalization()

    # 可对角化矩阵示例
    A1 = '[[4, -2], [1, 1]]'
    A2 = '[[2, 0, 0], [0, 3, 0], [0, 0, 1]]'  # 对角矩阵
    A3 = '[[1, 2, 0], [2, 1, 0], [0, 0, 3]]'  # 对称矩阵

    diag_1.step_generator.add_step(r"\textbf{可对角化矩阵演示}")

    cases = [
        ("一般可对角化矩阵", A1),
        ("对角矩阵", A2),
        ("对称矩阵", A3)
    ]

    for name, matrix in cases:
        diag_1.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            diag_1.auto_diagonalization(matrix)
            display(Math(diag_1.get_steps_latex()))
        except Exception as e:
            diag_1.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(diag_1.get_steps_latex()))


def demo_non_diagonalizable():
    """演示不可对角化矩阵"""
    diag_2 = Diagonalization()

    # 不可对角化矩阵示例
    A1 = '[[1, 1], [0, 1]]'
    A2 = '[[2, 1, 0], [0, 2, 1], [0, 0, 2]]'

    diag_2.step_generator.add_step(r"\textbf{不可对角化矩阵演示}")

    cases = [
        ("示例 1", A1),
        ("示例 2", A2)
    ]

    for name, matrix in cases:
        diag_2.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            diag_2.auto_diagonalization(matrix)
            display(Math(diag_2.get_steps_latex()))
        except Exception as e:
            diag_2.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(diag_2.get_steps_latex()))


def demo_complex():
    """演示复数矩阵对角化"""
    diag_3 = Diagonalization()

    # 复数矩阵示例
    A1 = '[[0, -1], [1, 0]]'  # 旋转矩阵
    A2 = '[[1, -2], [2, 1]]'  # 有复特征值

    diag_3.step_generator.add_step(r"\textbf{复数矩阵对角化演示}")

    cases = [
        ("旋转矩阵", A1),
        ("有复特征值的矩阵", A2)
    ]

    for name, matrix in cases:
        diag_3.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            diag_3.auto_diagonalization(matrix)
            display(Math(diag_3.get_steps_latex()))
        except Exception as e:
            diag_3.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(diag_3.get_steps_latex()))


def demo_special_cases():
    """演示特殊情况"""
    diag_4 = Diagonalization()

    # 特殊情况示例
    A1 = '[[1, 0, 0], [0, 1, 0], [0, 0, 1]]'  # 单位矩阵
    A2 = '[[2, 1], [0, 2]]'  # 不可对角化
    A3 = '[[3, 1, 0], [0, 3, 0], [0, 0, 4]]'  # 不可对角化

    diag_4.step_generator.add_step(r"\textbf{特殊情况演示}")

    cases = [
        ("单位矩阵", A1),
        ("2x2 不可对角化矩阵", A2),
        ("3x3 不可对角化矩阵", A3)
    ]

    for name, matrix in cases:
        diag_4.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            diag_4.auto_diagonalization(matrix)
            display(Math(diag_4.get_steps_latex()))
        except Exception as e:
            diag_4.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(diag_4.get_steps_latex()))


if __name__ == "__main__":
    # 运行各种演示
    demo_diagonalization()
    demo_non_diagonalizable()
    demo_complex()
    demo_special_cases()
