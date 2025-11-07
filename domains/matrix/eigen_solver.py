from collections import Counter
from sympy import Matrix, sympify, latex, zeros, eye, simplify, nsimplify, symbols, solve, factor, I
from IPython.display import display, Math

from domains.matrix import CommonStepGenerator


class EigenSolver:

    def __init__(self):
        self.step_counter = 0
        self.step_generator = CommonStepGenerator()

    def add_step(self, title):
        """显示步骤标题"""
        self.step_counter += 1
        self.step_generator.add_step(
            f"\\textbf{{步骤 {self.step_counter}: }} \\text{{{title}}}")

    def add_matrix(self, matrix, name="A"):
        """显示矩阵"""
        self.step_generator.add_step(f"{name} = {latex(matrix)}")

    def add_equation(self, equation):
        """显示方程"""
        self.step_generator.add_step(equation)

    def get_steps_latex(self):
        return self.step_generator.get_steps_latex()

    def simplify_matrix(self, matrix, method='auto'):
        """
        对矩阵的每个元素进行化简
        """
        if method == 'auto':
            has_symbols = any(any(element.free_symbols for element in row)
                              for row in matrix.tolist())
            method = 'simplify' if has_symbols else 'nsimplify'

        simplified_matrix = zeros(matrix.rows, matrix.cols)

        for i in range(matrix.rows):
            for j in range(matrix.cols):
                element = matrix[i, j]
                if method == 'simplify':
                    simplified_element = simplify(element)
                elif method == 'nsimplify':
                    simplified_element = nsimplify(element, rational=True)
                else:
                    simplified_element = element

                simplified_matrix[i, j] = simplified_element

        return simplified_matrix

    def parse_matrix_input(self, matrix_input):
        """解析矩阵输入"""
        try:
            if isinstance(matrix_input, str):
                matrix = Matrix(sympify(matrix_input))
            else:
                matrix = matrix_input
            return matrix
        except Exception as e:
            raise ValueError(f"无法解析矩阵输入: {matrix_input}, 错误: {str(e)}") from e

    def format_lambda_I(self, eigenvalue):
        """为复数或符号特征值添加括号"""
        has_complex = eigenvalue.has(I)

        is_symbolic = (isinstance(eigenvalue, sympify(
            'a').__class__) and eigenvalue.free_symbols)

        needs_parentheses = False
        if hasattr(eigenvalue, 'args') and len(eigenvalue.args) > 0:
            if any(op in str(eigenvalue) for op in ['+', '-', '*', '/']):
                needs_parentheses = True

        if not is_symbolic and not has_complex:
            if str(eigenvalue).startswith('-'):
                needs_parentheses = True

        if has_complex or is_symbolic or needs_parentheses:
            return f"({latex(eigenvalue)})I"
        else:
            return f"{latex(eigenvalue)}I"

    def analyze_matrix_structure(self, matrix, show_steps=True):
        """
        分析矩阵的特殊结构
        """
        n = matrix.rows
        special_types = []

        # 检查对角矩阵
        is_diagonal = True
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] != 0:
                    is_diagonal = False
                    break
            if not is_diagonal:
                break
        if is_diagonal:
            special_types.append("对角矩阵")

        # 检查上三角矩阵
        is_upper_triangular = True
        for i in range(1, n):
            for j in range(i):
                if matrix[i, j] != 0:
                    is_upper_triangular = False
                    break
            if not is_upper_triangular:
                break
        if is_upper_triangular:
            special_types.append("上三角矩阵")

        # 检查下三角矩阵
        is_lower_triangular = True
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i, j] != 0:
                    is_lower_triangular = False
                    break
            if not is_lower_triangular:
                break
        if is_lower_triangular:
            special_types.append("下三角矩阵")

        # 检查对称矩阵
        is_symmetric = True
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i, j] != matrix[j, i]:
                    is_symmetric = False
                    break
            if not is_symmetric:
                break
        if is_symmetric:
            special_types.append("对称矩阵")

        # 检查反对称矩阵
        is_skew_symmetric = True
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # 对角线元素应该为0
                    if matrix[i, i] != 0:
                        is_skew_symmetric = False
                        break
                else:
                    if matrix[i, j] + matrix[j, i] != 0:
                        is_skew_symmetric = False
                        break
            if not is_skew_symmetric:
                break
        if is_skew_symmetric:
            special_types.append("反对称矩阵")

        # 检查正交矩阵
        try:
            if n > 0:
                A_T = matrix.T
                product = matrix * A_T
                if simplify(product - eye(n)) == zeros(n, n):
                    special_types.append("正交矩阵")
        except:
            pass

        # 检查幂等矩阵
        try:
            if simplify(matrix * matrix - matrix) == zeros(n, n):
                special_types.append("幂等矩阵")
        except:
            pass

        # 检查幂零矩阵
        try:
            # 检查是否存在 k 使得A^k = 0
            is_nilpotent = False
            current_power = matrix
            for _ in range(1, n+1):
                if simplify(current_power) == zeros(n, n):
                    is_nilpotent = True
                    break
                current_power = current_power * matrix
            if is_nilpotent:
                special_types.append("幂零矩阵")
        except:
            pass

        if show_steps and special_types:
            # 去重并排序：更特殊的性质优先显示
            priority_order = ["对角矩阵", "上三角矩阵", "下三角矩阵", "对称矩阵",
                              "反对称矩阵", "正交矩阵", "幂等矩阵", "幂零矩阵"]
            sorted_types = [st for st in priority_order if st in special_types]

            self.step_generator.add_step(
                f"\\text{{矩阵类型: }} {', '.join(sorted_types)}")
            # 只显示最高优先级的性质
            if sorted_types:
                self.display_special_properties(sorted_types[0], matrix)

        return special_types

    def display_special_properties(self, matrix_type, _matrix):
        """
        显示特殊矩阵的性质(每个类型只显示一次)
        """
        properties = {
            "对角矩阵": [
                r"\text{性质: 特征值就是对角线元素}",
                r"\text{性质: 特征向量是标准基向量}"
            ],
            "上三角矩阵": [
                r"\text{性质: 特征值就是对角线元素}"
            ],
            "下三角矩阵": [
                r"\text{性质: 特征值就是对角线元素}"
            ],
            "对称矩阵": [
                r"\text{性质: 特征值都是实数}",
                r"\text{性质: 不同特征值对应的特征向量正交}"
            ],
            "反对称矩阵": [
                r"\text{性质: 特征值是纯虚数或零}"
            ],
            "正交矩阵": [
                r"\text{性质: 特征值的模长为1}",
                r"\text{性质: } |\lambda| = 1"
            ],
            "幂等矩阵": [
                r"\text{性质: 特征值只能是 0 或 1}"
            ],
            "幂零矩阵": [
                r"\text{性质: 所有特征值都是 0}"
            ]
        }

        if matrix_type in properties:
            for prop in properties[matrix_type]:
                self.step_generator.add_step(prop)

    def characteristic_polynomial_method(self, matrix_input, show_steps=True, is_clear=True):
        """
        特征多项式法求解特征值和特征向量
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{特征多项式法}")
            self.add_matrix(A, "A")
            self.step_generator.add_step(
                r"\text{原理: 求解特征方程 } \det(A - \lambda I) = 0")

        if A.rows != A.cols:
            if show_steps:
                self.step_generator.add_step(r"\text{错误: 特征值只适用于方阵}")
            return None, None

        n = A.rows

        # 分析矩阵结构
        self.analyze_matrix_structure(A, show_steps)

        # 步骤1: 构造特征矩阵
        if show_steps:
            self.add_step("构造特征矩阵 A - $\\lambda$I")

        lambda_sym = symbols('lambda')
        I_mat = eye(n)
        A_minus_lambda_I = A - lambda_sym * I_mat

        if show_steps:
            self.step_generator.add_step(
                f"A - \\lambda I = {latex(A_minus_lambda_I)}")

        # 步骤2: 计算特征多项式
        if show_steps:
            self.add_step("计算特征多项式")

        char_poly = A_minus_lambda_I.det()
        if show_steps:
            self.step_generator.add_step(
                f"\\det(A - \\lambda I) = {latex(char_poly)}")

        char_poly_simplified = simplify(char_poly)
        if show_steps and char_poly_simplified != char_poly:
            self.step_generator.add_step(
                f"\\text{{化简后: }} \\det(A - \\lambda I) = {latex(char_poly_simplified)}")

        # 因式分解特征多项式
        try:
            char_poly_factored = factor(char_poly_simplified)
            if show_steps and char_poly_factored != char_poly_simplified:
                self.step_generator.add_step(
                    f"\\text{{因式分解: }} \\det(A - \\lambda I) = {latex(char_poly_factored)}")
                char_poly_simplified = char_poly_factored
        except:
            if show_steps:
                self.step_generator.add_step(r"\text{无法进行因式分解}")

        # 步骤3: 求解特征值
        if show_steps:
            self.add_step("求解特征值")
            self.step_generator.add_step(
                f"\\text{{解特征方程: }} {latex(char_poly_simplified)} = 0")

        eigenvalues = solve(char_poly_simplified, lambda_sym, dict=False)

        # 处理复数特征值
        complex_eigenvalues = any(eig.has(I) for eig in eigenvalues)
        if complex_eigenvalues and show_steps:
            self.step_generator.add_step(r"\text{注意: 存在复数特征值}")

        # 处理重根情况
        eigenvalue_counts = Counter(eigenvalues)

        if show_steps:
            part = ',\;'.join(
                [f'\\lambda_{{{i+1}}} = {latex(eig)}' for i, eig in enumerate(eigenvalues)])
            self.step_generator.add_step(f"\\text{{特征值: }} {part}")

            # 显示代数重数
            for eig, count in eigenvalue_counts.items():
                if count > 1:
                    self.step_generator.add_step(
                        f"\\text{{特征值 }} {latex(eig)} \\text{{ 的代数重数为 }} {count}")

        # 步骤4: 对每个特征值求解特征向量
        if show_steps:
            self.add_step("求解特征向量")

        eigenvectors = {}
        geometric_multiplicities = {}

        for eigenvalue in set(eigenvalues):
            if show_steps:
                self.step_generator.add_step(
                    f"\\textbf{{求解特征值 }} \\lambda = {latex(eigenvalue)} \\textbf{{ 的特征向量}}")

            A_minus_eig_I = A - eigenvalue * eye(n)
            if show_steps:
                formatted_lambda = self.format_lambda_I(eigenvalue)
                self.step_generator.add_step(
                    f"A - {formatted_lambda} = {latex(A_minus_eig_I)}")

            nullspace = A_minus_eig_I.nullspace()

            geometric_multiplicity = len(nullspace)
            geometric_multiplicities[eigenvalue] = geometric_multiplicity

            if show_steps:
                if nullspace:
                    self.step_generator.add_step(
                        f"\\text{{找到 }} {len(nullspace)} \\text{{ 个线性无关的特征向量}}")
                    self.step_generator.add_step(
                        f"\\text{{几何重数: }} {geometric_multiplicity}")

                    for i, vec in enumerate(nullspace):
                        self.step_generator.add_step(
                            f"\\boldsymbol{{v}}_{{{i+1}}} = {latex(vec)}")
                else:
                    self.step_generator.add_step(r"\text{未找到非零特征向量}")

            eigenvectors[eigenvalue] = nullspace

        # 显示特征值和特征向量的总结
        if show_steps:
            self.display_eigen_summary(
                eigenvalues, eigenvectors, eigenvalue_counts, geometric_multiplicities)

        return eigenvalues, eigenvectors

    def display_eigen_summary(self, eigenvalues, eigenvectors, eigenvalue_counts, geometric_multiplicities):
        """
        显示特征值和特征向量的总结
        """
        self.add_step("特征分析总结")

        self.step_generator.add_step(r"\textbf{特征值总结:}")
        is_diagonalizable = True
        for i, eigenvalue in enumerate(set(eigenvalues)):
            algebraic = eigenvalue_counts[eigenvalue]
            geometric = geometric_multiplicities.get(eigenvalue, 0)
            self.step_generator.add_step(
                f"\\lambda_{{{i+1}}} = {latex(eigenvalue)}, \\text{{代数重数: }} {algebraic}, \\text{{几何重数: }} {geometric}")

            # 检查是否可对角化
            if algebraic != geometric:
                is_diagonalizable = False

        # 显示是否可对角化的结论
        if is_diagonalizable:
            self.step_generator.add_step(r"\textbf{结论: 矩阵可对角化}")
            self.step_generator.add_step(r"\text{原因: 所有特征值的代数重数等于几何重数}")
        else:
            self.step_generator.add_step(r"\textbf{结论: 矩阵不可对角化}")
            self.step_generator.add_step(r"\text{原因: 至少存在一个特征值的代数重数大于几何重数}")

        self.step_generator.add_step(r"\textbf{特征向量:}")
        for eigenvalue, vecs in eigenvectors.items():
            if vecs:
                self.step_generator.add_step(
                    f"\\text{{对于 }} \\lambda = {latex(eigenvalue)}:")
                for j, vec in enumerate(vecs):
                    self.step_generator.add_step(
                        f"\\boldsymbol{{v}}_{{{j+1}}} = {latex(vec)}")

    def auto_eigen_solver(self, matrix_input, show_steps=True, is_clear=True):
        """
        自动求解矩阵的特征值和特征向量
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.step_generator.add_step(r"\textbf{特征值求解器}")
            self.add_matrix(A, "A")

            if A.rows != A.cols:
                self.step_generator.add_step(r"\text{错误: 特征值只适用于方阵}")
                return None, None

        try:
            eigenvalues, eigenvectors = self.characteristic_polynomial_method(
                matrix_input, show_steps, is_clear=False)
            return eigenvalues, eigenvectors
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{求解失败: {str(e)}}}")
            return None, None


def demo_diagonal_and_triangular():
    """演示对角矩阵和三角矩阵"""
    eigen_solver = EigenSolver()

    eigen_solver.step_generator.add_step(r"\textbf{对角矩阵和三角矩阵演示}")

    matrices = [
        ("对角矩阵", "[[2,0,0],[0,3,0],[0,0,5]]"),
        ("上三角矩阵", "[[1,2,3],[0,4,5],[0,0,6]]"),
        ("下三角矩阵", "[[1,0,0],[2,3,0],[4,5,6]]")
    ]

    for name, matrix in matrices:
        eigen_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            eigen_solver.auto_eigen_solver(matrix)
            display(Math(eigen_solver.get_steps_latex()))
        except Exception as e:
            eigen_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(eigen_solver.get_steps_latex()))


def demo_special_matrices():
    """演示特殊矩阵"""
    eigen_solver = EigenSolver()

    eigen_solver.step_generator.add_step(r"\textbf{特殊矩阵演示}")

    matrices = [
        ("对称矩阵", "[[2,1,0],[1,3,1],[0,1,2]]"),
        ("反对称矩阵", "[[0,1,-2],[-1,0,3],[2,-3,0]]"),
        ("幂等矩阵", "[[1,0],[0,1]]"),  # 单位矩阵是幂等矩阵
        ("正交矩阵", "[[0,-1],[1,0]]")  # 旋转矩阵
    ]

    for name, matrix in matrices:
        eigen_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            eigen_solver.auto_eigen_solver(matrix)
            display(Math(eigen_solver.get_steps_latex()))
        except Exception as e:
            eigen_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(eigen_solver.get_steps_latex()))


def demo_complex_eigenvalues():
    """演示复数特征值"""
    eigen_solver = EigenSolver()

    eigen_solver.step_generator.add_step(r"\textbf{复数特征值演示}")

    matrices = [
        ("旋转矩阵", "[[0,-1],[1,0]]"),
        ("复数特征值矩阵", "[[1,-2],[1,3]]")
    ]

    for name, matrix in matrices:
        eigen_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            eigen_solver.auto_eigen_solver(matrix)
            display(Math(eigen_solver.get_steps_latex()))
        except Exception as e:
            eigen_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(eigen_solver.get_steps_latex()))


def demo_defective_matrices():
    """演示亏损矩阵"""
    eigen_solver = EigenSolver()

    eigen_solver.step_generator.add_step(r"\textbf{亏损矩阵演示}")

    matrices = [
        ("若尔当块", "[[2,1,0],[0,2,1],[0,0,2]]"),
        ("亏损矩阵", "[[1,1],[0,1]]")
    ]

    for name, matrix in matrices:
        eigen_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            eigen_solver.auto_eigen_solver(matrix)
            display(Math(eigen_solver.get_steps_latex()))
        except Exception as e:
            eigen_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(eigen_solver.get_steps_latex()))


def demo_symbolic_matrices():
    """演示符号矩阵"""
    eigen_solver = EigenSolver()

    eigen_solver.step_generator.add_step(r"\textbf{符号矩阵演示}")

    matrices = [
        ("2×2 符号矩阵", "[[a,b],[c,d]]"),
        ("对称符号矩阵", "[[p,q],[q,p]]")
    ]

    for name, matrix in matrices:
        eigen_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            eigen_solver.auto_eigen_solver(matrix)
            display(Math(eigen_solver.get_steps_latex()))
        except Exception as e:
            eigen_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(eigen_solver.get_steps_latex()))


def demo_general_matrices():
    """演示普通矩阵的特征值求解"""
    eigen_solver = EigenSolver()

    eigen_solver.step_generator.add_step(r"\textbf{普通矩阵特征值求解演示}")

    matrices = [
        ("2×2 实矩阵示例 1", "[[3,1],[1,7]]"),
        ("2×2 实矩阵示例 2", "[[2,4],[1,2]]"),
        ("3×3 实矩阵示例", "[[4,1,1],[1,4,1],[1,1,4]]"),
        ("具有重特征值的矩阵", "[[3,1,-1],[1,2,-1],[1,1,0]]"),
        ("不可对角化矩阵", "[[2,1,0],[0,2,0],[0,0,3]]"),
        ("随机 3x3 矩阵", "[[2,-1,0],[-1,2,-1],[0,-1,2]]"),
        ("全 1 矩阵", "[[1,1,1],[1,1,1],[1,1,1]]")
    ]

    for name, matrix in matrices:
        eigen_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            eigen_solver.auto_eigen_solver(matrix)
            display(Math(eigen_solver.get_steps_latex()))
        except Exception as e:
            eigen_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(eigen_solver.get_steps_latex()))


def demo_mixed_matrices():
    """演示混合类型的矩阵"""
    eigen_solver = EigenSolver()

    eigen_solver.step_generator.add_step(r"\textbf{混合类型矩阵演示}")

    matrices = [
        ("特征值为 0 的矩阵", "[[1,1],[1,1]]"),
        ("负特征值矩阵", "[[-1,2],[2,-1]]"),
        ("分数特征值矩阵", "[[1/2,1/3],[1/4,1/5]]"),
        ("大数矩阵", "[[100,50],[25,75]]"),
    ]

    for name, matrix in matrices:
        eigen_solver.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            eigen_solver.auto_eigen_solver(matrix)
            display(Math(eigen_solver.get_steps_latex()))
        except Exception as e:
            eigen_solver.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(eigen_solver.get_steps_latex()))


if __name__ == "__main__":
    demo_diagonal_and_triangular()
    demo_special_matrices()
    demo_complex_eigenvalues()
    demo_defective_matrices()
    demo_symbolic_matrices()
    demo_general_matrices()
    demo_mixed_matrices()
