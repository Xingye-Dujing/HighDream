from typing import List
from sympy import Eq, latex, symbols, Matrix, zeros, Rational, sympify
from IPython.display import display, Math

from core import CommonMatrixCalculator


class LinearSystemConverter(CommonMatrixCalculator):

    @staticmethod
    def str_to_Eq(expressions: List[str], get_unknowns=False):
        unknowns_str = ''
        expressions = expressions.split('\n')
        if '=' not in expressions[-1]:
            unknowns_str = expressions[-1]
            expressions = expressions[:-1]
        elif get_unknowns:
            raise ValueError("请在最后一行指定未知数")
        eq_list = []
        for i, expr in enumerate(expressions):
            if '=' not in expr:
                raise ValueError(f"第{i+1}行的方程格式错误")
            left, right = expr.split('=')
            eq = Eq(sympify(left), sympify(right))
            eq_list.append(eq)
        if get_unknowns:
            unkowns = [sympify(sym) for sym in unknowns_str.split(' ')]
            return eq_list, unkowns
        return eq_list

    def equations_to_matrix(self, equations: List[Eq], unknowns=None, parameters=None):
        """
        将线性方程组转换为矩阵形式 Ax = b
        支持 A 和 b 中包含参数

        equations: 方程列表
        unknowns: 未知数列表
        parameters: 参数列表(可选. 如果不指定, 则所有非未知数的符号都视为参数)
        """
        # 处理输入方程
        processed_eqs = []
        for eq in equations:
            processed_eqs.append(eq.lhs - eq.rhs)

        # 提取所有符号
        all_symbols = set()
        for eq in processed_eqs:
            all_symbols.update(eq.free_symbols)
        all_symbols = sorted(list(all_symbols), key=str)

        # 确定未知数和参数
        if unknowns is None:
            # 如果没有指定未知数，使用所有符号
            unknowns = all_symbols
            parameters = []
        else:
            # 如果指定了未知数，确定参数
            if parameters is None:
                parameters = [
                    sym for sym in all_symbols if sym not in unknowns]

        # 构建系数矩阵 A 和常数向量 b
        n_eq = len(processed_eqs)
        n_unknowns = len(unknowns)

        A = zeros(n_eq, n_unknowns)
        b = zeros(n_eq, 1)

        for i, eq in enumerate(processed_eqs):
            # 对每个方程，提取未知数的系数
            for j, unknown in enumerate(unknowns):
                # 提取unknown的系数(可能包含参数)
                coeff = eq.coeff(unknown)
                A[i, j] = coeff
                # 从方程中减去该项
                eq = eq - coeff * unknown

            # 剩余部分(包含参数和常数)放入b, 注意符号
            b[i, 0] = -eq

        # 创建未知数向量
        x = Matrix(unknowns)

        return A, x, b, parameters

    def matrix_to_equations(self, A, x, b):
        """
        将矩阵形式 Ax = b 转换为线性方程组
        支持A和b中包含参数
        """
        equations = []
        for i in range(A.rows):
            lhs = 0
            for j in range(A.cols):
                if A[i, j] != 0:
                    if lhs == 0:
                        lhs = A[i, j] * x[j]
                    else:
                        lhs += A[i, j] * x[j]
            equations.append(Eq(lhs, b[i]))

        return equations

    def show_equations_to_matrix(self, equations, unknowns=None, parameters=None, show_steps=True):
        """显示方程组到矩阵的转换过程, 支持参数"""
        if isinstance(equations, str):
            equations = self.str_to_Eq(equations)
        print(equations)

        if show_steps:
            self.step_generator.clear()
            self.add_step("方程组到矩阵的转换")

        A, x, b, params = self.equations_to_matrix(
            equations, unknowns, parameters)

        if show_steps:
            # 显示原始方程组
            self.add_step("原始线性方程组")
            eq_latex = r"\begin{cases}"
            for eq in equations:
                if isinstance(eq, Eq):
                    eq_latex += latex(eq.lhs) + " = " + latex(eq.rhs) + r" \\ "
                else:
                    eq_latex += latex(eq) + " = 0" + r" \\ "
            eq_latex += r"\end{cases}"
            self.step_generator.add_step(eq_latex)

            # 显示未知数和参数信息
            if unknowns:
                u_str = ''
                for u in unknowns:
                    u_str += latex(u) + r"\;\;"
                self.step_generator.add_step(
                    r"\text{未知数: }" + u_str)
            if params:
                p_str = ''
                for p in params:
                    p_str += latex(p) + r"\;\;"
                self.step_generator.add_step(
                    r"\text{参数: }" + p_str)

            # 显示矩阵形式
            self.add_step("矩阵形式")
            self.step_generator.add_step(rf"A = {latex(A)}")
            self.step_generator.add_step(rf"\boldsymbol{{x}} = {latex(x)}")
            self.step_generator.add_step(rf"\boldsymbol{{b}} = {latex(b)}")

            # 显示完整的矩阵方程
            self.add_step("完整矩阵方程")
            self.step_generator.add_step(f"{latex(A)} {latex(x)} = {latex(b)}")

        return A, x, b, params

    def show_matrix_to_equations(self, A, x, b, show_steps=True):
        """显示矩阵到方程组的转换过程，支持参数"""
        if show_steps:
            self.step_generator.clear()
            self.add_step("矩阵到方程组的转换")

        equations = self.matrix_to_equations(A, x, b)

        if show_steps:
            # 显示矩阵形式
            self.add_step("矩阵形式")
            self.step_generator.add_step(rf"A = {latex(A)}")
            self.step_generator.add_step(rf"\boldsymbol{{x}} = {latex(x)}")
            self.step_generator.add_step(rf"\boldsymbol{{b}} = {latex(b)}")
            self.step_generator.add_step(f"{latex(A)} {latex(x)} = {latex(b)}")

            # 显示转换后的方程组
            self.add_step("对应的线性方程组")
            eq_latex = r"\begin{cases}"
            for eq in equations:
                eq_latex += latex(eq) + r" \\ "
            eq_latex += r"\end{cases}"
            self.step_generator.add_step(eq_latex)

        return equations


def demo():
    converter = LinearSystemConverter()

    # 定义符号
    x, y, z = symbols('x y z')
    a, b, c, d, k, m, n = symbols('a b c d k m n')

    # 示例1：简单的参数系统
    converter.step_generator.add_step(r"\textbf{示例1: 简单的参数系统}")
    equations1 = [
        Eq(a*x + b*y, c),
        Eq(d*x - b*y, k)
    ]
    converter.show_equations_to_matrix(
        equations1, unknowns=[x, y])
    display(Math(converter.get_steps_latex()))

    # 示例2：混合情况 - 部分指定未知数
    converter.step_generator.add_step(r"\textbf{示例2: 混合情况}")
    equations2 = [
        Eq(2*x + a*y, b + 1),
        Eq(b*x - 3*y, c)
    ]
    converter.show_equations_to_matrix(
        equations2, unknowns=[x, y])
    display(Math(converter.get_steps_latex()))

    # 示例3：3变量带多个参数
    converter.step_generator.add_step(r"\textbf{示例3: 三变量多参数系统}")
    equations3 = [
        Eq(a*x + b*y + c*z, d),
        Eq(2*x + k*y - z, m),
        Eq(x + y + z, n)
    ]
    A3, x3, b3, _ = converter.show_equations_to_matrix(
        equations3, unknowns=[x, y, z])
    display(Math(converter.get_steps_latex()))

    # 示例4：从矩阵形式转回方程组（带参数）
    converter.step_generator.add_step(rf"\textbf{{示例4: 矩阵形式转回方程组(带参数)}}")
    converter.show_matrix_to_equations(A3, x3, b3)
    display(Math(converter.get_steps_latex()))

    # 示例5：明确指定参数
    converter.step_generator.add_step(r"\textbf{示例5: 明确指定参数}")
    equations5 = [
        Eq(k*x + m*y, n),
        Eq(2*k*x - y, 3*m)
    ]
    converter.show_equations_to_matrix(
        equations5, unknowns=[x, y], parameters=[k, m, n])
    display(Math(converter.get_steps_latex()))

    # 示例6：物理系统示例 - 弹簧质量系统
    converter.step_generator.add_step(r"\textbf{示例6: 物理系统示例}")
    F1, F2, k1, k2, k3 = symbols('F1 F2 k1 k2 k3')
    x1_sym, x2_sym = symbols('x1 x2')

    equations6 = [
        Eq((k1 + k2)*x1_sym - k2*x2_sym, F1),
        Eq(-k2*x1_sym + (k2 + k3)*x2_sym, F2)
    ]
    converter.show_equations_to_matrix(
        equations6, unknowns=[x1_sym, x2_sym])
    display(Math(converter.get_steps_latex()))

    converter.step_generator.add_step(r"\textbf{示例7: 数值示例}")
    x, y = symbols('x y')

    equations7 = [
        Eq(8*x - 3*y, 6),
        Eq(10*x + 2*y, 8)
    ]
    converter.show_equations_to_matrix(
        equations7, unknowns=[x, y])
    display(Math(converter.get_steps_latex()))

    converter.step_generator.add_step(r"\textbf{示例8: 分数示例}")
    x, y = symbols('x y')

    equations8 = [
        Eq(8*x - Rational(5, 6)*y - 6*a, 0),
        Eq(10*x + 2*y, 8)
    ]
    converter.show_equations_to_matrix(
        equations8, unknowns=[x, y])
    display(Math(converter.get_steps_latex()))


if __name__ == "__main__":
    demo()
