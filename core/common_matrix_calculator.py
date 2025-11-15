from abc import ABC

from sympy import Matrix, latex, simplify, sympify, zeros

from core import MatrixStepGenerator


class CommonMatrixCalculator(ABC):

    def __init__(self):
        self.step_generator = MatrixStepGenerator()

    def add_step(self, title):
        """显示步骤标题"""
        self.step_generator.add_step(f"\\text{{{title}}}")

    def add_matrix(self, matrix, name="A"):
        """显示矩阵"""
        self.step_generator.add_step(f"{name} = {latex(matrix)}")

    def add_vector(self, vector, name="x"):
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
            return Matrix(sympify(matrix_input))
        except Exception as e:
            raise ValueError(f"无法解析矩阵输入: {matrix_input}, 错误: {str(e)}") from e

    def parse_vector_input(self, vector_input):
        """解析向量输入"""
        try:
            # 处理向量输入，如 '[1,2,3]' 或 '[[1],[2],[3]]'
            if vector_input.startswith('[[') and vector_input.endswith(']]'):
                vector = Matrix(sympify(vector_input))
            else:
                # 转换为列向量格式
                vector_str = vector_input.strip('[]')
                elements = [sympify(x.strip())
                            for x in vector_str.split(',')]
                vector = Matrix(elements)
            return vector
        except Exception as e:
            raise ValueError(f"无法解析向量输入: {vector_input}, 错误: {str(e)}") from e

    def simplify_matrix(self, matrix):
        """
        对矩阵的每个元素进行化简
        """
        simplified_matrix = zeros(matrix.rows, matrix.cols)

        for i in range(matrix.rows):
            for j in range(matrix.cols):
                element = matrix[i, j]

                simplified_matrix[i, j] = simplify(element)

        return simplified_matrix
