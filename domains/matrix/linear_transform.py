from sympy import Matrix, latex, sympify, symbols
from IPython.display import display, Math

from domains.matrix import CommonStepGenerator


class LinearTransform:

    def __init__(self):
        self.step_counter = 0
        self.step_generator = CommonStepGenerator()

    def add_step(self, title):
        """显示步骤标题"""
        self.step_counter += 1
        self.step_generator.add_step(
            f"\\textbf{{步骤 {self.step_counter}: }} \\text{{{title}}}")

    def clear_steps(self):
        """清除步骤记录"""
        self.step_counter = 0
        self.step_generator.clear()

    def get_steps_latex(self):
        """获取步骤的LaTeX格式"""
        return self.step_generator.get_steps_latex()

    def add_matrix(self, matrix, name="A"):
        """添加矩阵到步骤"""
        self.step_generator.add_step(f"{name} = {latex(matrix)}")

    def add_vector(self, vector, name="x"):
        """添加向量到步骤"""
        self.step_generator.add_step(f"{name} = {latex(vector)}")

    def parse_basis_input(self, basis_matrix, name="B", vector_names=None):
        """解析基输入"""
        try:
            self.add_step(f"{name}组成的矩阵")
            if vector_names is None:
                vector_names = [
                    f"\\boldsymbol{{b}}_{{{i+1}}}" for i in range(basis_matrix.rows)]

            for i in range(basis_matrix.rows):
                vector_latex = latex(basis_matrix.col(i))
                self.step_generator.add_step(
                    f"{vector_names[i]} = {vector_latex}")

            matrix_symbol = 'C' if name == "新基" else 'B'
            self.add_matrix(basis_matrix, matrix_symbol)

        except Exception as e:
            raise ValueError(f"无法解析基输入: {basis_matrix}") from e

    def parse_input(self, transformation_input):
        """解析线性变换输入"""
        try:
            return Matrix(sympify(transformation_input))
        except Exception as e:
            raise ValueError(
                f"无法解析线性变换输入: {transformation_input}") from e

    def find_linear_transform_matrix(self, transformation_input, basis_input, show_steps=True):
        """求线性变换在给定基下的矩阵表示"""
        if show_steps:
            self.clear_steps()
            self.step_generator.add_step(r"\textbf{求线性变换在给定基下的矩阵表示}")

        # 解析基
        basis = self.parse_input(basis_input)
        self.parse_basis_input(basis, "给定基")

        # 解析线性变换
        self.add_step("线性变换(标准基下)")
        transformation = self.parse_input(transformation_input)

        if isinstance(transformation, Matrix):
            self.add_matrix(transformation, "T")

        # 应用线性变换到基向量
        transformed_basis_vectors = []

        for i in range(basis.cols):
            basis_vector = basis.col(i)
            transformed_vector = transformation * basis_vector

            if i == 0:
                self.add_step("应用线性变换到基向量")
            self.add_vector(
                basis_vector, f"\\boldsymbol{{b}}_{{{i+1}}}")
            self.add_vector(transformed_vector,
                            f"T(\\boldsymbol{{b}}_{{{i+1}}})")
            transformed_basis_vectors.append(transformed_vector)

        # 构建变换后的基向量矩阵
        T_basis_matrix = transformed_basis_vectors[0]
        for i in range(1, len(transformed_basis_vectors)):
            T_basis_matrix = T_basis_matrix.row_join(
                transformed_basis_vectors[i])

        self.add_matrix(T_basis_matrix, "T(B)")

        # 求解线性变换矩阵 A, 使得 T(B) = B * A
        self.add_step("求解线性变换矩阵")
        self.step_generator.add_step(
            r"\text{设线性变换矩阵为 } A \text{, 满足: } A = B^{-1} T(B)")

        try:
            basis_inv = basis.inv()
            self.add_matrix(basis_inv, r"\text{其中, } B^{-1}")

            A = basis_inv * T_basis_matrix
            self.add_matrix(A, "A")

            return A

        except Exception as e:
            self.step_generator.add_step(f"\\text{{计算失败: {str(e)}}}")
            return None

    def change_transform_basis(self, transform_matrix, from_basis_input, to_basis_input, show_steps=True):
        """将线性变换矩阵从一组基转换到另一组基"""
        if show_steps:
            self.clear_steps()
            self.step_generator.add_step(r"\textbf{线性变换矩阵的基变换}")

        # 解析输入
        transform_matrix = self.parse_input(transform_matrix)

        if isinstance(transform_matrix, Matrix):
            self.add_matrix(transform_matrix, "A")

        from_basis = self.parse_input(from_basis_input)
        to_basis = self.parse_input(to_basis_input)
        self.parse_basis_input(from_basis, "原基")
        self.parse_basis_input(to_basis, "新基", [
            f"\\boldsymbol{{c}}_{{{i+1}}}" for i in range(to_basis.rows)])

        # 计算基变换矩阵
        self.add_step("计算基变换矩阵")
        self.step_generator.add_step(
            r"\text{设从原基 } B \text{ 到新基 } C \text{ 的过渡矩阵为 } P")
        self.step_generator.add_step(r"\text{满足: } C = B P")
        self.step_generator.add_step(r"\text{即 } P = B^{-1} C")

        try:
            from_basis_inv = from_basis.inv()
            self.add_matrix(from_basis_inv, "B^{-1}")

            P = from_basis_inv * to_basis
            self.add_matrix(P, "P")

            # 计算新基下的线性变换矩阵
            self.add_step("计算新基下的线性变换矩阵")
            self.step_generator.add_step(
                r"\text{新基下的线性变换矩阵为: } A' = P^{-1} A P")

            P_inv = P.inv()
            self.add_matrix(P_inv, "其中,\;P^{-1}")

            A_prime = P_inv * transform_matrix * P
            self.add_matrix(A_prime, "A'")

            return A_prime

        except Exception as e:
            self.step_generator.add_step(f"\\text{{计算失败: {str(e)}}}")
            return None

    def compute(self, expression, operation):
        expressions = expression.split('\n')

        if operation in ['find_matrix', 'find_transform_matrix']:
            # For find_linear_transform_matrix: transformation, basis
            if len(expressions) >= 2:
                self.find_linear_transform_matrix(
                    expressions[0], expressions[1])

        elif operation in ['change_basis', 'basis_change', 'change_transform_basis']:
            # For change_transform_basis: transform_matrix, from_basis, to_basis
            if len(expressions) >= 3:
                self.change_transform_basis(
                    expressions[0], expressions[1], expressions[2])
            else:
                raise ValueError('需要三个矩阵')

        return self.get_steps_latex()


def demo_linear_transform():
    """演示线性变换矩阵求解"""
    transformer = LinearTransform()

    transformer.step_generator.add_step(r"\textbf{线性变换矩阵求解演示}")

    # 示例 1: 二维旋转变换
    transformer.step_generator.add_step(r"\text{例 1: 二维旋转变换(90 度)}")
    rotation_matrix = [[0, -1], [1, 0]]
    basis1 = [[1, 0], [0, 1]]  # 标准基

    try:
        transformer.find_linear_transform_matrix(rotation_matrix, basis1)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))

    # 示例 2: 三维旋转变换
    transformer.step_generator.add_step(r"\text{例 2: 三维变换}")
    matrix1 = [[2, 2, 0], [1, 1, 2], [1, 1, 2]]
    basis1 = [[1, -2, 1], [-1, 1, 1], [0, 1, 1]]

    try:
        transformer.find_linear_transform_matrix(matrix1, basis1)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))

    # 示例 3: 投影变换
    transformer.step_generator.add_step(r"\text{例 3: 到 $x$ 轴的投影变换}")
    projection_matrix = [[1, 0], [0, 0]]
    basis2 = [[2, 1], [1, 2]]

    try:
        transformer.find_linear_transform_matrix(
            projection_matrix, basis2)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))


def demo_basis_change():
    """演示线性变换矩阵的基变换"""
    transformer = LinearTransform()

    transformer.step_generator.add_step(r"\textbf{线性变换矩阵基变换演示}")

    # 示例 1: 旋转变换在不同基下的表示
    transformer.step_generator.add_step(r"\text{例 1: 旋转变换在不同基下的表示}")
    rotation_matrix = [[0, -1], [1, 0]]  # 90度旋转
    standard_basis = [[1, 0], [0, 1]]
    new_basis = [[1, 1], [1, -1]]

    try:
        transformer.change_transform_basis(
            rotation_matrix, standard_basis, new_basis)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))

    # 示例 2: 投影变换的基变换
    transformer.step_generator.add_step(r"\text{例 2: 投影变换的基变换}")
    projection_matrix = [[1, 0], [0, 0]]
    from_basis = [[1, 0], [0, 1]]
    to_basis = [[2, 1], [1, 2]]

    try:
        transformer.change_transform_basis(
            projection_matrix, from_basis, to_basis)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))

    # 示例 3: 三维空间的基变换
    transformer.step_generator.add_step(r"\text{例 3: 三维空间的基变换}")
    transform_3d = [[1, 2, 0], [0, 1, 1], [1, 0, 1]]
    from_basis_3d = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    to_basis_3d = [[1, 1, 0], [1, -1, 0], [0, 0, 2]]

    try:
        transformer.change_transform_basis(
            transform_3d, from_basis_3d, to_basis_3d)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))


def demo_symbolic_transform():
    """演示包含符号元素的线性变换"""
    transformer = LinearTransform()

    transformer.step_generator.add_step(r"\textbf{符号线性变换演示}")
    transformer.step_generator.add_step(r"\textbf{假设所有符号表达式不为 0}")

    # 示例 1: 带参数的二维线性变换
    transformer.step_generator.add_step(r"\text{例 1: 带参数的二维线性变换}")

    # 定义符号
    a, b, c, d = symbols('a b c d')

    # 符号矩阵
    symbolic_matrix = Matrix([
        [a, b],
        [c, d]
    ])

    basis1 = [[1, 0], [0, 1]]

    try:
        transformer.find_linear_transform_matrix(symbolic_matrix, basis1)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))

    # 示例 2: 符号基下的线性变换
    transformer.step_generator.add_step(r"\text{例 2: 符号基下的线性变换}")

    # 固定变换矩阵
    fixed_matrix = [[2, 1], [1, 3]]

    # 符号基
    k, m = symbols('k m')
    symbolic_basis = Matrix([
        [1, k],
        [m, 1]
    ])

    try:
        transformer.find_linear_transform_matrix(fixed_matrix, symbolic_basis)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))

    # 示例 3: 符号基变换
    transformer.step_generator.add_step(r"\text{例 3: 符号基变换}")

    # 符号变换矩阵
    p, q, r, s = symbols('p q r s')
    symbolic_transform = Matrix([
        [p, q],
        [r, s]
    ])

    # 符号基
    alpha, beta = symbols('alpha beta')
    from_basis = Matrix([
        [1, 0],
        [0, 1]
    ])
    to_basis = Matrix([
        [alpha, 1],
        [1, beta]
    ])

    try:
        transformer.change_transform_basis(
            symbolic_transform, from_basis, to_basis)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))

    # 示例 4: 三维符号变换
    transformer.step_generator.add_step(r"\text{例 4: 三维符号变换}")

    t, u, v = symbols('t u v')
    symbolic_3d_matrix = Matrix([
        [1, t, 0],
        [u, 2, v],
        [0, 1, 3]
    ])

    symbolic_3d_basis = Matrix([
        [1, 0, 1],
        [0, 1, t],
        [1, u, 0]
    ])

    try:
        transformer.find_linear_transform_matrix(
            symbolic_3d_matrix, symbolic_3d_basis)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))

    # 示例 5: 函数形式的符号变换
    transformer.step_generator.add_step(r"\text{例 5: 函数形式的符号变换}")

    def symbolic_function_transform(vector):
        """符号线性变换: T(x,y) = (ax + by, cx + dy)"""
        x, y = vector[0], vector[1]
        result = Matrix([a*x + b*y, c*x + d*y])
        desc = f'T(x,y) = ({a}x + {b}y, {c}x + {d}y)'
        return result, desc

    standard_basis = [[1, 0], [0, 1]]

    try:
        transformer.find_linear_transform_matrix(
            symbolic_function_transform, standard_basis)
        display(Math(transformer.get_steps_latex()))
    except Exception as e:
        transformer.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(transformer.get_steps_latex()))


if __name__ == "__main__":
    demo_linear_transform()
    demo_basis_change()
    demo_symbolic_transform()
