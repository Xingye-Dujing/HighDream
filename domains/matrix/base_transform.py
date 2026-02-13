from typing import Dict, List

from sympy import Matrix, latex

from core import CommonMatrixCalculator


# from sympy import symbols
# from IPython.display import Math, display


class BaseTransform(CommonMatrixCalculator):
    """A class for performing basis transformations and coordinate conversions in linear algebra.

    This class provides methods for computing transition matrices between different bases,
    converting vector coordinates between bases, and handling both numeric and symbolic calculations.
    """

    def show_input(self, basis_matrix: Matrix, name: str = "基", vector_names: str | List[str] = None):
        """Display the input basis matrix with appropriate formatting."""
        if vector_names is None:
            vector_names = [
                f"\\boldsymbol{{v}}_{{{i + 1}}}" for i in range(basis_matrix.cols)]

        self.add_step(f"{name}组成的矩阵")
        for i in range(basis_matrix.rows):
            vector_latex = latex(basis_matrix.col(i))
            self.step_generator.add_step(
                f"{vector_names[i]} = {vector_latex}")
        name = 'U' if name == '新基' else 'V'
        self.add_matrix(basis_matrix, name)

    def basis_change_matrix(self, old_basis: str | Matrix, new_basis: str | Matrix, show_steps: bool = True,
                            need_parse: bool = True,
                            is_clear: bool = True) -> Dict[str, Matrix] | None:
        """Calculate the transition matrix from the old basis to the new basis.

        Computes the transition matrix P such that new_basis = old_basis * P.
        """

        if is_clear:
            self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{基变换: 从旧基到新基的过渡矩阵}")

        # Parse inputs
        if need_parse:
            old_basis = self.parse_matrix_input(old_basis)
            new_basis = self.parse_matrix_input(new_basis)
            self.show_input(old_basis, "旧基",
                            [f"\\boldsymbol{{v}}_{{{i + 1}}}" for i in range(old_basis.cols)])
            self.show_input(new_basis, "新基",
                            [f"\\boldsymbol{{u}}_{{{i + 1}}}" for i in range(new_basis.cols)])

        # Calculate transition matrix P, where new_basis = old_basis * P.
        self.add_step("计算过渡矩阵")
        self.step_generator.add_step(
            r"\text{设过渡矩阵 } P \text{, 其满足: } U = VP (右乘)")
        self.step_generator.add_step(r"\text{即 } P = V^{-1} U")

        # P = old_basis^{-1} * new_basis
        try:
            try:
                old_basis_inv = old_basis.inv()
            except Exception:
                self.step_generator.add_step(
                    r"\text{V 不可逆, 这意味着 V 的列向量线性相关, 不构成基}")
                return None
            self.add_matrix(
                old_basis_inv, r"其中\;V^{-1}")

            P = old_basis_inv * new_basis
            self.add_matrix(P, "P")

            return P

        except Exception as e:
            self.step_generator.add_step(f"\\text{{错误: {e}}}")
            return None

    def coordinate_transform(self, vector_input: str, from_basis_input: str, to_basis_input: str,
                             show_steps: bool = True) -> Dict[str, Matrix] | None:
        """Transform vector coordinates from one basis to another."""
        self.step_generator.clear()
        to_coords = None
        if show_steps:
            self.step_generator.add_step(
                r"\textbf{坐标变换: 向量在不同基下的坐标转换}")

        # Parse inputs
        vector = self.parse_vector_input(vector_input)
        self.add_vector(vector, "[\\boldsymbol{x}]_{V}")
        from_basis = self.parse_matrix_input(from_basis_input)
        to_basis = self.parse_matrix_input(to_basis_input)
        self.show_input(from_basis, "旧基")
        self.show_input(to_basis, "新基", [
            f"\\boldsymbol{{u}}_{{{i + 1}}}" for i in range(to_basis.rows)])

        self.step_generator.add_step(r"\textbf{方法一: 利用过渡矩阵转换坐标}")
        # Calculate transition matrix from the old basis to the new basis.
        P = self.basis_change_matrix(
            from_basis, to_basis, show_steps=False, need_parse=False, is_clear=False)

        if P is None:
            return None

        # Calculate vector coordinates in the new basis
        self.add_step("计算向量在新基下的坐标")
        self.step_generator.add_step(
            r"\text{利用过渡矩阵转换坐标: } [\boldsymbol{x}]_{U} = P^{-1} [\boldsymbol{x}]_{V}")
        try:
            P_inv = P.inv()
            self.add_matrix(P_inv, r"其中\;P^{-1}")

            to_coords = P_inv * vector
            self.add_vector(to_coords, r"[\boldsymbol{x}]_{U}")
        except Exception:
            self.step_generator.add_step(f"\\text{{P 不可逆}}")

        try:
            self.step_generator.add_step(r"\textbf{方法二: 建立等量关系直接求解}")
            self.step_generator.add_step(
                r"\text{等量关系: } V [\boldsymbol{x}]_{V} = U [\boldsymbol{x}]_{U} = \boldsymbol{x}")
            self.step_generator.add_step(
                r"\text{即 } [\boldsymbol{x}]_{U} = U^{-1} V [\boldsymbol{x}]_{V}")
            U_inv = to_basis.inv()
            self.add_matrix(U_inv, r"其中\;U^{-1}")
            to_coords = U_inv * from_basis * vector
            self.add_vector(to_coords, r"[\boldsymbol{x}]_{U}")
        except Exception:
            self.step_generator.add_step(f"\\text{{U 不可逆}}")

        return {
            'from_coordinates': vector,
            'to_coordinates': to_coords,
            'transition_matrix': P
        }

    def standard_to_basis(self, vector_input: str, basis_input: str, show_steps: bool = True) -> Matrix:
        """Convert vector coordinates from the standard basis to a given basis."""

        self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{坐标变换: 从标准基到给定基}")

        vector = self.parse_vector_input(vector_input)
        self.add_vector(vector, "\\boldsymbol{x}")
        basis = self.parse_matrix_input(basis_input)
        self.show_input(basis, "目标基")

        self.add_step("计算向量在新基下的坐标")
        self.step_generator.add_step(
            r"\text{设向量 } \boldsymbol{x} \text{ 在新基下的坐标为 } [\boldsymbol{x}]_{V}")
        self.step_generator.add_step(
            r"\text{满足: } \boldsymbol{x} =  V [\boldsymbol{x}]_{V}")
        self.step_generator.add_step(
            r"\text{即 } [\boldsymbol{x}]_{V} = V^{-1} \boldsymbol{x}")

        coords = basis.solve(vector)
        self.add_vector(basis.inv(), r"其中\;V^{-1}")
        self.add_vector(coords, r"[\boldsymbol{x}]_{V}")

        return coords

    def basis_to_standard(self, coordinates_input: str, basis_input: str, show_steps: bool = True) -> Matrix:
        """Convert vector coordinates from a given basis to the standard basis."""
        self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{坐标变换: 从给定基到标准基}")

        coordinates = self.parse_vector_input(coordinates_input)
        self.add_vector(coordinates, r"[\boldsymbol{x}]_{V}")
        basis = self.parse_matrix_input(basis_input)
        self.show_input(basis, "当前基")

        self.add_step("计算向量在标准基下的表示")
        self.step_generator.add_step(
            r"\text{向量在标准基下的表示为: } \boldsymbol{x} = V [\boldsymbol{x}]_{V}")

        vector = basis * coordinates
        self.add_vector(vector, r"\boldsymbol{x}")

        return vector

    def compute_transform(self, expression: str, operation: str) -> str:
        """Compute various transformations based on provided expressions and operation type."""

        expressions = expression.split('\n')

        if operation in ['basis_change', 'change_basis']:
            # For basis_change_matrix: old_basis, new_basis
            if len(expressions) >= 2:
                self.basis_change_matrix(expressions[0], expressions[1])

        elif operation in ['coordinate_transform', 'coord_transform']:
            # For coordinate_transform: vector, from_basis, to_basis
            if len(expressions) >= 3:
                self.coordinate_transform(
                    expressions[0], expressions[1], expressions[2])

        elif operation in ['standard_to_basis', 'std_to_basis']:
            # For standard_to_basis: vector, basis
            if len(expressions) >= 2:
                self.standard_to_basis(expressions[0], expressions[1])

        elif operation in ['basis_to_standard', 'basis_to_std']:
            # For basis_to_standard: coordinates, basis
            if len(expressions) >= 2:
                self.basis_to_standard(expressions[0], expressions[1])

        return self.get_steps_latex()

# def demo_coordinate_transform():
#     """Demonstrate coordinate transformations with various examples.

#     Shows how to transform coordinates between different bases in 2D and 3D spaces,
#     including both numeric and symbolic examples.
#     """
#     transformer = BaseTransform()

#     transformer.step_generator.add_step(r"\textbf{坐标变换演示}")

#     # Example 1: 2D space coordinate transformation

#     transformer.step_generator.add_step(r"\text{例 1: 二维空间坐标变换}")
#     vector1 = [3, 2]
#     old_basis1 = [[2, 1], [1, 2]]
#     new_basis1 = [[1, 1], [1, -1]]

#     try:
#         transformer.coordinate_transform(
#             vector1, old_basis1, new_basis1)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")

#     display(Math(transformer.get_steps_latex()))

#     # Example 2: From the standard basis to given the basis.

#     transformer.step_generator.add_step(r"\text{例 2: 从标准基到给定基}")
#     vector2 = [4, 3]
#     basis2 = [[1, 1], [1, -1]]

#     try:
#         transformer.standard_to_basis(vector2, basis2)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")

#     display(Math(transformer.get_steps_latex()))

#     # Example 3: From the given basis to the standard basis.

#     transformer.step_generator.add_step(r"\text{例 3: 从给定基到标准基}")
#     coords3 = [2, 1]
#     basis3 = [[1, 1], [1, -1]]

#     try:
#         transformer.basis_to_standard(coords3, basis3)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")

#     display(Math(transformer.get_steps_latex()))

#     # Example 4: 3D space coordinate transformation

#     transformer.step_generator.add_step(r"\text{例 4: 三维空间坐标变换}")
#     vector4 = [1, 2, 3]
#     old_basis4 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Standard basis
#     new_basis4 = [[1, 1, 0], [1, -1, 0], [0, 0, 2]]

#     try:
#         transformer.coordinate_transform(
#             vector4, old_basis4, new_basis4)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")
#     display(Math(transformer.get_steps_latex()))


# def demo_basis_change():
#     """Demonstrate basis change computations.

#     Shows how to calculate transition matrices between different bases in 2D and 3D spaces.
#     """
#     transformer = BaseTransform()

#     transformer.step_generator.add_step(r"\textbf{基变换演示}")

#     # Example 1: Standard basis to another basis

#     old_basis1 = [[2, 1], [1, 2]]
#     new_basis1 = [[1, 1], [1, -1]]

#     try:
#         transformer.basis_change_matrix(old_basis1, new_basis1)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")

#     display(Math(transformer.get_steps_latex()))

#     # Example 2: 3D space basis change

#     old_basis2 = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
#     new_basis2 = [[1, 1, 0], [1, -1, 0], [0, 0, 2]]

#     try:
#         transformer.basis_change_matrix(old_basis2, new_basis2)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")
#     display(Math(transformer.get_steps_latex()))


# def demo_symbolic_basis_transform():
#     """Demonstrate basis transformations with symbolic elements.

#     Shows how to work with symbolic expressions in basis transformations and coordinate conversions.
#     """
#     transformer = BaseTransform()

#     transformer.step_generator.add_step(r"\textbf{符号基变换和坐标变换演示}")
#     transformer.step_generator.add_step(r"\textbf{假设所有符号表达式不为 0}")

#     # Example 1: Symbolic basis change

#     transformer.step_generator.add_step(r"\text{例 1: 符号基变换}")

#     # Define symbols
#     a, b, c, d = symbols('a b c d')

#     # Symbolic bases
#     old_basis_symbolic = Matrix([
#         [1, a],
#         [b, 1]
#     ])

#     new_basis_symbolic = Matrix([
#         [c, 1],
#         [1, d]
#     ])

#     try:
#         transformer.basis_change_matrix(old_basis_symbolic, new_basis_symbolic)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")

#     display(Math(transformer.get_steps_latex()))

#     # Example 2: Symbolic coordinate transformation

#     transformer.step_generator.add_step(r"\text{例 2: 符号坐标变换}")

#     # Symbolic vector coordinates
#     x, y = symbols('x y')
#     vector_symbolic = Matrix([x, y])

#     # Symbolic basis
#     basis_symbolic = Matrix([
#         [1, 2],
#         [3, 4]
#     ])

#     try:
#         transformer.standard_to_basis(vector_symbolic, basis_symbolic)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")

#     display(Math(transformer.get_steps_latex()))

#     # Example 3: Coordinate transformation between symbolic bases

#     transformer.step_generator.add_step(r"\text{例 3: 符号基之间的坐标变换}")

#     p, q = symbols('p q')
#     vector_symbolic2 = Matrix([p, q])

#     old_basis_symbolic2 = Matrix([
#         [1, 1],
#         [1, -1]
#     ])

#     new_basis_symbolic2 = Matrix([
#         [a, 0],
#         [0, b]
#     ])

#     try:
#         transformer.coordinate_transform(
#             vector_symbolic2, old_basis_symbolic2, new_basis_symbolic2)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")

#     display(Math(transformer.get_steps_latex()))

#     # Example 4: 3D symbolic basis change

#     transformer.step_generator.add_step(r"\text{例 4: 三维符号基变换}")

#     t, u, v = symbols('t u v')
#     old_basis_3d = Matrix([
#         [1, 0, t],
#         [0, 1, u],
#         [0, 0, 1]
#     ])

#     new_basis_3d = Matrix([
#         [1, 1, 0],
#         [1, -1, 0],
#         [v, 0, 1]
#     ])

#     try:
#         transformer.basis_change_matrix(old_basis_3d, new_basis_3d)
#     except Exception as e:
#         transformer.step_generator.add_step(
#             f"\\text{{错误: }} {str(e)}")

#     display(Math(transformer.get_steps_latex()))


# if __name__ == "__main__":
#     demo_basis_change()
#     demo_coordinate_transform()
#     demo_symbolic_basis_transform()
