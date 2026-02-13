from typing import Tuple

from sympy import Matrix, latex, zeros

from core import CommonMatrixCalculator


# from IPython.display import Math, display


class VectorProjectionSolver(CommonMatrixCalculator):

    def check_subspace_type(self, subspace_basis: str | Matrix, show_steps: bool = True) -> str:
        """Check the type of subspace

        Parameters:
            subspace_basis: Basis vectors of the subspace
            show_steps: Whether to show calculation steps.

        Returns:
            "zero": Zero subspace
            "line": Line (one-dimensional subspace)
            "plane": Plane (two-dimensional subspace)
            "hyperplane": Hyperplane
            "full_space": Full space
            "degenerate": Degenerate subspace (linearly dependent basis)
        """
        if show_steps:
            self.add_step("子空间分析")

        # Parse input
        A = self.parse_matrix_input(subspace_basis)
        m, n = A.rows, A.cols

        # Check if it is a zero matrix
        if A.norm() == 0:
            if show_steps:
                self.step_generator.add_step(r"\text{子空间类型: 零子空间}")
            return "zero"

        # Calculate rank
        rank_A = A.rank()

        if show_steps:
            self.step_generator.add_step(f"\\mathrm{{rank}}(A) = {rank_A}")

        # Determine the subspace type
        if rank_A < n:
            if show_steps:
                self.step_generator.add_step(r"\text{警告: 基向量线性相关}")
            subspace_type = "degenerate"
        elif rank_A == 0:
            subspace_type = "zero"
        elif rank_A == 1:
            subspace_type = "line"
        elif rank_A == 2:
            subspace_type = "plane"
        elif rank_A == m:
            subspace_type = "full_space"
        else:
            subspace_type = "hyperplane"

        type_names = {
            "zero": "零子空间",
            "line": "直线(一维子空间)",
            "plane": "平面(二维子空间)",
            "hyperplane": "超平面",
            "full_space": "全空间",
            "degenerate": "退化子空间(线性相关的基)"
        }

        if show_steps:
            self.step_generator.add_step(
                f"\\text{{子空间类型: {type_names[subspace_type]}}}")

        return subspace_type

    def gram_schmidt_orthogonalization(self, A: str | Matrix, show_steps: bool = True) -> Tuple[Matrix, Matrix]:
        """Gram-Schmidt orthogonalization process

        Parameters:
            A: Matrix containing basis vectors as columns
            show_steps: Whether to show calculation steps.

        Returns:
            Q: The orthogonal matrix
            R: Upper triangular matrix
        """
        if show_steps:
            self.add_step("Gram-Schmidt 正交化")

        if isinstance(A, str):
            A = self.parse_matrix_input(A)

        m, n = A.rows, A.cols
        Q = zeros(m, n)  # Orthogonal vector matrix
        R = zeros(n, n)  # Upper triangular matrix

        # Copy original vectors
        vectors = [A.col(i) for i in range(n)]

        for i in range(n):
            if show_steps:
                self.step_generator.add_step(f"\\text{{处理第 {i + 1} 个向量}}")
                self.add_vector(vectors[i], f"\\boldsymbol{{a_{i + 1}}}")

            # Start orthogonalization
            v = vectors[i].copy()

            desc = ""
            for j in range(i):

                # Calculate projection coefficient
                r_ji = vectors[i].dot(Q.col(j))
                R[j, i] = r_ji

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{投影系数: }} r_{{{j + 1}{i + 1}}} = "
                        f"\\boldsymbol{{a_{i + 1}}} \\cdot \\boldsymbol{{q_{j + 1}}} = {latex(r_ji)}"
                    )
                    self.step_generator.add_step(
                        f"\\text{{投影向量: }} {latex(r_ji)} \\cdot \\boldsymbol{{q_{j + 1}}}"
                    )
                    self.step_generator.add_step(
                        f"\\text{{减去在 }} \\boldsymbol{{q_{j + 1}}} \\text{{ 上的投影}}"
                    )
                    desc += f"- r_{{{j + 1}{i + 1}}} \\boldsymbol{{q_{j + 1}}}"

                # Subtract projection
                v = v - r_ji * Q.col(j)

                if show_steps:
                    if v == zeros(m, 1):
                        self.step_generator.add_step(
                            f"\\boldsymbol{{v_{i + 1}}}^{{({j + 1})}} = {latex(v)} = \\boldsymbol{{0}}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"\\boldsymbol{{v_{i + 1}}}^{{({j + 1})}} = {latex(v)}"
                        )

            # Calculate norm
            norm_v = v.norm()

            if norm_v > 0:
                # Normalize
                q_i = v / norm_v
                R[i, i] = norm_v

                if show_steps:
                    if i == 0:
                        self.step_generator.add_step(
                            f"\\boldsymbol{{v_{i + 1}}} = \\boldsymbol{{a_{i + 1}}} {desc}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"\\boldsymbol{{v_{i + 1}}} = \\boldsymbol{{a_{i + 1}}} {desc} = "
                            f"\\boldsymbol{{v_{i + 1}}}^{{({i})}}"
                        )
                    self.step_generator.add_step(
                        f"\\text{{范数: }} \\|\\boldsymbol{{v_{i + 1}}}\\| = {latex(norm_v)}"
                    )
                    self.step_generator.add_step("\\text{{单位向量: }}")
                    self.add_vector(
                        q_i,
                        f"\\boldsymbol{{q_{i + 1}}} = "
                        f"\\frac{{\\boldsymbol{{v_{i + 1}}}}}{{ \\|\\boldsymbol{{v_{i + 1}}}\\|}}"
                    )

                # Store orthogonal vector
                for k in range(m):
                    Q[k, i] = q_i[k]
            else:
                # Zero vector, linearly dependent
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{第 {i + 1} 个向量与前面某个向量线性相关, 跳过}}")
                R[i, i] = 0

        if show_steps:
            self.add_step("正交化结果")
            self.add_matrix(Q, "Q")
            self.add_matrix(R, "R")

        return Q, R

    def project_onto_line(self, vector: Matrix, line_direction: Matrix, show_steps: bool = True) -> Matrix:
        """Project onto a line (one-dimensional subspace)

        Parameters:
            vector: Vector to be projected
            line_direction: Direction vector of the line
            show_steps: Whether to show calculation steps.

        Returns:
            projection: Projection of vector onto the line
        """
        if show_steps:
            self.add_step("一维子空间投影(直线)")
            self.add_vector(vector, "\\boldsymbol{v}")
            self.add_vector(line_direction, "\\boldsymbol{u}")

        # Ensure direction vector is a unit vector
        u_norm = line_direction.norm()

        if show_steps:
            self.step_generator.add_step(
                f"\\text{{方向向量范数: }} \\|\\boldsymbol{{u}}\\| = {latex(u_norm)}"
            )

        if u_norm != 1:
            u_unit = line_direction / u_norm
            if show_steps:
                self.step_generator.add_step(r"\text{标准化方向向量: }")
                self.add_vector(u_unit, "\\boldsymbol{\\hat{u}}")
        else:
            u_unit = line_direction

        # Calculate projection
        projection_coeff = vector.dot(u_unit)
        projection = projection_coeff * u_unit

        if show_steps:
            part = ""
            for i in range(vector.shape[0]):
                if i == 0:
                    part += f"v_{{{i}}}u_{{{i}}}"
                else:
                    part += f"+ v_{{{i}}}u_{{{i}}}"
            self.step_generator.add_step(
                f"\\text{{投影系数: }} c = \\boldsymbol{{v}} \\cdot "
                f"\\boldsymbol{{\\hat{{u}}}} = {part} = {latex(projection_coeff)}"
            )
            self.step_generator.add_step(
                f"\\text{{投影向量: }} \\boldsymbol{{p}} = c \\cdot "
                f"\\boldsymbol{{\\hat{{u}}}} = {latex(projection_coeff)} \\cdot {latex(u_unit)}"
            )
            self.add_vector(projection, "\\boldsymbol{p}")

        return projection

    def project_onto_subspace(self, vector: str | Matrix, subspace_basis: str | Matrix,
                              show_steps: bool = True) -> Matrix | None:
        """Project onto a subspace (using least squares method)

        Parameters:
            vector: Vector to be projected
            subspace_basis: Basis vectors of the subspace
            show_steps: Whether to show calculation steps.

        Returns:
            projection: Projection of vector onto the subspace
        """
        if show_steps:
            self.add_step("子空间投影(最小二乘法)")
            self.add_vector(vector, "\\boldsymbol{v}")
            self.add_matrix(subspace_basis, "A")

        A = self.parse_matrix_input(subspace_basis)
        b = self.parse_vector_input(vector)

        # Use normal equation: A^T A x = A^T b
        A_T = A.T
        ATA = A_T * A
        ATb = A_T * b

        if show_steps:
            self.add_matrix(A_T, "A^T")
            self.add_matrix(ATA, "A^TA")
            self.add_vector(ATb, "A^T\\boldsymbol{b}")

        # Check if A^T A is invertible
        if ATA.det() == 0:
            if show_steps:
                self.step_generator.add_step(
                    r"\text{警告: } A^TA \text{ 不可逆，使用伪逆}")

            # Use Gram-Schmidt orthogonalization
            Q, _ = self.gram_schmidt_orthogonalization(A, show_steps)

            # Project onto the orthogonal basis
            projection = Q * (Q.T * b)

            if show_steps:
                self.step_generator.add_step(
                    r"\text{使用正交基计算投影: } \\boldsymbol{p} = Q Q^T \\boldsymbol{b}"
                )
                self.add_vector(projection, "\\boldsymbol{p}")

            return projection

        # Solve normal equations
        try:
            x = ATA.inv() * ATb
            projection = A * x

            if show_steps:
                self.add_step("求解正规方程")
                self.step_generator.add_step(
                    r"A^TA \boldsymbol{x} = A^T\boldsymbol{b}")
                self.add_vector(x, "\\boldsymbol{x}")
                self.step_generator.add_step(
                    r"\boldsymbol{p} = A \boldsymbol{x}")
                self.add_vector(projection, "\\boldsymbol{p}")

            return projection

        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{求解失败: {str(e)}}}")
            return None

    def auto_project_vector(self, vector_input: str, subspace_input: str, show_steps: bool = True) -> Matrix | None:
        """Automatically project vector onto subspace

        Parameters:
            vector_input: Vector to be projected
            subspace_input: Basis vectors of the subspace
            show_steps: Whether to show calculation steps.

        Returns:
            projection: Projection of vector onto the subspace
        """
        self.step_generator.clear()

        if show_steps:
            self.step_generator.add_step(r"\textbf{向量投影求解}")

        vector = self.parse_vector_input(vector_input)
        subspace_basis = self.parse_matrix_input(subspace_input)

        if show_steps:
            self.add_vector(vector, "\\boldsymbol{v}")
            self.add_matrix(subspace_basis, "A")

        # Check dimension compatibility
        if vector.rows != subspace_basis.rows:
            if show_steps:
                self.step_generator.add_step(r"\text{错误: 向量和子空间基的维度不匹配}")
            return None

        # Analyze the subspace type
        subspace_type = self.check_subspace_type(subspace_basis, show_steps)

        # Choose projection method based on subspace type
        if subspace_type == "zero":
            if show_steps:
                self.step_generator.add_step(
                    r"\text{投影到零子空间, 结果为 } \boldsymbol{0}")
            projection = zeros(vector.rows, 1)

        elif subspace_type == "line":
            # One-dimensional subspace, use vector projection formula
            direction_vector = subspace_basis.col(0)
            projection = self.project_onto_line(
                vector, direction_vector, show_steps)

        elif subspace_type == "degenerate":
            if show_steps:
                self.step_generator.add_step(
                    r"\text{处理退化子空间, 使用 Gram-Schmidt 正交化}")

            # Use Gram-Schmidt to get the orthogonal basis
            Q, _ = self.gram_schmidt_orthogonalization(
                subspace_basis, show_steps)

            # Project onto the orthogonal basis
            projection_coeffs = Q.T * vector
            projection = Q * projection_coeffs

            if show_steps:
                self.add_step("投影计算")
                self.add_vector(projection_coeffs,
                                "\\boldsymbol{c} = Q^T \\boldsymbol{v}")
                self.step_generator.add_step(
                    r"\boldsymbol{p} = Q \boldsymbol{c}")
                self.add_vector(projection, "\\boldsymbol{p}")

        else:
            # Multi-dimensional subspace, use least squares method
            projection = self.project_onto_subspace(
                vector, subspace_basis, show_steps)

        if projection is None:
            if show_steps:
                self.step_generator.add_step(r"\text{投影计算失败}")
            return None

        # Final result
        if show_steps:
            self.add_step("最终结果")
            self.add_vector(projection, "\\boldsymbol{p}_{\\text{proj}}")

        return projection

    def project_using_least_squares(self, vector_input: str, subspace_input: str, show_steps: bool = True) -> Matrix:
        """Project using least squares method

        Parameters:
            vector_input: Vector to be projected
            subspace_input: Basis vectors of the subspace
            show_steps: Whether to show calculation steps.

        Returns:
            projection: Projection of vector onto the subspace
        """
        return self.project_onto_subspace(vector_input, subspace_input, show_steps)

# # Demo functions
# def demo_line_projection():
#     """Demonstrate line projection"""
#     solver = VectorProjectionSolver()

#     solver.step_generator.add_step(r"\textbf{直线投影演示}")

#     # Example 1: Project onto coordinate axis
#     v1 = '[3,4]'
#     line1 = '[1,0]'  # x-axis

#     solver.step_generator.add_step(r"\textbf{示例 1: 投影到 $x$ 轴}")
#     solver.auto_project_vector(v1, line1)
#     display(Math(solver.get_steps_latex()))

#     # Example 2: Project onto diagonal line
#     v2 = '[2,3]'
#     line2 = '[1,1]'  # 45-degree line

#     solver.step_generator.add_step(r"\textbf{示例 2: 投影到 45 度线}")
#     solver.auto_project_vector(v2, line2)
#     display(Math(solver.get_steps_latex()))


# def demo_plane_projection():
#     """Demonstrate plane projection"""
#     solver = VectorProjectionSolver()

#     solver.step_generator.add_step(r"\textbf{平面投影演示}")

#     # Example 1: Project onto xy-plane
#     v1 = '[1,2,3]'
#     plane1 = '[[1,0],[0,1],[0,0]]'  # xy-plane

#     solver.step_generator.add_step(r"\textbf{示例 1: 投影到 $xy$ 平面}")
#     solver.auto_project_vector(v1, plane1)
#     display(Math(solver.get_steps_latex()))

#     # Example 2: Project onto oblique plane
#     v2 = '[1,1,1]'
#     plane2 = '[[1,0],[0,1],[1,1]]'  # Oblique plane

#     solver.step_generator.add_step(r"\textbf{示例 2: 投影到斜平面}")
#     solver.auto_project_vector(v2, plane2)
#     display(Math(solver.get_steps_latex()))


# def demo_degenerate_subspace():
#     """Demonstrate degenerate subspace projection"""
#     solver = VectorProjectionSolver()

#     solver.step_generator.add_step(r"\textbf{退化子空间投影演示}")

#     # Example: Linearly dependent basis
#     v = '[1,2,3]'
#     # Second column is multiple of first
#     degenerate_basis = '[[1,2,3],[2,4,2],[3,6,1]]'

#     solver.step_generator.add_step(r"\textbf{示例: 线性相关基}")
#     solver.auto_project_vector(v, degenerate_basis)
#     display(Math(solver.get_steps_latex()))


# def demo_gram_schmidt_process():
#     """Demonstrate Gram-Schmidt process"""
#     solver = VectorProjectionSolver()

#     solver.step_generator.add_step(r"\textbf{Gram-Schmidt 正交化演示}")

#     # Example
#     basis = '[[1,1,0],[1,0,1],[0,1,1]]'

#     solver.step_generator.add_step(r"\textbf{正交化过程}")
#     solver.gram_schmidt_orthogonalization(basis)
#     display(Math(solver.get_steps_latex()))


# def demo_symbolic_projection():
#     """Demonstrate symbolic projection"""
#     solver = VectorProjectionSolver()

#     solver.step_generator.add_step(r"\textbf{符号投影演示}")
#     solver.step_generator.add_step(r"\textbf{假设所有符号表达式不为 0}")

#     # Symbolic example
#     v = '[a,b,c]'
#     subspace = '[[1,0],[0,d],[0,0]]'  # xy-plane

#     solver.step_generator.add_step(r"\textbf{符号向量投影到 $xy$ 平面}")
#     solver.auto_project_vector(v, subspace)
#     display(Math(solver.get_steps_latex()))


# if __name__ == "__main__":
#     demo_line_projection()
#     demo_plane_projection()
#     demo_degenerate_subspace()
#     demo_gram_schmidt_process()
#     demo_symbolic_projection()
