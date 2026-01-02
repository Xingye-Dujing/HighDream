from typing import List, Union
from sympy import Matrix, latex, simplify, solve, symbols, sympify, zeros
# from IPython.display import Math, display

from core import CommonMatrixCalculator


class LinearDependence(CommonMatrixCalculator):
    """A class for checking linear dependence of vector groups using various methods.

    This class provides multiple approaches to determine whether a set of vectors
    are linearly dependent or independent, including definition method, row reduction,
    determinant method, Gram determinant method, and linear combination method.
    """

    def parse_vector_input(self, vectors_input: Union[str, List, Matrix]) -> Matrix:
        """Parse vector input into a matrix format.

        Args:
            vectors_input (str, list, Matrix): Input vectors in various formats

        Returns:
            Matrix: A matrix where each column represents a vector

        Raises:
            ValueError: If the input cannot be parsed or vectors have inconsistent dimensions
        """
        try:
            if isinstance(vectors_input, str):
                # Handle string input like "[[1,2],[3,4],[5,6]]"
                matrix = Matrix(sympify(vectors_input))
            elif isinstance(vectors_input, list):
                # Handle list of vectors
                if all(isinstance(v, Matrix) for v in vectors_input):
                    # If input is a list of Matrix objects
                    if len(vectors_input) == 0:
                        return Matrix([])
                    # Check that all vectors have the same dimension
                    dim = vectors_input[0].rows
                    if any(v.rows != dim for v in vectors_input):
                        raise ValueError(
                            "All vectors must have the same dimension")
                    # Combine vectors into a matrix (each column is a vector)
                    matrix = zeros(dim, len(vectors_input))
                    for i, vec in enumerate(vectors_input):
                        for j in range(dim):
                            matrix[j, i] = vec[j]
                else:
                    # If it's a list of lists of numbers or symbols
                    vectors = [Matrix(v) for v in vectors_input]
                    dim = vectors[0].rows
                    if any(v.rows != dim for v in vectors):
                        raise ValueError(
                            "All vectors must have the same dimension")
                    matrix = zeros(dim, len(vectors))
                    for i, vec in enumerate(vectors):
                        for j in range(dim):
                            matrix[j, i] = vec[j]
            else:
                matrix = vectors_input
            return matrix
        except Exception as e:
            raise ValueError(
                f"Unable to parse vector input: {vectors_input}, Error: {str(e)}") from e

    def display_vectors(self, vectors: List, name: str = "v") -> None:
        """Display a group of vectors.

        Args:
            vectors (list): List of vector matrices
            name (str): Base name for the vectors (default: "v")
        """
        if len(vectors) == 0:
            self.step_generator.add_step(
                f"\\boldsymbol{{{name}}} = \\emptyset")
            return

        vector_strs = []
        for i, vec in enumerate(vectors):
            vector_strs.append(
                f"\\boldsymbol{{{name}_{{{i+1}}}}} = {latex(vec)}")

        vector_strs = rf',\;'.join(vector_strs)
        self.step_generator.add_step(f"\\text{{向量组: }}{vector_strs}")

    def check_special_cases(self, vectors_input: str, show_steps: bool = True, is_clear: bool = True) -> bool:
        """Check special cases for linear dependence.

        Args:
            vectors_input: Input vectors in any supported format
            show_steps (bool): Whether to show calculation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            bool or None: True if linearly dependent, False if independent,
                         None if no special case detected
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        m, n = A.rows, A.cols

        # Check for zero vectors
        zero_vectors = sum(1 for i in range(
            n) if all(A[j, i] == 0 for j in range(m)))
        if zero_vectors > 0:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{发现 {zero_vectors} 个零向量}}")
                self.step_generator.add_step(r"\text{包含零向量的向量组一定线性相关}")
            return True

        # Check for single vector
        if n == 1:
            if show_steps:
                self.step_generator.add_step(r"\text{单个向量}")
                if any(A[j, 0] != 0 for j in range(m)):
                    self.step_generator.add_step(r"\text{非零向量线性无关}")
                    return False

                self.step_generator.add_step(r"\text{零向量线性相关}")
                return True

        # Check if number of vectors exceeds dimension
        if n > m:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{向量个数 ({n}) > 向量维度 ({m})}}")
                self.step_generator.add_step(
                    r"\text{在 } \mathbb{R}^m \text{ 中, 当向量个数大于维度时一定线性相关}")
            return True

        # Check for standard basis vectors
        if m == n:
            is_standard_basis = True
            for i in range(m):
                for j in range(n):
                    if i == j and A[i, j] != 1:
                        is_standard_basis = False
                        break
                    if i != j and A[i, j] != 0:
                        is_standard_basis = False
                        break
                if not is_standard_basis:
                    break

            if is_standard_basis:
                if show_steps:
                    self.step_generator.add_step(r"\textbf{标准基向量}")
                    self.step_generator.add_step(r"\text{标准基向量线性无关}")
                return False

        return None

    def by_definition(self, vectors_input: str, show_steps: bool = True, is_clear: bool = True) -> bool:
        """Method 1: Definition approach (solving homogeneous equations).

        Args:
            vectors_input: Input vectors in any supported format
            show_steps (bool): Whether to show calculation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            bool: True if linearly dependent, False if independent
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法一: 定义法}")
            self.display_vectors(vectors)
            self.step_generator.add_step(
                r"\text{原理: 判断方程 } k_1 \boldsymbol{v_1} + k_2 \boldsymbol{v_2} + \cdots + k_n \boldsymbol{v_n} = \boldsymbol{0} \text{ 是否有非零解}")

        m, n = A.rows, A.cols

        # Construct homogeneous linear system
        if show_steps:
            self.add_step("构造齐次线性方程组:")
            equation = " + ".join(
                [f"k_{{{i+1}}} \\boldsymbol{{v_{{{i+1}}}}}" for i in range(n)]) + " = \\boldsymbol{0}"
            self.add_equation(equation)

            self.add_step("对应的系数矩阵:")
            self.add_matrix(A, "A")

        # Solve the system of equations
        try:
            # Use sympy to solve the equations
            k_symbols = symbols(f'k_1:{n+1}')
            equations = []

            for i in range(m):
                eq = 0
                for j in range(n):
                    eq += k_symbols[j] * A[i, j]
                equations.append(eq)

            solutions = solve(equations, k_symbols, dict=True)

            # Determine the nature of solutions
            only_trivial = True
            for sol in solutions:
                if any(v != 0 for v in sol.values()):
                    only_trivial = False
                    break

            if only_trivial:
                if show_steps:
                    self.step_generator.add_step(r"\text{方程组只有零解}")
                    self.step_generator.add_step(r"\text{结论: 向量组线性无关}")
                return False

            if show_steps:
                self.step_generator.add_step(r"\text{方程组有非零解:}")
                for sol in solutions:
                    sol_str = ", ".join(
                        [f"{latex(k)} = {latex(v)}" for k, v in sol.items()])
                    self.step_generator.add_step(sol_str)

                self.step_generator.add_step(r"\text{结论: 向量组线性相关}")
            return True

        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{解方程时出错: {str(e)}}}")
            # Fall back to another method
            return self.by_rref(vectors_input, show_steps, is_clear=False)

    def by_rref(self, vectors_input: str, show_steps: bool = True, is_clear: bool = True) -> bool:
        """Method 2: Row reduced echelon form approach.

        Args:
            vectors_input: Input vectors in any supported format
            show_steps (bool): Whether to show calculation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            bool: True if linearly dependent, False if independent
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法二: 行简化阶梯形法}")
            self.display_vectors(vectors)
            self.step_generator.add_step(
                r"\text{原理: 矩阵的秩 < 向量个数} \Leftrightarrow  \text{线性相关}")

        n = A.cols

        if show_steps:
            self.add_step("构造向量组的矩阵:")
            self.add_matrix(A, "A")

        # Calculate row reduced echelon form
        rref_matrix, pivot_columns = A.rref()

        if show_steps:
            self.add_step("行简化阶梯形矩阵:")
            self.add_matrix(rref_matrix, "A_{rref}")
            self.step_generator.add_step(
                f"\\text{{主元列位置: }} {[c+1 for c in pivot_columns]}")
            self.step_generator.add_step(
                f"\\text{{主元个数 (秩): }} {len(pivot_columns)}")
            self.step_generator.add_step(f"\\text{{向量个数: }} {n}")

        rank = len(pivot_columns)

        if rank < n:
            if show_steps:
                self.step_generator.add_step(
                    r"\text{矩阵的秩 < 向量个数} \Leftrightarrow  \text{线性相关}")
                self.step_generator.add_step(r"\text{结论: 向量组线性相关}")

                # Show linear relationships
                self.add_step("线性关系推导:")
                for j in range(n):
                    if j not in pivot_columns:
                        # This column can be expressed as a linear combination of pivot columns
                        relation = f"\\boldsymbol{{v_{{{j+1}}}}} = "
                        terms = []
                        for i, pivot_col in enumerate(pivot_columns):
                            coeff = rref_matrix[i, j]
                            if coeff != 0:
                                terms.append(
                                    f"{latex(coeff)} \\boldsymbol{{v_{{{pivot_col+1}}}}}")
                        if terms:
                            relation += " + ".join(terms)
                            self.add_equation(relation)
            return True

        if show_steps:
            self.step_generator.add_step(
                r"\text{秩 = 向量个数} \Rightarrow \text{线性无关}")
            self.step_generator.add_step(r"\text{结论: 向量组线性无关}")
        return False

    def by_determinant(self, vectors_input: str, show_steps: bool = True, is_clear: bool = True) -> bool:
        """Method 3: Determinant approach (only applicable to square matrices).

        Args:
            vectors_input: Input vectors in any supported format
            show_steps (bool): Whether to show calculation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            bool or None: True if linearly dependent, False if independent,
                         None if not applicable (non-square matrix)
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法三: 行列式法}")
            self.display_vectors(vectors)
            self.step_generator.add_step(
                r"\text{原理: 方阵行列式} \neq 0 \Leftrightarrow \text{线性无关}")

        m, n = A.rows, A.cols

        if m != n:
            if show_steps:
                self.step_generator.add_step(r"\text{不是方阵, 无法使用行列式法}")
            return None

        if show_steps:
            self.add_step("计算行列式:")
            self.add_matrix(A, "A")

        try:
            det_A = A.det()
            simplified_det = simplify(det_A)

            if show_steps:
                self.step_generator.add_step(
                    f"\\det(A) = {latex(simplified_det)}")

            if simplified_det == 0:
                if show_steps:
                    self.step_generator.add_step(
                        r"\det(A) = 0 \Rightarrow \text{线性相关}")
                    self.step_generator.add_step(r"\text{结论: 向量组线性相关}")
                return True

            if show_steps:
                self.step_generator.add_step(
                    r"\det(A) \neq 0 \Rightarrow \text{线性无关}")
                self.step_generator.add_step(r"\text{结论: 向量组线性无关}")
            return False

        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{计算行列式时出错: {str(e)}}}")
            return None

    def by_gram_determinant(self, vectors_input: str, show_steps: bool = True, is_clear: bool = True) -> bool:
        """Method 4: Gram determinant approach.

        Args:
            vectors_input: Input vectors in any supported format
            show_steps (bool): Whether to show calculation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            bool or None: True if linearly dependent, False if independent,
                         None if computation fails
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法四: Gram 行列式法}")
            self.display_vectors(vectors)
            self.step_generator.add_step(
                r"\text{原理: Gram 行列式} \neq 0 \Leftrightarrow \text{线性无关}")

        if show_steps:
            self.add_step("构造 Gram 矩阵:")
            self.add_equation(r"G = A^T A")

        # Calculate Gram matrix
        G = A.T * A

        if show_steps:
            self.add_matrix(G, "G")

        # Calculate Gram determinant
        try:
            det_G = G.det()
            simplified_det = simplify(det_G)

            if show_steps:
                self.step_generator.add_step(
                    f"\\det(G) = {latex(simplified_det)}")

            if simplified_det == 0:
                if show_steps:
                    self.step_generator.add_step(
                        r"\det(G) = 0 \Rightarrow \text{线性相关}")
                    self.step_generator.add_step(r"\text{结论: 向量组线性相关}")
                return True

            if show_steps:
                self.step_generator.add_step(
                    r"\det(G) \neq 0 \Rightarrow \text{线性无关}")
                self.step_generator.add_step(r"\text{结论: 向量组线性无关}")
            return False

        except Exception as e:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{计算 Gram 行列式时出错: {str(e)}}}")
            return None

    def by_linear_combination(self, vectors_input: str, show_steps: bool = True, is_clear: bool = True) -> bool:
        """Method 5: Linear combination approach.

        Args:
            vectors_input: Input vectors in any supported format
            show_steps (bool): Whether to show calculation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            bool: True if linearly dependent, False if independent
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法五: 线性组合法}")
            self.display_vectors(vectors)
            self.step_generator.add_step(r"\text{原理: 逐个检查每个向量是否能被前面的向量线性表示}")

        m, n = A.rows, A.cols

        independent_vectors = []
        relations = []

        for i in range(n):
            current_vector = vectors[i]

            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{检查向量 }} v_{{{i+1}}} = {latex(current_vector)}")

            if len(independent_vectors) == 0:
                # First vector - add if it's not a zero vector
                if any(current_vector[j] != 0 for j in range(m)):
                    independent_vectors.append(current_vector)
                    if show_steps:
                        self.step_generator.add_step(r"\text{加入独立向量集}")
                else:
                    if show_steps:
                        self.step_generator.add_step(r"\text{零向量, 线性相关}")
                    return True
            else:
                # Check if current vector can be linearly represented by previous independent vectors
                # Construct system of equations
                coeff_symbols = symbols(f'c_1:{len(independent_vectors)+1}')
                equations = []

                for j in range(m):
                    eq = -current_vector[j]
                    for k, vec in enumerate(independent_vectors):
                        eq += coeff_symbols[k] * vec[j]
                    equations.append(eq)

                try:
                    solutions = solve(equations, coeff_symbols, dict=True)

                    if solutions and any(sol != {s: 0 for s in coeff_symbols} for sol in solutions):
                        # Non-trivial solution exists, indicating linear dependence
                        relation = f"\\boldsymbol{{v_{{{i+1}}}}} = "
                        terms = []
                        for sol in solutions:
                            for k, coeff in sol.items():
                                if coeff != 0:
                                    idx = coeff_symbols.index(k)
                                    terms.append(
                                        f"{latex(coeff)} \\boldsymbol{{v_{{{independent_vectors.index(vectors[idx])+1}}}}}")
                            break  # Take only the first solution

                        if terms:
                            relation += " + ".join(terms)
                            relations.append(relation)

                        if show_steps:
                            self.step_generator.add_step(
                                r"\text{可以被前面的向量线性表示} \Rightarrow 线性相关")
                            if relations:
                                self.add_equation(relation)
                        return True

                    # Only trivial solution, indicating linear independence
                    independent_vectors.append(current_vector)
                    if show_steps:
                        self.step_generator.add_step(
                            r"\text{不能被前面的向量线性表示} \Rightarrow \text{线性无关, 加入独立向量集}")
                except Exception as e:
                    if show_steps:
                        self.step_generator.add_step(
                            f"\\text{{求解时出错: {str(e)}}}")
                    # If solving fails, conservatively assume linear independence
                    independent_vectors.append(current_vector)

        if show_steps:
            self.step_generator.add_step(r"\text{所有向量都线性无关}")
            self.step_generator.add_step(r"\text{结论: 向量组线性无关}")
        return False

    def auto_check_dependence(self, vectors_input: str, show_steps: bool = True, is_clear: bool = True) -> bool:
        """Automatically check linear dependence of vector groups.

        Args:
            vectors_input: Input vectors in any supported format
            show_steps (bool): Whether to show calculation steps
            is_clear (bool): Whether to clear previous steps

        Returns:
            bool or None: True if linearly dependent, False if independent,
                         None if unable to determine
        """
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.step_generator.add_step(r"\textbf{自动判断线性相关性}")
            self.display_vectors(vectors)

        # First check special cases
        special_result = self.check_special_cases(
            vectors_input, show_steps, is_clear=False)
        if special_result is not None:
            return special_result

        if show_steps:
            self.step_generator.add_step(r"\text{检测到一般情况, 使用多种方法判断}")

        results = {}

        # Method 1: Definition approach
        try:
            results["definition"] = self.by_definition(
                vectors_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{定义法失败: {str(e)}}}")

        # Method 2: Row reduced echelon form approach
        try:
            results["rref"] = self.by_rref(
                vectors_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{行简化阶梯形法失败: {str(e)}}}")

        # Method 3: Determinant approach (only for square matrices)
        if A.rows == A.cols:
            try:
                results["determinant"] = self.by_determinant(
                    vectors_input, show_steps, is_clear=False)
            except Exception as e:
                if show_steps:
                    self.step_generator.add_step(f"\\text{{行列式法失败: {str(e)}}}")

        # Method 4: Gram determinant approach
        try:
            results["gram"] = self.by_gram_determinant(
                vectors_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{Gram行列式法失败: {str(e)}}}")

        # Method 5: Linear combination approach
        try:
            results["combination"] = self.by_linear_combination(
                vectors_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{线性组合法失败: {str(e)}}}")

        # Return the most reliable result
        if "rref" in results:
            return results["rref"]
        if "definition" in results:
            return results["definition"]
        if results:
            return next(iter(results.values()))

        return None


# # Demo functions
# def demo_basic_vectors():
#     """Demonstrate linear dependence checking for basic vector groups."""
#     checker = LinearDependence()

#     # Examples of vector groups in various situations
#     independent_2d = '[[1,0],[0,1]]'  # 2D independent
#     dependent_2d = '[[1,2],[2,4]]'    # 2D dependent
#     independent_3d = '[[1,0,0],[0,1,0],[0,0,1]]'  # 3D independent
#     dependent_3d = '[[1,2,3],[2,4,6],[3,6,9]]'    # 3D dependent

#     checker.step_generator.add_step(r"\textbf{基本向量组线性相关性判断演示}")

#     test_vectors = [
#         ("二维线性无关向量组", independent_2d),
#         ("二维线性相关向量组", dependent_2d),
#         ("三维线性无关向量组", independent_3d),
#         ("三维线性相关向量组", dependent_3d)
#     ]

#     for name, vectors in test_vectors:
#         checker.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             result = checker.auto_check_dependence(vectors)
#             status = "线性相关" if result else "线性无关"
#             checker.step_generator.add_step(f"\\textbf{{最终结果: {status}}}")
#             display(Math(checker.get_steps_latex()))
#         except Exception as e:
#             checker.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(checker.get_steps_latex()))


# def demo_special_cases():
#     """Demonstrate special cases."""
#     checker = LinearDependence()

#     # Special case examples
#     zero_vector = '[[0,0]]'           # Zero vector
#     single_vector = '[[1,2]]'         # Single non-zero vector
#     excess_vectors = '[[1,0],[0,1],[1,1]]'  # More vectors than dimension

#     checker.step_generator.add_step(r"\textbf{特殊情况演示}")

#     special_cases = [
#         ("零向量", zero_vector),
#         ("单个非零向量", single_vector),
#         ("向量个数大于维度", excess_vectors)
#     ]

#     for name, vectors in special_cases:
#         checker.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             result = checker.auto_check_dependence(vectors)
#             status = "线性相关" if result else "线性无关"
#             checker.step_generator.add_step(f"\\textbf{{最终结果: {status}}}")
#             display(Math(checker.get_steps_latex()))
#         except Exception as e:
#             checker.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(checker.get_steps_latex()))


# def demo_symbolic_vectors():
#     """Demonstrate symbolic vectors."""
#     checker = LinearDependence()

#     # Symbolic vector examples
#     symbolic_2d = '[[a,b],[c,d]]'
#     symbolic_2d_independent = '[[a,b],[2*a,2*b]]'
#     symbolic_3d = '[[a,b,c],[d,e,f],[g,h,i]]'

#     display(Math(r"\textbf{符号向量线性相关性判断演示}"))
#     display(Math(r"\textbf{假设所有符号表达式不为 0}"))

#     symbolic_vectors = [
#         ("2×2 符号向量组", symbolic_2d),
#         ("2×2 符号向量组(线性有关)", symbolic_2d_independent),
#         ("3×3 符号向量组", symbolic_3d)
#     ]

#     for name, vectors in symbolic_vectors:
#         checker.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             result = checker.auto_check_dependence(vectors)
#             if result is None:
#                 checker.step_generator.add_step(r"\text{无法确定线性相关性}")
#             else:
#                 status = "线性相关" if result else "线性无关"
#                 checker.step_generator.add_step(f"\\textbf{{最终结果: {status}}}")
#             display(Math(checker.get_steps_latex()))
#         except Exception as e:
#             checker.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(checker.get_steps_latex()))


# def demo_high_dimensional():
#     """Demonstrate high-dimensional vectors."""
#     checker = LinearDependence()

#     # High-dimensional vector examples
#     high_dim_independent = '[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]'
#     high_dim_dependent = '[[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16]]'

#     checker.step_generator.add_step(r"\textbf{高维向量线性相关性判断演示}")

#     high_dim_vectors = [
#         ("四维线性无关向量组", high_dim_independent),
#         ("四维线性相关向量组", high_dim_dependent)
#     ]

#     for name, vectors in high_dim_vectors:
#         checker.step_generator.add_step(f"\\textbf{{{name}}}")
#         try:
#             result = checker.auto_check_dependence(vectors)
#             status = "线性相关" if result else "线性无关"
#             checker.step_generator.add_step(f"\\textbf{{最终结果: {status}}}")
#             display(Math(checker.get_steps_latex()))
#         except Exception as e:
#             checker.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
#             display(Math(checker.get_steps_latex()))


# if __name__ == "__main__":
#     demo_basic_vectors()
#     demo_special_cases()
#     demo_symbolic_vectors()
#     demo_high_dimensional()
