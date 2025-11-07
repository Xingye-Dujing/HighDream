from sympy import Matrix, sympify, latex, zeros, simplify, nsimplify, symbols, solve
from IPython.display import display, Math

from domains.matrix import CommonStepGenerator


class LinearDependence:

    def __init__(self):
        self.step_generator = CommonStepGenerator()

    def add_step(self, title):
        """显示步骤标题"""
        self.step_generator.add_step(
            f"\\text{{{title}}}")

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

    def parse_vector_input(self, vectors_input):
        """解析向量输入"""
        try:
            if isinstance(vectors_input, str):
                # 处理字符串输入，如 "[[1,2],[3,4],[5,6]]"
                matrix = Matrix(sympify(vectors_input))
            elif isinstance(vectors_input, list):
                # 处理向量列表
                if all(isinstance(v, Matrix) for v in vectors_input):
                    # 如果输入是Matrix对象列表
                    if len(vectors_input) == 0:
                        return Matrix([])
                    # 检查所有向量维度是否相同
                    dim = vectors_input[0].rows
                    if any(v.rows != dim for v in vectors_input):
                        raise ValueError("所有向量必须具有相同的维度")
                    # 将向量组合成矩阵（每列是一个向量）
                    matrix = zeros(dim, len(vectors_input))
                    for i, vec in enumerate(vectors_input):
                        for j in range(dim):
                            matrix[j, i] = vec[j]
                else:
                    # 如果是数字或符号列表的列表
                    vectors = [Matrix(v) for v in vectors_input]
                    dim = vectors[0].rows
                    if any(v.rows != dim for v in vectors):
                        raise ValueError("所有向量必须具有相同的维度")
                    matrix = zeros(dim, len(vectors))
                    for i, vec in enumerate(vectors):
                        for j in range(dim):
                            matrix[j, i] = vec[j]
            else:
                matrix = vectors_input
            return matrix
        except Exception as e:
            raise ValueError(f"无法解析向量输入: {vectors_input}, 错误: {str(e)}") from e

    def display_vectors(self, vectors, name="v"):
        """显示向量组"""
        if len(vectors) == 0:
            self.step_generator.add_step(
                f"\\boldsymbol{{{name}}} = \\emptyset")
            return

        vector_strs = []
        for i, vec in enumerate(vectors):
            vector_strs.append(
                f"\\boldsymbol{{{name}_{{{i+1}}}}} = {latex(vec)}")

        vector_strs = ',\;'.join(vector_strs)
        self.step_generator.add_step(f"\\text{{向量组: }}{vector_strs}")

    def check_special_cases(self, vectors_input, show_steps=True, is_clear=True):
        """检查特殊情况"""
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        m, n = A.rows, A.cols

        # 检查零向量
        zero_vectors = sum(1 for i in range(
            n) if all(A[j, i] == 0 for j in range(m)))
        if zero_vectors > 0:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{发现 {zero_vectors} 个零向量}}")
                self.step_generator.add_step(r"\text{包含零向量的向量组一定线性相关}")
            return True

        # 检查单个向量
        if n == 1:
            if show_steps:
                self.step_generator.add_step(r"\text{单个向量}")
                if any(A[j, 0] != 0 for j in range(m)):
                    self.step_generator.add_step(r"\text{非零向量线性无关}")
                    return False
                else:
                    self.step_generator.add_step(r"\text{零向量线性相关}")
                    return True

        # 检查向量个数大于维度
        if n > m:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{向量个数 ({n}) > 向量维度 ({m})}}")
                self.step_generator.add_step(
                    r"\text{在 } \mathbb{R}^m \text{ 中, 当向量个数大于维度时一定线性相关}")
            return True

        # 检查标准基向量
        if m == n:
            is_standard_basis = True
            for i in range(m):
                for j in range(n):
                    if i == j and A[i, j] != 1:
                        is_standard_basis = False
                        break
                    elif i != j and A[i, j] != 0:
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

    def by_definition(self, vectors_input, show_steps=True, is_clear=True):
        """方法一：定义法(解齐次方程)"""
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

        # 构造齐次线性方程组
        if show_steps:
            self.add_step("构造齐次线性方程组:")
            equation = " + ".join(
                [f"k_{{{i+1}}} \\boldsymbol{{v_{{{i+1}}}}}" for i in range(n)]) + " = \\boldsymbol{0}"
            self.add_equation(equation)

            self.add_step("对应的系数矩阵:")
            self.add_matrix(A, "A")

        # 解方程组
        try:
            # 使用 sympy 解方程组
            k_symbols = symbols(f'k_1:{n+1}')
            equations = []

            for i in range(m):
                eq = 0
                for j in range(n):
                    eq += k_symbols[j] * A[i, j]
                equations.append(eq)

            solutions = solve(equations, k_symbols, dict=True)

            # 判断解的情况
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
            else:
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
            # 回退到其他方法
            return self.by_rref(vectors_input, show_steps, is_clear=False)

    def by_rref(self, vectors_input, show_steps=True, is_clear=True):
        """方法二：行简化阶梯形法"""
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

        # 计算行简化阶梯形
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

                # 显示线性关系
                self.add_step("线性关系推导:")
                for j in range(n):
                    if j not in pivot_columns:
                        # 这一列可以由主元列线性表示
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
        else:
            if show_steps:
                self.step_generator.add_step(
                    r"\text{秩 = 向量个数} \Rightarrow \text{线性无关}")
                self.step_generator.add_step(r"\text{结论: 向量组线性无关}")
            return False

    def by_determinant(self, vectors_input, show_steps=True, is_clear=True):
        """方法三：行列式法(仅适用于方阵)"""
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
            else:
                if show_steps:
                    self.step_generator.add_step(
                        r"\det(A) \neq 0 \Rightarrow \text{线性无关}")
                    self.step_generator.add_step(r"\text{结论: 向量组线性无关}")
                return False

        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{计算行列式时出错: {str(e)}}}")
            return None

    def by_gram_determinant(self, vectors_input, show_steps=True, is_clear=True):
        """方法四：Gram 行列式法"""
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.step_generator.add_step(r"\textbf{方法四: Gram 行列式法}")
            self.display_vectors(vectors)
            self.step_generator.add_step(
                r"\text{原理: Gram 行列式} \neq 0 \Leftrightarrow \text{线性无关}")

        m, n = A.rows, A.cols

        if show_steps:
            self.add_step("构造 Gram 矩阵:")
            self.add_equation(r"G = A^T A")

        # 计算Gram矩阵
        G = A.T * A

        if show_steps:
            self.add_matrix(G, "G")

        # 计算 Gram 行列式
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
            else:
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

    def by_linear_combination(self, vectors_input, show_steps=True, is_clear=True):
        """方法五：线性组合法"""
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
                # 第一个向量，只要不是零向量就加入
                if any(current_vector[j] != 0 for j in range(m)):
                    independent_vectors.append(current_vector)
                    if show_steps:
                        self.step_generator.add_step(r"\text{加入独立向量集}")
                else:
                    if show_steps:
                        self.step_generator.add_step(r"\text{零向量, 线性相关}")
                    return True
            else:
                # 检查当前向量是否能被前面的独立向量线性表示
                # 构造方程组
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
                        # 有非零解, 说明线性相关
                        relation = f"\\boldsymbol{{v_{{{i+1}}}}} = "
                        terms = []
                        for sol in solutions:
                            for k, coeff in sol.items():
                                if coeff != 0:
                                    idx = coeff_symbols.index(k)
                                    terms.append(
                                        f"{latex(coeff)} \\boldsymbol{{v_{{{independent_vectors.index(vectors[idx])+1}}}}}")
                            break  # 只取第一个解

                        if terms:
                            relation += " + ".join(terms)
                            relations.append(relation)

                        if show_steps:
                            self.step_generator.add_step(
                                r"\text{可以被前面的向量线性表示} \Rightarrow 线性相关")
                            if relations:
                                self.add_equation(relation)
                        return True
                    else:
                        # 只有零解，说明线性无关
                        independent_vectors.append(current_vector)
                        if show_steps:
                            self.step_generator.add_step(
                                r"\text{不能被前面的向量线性表示} \Rightarrow \text{线性无关, 加入独立向量集}")
                except Exception as e:
                    if show_steps:
                        self.step_generator.add_step(
                            f"\\text{{求解时出错: {str(e)}}}")
                    # 如果求解失败, 保守地认为线性无关
                    independent_vectors.append(current_vector)

        if show_steps:
            self.step_generator.add_step(r"\text{所有向量都线性无关}")
            self.step_generator.add_step(r"\text{结论: 向量组线性无关}")
        return False

    def auto_check_dependence(self, vectors_input, show_steps=True, is_clear=True):
        """自动判断向量组的线性相关性"""
        if is_clear:
            self.step_generator.clear()

        A = self.parse_vector_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.step_generator.add_step(r"\textbf{自动判断线性相关性}")
            self.display_vectors(vectors)

        # 首先检查特殊情况
        special_result = self.check_special_cases(
            vectors_input, show_steps, is_clear=False)
        if special_result is not None:
            return special_result

        if show_steps:
            self.step_generator.add_step(r"\text{检测到一般情况, 使用多种方法判断}")

        results = {}

        # 方法 1: 定义法
        try:
            results["definition"] = self.by_definition(
                vectors_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{定义法失败: {str(e)}}}")

        # 方法 2: 行简化阶梯形法
        try:
            results["rref"] = self.by_rref(
                vectors_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{行简化阶梯形法失败: {str(e)}}}")

        # 方法 3: 行列式法(仅适用于方阵)
        if A.rows == A.cols:
            try:
                results["determinant"] = self.by_determinant(
                    vectors_input, show_steps, is_clear=False)
            except Exception as e:
                if show_steps:
                    self.step_generator.add_step(f"\\text{{行列式法失败: {str(e)}}}")

        # 方法 4: Gram 行列式法
        try:
            results["gram"] = self.by_gram_determinant(
                vectors_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{Gram行列式法失败: {str(e)}}}")

        # 方法 5: 线性组合法
        try:
            results["combination"] = self.by_linear_combination(
                vectors_input, show_steps, is_clear=False)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{线性组合法失败: {str(e)}}}")

        # 返回最可靠的结果
        if "rref" in results:
            return results["rref"]
        elif "definition" in results:
            return results["definition"]
        elif results:
            return next(iter(results.values()))

        return None


# 演示函数
def demo_basic_vectors():
    """演示基本向量组的线性相关性判断"""
    checker = LinearDependence()

    # 各种情况的向量组示例
    independent_2d = '[[1,0],[0,1]]'  # 二维无关
    dependent_2d = '[[1,2],[2,4]]'    # 二维相关
    independent_3d = '[[1,0,0],[0,1,0],[0,0,1]]'  # 三维无关
    dependent_3d = '[[1,2,3],[2,4,6],[3,6,9]]'    # 三维相关

    checker.step_generator.add_step(r"\textbf{基本向量组线性相关性判断演示}")

    test_vectors = [
        ("二维线性无关向量组", independent_2d),
        ("二维线性相关向量组", dependent_2d),
        ("三维线性无关向量组", independent_3d),
        ("三维线性相关向量组", dependent_3d)
    ]

    for name, vectors in test_vectors:
        checker.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            result = checker.auto_check_dependence(vectors)
            status = "线性相关" if result else "线性无关"
            checker.step_generator.add_step(f"\\textbf{{最终结果: {status}}}")
            display(Math(checker.get_steps_latex()))
        except Exception as e:
            checker.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(checker.get_steps_latex()))


def demo_special_cases():
    """演示特殊情况"""
    checker = LinearDependence()

    # 特殊情况示例
    zero_vector = '[[0,0]]'           # 零向量
    single_vector = '[[1,2]]'         # 单个非零向量
    excess_vectors = '[[1,0],[0,1],[1,1]]'  # 向量个数大于维度

    checker.step_generator.add_step(r"\textbf{特殊情况演示}")

    special_cases = [
        ("零向量", zero_vector),
        ("单个非零向量", single_vector),
        ("向量个数大于维度", excess_vectors)
    ]

    for name, vectors in special_cases:
        checker.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            result = checker.auto_check_dependence(vectors)
            status = "线性相关" if result else "线性无关"
            checker.step_generator.add_step(f"\\textbf{{最终结果: {status}}}")
            display(Math(checker.get_steps_latex()))
        except Exception as e:
            checker.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(checker.get_steps_latex()))


def demo_symbolic_vectors():
    """演示符号向量"""
    checker = LinearDependence()

    # 符号向量示例
    symbolic_2d = '[[a,b],[c,d]]'
    symbolic_2d_independent = '[[a,b],[2*a,2*b]]'
    symbolic_3d = '[[a,b,c],[d,e,f],[g,h,i]]'

    display(Math(r"\textbf{符号向量线性相关性判断演示}"))
    display(Math(r"\textbf{假设所有符号表达式不为 0}"))

    symbolic_vectors = [
        ("2×2 符号向量组", symbolic_2d),
        ("2×2 符号向量组(线性有关)", symbolic_2d_independent),
        ("3×3 符号向量组", symbolic_3d)
    ]

    for name, vectors in symbolic_vectors:
        checker.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            result = checker.auto_check_dependence(vectors)
            if result is None:
                checker.step_generator.add_step(r"\text{无法确定线性相关性}")
            else:
                status = "线性相关" if result else "线性无关"
                checker.step_generator.add_step(f"\\textbf{{最终结果: {status}}}")
            display(Math(checker.get_steps_latex()))
        except Exception as e:
            checker.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(checker.get_steps_latex()))


def demo_high_dimensional():
    """演示高维向量"""
    checker = LinearDependence()

    # 高维向量示例
    high_dim_independent = '[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]'
    high_dim_dependent = '[[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16]]'

    checker.step_generator.add_step(r"\textbf{高维向量线性相关性判断演示}")

    high_dim_vectors = [
        ("四维线性无关向量组", high_dim_independent),
        ("四维线性相关向量组", high_dim_dependent)
    ]

    for name, vectors in high_dim_vectors:
        checker.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            result = checker.auto_check_dependence(vectors)
            status = "线性相关" if result else "线性无关"
            checker.step_generator.add_step(f"\\textbf{{最终结果: {status}}}")
            display(Math(checker.get_steps_latex()))
        except Exception as e:
            checker.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(checker.get_steps_latex()))


if __name__ == "__main__":
    demo_basic_vectors()
    demo_special_cases()
    demo_symbolic_vectors()
    demo_high_dimensional()
