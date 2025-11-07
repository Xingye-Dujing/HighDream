from sympy import Matrix, sympify, latex, zeros, eye, simplify, symbols
from IPython.display import display, Math

from domains.matrix import CommonStepGenerator


class LinearSystemSolver:

    def __init__(self):
        self.step_generator = CommonStepGenerator()

    def simplify_matrix(self, matrix):
        """
        对矩阵的每个元素进行化简
        """
        simplified_matrix = zeros(matrix.rows, matrix.cols)

        for i in range(matrix.rows):
            for j in range(matrix.cols):
                element = matrix[i, j]
                simplified_element = simplify(element)
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

    def parse_vector_input(self, vector_input):
        """解析向量输入"""
        try:
            if isinstance(vector_input, str):
                # 处理向量输入，如 '[1,2,3]' 或 '[[1],[2],[3]]'
                if vector_input.startswith('[[') and vector_input.endswith(']]'):
                    vector = Matrix(sympify(vector_input))
                else:
                    # 转换为列向量格式
                    vector_str = vector_input.strip('[]')
                    elements = [sympify(x.strip())
                                for x in vector_str.split(',')]
                    vector = Matrix(elements)
            else:
                vector = vector_input
            return vector
        except Exception as e:
            raise ValueError(f"无法解析向量输入: {vector_input}, 错误: {str(e)}") from e

    def add_step(self, title):
        """添加步骤标题"""
        self.step_generator.add_step(title)

    def add_matrix(self, matrix, name="M"):
        """添加矩阵"""
        self.step_generator.add_step(f"{name} = {latex(matrix)}")

    def add_vector(self, vector, name="v"):
        """添加向量"""
        self.step_generator.add_step(f"{name} = {latex(vector)}")

    def add_equation(self, equation):
        """添加方程"""
        self.step_generator.add_step(equation)

    def display_steps(self):
        """显示所有步骤"""
        display(Math(self.step_generator.get_steps_latex()))

    def get_steps_latex(self):
        return self.step_generator.get_steps_latex()

    def clear_steps(self):
        """清除所有步骤"""
        self.step_generator.clear()

    def display_system(self, A, b, variables=None):
        """显示线性方程组 Ax = b"""
        if variables is None:
            n = A.cols
            variables = [f'x_{i+1}' for i in range(n)]

        equations = []
        for i in range(A.rows):
            equation_terms = []
            for j in range(A.cols):
                if A[i, j] != 0:
                    if A[i, j] == 1:
                        equation_terms.append(f"{variables[j]}")
                    elif A[i, j] == -1:
                        equation_terms.append(f"-{variables[j]}")
                    else:
                        equation_terms.append(
                            f"{latex(A[i, j])}{variables[j]}")

            if equation_terms:
                equation_str = " + ".join(equation_terms).replace("+ -", "- ")
                equations.append(f"{equation_str} = {latex(b[i])}")
            else:
                equations.append(f"0 = {latex(b[i])}")

        for eq in equations:
            self.add_equation(eq)

    def is_square(self, matrix):
        """检查是否为方阵"""
        return matrix.rows == matrix.cols

    def check_system_type(self, A, b, show_steps=True):
        """检查线性方程组类型"""
        A = self.parse_matrix_input(A)
        b = self.parse_vector_input(b)
        m, n = A.rows, A.cols

        if show_steps:
            self.add_step("系统分析:")
            self.add_step(f"系数矩阵 A: {m} \\times {n}")
            self.add_matrix(A, "A")
            self.add_step(f"常数向量 \\boldsymbol{{b}}: {m} \\times 1")
            self.add_vector(b, "\\boldsymbol{b}")

        # 检查是否为方阵系统
        if m == n:
            det_A = A.det()
            if show_steps:
                self.add_step(f"\\det(A) = {latex(det_A)}")

            if det_A != 0:
                if show_steps:
                    self.add_step("系统有唯一解")
                return "unique_solution"
            else:
                # 检查秩来判断是有无穷多解还是无解
                rank_A = A.rank()
                augmented = A.row_join(b)
                rank_augmented = augmented.rank()

                if show_steps:
                    self.add_step(f"\\text{{rank}}(A) = {rank_A}")
                    self.add_step(f"\\text{{rank}}([A|b]) = {rank_augmented}")

                if rank_A == rank_augmented:
                    if show_steps:
                        self.add_step("系统有无穷多解")
                    return "singular_infinite"
                else:
                    if show_steps:
                        self.add_step("系统无解")
                    return "singular_no_solution"
        else:
            if m < n:
                if show_steps:
                    self.add_step("欠定系统: 方程数少于未知数, 通常有无穷多解")
                return "underdetermined"
            else:
                # 检查超定系统是否有解
                rank_A = A.rank()
                augmented = A.row_join(b)
                rank_augmented = augmented.rank()

                if show_steps:
                    self.add_step(f"\\text{{rank}}(A) = {rank_A}")
                    self.add_step(f"\\text{{rank}}([A|b]) = {rank_augmented}")

                if rank_A == rank_augmented:
                    if show_steps:
                        self.add_step("超定系统有精确解, 使用标准方法")
                    return "overdetermined_exact"
                else:
                    if show_steps:
                        self.add_step("超定系统无精确解, 使用最小二乘法")
                    return "overdetermined"

    def check_special_matrix(self, matrix):
        """检查特殊矩阵类型"""
        n = matrix.rows
        if n > matrix.cols:
            return "general"

        # 检查是否为单位矩阵
        if matrix == eye(n):
            return "identity"

        # 检查是否为对角矩阵
        is_diagonal = True
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] != 0:
                    is_diagonal = False
                    break
            if not is_diagonal:
                break
        if is_diagonal:
            return "diagonal"

        # 检查是否为置换矩阵(每行每列只有一个1, 其余为 0)
        is_permutation = True
        for i in range(n):
            row_ones = 0
            col_ones = 0
            for j in range(n):
                if matrix[i, j] not in [0, 1]:
                    is_permutation = False
                    break
                if matrix[i, j] == 1:
                    row_ones += 1
                if matrix[j, i] == 1:
                    col_ones += 1
            if row_ones != 1 or col_ones != 1:
                is_permutation = False
                break
        if is_permutation:
            return "permutation"

        # 检查是否为三角矩阵
        is_upper_triangular = True
        is_lower_triangular = True
        for i in range(n):
            for j in range(n):
                if i > j and matrix[i, j] != 0:
                    is_upper_triangular = False
                if i < j and matrix[i, j] != 0:
                    is_lower_triangular = False
        if is_upper_triangular:
            return "upper_triangular"
        if is_lower_triangular:
            return "lower_triangular"

        return "general"

    def solve_singular_system(self, A_input, b_input, show_steps=True):
        """处理奇异系统(方阵但行列式为0)"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        if show_steps:
            self.add_step("\\textbf{奇异系统求解: 方阵但行列式为0}")
            self.display_system(A, b)

        n = A.rows

        # 创建增广矩阵
        augmented = A.row_join(b)
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        # 高斯-约当消元得到简化行阶梯形
        rref_matrix, pivot_columns = augmented.rref()

        if show_steps:
            self.add_step("简化行阶梯形:")
            self.add_matrix(rref_matrix, "[A|\\boldsymbol{b}]_{rref}")

        # 检查是否有解
        has_solution = True
        for i in range(n):
            # 如果系数部分全为0但常数项不为0，则无解
            if all(rref_matrix[i, j] == 0 for j in range(n)) and rref_matrix[i, n] != 0:
                has_solution = False
                if show_steps:
                    self.add_step(
                        f"第 {i+1} 行: 0 = {latex(rref_matrix[i, n])} ≠ 0，系统无解")
                break

        if not has_solution:
            if show_steps:
                self.add_step("系统无解")
            return None

        # 识别主元列和自由变量列
        pivot_cols = []
        free_cols = []

        for j in range(n):
            is_pivot = False
            for i in range(n):
                if rref_matrix[i, j] == 1 and all(rref_matrix[i, k] == 0 for k in range(j)):
                    pivot_cols.append(j)
                    is_pivot = True
                    break
            if not is_pivot:
                free_cols.append(j)

        if show_steps:
            self.add_step(f"主元列: {[c+1 for c in pivot_cols]}")
            self.add_step(f"自由变量列: {[c+1 for c in free_cols]}")

        # 如果所有变量都是主元变量，说明有唯一解(虽然理论上奇异矩阵不应该有唯一解，但数值计算可能有误差)
        if len(free_cols) == 0:
            if show_steps:
                self.add_step("系统有唯一解")
            x = zeros(n, 1)
            for i in range(n):
                x[i] = rref_matrix[i, n]

            x_simplified = self.simplify_matrix(x)

            if show_steps:
                self.add_step("解:")
                self.add_vector(x_simplified, "\\boldsymbol{x}")
            return x_simplified

        # 创建自由变量符号
        free_vars = symbols(f't_1:{len(free_cols)+1}')

        # 构建解向量
        x = zeros(n, 1)

        # 为每个主元变量建立方程
        pivot_rows = []
        for i in range(n):
            if any(rref_matrix[i, j] != 0 for j in range(n)):
                pivot_rows.append(i)

        for i, row_idx in enumerate(pivot_rows):
            pivot_col = pivot_cols[i]
            x[pivot_col] = rref_matrix[row_idx, n]  # 特解部分

            # 减去自由变量的贡献
            for j, free_col in enumerate(free_cols):
                x[pivot_col] -= rref_matrix[row_idx, free_col] * free_vars[j]

        # 设置自由变量
        for j, free_col in enumerate(free_cols):
            x[free_col] = free_vars[j]

        if show_steps:
            self.add_step("通解:")
            self.add_equation(
                "\\boldsymbol{x} = \\boldsymbol{x_p} + \\sum t \\boldsymbol{h}")

            # 显示特解
            x_particular = zeros(n, 1)
            for i in range(n):
                if i in pivot_cols:
                    idx = pivot_cols.index(i)
                    x_particular[i] = rref_matrix[pivot_rows[idx], n]
                else:
                    x_particular[i] = 0

            self.add_vector(x_particular, "\\boldsymbol{x_p}")

            # 显示齐次解
            if free_cols:
                for j, free_col in enumerate(free_cols):
                    x_homogeneous = zeros(n, 1)
                    x_homogeneous[free_col] = 1
                    for i, pivot_col in enumerate(pivot_cols):
                        x_homogeneous[pivot_col] = - \
                            rref_matrix[pivot_rows[i], free_col]
                    self.add_vector(
                        x_homogeneous, f"\\boldsymbol{{h_{j+1}}}")

        if show_steps:
            self.add_vector(x, "\\boldsymbol{x}")

        return x

    def solve_underdetermined_system(self, A_input, b_input, show_steps=True):
        """处理欠定方程组(引入自由变量)"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        if show_steps:
            self.add_step("\\textbf{欠定系统求解: 引入自由变量}")
            self.display_system(A, b)

        m, n = A.rows, A.cols

        # 创建增广矩阵
        augmented = A.row_join(b)
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        # 高斯-约当消元得到简化行阶梯形
        rref_matrix, pivot_columns = augmented.rref()

        if show_steps:
            self.add_step("简化行阶梯形:")
            self.add_matrix(rref_matrix, "[A|\\boldsymbol{b}]_{rref}")

        # 检查是否有解
        for i in range(m):
            if all(rref_matrix[i, j] == 0 for j in range(n)) and rref_matrix[i, n] != 0:
                if show_steps:
                    self.add_step("系统无解")
                return None

        # 识别主元列和自由变量列
        pivot_cols = []
        free_cols = []

        for j in range(n):
            is_pivot = False
            for i in range(m):
                if rref_matrix[i, j] == 1 and all(rref_matrix[i, k] == 0 for k in range(j)):
                    pivot_cols.append(j)
                    is_pivot = True
                    break
            if not is_pivot:
                free_cols.append(j)

        if show_steps:
            self.add_step(f"主元列: {[c+1 for c in pivot_cols]}")
            self.add_step(f"自由变量列: {[c+1 for c in free_cols]}")

        # 创建自由变量符号
        free_vars = symbols(f't_1:{len(free_cols)+1}')

        # 构建解向量
        x = zeros(n, 1)

        for i, pivot_col in enumerate(pivot_cols):
            x[pivot_col] = rref_matrix[i, n]  # 特解部分

            # 减去自由变量的贡献
            for j, free_col in enumerate(free_cols):
                x[pivot_col] -= rref_matrix[i, free_col] * free_vars[j]

        for j, free_col in enumerate(free_cols):
            x[free_col] = free_vars[j]

        if show_steps:
            self.add_step("通解:")
            self.add_equation(
                "\\boldsymbol{x} = \\boldsymbol{x_p} + \\sum t \\boldsymbol{h}")

            # 显示特解和齐次解
            x_particular = zeros(n, 1)
            for i in range(n):
                if i in pivot_cols:
                    x_particular[i] = rref_matrix[pivot_cols.index(i), n]
                else:
                    x_particular[i] = 0

            self.add_vector(x_particular, "\\boldsymbol{x_p}")

            if free_cols:
                for j, free_col in enumerate(free_cols):
                    x_homogeneous = zeros(n, 1)
                    x_homogeneous[free_col] = 1
                    for i, pivot_col in enumerate(pivot_cols):
                        x_homogeneous[pivot_col] = -rref_matrix[i, free_col]
                    self.add_vector(
                        x_homogeneous, f"\\boldsymbol{{h_{j+1}}}")

        if show_steps:
            self.add_vector(x, "\\boldsymbol{x}")

        return x

    def solve_overdetermined_system(self, A_input, b_input, show_steps=True, simplify_result=True):
        """处理超定方程组(最小二乘法)"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        if show_steps:
            self.add_step("\\textbf{超定系统求解: 最小二乘法}")
            self.display_system(A, b)
            self.add_step("求解正规方程: A^TA\\boldsymbol{x} = A^T\\boldsymbol{b}")

        # 计算 A^T A 和 A^T b
        A_T = A.T
        ATA = A_T * A
        ATb = A_T * b

        if show_steps:
            self.add_matrix(A_T, "A^T")
            self.add_matrix(ATA, "A^TA")
            self.add_vector(ATb, "A^T\\boldsymbol{b}")

        # 检查 A^T A 是否可逆
        if ATA.det() == 0:
            if show_steps:
                self.add_step("警告: A^TA 不可逆，最小二乘解不唯一")
            return None

        # 求解正规方程
        try:
            x = ATA.inv() * ATb

            if simplify_result:
                x_simplified = self.simplify_matrix(x)
            else:
                x_simplified = x

            if show_steps:
                self.add_step("最小二乘解:")
                self.add_vector(x_simplified, "\\boldsymbol{x}")

                # 计算残差
                residual = A * x_simplified - b
                residual_norm = (residual.T * residual)[0]

                self.add_step("残差分析:")
                self.add_vector(
                    residual, "\\boldsymbol{r} = A\\boldsymbol{x} - \\boldsymbol{b}")
                self.add_step(
                    f"\\|\\boldsymbol{{r}}\\|^2 = {latex(residual_norm)}")

            return x_simplified

        except Exception as e:
            if show_steps:
                self.add_step(f"求解失败: {str(e)}")
            return None

    def solve_by_gaussian_elimination(self, A_input, b_input, show_steps=True):
        """方法一：高斯消元法"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # 检查系统类型
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            return self.solve_underdetermined_system(A, b, show_steps)
        elif system_type == "overdetermined":
            self.add_step("警告: 超定系统，高斯消元法可能无解")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法一: 高斯消元法}")
            self.display_system(A, b)

        m, n = A.rows, A.cols

        # 创建增广矩阵
        augmented = A.row_join(b)
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        # 高斯消元
        for i in range(min(m, n)):
            if show_steps:
                self.add_step(f"第 {i+1} 步: 处理第 {i+1} 列")

            # 寻找主元
            pivot_row = i
            for r in range(i, m):
                if augmented[r, i] != 0:
                    pivot_row = r
                    break

            # 行交换(如果需要)
            if pivot_row != i:
                if show_steps:
                    self.add_step(
                        f"行交换: R_{i+1} \\leftrightarrow  R_{pivot_row+1}")
                augmented.row_swap(i, pivot_row)
                if show_steps:
                    self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

            # 如果主元为 0, 跳过这一列
            if augmented[i, i] == 0:
                if show_steps:
                    self.add_step(f"第 {i+1} 列主元为 0，跳过")
                continue

            # 归一化主元行
            pivot = augmented[i, i]
            if pivot != 1:
                if show_steps:
                    self.add_step(f"归一化: R_{i+1} \\times {latex(1/pivot)}")
                augmented[i, :] = augmented[i, :] / pivot
                if show_steps:
                    self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

            # 消元其他行
            for j in range(i+1, m):
                if augmented[j, i] != 0:
                    factor = augmented[j, i]
                    if show_steps:
                        self.add_step(
                            f"消元: R_{j+1} - {latex(factor)} \\times R_{i+1}")
                    augmented[j, :] = augmented[j, :] - \
                        factor * augmented[i, :]
                    if show_steps:
                        self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        if show_steps:
            self.add_step("行阶梯形:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]_{ref}")

        # 回代求解
        x = zeros(n, 1)
        for i in range(min(m, n)-1, -1, -1):
            if augmented[i, i] == 0:
                continue

            sum_val = sum(augmented[i, j] * x[j] for j in range(i+1, n))
            x[i] = (augmented[i, n] - sum_val) / augmented[i, i]

            if show_steps:
                if i == min(m, n)-1:
                    self.add_step(
                        f"x_{{{i+1}}} = \\frac{{b_{{{i+1}}}}}{{A_{{{i+1}{i+1}}}}} = \\frac{{{latex(augmented[i, n])}}}{{{latex(augmented[i, i])}}} = {latex(x[i])}")
                else:
                    sum_terms = " + ".join(
                        [f"A_{{{i+1}{j+1}}} \\cdot x_{{{j+1}}}" for j in range(i+1, n)])
                    self.add_step(
                        f"x_{{{i+1}}} = \\frac{{b_{{{i+1}}} - ({sum_terms})}}{{A_{{{i+1}{i+1}}}}} = \\frac{{{latex(augmented[i, n])} - ({latex(sum_val)})}}{{{latex(augmented[i, i])}}} = {latex(x[i])}")

        # 化简结果
        x_simplified = self.simplify_matrix(x)

        if show_steps:
            self.add_step("最终解:")
            self.add_vector(x_simplified, "\\boldsymbol{x}")

            # 验证
            self.add_step("验证:")
            b_check = A * x_simplified
            b_check_simplified = self.simplify_matrix(b_check)
            self.add_vector(b_check_simplified, "A \\times \\boldsymbol{x}")
            self.add_vector(b, "\\boldsymbol{b}")
            if b_check_simplified == b:
                self.add_step(
                    "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
            else:
                self.add_step("验证失败")

        return x_simplified

    def solve_by_gauss_jordan(self, A_input, b_input, show_steps=True):
        """方法二：高斯-约当消元法"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # 检查系统类型
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            return self.solve_underdetermined_system(A, b, show_steps)
        elif system_type == "overdetermined":
            self.add_step("警告: 超定系统，高斯-约当消元法可能无解")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法二: 高斯-约当消元法}")
            self.display_system(A, b)

        m, n = A.rows, A.cols

        # 创建增广矩阵
        augmented = A.row_join(b)
        if show_steps:
            self.add_step("构造增广矩阵:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        # 高斯-约当消元
        for i in range(min(m, n)):
            if show_steps:
                self.add_step(f"第 {i+1} 步: 处理第 {i+1} 列")

            # 寻找主元
            pivot_row = i
            for r in range(i, m):
                if augmented[r, i] != 0:
                    pivot_row = r
                    break

            # 行交换(如果需要)
            if pivot_row != i:
                if show_steps:
                    self.add_step(
                        f"行交换: R_{i+1} \\leftrightarrow  R_{pivot_row+1}")
                augmented.row_swap(i, pivot_row)
                if show_steps:
                    self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

            # 如果主元为0，跳过这一列
            if augmented[i, i] == 0:
                if show_steps:
                    self.add_step(f"第 {i+1} 列主元为 0, 跳过")
                continue

            # 归一化主元行
            pivot = augmented[i, i]
            if pivot != 1:
                if show_steps:
                    self.add_step(f"归一化: R_{i+1} \\times {latex(1/pivot)}")
                augmented[i, :] = augmented[i, :] / pivot
                if show_steps:
                    self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

            # 消元其他行
            for j in range(m):
                if j != i and augmented[j, i] != 0:
                    factor = augmented[j, i]
                    if show_steps:
                        self.add_step(
                            f"消元: R_{j+1} - {latex(factor)} \\times R_{i+1}")
                    augmented[j, :] = augmented[j, :] - \
                        factor * augmented[i, :]
                    if show_steps:
                        self.add_matrix(augmented, "[A|\\boldsymbol{b}]")

        if show_steps:
            self.add_step("简化行阶梯形:")
            self.add_matrix(augmented, "[A|\\boldsymbol{b}]_{rref}")

        # 提取解
        x = zeros(n, 1)
        for i in range(min(m, n)):
            if augmented[i, i] != 0:
                x[i] = augmented[i, n]

        # 化简结果
        x_simplified = self.simplify_matrix(x)

        if show_steps:
            self.add_step("最终解:")
            self.add_vector(x_simplified, "\\boldsymbol{x}")

            # 验证
            self.add_step("验证:")
            b_check = self.simplify_matrix(A * x_simplified)
            self.add_vector(b_check, "A \\times \\boldsymbol{x}")
            self.add_vector(b, "\\boldsymbol{b}")
            if b.equals(b_check):
                self.add_step(
                    "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
            else:
                self.add_step("验证失败")

        return x_simplified

    def solve_by_matrix_inverse(self, A_input, b_input, show_steps=True):
        """方法三：矩阵求逆法"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # 检查系统类型
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            self.add_step("警告: 欠定系统，矩阵求逆法不适用")
            return self.solve_underdetermined_system(A, b, show_steps)
        elif system_type == "overdetermined":
            self.add_step("警告: 超定系统，矩阵求逆法不适用")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法三: 矩阵求逆法}")
            self.display_system(A, b)

        # 检查是否可逆
        if not self.is_square(A):
            if show_steps:
                self.add_step("矩阵不是方阵, 不能使用求逆法")
            return None

        det_A = A.det()
        if det_A == 0:
            if show_steps:
                self.add_step("矩阵行列式为 0, 不可逆")
            return None

        if show_steps:
            self.add_step("计算逆矩阵:")
            self.add_step(f"\\det(A) = {latex(det_A)}")

        # 计算逆矩阵
        try:
            A_inv = A.inv()
            A_inv_simplified = self.simplify_matrix(A_inv)

            if show_steps:
                self.add_matrix(A_inv_simplified, "A^{-1}")

            # 计算解
            x = A_inv_simplified * b
            x_simplified = self.simplify_matrix(x)

            if show_steps:
                self.add_step("计算解:")
                self.add_equation(
                    "\\boldsymbol{x} = A^{-1} \\cdot \\boldsymbol{b}")
                self.add_vector(x_simplified, "\\boldsymbol{x}")

                # 验证
                self.add_step("验证:")
                b_check = A * x_simplified
                b_check_simplified = self.simplify_matrix(b_check)
                self.add_vector(b_check_simplified,
                                "A \\times \\boldsymbol{x}")
                self.add_vector(b, "\\boldsymbol{b}")
                if b_check_simplified == b:
                    self.add_step(
                        "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
                else:
                    self.add_step("验证失败")

            return x_simplified

        except Exception as e:
            if show_steps:
                self.add_step(f"求逆失败: {str(e)}")
            return None

    def solve_by_lu_decomposition(self, A_input, b_input, show_steps=True):
        """方法四：LU 分解法"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # 检查系统类型
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            self.add_step("警告: 欠定系统，LU 分解法不适用")
            return self.solve_underdetermined_system(A, b, show_steps)
        elif system_type == "overdetermined":
            self.add_step("警告: 超定系统，LU 分解法不适用")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法四: LU 分解法}")
            self.display_system(A, b)

        if not self.is_square(A):
            if show_steps:
                self.add_step("矩阵不是方阵, 不能使用 LU 分解法")
            return None

        n = A.rows

        # 进行 LU 分解(使用 Doolittle 方法)
        L = eye(n)
        U = zeros(n)

        for i in range(n):
            # 计算 U 的第 i 行
            for j in range(i, n):
                sum_val = sum(L[i, k] * U[k, j] for k in range(i))
                U[i, j] = A[i, j] - sum_val

            # 计算 L 的第 i 列
            for j in range(i+1, n):
                sum_val = sum(L[j, k] * U[k, i] for k in range(i))
                if U[i, i] == 0:
                    if show_steps:
                        self.add_step("不能进行 LU 分解")
                    return None
                L[j, i] = (A[j, i] - sum_val) / U[i, i]

        if show_steps:
            self.add_step("LU 分解:")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            # 验证 LU 分解
            self.add_step("验证 LU 分解:")
            LU_product = L * U
            self.add_matrix(LU_product, "L \\times U")
            if LU_product == A:
                self.add_step("LU 分解正确")
            else:
                self.add_step("LU 分解错误")

        # 解方程组 L * y = b 和 U * x = y
        if show_steps:
            self.add_step("解方程组:")
            self.add_equation(
                "解: L \\cdot \\boldsymbol{y} = \\boldsymbol{b} \\quad \\text{和} \\quad U \\cdot \\boldsymbol{x} = \\boldsymbol{y}")

        # 前代法解 L * y = b
        y = zeros(n, 1)
        if show_steps:
            self.add_step(
                "(1) 前代法求解 L \\cdot \\boldsymbol{y} = \\boldsymbol{b}")

        for i in range(n):
            sum_val = sum(L[i, j] * y[j] for j in range(i))
            y[i] = (b[i] - sum_val) / L[i, i]

            if show_steps:
                if i == 0:
                    self.add_step(
                        f"y_{{{i+1}}} = \\frac{{b_{{{i+1}}}}}{{L_{{{i+1}{i+1}}}}} = \\frac{{{latex(b[i])}}}{{{latex(L[i, i])}}} = {latex(y[i])}")
                else:
                    sum_terms = " + ".join(
                        [f"L_{{{i+1}{j+1}}} \\cdot y_{{{j+1}}}" for j in range(i)])
                    self.add_step(
                        f"y_{{{i+1}}} = \\frac{{b_{{{i+1}}} - ({sum_terms})}}{{L_{{{i+1}{i+1}}}}} = \\frac{{{latex(b[i])} - ({latex(sum_val)})}}{{{latex(L[i, i])}}} = {latex(y[i])}")

        if show_steps:
            self.add_vector(y, "\\boldsymbol{y}")

        # 回代法解 U * x = y
        x = zeros(n, 1)
        if show_steps:
            self.add_step(
                "(2) 回代法求解 U \\cdot \\boldsymbol{x} = \\boldsymbol{y}")

        for i in range(n-1, -1, -1):
            sum_val = sum(U[i, j] * x[j] for j in range(i+1, n))
            x[i] = (y[i] - sum_val) / U[i, i]

            if show_steps:
                if i == n-1:
                    self.add_step(
                        f"x_{{{i+1}}} = \\frac{{y_{{{i+1}}}}}{{U_{{{i+1}{i+1}}}}} = \\frac{{{latex(y[i])}}}{{{latex(U[i, i])}}} = {latex(x[i])}")
                else:
                    sum_terms = " + ".join(
                        [f"U_{{{i+1}{j+1}}} \\cdot x_{{{j+1}}}" for j in range(i+1, n)])
                    self.add_step(
                        f"x_{{{i+1}}} = \\frac{{y_{{{i+1}}} - ({sum_terms})}}{{U_{{{i+1}{i+1}}}}} = \\frac{{{latex(y[i])} - ({latex(sum_val)})}}{{{latex(U[i, i])}}} = {latex(x[i])}")

        x_simplified = self.simplify_matrix(x)

        if show_steps:
            self.add_step("最终解:")
            self.add_vector(x_simplified, "\\boldsymbol{x}")

            # 验证
            self.add_step("验证:")
            b_check = A * x_simplified
            b_check_simplified = self.simplify_matrix(b_check)
            self.add_vector(b_check_simplified, "A \\times \\boldsymbol{x}")
            self.add_vector(b, "\\boldsymbol{b}")
            if b_check_simplified == b:
                self.add_step(
                    "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
            else:
                self.add_step("验证失败")

        return x_simplified

    def solve_by_cramers_rule(self, A_input, b_input, show_steps=True):
        """方法五：克莱姆法则"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # 检查系统类型
        system_type = self.check_system_type(A, b, False)

        if system_type == "underdetermined":
            self.add_step("警告: 欠定系统，克莱姆法则不适用")
            return self.solve_underdetermined_system(A, b, show_steps)
        elif system_type == "overdetermined":
            self.add_step("警告: 超定系统，克莱姆法则不适用")
            return self.solve_overdetermined_system(A, b, show_steps)

        if show_steps:
            self.add_step("\\textbf{方法五: 克莱姆法则}")
            self.display_system(A, b)

        if not self.is_square(A):
            if show_steps:
                self.add_step("矩阵不是方阵, 不能使用克莱姆法则")
            return None

        n = A.rows

        # 计算系数矩阵的行列式
        det_A = A.det()
        if show_steps:
            self.add_step(f"\\det(A) = {latex(det_A)}")

        if det_A == 0:
            if show_steps:
                self.add_step("行列式为 0, 克莱姆法则不适用")
            return None

        # 使用克莱姆法则求解
        x = zeros(n, 1)

        for i in range(n):
            # 创建替换矩阵
            A_i = A.copy()
            for j in range(n):
                A_i[j, i] = b[j]

            det_A_i = A_i.det()

            if show_steps:
                self.add_step(f"计算 x_{{{i+1}}}:")
                self.add_matrix(A_i, f"A_{{{i+1}}}")
                self.add_step(f"\\det(A_{{{i+1}}}) = {latex(det_A_i)}")
                self.add_step(
                    f"x_{{{i+1}}} = \\frac{{\\det(A_{{{i+1}}})}}{{\\det(A)}} = \\frac{{{latex(det_A_i)}}}{{{latex(det_A)}}}")

            x[i] = det_A_i / det_A

        x_simplified = self.simplify_matrix(x)

        if show_steps:
            self.add_step("最终解:")
            self.add_vector(x_simplified, "\\boldsymbol{x}")

            # 验证
            self.add_step("验证:")
            b_check = A * x_simplified
            b_check_simplified = self.simplify_matrix(b_check)
            self.add_vector(b_check_simplified, "A \\times \\boldsymbol{x}")
            self.add_vector(b, "\\boldsymbol{b}")
            if b_check_simplified == b:
                self.add_step(
                    "验证通过: A \\times \\boldsymbol{x} = \\boldsymbol{b}")
            else:
                self.add_step("验证失败")

        return x_simplified

    def solve(self, A_input, b_input, method='auto', show_steps=True):
        """主求解函数"""
        self.clear_steps()
        A = self.parse_matrix_input(A_input)
        b = self.parse_vector_input(b_input)

        # 自动选择方法
        if method == 'auto':
            system_type = self.check_system_type(A, b, show_steps)

            if system_type == "unique_solution":
                if A.rows <= 3:
                    method = 'gauss_jordan'
                else:
                    method = 'lu'
            elif system_type == "underdetermined":
                method = 'underdetermined'
            elif system_type == "overdetermined":
                method = 'overdetermined'
            elif system_type == "singular_infinite":
                method = 'singular'
            elif system_type == "singular_no_solution":
                return None
            else:
                method = 'gauss_jordan'

        # 根据选择的方法求解
        if method == 'gaussian':
            return self.solve_by_gaussian_elimination(A, b, show_steps)
        elif method == 'gauss_jordan':
            return self.solve_by_gauss_jordan(A, b, show_steps)
        elif method == 'inverse':
            return self.solve_by_matrix_inverse(A, b, show_steps)
        elif method == 'lu':
            return self.solve_by_lu_decomposition(A, b, show_steps)
        elif method == 'cramer':
            return self.solve_by_cramers_rule(A, b, show_steps)
        elif method == 'underdetermined':
            return self.solve_underdetermined_system(A, b, show_steps)
        elif method == 'overdetermined':
            return self.solve_overdetermined_system(A, b, show_steps)
        elif method == 'singular':
            return self.solve_singular_system(A, b, show_steps)
        else:
            raise ValueError(f"未知的求解方法: {method}")


# 演示函数
def demo_underdetermined_systems():
    """演示欠定方程组求解"""
    solver = LinearSystemSolver()

    # 欠定系统示例
    under_A1 = '[[1,2,3],[4,5,6]]'  # 2方程3未知数
    under_b1 = '[7,8]'

    under_A2 = '[[1,1,1,1],[0,1,1,1]]'  # 2方程4未知数
    under_b2 = '[5,3]'

    solver.add_step("\\textbf{欠定线性方程组求解演示}")

    under_systems = [(under_A1, under_b1), (under_A2, under_b2)]

    for i, (A, b) in enumerate(under_systems, 1):
        solver.add_step(f"\\textbf{{欠定系统示例 {i}}}")
        try:
            result = solver.solve_underdetermined_system(A, b)
            solver.display_steps()
            solver.clear_steps()
        except Exception as e:
            solver.add_step(f"\\text{{错误: }} {str(e)}")
            solver.display_steps()
            solver.clear_steps()


def demo_overdetermined_systems():
    """演示超定方程组求解"""
    solver = LinearSystemSolver()

    # 超定系统示例
    over_A1 = '[[1,2],[3,4],[5,6],[2,6]]'
    over_b1 = '[7,8,9,3]'

    over_A2 = '[[1,1],[2,1],[3,1]]'
    over_b2 = '[3,4,5]'

    solver.add_step("\\textbf{超定线性方程组求解演示}")

    over_systems = [(over_A1, over_b1), (over_A2, over_b2)]

    for i, (A, b) in enumerate(over_systems, 1):
        solver.add_step(f"\\textbf{{超定系统示例 {i}}}")
        try:
            result = solver.solve_overdetermined_system(A, b)
            if result is not None:
                solver.add_vector(result, "\\boldsymbol{x}")
            solver.display_steps()
            solver.clear_steps()
        except Exception as e:
            solver.add_step(f"\\text{{错误: }} {str(e)}")
            solver.display_steps()
            solver.clear_steps()


def demo_basic_systems():
    """演示基本线性方程组求解"""
    solver = LinearSystemSolver()

    # 可解系统示例
    A1 = '[[2,1],[1,3]]'
    b1 = '[5,10]'

    A2 = '[[1,2,3],[0,1,4],[5,6,0]]'
    b2 = '[14,7,8]'

    A3 = '[[1,1],[2,3]]'
    b3 = '[5,13]'

    solver.add_step("\\textbf{基本线性方程组求解演示}")

    test_systems = [(A1, b1), (A2, b2), (A3, b3)]

    for i, (A, b) in enumerate(test_systems, 1):
        solver.add_step(f"\\textbf{{示例 {i}}}")
        try:
            solver.solve(A, b, method='auto', show_steps=True)
            solver.display_steps()
            solver.clear_steps()
        except Exception as e:
            solver.add_step(f"\\text{{错误: }} {str(e)}")
            solver.display_steps()
            solver.clear_steps()


def demo_special_matrices():
    """演示特殊矩阵系统求解"""
    solver = LinearSystemSolver()

    # 特殊矩阵示例
    identity_A = '[[1,0,0],[0,1,0],[0,0,1]]'
    diagonal_A = '[[2,0,0],[0,3,0],[0,0,5]]'
    permutation_A = '[[0,1,0],[0,0,1],[1,0,0]]'
    upper_triangular_A = '[[1,2,3],[0,4,5],[0,0,6]]'
    lower_triangular_A = '[[1,0,0],[2,3,0],[4,5,6]]'

    b = '[1,2,3]'

    solver.add_step("\\textbf{特殊矩阵系统求解演示}")

    special_cases = [
        ("单位矩阵", identity_A, b),
        ("对角矩阵", diagonal_A, b),
        ("置换矩阵", permutation_A, b),
        ("上三角矩阵", upper_triangular_A, b),
        ("下三角矩阵", lower_triangular_A, b)
    ]

    for name, A, b in special_cases:
        solver.add_step(f"\\textbf{{{name}}}")
        try:
            # 使用高斯消元法演示特殊矩阵求解
            solver.solve_by_gaussian_elimination(A, b, show_steps=True)
            solver.display_steps()
            solver.clear_steps()
        except Exception as e:
            solver.add_step(f"\\text{{错误: }} {str(e)}")
            solver.display_steps()
            solver.clear_steps()


def demo_singular_systems():
    """演示奇异系统情况"""
    solver = LinearSystemSolver()

    # 奇异系统示例
    singular_A1 = '[[1,2,3],[4,5,6],[7,8,9]]'  # 行线性相关，无穷多解
    singular_b1 = '[1,2,3]'

    singular_A2 = '[[1,1],[1,1]]'  # 两行相同，无穷多解
    singular_b2 = '[2,2]'

    singular_A3 = '[[1,1],[1,1]]'  # 两行相同但常数项不同，无解
    singular_b3 = '[2,3]'

    singular_A4 = '[[1,2],[2,4]]'  # 第二行是第一行的倍数
    singular_b4 = '[1,2]'  # 有解

    singular_A5 = '[[1,2],[2,4]]'  # 第二行是第一行的倍数
    singular_b5 = '[1,3]'  # 无解

    solver.add_step("\\textbf{奇异系统演示}")

    singular_systems = [
        ("无穷多解示例1", singular_A1, singular_b1),
        ("无穷多解示例2", singular_A2, singular_b2),
        ("无解示例1", singular_A3, singular_b3),
        ("无穷多解示例3", singular_A4, singular_b4),
        ("无解示例2", singular_A5, singular_b5)
    ]

    for name, A, b in singular_systems:
        solver.add_step(f"\\textbf{{{name}}}")
        try:
            result = solver.solve(A, b, method='auto', show_steps=True)
            solver.display_steps()
            solver.clear_steps()
        except Exception as e:
            solver.add_step(f"\\text{{错误: }} {str(e)}")
            solver.display_steps()
            solver.clear_steps()


def demo_symbolic_systems():
    """演示符号系统求解"""
    solver = LinearSystemSolver()

    # 符号系统
    symbolic_A_2x2 = '[[a,b],[c,d]]'
    symbolic_b_2x2 = '[p,q]'

    symbolic_A_3x3 = '[[a,b,c],[d,e,f],[g,h,i]]'
    symbolic_b_3x3 = '[p,q,r]'

    solver.add_step("\\textbf{符号系统求解演示}")
    solver.add_step("\\textbf{假设所有符号表达式不为 0, 可作分母}")

    solver.add_step("\\textbf{2×2 符号系统}")
    try:
        solver.solve(symbolic_A_2x2, symbolic_b_2x2,
                     method='auto', show_steps=True)
        solver.display_steps()
        solver.clear_steps()
    except Exception as e:
        solver.add_step(f"\\text{{错误: }} {str(e)}")
        solver.display_steps()
        solver.clear_steps()

    solver.add_step("\\textbf{3×3 符号系统}")
    try:
        solver.solve(symbolic_A_3x3, symbolic_b_3x3,
                     method='auto', show_steps=True)
        solver.display_steps()
        solver.clear_steps()
    except Exception as e:
        solver.add_step(f"\\text{{错误: }} {str(e)}")
        solver.display_steps()
        solver.clear_steps()


def demo_all_methods():
    """演示所有求解方法"""
    solver = LinearSystemSolver()

    # 测试系统
    A = '[[2,1],[1,3]]'
    b = '[5,10]'

    methods = [
        ('gaussian', '高斯消元法'),
        ('gauss_jordan', '高斯-约当消元法'),
        ('inverse', '矩阵求逆法'),
        ('lu', 'LU分解法'),
        ('cramer', '克莱姆法则')
    ]

    solver.add_step("\\textbf{所有求解方法演示}")

    for method_key, method_name in methods:
        solver.add_step(f"\\textbf{{{method_name}}}")
        try:
            result = solver.solve(A, b, method=method_key, show_steps=True)
            solver.display_steps()
            solver.clear_steps()
        except Exception as e:
            solver.add_step(f"\\text{{错误: }} {str(e)}")
            solver.display_steps()
            solver.clear_steps()


def demo_auto_solve():
    """演示自动求解功能"""
    solver = LinearSystemSolver()

    # 各种类型的系统
    systems = [
        ("唯一解系统", '[[2,1],[1,3]]', '[5,10]'),
        ("欠定系统", '[[1,2,3],[4,5,6]]', '[7,8]'),
        ("超定系统", '[[1,2],[3,4],[5,6]]', '[7,8,9]'),
        ("上三角系统", '[[1,2,3],[0,4,5],[0,0,6]]', '[1,2,3]'),
        ("对角系统", '[[2,0,0],[0,3,0],[0,0,5]]', '[4,6,10]')
    ]

    solver.add_step("\\textbf{自动求解功能演示}")

    for name, A, b in systems:
        solver.add_step(f"\\textbf{{{name}}}")
        try:
            result = solver.solve(A, b, method='auto', show_steps=True)
            solver.display_steps()
            solver.clear_steps()
        except Exception as e:
            solver.add_step(f"\\text{{错误: }} {str(e)}")
            solver.display_steps()
            solver.clear_steps()


if __name__ == "__main__":
    # 运行所有演示
    demo_basic_systems()
    demo_special_matrices()
    demo_singular_systems()
    demo_symbolic_systems()
    demo_underdetermined_systems()
    demo_overdetermined_systems()
    demo_all_methods()
    demo_auto_solve()
