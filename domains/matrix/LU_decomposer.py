from sympy import Matrix, latex, zeros, eye, symbols, Symbol
from IPython.display import display, Math

from core import CommonMatrixCalculator


class LUDecomposition(CommonMatrixCalculator):

    def is_square(self, matrix):
        """检查是否为方阵"""
        return matrix.rows == matrix.cols

    def lu_decomposition_gaussian(self, matrix_input, show_steps=True):
        """
        角度一: 高斯消元法进行 LU 分解
        通过高斯消元过程得到 U, 同时记录消元系数得到 L
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{方法一: 高斯消元法}")

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("LU 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")

        # 初始化L和U矩阵
        L = eye(n)
        U = A.copy()

        if show_steps:
            self.add_step("初始化:")
            self.add_matrix(L, "L_0")
            self.add_matrix(U, "U_0")

        # 高斯消元过程
        for k in range(n-1):  # 主元列
            if show_steps:
                self.step_generator.add_step(f"\\text{{第 {k+1} 步消元:}}")

            # 检查主元是否为 0
            if U[k, k] == 0:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\textbf{{警告: 主元 }} u_{{{k+1}{k+1}}} \\textbf{{ = 0, 可能需要行交换或进行 PLU 分解}}")
                return None

            for i in range(k+1, n):  # 要消元的行
                # 计算消元系数
                factor = U[i, k] / U[k, k]
                L[i, k] = factor

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{计算消元系数: }} l_{{{i+1}{k+1}}} = \\frac{{u_{{{i+1}{k+1}}}^{{({k})}}}}{{u_{{{k+1}{k+1}}}^{{({k})}}}} = " +
                        f"\\frac{{{latex(U[i,k])}}}{{{latex(U[k,k])}}} = {latex(factor)}"
                    )

                # 执行行操作 - 为每个元素显示计算过程
                if show_steps:
                    self.step_generator.add_step(f"\\text{{更新第 {i+1} 行:}}")

                for j in range(k, n):
                    old_value = U[i, j]
                    new_value = U[i, j] - factor * U[k, j]
                    U[i, j] = new_value

                    if show_steps:
                        # 显示每个元素的详细计算过程
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}}^{{({k+1})}} = u_{{{i+1}{j+1}}}^{{({k})}} - l_{{{i+1}{k+1}}} \\cdot u_{{{k+1}{j+1}}}^{{({k})}} = " +
                            f"{latex(old_value)} - {latex(factor)} \\cdot {latex(U[k,j])} = {latex(new_value)}"
                        )

            if show_steps:
                self.add_step(f"第 {k+1} 步消元后:")
                self.add_matrix(L, f"L_{{{k+1}}}")
                self.add_matrix(U, f"U_{{{k+1}}}")

        # 最终验证分解结果
        if show_steps:
            self.add_step("最终结果:")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            self.add_step("最终验证:")
            self.step_generator.add_step(r"\text{验证: } L \times U = A")
            L_times_U = L * U
            self.add_matrix(L_times_U, "L \\times U")
            self.add_matrix(A, "A")

            if L_times_U == A:
                self.step_generator.add_step("分解正确")
            else:
                self.step_generator.add_step("分解错误")

        return L, U

    def lu_decomposition_doolittle(self, matrix_input, show_steps=True):
        """
        角度二：Doolittle直接分解法
        通过直接计算 L 和 U 的元素, 利用矩阵乘法规则
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{方法二: 假设形式, 待定系数法}")

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("LU 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")
            self.step_generator.add_step(
                r"\text{假设: } L = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ l_{21} & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ l_{n1} & l_{n2} & \cdots & 1 \end{bmatrix}, \quad U = \begin{bmatrix} u_{11} & u_{12} & \cdots & u_{1n} \\ 0 & u_{22} & \cdots & u_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & u_{nn} \end{bmatrix}")

        # 初始化L和U矩阵
        L = eye(n)
        U = zeros(n)

        if show_steps:
            # 创建初始符号矩阵
            L_symbolic = eye(n)
            U_symbolic = zeros(n)

            for i in range(n):
                for j in range(n):
                    if i > j:  # L的下三角部分（对角线下）
                        L_symbolic[i, j] = Symbol(f'l_{{{i+1}{j+1}}}')
                    elif i < j:  # U的上三角部分（对角线上）
                        U_symbolic[i, j] = Symbol(f'u_{{{i+1}{j+1}}}')
                    elif i == j:  # 对角线
                        U_symbolic[i, j] = Symbol(f'u_{{{i+1}{i+1}}}')

            self.add_step("初始化 L 和 U:")
            self.add_matrix(L_symbolic, "L_0")
            self.add_matrix(U_symbolic, "U_0")

        # Doolittle 算法
        for i in range(n):
            if show_steps and i < n-1:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 计算 L 的第 {i+1} 列和 U 的第 {i+1} 行}}")
            elif show_steps:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 计算 U 的第 {i+1} 行}}")

            # 计算U的第i行
            for j in range(i, n):
                sum_val = 0
                sum_terms = []
                for k in range(i):
                    product = L[i, k] * U[k, j]
                    sum_val += product
                    sum_terms.append(
                        f"l_{{{i+1}{k+1}}} \\cdot u_{{{k+1}{j+1}}}")

                U[i, j] = A[i, j] - sum_val

                if show_steps:
                    if sum_terms:
                        sum_expr = " + ".join(sum_terms)
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = a_{{{i+1}{j+1}}} - ({sum_expr}) = " +
                            f"{latex(A[i,j])} - ({latex(sum_val)}) = {latex(U[i,j])}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = a_{{{i+1}{j+1}}} = {latex(A[i,j])}")

            if U[i, i] == 0 and i < n-1:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\textbf{{警告: }} u_{{{i+1}{i+1}}} \\textbf{{= 0, 可能需要行交换或进行 PLU 分解}}")
                return None

            # 计算L的第i列 (i+1行开始)
            for j in range(i+1, n):
                sum_val = 0
                sum_terms = []
                for k in range(i):
                    product = L[j, k] * U[k, i]
                    sum_val += product
                    sum_terms.append(
                        f"l_{{{j+1}{k+1}}} \\cdot u_{{{k+1}{i+1}}}")

                L[j, i] = (A[j, i] - sum_val) / U[i, i]

                if show_steps:
                    if sum_terms:
                        sum_expr = " + ".join(sum_terms)
                        self.step_generator.add_step(
                            f"l_{{{j+1}{i+1}}} = \\frac{{a_{{{j+1}{i+1}}} - ({sum_expr})}}{{u_{{{i+1}{i+1}}}}} = " +
                            f"\\frac{{{latex(A[j,i])} - ({latex(sum_val)})}}{{{latex(U[i,i])}}} = {latex(L[j,i])}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"l_{{{j+1}{i+1}}} = \\frac{{a_{{{j+1}{i+1}}}}}{{u_{{{i+1}{i+1}}}}} = " +
                            f"\\frac{{{latex(A[j,i])}}}{{{latex(U[i,i])}}} = {latex(L[j,i])}"
                        )

            if show_steps and i < n-1:
                # 创建当前步骤的显示矩阵 - 完全基于实际计算值
                L_display = eye(n)
                U_display = zeros(n)

                # 填充已计算出的值
                for r in range(n):
                    for c in range(n):
                        if r <= i or (r > i and c <= i):  # L 中已计算的部分
                            L_display[r, c] = L[r, c]
                        else:  # L 中未计算的部分
                            if r > c:
                                L_display[r, c] = Symbol(f'l_{{{r+1}{c+1}}}')

                        if c <= i or (r <= i and c > i):  # U 中已计算的部分
                            U_display[r, c] = U[r, c]
                        else:  # U中未计算的部分
                            if r <= c:
                                U_display[r, c] = Symbol(f'u_{{{r+1}{c+1}}}')

                self.add_step(f"第 {i+1} 步后的 L 和 U:")
                self.add_matrix(L_display, f"L_{{{i+1}}}")
                self.add_matrix(U_display, f"U_{{{i+1}}}")

        # 验证分解结果
        if show_steps:
            self.add_step("最终结果:")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            self.step_generator.add_step(r"\text{验证: } L \times U = A")
            L_times_U = L * U
            self.add_matrix(L_times_U, "L \\times U")
            self.add_matrix(A, "A")

            if L_times_U == A:
                self.step_generator.add_step(r"\text{分解正确}")
            else:
                self.step_generator.add_step(r"\text{分解错误}")

        return L, U

    def lu_decomposition_crout(self, matrix_input, show_steps=True):
        """
        角度三：Crout分解法
        L 的对角线元素为1, U 的对角线元素需要计算
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{方法三: 另一种假设形式法}")

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("LU 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")
            self.step_generator.add_step(
                r"\text{假设: } L = \begin{bmatrix} l_{11} & 0 & \cdots & 0 \\ l_{21} & l_{22} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ l_{n1} & l_{n2} & \cdots & l_{nn} \end{bmatrix}, \quad U = \begin{bmatrix} 1 & u_{12} & \cdots & u_{1n} \\ 0 & 1 & \cdots & u_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}")

        # 初始化L和U矩阵
        L = zeros(n)
        U = eye(n)

        if show_steps:
            # 创建初始符号矩阵
            L_symbolic = zeros(n)
            U_symbolic = eye(n)

            for i in range(n):
                for j in range(n):
                    if i >= j:  # L的下三角部分（包括对角线）
                        L_symbolic[i, j] = Symbol(f'l_{{{i+1}{j+1}}}')
                    elif i < j:  # U的上三角部分（对角线上）
                        U_symbolic[i, j] = Symbol(f'u_{{{i+1}{j+1}}}')

            self.add_step("初始化 L 和 U:")
            self.add_matrix(L_symbolic, "L_0")
            self.add_matrix(U_symbolic, "U_0")

        # Crout算法
        for i in range(n):
            if show_steps and i < n-1:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 计算 L 的第 {i+1} 列和 U 的第 {i+1} 行}}")
            else:
                self.step_generator.add_step(
                    f"\\text{{第 {i+1} 步: 计算 L 的第 {i+1} 列}}")

            # 计算L的第i列
            for j in range(i, n):
                sum_val = 0
                sum_terms = []
                for k in range(i):
                    product = L[j, k] * U[k, i]
                    sum_val += product
                    sum_terms.append(
                        f"l_{{{j+1}{k+1}}} \\cdot u_{{{k+1}{i+1}}}")

                L[j, i] = A[j, i] - sum_val

                if show_steps:
                    if sum_terms:
                        sum_expr = " + ".join(sum_terms)
                        self.step_generator.add_step(
                            f"l_{{{j+1}{i+1}}} = a_{{{j+1}{i+1}}} - ({sum_expr}) = " +
                            f"{latex(A[j,i])} - ({latex(sum_val)}) = {latex(L[j,i])}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"l_{{{j+1}{i+1}}} = a_{{{j+1}{i+1}}} = {latex(A[j,i])}")

            if L[i, i] == 0:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\textbf{{警告: }} l_{{{i+1}{i+1}}} \\textbf{{= 0, 可能需要行交换或进行 PLU 分解}}")
                return None

            # 计算U的第i行(i+1列开始)
            for j in range(i+1, n):
                sum_val = 0
                sum_terms = []
                for k in range(i):
                    product = L[i, k] * U[k, j]
                    sum_val += product
                    sum_terms.append(
                        f"l_{{{i+1}{k+1}}} \\cdot u_{{{k+1}{j+1}}}")

                U[i, j] = (A[i, j] - sum_val) / L[i, i]

                if show_steps:
                    if sum_terms:
                        sum_expr = " + ".join(sum_terms)
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = \\frac{{a_{{{i+1}{j+1}}} - ({sum_expr})}}{{l_{{{i+1}{i+1}}}}} = " +
                            f"\\frac{{{latex(A[i,j])} - ({latex(sum_val)})}}{{{latex(L[i,i])}}} = {latex(U[i,j])}"
                        )
                    else:
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = \\frac{{a_{{{i+1}{j+1}}}}}{{l_{{{i+1}{i+1}}}}} = " +
                            f"\\frac{{{latex(A[i,j])}}}{{{latex(L[i,i])}}} = {latex(U[i,j])}"
                        )

            if show_steps and i < n-1:
                # 创建当前步骤的显示矩阵 - 修正：确保已计算的值都显示为数值
                L_display = zeros(n)
                U_display = eye(n)  # U的对角线始终为1

                # 填充L矩阵：已计算的部分显示数值，未计算的部分显示符号
                for r in range(n):
                    for c in range(n):
                        if r >= c:  # L 的下三角部分
                            if c <= i:  # 已计算的列（第0到第i列）
                                L_display[r, c] = L[r, c]  # 显示数值
                            else:  # 未计算的列
                                L_display[r, c] = Symbol(f'l_{{{r+1}{c+1}}}')

                        # U矩阵：已计算的部分显示数值，未计算的部分显示符号
                        if r < c:  # U的上三角部分（不包括对角线）
                            if r <= i and c <= n:  # 已计算的行（第0到第i行）
                                U_display[r, c] = U[r, c]  # 显示数值
                            else:  # 未计算的行
                                U_display[r, c] = Symbol(f'u_{{{r+1}{c+1}}}')

                self.add_step(f"第 {i+1} 步后的 L 和 U:")
                self.add_matrix(L_display, f"L_{{{i+1}}}")
                self.add_matrix(U_display, f"U_{{{i+1}}}")

        # 验证分解结果
        if show_steps:
            self.add_step("最终结果:")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            self.step_generator.add_step(r"\text{验证: } L \times U = A")
            L_times_U = L * U
            self.add_matrix(L_times_U, "L \\times U")
            self.add_matrix(A, "A")

            if L_times_U == A:
                self.step_generator.add_step(r"\text{分解正确}")
            else:
                self.step_generator.add_step(r"\text{分解错误}")

        return L, U

    def plu_decomposition(self, matrix_input, show_steps=True):
        """
        PLU 分解 - 带部分主元选择的高斯消元法
        返回 P, L, U 使得 PA = LU
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{PLU 分解 (带部分主元选择)}")

        A = self.parse_matrix_input(matrix_input)

        if not self.is_square(A):
            raise ValueError("PLU 分解只适用于方阵")

        n = A.rows

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(f"\\text{{矩阵维度: }} {n} \\times {n}")

        # 初始化矩阵
        P = eye(n)  # 置换矩阵
        L = eye(n)  # 单位下三角矩阵
        U = A.copy()  # 上三角矩阵

        if show_steps:
            self.add_step("初始化:")
            self.add_matrix(P, "P_0")
            self.add_matrix(L, "L_0")
            self.add_matrix(U, "U_0")

        # 记录行交换
        pivot_history = []

        # 高斯消元过程（带部分主元选择）
        for k in range(n-1):
            if show_steps:
                self.step_generator.add_step(f"\\text{{第 {k+1} 步消元:}}")

            # 寻找主元
            pivot_row = k
            max_val = abs(U[k, k])

            for i in range(k+1, n):
                if abs(U[i, k]) > max_val:
                    max_val = abs(U[i, k])
                    pivot_row = i

            # 如果需要行交换
            if pivot_row != k:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{行交换: 将第 {k+1} 行与第 {pivot_row+1} 行交换, 因为 }} |{latex(U[pivot_row, k])}| > |{latex(U[k, k])}|"
                    )

                # 交换 U 的行
                U.row_swap(k, pivot_row)

                # 交换 L 的行(只交换已计算的部分)
                for j in range(k):
                    L[k, j], L[pivot_row, j] = L[pivot_row, j], L[k, j]

                # 交换 P 的行
                P.row_swap(k, pivot_row)

                pivot_history.append((k, pivot_row))

                if show_steps:
                    self.add_step(f"行交换后:")
                    self.add_matrix(P, f"P_{{{k+1}}}")
                    self.add_matrix(L, f"L_{{{k+1}}}")
                    self.add_matrix(U, f"U_{{{k+1}}}")

            # 检查主元是否为0
            if U[k, k] == 0:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\textbf{{警告: 主元为 0, 矩阵可能奇异}}")
                return None

            # 消元过程
            for i in range(k+1, n):
                # 计算消元系数
                factor = U[i, k] / U[k, k]
                L[i, k] = factor

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{计算消元系数: }} l_{{{i+1}{k+1}}} = \\frac{{u_{{{i+1}{k+1}}}}}{{u_{{{k+1}{k+1}}}}} = " +
                        f"\\frac{{{latex(U[i,k])}}}{{{latex(U[k,k])}}} = {latex(factor)}"
                    )

                # 更新 U 的第 i 行
                if show_steps:
                    self.step_generator.add_step(f"\\text{{更新第 {i+1} 行:}}")

                for j in range(k, n):
                    old_value = U[i, j]
                    new_value = U[i, j] - factor * U[k, j]
                    U[i, j] = new_value

                    if show_steps and j == k:  # 只显示第一个元素的详细计算
                        self.step_generator.add_step(
                            f"u_{{{i+1}{j+1}}} = u_{{{i+1}{j+1}}} - l_{{{i+1}{k+1}}} \\cdot u_{{{k+1}{j+1}}} = " +
                            f"{latex(old_value)} - {latex(factor)} \\cdot {latex(U[k,j])} = {latex(new_value)}"
                        )

            if show_steps:
                self.add_step(f"第 {k+1} 步消元后:")
                self.add_matrix(L, f"L_{{{k+1}}}")
                self.add_matrix(U, f"U_{{{k+1}}}")

        # 最终结果
        if show_steps:
            self.add_step("最终结果:")
            self.add_matrix(P, "P")
            self.add_matrix(L, "L")
            self.add_matrix(U, "U")

            # 验证分解
            self.step_generator.add_step(r"\text{验证: }")
            P_times_A = P * A
            L_times_U = L * U
            self.add_matrix(P_times_A, "P \\times A")
            self.add_matrix(L_times_U, "L \\times U")

            if P_times_A == L_times_U:
                self.step_generator.add_step(r"\text{分解正确: } PA = LU")
            else:
                self.step_generator.add_step(r"\text{分解错误}")

            # 显示行交换历史
            if pivot_history:
                self.add_step("行交换历史:")
                for i, (old_row, new_row) in enumerate(pivot_history):
                    self.step_generator.add_step(
                        f"\\text{{步骤 {i+1}: 行 {old_row+1}}} \\leftrightarrow 行 \\text{{{new_row+1}}}")

        return P, L, U

    def check_lu_conditions(self, matrix_input, show_steps=True):
        """
        检查矩阵是否满足LU分解的条件
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{检查LU分解条件}")

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_matrix(A, "A")

        n = A.rows
        conditions_met = True

        # 检查是否为方阵
        if not self.is_square(A):
            if show_steps:
                self.step_generator.add_step(r"\text{矩阵不是方阵, 无法进行LU分解}")
            return False

        if show_steps:
            self.step_generator.add_step(r"\text{矩阵是方阵}")

        # 检查顺序主子式是否都不为0
        for k in range(1, n+1):
            submatrix = A[:k, :k]
            det = submatrix.det()

            if show_steps:
                self.add_matrix(submatrix, f"A_{{{k}}}")
                self.step_generator.add_step(
                    f"\\text{{主子式 }} \\det(A_{{{k}}}) = {latex(det)}")

            if det == 0:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{第 {k} 阶顺序主子式为 0, 可能需要行交换}}")
                conditions_met = False
            else:
                if show_steps:
                    self.step_generator.add_step(f"\\text{{第 {k} 阶顺序主子式不为 0}}")

        if conditions_met:
            if show_steps:
                self.step_generator.add_step(rf"\text{{矩阵满足 LU 分解条件(不需要行交换)}}")
        else:
            if show_steps:
                self.step_generator.add_step(
                    r"\text{矩阵不满足 LU 分解条件, 可能需要行交换或使用 PLU 分解}")

        return conditions_met

    def auto_lu_decomposition(self, matrix_input, show_steps=True):
        """
        自动选择 LU 或 PLU 分解
        根据矩阵条件自动判断是否需要行交换
        """
        if show_steps:
            self.step_generator.clear()
            self.step_generator.add_step(r"\textbf{自动 LU/PLU 分解}")

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_matrix(A, "A")

        # 检查条件
        can_do_lu = self.check_lu_conditions(matrix_input, show_steps=False)

        if can_do_lu:
            if show_steps:
                self.step_generator.add_step(r"\text{矩阵满足 LU 分解条件，使用标准 LU 分解}")
            return self.lu_decomposition_doolittle(matrix_input, show_steps)
        else:
            if show_steps:
                self.step_generator.add_step(r"\text{矩阵不满足 LU 分解条件，使用 PLU 分解}")
            return self.plu_decomposition(matrix_input, show_steps)


def demo():
    """演示LU分解"""
    lu = LUDecomposition()

    # 示例矩阵
    A1 = '[[2,1,1],[4,3,3],[8,7,9]]'
    A2 = '[[1,2,3],[2,5,7],[3,7,10]]'
    A3 = '[[2,4,6],[1,3,7],[1,1,1]]'

    # 高斯消元法
    try:
        lu.lu_decomposition_gaussian(A1)
        display(Math(lu.get_steps_latex()))
    except Exception as e:
        lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(lu.get_steps_latex()))

    # Doolittle 方法
    try:
        lu.lu_decomposition_doolittle(A2)
        display(Math(lu.get_steps_latex()))
    except Exception as e:
        lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(lu.get_steps_latex()))

    # Crout 方法
    try:
        lu.lu_decomposition_crout(A3)
        display(Math(lu.get_steps_latex()))
    except Exception as e:
        lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(lu.get_steps_latex()))


def demo_plu():
    """演示 PLU 分解"""
    lu = LUDecomposition()

    # 需要行交换的矩阵
    A_need_pivot_1 = '[[0,1,1],[1,1,1],[2,3,4]]'
    A_need_pivot_2 = '[[1,2,3],[4,5,6],[7,8,9]]'
    A_need_pivot_3 = '[[0,1,1],[4,3,3],[8,7,9]]'

    lu.step_generator.add_step(r"\textbf{PLU 分解演示}")

    cases = [A_need_pivot_1, A_need_pivot_2, A_need_pivot_3]

    for matrix in cases:
        try:
            lu.plu_decomposition(matrix)
            display(Math(lu.get_steps_latex()))
        except Exception as e:
            lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(lu.get_steps_latex()))


def demo_auto():
    """演示自动选择分解方法"""
    lu = LUDecomposition()

    # 各种测试矩阵
    test_matrices = [
        ("可 LU 分解的矩阵", '[[2,1,1],[4,3,3],[8,7,9]]'),
        ("需要 PLU 的矩阵", '[[0,1,1],[1,1,1],[2,3,4]]'),
        ("对角占优矩阵", '[[3,1,1],[1,4,2],[1,1,5]]')
    ]

    lu.step_generator.add_step(r"\textbf{自动 LU/PLU 分解演示}")

    for name, matrix in test_matrices:
        lu.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            lu.auto_lu_decomposition(matrix)
            display(Math(lu.get_steps_latex()))
        except Exception as e:
            lu.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
            display(Math(lu.get_steps_latex()))


def demo_symbolic():
    """演示符号矩阵的 LU 分解"""
    lu = LUDecomposition()

    a, b, c, d, e, f, g, h, i = symbols('a b c d e f g h i')

    A_sym = Matrix([[a, b, c], [d, e, f], [g, h, i]])

    lu.step_generator.add_step(r"\textbf{符号矩阵 LU 分解}")
    lu.step_generator.add_step(r"\textbf{假设所有符号表达式不为 0, 可作主元}")
    lu.add_matrix(A_sym, "A")

    try:
        lu.lu_decomposition_gaussian(A_sym)
        display(Math(lu.get_steps_latex()))
    except Exception as ex:
        lu.step_generator.add_step(f"\\text{{错误: }} {str(ex)}")
        display(Math(lu.get_steps_latex()))

    B_sym = Matrix([[a, b, c], [4, 5, 6], [g, h, i]])
    lu.lu_decomposition_gaussian(B_sym)
    display(Math(lu.get_steps_latex()))


def demo_special_cases():
    """演示特殊情况"""
    lu = LUDecomposition()

    # 对角矩阵
    diag = '[[2,0,0],[0,3,0],[0,0,5]]'

    # 三角矩阵
    triangular = '[[1,2,3],[0,4,5],[0,0,6]]'

    # 可能无法分解的矩阵
    problematic = '[[0,1],[1,0]]'

    lu.step_generator.add_step(r"\textbf{特殊情况演示}")

    cases = [
        ("对角矩阵", diag),
        ("上三角矩阵", triangular),
        ("可能无法分解的矩阵", problematic)
    ]

    for name, matrix in cases:
        display(Math(f"\\textbf{{{name}}}"))
        try:
            lu.lu_decomposition_crout(matrix)
            display(Math(lu.get_steps_latex()))
        except Exception as e:
            lu.step_generator.add_step(f"\\text{{分解失败: }} {str(e)}")
            display(Math(lu.get_steps_latex()))


if __name__ == "__main__":
    # 运行数值演示
    demo()
    # 运行 PLU 演示
    demo_plu()
    # 运行自动分解演示
    demo_auto()
    # 运行特殊情况演示
    demo_special_cases()
    # 运行符号演示
    demo_symbolic()
