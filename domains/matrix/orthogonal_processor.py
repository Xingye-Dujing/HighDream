from sympy import latex, zeros, simplify, eye, Symbol
from IPython.display import display, Math

from core import CommonMatrixCalculator


class OrthogonalProcessor(CommonMatrixCalculator):

    def check_special_cases_orthogonal_set(self, vectors_input, show_steps=True):
        """检查正交集的特殊情况"""
        A = self.parse_matrix_input(vectors_input)

        if show_steps:
            self.add_step("特殊情况检查")

        # 检查零向量
        zero_vectors = []
        for i in range(A.cols):
            if all(A[j, i] == 0 for j in range(A.rows)):
                zero_vectors.append(i+1)

        if zero_vectors:
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{发现零向量: 第 {zero_vectors} 个向量}}")
                self.step_generator.add_step(r"\text{包含零向量的向量集不可能是正交集或规范正交集}")
            return "has_zero"

        # 检查单个向量
        if A.cols == 1:
            if show_steps:
                self.step_generator.add_step(r"\text{单个向量}")
                norm = A.col(0).norm()
                if norm == 1:
                    self.step_generator.add_step(r"\text{单位向量，是规范正交集}")
                    return "orthonormal_single"
                else:
                    self.step_generator.add_step(r"\text{非单位向量，是正交集但不是规范正交集}")
                    return "orthogonal_single"

        return None

    def is_orthogonal_set(self, vectors_input, show_steps=True, is_clear=True):
        """判断向量集是否为正交集"""
        if is_clear:
            self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{判断向量集是否为正交集}")

        A = self.parse_matrix_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(r"\text{原理: 任意两个不同向量的内积为 0}")

        # 检查特殊情况
        special_result = self.check_special_cases_orthogonal_set(
            vectors_input, show_steps)
        if special_result == "has_zero":
            return False
        elif special_result in ["orthogonal_single", "orthonormal_single"]:
            return True

        n = A.cols

        if show_steps:
            self.add_step("计算所有向量对的内积")

        all_orthogonal = True

        for i in range(n):
            for j in range(i+1, n):
                dot_product = vectors[i].dot(vectors[j])
                simplified_dot = simplify(dot_product)

                if show_steps:
                    self.step_generator.add_step(
                        f"\\boldsymbol{{v_{{{i+1}}}}} \\cdot \\boldsymbol{{v_{{{j+1}}}}} = {latex(simplified_dot)}"
                    )

                if simplified_dot != 0:
                    all_orthogonal = False
                    if show_steps:
                        self.step_generator.add_step(
                            f"\\text{{向量 {i+1} 和向量 {j+1} 不正交}}"
                        )

        if all_orthogonal:
            if show_steps:
                self.step_generator.add_step(r"\text{所有向量对都正交}")
                self.step_generator.add_step(r"\text{结论: 向量集是正交集}")
            return True
        else:
            if show_steps:
                self.step_generator.add_step(r"\text{存在不正交的向量对}")
                self.step_generator.add_step(r"\text{结论: 向量集不是正交集}")
            return False

    def is_orthonormal_set(self, vectors_input, show_steps=True, is_clear=True):
        """判断向量集是否为规范正交集"""
        if is_clear:
            self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{判断向量集是否为规范正交集}")

        A = self.parse_matrix_input(vectors_input)
        vectors = [A.col(i) for i in range(A.cols)]

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(r"\text{原理: 向量两两正交且每个向量的范数为 1}")

        # 检查特殊情况
        special_result = self.check_special_cases_orthogonal_set(
            vectors_input, show_steps)
        if special_result == "has_zero":
            return False
        elif special_result == "orthonormal_single":
            return True
        elif special_result == "orthogonal_single":
            return False

        # 首先检查是否正交
        if not self.is_orthogonal_set(vectors_input, False, False):
            if show_steps:
                self.step_generator.add_step(r"\text{向量集不正交，因此不是规范正交集}")
            return False

        if show_steps:
            self.add_step("检查每个向量的范数")

        all_unit_norm = True
        n = A.cols

        for i in range(n):
            norm_sq = vectors[i].dot(vectors[i])
            simplified_norm_sq = simplify(norm_sq)

            if show_steps:
                self.step_generator.add_step(
                    f"\\|\\boldsymbol{{v_{{{i+1}}}}}\\|^2 = \\boldsymbol{{v_{{{i+1}}}}} \\cdot \\boldsymbol{{v_{{{i+1}}}}} = {latex(simplified_norm_sq)}"
                )

            if simplified_norm_sq != 1:
                all_unit_norm = False
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{向量 {i+1} 的范数不为 1}}"
                    )

        if all_unit_norm:
            if show_steps:
                self.step_generator.add_step(r"\text{所有向量范数都为 1}")
                self.step_generator.add_step(r"\text{结论: 向量集是规范正交集}")
            return True
        else:
            if show_steps:
                self.step_generator.add_step(r"\text{存在范数不为 1 的向量}")
                self.step_generator.add_step(r"\text{结论: 向量集是正交集但不是规范正交集}")
            return False

    def is_orthogonal_matrix(self, matrix_input, show_steps=True, is_clear=True):
        """判断矩阵是否为正交矩阵"""
        if is_clear:
            self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{判断矩阵是否为正交矩阵}")

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(
                r"\text{原理: } A^T A = I \text{ 且 } A A^T = I")

        m, n = A.rows, A.cols

        # 检查是否为方阵
        if m != n:
            if show_steps:
                self.step_generator.add_step(r"\text{不是方阵，因此不是正交矩阵}")
            return False

        if show_steps:
            self.add_step("计算 $A^T A$")

        A_T = A.T
        ATA = A_T * A

        if show_steps:
            self.add_matrix(A_T, "A^T")
            self.add_matrix(ATA, "A^T A")

        # 检查 A^T A 是否等于单位矩阵
        I = eye(n)
        is_orthogonal = True

        for i in range(n):
            for j in range(n):
                element = simplify(ATA[i, j])
                expected = I[i, j]

                if element != expected:
                    is_orthogonal = False
                    if show_steps:
                        self.step_generator.add_step(
                            f"\\text{{位置 ({i+1},{j+1}): 得到 ${latex(element)}$, 期望 {latex(expected)}}}"
                        )

        if is_orthogonal:
            if show_steps:
                self.step_generator.add_step(r"A^T A = I")
                self.step_generator.add_step(r"\text{结论: 矩阵是正交矩阵}")
            return True
        else:
            if show_steps:
                self.step_generator.add_step(r"A^T A \neq I")
                self.step_generator.add_step(r"\text{结论: 矩阵不是正交矩阵}")
            return False

    def gram_schmidt_orthonormalization(self, vectors_input, show_steps=True, is_clear=True):
        """使用 Gram-Schmidt 过程构建规范正交基"""
        if is_clear:
            self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{Gram-Schmidt 规范正交化过程}")

        A = self.parse_matrix_input(vectors_input)

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(r"\text{原理: 逐步正交化并单位化向量}")

        m, n = A.rows, A.cols
        Q = zeros(m, n)  # 规范正交向量矩阵
        R = zeros(n, n)  # 上三角矩阵

        # 复制原始向量
        vectors = [A.col(i) for i in range(n)]
        orthogonal_vectors = []

        if show_steps:
            self.add_step("开始 Gram-Schmidt 过程")

        for i in range(n):
            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{步骤 {i+1}: 处理向量 }} \\boldsymbol{{a_{i+1}}}")
                if i == 0:
                    self.add_vector(
                        vectors[i], f"v_{1} = a_{1}")
                else:
                    self.add_vector(vectors[i], f"a_{i+1}")

            # 开始正交化
            v = vectors[i].copy()

            if show_steps and i > 0:
                self.step_generator.add_step(r"\text{减去在前面的正交向量上的投影:}")

            projection_terms = []
            for j in range(i):
                # 计算投影系数
                r_ji = vectors[i].dot(orthogonal_vectors[j])
                R[j, i] = r_ji

                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{在 }}\\boldsymbol{{q_{j+1}}}\\text{{ 上的投影系数: }} r_{{{j+1}{i+1}}} = \\boldsymbol{{a_{i+1}}} \\cdot \\boldsymbol{{q_{j+1}}} = {latex(r_ji)}"
                    )
                    self.step_generator.add_step(
                        f"\\text{{投影分量: }} {latex(r_ji)} \\cdot \\boldsymbol{{q_{j+1}}} = {latex(r_ji * orthogonal_vectors[j])}"
                    )

                # 减去投影
                v = v - r_ji * orthogonal_vectors[j]
                projection_terms.append(
                    f"{latex(r_ji)}\\boldsymbol{{q_{j+1}}}")

            if show_steps and i > 0:
                projection_str = " - ".join(projection_terms)
                self.step_generator.add_step(
                    f"\\boldsymbol{{v_{i+1}}} = \\boldsymbol{{a_{i+1}}} - ({projection_str})")
                self.add_vector(v, f"v_{i+1}")

            # 计算范数
            norm_v = v.norm()
            R[i, i] = norm_v

            if show_steps:
                self.step_generator.add_step(
                    f"\\text{{正交向量的范数: }} \\|\\boldsymbol{{v_{i+1}}}\\| = {latex(norm_v)}")

            # 如果范数含有符号, 则假定其为正值(不为 0)
            if norm_v.has(Symbol) or norm_v > 0:
                # 单位化
                q_i = v / norm_v
                orthogonal_vectors.append(q_i)

                # 存储到 Q 矩阵
                for k in range(m):
                    Q[k, i] = q_i[k]

                if show_steps:
                    self.step_generator.add_step(r"\text{单位化: }")
                    self.step_generator.add_step(
                        f"\\boldsymbol{{q_{i+1}}} = \\frac{{\\boldsymbol{{v_{i+1}}}}}{{ \\|\\boldsymbol{{v_{i+1}}}\\| }} = \\frac{{ {latex(v)} }}{{ {latex(norm_v)} }}")
                    self.add_vector(q_i, f"q_{i+1}")
            else:
                if show_steps:
                    self.step_generator.add_step(
                        f"\\text{{警告: 第 {i+1} 个向量与前面向量线性相关，无法添加到规范正交基}}")
                # 对于线性相关的情况，我们添加零向量
                orthogonal_vectors.append(zeros(m, 1))

        if show_steps:
            self.add_step("Gram-Schmidt 过程完成")
            # 化简矩阵
            Q = self.simplify_matrix(Q)
            R = self.simplify_matrix(R)
            self.add_matrix(Q, "Q")
            self.add_matrix(R, "R")

            # 验证结果
            self.add_step("验证结果")
            if self.is_orthonormal_set(Q, False, False):
                self.step_generator.add_step(r"\text{生成的向量集是规范正交集}")
            else:
                self.step_generator.add_step(r"\text{警告: 生成的向量集不是规范正交集}")

        return Q, R

    def qr_decomposition(self, matrix_input, show_steps=True, is_clear=True):
        """矩阵的 QR 分解"""
        if is_clear:
            self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{矩阵的 QR 分解}")

        A = self.parse_matrix_input(matrix_input)

        if show_steps:
            self.add_matrix(A, "A")
            self.step_generator.add_step(
                r"\text{原理: } A = QR \text{, 其中 } Q \text{ 是正交矩阵, } R \text{ 是上三角矩阵}")

        m, n = A.rows, A.cols

        # 使用 Gram-Schmidt 过程
        if show_steps:
            self.step_generator.add_step(r"\text{使用 Gram-Schmidt 过程进行 QR 分解}")

        Q, R = self.gram_schmidt_orthonormalization(
            matrix_input, show_steps, False)

        if show_steps:
            self.add_step("验证分解结果")

            # 计算 Q * R
            QR = Q * R
            self.add_matrix(QR, "QR")

            # 检查是否等于 A
            is_correct = True
            for i in range(m):
                for j in range(n):
                    if simplify(QR[i, j] - A[i, j]) != 0:
                        is_correct = False
                        break
                if not is_correct:
                    break

            if is_correct:
                self.step_generator.add_step(r"QR = A \Rightarrow \text{分解正确}")
            else:
                self.step_generator.add_step(
                    r"QR \neq A \Rightarrow \text{分解可能有误}")

            # 检查 Q 是否正交
            if m == n:  # 只有方阵才能是正交矩阵
                if self.is_orthogonal_matrix(Q, False, False):
                    self.step_generator.add_step(r"Q \text{ 是正交矩阵}")
                else:
                    self.step_generator.add_step(r"Q \text{ 不是正交矩阵}")
            else:
                if self.is_orthonormal_set(Q, False, False):
                    self.step_generator.add_step(r"Q \text{ 的列是规范正交的}")
                else:
                    self.step_generator.add_step(r"Q \text{ 的列不是规范正交的}")

            # 检查 R 是否上三角
            is_upper_triangular = True
            for i in range(R.rows):
                for j in range(i):
                    if R[i, j] != 0:
                        is_upper_triangular = False
                        break
                if not is_upper_triangular:
                    break

            if is_upper_triangular:
                self.step_generator.add_step(r"R \text{ 是上三角矩阵}")
            else:
                self.step_generator.add_step(r"R \text{ 不是上三角矩阵}")

        return Q, R

    def auto_orthogonal_analysis(self, input_data, show_steps=True):
        """自动进行正交性分析"""
        self.step_generator.clear()
        if show_steps:
            self.step_generator.add_step(r"\textbf{自动正交性分析}")

        try:
            A = self.parse_matrix_input(input_data)
        except Exception:
            # 可能是单个向量
            A = self.parse_vector_input(input_data)
            A = A.T  # 转换为行向量以便统一处理

        if show_steps:
            self.add_matrix(A, "A")

        results = {}

        # 分析向量集的正交性
        if show_steps:
            self.add_step("向量集正交性分析")

        is_orthogonal = self.is_orthogonal_set(A, show_steps, False)
        is_orthonormal = self.is_orthonormal_set(A, show_steps, False)

        results["orthogonal_set"] = is_orthogonal
        results["orthonormal_set"] = is_orthonormal

        # 分析矩阵的正交性(如果是方阵)
        if A.rows == A.cols:
            if show_steps:
                self.add_step("矩阵正交性分析")

            is_orthogonal_matrix = self.is_orthogonal_matrix(
                A, show_steps, False)
            results["orthogonal_matrix"] = is_orthogonal_matrix

        # 进行 QR 分解
        if show_steps:
            self.add_step("QR 分解")

        try:
            Q, R = self.qr_decomposition(A, show_steps, False)
            results["qr_decomposition"] = (Q, R)
        except Exception as e:
            if show_steps:
                self.step_generator.add_step(f"\\text{{QR 分解失败: {str(e)}}}")
            results["qr_decomposition"] = None

        return results


# 演示函数
def demo_orthogonal_sets():
    """演示正交集判断"""
    processor = OrthogonalProcessor()

    processor.step_generator.add_step(r"\textbf{正交集判断演示}")

    # 各种情况的示例
    orthogonal_set = '[[1,0],[0,1]]'  # 规范正交集
    orthogonal_not_normal = '[[2,0],[0,3]]'  # 正交但不规范
    not_orthogonal = '[[1,1,2],[1,0,1],[3,1,2]]'  # 不正交
    single_vector = '[[1,0]]'  # 单个向量
    zero_vector = '[[0,0],[1,0]]'  # 包含零向量

    test_cases = [
        ("规范正交集", orthogonal_set),
        ("正交但不规范集", orthogonal_not_normal),
        ("非正交集", not_orthogonal),
        ("单个向量", single_vector),
        ("包含零向量的集", zero_vector)
    ]

    for name, vectors in test_cases:
        processor.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            processor.auto_orthogonal_analysis(vectors)
        except Exception as e:
            processor.step_generator.add_step(f"\\text{{错误: }} {str(e)}")

        display(Math(processor.get_steps_latex()))


def demo_orthogonal_matrices():
    """演示正交矩阵判断"""
    processor = OrthogonalProcessor()

    processor.step_generator.add_step(r"\textbf{正交矩阵判断演示}")

    # 各种情况的示例
    identity = '[[1,0],[0,1]]'  # 单位矩阵
    rotation = '[[0,-1],[1,0]]'  # 旋转矩阵
    not_orthogonal = '[[1,2],[3,4]]'  # 非正交矩阵
    reflection = '[[1,0],[0,-1]]'  # 反射矩阵

    test_cases = [
        ("单位矩阵", identity),
        ("旋转矩阵", rotation),
        ("非正交矩阵", not_orthogonal),
        ("反射矩阵", reflection)
    ]

    for name, matrix in test_cases:
        processor.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            processor.is_orthogonal_matrix(matrix)
        except Exception as e:
            processor.step_generator.add_step(f"\\text{{错误: }} {str(e)}")

        display(Math(processor.get_steps_latex()))


def demo_gram_schmidt():
    """演示 Gram-Schmidt 过程"""
    processor = OrthogonalProcessor()

    processor.step_generator.add_step(r"\textbf{Gram-Schmidt 规范正交化演示}")

    # 各种情况的示例
    independent_2d = '[[1,1],[2,0]]'
    independent_3d = '[[1,1,0],[1,0,1],[0,1,1]]'
    dependent_vectors = '[[1,2],[2,4],[3,6]]'  # 线性相关

    test_cases = [
        ("二维独立向量", independent_2d),
        ("三维独立向量", independent_3d),
        ("线性相关向量", dependent_vectors)
    ]

    for name, vectors in test_cases:
        processor.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            processor.gram_schmidt_orthonormalization(vectors)
        except Exception as e:
            processor.step_generator.add_step(f"\\text{{错误: }} {str(e)}")
        display(Math(processor.get_steps_latex()))


def demo_qr_decomposition():
    """演示 QR 分解"""
    processor = OrthogonalProcessor()

    processor.step_generator.add_step(r"\textbf{QR 分解演示}")

    # 各种情况的示例
    square_matrix = '[[1,1],[2,0]]'
    rectangular = '[[1,2,3],[4,5,6]]'
    symmetric = '[[2,1],[1,2]]'

    test_cases = [
        ("方阵", square_matrix),
        ("矩形矩阵", rectangular),
        ("对称矩阵", symmetric)
    ]

    for name, matrix in test_cases:
        processor.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            processor.qr_decomposition(matrix)
        except Exception as e:
            processor.step_generator.add_step(f"\\text{{错误: }} {str(e)}")

        display(Math(processor.get_steps_latex()))


def demo_symbolic_cases():
    """演示符号情况"""
    processor = OrthogonalProcessor()

    display(Math(r"\textbf{符号情况演示}"))
    display(Math(r"\textbf{假设所有符号表达式各自满足一定的条件}"))

    # 符号示例
    symbolic_orthogonal = '[[a,0],[0,b]]'
    symbolic_matrix = '[[a,b],[c,d]]'

    test_cases = [
        ("符号正交集", symbolic_orthogonal),
        ("符号矩阵", symbolic_matrix)
    ]

    for name, data in test_cases:
        processor.step_generator.add_step(f"\\textbf{{{name}}}")
        try:
            processor.auto_orthogonal_analysis(data)
        except Exception as e:
            processor.step_generator.add_step(f"\\text{{错误: }} {str(e)}")

        display(Math(processor.get_steps_latex()))


if __name__ == "__main__":
    demo_orthogonal_sets()
    demo_orthogonal_matrices()
    demo_gram_schmidt()
    demo_qr_decomposition()
    demo_symbolic_cases()
