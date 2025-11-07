from sympy import Matrix, simplify, sympify, latex
from domains.matrix import RefStepGenerator


class DeterminantCalculator:

    def __init__(self):
        self.step_generator = RefStepGenerator()

    def _reset(self):
        self.step_generator.reset()

    def _parse_matrix_input(self, matrix_input: str) -> Matrix:
        try:
            M = Matrix(sympify(matrix_input))
            if M.shape[0] != M.shape[1]:  # 矩阵必须为方阵
                raise ValueError("矩阵必须为方阵")
            return M
        except Exception as e:
            raise ValueError("无法解析矩阵输入, 请使用格式如 '[[1,2],[3,4]]'") from e

    def _add_section_title(self, title: str):
        # 换行
        self.step_generator.add_step("", "")
        self.step_generator.add_step(title, "")
        self.step_generator.add_step("", "")

    def _best_expand_axis(self, M: Matrix):
        n = M.shape[0]
        min_nonzeros = n + 1
        best = ("row", 0)
        for i in range(n):
            cnt = sum(1 for x in M.row(i) if simplify(x) != 0)
            if cnt < min_nonzeros:
                min_nonzeros = cnt
                best = ("row", i)
        for j in range(n):
            cnt = sum(1 for x in M.col(j) if simplify(x) != 0)
            if cnt < min_nonzeros:
                min_nonzeros = cnt
                best = ("col", j)
        return best

    def _method_laplace(self, A: Matrix):
        # 处理 1x1 与 2x2 的直接结束情形
        n = A.shape[0]
        original_n = n
        if n == 1:
            val = simplify(A[0, 0])
            self.step_generator.add_step(val, rf"$|1×1\,行列式|=元素本身$")
            return val
        if n == 2:
            res = simplify(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
            self.step_generator.add_step(res, rf"$2×2\,行列式$")
            return res

        # 把一个 (coeff, obj) 项格式化成 latex(obj 可能是 Matrix 或 scalar)
        def term_to_latex(term):
            coeff, obj = term
            # 忽略系数为 0 的项
            if coeff == 0:
                return None
            if isinstance(obj, Matrix):
                # obj 是子矩阵: 显示 coeff · det(obj)
                if coeff == 1:
                    return f"{latex(obj)}"
                elif coeff == -1:
                    return f"-{latex(obj)}"
                else:
                    return f"{latex(coeff)} \\cdot {latex(obj)}"
            else:
                # obj 是标量
                if coeff == 1:
                    return latex(obj)
                elif coeff == -1:
                    return f"-{latex(obj)}"
                else:
                    return f"{latex(sympify(coeff * obj))}"

        # 构造表达式的 latex(把各项用 + 拼起来, 并把 '+-' 替换为 '-')
        def expr_to_latex(terms):
            parts = []
            for t in terms:
                s = term_to_latex(t)
                if s:
                    parts.append(s)
            if not parts:
                return "0"
            expr = "+".join(parts)
            expr = expr.replace("+-", "-")
            expr = expr.replace("--", "+")
            return expr

        # active_terms 保存当前表达式的项列表
        # 项为 (coeff, obj), 表示 coeff * det(obj)
        active_terms = [(1, A)]

        # 主循环: 每次找到第一个需要展开的项并展开替换, 记录替换前后表达式
        while True:
            idx_to_expand = None
            # 找到第一个需要展开的项
            for idx, (coeff, obj) in enumerate(active_terms):
                if isinstance(obj, Matrix) and obj.shape[0] > 1 and simplify(coeff) != 0:
                    idx_to_expand = idx
                    break

            # 都没有需要展开的矩阵项, 跳出
            if idx_to_expand is None:
                break

            coeff, M = active_terms[idx_to_expand]
            axis, ind = self._best_expand_axis(M)
            expand_desc = rf"按第\,{ind+1}\,{'行' if axis == 'row' else '列'}展开"

            n = M.shape[0]
            new_terms = []
            for j in range(n):
                if axis == "row":
                    a = simplify(M[ind, j])
                    if a == 0:
                        continue
                    s = (-1) ** (ind + j)
                    minor = M.minor_submatrix(ind, j)
                else:
                    a = simplify(M[j, ind])
                    if a == 0:
                        continue
                    s = (-1) ** (j + ind)
                    minor = M.minor_submatrix(j, ind)

                new_coeff = simplify(coeff * s * a)

                # 若余子式为 1x1 或 2x2, 直接算出标量以便能马上替换为标量项
                if minor.shape[0] == 1:
                    scalar = simplify(minor[0, 0])
                    new_terms.append((new_coeff, scalar))
                # 3x3 矩阵还是要显示余子式展开
                elif minor.shape[0] == 2 and original_n != 3:
                    det2 = simplify(
                        minor[0, 0] * minor[1, 1] - minor[0, 1] * minor[1, 0])
                    new_terms.append((new_coeff, det2))
                else:
                    new_terms.append((new_coeff, minor))

            # 在 active_terms 中用 new_terms 替换第 idx_to_expand 项
            active_terms = active_terms[:idx_to_expand] + \
                new_terms + active_terms[idx_to_expand + 1:]
            new_expr = expr_to_latex(active_terms)

            # 记录这一步的连等式
            self.step_generator.add_step(
                rf"$\;\Rightarrow\;{new_expr}$", expand_desc)
            self.step_generator.add_step("", "")

        final_sum = simplify(sum(simplify(c * o) for c, o in active_terms))
        self.step_generator.add_step(
            rf'$\;\Rightarrow\;{latex(final_sum)}$', "最终结果")

        return final_sum

    # Sarrus(3x3)
    def _method_sarrus(self, A: Matrix):
        if A.shape != (3, 3):
            return None
        a, b, c = A[0, 0], A[0, 1], A[0, 2]
        d, e, f = A[1, 0], A[1, 1], A[1, 2]
        g, h, i = A[2, 0], A[2, 1], A[2, 2]
        pos = simplify(a * e * i + b * f * g + c * d * h)
        neg = simplify(c * e * g + a * f * h + b * d * i)
        res = simplify(pos - neg)
        self.step_generator.add_step(
            pos, rf"$主和\,=\,{latex(a)} \times {latex(e)} \times {latex(i)}+{latex(b)} \times {latex(f)} \times {latex(g)}+{latex(c)} \times {latex(d)} \times {latex(h)}$")
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            neg, rf"$副和\,=\,{latex(c)} \times {latex(e)} \times {latex(g)}+{latex(a)} \times {latex(f)} \times {latex(h)}+{latex(b)} \times {latex(d)} \times {latex(i)}$")
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            res, rf"$结果\,=\,主和 -\,副和\,=\,{latex(res)}$")
        return res

    def _method_lu(self, A: Matrix):
        try:
            L, U, perm = A.LUdecomposition()
        except Exception:
            self.step_generator.add_step(A.copy(), "LU 分解失败或不可用")
            return None

        # 构造置换矩阵 P
        P = Matrix.eye(A.rows)
        for (i, j) in perm:
            P.row_swap(i, j)
        # 计算置换次数(行交换次数)
        swap_count = len(perm)
        # 每次行交换行列式符号翻转
        det_P = (-1) ** swap_count

        # 计算最终行列式：|A| = |P^-1| * |L| * |U| = |P| * |L| * |U|
        det_val = simplify(det_P * L.det() * U.det())

        # 步骤记录
        if perm:
            # 有行交换
            self.step_generator.add_step(
                P.copy(), rf"$P(置换矩阵),行交换次数={swap_count}.\;|P| = (-1)^{{{swap_count}}} = {det_P}$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(L.copy(), "$L(下三角矩阵)$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(U.copy(), "$U(上三角矩阵)$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                det_val,
                rf"$|A|=|P| \cdot |L| \cdot |U|={det_P} \times {latex(L.det())} \times {latex(U.det())}={latex(det_val)}$"
            )
        else:
            # 无行交换
            self.step_generator.add_step(L.copy(), "$L(下三角矩阵)$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(U.copy(), "$U(上三角矩阵)$")
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                det_val,
                f"$|A|=|L| \\cdot |U|={latex(L.det())} \\times {latex(U.det())}={latex(det_val)}$"
            )

        return det_val

    # 性质法(快速检测倍行/三角等)
    def _method_properties(self, A: Matrix):
        n = A.rows
        m = A.cols
        # 全零行
        for i in range(n):
            if all(simplify(A[i, j]) == 0 for j in range(m)):
                self.step_generator.add_step(
                    0, rf"$第\,{i+1}\,行全为\,0$")
                return 0
        # 全零列
        for j in range(m):
            if all(simplify(A[i, j]) == 0 for i in range(n)):
                self.step_generator.add_step(
                    0, rf"$第\,{j+1}\,列全为\,0$")
                return 0

        # 相同行或成比例行
        for i in range(n):
            for j in range(i + 1, n):
                if all(simplify(A[i, k] - A[j, k]) == 0 for k in range(A.cols)):
                    self.step_generator.add_step(
                        0, rf"$第\,{i+1}\,行\,=\,第\,{j+1}\,行$")
                    return 0
                # 成比例(注意防止除以0)
                try:
                    ratios = []
                    valid = True
                    for k in range(A.cols):
                        if A[i, k] == 0 and A[j, k] == 0:
                            continue
                        if A[i, k] == 0:
                            valid = False
                            break
                        ratios.append(simplify(A[j, k] / A[i, k]))
                    if valid and ratios and all(r == ratios[0] for r in ratios):
                        self.step_generator.add_step(
                            0, rf"$第\,{j+1}\,行\,=\,({latex(ratios[0])}) \\cdot 第\,{i+1}\,行$")
                        return 0
                except Exception:
                    pass

        # 相同列或成比例列
        for i in range(m):
            for j in range(i + 1, m):
                if all(simplify(A[k, i] - A[k, j]) == 0 for k in range(A.rows)):
                    self.step_generator.add_step(
                        0, rf"$第\,{i+1}\,列\,=\,第\,{j+1}\,列$")
                    return 0
                # 成比例(注意防止除以0)
                try:
                    ratios = []
                    valid = True
                    for k in range(A.rows):
                        if A[k, i] == 0 and A[k, j] == 0:
                            continue
                        if A[k, i] == 0:
                            valid = False
                            break
                        ratios.append(simplify(A[k, j] / A[k, i]))
                    if valid and ratios and all(r == ratios[0] for r in ratios):
                        self.step_generator.add_step(
                            0, rf"$第\,{j+1}\,列\,=\,({latex(ratios[0])}) \\cdot 第\,{i+1}\,列$")
                        return 0
                except Exception:
                    pass

        # 单位 / 对角 / 三角
        is_diag = all((i == j) or (A[i, j] == 0)
                      for i in range(n) for j in range(n))
        is_upper = all((i <= j) or (A[i, j] == 0)
                       for i in range(n) for j in range(n))
        is_lower = all((i >= j) or (A[i, j] == 0)
                       for i in range(n) for j in range(n))
        if is_diag or is_upper or is_lower:
            val = simplify(A.det())
            self.step_generator.add_step(
                val, rf"$|对角/三角矩阵|\,=\,对角线元素乘积\,=\,{latex(val)}$")
            return val

        return None

    # 行变换(高斯消元法)
    def _method_gaussian_elimination(self, matrix: Matrix):
        M = matrix.copy()
        rows = M.rows
        sign = 1

        for pivot_row in range(rows):
            pivot_elem = M[pivot_row, pivot_row]
            if pivot_elem == 0:
                found = False
                for swap_row in range(pivot_row + 1, rows):
                    if M[swap_row, pivot_row] != 0:
                        M.row_swap(pivot_row, swap_row)
                        self.step_generator.add_step("", "")
                        self.step_generator.add_step(
                            M.copy(), rf"$交换第\,{pivot_row+1}\,行和\,{swap_row+1}\,行(正负号取反一次)$")
                        sign *= -1
                        found = True
                        pivot_elem = M[pivot_row, pivot_row]
                        break
                if not found:
                    # 该列没有主元, 继续下一列
                    self.step_generator.add_step("", "")
                    self.step_generator.add_step(
                        M.copy(), rf"$第\,{pivot_row+1}\,列无可用主元, 跳过(此时已可得出行列式的值为\,0)$")
                    continue

            # 假定所有符号表达式(分母)不为 0, 继续消元
            for r in range(pivot_row + 1, rows):
                if simplify(M[r, pivot_row]) == 0:
                    continue
                factor = simplify(M[r, pivot_row] / pivot_elem)
                # R_r <- R_r - factor * R_pivot
                M.row_op(r, lambda v, j, pr=pivot_row,
                         factor=factor: simplify(v - factor * M[pr, j]))
                self.step_generator.add_step("", "")

                if factor == 1:
                    self.step_generator.add_step(
                        M.copy(), f"$R_{{{r+1}}} - R_{{{pivot_row+1}}} \\to R_{{{r+1}}}$")
                elif factor == -1:
                    self.step_generator.add_step(
                        M.copy(), f"$R_{{{r+1}}} + R_{{{pivot_row+1}}} \\to R_{{{r+1}}}$")
                elif factor < 0:
                    self.step_generator.add_step(
                        M.copy(), f"$R_{{{r+1}}} + \\left({latex(-factor)}\\right) \\cdot R_{{{pivot_row+1}}} \\to R_{{{r+1}}}$")
                else:
                    self.step_generator.add_step(
                        M.copy(), f"$R_{{{r+1}}} - \\left({latex(factor)}\\right) \\cdot R_{{{pivot_row+1}}} \\to R_{{{r+1}}}$")

        # 消元完成, 计算对角乘积并乘以 sign
        diag_prod = 1
        for i in range(rows):
            diag_prod *= M[i, i]
        det_val = simplify(sign * diag_prod)
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            det_val, rf"$已化为上三角矩阵:\;|A|\,=\,{'' if sign == 1 else '-'}对角线元素乘积$")
        return det_val

    def compute_list(self, matrix_input: str):
        self._reset()
        A = self._parse_matrix_input(matrix_input)
        self.step_generator.add_step(A.copy(), "输入矩阵 $A$")
        method_count = 1

        # 性质法
        self._add_section_title(rf"$法\,{method_count}:\,特殊结构检测$")
        res = self._method_properties(A)
        if res is None:
            self.step_generator.add_step(f"$\\Rightarrow 无特殊结构$", "")
        method_count += 1

        # 拉普拉斯展开法(若矩阵不太大)
        n = A.shape[0]
        if n <= 6:
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                rf"$法\,{method_count}:\;$拉普拉斯展开法", "")
            if n > 3:
                self.step_generator.add_step("", "")
                self.step_generator.add_step(
                    rf"$3 \times 3\,行列式不显示展开的余子式, 直接出结果, 以减少中间过程$")
            self.step_generator.add_step("", "")
            self._method_laplace(A)
            method_count += 1
        else:
            self.step_generator.add_step(
                "", rf"矩阵维度较大, 展开过于复杂, 跳过完整$\,$拉普拉斯展开(可自行启用)")

        # LU/PLU 分解
        self._add_section_title(rf"$法\,{method_count}:\;$LU/PLU$\,$分解法")
        self._method_lu(A)
        method_count += 1

        # 对角线法则(3x3)
        if n == 3:
            self._add_section_title(rf"$法\,{method_count}:\;$对角线法则(3×3)")
            self._method_sarrus(A)
            method_count += 1

        # 行变换法(高斯消元法)
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            rf"$法\,{method_count}:\,行变换法(高斯消元法)$", rf"$假设所有符号表达式不为\,0$")
        self.step_generator.add_step("", "")
        self.step_generator.add_step(
            rf"在计算行列式时, 即使分类讨论最后也会统一成该假设下的结果", rf"$所以此处并不需要进行是否为\,0\,的讨论$")
        self._method_gaussian_elimination(A)
        method_count += 1

        steps, explanations = self.step_generator.get_steps()
        return steps, explanations

    def compute_latex(self, matrix_input: str):
        steps, explanations = self.compute_list(matrix_input)
        return self.step_generator.get_latex(steps, explanations)
