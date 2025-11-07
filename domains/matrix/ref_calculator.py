from sympy import Matrix, sympify, simplify, Eq, Ne, latex
from domains.matrix import (RefStepGenerator, BranchManager, BranchNode,
                            apply_swap_rule, apply_scale_rule, apply_elimination_rule)


class RefCalculator():

    def __init__(self):
        self.step_generator = RefStepGenerator()
        self.target_form: str = None
        # 全局记录检测到的可能为 0 的表达式(即分母候选)
        self.denominators = set()
        self.branch_manager = BranchManager()

    def _reset_process(self):
        self.step_generator.reset()
        self.denominators.clear()

    def _parse_matrix_input(self, matrix_input: str):
        try:
            return Matrix(sympify(matrix_input))
        except Exception as e:
            raise ValueError("无法解析矩阵输入. 请使用有效的矩阵格式 '[[1, 2], [3, 4]]'") from e

    # 条件下的零/非零判定
    @staticmethod
    def _apply_equalities_to_expr(expr, conditions):
        """把条件集合中的等式替换到表达式上, 返回替换结果并 simplify"""
        subs_map = {}
        for c in conditions:
            try:
                if isinstance(c, (Eq,)) and c.lhs is not None:
                    subs_map[c.lhs] = c.rhs
            except Exception:
                continue
        if subs_map:
            try:
                return expr.subs(subs_map)
            except Exception:
                return expr
        return expr

    @staticmethod
    def _is_zero_under(expr, conditions) -> bool:
        """判断在给定条件下 expr 是否确定为 0"""
        expr_sub = RefCalculator._apply_equalities_to_expr(expr, conditions)
        try:
            s = simplify(expr_sub)
            return s.equals(0)
        except Exception:
            return False

    @staticmethod
    def _is_nonzero_under(expr, conditions) -> bool:
        """判断在给定条件下 expr 是否确定非 0(若不能确定返回 False)"""
        expr_sub = RefCalculator._apply_equalities_to_expr(expr, conditions)
        try:
            s = simplify(expr_sub)
            if s == 0:
                return False
            if len(s.free_symbols) == 0:
                return True
            # 若条件中已有明确 Ne(expr,0), 则视为非零
            for c in conditions:
                try:
                    if isinstance(c, (Ne,)) and c.lhs == expr:
                        return True
                except Exception:
                    pass
            return False
        except Exception:
            return False

    # 分支内处理单个分支
    def _process_branch(self, branch: BranchNode, target_form: str) -> BranchNode:
        """
        在 branch 的条件下处理矩阵（尝试完成到 target_form）。
        如果遇到需要分裂的分母，会把 branch.finished 设为 False 并返回，
        由外层进行分裂处理。
        """
        matrix = branch.matrix.copy()
        rows, cols = matrix.rows, matrix.cols
        pivot_row = 0

        while pivot_row < rows:
            for col in range(cols):
                if pivot_row >= rows:
                    break
                # 搜寻可作为主元的行：优先选当前条件下确定非零的行；否则挑选可能非零的一行
                swap_r = None
                possible_r = None
                for r in range(pivot_row, rows):
                    elem = matrix[r, col]
                    # 确定非零
                    if self._is_nonzero_under(elem, branch.conditions):
                        swap_r = r
                        break
                    # 不确定是非零(break), 也不确定是零 -> 可能为 0(取第一个可能行)
                    if not self._is_zero_under(elem, branch.conditions) and possible_r is None:
                        possible_r = r
                if swap_r is None:
                    swap_r = possible_r

                if swap_r is None:
                    # 该列没有主元，继续下列
                    continue

                # 若需换行
                if swap_r != pivot_row:
                    matrix, expl = apply_swap_rule(
                        matrix, pivot_row, swap_r)
                    branch.add_step(matrix.copy(), expl)
                # 若 pivot_elem 含符号且当前条件不足以判定非零, 则记录为需讨论分母
                if swap_r == possible_r:
                    pivot_elem = matrix[pivot_row, col]
                    self.denominators.add(pivot_elem)
                    branch.add_step(
                        matrix.copy(), f"需对候选主元 ${latex(pivot_elem)}$ 进行是否为 0 的讨论")
                    branch.finished = False
                    return branch

                # 放缩主元至 1(若需要)
                matrix, expl = apply_scale_rule(matrix, pivot_row, col)
                if expl:
                    branch.matrix = matrix.copy()
                    branch.add_step(matrix.copy(), expl)

                # 消去其它行或下方行, 取决于 target_form
                if target_form == 'rref':
                    for r in range(rows):
                        if r == pivot_row or matrix[r, col] == 0:
                            continue
                        matrix, expl = apply_elimination_rule(
                            matrix, pivot_row, col, r)
                        if expl:
                            branch.matrix = matrix.copy()
                            branch.add_step(matrix.copy(), expl)
                else:  # ref
                    for r in range(pivot_row + 1, rows):
                        # 本身为 0, 跳过
                        if matrix[r, col] == 0:
                            continue
                        matrix, expl = apply_elimination_rule(
                            matrix, pivot_row, col, r)
                        if expl:
                            branch.matrix = matrix.copy()
                            branch.add_step(matrix.copy(), expl)

                break
            pivot_row += 1

        branch.finished = True
        branch.add_step(
            matrix.copy(), rf"$最终\;{target_form.upper()}\;形式(在分支条件下)$")
        return branch

    def _split_branch_on_denominator(self, branch: BranchNode, denom_expr):
        """
        在 denom_expr 上对 branch 做二分裂:
          denom != 0 分支(直接继续)
          denom == 0 分支(在矩阵中代入 denom=0, 再继续)
        返回生成的新分支列表(仅返回可满足的分支)
        """
        den = denom_expr
        baseconds = branch.conditions.copy()

        bA = BranchNode(conditions=baseconds.copy(),
                        matrix=branch.matrix.copy(), steps=branch.steps.copy())
        bB = BranchNode(conditions=baseconds.copy(),
                        matrix=branch.matrix.copy(), steps=branch.steps.copy())

        # A: den != 0
        bA.conditions.append(BranchManager.cond_ne(den))
        bA.add_step(bA.matrix.copy(), f"假设 ${latex(den)} \\neq 0$(分支)")

        # B: den == 0 -> 对矩阵进行代入
        bB.conditions.append(BranchManager.cond_eq(den))
        try:
            bB.matrix = bB.matrix.applyfunc(lambda x: simplify(x.subs(den, 0)))
        except Exception:
            try:
                bB.matrix = bB.matrix.subs(den, 0)
            except Exception:
                pass
        bB.add_step(bB.matrix.copy(), f"假设 ${latex(den)} = 0$(分支)")

        out = []
        if BranchManager.is_satisfiable(bA.conditions):
            out.append(bA)
        if BranchManager.is_satisfiable(bB.conditions):
            out.append(bB)
        return out

    def _compute_branches(self, original_matrix: Matrix, target_form: str):
        """
        使用 BFS(队列) 展开分支:
          每次处理队列头的分支，调用 _process_branch
          若分支未 finished, 则依据 self.denominators 找到可用于分裂的表达式并分裂(优先最早出现的)
          最终收集所有 finished 的叶子并调用 BranchManager.merge_leaves 合并等价结果
        """
        root = BranchNode(
            conditions=[], matrix=original_matrix.copy(), steps=[])
        queue = [root]
        finished_leaves = []

        # 清空 denominators(确保从 0 开始收集)
        self.denominators = set()

        while queue:
            branch = queue.pop(0)
            processed = self._process_branch(branch, target_form)
            if not processed.finished:
                # 找到需要分裂的 denominators(排除该分支已有条件涵盖的表达式)
                to_split = []
                for den in list(self.denominators):
                    already = any((hasattr(c, 'lhs') and c.lhs == den)
                                  for c in branch.conditions)
                    if not already:
                        to_split.append(den)
                if not to_split:
                    # 若没有可分裂的表达式, 则该分支卡住 -> 标记为不可行
                    branch.reason = "无可分裂的表达式但未完成"
                    continue
                # 取第一个表达式分裂
                den = to_split[0]
                new_branches = self._split_branch_on_denominator(branch, den)
                for nb in new_branches:
                    queue.append(nb)
                continue
            else:
                if BranchManager.is_satisfiable(branch.conditions):
                    finished_leaves.append(branch)
                else:
                    continue

        merged = self.branch_manager.merge_leaves(finished_leaves)
        return merged

    def compute_list(self, matrix_input: str, target_form='rref'):
        self._reset_process()
        self.target_form = target_form
        original_matrix = self._parse_matrix_input(matrix_input)

        leaves = self._compute_branches(original_matrix, target_form)

        self.step_generator.add_step(original_matrix, "初始矩阵")

        # 将每个叶子的步骤写入 step_generator(带分支标题与条件)
        for i, leaf in enumerate(leaves):
            cond_latex = BranchManager.conditions_to_latex(leaf.conditions)
            # 换行
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                f"\\quad\\quad 分支 {i+1}", f"条件：${cond_latex}$")
            for mat_or_str, expl in leaf.steps:
                # 换行
                self.step_generator.add_step("", "")
                self.step_generator.add_step(mat_or_str, expl)

        if not leaves:
            self.step_generator.add_step(
                original_matrix, "在所有划分下均无可行解(所有分支均不可满足)")
        else:
            # 换行
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                "\\quad\\quad 最终结果汇总",
                "每个分支下的最终矩阵与条件"
            )
            for i, leaf in enumerate(leaves):
                cond_latex = BranchManager.conditions_to_latex(leaf.conditions)
                # 取该分支的最后一步矩阵
                if isinstance(leaf.steps[-1][0], str):
                    final_matrix = leaf.steps[-2][0]
                else:
                    final_matrix = leaf.steps[-1][0]
                # 换行
                self.step_generator.add_step("", "")
                self.step_generator.add_step(
                    final_matrix,
                    f"分支 {i+1}: 条件 ${cond_latex}$"
                )

        steps, explanations = self.step_generator.get_steps()
        return steps, explanations

    def compute_latex(self, matrix_input: str, target_form='rref'):
        steps, explanations = self.compute_list(matrix_input, target_form)
        return self.step_generator.get_latex(steps, explanations)
