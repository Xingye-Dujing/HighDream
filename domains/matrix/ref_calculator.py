from typing import List, Tuple, Union

from sympy import Eq, Expr, Matrix, Ne, latex, simplify, sympify
from sympy.core.relational import Relational

from domains.matrix import (
    BranchManager, BranchNode, RefStepGenerator, apply_elimination_rule,
    apply_scale_rule, apply_swap_rule
)


class RefCalculator:
    """A calculator for computing Row Echelon Form (REF) or Reduced Row Echelon Form (RREF)
    of matrices with symbolic entries, handling branching cases when denominators may be zero.

    This class systematically handles the computation of REF/RREF forms by:
    1. Tracking potential zero denominators during elimination
    2. Creating branches for different cases (denominator = 0 vs != 0)
    3. Processing each branch under its specific conditions
    4. Merging equivalent results from different branches.
    """

    def __init__(self) -> None:
        """Initialize the REF calculator with default components."""
        self.step_generator = RefStepGenerator()
        self.target_form: str | None = None
        # Global record of expressions that might be zero (denominator candidates)
        self.denominators = set()
        self.branch_manager = BranchManager()

    def _reset_process(self) -> None:
        """Reset the calculation process by clearing steps and denominators."""
        self.step_generator.reset()
        self.denominators.clear()

    @staticmethod
    def _parse_matrix_input(matrix_input: str) -> Matrix:
        """Parse string input into a SymPy Matrix.

        Args:
            matrix_input (str): String representation of a matrix in format like '[[1, 2], [3, 4]]'

        Returns:
            Matrix: Parsed SymPy Matrix object

        Raises:
            ValueError: If the matrix cannot be parsed from the input string.
        """
        try:
            return Matrix(sympify(matrix_input))
        except Exception as e:
            raise ValueError("无法解析矩阵输入. 请使用有效的矩阵格式 '[[1, 2], [3, 4]]'") from e

    # Zero/non-zero determination under conditions
    @staticmethod
    def _apply_equalities_to_expr(expr: Expr, conditions: List[Relational]) -> Expr:
        """Apply equality substitutions from conditions to an expression.

        Args:
            expr: The expression to apply substitutions to
            conditions: List of condition objects, potentially containing Eq instances.

        Returns:
            The expression after applying substitutions and simplification
        """
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
    def _is_zero_under(expr: Expr, conditions: List[Relational]) -> bool:
        """Determine if an expression is definitely zero under given conditions.

        Args:
            expr: Expression to evaluate
            conditions: List of conditions that may affect the expression value.

        Returns:
            bool: True if expression is definitely zero under conditions, False otherwise.
        """
        expr_sub = RefCalculator._apply_equalities_to_expr(expr, conditions)
        try:
            s = simplify(expr_sub)
            return s.equals(0)
        except Exception:
            return False

    @staticmethod
    def _is_nonzero_under(expr: Expr, conditions: List[Relational]) -> bool:
        """Determine if an expression is definitely non-zero under given conditions.

        Args:
            expr: Expression to evaluate
            conditions: List of conditions that may affect the expression value.

        Returns:
            bool: True if expression is definitely non-zero under conditions, False otherwise.
        """
        expr_sub = RefCalculator._apply_equalities_to_expr(expr, conditions)
        try:
            s = simplify(expr_sub)
            if s == 0:
                return False
            if len(s.free_symbols) == 0:
                return True
            # If conditions explicitly state Ne(expr,0), consider it non-zero
            for c in conditions:
                try:
                    if isinstance(c, (Ne,)) and c.lhs == expr:
                        return True
                except Exception:
                    pass
            return False
        except Exception:
            return False

    # Process a single branch within its conditions
    def _process_branch(self, branch: BranchNode, target_form: str) -> BranchNode:
        """Process a matrix under branch conditions to attempt completion to focus on form.

        If a denominator requiring case analysis is encountered, the branch.
        The finished flag
        will be set to False and returned for outer-level splitting processing.

        Args:
            branch (BranchNode): The branch node containing matrix, conditions and steps.
            target_form (str): Target form, either 'ref' or 'rref'.

        Returns:
            BranchNode: The processed branch node, possibly unfinished
        """
        matrix = branch.matrix.copy()
        rows, cols = matrix.rows, matrix.cols
        pivot_row = 0

        while pivot_row < rows:
            for col in range(cols):
                if pivot_row >= rows:
                    break
                # Search for a suitable pivot row: prefer definitely non-zero, then possibly non-zero.
                swap_r = None
                possible_r = None
                for r in range(pivot_row, rows):
                    elem = matrix[r, col]
                    # Definitely non-zero
                    if self._is_nonzero_under(elem, branch.conditions):
                        swap_r = r
                        break
                    # Uncertain whether zero or non-zero -> possibly non-zero (take first such rows)
                    if not self._is_zero_under(elem, branch.conditions) and possible_r is None:
                        possible_r = r
                if swap_r is None:
                    swap_r = possible_r

                if swap_r is None:
                    # No pivot in this column, continue to next column
                    continue

                # Swap rows if needed
                if swap_r != pivot_row:
                    matrix, expl = apply_swap_rule(
                        matrix, pivot_row, swap_r)
                    branch.add_step(matrix.copy(), expl)
                # If the pivot element contains symbols and current conditions are not enough
                # to determine it is non-zero, record it as a denominator for case analysis.
                if swap_r == possible_r:
                    pivot_elem = matrix[pivot_row, col]
                    self.denominators.add(pivot_elem)
                    branch.add_step(
                        matrix.copy(), f"需对候选主元 ${latex(pivot_elem)}$ 进行是否为 0 的讨论")
                    branch.finished = False
                    return branch

                # Scale pivot to 1 (if needed)
                matrix, expl = apply_scale_rule(matrix, pivot_row, col)
                if expl:
                    branch.matrix = matrix.copy()
                    branch.add_step(matrix.copy(), expl)

                # Eliminate other rows depending on target_form.
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
                        # Skip if already zero
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

    @staticmethod
    def _split_branch_on_denominator(branch: BranchNode, denom_expr: Expr) -> List[BranchNode]:
        """Split a branch into two based on a denominator expression:
          - denom != 0 branch (continue directly)
          - denom == 0 branch (substitute denom=0 in the matrix, then continue)

        Only returns satisfiable branches.

        Args:
            branch (BranchNode): The branch to split
            denom_expr: The denominator expression to split on.

        Returns:
            list: New branch nodes generated from splitting
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

        # B: den == 0 -> substitute den=0 in matrix
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

    def _compute_branches(self, original_matrix: Matrix, target_form: str) -> List[BranchNode]:
        """Expand branches using BFS (queue-based):
          1. Process the front branch in the queue using _process_branch
          2. If branch is unfinished, split based on self.denominators (prioritizing the earliest appearing)
          3. Collect all finished leaves and merge equivalent results using BranchManager.merge_leaves

        Args:
            original_matrix (Matrix): The original matrix to process
            target_form (str): Target form ('ref' or 'rref')

        Returns:
            list: Merged finished branch leaves
        """
        root = BranchNode(
            conditions=[], matrix=original_matrix.copy(), steps=[])
        queue = [root]
        finished_leaves = []

        # Clear denominators (ensure the fresh collection from start)
        self.denominators = set()

        while queue:
            branch = queue.pop(0)
            processed = self._process_branch(branch, target_form)
            if not processed.finished:
                # Find denominators requiring splitting (excluding those already covered by branch conditions)
                to_split = []
                for den in list(self.denominators):
                    already = any((hasattr(c, 'lhs') and c.lhs == den)
                                  for c in branch.conditions)
                    if not already:
                        to_split.append(den)
                if not to_split:
                    # No splittable expressions -> branch stuck, mark as unsatisfiable
                    branch.reason = "无可分裂的表达式但未完成"
                    continue
                # Split on first available expression
                den = to_split[0]
                new_branches = self._split_branch_on_denominator(branch, den)
                for nb in new_branches:
                    queue.append(nb)
                continue

            if BranchManager.is_satisfiable(branch.conditions):
                finished_leaves.append(branch)
            else:
                continue

        merged = self.branch_manager.merge_leaves(finished_leaves)
        return merged

    def compute_list(self, matrix_input: str, target_form='rref') -> Tuple[List[Union[Expr, Matrix]], List[str]]:
        """Compute the REF/RREF form with detailed steps and explanations.

        Args:
            matrix_input (str): String representation of matrix
            target_form (str): Target form, either 'ref' or 'rref'. Defaults to 'rref'.

        Returns:
            tuple: (steps, explanations) lists containing the computation steps
        """
        self._reset_process()
        self.target_form = target_form
        original_matrix = self._parse_matrix_input(matrix_input)

        leaves = self._compute_branches(original_matrix, target_form)

        self.step_generator.add_step(original_matrix, "初始矩阵")

        # Write each leaf's steps to step_generator (with branch title and conditions)
        for i, leaf in enumerate(leaves):
            cond_latex = BranchManager.conditions_to_latex(leaf.conditions)
            # Line break
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                f"\\quad\\quad 分支 {i + 1}", f"条件：${cond_latex}$")
            for mat_or_str, expl in leaf.steps:
                # Line break
                self.step_generator.add_step("", "")
                self.step_generator.add_step(mat_or_str, expl)

        if not leaves:
            self.step_generator.add_step(
                original_matrix, "在所有划分下均无可行解(所有分支均不可满足)")
        else:
            # Line break
            self.step_generator.add_step("", "")
            self.step_generator.add_step(
                "\\quad\\quad 最终结果汇总",
                "每个分支下的最终矩阵与条件"
            )
            for i, leaf in enumerate(leaves):
                cond_latex = BranchManager.conditions_to_latex(leaf.conditions)
                # Get the final matrix from the last step of branch
                if isinstance(leaf.steps[-1][0], str):
                    final_matrix = leaf.steps[-2][0]
                else:
                    final_matrix = leaf.steps[-1][0]
                # Line break
                self.step_generator.add_step("", "")
                self.step_generator.add_step(
                    final_matrix,
                    f"分支 {i + 1}: 条件 ${cond_latex}$"
                )

        steps, explanations = self.step_generator.get_steps()
        return steps, explanations

    def compute_latex(self, matrix_input: str, target_form='rref') -> str:
        """Compute the REF/RREF form and return LaTeX formatted output.

        Args:
            matrix_input (str): String representation of matrix
            target_form (str): Target form, either 'ref' or 'rref'. Defaults to 'rref'.

        Returns:
            str: LaTeX formatted computation steps and explanations
        """
        steps, explanations = self.compute_list(matrix_input, target_form)
        return self.step_generator.get_latex(steps, explanations)
