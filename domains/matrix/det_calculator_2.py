from typing import Deque, Dict, List, Tuple
from collections import deque
from sympy import Determinant, Expr, Matrix, Symbol, latex, simplify, sympify

from core import BaseCalculator
from utils import Context, MatcherList, Operation, RuleDict
from domains.matrix import DetStepGenerator, MATCHER_LIST, RULE_DICT


class DetCalculator(BaseCalculator):
    def __init__(self) -> None:
        super().__init__()
        self.step_generator = DetStepGenerator()

    def init_key_property(self) -> None:
        self.operation: Operation = Determinant
        self.rule_dict: RuleDict = RULE_DICT
        self.matcher_list: MatcherList = MATCHER_LIST

    def _get_cached_result(self, expr: Expr, operation: Operation, **context: Context) -> Operation:
        """Return a cached result if available, otherwise compute and cache the result."""
        # Solve the problem: unhashable type: 'MutableDenseMatrix'
        key = (str(expr), str(context))
        if key not in self.cache:
            self.cache[key] = self._perform_operation(
                expr, operation, **context)
        return self.cache[key]

    def _do_compute(self, expr: str, operation: Operation, **context: Context) -> None:
        """Perform the core symbolic computation and record each evaluation step."""
        self.reset_process()
        expr = self._sympify(expr)
        # Extract the unique variable from the expression for calculating.
        symbol_list = list(expr.free_symbols)
        if len(symbol_list) > 1:
            raise ValueError(
                "仅允许出现一个字母变量.")
        if symbol_list:
            context['variable'] = symbol_list[0]
        else:
            context['variable'] = Symbol("x")

        initial_operation = self._get_cached_result(expr, operation, **context)
        self.step_generator.add_step(initial_operation)

        # BFS using a queue.
        queue: Deque[Expr] = deque([expr])
        # Solve the problem: unhashable type: 'MutableDenseMatrix'
        expr_key = str(expr)
        expr_to_operation: Dict[Expr, Operation] = {
            expr_key: initial_operation}

        while queue:
            direct_compute = False

            current_expr = queue.popleft()
            # Solve the problem: unhashable type: 'MutableDenseMatrix'
            current_expr_key = str(current_expr)
            current_operation = expr_to_operation.get(current_expr_key)

            if current_expr_key in self.processed:
                direct_compute = True
            self.processed.add(current_expr_key)

            # Extract the unique variable from the expression for substitution.
            symbol_list = list(current_expr.free_symbols)
            if symbol_list:
                context['variable'] = symbol_list[0]
            elif self.step_generator.subs_dict:
                context['variable'] = list(
                    self.step_generator.subs_dict.keys())[-1]
            else:
                context['variable'] = Symbol("x")

            new_expr, explanation, expr_to_operation = self._update_expression(
                current_expr, operation, expr_to_operation, direct_compute, **context)

            current_step = expr_to_operation[expr_key]
            current_step = self._step_expr_postprocess(current_step)

            if self.sudden_end[0]:
                self.step_generator.add_step(
                    None, f'${latex(self.sudden_end[1])}$ 无法用初等函数表示, 退出计算.')
                break

            self.step_generator.add_step(current_step, explanation)

            if new_expr != current_operation:
                # Extract sub-expressions to continue processing.
                sub_exprs = [new_expr] if isinstance(
                    new_expr, operation) else new_expr.atoms(operation)
                for s in sub_exprs:
                    sub_expr = s.args[0]
                    expr_to_operation[str(sub_expr)] = s
                    queue.append(sub_expr)

        if not self.sudden_end[0]:
            exprs, _ = self.step_generator.get_steps()
            final_expr = exprs[-1]

            # Final simplification
            simplified_expr = simplify(final_expr, inverse=True)
            if simplified_expr != final_expr:
                self.step_generator.steps[-1] = simplified_expr
                final_expr = simplified_expr

            self._final_postprocess(final_expr)

    def _sympify(self, matrix_expr: str) -> Matrix:
        """Convert the input expression to a SymPy matrix."""
        return Matrix(sympify(matrix_expr))

    def _perform_operation(self, matrix: Matrix, _operation: Operation, **_context: Context) -> Operation:
        return Determinant(matrix)

    def compute_list(self, matrix_expr: str) -> Tuple[List[Expr], List[str]]:
        self._compute(matrix_expr)
        return self.step_generator.get_steps()

    def compute_latex(self, matrix_expr: str) -> str:
        self._compute(matrix_expr)
        return self.step_generator.get_latex()
