from typing import List, Tuple
from sympy import Determinant, Expr, Matrix, sympify

from core import BaseCalculator
from utils import Context, MatcherList, Operation, RuleDict
from domains.matrix import DetStepGenerator, MATCHER_LIST, RULE_DICT


class DetCalculator(BaseCalculator):
    def __init__(self):
        super().__init__()
        self.step_generator = DetStepGenerator()

    def init_key_property(self):
        self.operation: Operation = Determinant
        self.rule_dict: RuleDict = RULE_DICT
        self.matcher_list: MatcherList = MATCHER_LIST

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
