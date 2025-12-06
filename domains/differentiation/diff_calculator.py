from typing import List, Tuple
from sympy import Derivative, Expr, Symbol

from core import BaseCalculator, SelectRuleCalculator
from utils import MatcherList, Operation, RuleDict
from domains.differentiation import MATCHER_LIST, RULE_DICT


def create_diff_calculator(base_class):
    class DiffCalculatorImpl(base_class):
        """Symbolic differentiation calculator that support step-by-step evaluation.

        Examples:
        >>> from sympy import Symbol
        >>> from IPython.display import display, Math
        >>> from domains import DiffCalculator

        >>> x = Symbol('x')
        >>> expr = "sin(x**x)+cos(1/x)-ln(1/x)"

        >>> calculator = DiffCalculator()
        >>> latex_output = calculator.compute_latex(expr, x)

        >>> display(Math((latex_output)))
        """

        def init_key_property(self) -> None:
            self.operation: Operation = Derivative
            self.rule_dict: RuleDict = RULE_DICT
            self.matcher_list: MatcherList = MATCHER_LIST

        def compute_list(self, expr: str, var: Symbol) -> Tuple[List[Expr], List[str]]:
            """Define Derivative context."""
            return super().compute_list(expr, variable=var)

        def compute_latex(self, expr: str, var: Symbol) -> str:
            """Define Derivative context."""
            return super().compute_latex(expr, variable=var)

    return DiffCalculatorImpl


DiffCalculator = create_diff_calculator(BaseCalculator)
SelectDiffCalculator = create_diff_calculator(SelectRuleCalculator)
