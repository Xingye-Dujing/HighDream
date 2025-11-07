# TODO 对每一个表达式先进行最适合求导的等价变换，再计算(sympy 化简后的表达式不一定是最适合求导的表达式) - expression_parser.py
# TODO 支持对含有不确定量，但不确定量是一个常数的函数求导，结果用不确定量表示(偏导的一种特例)
# TODO 支持求偏导

from typing import List, Tuple
from sympy import Derivative, Expr, Symbol

from core import BaseCalculator
from utils import MatcherList, Operation, RuleDict
from domains.differentiation import MATCHER_LIST, RULE_DICT


class DiffCalculator(BaseCalculator):
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
