# TODO 对输入表达式进行最适合积分的等价变换(如三角恒等变形、部分分式分解等), 再求值 - expression_parser.py
# TODO 支持定积分计算(增加上下限参数)
# TODO 支持多重积分(偏积分)

from typing import List, Tuple
from sympy import Expr, Integral, Symbol

from core import BaseCalculator
from utils import MatcherList, Operation, RuleDict
from domains.integral import MATCHER_LIST, RULE_DICT

# Define the constant of integration C for indefinite integrals.
C = Symbol('C')


class IntegralCalculator(BaseCalculator):
    """Symbolic integral calculator that support step-by-step evaluation.

    Examples:
    >>> from sympy import Symbol
    >>> from IPython.display import display, Math
    >>> from domains import IntegralCalculator

    >>> expr =  'x**3+sin(x)'
    >>> x = Symbol('x')

    >>> calculator = IntegralCalculator()
    >>> latex_output = calculator.compute_latex(expr, x)

    >>> display(Math(latex_output))
    """

    def init_key_property(self) -> None:
        self.operation: Operation = Integral
        self.rule_dict: RuleDict = RULE_DICT
        self.matcher_list: MatcherList = MATCHER_LIST

    def _final_postprocess(self, final_expr: Expr) -> None:
        """Add constant of integration (+C) for indefinite integrals without the integral symbol.

        Ensure the constant of integration (+C) appears only in explanatory text or final output,
        never embedded in the symbolic antiderivative expression.
        """
        super()._final_postprocess(final_expr)
        # Add C to the last step
        self.step_generator.steps[-1] += C
        try:
            # When expression experience simplification without assumptions,
            # add C to the penultimate expression.
            if not self.step_generator.steps[-2].has(C):
                self.step_generator.steps[-2] += C
            # When no rule applies to the remaining Integral, fall back to SymPy's Integral.doit() and
            # the result experiences simplification without assumptions and simplification with assumptions.
            # add C to the third-to-last expression.
            if len(self.step_generator.steps) > 3 and not self.step_generator.steps[-3].has(C):
                self.step_generator.steps[-3] += C
        except IndexError:
            pass

    def compute_list(self, expr: str, var: Symbol) -> Tuple[List[Expr], List[str]]:
        """Define Integral context."""
        return super().compute_list(expr, variable=var)

    def compute_latex(self, expr: str, var: Symbol) -> str:
        """Define Integral context."""
        return super().compute_latex(expr, variable=var)
