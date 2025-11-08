# TODO 对输入表达式进行最适合积分的等价变换(如三角恒等变形、部分分式分解等), 再求值 - expression_parser.py
# TODO 支持定积分计算(增加上下限参数)
# TODO 支持多重积分(偏积分)

from typing import List, Tuple
from sympy import Expr, Integral, Symbol, simplify

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

    @staticmethod
    def _step_expr_postprocess(step_expr: Expr) -> Expr:
        """Add constant of integration (+C) for indefinite integrals without the integral symbol."""
        if step_expr.has(Integral):
            return step_expr
        return step_expr + C

    def _final_postprocess(self, final_expr: Expr) -> None:
        """Ensure the constant of integration (+C) appears only in explanatory text or final output,
        never embedded in the symbolic antiderivative expression.
        """
        if not final_expr.free_symbols:
            return

        final_expr -= C

        # Map each free symbol to a new symbol with positive=True, real=True
        assumption_map = {
            s: Symbol(s.name, positive=True, real=True)
            for s in final_expr.free_symbols
        }

        # Replace symbols with their assumed counterparts
        expr_with_assumptions = final_expr.xreplace(assumption_map)
        simplified_expr = simplify(expr_with_assumptions) + C

        # Must compare strings, because variables must be not equal due to different assumptions
        if str(simplified_expr) != str(final_expr):
            self.step_generator.add_step(
                simplified_expr,
                "假设所有变量为正实数, 化简表达式"
            )

    def compute_list(self, expr: str, var: Symbol) -> Tuple[List[Expr], List[str]]:
        """Define Integral context."""
        return super().compute_list(expr, variable=var)

    def compute_latex(self, expr: str, var: Symbol) -> str:
        """Define Integral context."""
        return super().compute_latex(expr, variable=var)
