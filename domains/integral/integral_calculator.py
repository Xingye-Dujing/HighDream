from typing import List, Tuple

from sympy import Add, Expr, Integral, Symbol, simplify

from core import BaseCalculator, SelectRuleCalculator
from domains.integral import MATCHER_LIST, RULE_DICT

# Define the constant of integration C for indefinite integrals.
C = Symbol('C')


def create_integral_calculator(base_class):
    """Factory function to create integral calculator classes."""

    class IntegralCalculatorImpl(base_class):
        """Symbolic integral calculator that supports step-by-step evaluation.

        Examples:
        >>> from sympy import Symbol
        >>> from IPython.display import display, Math
        >>> from domains import IntegralCalculator

        >>> expr = 'x**3+sin(x)'
        >>> x = Symbol('x')

        >>> calculator = IntegralCalculator()
        >>> latex_output = calculator.compute_latex(expr, x)

        >>> display(Math(latex_output))
        """

        def __init__(self):
            super().__init__(Integral, RULE_DICT, MATCHER_LIST)

        @staticmethod
        def _merge_constants_with_c(expr: Expr) -> Expr:
            """Merge all constant terms to C."""

            if expr.is_number:
                return C

            expand_expr = expr.expand()
            if not isinstance(expand_expr, Add):
                return expr + C

            other_terms = [
                term for term in expand_expr.args if not term.is_number]
            # If there are more than 4 terms, don't merge constant terms to C to avoid complexity.
            if len(other_terms) > 4 and not isinstance(expr, Add):
                return expr + C
            return simplify(Add(*other_terms)) + C

        def _final_postprocess(self, final_expr: Expr) -> None:
            """Add constant of integration (+C) for indefinite integrals without the integral symbol.

            Ensure the constant of integration (+C) appears only in explanatory text or final output,
            never embedded in the symbolic antiderivative expression.
            """
            super().final_postprocess(final_expr)
            # Add C to the last step
            self.step_generator.steps[-1] = self._merge_constants_with_c(
                self.step_generator.steps[-1])
            try:
                # When expression experiences simplification without assumptions,
                # add C to the penultimate expression.
                if not self.step_generator.steps[-2].has(C) and not self.step_generator.steps[-2].has(Integral):
                    self.step_generator.steps[-2] = self._merge_constants_with_c(
                        self.step_generator.steps[-2])
                # When no rule applies to the remaining Integral, fall back to SymPy's Integral.doit() and
                # the result experiences simplification without assumptions and simplification with assumptions.
                # Add C to the third-to-last expression.
                if not self.step_generator.steps[-3].has(C) and not self.step_generator.steps[-3].has(Integral):
                    self.step_generator.steps[-3] = self._merge_constants_with_c(
                        self.step_generator.steps[-3])
            except (IndexError, AttributeError):
                pass

        def compute_list(self, expr: str, var: Symbol) -> Tuple[List[Expr], List[str]]:
            """Define Integral context."""
            return super().compute_list(expr, variable=var)

        def compute_latex(self, expr: str, var: Symbol) -> str:
            """Define Integral context."""
            return super().compute_latex(expr, variable=var)

    return IntegralCalculatorImpl


IntegralCalculator = create_integral_calculator(BaseCalculator)

SelectIntegralCalculator = create_integral_calculator(SelectRuleCalculator)
