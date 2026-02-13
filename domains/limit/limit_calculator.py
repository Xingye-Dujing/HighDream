from typing import List, Tuple

from sympy import AccumBounds, Basic, Expr, Limit, S, Symbol, latex, sympify, zoo

from core import BaseCalculator, SelectRuleCalculator
from domains.limit import LHOPITAL_RULES, LimitStepGenerator, MATCHER_LIST, RULE_DICT
from utils import Context, Operation, RuleFunction
from .limit_help_func import detect_feasible_directions


def create_limit_calculator(base_class):
    """Factory function to create limit calculator classes."""

    class LimitCalculatorImpl(base_class):
        """Symbolic limit calculator that support step-by-step evaluation."""

        def __init__(self, max_lhopital: int = 5) -> None:
            super().__init__(Limit, RULE_DICT, MATCHER_LIST)
            # Rationalizing the denominator is helpful for solving limits
            self.is_radsimp: bool = True
            # Use a specialized step generator for two-sided limits' steps generator.
            self.step_generator = LimitStepGenerator()
            # Define lhopital rules
            self._lhopital_rules = LHOPITAL_RULES
            # Prevent infinite loops when applying lhopital rules.
            self._lhopital_count = 0
            # The Maximum number of times lhopital rules may be applied.
            self._lhopital_max = max_lhopital

        def reset_process(self) -> None:
            super().reset_process()
            self._lhopital_count = 0

        @staticmethod
        def _context_split(**context: Context) -> Tuple[Symbol, Expr | int, str]:
            return context.get('variable', Symbol('x')), context.get('point', 0), context.get('direction', '+')

        def _perform_operation(self, expr: Expr, operation: Operation, **context: Context) -> Operation:
            var, point, direction = self._context_split(**context)
            return operation(expr, var, point, dir=direction)

        def _check_rule_is_can_apply(self, rule: RuleFunction) -> bool:
            # Prevent infinite loops when applying lhopital rules.
            if rule.__name__ in self._lhopital_rules:
                self._lhopital_count += 1
                if self._lhopital_count > self._lhopital_max:
                    return False
            return True

        def _final_postprocess(self, final_expr: Expr) -> None:
            if final_expr.has(AccumBounds):
                self.step_generator.add_step(S.NaN, "函数在极限点附近振荡, 极限不存在")

        def _compute_single_direction(self, expr: Basic | Expr, dire: str, **context: Context) \
                -> Tuple[List[Expr], List[str], Expr]:
            context['direction'] = dire
            self._do_compute(expr, self.operation, **context)
            steps, explanations = self.step_generator.get_steps()
            final_result = steps[-1] if steps else None
            return steps, explanations, final_result

        def _add_left_limit_step(self, expr: Expr, var: Symbol, point: Expr) -> None:
            self.step_generator.add_step(
                Limit(expr, var, point, dir='-'),
                "计算左极限"
            )

        def _add_right_limit_step(self, expr: Expr, var: Symbol, point: Expr) -> None:
            self.step_generator.add_step(
                Limit(expr, var, point, dir='+'),
                "计算右极限"
            )

        def _add_both_limit_step(self, expr: Expr, var: Symbol, point: Expr) -> None:
            self.step_generator.add_step(
                Limit(expr, var, point, dir='+-'),
                ""
            )

        def _add_limit_step(self, direction, expr: Basic | Expr, var: Symbol, point: Expr, steps: List[Expr],
                            explanations: List[str]) -> None:
            if direction == '+':
                self._add_right_limit_step(expr, var, point)
            elif direction == '-':
                self._add_left_limit_step(expr, var, point)
            else:
                self._add_both_limit_step(expr, var, point)
            self.step_generator.steps += steps[1:]
            self.step_generator._explanations += explanations[1:]

        def _add_final_step(self, left_result: Expr, right_result: Expr) -> None:
            explanation = (
                rf"最终结论: 左极限 = ${latex(left_result)}$, "
                rf"右极限 = ${latex(right_result)}$, "
            )
            # Check if left limit or right limit NaN
            if S.NaN in (left_result, right_result):
                final_result = zoo
                explanation += "至少一个极限为 NaN, 故极限不存在"
            # Check if the left limit and right limit are equal
            elif left_result.equals(right_result):
                final_result = left_result
                explanation += rf"左右极限相等, 故极限存在, 值为 ${latex(final_result)}$"
            else:
                final_result = zoo
                explanation += "左右极限不相等, 故极限不存在"
            self.step_generator.add_step(final_result, explanation)

        def _compute_both_directions(self, expr: Basic | Expr, **context: Context) -> None:
            var, point, _ = self._context_split(**context)

            # Compute left limit
            left_steps, left_explanations, left_result = self._compute_single_direction(
                expr, '-', **context)
            # Compute right limit
            right_steps, right_explanations, right_result = self._compute_single_direction(
                expr, '+', **context)

            self.step_generator.reset()
            # Add left limit's steps
            self._add_limit_step(
                '-', expr, var, point, left_steps, left_explanations)
            # Add right limit's steps
            self._add_limit_step(
                '+', expr, var, point, right_steps, right_explanations)
            # Add the final step
            self._add_final_step(left_result, right_result)

        def _compute(self, expr: str, **context: Context) -> None:
            """Compute the limit in the specified direction after validating its feasibility."""
            var, point, direction = self._context_split(**context)
            expr, point = sympify(expr), sympify(point)

            feasible_directions = detect_feasible_directions(expr, var, point)
            # Validate both directions' feasibility
            if direction == 'both':
                if not feasible_directions['left'] and not feasible_directions['right']:
                    self.reset_process()
                    self.step_generator.add_step(
                        Limit(expr, var, point, '+-'),
                        rf"初始检测: 在 $\, x \to {point} \,$ 处, 左右邻域均违反定义域约束, 极限不存在."
                    )
                    self.step_generator.add_step(zoo, "结论: 极限不存在")
                    return
                if not feasible_directions['left']:
                    self.step_generator.add_step(
                        Limit(expr, var, point, '+-'),
                        rf"初始检测: 在 $\, x \to {point}^- \,$ 处, 左邻域违反定义域约束, 仅能尝试计算右极限."
                    )
                    steps, explanations, right_result = self._compute_single_direction(
                        expr, '+', **context)
                    self.step_generator.reset()
                    self._add_limit_step(
                        '+-', expr, var, point, steps, explanations)
                    self.step_generator.add_step(
                        right_result,
                        f"结论: 由于左极限不可计算, 仅右极限存在, 值为 ${latex(right_result)}$."
                    )
                    return
                if not feasible_directions['right']:
                    self.step_generator.add_step(
                        Limit(expr, var, point, '+-'),
                        rf"$初始检测: 在$\, x \to {point}^- \,$ 处, 右邻域违反定义域约束, 仅能尝试计算左极限."
                    )

                    steps, explanations, left_result = self._compute_single_direction(
                        expr, '-', **context)
                    self.step_generator.reset()
                    self._add_limit_step(
                        '+-', expr, var, point, steps, explanations)
                    self.step_generator.add_step(
                        left_result,
                        f"结论: 由于右极限不可计算, 仅左极限存在, 值为 ${latex(left_result)}$."
                    )
                    return
                self._compute_both_directions(expr, **context)
            else:
                # Single direction
                if direction == '+' and not feasible_directions['right']:
                    self.reset_process()
                    self.step_generator.add_step(
                        Limit(expr, var, point, '+'),
                        rf"初始检测: 在$\, x \to {point}^- \,$ 处, 右邻域违反定义域约束，右极限不可计算."
                    )
                    return
                if direction == '-' and not feasible_directions['left']:
                    self.reset_process()
                    self.step_generator.add_step(
                        Limit(expr, var, point, '-'),
                        rf"初始检测: 在$\, x \to {point}^- \,$ 处, 左邻域违反定义域约束, 左极限不可计算."
                    )
                    return
                self._compute_single_direction(expr, direction, **context)

        def compute_list(self, expr: str, var: Symbol = Symbol('x'), point: Expr = 0, direction: str = '+') \
                -> Tuple[List[Expr], List[str]]:
            return super().compute_list(expr, variable=var, point=point, direction=direction)

        def compute_latex(self, expr: str, var: Symbol = Symbol('x'), point: Expr = 0, direction: str = '+') -> str:
            self._compute(expr, variable=var,
                          point=point, direction=direction)
            if direction == 'both':
                return self.step_generator.to_latex_both()
            return self.step_generator.to_latex()

    return LimitCalculatorImpl


LimitCalculator = create_limit_calculator(BaseCalculator)

SelectLimitCalculator = create_limit_calculator(SelectRuleCalculator)
