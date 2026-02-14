from abc import ABC, abstractmethod
from collections import deque
from functools import lru_cache
from typing import Deque, Dict, List, Tuple

from sympy import Expr, Matrix, Symbol, expand_log, latex, radsimp, simplify, sympify

from utils import (
    Context, MatcherList, Operation, RuleContext,
    RuleDict, RuleFunction, is_elementary_expression
)
from .base_step_generator import BaseStepGenerator
from .rule_registry import RuleRegistry


class BaseCalculator(ABC):
    """Abstract base class for symbolic expression evaluators that support step-by-step evaluation."""

    def __init__(self, operation: [Operation | None] = None, rule_dict: [RuleDict | None] = None,
                 matcher_list: [MatcherList | None] = None) -> None:
        # Note: The following three attributes must be initialized in the subclass.
        self.operation: [Operation | None] = operation
        self.rule_dict: [RuleDict | None] = rule_dict
        self.matcher_list: [MatcherList | None] = matcher_list
        self._rule_registry = RuleRegistry()
        self.step_generator = BaseStepGenerator()
        self.processed: set = set()
        self.cache: dict = {}
        # Note: This list always contains exactly two elements.
        self.sudden_end: List[bool | Expr | None] = [False, None]
        # Only limit require rationalization of the denominator,
        # Because rationalizing the denominator facilitates solving limit.
        self.is_radsimp: bool = False
        self._validate_properties()
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Register all rules and matchers."""
        self._rule_registry.register_all(self.rule_dict, self.matcher_list)

    def _validate_properties(self) -> None:
        """Validate the key properties of the calculator."""
        required_attrs = ['operation', 'rule_dict', 'matcher_list']
        missing_attrs = [attr for attr in required_attrs if getattr(self, attr) is None]
        if missing_attrs:
            raise ValueError(f"Attributes {missing_attrs} must be initialized in init_key_property().")

    def reset_process(self) -> None:
        """Reset internal state to prepare for a new calculation.
        Clears the set of processed expressions and resets the step generator."""

        self.processed = set()
        self.step_generator.reset()
        self.sudden_end = [False, None]

    @staticmethod
    def _context_split(**context: Context) -> Symbol:
        """Only fit Derivative, Integral."""
        return context.get('variable')

    def _perform_operation(self, expr: Expr, operation: Operation, **context: Context) -> Operation:
        """Perform the specified operation on the expression.
        Only fit Derivative, Integral."""

        var = self._context_split(**context)
        return operation(expr, var)

    def _get_cached_operation(self, expr: Expr, operation: Operation, **context: Context) -> Operation:
        """Return a cached operation if available, otherwise compute and cache the operation."""
        key = (expr, str(context))
        if key not in self.cache:
            self.cache[key] = self._perform_operation(expr, operation, **context)
        return self.cache[key]

    @staticmethod
    def _step_expr_postprocess(step_expr: Expr) -> Expr:
        """Postprocess a step expression before adding it to the step generator.
        E.g., IntegralCalculator: Add constant of integration"""

        return step_expr

    @lru_cache(maxsize=128)
    def _cached_simplify(self, expr: Expr) -> Expr:
        """Return a simplified version of the expression, using caching to avoid redundant computation.
        Note: Force log simplification."""

        if self.is_radsimp:
            return expand_log(radsimp(expr), force=True)
        return expand_log(expr, force=True)

    def _get_context_dict(self, **context: Context) -> RuleContext:
        context_dict = {}
        for key, value in context.items():
            context_dict[key] = value
        # Add step_generator to the rule context to allow rule functions to control steps freely by accessing the
        # step generator.
        context_dict['step_generator'] = self.step_generator
        return context_dict

    @staticmethod
    def _check_rule_is_can_apply(_rule: RuleFunction) -> bool:
        return True

    def _apply_rule(self, expr: Expr, operation: Operation, **context: Context) -> Tuple[Expr, str]:
        """Apply the most appropriate rule to the expression and return result with explanation."""
        rule_context: RuleContext = self._get_context_dict(**context)

        for rule in self._rule_registry.get_applicable_rules(expr, rule_context):
            if not self._check_rule_is_can_apply(rule):
                continue

            result = rule(expr, rule_context)
            if result:
                print(f"rule: {rule.__name__}")
                return result

        # Fallback to SymPy if no rule matches
        operation_obj = self._perform_operation(expr, operation, **context)
        operation_val = operation_obj.doit()
        if not is_elementary_expression(operation_val):
            self.sudden_end = [True, operation_obj]
        return operation_val, f"需手动计算表达式: ${latex(operation_obj)}$"

    def _update_expression(self, current_expr: Expr | Matrix, operation: Operation,
                           expr_to_operation: Dict[str | Expr, Operation], direct_compute: bool, **context: Context) \
            -> Tuple[Expr, str, Dict[Expr, Operation]]:

        current_operation = self._get_cached_operation(current_expr, operation, **context)
        if direct_compute:
            new_expr = current_operation.doit()
            if not is_elementary_expression(new_expr):
                self.sudden_end = [True, current_operation]
            explanation = f"${latex(current_operation)}$ 之前已计算过，不再显示中间过程"
        else:
            new_expr, explanation = self._apply_rule(current_expr, operation, **context)
        # Replace all occurrences of operation(current, var) with new_expr
        for key in list(expr_to_operation.keys()):
            expr_to_operation[key] = expr_to_operation[key].subs(current_operation, new_expr)
        return new_expr, explanation, expr_to_operation

    def _back_subs(self, final_expr: Expr) -> Expr:
        """Perform back substitution by iterating through the substitution dictionary in reverse order.

        The keys are post-substitution variables and values are pre-substitution expressions.
        Since substitutions were added in order, back substitution requires reverse iteration.

        Args:
            final_expr: The final expression to perform back substitution on."""

        subs_dict = self.step_generator.subs_dict
        if not subs_dict:
            return final_expr
        # Iterate through the substitution dictionary in reverse order,
        # This ensures proper back substitution since later substitutions depend on earlier ones.
        for key, value in reversed(subs_dict.items()):
            final_expr = final_expr.subs(key, value)

        self.step_generator.add_step(final_expr, "回代换元变量")

        return simplify(final_expr, inverse=True)

    def final_postprocess(self, final_expr: Expr) -> None:
        """Apply domain-aware simplification by assuming all free symbols are positive real numbers.

        This step helps reduce expressions like sqrt(x^2) to x, log(x^2)/2 to log(x), etc.,
        which SymPy avoids under generic assumptions to preserve mathematical correctness."""

        if not final_expr.free_symbols:
            return

        # 1. Back substitute
        self._back_subs(final_expr)

        # # 2. Assume the variable are positive real numbers and simplify final expression
        # var = final_expr.free_symbols.pop()

        # _t = Symbol('t', real=True, positive=True)
        # simplified_expr = simplify(final_expr.subs(var, _t).subs(_t, var).replace(
        #     Abs, lambda arg: arg))
        # if simplified_expr != final_expr:
        #     self.step_generator.add_step(simplified_expr, "直接去掉所有绝对值, 再次化简表达式")

    def _sympify(self, expr: str) -> Expr:
        """Convert the input expression to a SymPy expression."""
        return sympify(expr)  # type: ignore

    def _do_compute(self, expr: str, operation: Operation, **context: Context) -> None:
        """Perform the core symbolic computation and record each evaluation step."""
        self.reset_process()
        expr = self._sympify(expr)
        # Extract the unique variable from the expression for calculating.
        symbol_list = list(expr.free_symbols)
        if len(symbol_list) > 1:
            raise ValueError("仅允许出现一个字母变量.")
        context['variable'] = symbol_list[0] if symbol_list else Symbol("x")

        initial_operation = self._get_cached_operation(expr, operation, **context)
        self.step_generator.add_step(initial_operation)

        simple_expr = self._cached_simplify(expr)
        if simple_expr != expr:
            expr = simple_expr
            initial_operation = self._get_cached_operation(expr, operation, **context)
            self.step_generator.add_step(initial_operation, "简化表达式")

        # BFS using a queue.
        queue: Deque[Expr] = deque([expr])
        expr_to_operation: Dict[Expr, Operation] = {expr: initial_operation}

        while queue:
            direct_compute = False

            current_expr = queue.popleft()
            current_operation = expr_to_operation.get(current_expr)

            if current_expr in self.processed:
                direct_compute = True
            self.processed.add(current_expr)

            # Extract the unique variable from the expression for substitution.
            symbol_list = list(current_expr.free_symbols)
            if symbol_list:
                context['variable'] = symbol_list[0]
            elif self.step_generator.subs_dict:
                context['variable'] = list(self.step_generator.subs_dict.keys())[-1]
            else:
                context['variable'] = Symbol("x")

            new_expr, explanation, expr_to_operation = self._update_expression(
                current_expr, operation, expr_to_operation, direct_compute, **context)

            current_step = expr_to_operation[expr]
            current_step = self._step_expr_postprocess(current_step)

            if self.sudden_end[0]:
                self.step_generator.add_step(
                    None, f'${latex(self.sudden_end[1])}$ 无法用初等函数表示, 退出计算.')
                break

            self.step_generator.add_step(current_step, explanation)

            if new_expr != current_operation:
                # Extract sub-expressions to continue processing.
                for s in list(new_expr.atoms(operation)):
                    sub_expr: Expr = s.args[0]  # type: ignore
                    expr_to_operation[sub_expr] = s
                    queue.append(sub_expr)

        if not self.sudden_end[0]:
            exprs, _ = self.step_generator.get_steps()
            final_expr = exprs[-1]

            # Final simplification
            simplified_expr = simplify(final_expr, inverse=True)
            if simplified_expr != final_expr:
                self.step_generator.steps[-1] = simplified_expr
                final_expr = simplified_expr

            self.final_postprocess(final_expr)

    def _compute(self, expr: str, **context: Context) -> None:
        """Compute the step-by-step evaluation of the given expression.

        Args:
            expr: A string representation of the symbolic expression to evaluate.
            **context: The evaluation context."""

        self._do_compute(expr, self.operation, **context)

    @abstractmethod
    def compute_list(self, expr: str, **context: Context) -> Tuple[List[Expr], List[str]]:
        """Compute the step-by-step evaluation of the given expression and return it as a tuple.

        Args:
            expr: A string representation of the symbolic expression to evaluate.
            **context: The evaluation context.

        Returns:
            Tuple:
            - A list of symbolic expressions representing each evaluation step.
            - A list of strings describing each step."""

        self._compute(expr, **context)
        return self.step_generator.get_steps()

    @abstractmethod
    def compute_latex(self, expr: str, **context: Context) -> str:
        """Compute the step-by-step evaluation of the given expression and return it as a LaTeX string.

        Args:
            expr: A string representation of the symbolic expression to evaluate.
            **context: The evaluation context, which can include variables, points and directions.

        Returns:
            A LaTeX string representing the step-by-step evaluation process.
            To render it in a Jupyter notebook, use: ``display(Math(latex_string))``."""

        self._compute(expr, **context)
        return self.step_generator.to_latex()
