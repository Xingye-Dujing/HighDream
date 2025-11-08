from abc import ABC, abstractmethod
from collections import deque
from functools import lru_cache
from typing import Deque, Dict, List, Tuple

from sympy import Expr, Symbol, latex, log, simplify, sympify

from utils import Context, MatcherList, Operation, RuleContext, RuleDict, RuleFunction
from .base_step_generator import BaseStepGenerator
from .rule_registry import RuleRegistry


class BaseCalculator(ABC):
    """Abstract base class for symbolic expression evaluators that support step-by-step evaluation."""

    def __init__(self) -> None:
        self.rule_registry = RuleRegistry()
        self.step_generator = BaseStepGenerator()
        self.processed: set = set()
        self.cache: dict = {}
        self.init_key_property()
        self._validate_properties()
        self._initialize_rules()

    @abstractmethod
    def init_key_property(self) -> None:
        """Initialize key properties of the calculator.

        These attributes will be validated after this method is called.
        """
        self.operation: Operation = None
        self.rule_dict: RuleDict = None
        self.matcher_list: MatcherList = None

    def _validate_properties(self) -> None:
        """Validate the key properties of the calculator."""
        required_attrs: List[str] = ['operation', 'rule_dict', 'matcher_list']
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(
                    f"Attribute '{attr}' must be initialized in init_key_property")

    def reset_process(self) -> None:
        """Reset internal state to prepare for a new calculation.

        Clears the set of processed expressions and resets the step generator.
        """
        self.processed = set()
        self.step_generator.reset()

    def _context_split(self, **context: Context) -> Symbol:
        return context.get('variable', Symbol('x'))

    def _perform_operation(self, expr: Expr, operation: Operation, **context: Context) -> Operation:
        """Perform the specified operation on the expression and return the result.

        Only fit Derivative, Integral."""
        var = self._context_split(**context)
        return operation(expr, var)

    def _get_cached_result(self, expr: Expr, operation: Operation, **context: Context) -> Operation:
        """Return a cached result if available, otherwise compute and cache the result."""
        key = (str(expr), str(context))
        if key not in self.cache:
            self.cache[key] = self._perform_operation(
                expr, operation, **context)
        return self.cache[key]

    def _preprocess_expression(self, expr: Expr) -> Expr:
        """Apply custom simplifications not covered by SymPy's simplify."""
        # ln(1/f(x)) = -ln(f(x))
        expr = expr.replace(
            lambda e: e.is_Function and e.func == log and len(
                e.args) == 1 and e.args[0].is_Pow and e.args[0].args[1] == -1,
            lambda e: -log(e.args[0].args[0])
        )
        return expr

    @staticmethod
    def _step_expr_postprocess(step_expr: Expr) -> Expr:
        """Postprocess a step expression before adding it to the step generator.

        eg. IntegralCalculator: Add constant of integration
        """
        return step_expr

    @lru_cache(maxsize=128)
    def _cached_simplify(self, expr: Expr) -> Expr:
        """Return a simplified version of the expression, using caching to avoid redundant computation."""
        return self._preprocess_expression(simplify(expr))

    def _initialize_rules(self) -> None:
        """Register all rules and matchers."""
        self.rule_registry.register_all(self.rule_dict, self.matcher_list)

    def _get_context_dict(self, **context: Context) -> RuleContext:
        context_dict = {}
        for key, value in context.items():
            context_dict[key] = value
        return context_dict

    def _check_rule_is_can_apply(self, _rule: RuleFunction) -> bool:
        return True

    def _apply_rule(self, expr: Expr, operation: Operation, **context: Context) -> Tuple[Expr, str]:
        """Apply the most appropriate rule to the expression and return result with explanation."""
        rule_context: RuleContext = self._get_context_dict(**context)

        for rule in self.rule_registry.get_applicable_rules(expr, rule_context):
            if not self._check_rule_is_can_apply(rule):
                continue
            # print(f"rule: {rule.__name__}")
            result = rule(expr, rule_context)
            if result:
                return result

        # Fallback to SymPy if no rule matches
        operation_obj = self._perform_operation(expr, operation, **context)
        return operation_obj.doit(), f"需手动计算表达式: ${latex(operation_obj)}$"

    def _update_expression(self, current_expr: Expr, operation: Operation, expr_to_operation: Dict[Expr, Operation], **context: Context) -> Tuple[Expr, str, Dict[Expr, Operation]]:
        # Extract the unique variable from the expression for substitution(u-substitution).
        # context['variable'] = list(current_expr.free_symbols)[0]
        new_expr, explanation = self._apply_rule(
            current_expr, operation, **context)
        # Replace all occurrences of operation(current, var) with new_expr
        for key in list(expr_to_operation.keys()):
            expr_to_operation[key] = expr_to_operation[key].subs(
                self._get_cached_result(
                    current_expr, operation, **context), new_expr
            )
        return new_expr, explanation, expr_to_operation

    def _final_postprocess(self, _final_expr: Expr) -> None:
        pass

    def _do_compute(self, expr: str, operation: Operation, **context: Context) -> None:
        """Perform the core symbolic computation and record each evaluation step."""
        self.reset_process()
        expr = sympify(expr)

        initial_operation = self._get_cached_result(expr, operation, **context)
        self.step_generator.add_step(initial_operation)

        simple_expr = self._cached_simplify(expr)
        if simple_expr != expr:
            expr = simple_expr
            initial_operation = self._get_cached_result(
                expr, operation, **context)
            self.step_generator.add_step(initial_operation, "简化表达式")

        # BFS using a queue.
        queue: Deque[Expr] = deque([expr])
        expr_to_operation: Dict[Expr, Operation] = {expr: initial_operation}
        # Avoid recomputing duplicate expressions.
        seen = set([expr])

        while queue:
            current_expr = queue.popleft()
            current_operation = expr_to_operation.get(current_expr)

            if current_expr in self.processed:
                continue
            self.processed.add(current_expr)

            new_expr, explanation, expr_to_operation = self._update_expression(
                current_expr, operation, expr_to_operation, **context)

            current_step = expr_to_operation[expr]
            current_step = self._step_expr_postprocess(current_step)
            self.step_generator.add_step(current_step, explanation)

            if new_expr != current_operation:
                # Extract sub-expressions to continue processing.
                sub_exprs = [new_expr] if isinstance(
                    new_expr, operation) else new_expr.atoms(operation)
                for s in sub_exprs:
                    sub_expr = s.args[0]
                    if sub_expr not in seen:
                        seen.add(sub_expr)
                        expr_to_operation[sub_expr] = s
                        queue.append(sub_expr)

        # Final simplification
        exprs, _ = self.step_generator.get_steps()
        if exprs:
            final_expr = exprs[-1]
            simplified_expr = self._cached_simplify(final_expr)
            if simplified_expr != final_expr:
                self.step_generator.add_step(simplified_expr, "简化表达式")

        self._final_postprocess(final_expr)

    def _compute(self, expr: str, **context: Context) -> None:
        """Compute the step-by-step evaluation of the given expression.

        Args:
            expr: A string representation of the symbolic expression to evaluate.
            **context: The evaluation context.
        """
        self._do_compute(expr, self.operation, **context)

    def compute_list(self, expr: str, **context: Context) -> Tuple[List[Expr], List[str]]:
        """Compute the step-by-step evaluation of the given expression and return it as a tuple.

        Args:
            expr: A string representation of the symbolic expression to evaluate.
            **context: The evaluation context.

        Returns:
            Tuple:
            - A list of symbolic expressions representing each evaluation step.
            - A list of strings describing each step.
        """
        self._compute(expr, **context)
        return self.step_generator.get_steps()

    def compute_latex(self, expr: str, **context: Context) -> str:
        """Compute the step-by-step evaluation of the given expression and return it as a LaTeX string.

        Args:
            expr: A string representation of the symbolic expression to evaluate.
            **context: The evaluation context, which can include variables, points, and directions.

        Returns:
            A LaTeX string representing the step-by-step evaluation process.
            To render it in a Jupyter notebook, use: ``display(Math(latex_string))``.
        """
        self._compute(expr, **context)
        return self.step_generator.to_latex()
