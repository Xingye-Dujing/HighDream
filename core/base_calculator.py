from abc import ABC, abstractmethod
from collections import deque
from functools import lru_cache
from typing import Deque, Dict, List, Tuple

from sympy import Expr, Matrix, Symbol

from utils import (
    Context, MatcherList, Operation, RuleContext,
    RuleDict, RuleFunction, is_elementary_expression
)
from utils.cache_utils import (
    cached_doit, cached_latex, cached_simplify,
    cached_simplify_radsimp, cached_sympify,
)
from utils.assumptions import refine_with_flags
from .base_step_generator import BaseStepGenerator
from .rule_registry import RuleRegistry


@lru_cache(maxsize=2048)
def _cached_perform_operation(expr: Expr, operation_cls_name: str,
                               var_name: str, point_str: str, direction: str) -> Operation:
    from sympy import Derivative, Integral, Limit  # pylint: disable=import-outside-toplevel
    mapping = {'Derivative': Derivative, 'Integral': Integral, 'Limit': Limit}
    cls = mapping.get(operation_cls_name)
    if cls is None:
        raise ValueError(f"Unknown operation: {operation_cls_name}")
    var = Symbol(var_name) if var_name else Symbol('x')
    if cls is Limit:
        return cls(expr, var, cached_sympify(point_str), dir=direction)
    return cls(expr, var)


class BaseCalculator(ABC):
    """Abstract base class for symbolic expression evaluators that support step-by-step evaluation."""

    # When False the engine accepts only a single free symbol (existing behavior).
    # Differentiation enables this to support partial derivatives, keeping the
    # requested variable fixed and treating all other symbols as constants.
    _allow_multi_symbols: bool = False

    # When True the integral engine honors the requested variable and treats every
    # other free symbol as a constant parameter (e.g. ``a`` in sqrt(a^2 - x^2)).
    # Opt-in and instance-level: only the user-facing request path flips it. Internal
    # callers (ODE reduction, applied ops) keep it False so their >1-symbol integrals
    # still raise and trigger their own fallbacks, exactly as before.
    _allow_parameters: bool = False

    def __init__(self, operation: List[Operation | None] = None, rule_dict: List[RuleDict | None] = None,
                 matcher_list: List[MatcherList | None] = None) -> None:
        # Note: The following three attributes must be initialized in the subclass.
        self.operation: List[Operation | None] = operation
        self.rule_dict: List[RuleDict | None] = rule_dict
        self.matcher_list: List[MatcherList | None] = matcher_list
        self._rule_registry = RuleRegistry()
        self.step_generator = BaseStepGenerator()
        self.processed: set = set()
        self.cache: dict = {}
        # Note: This list always contains exactly two elements.
        self.sudden_end: List[bool | Expr | None] = [False, None]
        # Only limit require rationalization of the denominator,
        # Because rationalizing the denominator facilitates solving limit.
        self.is_radsimp: bool = False
        # Optional user assumptions (e.g. x>0) applied as a final, honest simplify
        # step in final_postprocess; set from the outside (per calculator instance)
        # so no compute_latex/compute_list signature has to change.
        self.assum_flags: dict = {}
        self.assumption_note: str = ''
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
        try:
            op_name = operation.__name__
        except AttributeError:
            op_name = operation.__class__.__name__
        var = str(context.get('variable', 'x'))
        point = str(context.get('point', '0'))
        direction = str(context.get('direction', '+'))
        return _cached_perform_operation(expr, op_name, var, point, direction)

    @staticmethod
    def _step_expr_postprocess(step_expr: Expr) -> Expr:
        """Postprocess a step expression before adding it to the step generator.
        E.g., IntegralCalculator: Add constant of integration"""

        return step_expr

    def _cached_simplify(self, expr: Expr) -> Expr:
        """Return a simplified version of the expression, using caching to avoid redundant computation.
        Note: Force log simplification."""

        return cached_simplify_radsimp(expr, self.is_radsimp)

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

            try:
                result = rule(expr, rule_context)
            except Exception:  # noqa
                # A single rule raising (e.g. a SymPy oracle hitting RecursionError on a
                # pathological expression) must not abort the whole solve; skip it and let
                # the next applicable rule (or the doit fallback) handle the expression.
                continue
            if result:
                print(f"rule: {rule.__name__}")
                return result

        # Fallback to SymPy if no rule matches
        operation_obj = self._perform_operation(expr, operation, **context)
        operation_val = cached_doit(operation_obj)
        if not is_elementary_expression(operation_val):
            self.sudden_end = [True, operation_obj]
        return operation_val, f"需手动计算表达式: ${cached_latex(operation_obj)}$"

    def _update_expression(self, current_expr: Expr | Matrix, operation: Operation,
                           expr_to_operation: Dict[str | Expr, Operation], direct_compute: bool,
                           stored_operation: Operation = None, **context: Context) \
            -> Tuple[Expr, str, Dict[Expr, Operation]]:

        # Use the ACTUAL stored operation (it carries the substituted dummy's assumptions,
        # e.g. Symbol(u, real=True, positive=True)) so the write-back subs() below matches;
        # rebuilding it via _get_cached_operation drops those assumptions and silently no-ops.
        current_operation = stored_operation if stored_operation is not None \
            else self._get_cached_operation(current_expr, operation, **context)
        if direct_compute:
            new_expr = cached_doit(current_operation)
            if not is_elementary_expression(new_expr):
                self.sudden_end = [True, current_operation]
            explanation = f"${cached_latex(current_operation)}$ 之前已计算过，不再显示中间过程"
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

        return cached_simplify(final_expr, inverse=True)

    def final_postprocess(self, final_expr: Expr) -> None:
        """Apply domain-aware simplification by assuming all free symbols are positive real numbers.

        This step helps reduce expressions like sqrt(x^2) to x, log(x^2)/2 to log(x), etc.,
        which SymPy avoids under generic assumptions to preserve mathematical correctness."""

        if not final_expr.free_symbols:
            return

        # 1. Back substitute
        back_substituted = self._back_subs(final_expr)
        # The back-substitution step was recorded with the unsimplified expression;
        # write the simplified form back so the final answer shown is the one the
        # step claims to produce.
        if back_substituted is not final_expr and self.step_generator.steps:
            self.step_generator.steps[-1] = back_substituted

        # 2. Apply user assumptions (e.g. x>0) as an explicit, honest final step so
        #    sqrt(x^2)->x, log(x^2)/2->log(x), etc. Soundly confined: only symbols the
        #    user constrained are simplified; others are left untouched.
        if self.assum_flags:
            steps = self.step_generator.steps
            base = steps[-1] if steps and steps[-1] is not None else final_expr
            refined = refine_with_flags(base, self.assum_flags)
            if refined != base:
                note = "依假设化简最终结果"
                if self.assumption_note:
                    note += f": {self.assumption_note}"
                self.step_generator.add_step(refined, note)

    def _sympify(self, expr: str) -> Expr:
        """Convert the input expression to a SymPy expression."""
        return cached_sympify(expr)

    def _do_compute(self, expr: str, operation: Operation, **context: Context) -> None:
        """Perform the core symbolic computation and record each evaluation step."""
        self.reset_process()
        expr = self._sympify(expr)
        symbol_list = list(expr.free_symbols)
        passed_var = context.get('variable')

        if self._allow_multi_symbols:
            # Partial differentiation: honor the explicitly requested variable and
            # treat every other free symbol as a constant.
            if isinstance(passed_var, Symbol):
                context['variable'] = passed_var
            elif symbol_list:
                context['variable'] = symbol_list[0]
            else:
                context['variable'] = Symbol("x")
        else:
            # Single-variable behavior unless the caller opted into parameters.
            if len(symbol_list) > 1:
                if not self._allow_parameters:
                    raise ValueError("仅允许出现一个字母变量.")
                # Honor the requested variable and treat the rest as constants; the
                # per-node re-derivation below still tracks the working variable
                # through substitutions via the operation's own variables.
                context['variable'] = passed_var if isinstance(passed_var, Symbol) \
                    else symbol_list[0]
            elif isinstance(passed_var, Symbol):
                # The caller named the working variable (the integral UI's variable
                # selector): honor it even when the integrand does not contain it.
                # Otherwise int(2, t) silently became int(2, x).
                context['variable'] = passed_var
            else:
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

            # Determine the working variable for this sub-expression. Prefer the real
            # integration/differentiation variable carried by the stored operation (it
            # preserves the substituted dummy's identity); only re-derive from free
            # symbols when the operation is unavailable (e.g. matrix path keys by str).
            if not self._allow_multi_symbols:
                op_vars = getattr(current_operation, 'variables', None) \
                    if current_operation is not None else None
                if op_vars:
                    context['variable'] = op_vars[0]
                else:
                    symbol_list = list(current_expr.free_symbols)
                    if symbol_list:
                        context['variable'] = symbol_list[0]
                    elif self.step_generator.subs_dict:
                        context['variable'] = list(self.step_generator.subs_dict.keys())[-1]
                    else:
                        context['variable'] = Symbol("x")
            else:
                # Partial integration keeps the requested variable (other free
                # symbols are constants), BUT a u-substitution inside it introduces
                # a dummy that must be integrated over. Without this, ∫(x+y+z)dx
                # became ∫u du displayed then evaluated as ∫u dx = u·x, and the
                # iterated/triple integral returned a wrong value.
                op_vars = getattr(current_operation, 'variables', None) \
                    if current_operation is not None else None
                if op_vars and op_vars[0] in self.step_generator.subs_dict:
                    context['variable'] = op_vars[0]

            new_expr, explanation, expr_to_operation = self._update_expression(
                current_expr, operation, expr_to_operation, direct_compute,
                stored_operation=current_operation, **context)

            current_step = expr_to_operation[expr]
            current_step = self._step_expr_postprocess(current_step)

            if self.sudden_end[0]:
                self.step_generator.add_step(
                    None, f'${cached_latex(self.sudden_end[1])}$ 无法用初等函数表示, 退出计算.')
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
            simplified_expr = cached_simplify(final_expr, inverse=True)
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
