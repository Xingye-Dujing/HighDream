"""Base class for the web-UI manual step-by-step solver.

Wraps a Select*Calculator and exposes a non-blocking API that lets a web
client pick which rule to apply at every derivation step. Domain-specific
subclasses (diff / integral / limit) live under ``domains/`` and only
override configuration (calculator class, rule display names, any
domain-specific hooks).
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from sympy import Expr, Symbol, latex, sympify

from utils import Operation


class BaseManualStepSolver:
    """Orchestrates step-by-step computation with explicit rule choices.

    This class directly uses the corresponding Select*Calculator for rule
    registration, caching, and operation execution. It does not re-implement
    the core symbolic computation; instead, it manages the BFS queue and step
    recording manually to allow user-driven rule selection.
    """

    rule_display_names: Dict[str, str] = {}  # rule_key -> Chinese name

    def __init__(self, expression: str, variable: str = 'x',
                 point: Any = 0, direction: str = '+') -> None:
        self.calculator = self._create_calculator()
        self._init_calculator()

        self.variable = Symbol(variable)
        self.point = sympify(point)
        self.direction = direction

        expr = self.calculator._sympify(expression)
        simple = self.calculator._cached_simplify(expr)
        simplified = simple != expr
        if simplified:
            expr = simple

        operation = self.calculator.operation
        initial_op = self.calculator._perform_operation(
            expr, operation, **self._build_context(expr))

        self.calculator.reset_process()
        self.calculator.step_generator.add_step(initial_op)
        if simplified:
            self.calculator.step_generator.add_step(initial_op, "简化表达式")

        self.steps: List[Expr] = list(self.calculator.step_generator.steps)
        self.explanations: List[str] = list(
            self.calculator.step_generator._explanations)

        self.top_expr: Expr = expr
        self.expr_to_operation: Dict[Expr, Operation] = {expr: initial_op}

        self.pending: Deque[Expr] = deque()
        self.current_expr: Optional[Expr] = expr

        self.done: bool = False
        self.error: Optional[str] = None

        self._visited: set = set()

    def _init_calculator(self) -> None:
        """Hook for subclasses to perform domain-specific calculator setup."""

    def _create_calculator(self):
        """Return a freshly-constructed Select*Calculator for this domain.

        Subclasses MUST override this.
        """
        raise NotImplementedError(
            "Subclasses must override `_create_calculator`.")

    # ------------------------------------------------------------------ context

    def _build_context(self, expr: Expr) -> Dict[str, Any]:
        """Build the kwargs dict that the wrapped calculator expects."""
        ctx: Dict[str, Any] = {'variable': self._current_variable(expr)}
        self._extend_context(ctx, expr)
        return ctx

    def _extend_context(self, ctx: Dict[str, Any], expr: Expr) -> None:
        """Hook for subclasses to add domain-specific context entries."""

    def _current_variable(self, expr: Expr) -> Symbol:
        """Return the primary free symbol of expr, falling back to self.variable."""
        syms = list(expr.free_symbols)
        if syms:
            return syms[0]
        return self.variable

    # --------------------------------------------------------------- public API

    def state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the current solver state."""
        rules = self.applicable_rules() if not self.done else []
        current_latex: Optional[str] = None
        try:
            current_latex = latex(self.current_expr)
        except TypeError:
            current_latex = repr(self.current_expr)
        except Exception:
            pass
        return {
            'done': self.done,
            'error': self.error,
            'domain': self.domain,
            'top_level_latex': latex(self._current_top_operation()),
            'current_expr_latex': current_latex,
            'pending': [
                latex(self.calculator._get_cached_operation(
                    sub, self.calculator.operation, **self._build_context(sub))
                ) for sub in self.pending
            ],
            'steps': [
                {'latex': latex(s) if s is not None else '',
                 'explanation': e}
                for s, e in zip(self.steps, self.explanations)
            ],
            'applicable_rules': rules,
        }

    _domain: str = ''  # overridden by subclasses ('diff', 'integral', 'limit')

    @property
    def domain(self) -> str:
        """Stable domain identifier used by the API layer."""
        return self._domain

    def applicable_rules(self) -> List[Dict[str, str]]:
        """List rules that can be applied to the current expression."""
        if self.done or self.current_expr is None:
            return []
        ctx = self._build_context(self.current_expr)
        rule_ctx = self.calculator._get_context_dict(**ctx)
        registry_rules = self.calculator._rule_registry._rules

        saved_snapshot = self._snapshot_solver_state()

        out: List[Dict[str, str]] = []
        seen = set()
        try:
            for rule in self.calculator._rule_registry.get_applicable_rules(
                    self.current_expr, rule_ctx):
                if not self.calculator._check_rule_is_can_apply(rule):
                    continue
                rule_key = None
                for key, func in registry_rules.items():
                    if func is rule:
                        rule_key = key
                        break
                if rule_key is None or rule_key in seen:
                    continue
                seen.add(rule_key)
                display = self.rule_display_names.get(rule_key, rule_key)

                backup = self._backup_step_generator()
                try:
                    preview_result = rule(self.current_expr, rule_ctx)
                    if preview_result:
                        preview_expr, _ = preview_result
                        preview_latex = latex(preview_expr)
                        out.append({
                            'name': rule_key,
                            'display_name': display,
                            'latex_preview': preview_latex,
                        })
                except Exception:
                    out.append({
                        'name': rule_key,
                        'display_name': display,
                        'latex_preview': None,
                    })
                finally:
                    self._restore_step_generator(backup)
        finally:
            self._restore_solver_state(saved_snapshot)
        return out

    def apply_rule(self, rule_name: str) -> Dict[str, Any]:
        self.error = None
        if self.done:
            raise RuntimeError("Solver already finished.")
        if self.current_expr is None:
            raise RuntimeError("No current expression.")

        registry_rules = self.calculator._rule_registry._rules
        if rule_name not in registry_rules:
            raise KeyError(f"Rule not registered: {rule_name}")
        rule = registry_rules[rule_name]

        if not self.calculator._check_rule_is_can_apply(rule):
            self.error = (f"规则 '{rule_name}' 已被限制 (例如 l'Hôpital 次数超限), "
                          "请选择其他规则或使用 SymPy 回退.")
            return self.state()

        ctx = self._build_context(self.current_expr)
        rule_ctx = self.calculator._get_context_dict(**ctx)

        step_count_before = len(self.calculator.step_generator.steps)
        try:
            result = rule(self.current_expr, rule_ctx)
        except Exception as e:
            self.error = f"Rule '{rule_name}' raised: {e}"
            return self.state()

        if not result:
            self.error = (f"规则 '{rule_name}' 无法应用到当前表达式, "
                          "请重新选择或使用 SymPy 回退.")
            del self.calculator.step_generator.steps[step_count_before:]
            del self.calculator.step_generator._explanations[step_count_before:]
            self._sync_from_generator()
            return self.state()

        new_expr, explanation = result
        self._commit_step(self.current_expr, new_expr, explanation)
        self._advance_pending()
        return self.state()

    def fallback(self) -> Dict[str, Any]:
        """Use SymPy's .doit() on the current sub-expression."""
        if self.done:
            raise RuntimeError("Solver already finished.")
        if self.current_expr is None:
            raise RuntimeError("No current expression.")

        ctx = self._build_context(self.current_expr)
        operation_obj = self.calculator._perform_operation(
            self.current_expr, self.calculator.operation, **ctx)
        new_expr = operation_obj.doit()
        explanation = f"手动计算（SymPy 回退）: ${latex(operation_obj)}$"
        self._commit_step(self.current_expr, new_expr, explanation)
        self._advance_pending()
        return self.state()

    def finish(self) -> Dict[str, Any]:
        """Run final post-processing (back-substitution + simplify)."""
        if self.done:
            return self.state()
        self.done = True
        if not self.steps:
            return self.state()
        final_expr = self.steps[-1]
        try:
            self.calculator.final_postprocess(final_expr)
        except Exception:
            pass
        self._sync_from_generator()
        return self.state()

    # ---------------------------------------------------------------- internals

    def _current_top_operation(self) -> Expr:
        return self.expr_to_operation.get(self.top_expr, self.top_expr)

    def _sync_from_generator(self) -> None:
        self.steps = list(self.calculator.step_generator.steps)
        self.explanations = list(self.calculator.step_generator._explanations)

    def _commit_step(self, old_expr: Expr, new_expr: Expr,
                     explanation: str) -> None:
        self._visited.add(old_expr)

        self._sync_from_generator()

        old_op = self.calculator._get_cached_operation(
            old_expr, self.calculator.operation,
            **self._build_context(old_expr))
        for key in list(self.expr_to_operation.keys()):
            self.expr_to_operation[key] = (
                self.expr_to_operation[key].subs(old_op, new_expr))

        current_top = self._current_top_operation()
        if not self.steps or self.steps[-1] != current_top:
            self.calculator.step_generator.add_step(current_top, explanation)
            self._sync_from_generator()

        for atom in list(new_expr.atoms(self.calculator.operation)):
            sub = atom.args[0]
            self.expr_to_operation[sub] = atom
            self.pending.append(sub)

    def _advance_pending(self) -> None:
        while self.pending:
            nxt = self.pending.popleft()
            if nxt in self._visited:
                ctx = self._build_context(nxt)
                operation_obj = self.calculator._perform_operation(
                    nxt, self.calculator.operation, **ctx)
                new_expr = operation_obj.doit()
                explanation = (f"${latex(operation_obj)}$ 之前已计算过, "
                               "不再显示中间过程")
                self._commit_step(nxt, new_expr, explanation)
                continue
            self.current_expr = nxt
            self._visited.add(nxt)
            return

        self.current_expr = None
        self.finish()

    def _backup_step_generator(self) -> Dict[str, Any]:
        gen = self.calculator.step_generator
        return {
            'steps': list(gen.steps),
            'explanations': list(gen._explanations),
            'subs_dict': dict(gen.subs_dict),
            'available_sym_chars': list(gen.available_sym_chars),
        }

    def _restore_step_generator(self, backup: Dict[str, Any]) -> None:
        gen = self.calculator.step_generator
        gen.steps = backup['steps']
        gen._explanations = backup['explanations']
        gen.subs_dict = backup['subs_dict']
        gen.available_sym_chars = backup['available_sym_chars']

    def _snapshot_solver_state(self) -> Dict[str, Any]:
        """Hook-friendly snapshot of solver-wide state (not step_generator).

        Used around ``applicable_rules`` so listing rules does not mutate
        domain-specific bookkeeping (e.g. l'Hôpital cap counters).
        """
        return {}

    def _restore_solver_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore whatever ``_snapshot_solver_state`` captured."""
