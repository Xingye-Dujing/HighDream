"""Manual step-by-step solver for the web UI.

Wraps a Select*Calculator and exposes a non-blocking API that lets a web
client pick which rule to apply at every derivation step.
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from sympy import Expr, Symbol, latex, sympify

from domains import (
    SelectDiffCalculator, SelectIntegralCalculator, SelectLimitCalculator,
)
from utils import Operation


# Domain -> Select*Calculator class
_CALCULATOR_BY_DOMAIN = {
    'diff': SelectDiffCalculator,
    'integral': SelectIntegralCalculator,
    'limit': SelectLimitCalculator,
}

# Chinese display names for rule keys, organized by domain
_RULE_DISPLAY_NAMES: Dict[str, Dict[str, str]] = {
    'diff': {
        'add': '加法法则',
        'mul': '乘法法则',
        'div': '除法法则',
        'chain': '链式法则',
        'const': '常数求导',
        'var': '变量求导',
        'pow': '幂函数求导',
        'sin': '正弦求导',
        'cos': '余弦求导',
        'tan': '正切求导',
        'sec': '正割求导',
        'csc': '余割求导',
        'cot': '余切求导',
        'asin': '反正弦求导',
        'acos': '反余弦求导',
        'atan': '反正切求导',
        'exp': '指数求导',
        'log': '对数求导',
        'sinh': '双曲正弦求导',
        'cosh': '双曲余弦求导',
        'tanh': '双曲正切求导',
    },
    'integral': {
        'add': '加法展开',
        'const': '常数积分',
        'var': '变量积分',
        'mul_const': '常数乘法',
        'pow': '幂函数积分',
        'exp': '指数函数积分',
        'log': '对数函数积分',
        'sin': '正弦函数积分',
        'cos': '余弦函数积分',
        'tan': '正切函数积分',
        'sec': '正割函数积分',
        'csc': '余割函数积分',
        'cot': '余切函数积分',
        'sinh': '双曲正弦积分',
        'cosh': '双曲余弦积分',
        'tanh': '双曲正切积分',
        'sech': '双曲正割积分',
        'csch': '双曲余割积分',
        'coth': '双曲余切积分',
        'inverse_trig': '反三角函数积分',
        'inverse_tangent_linear': '线性反正切积分',
        'sin_power': '正弦幂积分',
        'cos_power': '余弦幂积分',
        'tan_power': '正切幂积分',
        'logarithmic': '对数函数积分',
        'parts': '分部积分法',
        'substitution': '换元积分法',
        'f_x_mul_exp_g_x': 'f(x)e^g(x) 型积分',
        'quotient_diff_form': '商微分形式',
        'quadratic_sqrt_reciprocal': '二次根式倒数积分',
        'sqrt_div_sqrt': '根式相除积分',
        'weierstrass_substitution': '万能公式代换',
    },
    'limit': {
        'sin_over_x': '重要极限 sin(x)/x',
        'one_plus_one_over_x_pow_x': '重要极限 (1+1/x)^x',
        'ln_one_plus_x_over_x': '重要极限 ln(1+x)/x',
        'exp_minus_one_over_x': '重要极限 (e^x-1)/x',
        'g_over_sin': 'g(x)/sin(x) 型',
        'g_over_ln_one_plus': 'g(x)/ln(1+x) 型',
        'g_over_exp_minus_one': 'g(x)/(e^x-1) 型',
        'mul_split': '乘法拆分',
        'add_split': '加法拆分',
        'div_split': '除法拆分',
        'direct_substitution': '直接代入',
        'conjugate_rationalize': '共轭有理化',
        'small_o_add': '小 o 加法',
        'const_inf_add': '常数+无穷',
        'const_inf_mul': '常数×无穷',
        'const_inf_div': '常数/无穷',
        'const_zero_div': '常数/零',
        'lhopital_direct': '洛必达法则 (0/0, ∞/∞)',
        'lhopital_zero_times_inf': '洛必达 (0·∞)',
        'lhopital_inf_minus_inf': '洛必达 (∞-∞)',
        'lhopital_power': '洛必达 (幂指型)',
        'pow': '幂指转换',
        'exp': '指数函数极限',
        'log': '对数函数极限',
        'sin': '正弦函数极限',
        'cos': '余弦函数极限',
        'tan': '正切函数极限',
        'sec': '正割函数极限',
        'csc': '余割函数极限',
        'cot': '余切函数极限',
        'asin': '反正弦极限',
        'acos': '反余弦极限',
        'atan': '反正切记限',
        'sinh': '双曲正弦极限',
        'cosh': '双曲余弦极限',
        'tanh': '双曲正切极限',
    },
}


class ManualStepSolver:
    """Orchestrates step-by-step computation with explicit rule choices."""

    def __init__(self, domain: str, expression: str, variable: str = 'x',
                 point: Any = 0, direction: str = '+') -> None:
        if domain not in _CALCULATOR_BY_DOMAIN:
            raise ValueError(f"Unsupported domain: {domain}. "
                             f"Expected one of {list(_CALCULATOR_BY_DOMAIN)}")

        self.domain = domain
        self.calculator = _CALCULATOR_BY_DOMAIN[domain]()
        # Limit uses a specialized step generator + rationalization setting
        if domain == 'limit':
            self.calculator._lhopital_count = 0

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

        # Reset the wrapped calculator's step generator so it starts clean.
        self.calculator.reset_process()
        self.calculator.step_generator.add_step(initial_op)
        if simplified:
            self.calculator.step_generator.add_step(initial_op, "简化表达式")

        # Mirrors of step_generator state (kept in sync for easy serialization)
        self.steps: List[Expr] = list(self.calculator.step_generator.steps)
        self.explanations: List[str] = list(
            self.calculator.step_generator._explanations)

        # Top-level expression whose Operation atoms get replaced as we go.
        self.top_expr: Expr = expr
        self.expr_to_operation: Dict[Expr, Operation] = {expr: initial_op}

        # BFS queue of sub-expressions discovered during rule application.
        self.pending: Deque[Expr] = deque()
        self.current_expr: Optional[Expr] = expr

        self.done: bool = False
        self.error: Optional[str] = None

        # Mirror the auto calculator: mark expressions we've already visited
        # so repeated occurrences are computed directly instead of expanded.
        self._visited: set = set()

    # ------------------------------------------------------------------ context

    def _build_context(self, expr: Expr) -> Dict[str, Any]:
        """Build the kwargs dict that the wrapped calculator expects."""
        ctx: Dict[str, Any] = {'variable': self._current_variable(expr)}
        if self.domain == 'limit':
            ctx['point'] = self.point
            ctx['direction'] = self.direction
        return ctx

    def _current_variable(self, _expr: Expr) -> Symbol:
        """Always use the solver's variable to ensure consistent Symbol identity
        across all Operation objects, so that subs() matching works correctly."""
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

    def applicable_rules(self) -> List[Dict[str, str]]:
        """List rules that can be applied to the current expression."""
        if self.done or self.current_expr is None:
            return []
        ctx = self._build_context(self.current_expr)
        rule_ctx = self.calculator._get_context_dict(**ctx)
        registry_rules = self.calculator._rule_registry._rules

        # Listing rules must not consume the l'Hôpital cap — snapshot and
        # restore the counter so only the actual apply_rule call increments it.
        saved_lhopital = getattr(self.calculator, '_lhopital_count', None)

        out: List[Dict[str, str]] = []
        seen = set()
        try:
            for rule in self.calculator._rule_registry.get_applicable_rules(
                    self.current_expr, rule_ctx):
                # Respect the limit calculator's l'Hôpital cap
                if not self.calculator._check_rule_is_can_apply(rule):
                    continue
                # Find the registry key for this rule function.
                rule_key = None
                for key, func in registry_rules.items():
                    if func is rule:
                        rule_key = key
                        break
                if rule_key is None or rule_key in seen:
                    continue
                seen.add(rule_key)
                names = _RULE_DISPLAY_NAMES.get(self.domain, {})
                display = names.get(rule_key, rule_key)

                # Compute preview by calling the rule with saved/restored state.
                # If the rule returns None it can't apply — skip it entirely.
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
                    # Rule raised — might still work in apply_rule, keep it.
                    out.append({
                        'name': rule_key,
                        'display_name': display,
                        'latex_preview': None,
                    })
                finally:
                    self._restore_step_generator(backup)
        finally:
            if saved_lhopital is not None:
                self.calculator._lhopital_count = saved_lhopital
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
            # Undo any step_generator mutations the rule performed.
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
            # Best-effort; surface whatever we have.
            pass
        self._sync_from_generator()
        return self.state()

    # ---------------------------------------------------------------- internals

    def _current_top_operation(self) -> Expr:
        """Top-level expression with all processed Operation atoms replaced."""
        return self.expr_to_operation.get(self.top_expr, self.top_expr)

    def _sync_from_generator(self) -> None:
        self.steps = list(self.calculator.step_generator.steps)
        self.explanations = list(self.calculator.step_generator._explanations)

    def _commit_step(self, old_expr: Expr, new_expr: Expr,
                     explanation: str) -> None:
        """Replace old_expr with new_expr in the top-level expression and
        record a step. Does NOT advance the pending queue — the caller
        (apply_rule / fallback) must call _advance_pending afterwards."""

        self._visited.add(old_expr)

        # Sync any steps the rule appended directly to step_generator.
        self._sync_from_generator()

        # Substitute into the top-level Operation map.
        old_op = self.calculator._get_cached_operation(
            old_expr, self.calculator.operation,
            **self._build_context(old_expr))
        for key in list(self.expr_to_operation.keys()):
            self.expr_to_operation[key] = (
                self.expr_to_operation[key].subs(old_op, new_expr))

        # Record a visible top-level step.
        current_top = self._current_top_operation()
        if not self.steps or self.steps[-1] != current_top:
            self.calculator.step_generator.add_step(current_top, explanation)
            self._sync_from_generator()

        # Enqueue newly-created Operation atoms from new_expr.
        for atom in list(new_expr.atoms(self.calculator.operation)):
            sub = atom.args[0]
            self.expr_to_operation[sub] = atom
            self.pending.append(sub)

    def _advance_pending(self) -> None:
        """Pop from the pending queue. If the popped sub-expression has been
        visited before, compute it directly (doit) and keep popping until we
        reach a fresh sub-expression or the queue empties."""
        while self.pending:
            nxt = self.pending.popleft()
            if nxt in self._visited:
                # Direct-compute for a repeated sub-expression (no rule UI).
                ctx = self._build_context(nxt)
                operation_obj = self.calculator._perform_operation(
                    nxt, self.calculator.operation, **ctx)
                new_expr = operation_obj.doit()
                explanation = (f"${latex(operation_obj)}$ 之前已计算过, "
                               "不再显示中间过程")
                self._commit_step(nxt, new_expr, explanation)
                continue
            # Fresh sub-expression: expose it to the user and stop.
            self.current_expr = nxt
            self._visited.add(nxt)
            return

        # Queue empty: nothing left to do — finalize.
        self.current_expr = None
        self.finish()

    def _backup_step_generator(self) -> Dict[str, Any]:
        """Snapshot step_generator state for safe rule preview computation."""
        gen = self.calculator.step_generator
        return {
            'steps': list(gen.steps),
            'explanations': list(gen._explanations),
            'subs_dict': dict(gen.subs_dict),
            'available_sym_chars': list(gen.available_sym_chars),
        }

    def _restore_step_generator(self, backup: Dict[str, Any]) -> None:
        """Restore step_generator from a snapshot."""
        gen = self.calculator.step_generator
        gen.steps = backup['steps']
        gen._explanations = backup['explanations']
        gen.subs_dict = backup['subs_dict']
        gen.available_sym_chars = backup['available_sym_chars']
