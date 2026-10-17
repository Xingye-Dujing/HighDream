from collections import deque
from typing import Callable, List, Optional, Set, Tuple

from sympy import (
    Basic, Expr, Mul, Symbol, preorder_traversal, srepr, expand, factor,
    expand_trig, logcombine, expand_log, apart, cancel, radsimp, tan, sin, cos,
    powsimp, trigsimp, collect, ratsimp
)

from utils.cache_utils import cached_sympify


class ExpressionParser:  # pylint: disable=too-few-public-methods
    """A class for parsing and transforming mathematical expressions using various symbolic computation rules.

    This parser applies transformation rules to generate equivalent forms of an expression up to a specified depth.
    """

    def __init__(self, max_depth: int = 3, sort_strategy: Optional[Callable[[Expr], float]] = None) -> None:
        """Initialize the ExpressionParser with maximum derivation depth and optional sorting strategy.

        Args:
            max_depth: Maximum depth of the derivation tree (default: 3)
            sort_strategy: Optional function to sort derived expressions by a custom criterion
        """
        self.max_depth = max_depth
        self.sort_strategy = sort_strategy
        self.transform_rules: List[Tuple[str, Callable[[Expr], List[Expr]]]] = []
        self._initialize_rules()

    @staticmethod
    def _try_transform(expr: Expr, func: Callable, *args, **kwargs) -> List[Expr]:
        """Safely apply a transformation function to an expression.

        Args:
            expr: The expression to transform
            func: The transformation function to apply
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function.

        Returns:
            List containing the transformed expression if successful, otherwise empty list.
        """
        try:
            result = func(expr, *args, **kwargs)
            if result is not None and result != expr:
                return [result]
        except Exception:
            return []
        return []

    def _initialize_rules(self) -> None:
        """Initialize the list of transformation rules with their names and corresponding functions."""

        self.transform_rules.extend([
            ("三角展开", self._expand_trig_transform),
            ("积化和差", self._product_to_sum_transform),
            ("多项式展开", self._polynomial_expand_transform),
            ("因式分解", self._factor_transform),
            ("tan 拆开", self._tan_to_sin_cos_transform),
            ("对数合并", self._combine_logarithms_transform),
            ("对数展开", self._expand_logarithmic_transform),
            ("部分分式", self._apart_transform),
            ("分式约分", self._cancel_transform),
            ("有理化", self._radsimp_transform),
            ("幂化简", self._powsimp_transform),
            ("三角化简", self._reciprocal_trig_transform),
            ("同类项合并", self._collect_transform),
            ("根式去嵌套", self._sqrtdenest_transform),
            ("有理式通分", self._ratsimp_transform),
        ])

    def parse(self, expression: str) -> List[Tuple[Expr, str]]:
        """Parse an expression and generate all possible derivations within the maximum depth.

        Args:
            expression: String representation of the mathematical expression

        Returns:
            List of tuples containing derived expressions, and their derivation steps.
        """
        expr = cached_sympify(expression)

        def canonical_key(e: Basic | Expr) -> str:
            return srepr(e)

        seen: Set[str] = {canonical_key(expr)}
        derivations: List[Tuple[Expr, str]] = []
        queue = deque([(expr, 0, "原始表达式")])

        while queue:
            current_expr, depth, reason = queue.popleft()
            if depth >= self.max_depth:
                continue

            for rule_name, rule in self.transform_rules:
                for new_expr in self._apply_rule_to_subexpressions(rule, current_expr):
                    key = canonical_key(new_expr)
                    if key not in seen:
                        seen.add(key)
                        step_reason = f"{reason} -> {rule_name}"
                        derivations.append((new_expr, step_reason))
                        queue.append((new_expr, depth + 1, step_reason))

        if self.sort_strategy:
            derivations.sort(key=lambda x: self.sort_strategy(x[0]))

        return derivations

    @staticmethod
    def _apply_rule_to_subexpressions(rule: Callable[[Expr], List[Expr]], expr: Expr) -> List[Expr]:
        """Apply a transformation rule to all subexpressions of an expression.

        Args:
            rule: The transformation rule function to apply
            expr: The expression to which the rule will be applied.

        Returns:
            List of all valid transformed expressions
        """
        results = []
        added_keys = set()

        def add_candidate(e: Expr):
            key = srepr(e)
            if key not in added_keys and e != expr:
                added_keys.add(key)
                results.append(e)

        try:
            for t in rule(expr):
                add_candidate(t)
        except Exception:
            pass

        try:
            for sub in preorder_traversal(expr):
                if sub == expr:
                    continue
                try:
                    for t in rule(sub):  # type:ignore
                        try:
                            new_expr = expr.xreplace({sub: t})
                        except Exception:
                            new_expr = expr.subs(sub, t)
                        add_candidate(new_expr)
                except Exception:
                    pass
        except Exception:
            pass

        return results

    # Transformation rules
    def _polynomial_expand_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, expand)

    def _factor_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, factor)

    def _product_to_sum_transform(self, expr: Expr) -> List[Expr]:
        """Rewrite a product of two trig factors (sin/cos) into an equivalent sum.

        Applies the product-to-sum identities to the first two sin/cos factors and
        keeps any remaining factors (numeric or symbolic) as a multiplier, so it also
        covers products with more than two trig factors or a non-numeric coefficient.
        The result is accepted only when it differs structurally yet is mathematically
        equal to the input.
        """
        try:
            if not expr.is_Mul:
                return []
            trig = [f for f in expr.args if isinstance(f, (sin, cos))]
            if len(trig) < 2:
                return []
            f1, f2 = trig[0], trig[1]
            a, b = f1.args[0], f2.args[0]
            if isinstance(f1, sin) and isinstance(f2, cos):
                base = (sin(a + b) + sin(a - b)) / 2
            elif isinstance(f1, cos) and isinstance(f2, sin):
                base = (sin(a + b) + sin(b - a)) / 2
            elif isinstance(f1, cos) and isinstance(f2, cos):
                base = (cos(a + b) + cos(a - b)) / 2
            else:  # sin * sin
                base = (cos(a - b) - cos(a + b)) / 2
            rest = [f for f in expr.args if f is not f1 and f is not f2]
            result = Mul(*rest) * base if rest else base
            if result != expr and result.equals(expr):
                return [result]
        except Exception:
            return []
        return []

    def _expand_trig_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, expand_trig)

    def _combine_logarithms_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, logcombine, force=True)

    def _expand_logarithmic_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, expand_log, force=True)

    def _apart_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, apart)

    def _cancel_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, cancel)

    def _radsimp_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, radsimp)

    def _powsimp_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, powsimp, force=True)

    def _reciprocal_trig_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, trigsimp, reciprocal=True)

    def _collect_transform(self, expr: Expr) -> List[Expr]:
        syms = sorted(expr.atoms(Symbol), key=str)
        if not syms:
            return []
        return self._try_transform(expr, collect, syms)

    def _sqrtdenest_transform(self, expr: Expr) -> List[Expr]:
        from sympy import sqrtdenest  # pylint: disable=import-outside-toplevel
        return self._try_transform(expr, sqrtdenest)

    def _ratsimp_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, ratsimp)

    @staticmethod
    def _tan_to_sin_cos_transform(expr: Expr) -> List[Expr]:
        """Transform tangent functions into sine over cosine ratios.

        Args:
            expr: The expression potentially containing tangent functions

        Returns:
            List containing the transformed expression or empty list if no transformation occurred.
        """
        try:
            replaced = expr.replace(lambda e: getattr(e, "func", None) == tan,
                                    lambda e: sin(e.args[0]) / cos(e.args[0]))
            if replaced != expr:
                return [replaced]
        except Exception:
            pass
        return []
