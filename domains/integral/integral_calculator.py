from typing import List, Optional, Tuple

from sympy import (Add, Eq, Integer, S, Symbol, Expr, Integral, fraction, latex,
                   limit, log, oo, preorder_traversal, simplify, solveset)

from core import BaseCalculator, SelectRuleCalculator
from core.cached_calculator import DBCachedCalculator
from domains.integral import MATCHER_LIST, RULE_DICT, N_MATCHER_LIST

# Define the constant of integration C for indefinite integrals.
C = Symbol('C')


def create_integral_calculator(base_class):
    """Factory function to create integral calculator classes."""

    class IntegralCalculatorImpl(base_class):
        """Symbolic integral calculator that supports step-by-step evaluation.

        Indefinite integrals end with the constant of integration ``+C``. When a
        definite upper bound is supplied the ``+C`` step is skipped and the
        Newton-Leibniz formula ``F(b) - F(a)`` is appended instead.

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

        def __init__(self, include_special_matchers: bool = True):
            if include_special_matchers:
                super().__init__(Integral, RULE_DICT, MATCHER_LIST)
            else:
                super().__init__(Integral, RULE_DICT, N_MATCHER_LIST)
            # Definite-integral state; populated by compute_list / compute_latex so
            # that reset_process() (run inside the engine) cannot clear it.
            self._var: Symbol = Symbol('x')
            self._lower: Optional[Expr] = None
            self._upper: Optional[Expr] = None

        @property
        def _is_definite(self) -> bool:
            """A definite integral is only requested when an upper bound is given."""
            return self._upper is not None

        @staticmethod
        def _merge_constants_with_c(expr: Expr) -> Expr:
            """Merge all constant terms to C."""

            if expr.is_number:
                return C

            expand_expr = expr.expand()
            if not isinstance(expand_expr, Add):
                return expr + C

            other_terms = [term for term in expand_expr.args if not term.is_number]
            # If there are more than 4 terms, don't merge constant terms to C to avoid complexity.
            if len(other_terms) > 4 and not isinstance(expr, Add):
                return expr + C
            return simplify(Add(*other_terms)) + C

        def final_postprocess(self, final_expr: Expr) -> None:
            """Add ``+C`` for indefinite integrals, or apply the definite evaluation."""

            super().final_postprocess(final_expr)

            if self._is_definite:
                # Use the back-substituted antiderivative (super() added a "回代" step
                # in terms of the original variable when a substitution was used); using
                # the raw final_expr would leave a stray substitution symbol and break
                # the Newton-Leibniz evaluation.
                steps = self.step_generator.steps
                antiderivative = steps[-1] if steps and steps[-1] is not None else final_expr
                self._append_definite_evaluation(antiderivative)
                return

            self._append_constant_of_integration()

        def _append_constant_of_integration(self) -> None:
            """Merge constant terms into ``+C`` across the final steps."""

            # Add C to the last step
            self.step_generator.steps[-1] = self._merge_constants_with_c(self.step_generator.steps[-1])
            try:
                # When expression experiences simplification without assumptions,
                # add C to the penultimate expression.
                if not self.step_generator.steps[-2].has(Integral) and not self.step_generator.steps[-2].has(C):
                    self.step_generator.steps[-2] = self._merge_constants_with_c(self.step_generator.steps[-2])
                # When no rule applies to the remaining Integral, fall back to SymPy's Integral.doit() and
                # the result experiences simplification without assumptions and simplification with assumptions.
                # Add C to the third-to-last expression.
                if not self.step_generator.steps[-3].has(Integral) and not self.step_generator.steps[-3].has(C):
                    self.step_generator.steps[-3] = self._merge_constants_with_c(self.step_generator.steps[-3])
            except (IndexError, AttributeError):
                pass

        def _resolve_antiderivative(self, value: Expr) -> Expr:
            """Resolve any leftover (unevaluated) Integral via SymPy's doit()."""

            if value.has(Integral):
                return value.doit()
            return value

        def _eval_at_bound(self, F: Expr, var: Symbol, bound, direction: str):
            """Value of the antiderivative at a bound, via a one-sided limit when the
            bound is infinite or the direct substitution is non-finite (improper case).

            Returns ``(value, is_improper)``.
            """
            direct = F.subs(var, bound)
            infinite = bound in (oo, -oo) or getattr(bound, 'is_infinite', False)
            bad = (direct.has(Integral, Symbol('Limit'))
                   or getattr(direct, 'is_finite', None) is False
                   or getattr(direct, 'is_nan', None) is True)
            if infinite or bad:
                try:
                    val = limit(F, var, bound, dir='-' if direction == 'upper' else '+')
                except Exception:  # pylint: disable=broad-exception-caught
                    val = direct
                return val, True
            return direct, False

        def _integrand_of(self) -> Optional[Expr]:
            """The original integrand, taken from the first Integral step (for pole scans)."""
            steps = self.step_generator.steps
            first = steps[0] if steps else None
            if isinstance(first, Integral) and first.args:
                return first.args[0]
            return None

        def _one_sided(self, F: Expr, var: Symbol, pt, side: str):
            """One-sided limit of the antiderivative at ``pt`` ('right' = +, 'left' = -)."""
            try:
                return limit(F, var, pt, dir='+' if side == 'right' else '-')
            except Exception:  # pylint: disable=broad-exception-caught
                return F.subs(var, pt)

        def _interior_poles(self, f: Expr, var: Symbol, a, b) -> list:
            """Points strictly inside (a, b) where the integrand blows up (interior poles).

            Candidates come from denominator roots and logarithm-argument zeros; each is
            confirmed by checking a one-sided limit of the integrand is infinite, so
            removable singularities are not mis-flagged."""
            cands = set()
            try:
                den = fraction(simplify(f))[1]
                for r in solveset(Eq(den, 0), var, domain=S.Reals):
                    if getattr(r, 'is_number', False) and getattr(r, 'is_finite', False):
                        cands.add(r)
                for e in preorder_traversal(f):
                    if isinstance(e, log):
                        g = e.args[0]
                        for r in solveset(Eq(g, 0), var, domain=S.Reals):
                            if getattr(r, 'is_number', False) and getattr(r, 'is_finite', False):
                                cands.add(r)
            except Exception:  # pylint: disable=broad-exception-caught
                return []
            lo, hi = (a, b) if (a <= b) else (b, a)
            inside = []
            for c in cands:
                try:
                    if not (lo < c < hi):
                        continue
                    lp = limit(f, var, c, dir='+')
                    lm = limit(f, var, c, dir='-')
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
                blows = (getattr(lp, 'is_infinite', False) or getattr(lm, 'is_infinite', False)
                         or lp in (S.ComplexInfinity,) or lm in (S.ComplexInfinity,))
                if blows:
                    inside.append(c)
            return sorted(set(inside), key=float)

        def _append_split_evaluation(self, F: Expr, var: Symbol, a, b, poles: list) -> None:
            """Evaluate the definite integral by splitting at interior poles; each side must converge."""
            a, b = (a, b) if (a <= b) else (b, a)
            pts = [a] + list(poles) + [b]
            pole_set = set(poles)
            total = Integer(0)
            dv: Optional[Tuple] = None
            for p, q in zip(pts[:-1], pts[1:]):
                vp = self._one_sided(F, var, p, 'right') if p in pole_set else F.subs(var, p)
                vq = self._one_sided(F, var, q, 'left') if q in pole_set else F.subs(var, q)
                if not (getattr(vp, 'is_finite', False) and getattr(vq, 'is_finite', False)):
                    dv = (p, q)
                    break
                total = total + (vq - vp)

            poles_tex = ", ".join(latex(c) for c in poles)
            if dv is not None:
                self.step_generator.add_step(
                    S.NaN,
                    rf"被积函数在区间内部有瑕点 $x = {poles_tex}$, 拆出的子区间 "
                    rf"$[{latex(dv[0])}, {latex(dv[1])}]$ 上的反常积分发散, 不能用牛顿-莱布尼茨公式, "
                    rf"故原反常积分发散")
                return

            result = simplify(total)
            self.step_generator.add_step(
                None, rf"被积函数在区间内部有瑕点 $x = {poles_tex}$, 按瑕点拆分为若干反常积分")
            self.step_generator.add_step(
                result, rf"各子区间反常积分均收敛, 求和得 $\int_{{{latex(a)}}}^{{{latex(b)}}} = {latex(result)}$")

        def _append_definite_evaluation(self, antiderivative: Expr) -> None:
            """Append the Newton-Leibniz steps; use endpoint limits for improper cases."""

            var, upper = self._var, self._upper
            lower = self._lower if self._lower is not None else 0
            lower = 0 if lower is None else lower

            F = self._resolve_antiderivative(antiderivative)

            # Interior poles invalidate Newton-Leibniz across a discontinuity
            # (e.g. PV-like 0 for int(1/x,-1,1) or the spurious -2 for int(1/x^2,-1,1));
            # split the interval and treat each side as an improper integral.
            f = self._integrand_of()
            if f is not None:
                poles = self._interior_poles(f, var, lower, upper)
                if poles:
                    self._append_split_evaluation(F, var, lower, upper, poles)
                    return

            fu, imp_u = self._eval_at_bound(F, var, upper, 'upper')
            fl, imp_l = self._eval_at_bound(F, var, lower, 'lower')

            lb, ub = latex(lower), latex(upper)
            if imp_u or imp_l:
                self.step_generator.add_step(
                    None,
                    r"端点为无穷或瑕点, 按反常积分取单侧极限: $\lim_{} F\bigr|_{"
                    + lb + r"}^{" + ub + r"}$"
                )

            diff_expr = fu - fl
            result = simplify(diff_expr)

            finite = getattr(result, 'is_finite', None) is True
            concl = "" if finite else r", 该反常积分发散(或不存在)"
            self.step_generator.add_step(
                result,
                rf"应用牛顿-莱布尼茨公式: $\left. {latex(F)} \right|_{{{lb}}}^{{{ub}}} "
                rf"= {latex(diff_expr)} = {latex(result)}$" + concl
            )

        def compute_list(self, expr: str, var: Symbol,
                         lower: Optional[Expr] = None,
                         upper: Optional[Expr] = None) -> Tuple[List[Expr], List[str]]:
            """Define Integral context, capturing optional definite bounds."""
            self._var, self._lower, self._upper = var, lower, upper
            return super().compute_list(expr, variable=var)

        def compute_latex(self, expr: str, var: Symbol,
                          lower: Optional[Expr] = None,
                          upper: Optional[Expr] = None) -> str:
            """Define Integral context, capturing optional definite bounds."""
            self._var, self._lower, self._upper = var, lower, upper
            return super().compute_latex(expr, variable=var)

    return IntegralCalculatorImpl


IntegralCalculator = create_integral_calculator(BaseCalculator)
SelectIntegralCalculator = create_integral_calculator(SelectRuleCalculator)


class CachedIntegralCalculator(DBCachedCalculator, IntegralCalculator):
    """Integral calculator with database cache support."""
