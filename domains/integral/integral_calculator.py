from typing import List, Optional, Tuple

import itertools

from sympy import (Add, Eq, Integer, Interval, S, Symbol, Expr, Integral, fraction,
                   latex, limit, log, oo, preorder_traversal, simplify, solveset, sympify,
                   trigsimp)

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

        # Auto (web/cache) path may apply the mirror/parity definite shortcuts;
        # the manual /rule_select path must not (a human picks every step there).
        _definite_symmetry_enabled = base_class is not SelectRuleCalculator

        def __init__(self, include_special_matchers: bool = True):
            if include_special_matchers:
                super().__init__(Integral, RULE_DICT, MATCHER_LIST)
            else:
                super().__init__(Integral, RULE_DICT, N_MATCHER_LIST)
            # Remembered so the DB cache can key on this choice: the two matcher
            # sets produce visibly different derivations.
            self._include_special_matchers = bool(include_special_matchers)
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

            steps = self.step_generator.steps
            # super() recorded the back-substituted antiderivative (in the original
            # variable) as the last step; final_expr itself may still carry a
            # substitution dummy, so validate the recorded step.
            substituted = steps[-1] if steps and steps[-1] is not None else final_expr
            if self._parametric_result_is_suspect(substituted):
                # A parametric closed-form antiderivative that fails numeric validation
                # is reported as UNEVAL (None) rather than shipping a wrong expression.
                self.step_generator.steps[-1] = None
                return

            self._append_constant_of_integration()

        def _original_integrand(self) -> Optional[Expr]:
            """Return the integrand of the first (input) Integral step, or None."""
            for step in self.step_generator.steps:
                if isinstance(step, Integral):
                    return step.args[0]
            return None

        def _parametric_result_is_suspect(self, antiderivative: Expr) -> bool:
            """True when a parametric closed-form antiderivative fails numeric checks.

            Scope is deliberately narrow: it returns False unless the caller opted into
            parameters (``_allow_parameters``) *and* the integrand carries a symbol other
            than the integration variable. Single-variable results — the whole existing
            regression surface — are therefore never touched. Unevaluated (residual
            ``Integral``) results are also not touched, leaving the honest UNEVAL flow."""
            if not getattr(self, '_allow_parameters', False):
                return False
            if antiderivative is None or antiderivative.has(Integral):
                return False
            integrand = self._original_integrand()
            if integrand is None:
                return False
            var = self._var if self._var in integrand.free_symbols else \
                (list(integrand.free_symbols)[0] if integrand.free_symbols else self._var)
            params = [s for s in integrand.free_symbols if s != var]
            if not params:
                return False  # single-variable: never suspect (out of guard scope)

            deriv = antiderivative.diff(var)
            checks = 0
            # A spread of (parameter, variable) samples. A result is trusted only when
            # every *clean* (real, finite) sample matches; a single clean mismatch means
            # it is definitively wrong. If no sample evaluates cleanly we cannot confirm
            # it, so flag as suspect rather than risk shipping an unverified answer.
            for pval in (1.7, 2.3, 0.9):
                assign = {p: pval for p in params}
                for xv in (0.31, 0.73, 1.19):
                    sample = dict(assign)
                    sample[var] = xv
                    try:
                        d = complex(deriv.subs(sample).evalf())
                        f = complex(integrand.subs(sample).evalf())
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
                    if d.imag or f.imag or abs(d) > 1e12 or abs(f) > 1e12:
                        continue  # non-real / explosive: skip this sample
                    checks += 1
                    if abs(d - f) > 1e-4 * (1.0 + abs(f)):
                        return True  # a clean mismatch: definitely wrong
            if checks == 0:
                return True  # could not confirm on any sample: treat as suspect
            return False

        def _midpoint_integral(self, f_assigned, var, lo, hi, n=1200):
            """Composite midpoint rule for the (parameter-substituted) integrand.

            Midpoints avoid evaluating at the interval endpoints, which is helpful
            for integrands with a vertical tangent or removable singularity there
            (e.g. sqrt(a^2 - x^2) at x = a). Returns None if any sample is non-real."""
            h = (hi - lo) / n
            total = 0.0
            for i in range(n):
                xv = lo + h * (i + 0.5)
                try:
                    val = complex(f_assigned.subs(var, xv).evalf())
                except (TypeError, ValueError, ZeroDivisionError):
                    return None
                if val.imag and abs(val.imag) > 1e-6:
                    return None
                total += val.real
            return total * h

        def _definite_parametric_is_suspect(self, result, var, lower, upper) -> bool:
            """True when a parametric definite integral's closed value fails numeric check.

            Confines itself to the parameter path (``_allow_parameters``) and to results
            that actually carry a free symbol, so the large, numeric-only definite test
            surface is untouched. The closed form is compared to a midpoint-rule numeric
            integral after substituting concrete positive values for the parameters; a
            non-finite value (e.g. a stray ``zoo``) or a clear mismatch becomes an honest
            UNEVAL. When no valid numeric reference can be formed the result is kept."""
            if not getattr(self, '_allow_parameters', False):
                return False
            if result is None:
                return False
            lower = sympify(lower)
            upper = sympify(upper)
            params = set(result.free_symbols) | set(lower.free_symbols) | set(upper.free_symbols)
            params.discard(var)
            if not params:
                return False  # numeric-valued definite integral: trusted
            f = self._integrand_of()
            if f is None:
                return False
            for pval in (1.7, 2.3):
                assign = {p: pval for p in params}
                try:
                    rv = result.subs(assign)
                    # A closed form containing zoo/oo/nan evaluates to a non-finite
                    # number; plain float comparisons against nan/inf are always False,
                    # so test is_finite explicitly before any numeric comparison.
                    if getattr(rv, 'is_finite', None) is not True:
                        return True
                    rval = complex(rv.evalf())
                except (TypeError, ValueError, ZeroDivisionError):
                    return True  # closed form not evaluable at a sample: suspect
                lo = complex(lower.subs(assign).evalf())
                hi = complex(upper.subs(assign).evalf())
                if lo.imag or hi.imag or abs(lo) > 1e12 or abs(hi) > 1e12 or lo.real >= hi.real:
                    continue  # interval invalid at this sample: try next
                if rval.imag and abs(rval.imag) > 1e-6:
                    return True  # complex value for real data: suspect
                try:
                    ref = self._midpoint_integral(f.subs(assign), var, lo.real, hi.real)
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if ref is None:
                    continue
                # ``<=`` (not ``>``) so a NaN difference evaluates as "not agreeing"
                # and is flagged, since NaN comparisons are always False.
                agrees = abs(rval.real - ref) <= 1e-2 * (1.0 + abs(ref))
                if not agrees:
                    return True  # clear numeric disagreement (or NaN diff): wrong
            return False  # matched or no reference available: keep the result


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
            # Treat "not a definite finite number" as improper: S.NaN exposes neither
            # is_nan nor is_finite==False, so a positive is_finite test is what catches
            # the finite-endpoint blow-up (e.g. log(x) at 0 -> NaN) and triggers a limit.
            bad = (direct.has(Integral, Symbol('Limit'))
                   or getattr(direct, 'is_finite', None) is not True)
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
            removable singularities are not mis-flagged. A periodic denominator makes
            solveset return an infinite ImageSet over Reals (iterating it never ends), so
            for concrete finite bounds the search is confined to Interval.open(a, b);
            otherwise iteration is capped. Symbolic bounds cannot be compared, so they
            are skipped rather than crashing on ``a <= b``."""
            if not (getattr(a, 'is_number', False) and getattr(b, 'is_number', False)):
                return []
            try:
                lo, hi = (a, b) if (a <= b) else (b, a)
            except Exception:  # pylint: disable=broad-exception-caught
                return []
            finite = getattr(lo, 'is_finite', True) and getattr(hi, 'is_finite', True)
            dom = Interval.open(lo, hi) if finite else S.Reals

            cands = set()

            def _collect(solutions):
                try:
                    for r in itertools.islice(solutions, 0, 128):
                        if getattr(r, 'is_number', False) and getattr(r, 'is_finite', False):
                            cands.add(r)
                except TypeError:
                    pass  # a non-iterable solveset result (e.g. an unevaluated set)

            try:
                den = fraction(simplify(f))[1]
                _collect(solveset(Eq(den, 0), var, domain=dom))
                for e in preorder_traversal(f):
                    if isinstance(e, log):
                        _collect(solveset(Eq(e.args[0], 0), var, domain=dom))
            except Exception:  # pylint: disable=broad-exception-caught
                return []

            inside = []
            for c in cands:
                try:
                    if not lo < c < hi:
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
                    r"端点为无穷或瑕点, 按反常积分取单侧极限: $\left. F \right|_{"
                    + lb + r"}^{" + ub + r"}$"
                )

            diff_expr = fu - fl
            result = simplify(diff_expr)

            if self._definite_parametric_is_suspect(result, var, lower, upper):
                # A parametric definite value that fails numeric validation is reported
                # as UNEVAL (None) instead of shipping a wrong expression (e.g. one
                # containing a stray ``zoo`` from an unevaluable symbolic bound).
                self.step_generator.add_step(
                    None,
                    r"参数定积分闭式经数值校验未通过, 按未能求得处理 (UNEVAL), 不输出可能错误的结果。")
                return

            finite = getattr(result, 'is_finite', None) is True
            concl = "" if finite else r", 该反常积分发散(或不存在)"
            self.step_generator.add_step(
                result,
                rf"应用牛顿-莱布尼茨公式: $\left. {latex(F)} \right|_{{{lb}}}^{{{ub}}} "
                rf"= {latex(diff_expr)} = {latex(result)}$" + concl
            )

        def _try_definite_symmetry(self, expr, var, lower, upper):
            """Mirror/king (constant-sum) and odd-parity definite shortcuts (no recursion).

            Only fires for finite numeric bounds with no interior poles, so it can never
            mask a divergent integral. Returns (steps, explanations) recorded into the step
            generator, or None to fall through to the normal Newton-Leibniz path."""
            a, b = lower, upper
            a, b = sympify(a), sympify(b)
            if not (getattr(a, 'is_number', False) and getattr(b, 'is_number', False)
                    and getattr(a, 'is_finite', True) and getattr(b, 'is_finite', True)):
                return None
            if a == b:
                return None
            try:
                f = self._sympify(expr)
            except Exception:  # pylint: disable=broad-exception-caught
                return None
            if self._interior_poles(f, var, a, b):
                return None
            ab = a + b
            fsub = f.subs(var, ab - var)
            g = simplify(trigsimp(f + fsub))
            is_const = g.is_number and getattr(g, 'is_finite', True)
            is_odd = (simplify(ab) == 0
                      and simplify(trigsimp(f + f.subs(var, -var))) == 0)
            if not (is_const or is_odd):
                # King property with an x-factor: f = var*h with h symmetric about
                # (a+b)/2, e.g. int_0^pi x*f(sin x)dx = (pi/2)*int_0^pi f(sin x)dx.
                # The reduced integral is solved recursively; the indefinite
                # antiderivative of such f is often non-elementary even though the
                # definite value is, so this shortcut is essential (not just faster).
                h = simplify(f / var)
                if (h != f and h.free_symbols and h.free_symbols <= {var}
                        and simplify(trigsimp(h - h.subs(var, ab - var))) == 0):
                    fired = self._king_xfactor(f, h, var, a, b, ab)
                    if fired is not None:
                        return fired
                return None

            v = latex(var)
            la, lb = latex(a), latex(b)
            ih = "\\int_{" + la + "}^{" + lb + "}"
            self.step_generator.reset()
            if is_const:
                span = b - a
                half = simplify(g * span / 2)
                self.step_generator.add_step(
                    Integral(f, (var, a, b)),
                    "记 $I=" + ih + latex(f) + "\\,d" + v + "$")
                self.step_generator.add_step(
                    Integral(fsub, (var, a, b)),
                    "区间再现(镜像): 令 $t=" + latex(ab) + "-" + v + "$, 则 $I=" + ih
                    + latex(fsub) + "\\,d" + v + "$")
                self.step_generator.add_step(
                    Integral(g, (var, a, b)),
                    "两式相加: $2I=" + ih + "[f(" + v + ")+f(" + latex(ab) + "-" + v
                    + ")]\\,d" + v + "=\\int_{" + la + "}^{" + lb + "} " + latex(g) + "\\,d" + v + "$")
                self.step_generator.add_step(
                    g * span,
                    "和式为常数 $" + latex(g) + "$: $\\int_{" + la + "}^{" + lb + "} "
                    + latex(g) + "\\,d" + v + "=" + latex(g * span) + "$")
                self.step_generator.add_step(
                    half,
                    "故 $I=\\dfrac{" + latex(g * span) + "}{2}=" + latex(half) + "$")
            else:
                self.step_generator.add_step(
                    Integral(f, (var, a, b)),
                    "记 $I=" + ih + latex(f) + "\\,d" + v + "$")
                self.step_generator.add_step(
                    Integer(0),
                    "积分区间 $[" + la + "," + lb + "]$ 关于原点对称, 且 $f(-" + v + ")=-f("
                    + v + ")$ 为奇函数, 故 $I=0$")
            return self.step_generator.get_steps()

        def _king_xfactor(self, f, h, var, a, b, ab):
            """King property with an x-factor: if f = var*h and h is symmetric about
            (a+b)/2, then I = ∫_a^b f dx = (a+b)/2 · ∫_a^b h dx. Solves the reduced
            integral recursively (its indefinite form is often non-elementary)."""
            sub = IntegralCalculator()
            sub._definite_symmetry_enabled = False
            try:
                jsteps, _ = sub.compute_list(h, var, lower=a, upper=b)
            except Exception:  # pylint: disable=broad-exception-caught
                return None
            jval = jsteps[-1] if jsteps else None
            if jval is None or jval.has(Integral):
                return None
            result = simplify(ab / 2 * jval)
            v = latex(var)
            la, lb = latex(a), latex(b)
            ih = "\\int_{" + la + "}^{" + lb + "}"
            self.step_generator.reset()
            self.step_generator.add_step(
                Integral(f, (var, a, b)), "记 $I=" + ih + latex(f) + "\\,d" + v + "$")
            self.step_generator.add_step(
                Integral((ab - var) * h, (var, a, b)),
                "区间再现: 令 $t=" + latex(ab) + "-" + v + "$, 因 $h(" + latex(ab) + "-" + v
                + ")=h(" + v + ")$, 得 $I=" + ih + latex((ab - var) * h) + "\\,d" + v + "$")
            self.step_generator.add_step(
                ab * Integral(h, (var, a, b)) / 2,
                "两式相加: $2I=" + ih + "\\left(" + v + "+(" + latex(ab) + "-" + v
                + ")\\right)h\\,d" + v + "=" + latex(ab) + ih + latex(h) + "\\,d" + v + "$")
            self.step_generator.add_step(
                result,
                "于是 $I=\\dfrac{" + latex(ab) + "}{2}" + ih + latex(h) + "\\,d" + v
                + "=\\dfrac{" + latex(ab) + "}{2}\\cdot\\left(" + latex(jval)
                + "\\right)=" + latex(result) + "$")
            return self.step_generator.get_steps()

        def compute_list(self, expr: str, var: Symbol,
                         lower: Optional[Expr] = None,
                         upper: Optional[Expr] = None) -> Tuple[List[Expr], List[str]]:
            """Define Integral context, capturing optional definite bounds."""
            self._var, self._lower, self._upper = var, lower, upper
            if self._definite_symmetry_enabled and lower is not None and upper is not None:
                fired = self._try_definite_symmetry(expr, var, lower, upper)
                if fired is not None:
                    return fired
            return super().compute_list(expr, variable=var)

        def compute_latex(self, expr: str, var: Symbol,
                          lower: Optional[Expr] = None,
                          upper: Optional[Expr] = None) -> str:
            """Define Integral context, capturing optional definite bounds."""
            self._var, self._lower, self._upper = var, lower, upper
            if self._definite_symmetry_enabled and lower is not None and upper is not None:
                fired = self._try_definite_symmetry(expr, var, lower, upper)
                if fired is not None:
                    return self.step_generator.to_latex()
            return super().compute_latex(expr, variable=var)

    return IntegralCalculatorImpl


IntegralCalculator = create_integral_calculator(BaseCalculator)
SelectIntegralCalculator = create_integral_calculator(SelectRuleCalculator)


class CachedIntegralCalculator(DBCachedCalculator, IntegralCalculator):
    """Integral calculator with database cache support."""
