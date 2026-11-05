"""Shared helpers for the applied-calculus operation package.

These deliberately REUSE the core rule-driven engines (``DiffCalculator`` /
``IntegralCalculator``) so applied operations show genuine step-by-step
derivations, matching the main calculus engines rather than one-line formulas.
"""

from sympy import Symbol

from config import HARD_MAX_DIFF_ORDER, HARD_MAX_ITERATIONS
from core.base_step_generator import format_note

from utils.cache_utils import cached_sympify as sympify


def get_var(data):
    return Symbol(data.get('variable', 'x') or 'x')


def to_num(value, default=None):
    if value is None or str(value).strip() == '':
        return default
    return sympify(str(value))


def to_count(value, default: int, low: int = 1, high: int = HARD_MAX_ITERATIONS) -> int:
    """Clamp a user-selected iteration/order count to a safe range.

    Applied numerical methods loop the requested number of times; an unbounded
    ``order=10**9`` would monopolize a worker long past the task timeout.
    """
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        count = default
    return max(low, min(high, count))


def expr_text(data):
    return data.get('expression', '')


def steps_value(steps):
    """The resolved value = the last concrete expression of a ``[(expr, note), ...]``
    step list. Applied handlers use it to record a composable result (via the same
    rule-engine rows they display) so the universal box can substitute it back."""
    concrete = [e for (e, _n) in steps if e is not None]
    return concrete[-1] if concrete else None


def render_note(text) -> str:
    """Prose -> ``\\textrm{}``, ``$...$`` -> live LaTeX (MathJax-safe, no literal $)."""
    return format_note(text, prose_command='textrm')


def align(items) -> str:
    """Build a ``\\begin{align}`` block from ``[(body_or_None, note), ...]``."""
    lines = []
    first = True
    for body, note in items:
        # Rules inject display-only sub-steps using the string 'None' as a sentinel for
        # "no left-hand side" (mirrors core.base_step_generator.to_latex).
        if body is not None and body != 'None':
            body_latex = body if isinstance(body, str) else _latex(body)
            prefix = '' if first else '&='
            first = False
            line = prefix + body_latex
        else:
            line = '' if first else '&\\text{}'
            first = False
        if note:
            line += r" \quad " + note
        lines.append(line)
    return r"\begin{align}" + r"\\".join(lines) + r"\end{align}"


def _latex(obj):
    from sympy import latex  # pylint: disable=import-outside-toplevel
    return latex(obj)


def derivative_steps(f_str, x):
    """Full rule-driven derivative derivation rows via the core DiffCalculator.

    DiffCalculator already honors ``x`` and treats every other free symbol as a
    constant, so the same call yields genuine partial-derivative steps for
    multivariable operations.
    """
    from domains import DiffCalculator  # pylint: disable=import-outside-toplevel
    steps, explanations = DiffCalculator().compute_list(f_str, x)
    return [(e, render_note(r)) for e, r in zip(steps, explanations) if r != ''] \
        or [(steps[-1], '')]


def derivative_value(f_str, x, order=1):
    """The n-th derivative VALUE, obtained by running the rule engine ``order`` times.

    Each application is a genuine DiffCalculator derivation; the previous result is
    fed back in, so higher derivatives never fall back to a black-box ``.diff(n)``.
    """
    from domains import DiffCalculator  # pylint: disable=import-outside-toplevel
    calc = DiffCalculator()
    current = sympify(str(f_str))
    for _ in range(max(1, min(int(order), HARD_MAX_DIFF_ORDER))):
        steps, _expl = calc.compute_list(str(current), x)
        concrete = [s for s in steps if s is not None]
        current = concrete[-1] if concrete else current
    return current


def integral_steps(integrand, x, lower=None, upper=None, allow_multi=False):
    """Full rule-driven integration rows via the core IntegralCalculator.

    ``allow_multi=True`` integrates with respect to ``x`` while treating every other
    free symbol as a constant (a partial integral), instead of the engine's default
    one-symbol rejection. Used for the inner integral of iterated integrals.
    """
    from domains import IntegralCalculator  # pylint: disable=import-outside-toplevel
    calc = IntegralCalculator()
    if allow_multi:
        calc._allow_multi_symbols = True
    kwargs = {}
    if upper is not None:
        kwargs['upper'] = upper
        kwargs['lower'] = 0 if lower is None else lower
    steps, explanations = calc.compute_list(str(integrand), x, **kwargs)
    return [(e, render_note(r)) for e, r in zip(steps, explanations) if r != ''] \
        or [(steps[-1], '')]


def integral_value(integrand, x, allow_multi=False):
    """The antiderivative VALUE produced by the rule engine (constant of integration dropped).

    ``allow_multi=True`` integrates with respect to ``x`` treating other symbols as
    constants (a partial integral), used by exact-ODE potential functions.
    """
    from domains import IntegralCalculator  # pylint: disable=import-outside-toplevel
    expr = sympify(str(integrand))
    if expr.is_number:
        # The engine defaults a constant integrand to wrt x; honor the requested var.
        return expr * x
    calc = IntegralCalculator()
    if allow_multi:
        calc._allow_multi_symbols = True
    steps, _expl = calc.compute_list(str(integrand), x)
    concrete = [s for s in steps if s is not None]
    value = concrete[-1] if concrete else expr
    return value.subs(Symbol('C'), 0) if value.has(Symbol('C')) else value
