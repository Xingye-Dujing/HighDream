"""Shared helpers for the applied-calculus operation package.

These deliberately REUSE the core rule-driven engines (``DiffCalculator`` /
``IntegralCalculator``) so applied operations show genuine step-by-step
derivations, matching the main calculus engines rather than one-line formulas.
"""

from sympy import Symbol

from utils.cache_utils import cached_sympify as sympify


def get_var(data):
    return Symbol(data.get('variable', 'x') or 'x')


def to_num(value, default=None):
    if value is None or str(value).strip() == '':
        return default
    return sympify(str(value))


def expr_text(data):
    return data.get('expression', '')


def render_note(text) -> str:
    """Prose -> ``\\textrm{}``, ``$...$`` -> live LaTeX (MathJax-safe, no literal $)."""
    parts = str(text).split('$')
    out = []
    for i, seg in enumerate(parts):
        if seg == '':
            continue
        out.append(seg if i % 2 == 1 else r"\textrm{" + seg + r"}")
    return ''.join(out)


def align(items) -> str:
    """Build a ``\\begin{align}`` block from ``[(body_or_None, note), ...]``."""
    lines = []
    first = True
    for body, note in items:
        if body is not None:
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
