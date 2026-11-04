from typing import List, Tuple, Union

from sympy import Basic, Expr, Matrix, latex

import config
from utils.cache_utils import cached_sympify
from utils.matrix_latex import latex_to_nested


def _wrap_latex(expr: Expr) -> str:
    """Add parentheses in LaTeX when needed for proper mathematical notation.

    Parentheses are added for non-atomic expressions, except for certain operations
    where they're typically omitted for clarity (e.g., exponents, indices).
    """
    expr_latex = latex(expr)

    # Atomic expressions don't need parentheses
    if expr.is_Atom:
        return expr_latex

    # Special handling for common mathematical operations where parentheses
    # are typically not wrapped.
    if expr.is_Pow:  # Exponentiation
        return expr_latex
    if expr.is_Indexed:  # Indexed objects
        return expr_latex
    if expr.is_Function:  # Function applications
        return expr_latex

    # For other non-atomic expressions, add parentheses for clarity
    return f"\\left({expr_latex}\\right)"


def wrap_latex(*expr: List[Basic | Expr]) -> Union[str, Tuple]:
    return _wrap_latex(expr[0]) if len(expr) == 1 else tuple(_wrap_latex(e) for e in expr)


def _fragment_latex(fragment: str) -> str:
    """Best-effort LaTeX for one operator argument (matrix literals included)."""
    fragment = fragment.strip()
    if not fragment:
        return ''
    try:
        nested = latex_to_nested(fragment)
        obj = cached_sympify(nested)
        if isinstance(obj, (list, tuple, dict)):
            # Matrix literal argument, e.g. det([[1,2],[3,4]]).
            return latex(Matrix(obj))
        return latex(obj)
    except Exception:  # pylint: disable=broad-exception-caught
        # Keep the original text (escaped by the client) rather than failing the
        # whole preview; MathJax renders most leftovers readably.
        return fragment


def _operator_preview(expr: str) -> str:
    """Render ``name(args)`` operator input as ``\\operatorname{name}(args)``.

    Used only for the live preview: the real computation path scores each operator
    through its own engine. Parsing operator input as a plain expression made
    SymPy's implicit-multiplication fallback split ``diff(...)`` into ``d*i*f*f``.
    """
    from core.latex_syntax import (  # pylint: disable=import-outside-toplevel
        _find_innermost_op, _fold_matrix_ops, _normalize_latex_ops, _split_top_args,
    )

    text = _normalize_latex_ops(_fold_matrix_ops(latex_to_nested(expr)))
    guard = 0
    while True:
        guard += 1
        if guard > 64:
            break
        loc = _find_innermost_op(text)
        if loc is None:
            break
        args = [_fragment_latex(a) for a in _split_top_args(loc['inner'])]
        replacement = (
            '\\operatorname{' + loc['name'] + '}\\left(' + ','.join(args) + '\\right)'
        )
        text = text[:loc['start']] + replacement + text[loc['end']:]
    return text.replace('*', '\\cdot ')


def str_to_latex(expr: str, operation_type: str = None) -> str:
    """Convert a user-provided string expression to LaTeX for frontend preview.

    This function safely parses and renders expressions based on context:
    - For matrix-related operations, it is rendered as a matrix.
    - For multi-line inputs, each line is rendered separately.
    - For linear systems, equations are formatted in a cases' environment.
    - Operator input (``diff(...)``, ``\\int``, ...) is rendered structurally
      instead of being fed to SymPy as a plain expression.
    - Otherwise, the expression is rendered as-is via SymPy's latex().
    """
    if operation_type in config.RENDER_SINGLE_MATRIX:
        return latex(Matrix(cached_sympify(latex_to_nested(expr))))

    if operation_type in config.RENDER_MANY_ROWS_ONLY_NUMBER_MATRIX:
        parts = []
        for line in latex_to_nested(expr).strip().split('\n'):
            sym = cached_sympify(line)
            parts.append(
                latex(sym) if hasattr(sym, 'is_number') and sym.is_number else latex(Matrix(sym)))
        return r"\quad\quad".join(parts)

    if operation_type == 'linear-system':
        from domains.matrix.linear_system_converter import (  # pylint: disable=import-outside-toplevel
            LinearSystemConverter,
        )
        equations = LinearSystemConverter.str_to_Eq(expr)
        eq_latex = [latex(eq) for eq in equations]
        return r"\begin{cases} " + r" \\ ".join(eq_latex) + r" \end{cases}"

    from core.input_validation import (  # pylint: disable=import-outside-toplevel
        looks_like_operator_input,
    )
    if looks_like_operator_input(expr):
        preview = _operator_preview(expr)
        if preview:
            return preview

    parsed = cached_sympify(latex_to_nested(expr))
    if isinstance(parsed, (list, tuple)):
        try:
            return latex(Matrix(parsed))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return latex(parsed)
