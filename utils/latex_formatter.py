from typing import Tuple, Union
from sympy import Expr, Matrix, latex, radsimp, sympify
from domains.matrix.linear_system_converter import LinearSystemConverter
import config


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
    # are typically not wrapped
    if expr.is_Pow:  # Exponentiation
        return expr_latex
    if expr.is_Indexed:  # Indexed objects
        return expr_latex
    if expr.is_Function:  # Function applications
        return expr_latex

    # For other non-atomic expressions, add parentheses for clarity
    return f"\\left({expr_latex}\\right)"


def wrap_latex(*expr: Expr) -> Union[str, Tuple]:
    return _wrap_latex(expr[0]) if len(expr) == 1 else tuple(_wrap_latex(e) for e in expr)


def str_to_latex(expr: str, operation_type: str = None) -> str:
    """Convert a user-provided string expression to LaTeX for frontend preview.

    This function safely parses and renders expressions based on context:
    - For matrix-related operations, it is rendered as a matrix.
    - For multi-line inputs, each line is rendered separately.
    - For linear systems, equations are formatted in a cases environment.
    - Otherwise, the expression is rendered as-is via SymPy's latex().
    """
    if operation_type in config.RENDER_SINGLE_MATRIX:
        return latex(Matrix(sympify(expr)))

    if operation_type in config.RENDER_MANY_ROWS_ONLY_NUMBER_MATRIX:
        parts = []
        for line in expr.strip().split('\n'):
            sym = sympify(line)
            parts.append(
                latex(sym) if hasattr(sym, 'is_number') and sym.is_number else latex(Matrix(sym)))
        return r"\quad\quad".join(parts)

    if operation_type == 'linear-system':
        equations = LinearSystemConverter.str_to_Eq(expr)
        eq_latex = [latex(eq) for eq in equations]
        return r"\begin{cases} " + r" \\ ".join(eq_latex) + r" \end{cases}"

    return latex(radsimp(sympify(expr)))
