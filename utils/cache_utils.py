import re
from functools import lru_cache

from sympy import Expr, evaluate, expand_log, latex, log, radsimp, simplify, sympify

from config import LARGE_POWER_EXPONENT_DIGITS

_HUGE_POWER_RE = re.compile(r'(?:\*\*|\^)\s*-?(\d+)')
_LATEX_COMMAND_RE = re.compile(r'\\[a-zA-Z]')
# MathLive renders powers/subscripts with braced ^{...} / _{...} (SymPy source uses
# ** and bare names), so a braced caret/underscore marks WYSIWYG LaTeX input.
_LATEX_MARK_RE = re.compile(r'\^\{|_\{')


def _has_huge_power_literal(expr_str: str) -> bool:
    """True if the input has a power whose integer exponent literal is very large.

    Detection uses the exponent literal's digit count as a cheap pre-check so we
    can avoid ever materializing huge integer powers such as ``2**1000000``.
    """
    return any(len(match.group(1)) >= LARGE_POWER_EXPONENT_DIGITS
               for match in _HUGE_POWER_RE.finditer(expr_str))


def _is_latex(expr_str: str) -> bool:
    """Heuristic: a backslash command or a caret/braced-subscript marks LaTeX input.

    A bare ``^`` (e.g. MathLive's ``x^2+3x``) is also treated as LaTeX because
    raw ``sympify`` rejects both ``^`` and implicit multiplication, whereas
    ``parse_latex`` handles them. ``_`` is deliberately excluded (common in
    plain identifiers); MathLive subscripts always emit the braced ``_{`` form.
    """
    return bool(_LATEX_COMMAND_RE.search(expr_str)
                or _LATEX_MARK_RE.search(expr_str)
                or '^' in expr_str)


# MathLive emits LaTeX that parse_latex mis-handles; normalize it first.
_TEXT_RE = re.compile(r'\\(?:text|textrm|textnormal|mathrm|mbox|operatorname)\s*\{[^{}]*\}')
_LN_RE = re.compile(r'(?<![A-Za-z])\\ln\b')
_EPOW_BRACE_RE = re.compile(r'(?<![A-Za-z\\])e\^\{([^{}]*)\}')
_EPOW_CHAR_RE = re.compile(r'(?<![A-Za-z\\])e\^([A-Za-z0-9])')
_UNFILLED_RES = (
    re.compile(r'\\square\b|\\Box\b|\\bigbox\b|\\placeholder\b'),
    re.compile(r'\\frac\s*\{\s*\}\s*\{\s*\}'),
    re.compile(r'\\sqrt\s*(?:\[\s*\])?\s*\{\s*\}'),
    re.compile(r'\\(?:int|sum|prod|lim|d?frac)\b[^{}]*\{\s*\}'),
)


def _normalize_mathlive_latex(s: str) -> str:
    """Turn MathLive WYSIWYG LaTeX into a form parse_latex handles correctly.

    Raises ValueError with a friendly message when the input still contains
    unfilled placeholder boxes, so the UI can tell the user to complete them."""
    if any(rx.search(s) for rx in _UNFILLED_RES):
        raise ValueError('请先填写输入框中的占位（□）后再计算')
    s = _TEXT_RE.sub('', s)               # drop prose labels
    s = _LN_RE.sub(r'\\log', s)           # \ln -> \lnatural log
    s = _EPOW_BRACE_RE.sub(r'\\exp{\1}', s)   # e^{x} -> \exp{x} (true exponential)
    s = _EPOW_CHAR_RE.sub(r'\\exp{\1}', s)
    return s


@lru_cache(maxsize=512)
def cached_latex(expr) -> str:
    return latex(expr)


@lru_cache(maxsize=256)
def cached_simplify(expr: Expr, inverse: bool = True) -> Expr:
    return simplify(expr, inverse=inverse)


@lru_cache(maxsize=1024)
def cached_doit(operation) -> Expr:
    return operation.doit()


@lru_cache(maxsize=1024)
def cached_sympify(expr_str: str) -> Expr:
    # Callers may pass an already-sympified object (e.g. the limit calculator
    # pre-sympifies); sympify is then idempotent and the regex guard must not run.
    if not isinstance(expr_str, str):
        return sympify(expr_str)  # type: ignore
    if _has_huge_power_literal(expr_str):
        # Parse without evaluation so a huge integer power stays symbolic instead
        # of being computed (which wastes memory and breaks int -> str rendering).
        # Checked before the LaTeX branch because a bare ``^`` now marks LaTeX and
        # parse_latex would otherwise materialize the huge value.
        with evaluate(False):
            return sympify(expr_str)
    if _is_latex(expr_str):
        # The interactive WYSIWYG cell input emits LaTeX; normalize MathLive quirks
        # (raising a friendly error on unfilled placeholders) then parse.
        normalized = _normalize_mathlive_latex(expr_str)
        try:
            from sympy.parsing.latex import parse_latex  # pylint: disable=import-outside-toplevel
            return parse_latex(normalized)
        except Exception:  # pylint: disable=broad-exception-caught
            return sympify(expr_str)  # type: ignore
    return sympify(expr_str)  # type: ignore


@lru_cache(maxsize=128)
def cached_simplify_radsimp(expr: Expr, is_radsimp: bool) -> Expr:
    if is_radsimp:
        expr = radsimp(expr)
    if expr.has(log):
        expr = expand_log(expr, force=True)
    return expr
