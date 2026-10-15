import re
from functools import lru_cache

from sympy import Expr, Mul, S, bottom_up, evaluate, expand_log, latex, log, radsimp, simplify, sympify

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

# parse_latex only knows BACKSLASH-prefixed function names; a bare ``sin(...)``,
# ``exp(...)``, ``asin(...)``, ``sqrt(...)`` is split into a product of symbols
# (sin -> s*i*n). A bare ``^`` routes input through parse_latex, so sympy-style
# text like ``x*exp(x^2)`` would otherwise be mangled. Re-prefix known functions.
_BARE_FUNCS = {'exp', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'sinh', 'cosh',
               'tanh', 'coth', 'sech', 'csch', 'asin', 'acos', 'atan',
               'arcsin', 'arccos', 'arctan', 'ln', 'log'}
_BARE_CALL_RE = re.compile(r'(?<![A-Za-z\\])([A-Za-z]+)\s*\(')
_BARE_SQRT_RE = re.compile(r'(?<![A-Za-z\\])sqrt\s*\(')


def _match_paren(s: str, open_idx: int):
    """Return (inner_text, index_after_close) for the '(' at open_idx, or (None, ...)."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:i], i + 1
    return None, len(s)


def _sqrt_to_braces(s: str) -> str:
    """Rewrite bare ``sqrt(ARG)`` -> ``\\sqrt{ARG}`` (parse_latex rejects ``\\sqrt(``)."""
    out, i = [], 0
    while True:
        m = _BARE_SQRT_RE.search(s, i)
        if not m:
            out.append(s[i:])
            break
        inner, after = _match_paren(s, m.end() - 1)
        if inner is None:
            out.append(s[i:m.end()])
            i = m.end()
            continue
        out.append(s[i:m.start()] + '\\sqrt{' + _sqrt_to_braces(inner) + '}')
        i = after
    return ''.join(out)


def _prefix_bare_funcs(s: str) -> str:
    """Prefix ``\\`` to bare known function names before a '(' (sqrt already handled)."""
    def rep(match):
        name = match.group(1)
        return '\\' + name + '(' if name.lower() in _BARE_FUNCS else match.group(0)
    return _BARE_CALL_RE.sub(rep, s)


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
    # Re-serialize bare (non-backslash) function names parse_latex would split:
    # sqrt(...) -> \sqrt{...} (brace-only), and exp/sin/asin/... -> \exp(\...).
    s = _sqrt_to_braces(s)
    s = _prefix_bare_funcs(s)
    return s


def _drop_unit_factor(node):
    """Drop parse_latex's redundant ``Integer(1)`` factor from a product.

    parse_latex turns ``1/x^2`` into ``Mul(1, 1/x**2)``, whose LaTeX shows a
    stray leading ``1`` (``1 \frac{1}{x^{2}}``). Only removes identity factors
    and leaves a genuine lone ``1`` intact."""
    if isinstance(node, Mul) and any(a is S.One for a in node.args):
        kept = [a for a in node.args if a is not S.One]
        return Mul(*kept) if kept else S.One
    return node


@lru_cache(maxsize=512)
def cached_latex(expr) -> str:
    return latex(expr)


@lru_cache(maxsize=256)
def cached_simplify(expr: Expr, inverse: bool = True) -> Expr:
    return simplify(expr, inverse=inverse)


@lru_cache(maxsize=1024)
def cached_doit(operation) -> Expr:
    return operation.doit()


# Users type "infinity" several ways in bound fields (∞, 中文, the word). sympify
# rejects the unicode char and treats the word as a bare Symbol, so a definite
# bound like upper='∞' fails to parse and the whole integral returns no result.
# Map every spelling to SymPy's oo before parsing.
_INF_WORD_RE = re.compile(r'(?<![A-Za-z_])[Ii]nfinity(?![A-Za-z_])')
_INF_TOKENS = ('负无穷大', '正无穷大', '无穷大', '负无穷', '正无穷', '无穷')


def _normalize_infinities(s: str) -> str:
    """Rewrite unicode/中文/word infinity tokens to sympy's ``oo``.

    Longest tokens are replaced first so ``负无穷大`` never leaves a ``无穷大``
    fragment; a leading ``-`` (e.g. ``-∞``) survives adjacency to become ``-oo``."""
    s = s.replace('\u221e', 'oo')
    for token in _INF_TOKENS:
        if token in s:
            s = s.replace(token, '-oo' if token.startswith('负') else 'oo')
    return _INF_WORD_RE.sub('oo', s)


@lru_cache(maxsize=1024)
def cached_sympify(expr_str: str) -> Expr:
    # Callers may pass an already-sympified object (e.g. the limit calculator
    # pre-sympifies); sympify is then idempotent and the regex guard must not run.
    if not isinstance(expr_str, str):
        return sympify(expr_str)  # type: ignore
    expr_str = _normalize_infinities(expr_str)
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
            return bottom_up(parse_latex(normalized), _drop_unit_factor)
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
