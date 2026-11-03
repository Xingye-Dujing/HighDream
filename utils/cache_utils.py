import re
from functools import lru_cache

from sympy import (Expr, Mul, S, Symbol, bottom_up, evaluate, expand_log,
                   latex, log, pi, radsimp, simplify, sympify)
from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                        parse_expr, standard_transformations)

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


def has_unfilled_placeholders(expr_str: str) -> bool:
    """True when the UI still has an empty template slot (□ / empty fraction)."""
    return any(rx.search(str(expr_str)) for rx in _UNFILLED_RES)


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
# Only real prose wrappers are dropped here.  \mathrm/\operatorname are kept:
# dropping a function name such as \operatorname{re}(z) used to turn it into
# just (z), silently losing the function.
_TEXT_RE = re.compile(r'\\(?:text|textrm|textnormal|mbox)\s*\{[^{}]*\}')
_LN_RE = re.compile(r'(?<![A-Za-z])\\ln\b')
_EPOW_BRACE_RE = re.compile(r'(?<![A-Za-z\\])e\^\{([^{}]*)\}')
_EPOW_CHAR_RE = re.compile(r'(?<![A-Za-z\\])e\^([A-Za-z0-9])')
_OPNAME_RE = re.compile(r'\\(?:operatorname|mathrm)\s*\{([^{}]*)\}')
# MathLive/SymPy spellings that parse_latex understands only with a backslash.
_OPNAME_MAP = {
    're': r'\Re', 'Re': r'\Re', 'im': r'\Im', 'Im': r'\Im',
    'arg': r'\arg', 'ln': r'\ln', 'log': r'\log', 'exp': r'\exp',
    'sin': r'\sin', 'cos': r'\cos', 'tan': r'\tan', 'cot': r'\cot',
    'sec': r'\sec', 'csc': r'\csc', 'sinh': r'\sinh', 'cosh': r'\cosh',
    'tanh': r'\tanh', 'coth': r'\coth', 'asin': r'\arcsin',
    'acos': r'\arccos', 'atan': r'\arctan', 'arcsin': r'\arcsin',
    'arccos': r'\arccos', 'arctan': r'\arctan',
}
_DIFF_D_RE = re.compile(r'\\mathrm\s*\{\s*d\s*\}')
# parse_latex (1.13) turns \pi into Symbol('pi') instead of the constant pi.
_CONST_SYMBOLS = {Symbol('pi'): pi}
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
# parse_latex has no ``\asin``/``\acos``/``\atan`` entry (they bind to an undefined
# Function and defeat every real-function rule). Its real inverse-trig spellings are
# ``\arcsin``/``\arccos``/``\arctan``. Sympy-style input uses the short names, so alias
# them here before parse_latex sees the string.
_INVTRIG_ALIAS = {'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan'}
_INVTRIG_BSLASH_RE = re.compile(r'\\(asin|acos|atan)\b')
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


_ATOM_NAME_RE = re.compile(r'[A-Za-z]+')
_CMD_NAME_RE = re.compile(r'\\[A-Za-z]+')


def _scan_bare_power_atom(s, j):
    """Locate the single operand of a bare ``^`` that starts at index j.

    Returns ``(atom_text, index_after)`` or ``(None, j)`` when there is nothing safe
    to wrap (end of string, or already a braced ``^{...}``). Recognises an optional
    sign then a ``(...)`` group, a ``\\name``(optionally called), an identifier
    (optionally followed by a ``(…)`` call), or a number."""
    n = len(s)
    if j >= n or s[j] == '{':
        return None, j
    k = j
    if s[k] in '+-':
        k += 1
    if k >= n:
        return None, j
    if s[k] == '(':
        inner, after = _match_paren(s, k)
        return (s[j:after], after) if inner is not None else (None, j)
    if s[k] == '\\':
        m = _CMD_NAME_RE.match(s, k)
        if not m:
            return None, j
        after = m.end()
        if after < n and s[after] == '(':
            _inner, after2 = _match_paren(s, after)
            if _inner is not None:
                after = after2
        return s[j:after], after
    m = _ATOM_NAME_RE.match(s, k)
    if m:
        after = m.end()
        if after < n and s[after] == '(':
            _inner, after2 = _match_paren(s, after)
            if _inner is not None:
                after = after2
        return s[j:after], after
    if s[k].isdigit() or s[k] == '.':
        e = k
        while e < n and (s[e].isdigit() or s[e] == '.'):
            e += 1
        return s[j:e], e
    return None, j


def _brace_bare_powers(s: str) -> str:
    """Wrap a bare ``^`` operand in braces so parse_latex keeps the whole exponent.

    parse_latex silently truncates an unbraced multi-token exponent — ``3^sin(x)``
    parses to just ``3`` — whereas braced ``3^{sin(x)}`` is correct. MathLive emits
    ``x^{2}`` already braced, so those are left untouched. Subscripts (``_``) are NOT
    touched because a bare ``_`` is a common identifier character. Runs before the
    bare-function re-prefixing so an exponent like ``sin(x)`` is captured intact."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == '^' and (i == 0 or s[i - 1] != '\\'):
            atom, after = _scan_bare_power_atom(s, i + 1)
            if atom:
                out.append('^{' + atom + '}')
                i = after
                continue
        out.append(ch)
        i += 1
    return ''.join(out)


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
        low = name.lower()
        if low in _BARE_FUNCS:
            return '\\' + _INVTRIG_ALIAS.get(low, low) + '('
        return match.group(0)
    return _BARE_CALL_RE.sub(rep, s)


def _replace_opnames(s: str) -> str:
    """Turn ``\\operatorname{re}`` / ``\\mathrm{d}`` into parse_latex spellings.

    Known function names get the backslash form parse_latex understands; other
    names keep their text so a matrix operator (``\\mathrm{rank}``) can still be
    recognized before this scalar parser is reached.
    """
    # \operatorname{conjugate}(z) has no parse_latex spelling; overline does.
    s = re.sub(
        r'\\operatorname\s*\{\s*(?:conjugate|conj)\s*\}\s*\(([^()]*)\)',
        r'\\overline{\1}',
        s,
    )

    def repl(match):
        name = match.group(1).strip()
        mapped = _OPNAME_MAP.get(name)
        if mapped:
            return mapped + ' '
        return name + ' '

    return _OPNAME_RE.sub(repl, s)


_ABS_TOKEN = 987654321


def _extract_abs_groups(s: str):
    """Protect ``|x|`` / ``Abs(x)`` while parse_latex parses the rest.

    SymPy's parse_latex has no absolute-value syntax.  Replace each group with a
    unique integer literal, parse the outer expression, then substitute
    ``Abs(inner)`` back.  Returns ``(rewritten, {token: inner})``.
    """
    groups = {}

    def repl(match):
        inner = match.group(1)
        token = str(_ABS_TOKEN + len(groups))
        groups[token] = inner
        return token

    s = re.sub(r'\\left\|([\s\S]*?)\\right\|', repl, s)
    s = re.sub(r'\\operatorname\s*\{\s*abs\s*\}\s*\(([^()]*)\)', repl, s)
    s = re.sub(r'(?<![A-Za-z\\])Abs\s*\(([^()]*)\)', repl, s)
    s = re.sub(r'(?<![A-Za-z\\])abs\s*\(([^()]*)\)', repl, s)
    return s, groups


def _restore_abs_groups(expr, groups):
    if not groups or not hasattr(expr, 'subs'):
        return expr
    from sympy import Abs, Integer  # pylint: disable=import-outside-toplevel
    for token, inner in groups.items():
        value = cached_sympify(inner)
        expr = expr.subs(Integer(int(token)), Abs(value))
    return expr


def _normalize_mathlive_latex(s: str) -> str:
    """Turn MathLive WYSIWYG LaTeX into a form parse_latex handles correctly.

    Raises ValueError with a friendly message when the input still contains
    unfilled placeholder boxes, so the UI can tell the user to complete them."""
    if any(rx.search(s) for rx in _UNFILLED_RES):
        raise ValueError('请先填写输入框中的占位（□）后再计算')
    s = _TEXT_RE.sub('', s)               # drop prose labels
    s = _replace_opnames(s)               # \operatorname{re} -> \Re, \mathrm{d} -> d
    s = _DIFF_D_RE.sub('d', s)            # \mathrm{d}x -> dx
    s = _LN_RE.sub(r'\\log', s)           # \ln -> \log
    s = _brace_bare_powers(s)             # 3^sin(x) -> 3^{sin(x)} (parse_latex truncates bare carets)
    s = _EPOW_BRACE_RE.sub(r'\\exp{\1}', s)   # e^{x} -> \exp{x} (true exponential)
    s = _EPOW_CHAR_RE.sub(r'\\exp{\1}', s)
    # \asin / \acos / \atan (MathLive may emit these) -> parse_latex's real names.
    s = _INVTRIG_BSLASH_RE.sub(lambda m: '\\' + _INVTRIG_ALIAS[m.group(1)], s)
    # Re-serialize bare (non-backslash) function names parse_latex would split:
    # sqrt(...) -> \sqrt{...} (brace-only), and exp/sin/asin/... -> \exp(\...).
    s = _sqrt_to_braces(s)
    s = _prefix_bare_funcs(s)
    return s


def _substitute_known_constants(expr):
    """Replace parse_latex's Symbol('pi') with the real constant pi."""
    if hasattr(expr, 'subs'):
        return expr.subs(_CONST_SYMBOLS, simultaneous=True)
    return expr


def _drop_unit_factor(node):
    """Drop parse_latex's redundant ``Integer(1)`` factor from a product.

    parse_latex turns ``1/x^2`` into ``Mul(1, 1/x**2)``, whose LaTeX shows a
    stray leading ``1`` (``1 \frac{1}{x^{2}}``). Only removes identity factors
    and leaves a genuine lone ``1`` intact."""
    if isinstance(node, Mul) and any(a is S.One for a in node.args):
        kept = [a for a in node.args if a is not S.One]
        return Mul(*kept) if kept else S.One
    return node


def _flatten_mul(node):
    """Rebuild a ``Mul`` with evaluation on so nested products collapse.

    parse_latex can leave products unevaluated — e.g. an exponent as
    ``Mul(1/2, Mul(-1, x))`` instead of the canonical ``-x/2``. Structural
    matching in the substitution rule compares expressions with ``==``/
    ``free_symbols``, which fails on the non-canonical shape, so an otherwise
    solvable integral comes back unresolved. ``Mul(*args)`` forces sympy to
    re-associate and fold the numeric coefficients. Used with ``bottom_up`` so
    operands (and exponents) are flattened before their parents rebuild."""
    if node.is_Mul:
        return Mul(*node.args)
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
        prepared, abs_groups = _extract_abs_groups(expr_str)
        normalized = _normalize_mathlive_latex(prepared)
        try:
            from sympy.parsing.latex import parse_latex  # pylint: disable=import-outside-toplevel
            parsed = bottom_up(parse_latex(normalized), _drop_unit_factor)
            parsed = _substitute_known_constants(parsed)
            parsed = _restore_abs_groups(parsed, abs_groups)
            return bottom_up(parsed, _flatten_mul)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Never silently fall back to sympify for LaTeX: it turns unknown
            # commands into products of symbols (e.g. \operatorname{re} -> re),
            # which produces confident but wrong answers.
            raise ValueError(f'无法解析该数学输入，请检查 LaTeX 写法: {exc}') from exc
    try:
        return sympify(expr_str)  # type: ignore
    except Exception:  # pylint: disable=broad-exception-caught
        # Plain text may use textbook implicit multiplication (x y, 2x, 2(x+1)).
        # Try SymPy's parser with that transformation before surfacing an error.
        try:
            return parse_expr(
                expr_str,
                transformations=standard_transformations + (implicit_multiplication_application,),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise ValueError(f'无法解析表达式: {expr_str}') from exc


@lru_cache(maxsize=128)
def cached_simplify_radsimp(expr: Expr, is_radsimp: bool) -> Expr:
    if is_radsimp:
        expr = radsimp(expr)
    if expr.has(log):
        expr = expand_log(expr, force=True)
    return expr
