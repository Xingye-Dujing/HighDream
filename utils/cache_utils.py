import ast
import builtins
import math
import re
from functools import lru_cache

from sympy import (Add, Expr, Mul, S, Symbol, bottom_up, evaluate, expand_log,
                   latex, log, pi, radsimp, simplify, sympify)
from sympy.parsing.sympy_parser import (convert_xor,
                                        implicit_multiplication_application,
                                        parse_expr, standard_transformations)

from config import LARGE_POWER_EXPONENT_DIGITS
from utils.expression_utils import extract_paren_arg, find_matching_paren

# MathLive writes powers as ^{N} while SymPy source uses **N; both spellings
# must be detected before the exponent literal is ever materialized.
_HUGE_POWER_RE = re.compile(r'(?:\*\*|\^)\s*(?:\{\s*)?-?(\d+)')
_LATEX_COMMAND_RE = re.compile(r'\\[a-zA-Z]')
# MathLive renders powers/subscripts with braced ^{...} / _{...} (SymPy source uses
# ** and bare names), so a braced caret/underscore marks WYSIWYG LaTeX input.
_LATEX_MARK_RE = re.compile(r'\^\{|_\{')
# Cap on exponent magnitudes we are willing to let Python materialize at all.
# ``2**10**9`` / ``2**(10**9)`` have no huge flat literal but would still build
# a 10**9-exponent power, so towers of integer literals are scanned for their
# outermost exponent's value instead of looking only at the literal text.
_EXPONENT_CAP = 10_000
_INT_TOKEN = r'-?\d+'
_TOWER_RE = re.compile(_INT_TOKEN + r'(?:\s*(?:\*\*|\^)\s*' + _INT_TOKEN + r'){2,}')
_PAREN_POWER_RE = re.compile(r'(?:\*\*|\^)\s*\(')


def _int_tokens(text: str):
    """Integer tokens of a pure integer power tower like ``3**5**2``, or None."""
    nums = []
    for part in re.split(r'(?:\*\*|\^)', text):
        part = part.strip().strip('{}').strip()
        if not re.fullmatch(r'-?\d+', part):
            return None
        nums.append(int(part))
    return nums


def _capped_tower_value(tokens):
    """Right-associative value of an integer power tower, capped at _EXPONENT_CAP.

    ``2**10**9`` has exponent ``10**9``; the 1e9 must be detected without
    ever computing ``2**1e9``. Any step exceeding _EXPONENT_CAP stops the
    evaluation and reports the cap plus one instead."""
    value = abs(tokens[-1])
    for token in reversed(tokens[:-1]):
        if value > _EXPONENT_CAP:
            return _EXPONENT_CAP + 1
        value = abs(token) ** value
    return value


def _arith_only_value(text: str) -> bool | float | None:
    """Value of a string containing only ``+ - * /`` and integer literals (or
    ``math.inf``), or None when it is not such a string or not finite.

    Used to judge whether a parenthesised exponent such as ``(1000000*1000000)``
    is huge without letting Python compute ``2**10**12``."""
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError:
        return None

    def _fold(node):
        if isinstance(node, ast.Expression):
            return _fold(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _fold(node.operand)
            if value is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = _fold(node.left), _fold(node.right)
            if left is None or right is None:
                return None
            try:
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                elif isinstance(node.op, ast.Div):
                    if right == 0:
                        return None
                    value = left / right
                else:
                    return None
            except OverflowError:
                return math.inf
            return math.inf if abs(value) > 1e300 else value
        return None

    return _fold(tree)


def _has_huge_power_literal(expr_str: str, _depth: int = 0) -> bool:
    """True if the input can produce an enormous integer power when evaluated.

    Catches three shapes, none of which are ever materialized:
    - a flat exponent literal with many digits (``2**1000000``)
    - a tower of integer literals whose outermost exponent is huge (``2**10**9``)
    - a parenthesised tower used as an exponent (``2**(10**9)``)

    ``_depth`` caps the parenthesised recursion so a pathological nested input
    cannot overflow the interpreter stack.
    """
    if _depth > 8:
        return True
    if any(len(match.group(1)) >= LARGE_POWER_EXPONENT_DIGITS
           for match in _HUGE_POWER_RE.finditer(expr_str)):
        return True
    for match in _TOWER_RE.finditer(expr_str):
        tokens = _int_tokens(match.group(0))
        if tokens and _capped_tower_value(tokens[1:]) > _EXPONENT_CAP:
            return True
    for match in _PAREN_POWER_RE.finditer(expr_str):
        inner = extract_paren_arg(expr_str, match.end() - 1)
        if not inner:
            continue
        value = _arith_only_value(inner)
        if value is not None and (value is math.inf or abs(value) > _EXPONENT_CAP):
            return True
        tokens = _int_tokens(inner)
        if tokens is not None and _capped_tower_value(tokens) > _EXPONENT_CAP:
            return True
        if _has_huge_power_literal(inner, _depth + 1):
            return True
    return False


def has_unfilled_placeholders(expr_str: str) -> bool:
    """True when the UI still has an empty template slot (□ / empty fraction)."""
    return any(rx.search(str(expr_str)) for rx in _UNFILLED_RES)


def _is_latex(expr_str: str) -> bool:
    """Heuristic: a backslash command, braced index, or equation marks LaTeX.

    Bare ``x^2`` is handled by the plain-text parser: it supports ``^``
    (convert_xor), implicit multiplication, function names and Greek variable
    names. Routing it through parse_latex corrupted multi-letter names
    (``pi*x^2`` became ``i*p*x**2``, ``alpha*x^2`` became ``a**2*h*l*p*x**2``).
    Equations (``x^2+y^2=1``) still use LaTeX because the plain parser has no
    ``=`` syntax. MathLive's WYSIWYG output braces its exponents (``x^{2}``) and
    subscripts (``_{...}``), so it still routes here.
    """
    return bool(_LATEX_COMMAND_RE.search(expr_str)
                or _LATEX_MARK_RE.search(expr_str)
                or '=' in expr_str)


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
_CONJUGATE_RE = re.compile(
    r'\\operatorname\s*\{\s*(?:conjugate|conj)\s*\}\s*\(([^()]*)\)')
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
    close = find_matching_paren(s, open_idx)
    if close is None:
        return None, len(s)
    return extract_paren_arg(s, open_idx), close + 1


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
    s = _CONJUGATE_RE.sub(r'\\overline{\1}', s)

    def repl(match):
        name = match.group(1).strip()
        mapped = _OPNAME_MAP.get(name)
        if mapped:
            return mapped + ' '
        # An unknown \operatorname{foo}(...) would be split by parse_latex into
        # a product of single-letter symbols ("rank" -> a*n*r*k(x)), yielding a
        # confident but WRONG answer. Matrix operators and the differential
        # "\mathrm{d}" are not followed by "(", so they still keep their text
        # for the upstream folding / _DIFF_D_RE pass.
        if re.match(r'\s*\(', s[match.end():]):
            raise ValueError(f'不支持的运算符名称: {name}')
        return name + ' '

    return _OPNAME_RE.sub(repl, s)


_ABS_TOKEN = 987654321
# Match the INNERMOST \left|...\right| first. The old non-greedy [\s\S]*?
# stopped at the first \right| and left a stray "\right|" for nested groups.
_ABS_LEFT_INNER_RE = re.compile(r'\\left\|((?:(?!\\left\|)[\s\S])*?)\\right\|')
_ABS_CALL_RES = (
    re.compile(r'\\operatorname\s*\{\s*abs\s*\}\s*\('),
    re.compile(r'(?<![A-Za-z\\])Abs\s*\('),
    re.compile(r'(?<![A-Za-z\\])abs\s*\('),
)


def _extract_abs_groups(s: str):
    """Protect ``|x|`` / ``Abs(x)`` while parse_latex parses the rest.

    SymPy's parse_latex has no absolute-value syntax.  Replace each group with
    a unique integer literal, parse the outer expression, then substitute
    ``Abs(inner)`` back.  Nested groups and arguments containing parentheses
    (``Abs(f(x))``) are handled with balanced-paren scanning, innermost first.
    Returns ``(rewritten, {token: inner})``.
    """
    groups = {}

    def new_token(inner):
        # Expand already-protected tokens back into LaTeX so nested groups
        # restore recursively instead of leaving a bare integer inside Abs().
        for token, value in list(groups.items()):
            if token in inner:
                inner = inner.replace(token, r'\left|' + value + r'\right|')
        token = str(_ABS_TOKEN + len(groups))
        groups[token] = inner
        return token

    def repl(match):
        return new_token(match.group(1))

    for _ in range(64):
        s, count = _ABS_LEFT_INNER_RE.subn(repl, s)
        if not count:
            break

    for pattern in _ABS_CALL_RES:
        for _ in range(64):
            match = pattern.search(s)
            if not match:
                break
            open_idx = s.index('(', match.start())
            close_idx = find_matching_paren(s, open_idx)
            if close_idx is None:
                break
            token = new_token(s[open_idx + 1:close_idx])
            s = s[:match.start()] + token + s[close_idx + 1:]
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
    """Rebuild an ``Add``/``Mul`` with evaluation on so nesting collapses.

    parse_latex can leave sums/products unevaluated — e.g. ``1/((x**2+2*x)+5)``
    (a nested Add as the denominator) or an exponent as ``Mul(1/2, Mul(-1, x))``
    instead of the canonical ``-x/2``. Structural matching in the rule engine
    compares expressions with ``==``/``free_symbols`` and degree/coefficient
    extraction, which fails on the non-canonical shape. The nested-Add case was
    severe: ``integral('1/(x^2+2*x+5)')`` missed the direct arctan rule, fell
    into integration by parts and ran for many minutes. ``Add(*args)`` /
    ``Mul(*args)`` force sympy to re-associate (and fold numeric coefficients)
    without touching ``Pow`` operands, so a huge unevaluated power stays
    unevaluated. Used with ``bottom_up`` so operands are flattened first."""
    if node.is_Mul:
        return Mul(*node.args)
    if node.is_Add:
        return Add(*node.args)
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


# -- Hardening -------------------------------------------------------------
# ``sympify``/``parse_expr`` evaluate generated Python code.  Without the two
# guards below a plain ``__import__('os').system(...)`` would run with full
# builtins.  User input is parsed in a namespace without builtins and rejected
# when it contains Python constructs a mathematical expression never needs.
_DANGEROUS_NAMES = frozenset({
    'eval', 'exec', 'compile', 'open', 'input', 'breakpoint', 'globals',
    'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr',
    'importlib', 'builtins', 'os', 'sys', 'subprocess', 'pathlib', 'io',
    'type', 'object', 'super', 'property', 'classmethod', 'staticmethod',
    'bytes', 'bytearray', 'memoryview', 'print', 'exit', 'quit', 'help',
    'license', 'credits', 'format', 'iter', 'next', 'filter', 'map',
})
_FORBIDDEN_AST_NODES = (
    ast.Attribute, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr,
    ast.JoinedStr, ast.FormattedValue, ast.Import, ast.ImportFrom,
    ast.Global, ast.Nonlocal,
)
_UNSAFE_TEXT_RE = re.compile(r'__|\.\s*[A-Za-z_]')


def _build_safe_globals():
    """SymPy's own ``from sympy import *`` namespace, minus every builtin.

    A small allowlist of *pure* builtins is re-added explicitly: ``sympify``
    normally exposes them, and without ``abs`` the input ``abs(x)/x`` became
    an undefined function instead of ``Abs(x)/x``, silently breaking limits.
    Nothing here can execute code (no import/eval/open/getattr/attributes).
    """
    namespace = {}
    exec('from sympy import *', namespace)  # pylint: disable=exec-used
    namespace['__builtins__'] = {}
    namespace['abs'] = namespace['Abs']
    namespace['min'] = namespace['Min']
    namespace['max'] = namespace['Max']
    for name in ('round', 'sum', 'len', 'pow', 'range', 'enumerate', 'zip',
                 'sorted', 'reversed', 'all', 'any', 'bool', 'int', 'float',
                 'complex', 'str', 'list', 'tuple', 'dict', 'set', 'frozenset',
                 'slice'):
        namespace[name] = getattr(builtins, name)
    return namespace


_SAFE_GLOBALS = _build_safe_globals()
# ``sympify`` parses strings with exactly the first set of transformations;
# reproducing it keeps ``xy`` a single symbol (no implicit multiplication),
# while the second set is the textbook-shorthand fallback the old code used.
_PLAIN_TRANSFORMATIONS = standard_transformations + (convert_xor,)
_IMPLICIT_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)

# Plain-text exponential spellings: e^x, e**x, e^{...}, e^(...). The prefix
# capture lets "2e^x"/"xe^x" keep their implicit multiplication (2*exp(x)).
_PLAIN_EPOW_RE = re.compile(
    r'(^|[\s\S])e\s*(?:\^|\*\*)\s*'
    r'(\{[^{}]*\}|\([^()]*\)|[A-Za-z0-9]+)')

# Greek-letter variable names collide with SymPy functions (``beta``, ``gamma``,
# ``zeta``, ...). ``beta*x`` used to raise ``TypeError: FunctionClass * Symbol``.
# Binding them as plain symbols in the parser's local dict makes textbook input
# such as ``alpha*x^2`` behave as the user expects.
_GREEK_VARIABLES = frozenset({
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'mu', 'nu', 'xi', 'rho', 'sigma', 'tau', 'phi', 'chi',
    'psi', 'omega',
})


def _plain_local_dict(expr_str: str):
    """Symbol overrides for Greek-letter identifiers present in ``expr_str``."""
    names = [name for name in _GREEK_VARIABLES
             if re.search(rf'(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])', expr_str)]
    if not names:
        return None
    return {name: Symbol(name) for name in names}


def _reject_unsafe_syntax(expr_str: str) -> None:
    """Reject Python syntax that a mathematical expression never needs."""
    if len(expr_str) > 4096:
        # Avoid pathological AST parsing on huge power literals; the blacklist
        # still applies and the builtin-free namespace remains in force.
        if _UNSAFE_TEXT_RE.search(expr_str):
            raise ValueError('表达式包含不允许的语法')
        return
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError:
        # "x y" / "2x" / "2(x+1)" are not valid Python; the implicit
        # multiplication transformation handles them after this check.
        if _UNSAFE_TEXT_RE.search(expr_str):
            raise ValueError('表达式包含不允许的语法') from None
        return
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            raise ValueError('表达式包含不允许的语法')
        if isinstance(node, ast.Name):
            if node.id.startswith('__') or node.id in _DANGEROUS_NAMES:
                raise ValueError('表达式包含不允许的语法')
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Call)):
                raise ValueError('表达式包含不允许的语法')
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if '__' in node.value:
                raise ValueError('表达式包含不允许的语法')


def _rewrite_plain_e_powers(expr_str: str) -> str:
    """Turn plain-text ``e^x`` / ``e**x`` into ``exp(x)``.

    The LaTeX branch already does this (MathLive emits ``e^{x}``); keeping the
    same meaning in the plain branch preserves textbook input. Braced and
    parenthesised exponents are kept as one unit; a preceding identifier gets an
    explicit ``*`` so ``2e^x`` / ``xe^x`` do not fuse into one symbol.
    """
    if 'e' not in expr_str:
        return expr_str

    def replace(match):
        prefix, exponent = match.group(1), match.group(2)
        if exponent[0] in '{(':
            exponent = exponent[1:-1]
        glue = '*' if prefix and (prefix[-1].isalnum() or prefix[-1] in '_)]') else ''
        return f'{prefix}{glue}exp({exponent})'

    return _PLAIN_EPOW_RE.sub(replace, expr_str)


def _parse_plain_input(expr_str: str, trusted: bool) -> Expr:
    """Reproduce ``sympify``'s string parsing without exposing builtins."""
    if not trusted:
        _reject_unsafe_syntax(expr_str)
    expr_str = _rewrite_plain_e_powers(expr_str)
    global_dict = None if trusted else _SAFE_GLOBALS
    local_dict = _plain_local_dict(expr_str)
    try:
        return parse_expr(expr_str, global_dict=global_dict,
                          local_dict=local_dict,
                          transformations=_PLAIN_TRANSFORMATIONS)
    except Exception:  # pylint: disable=broad-exception-caught
        try:
            return parse_expr(expr_str, global_dict=global_dict,
                              local_dict=local_dict,
                              transformations=_IMPLICIT_TRANSFORMATIONS)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise ValueError(f'无法解析表达式: {expr_str}') from exc


@lru_cache(maxsize=1024)
def cached_sympify(expr_str: str, trusted: bool = False) -> Expr:
    """Parse user text/LaTeX into a SymPy object.

    ``trusted=True`` is reserved for internally generated SymPy source strings;
    every HTTP-facing caller must keep the default so the builtin-free
    namespace and the syntax guard stay in force.
    """
    # Callers may pass an already-sympified object (e.g. the limit calculator
    # pre-sympifies); sympify is then idempotent and the regex guard must not run.
    if not isinstance(expr_str, str):
        return sympify(expr_str)  # type: ignore
    expr_str = _normalize_infinities(expr_str)
    if not trusted:
        # Run the syntax guard before choosing a parser: an input containing
        # "=" now routes to the LaTeX branch, and that must not become a way to
        # skip the blacklist (e.g. exec("x=1")).
        _reject_unsafe_syntax(expr_str)
    # Braced MathLive powers (2^{1000000}) are LaTeX: parse_latex keeps them as
    # an unevaluated Pow, so they must NOT go through the plain-text parser.
    if _has_huge_power_literal(expr_str) and not _is_latex(expr_str):
        # Parse without evaluation so a huge integer power stays symbolic instead
        # of being computed (which wastes memory and breaks int -> str rendering).
        # Checked before the LaTeX branch because a bare ``^`` now marks LaTeX and
        # parse_latex would otherwise materialize the huge value.
        with evaluate(False):
            return _parse_plain_input(expr_str, trusted)
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
    return _parse_plain_input(expr_str, trusted)


@lru_cache(maxsize=128)
def cached_simplify_radsimp(expr: Expr, is_radsimp: bool) -> Expr:
    if is_radsimp:
        expr = radsimp(expr)
    if expr.has(log):
        expr = expand_log(expr, force=True)
    return expr
