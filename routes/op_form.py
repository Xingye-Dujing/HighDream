r"""Bidirectional converter between the universal-box OPERATOR form and the MathLive
LaTeX WYSIWYG form for the operators the notebook uses.

Single source of truth for BOTH the docs/ migration and the runtime per-cell
"转换形式" toggle, so the two never drift. Reuses the existing argument splitters
and the LaTeX->nested-matrix / LaTeX-operator normalizers.

  operator form : int(1/(x**2+1), x), diff(x*sin(x), x), limit(..., x, 0, +),
                  det([[a,b],[c,d]]), inverse([[..]]), add([[..]],[[..]])
  latex form    : \int (\frac{1}{x^{2}+1})\, dx, \frac{d}{d x}(x \sin{\left(x \right)}),
                  \lim_{x \to 0^{+}}(...), \det \begin{bmatrix}a & b\\c & d\end{bmatrix},
                  \begin{bmatrix}..\end{bmatrix}^{-1}, bmatA + bmatB

Unrecognised expressions are returned unchanged (a no-op) so the toggle is always safe.
"""
# LaTeX templates are clearer with %-interpolation than f-strings (many literal braces).
# pylint: disable=consider-using-f-string

import re

from sympy import latex

from utils.matrix_latex import latex_to_nested
from utils.cache_utils import _has_huge_power_literal, cached_sympify

from core.latex_syntax import (_extract_paren_arg, _fold_matrix_ops,
                               _normalize_latex_ops, _split_top_args)

# operator name -> LaTeX display prefix for a single-matrix operand (bare prefix + matrix)
_MAT_PREFIX = {
    'det': r'\det ', 'determinant': r'\det ',
    'rank': r'\mathrm{rank} ', 'trace': r'\mathrm{tr} ', 'tr': r'\mathrm{tr} ',
    'ref': r'\mathrm{rref} ', 'rref': r'\mathrm{rref} ', 'rowreduce': r'\mathrm{rref} ',
    'lu': r'\mathrm{lu} ', 'diag': r'\mathrm{diag} ', 'diagonalize': r'\mathrm{diag} ',
    'svd': r'\mathrm{svd} ', 'singular': r'\mathrm{svd} ',
    'schur': r'\mathrm{schur} ', 'eigen': r'\mathrm{eigen} ',
    'jordan': r'\mathrm{jordan} ', 'charpoly': r'\mathrm{charpoly} ',
    'orthogonal': r'\mathrm{orthogonalize} ', 'orthogonalize': r'\mathrm{orthogonalize} ',
    'gramschmidt': r'\mathrm{orthogonalize} ',
    'nullspace': r'\mathrm{nullspace} ', 'colspace': r'\mathrm{colspace} ',
    'rowspace': r'\mathrm{rowspace} ',
    'inverse': None, 'inv': None, 'invert': None,  # handled as ^{-1}
    'transpose': None,                              # handled as ^{T}
    'pinv': None,                                   # handled as ^{+}
}
_BIN_OP = {'add': '+', 'subtract': '-', 'minus': '-', 'multiply': r'\cdot', 'matmul': r'\cdot'}
_EXPR_ONLY = {'simplify', 'expand', 'factor', 'equivalent', 'expr'}
_LATEX_SIDES = {'+': '^{+}', '-': '^{-}', '+-': '^{\\pm}', 'pm': '^{\\pm}', '\\pm': '^{\\pm}'}
_OP_SIDES = {'+': '+', '-': '-', '^{+}': '+', '^{-}': '-', '^{\\pm}': '+-', '^+': '+', '^-': '-'}
# spellings SymPy/the box accept for an infinite limit point (one-sided == two-sided there)
_INF_POINTS = {'oo', '+oo', '-oo', 'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity'}


def _is_mat(arg):
    return isinstance(arg, str) and arg.strip().startswith('[[')


def _L(expr):
    """Latex of a scalar/expression operand.

    Normalizes to pure SymPy syntax (``^``->``**``, ``ln``->``log``) before parsing.
    The caret rewrite means cached_sympify takes its plain-text branch, so the
    builtin-free parser guard applies without mangling a bare ``sqrt(...)`` into
    ``s*q*r*t``. Falls back to raw text."""
    try:
        s = re.sub(r'\s*\^\s*', '**', str(expr))
        s = re.sub(r'\bln\s*\(', 'log(', s)
        # Do not materialize a giant integer power (2**1000000 etc.) here: it can
        # burn CPU/memory and then crash latex()'s int->str conversion.
        if _has_huge_power_literal(s):
            return str(expr)
        return latex(cached_sympify(s))
    except Exception:  # noqa
        return str(expr)


def _caret(text):
    """Display form for an operand: keep its own text, only ``**``->``^``.

    The input parser now round-trips bare function names + carets, so we must NOT
    re-serialize the operand via sympy.latex (that emits ``\\operatorname{atan}`` and
    nested ``e^{x^{2}}`` which do NOT parse back). Verbatim keeps it correct."""
    return re.sub(r'\s*\*\*\s*', '^', str(text))


def _pt(text):
    """Limit point display: oo/-oo -> \\infty/-\\infty, else verbatim caret form."""
    t = str(text).strip()
    if t in ('oo', 'inf', 'infinity', '+oo'):
        return r'\infty'
    if t in ('-oo', '-inf', '-infinity'):
        return r'-\infty'
    return _caret(t)


def nested_to_bmatrix(nested):
    """``[[a,b],[c,d]]`` -> ``\\begin{bmatrix}a & b\\\\c & d\\end{bmatrix}``."""
    s = nested.strip()
    if not s.startswith('['):
        return _L(s)
    rows = re.findall(r'\[([^\[\]]*)\]', s)
    if not rows:
        return _L(s)
    body = '\\\\'.join(' & '.join(c.strip() for c in r.split(',')) for r in rows)
    return r'\begin{bmatrix}' + body + r'\end{bmatrix}'


def _mat_operand(arg):
    return nested_to_bmatrix(arg) if _is_mat(arg) else _L(arg)


def _parse_call(expr):
    """Return (lower_name, [args]) for a leading ``name(...)`` call, else None."""
    m = re.match(r'\s*([A-Za-z_][A-Za-z_0-9]*)\s*\(', expr)
    if not m:
        return None
    inner = _extract_paren_arg(expr, m.end() - 1)
    if inner is None:
        return None
    return m.group(1).lower(), _split_top_args(inner)


def operator_to_latex(expr):
    """Rewrite one universal-box operator call into MathLive-renderable LaTeX."""
    parsed = _parse_call(expr)
    if not parsed:
        return expr
    name, a = parsed
    if name in ('int', 'integrate', 'improper', 'improperintegral'):
        body, var = (a + ['x'] * 2)[:2]
        if len(a) >= 4 and str(a[2]).strip() and str(a[3]).strip():
            return r'\int_{%s}^{%s} (%s)\, d%s' % (_caret(a[2]), _caret(a[3]), _caret(body), var)
        return r'\int (%s)\, d%s' % (_caret(body), var)
    if name in ('diff', 'derivative', 'd'):
        body, var = (a + ['x'] * 2)[:2]
        if len(a) >= 3 and str(a[2]).strip() not in ('', '1'):
            return r'\frac{d^{%s}}{d %s^{%s}} (%s)' % (a[2], var, a[2], _caret(body))
        return r'\frac{d}{d %s} (%s)' % (var, _caret(body))
    if name == 'limit':
        if len(a) < 3:
            return expr
        body, var, point = a[0], a[1], a[2]
        if len(a) >= 4:
            side = _LATEX_SIDES.get(a[3].strip(), '')
        elif str(point).strip().lower() not in _INF_POINTS:
            # A 3-arg limit computes one-sided from the right at runtime (default '+');
            # mirror that so the "转换形式" toggle never flips it to a two-sided limit
            # (e.g. lim_{x->0} 1/x would otherwise change from +oo to complex oo).
            side = _LATEX_SIDES['+']
        else:
            side = ''  # at +/-oo the one- and two-sided limits coincide
        return r'\lim_{%s \to %s%s} (%s)' % (var, _pt(point), side, _caret(body))
    if name in _BIN_OP and len(a) >= 2:
        return '%s %s %s' % (_mat_operand(a[0]), _BIN_OP[name], _mat_operand(a[1]))
    if name in _MAT_PREFIX:
        if not a or not _is_mat(a[0]):
            return expr
        bm = nested_to_bmatrix(a[0])
        if name in ('inverse', 'inv', 'invert'):
            return bm + '^{-1}'
        if name == 'transpose':
            return bm + '^{T}'
        if name == 'pinv':
            return bm + '^{+}'
        return _MAT_PREFIX[name] + bm
    if name in _EXPR_ONLY and a:
        return _caret(a[0])
    return expr


def latex_to_operator(expr):
    """Rewrite MathLive LaTeX back into the universal-box operator form.

    Reuses the existing LaTeX->nested-matrix and calculus-operator normalizers, then
    folds the matrix display spellings into operator calls."""
    if not isinstance(expr, str) or not expr.strip():
        return expr
    work = _normalize_latex_ops(latex_to_nested(expr))
    work = _fold_matrix_ops(work)
    return work.strip()


# Sniff the current representation: any MathLive LaTeX construct means "latex".
_LATEX_MARK = re.compile(
    r'\\(?:int|frac\{d|lim|sum|prod|det|mathrm|operatorname|begin\{[bvpBVP]?matrix)')


def toggle_form(expr):
    """Flip one expression between the LaTeX WYSIWYG form and the operator form.

    Idempotent-safe: an already-LaTeX string converts to operator and vice versa;
    anything unrecognised is returned unchanged."""
    if not isinstance(expr, str) or not expr.strip():
        return expr
    return latex_to_operator(expr) if _LATEX_MARK.search(expr) else operator_to_latex(expr)
