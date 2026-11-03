# LaTeX templates are clearer with %-interpolation than f-strings (many literal braces).
# pylint: disable=consider-using-f-string
"""Parsing/normalizing the structural LaTeX emitted by the WYSIWYG editor.

This module owns two related concerns that used to be buried in the 1,600-line
``compute_service``:

* turning structural LaTeX (``\\int``, ``\\lim``, ``\\frac{d}{dx}``, matrix
  inverse/transpose) into the operator-call syntax used by the dispatcher;
* scanning those operator calls (balanced parentheses, innermost-first).

The functions are pure and cached where the input is a string, so repeated
requests do not recompile the same regular expressions or redo the same work.
"""

import re
from functools import lru_cache

from utils.expression_utils import (extract_paren_arg, find_matching_paren,
                                    grab_balanced_paren, split_top_level)
from core.operator_registry import _AUTO_FUNCS

_OP_CALL_RX = re.compile(r'(?<![A-Za-z_\\])([A-Za-z][A-Za-z0-9]*)\s*\(')
_SUM_RE = re.compile(r'\\sum\s*_\{([^{}]*?)\}(?:\s*\^\{([^{}]*)\})?')
_PROD_RE = re.compile(r'\\prod\s*_\{([^{}]*?)\}(?:\s*\^\{([^{}]*)\})?')
_DIFF_RE = re.compile(r'\\frac\{d(?:\^\{?(\w+)\}?)?\}\{d\s*(\w)\}')
_PARTIAL_RE = re.compile(r'\\partial\s*_\s*\{?(\w)\}?')
_INTEGRAL_RE = re.compile(r'\\int(?![A-Za-z])(?:\s*_\{([^{}]*)\})?(?:\s*\^\{([^{}]*)\})?')
_LIMIT_RE = re.compile(r'\\lim\s*_\{')
_DIRECTION_RE = re.compile(r'\^\{\s*(\\pm|pm|[+-])\s*\}\s*$')
_DIRECTION_SHORT_RE = re.compile(r'\^\s*([+-])\s*$')

_MAT = r'(\[\[.*?\]\])'


# ---------------------------------------------------------------------------
# Parenthesis / argument helpers (kept as public names for backward compatibility)
# ---------------------------------------------------------------------------

def _paren_close_index(text, open_idx):
    """Index of the ')' matching the '(' at ``open_idx`` (balanced), else None."""
    return find_matching_paren(text, open_idx)


def _grab_balanced_paren(text, search_from):
    """If a balanced ``(...)`` group starts at/after ``search_from``, return it."""
    return grab_balanced_paren(text, search_from)


def _extract_paren_arg(text: str, open_idx: int):
    """Return the substring inside the balanced parentheses starting at open_idx."""
    return extract_paren_arg(text, open_idx)


def _split_top_args(argstr: str):
    """Split a function argument string on top-level commas."""
    return split_top_level(argstr, ',')


# ---------------------------------------------------------------------------
# Operator-call scanning
# ---------------------------------------------------------------------------

def _iter_op_calls(text):
    """All ``name(...)`` calls whose ``name`` is a supported operator, in order.

    Each entry: name/start/open/end/inner/raw. Unknown names (sin, exp, foo) are
    intentionally NOT treated as operators so they flow through as plain functions.
    """
    calls = []
    for match in _OP_CALL_RX.finditer(text):
        name = match.group(1).lower()
        if name not in _AUTO_FUNCS:
            continue
        open_idx = match.end() - 1
        close = find_matching_paren(text, open_idx)
        if close is None:
            continue
        calls.append({
            'name': name,
            'start': match.start(),
            'open': open_idx,
            'end': close + 1,
            'inner': text[open_idx + 1:close],
            'raw': text[match.start():close + 1],
        })
    return calls


def _find_innermost_op(text):
    """The first operator call whose arguments contain no further operator call."""
    calls = _iter_op_calls(text)
    for call in calls:
        if not _iter_op_calls(call['inner']):
            return call
    return calls[0] if calls else None


# ---------------------------------------------------------------------------
# Structural LaTeX -> operator calls
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _normalize_latex_ops(text):
    """Rewrite the LaTeX operator forms the MathLive keyboard emits.

    The parenthesized body (and an integral's trailing ``dX``) is consumed so it
    is not duplicated. Bodies are normalized recursively so nested operators
    convert too. Unrecognized spellings are left untouched for the parser.
    """
    return _nl_limit(_nl_integral(_nl_partial(_nl_diff(_nl_product(_nl_sum(text))))))


def _nl_sum(src):
    """``\\sum_{var=low}^{high} (BODY)`` -> ``sum(BODY, var, low, high)``."""
    out, i = [], 0
    while True:
        match = _SUM_RE.search(src, i)
        if not match:
            out.append(src[i:])
            break
        sub = match.group(1)
        high = (match.group(2) or '').strip()
        mvar = re.match(r'\s*(\w+)\s*=\s*(.+)', sub)
        var = mvar.group(1) if mvar else (sub.strip() or 'n')
        low = mvar.group(2).strip() if mvar else ''
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        out.append(src[i:match.start()])
        nb = _normalize_latex_ops(body)
        if high != '':
            out.append(rf'sum({nb},{var},{low or 1},{high})')
        else:
            out.append(rf'sum({nb},{var})')
        i = after
    return ''.join(out)


def _nl_product(src):
    """``\\prod_{var=low}^{high} (BODY)`` -> ``prod(BODY, var, low, high)``."""
    out, i = [], 0
    while True:
        match = _PROD_RE.search(src, i)
        if not match:
            out.append(src[i:])
            break
        sub = match.group(1)
        high = (match.group(2) or '').strip()
        mvar = re.match(r'\s*(\w+)\s*=\s*(.+)', sub)
        var = mvar.group(1) if mvar else (sub.strip() or 'n')
        low = mvar.group(2).strip() if mvar else ''
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        out.append(src[i:match.start()])
        nb = _normalize_latex_ops(body)
        if high != '':
            out.append(rf'prod({nb},{var},{low or 1},{high})')
        else:
            out.append(rf'prod({nb},{var})')
        i = after
    return ''.join(out)


def _nl_diff(src):
    """``\\frac{d}{dX}`` / ``\\frac{d^{n}}{dX^{n}}`` + ``(BODY)`` -> ``diff(BODY, X[, n])``."""
    out, i = [], 0
    while True:
        match = _DIFF_RE.search(src, i)
        if not match:
            out.append(src[i:])
            break
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        out.append(src[i:match.start()])
        order, var = match.group(1), match.group(2)
        tail = rf',{order}' if order else ''
        out.append(rf'diff({_normalize_latex_ops(body)},{var}{tail})')
        i = after
    return ''.join(out)


def _nl_partial(src):
    """``\\partial_{X} (BODY)`` -> ``diff(BODY, X)``."""
    out, i = [], 0
    while True:
        match = _PARTIAL_RE.search(src, i)
        if not match:
            out.append(src[i:])
            break
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        out.append(src[i:match.start()])
        out.append(rf'diff({_normalize_latex_ops(body)},{match.group(1)})')
        i = after
    return ''.join(out)


def _nl_integral(src):
    """``\\int [_{a}] [^{b}] (BODY) \\, dX`` -> ``int(BODY, X[, a, b])``."""
    out, i = [], 0
    while True:
        match = _INTEGRAL_RE.search(src, i)
        if not match:
            out.append(src[i:])
            break
        low = (match.group(1) or '').strip()
        up = (match.group(2) or '').strip()
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        mvar = re.match(r'(?:[\s\\,]*)d\s*(\w)', src[after:])
        var = mvar.group(1) if mvar else 'x'
        consume_end = after + (mvar.end() if mvar else 0)
        out.append(src[i:match.start()])
        nb = _normalize_latex_ops(body)
        if up != '':
            out.append(rf'int({nb},{var},{low or 0},{up})')
        else:
            out.append(rf'int({nb},{var})')
        i = consume_end
    return ''.join(out)


def _nl_limit(src):
    """``\\lim_{X \\to a[\\;^{+|-|\\pm}]\\;} (BODY)`` -> ``limit(BODY, X, a[, dir])``.

    Reads the subscript with a balanced-brace scan (so ``0^{+}`` / ``\\infty``
    survive) and maps a trailing ``^{+}``/``^{-}``/``^{\\pm}`` to the direction
    argument; no marker leaves it unspecified (the engine's default).
    """
    out, i = [], 0
    while True:
        match = _LIMIT_RE.search(src, i)
        if not match:
            out.append(src[i:])
            break
        start = match.end()
        j = start
        depth = 1
        while j < len(src) and depth:
            if src[j] == '{':
                depth += 1
            elif src[j] == '}':
                depth -= 1
            j += 1
        if depth != 0:                      # unbalanced subscript: leave untouched
            out.append(src[i:match.end()])
            i = match.end()
            continue
        inner = src[start:j - 1]
        body, after = _grab_balanced_paren(src, j)
        if body is None:
            out.append(src[i:j])
            i = j
            continue
        out.append(src[i:match.start()])
        left, _, right = inner.partition('\\to')
        var = left.strip() or 'x'
        target = right.strip()
        direction = ''
        mdir = _DIRECTION_RE.search(target) or _DIRECTION_SHORT_RE.search(target)
        if mdir:
            raw = mdir.group(1)
            direction = {'+': '+', '-': '-', 'pm': '+-', '\\pm': '+-'}.get(raw, '')
            target = target[:mdir.start()].strip()
        point = target or '0'
        extra = f',{direction}' if direction else ''
        out.append(rf'limit({_normalize_latex_ops(body)},{var},{point}{extra})')
        i = after
    return ''.join(out)


# ---------------------------------------------------------------------------
# Matrix display spellings -> operator calls
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _fold_matrix_ops(text):
    """Fold ``M^{-1}``, ``\\det M``, ``\\mathrm{op} M`` and matrix arithmetic."""
    s = text
    s = re.sub(
        _MAT + r'\^\{?(-1|T|t|\+|\\top)\}?',
        lambda m: '%s(%s)' % ({
            '-1': 'inverse', 'T': 'transpose', 't': 'transpose',
            '+': 'pinv', '\\top': 'transpose',
        }[m.group(2)], m.group(1)),
        s,
    )
    s = re.sub(r'\\det\s*' + _MAT, lambda m: 'det(%s)' % m.group(1), s)
    s = re.sub(
        r'\\(?:mathrm|operatorname)\{(rank|rref|lu|diag|svd|singular|schur|eigen|'
        r'orthogonalize|trace|tr|jordan|charpoly|nullspace|colspace|rowspace|inverse)\}\s*' + _MAT,
        lambda m: '%s(%s)' % (
            'ref' if m.group(1) == 'rref' else
            'trace' if m.group(1) == 'tr' else m.group(1),
            m.group(2),
        ),
        s,
    )
    s = re.sub(
        r'\b(det|rank|rref|ref|lu|diag|svd|singular|schur|eigen|orthogonalize|'
        r'gramschmidt|trace|tr|inverse)\s+' + _MAT,
        lambda m: '%s(%s)' % (
            'trace' if m.group(1) == 'tr' else m.group(1), m.group(2)),
        s,
    )
    s = re.sub(_MAT + r'\s*(?:\\cdot|\*)\s*' + _MAT,
               lambda m: 'multiply(%s,%s)' % (m.group(1), m.group(2)), s)
    s = re.sub(_MAT + r'\s*-\s*' + _MAT,
               lambda m: 'subtract(%s,%s)' % (m.group(1), m.group(2)), s)
    s = re.sub(_MAT + r'\s*\+\s*' + _MAT,
               lambda m: 'add(%s,%s)' % (m.group(1), m.group(2)), s)
    return s
