"""Small, dependency-free helpers shared by the parsers and matrix code.

Keeping these in one place avoids the subtle drift between the several copies of
"find the matching parenthesis" and "split only at the top level" that used to
live in ``compute_service``, ``cache_utils`` and ``matrix_calculator``.
"""

from typing import List, Optional, Tuple


def strip_outer_delimiters(text: str) -> str:
    """Remove a whole-input pair of ``$...$`` / ``\\(...\\)`` / ``\\[...\\]``.

    Only strips when the delimiters wrap the *entire* input.  If the same
    delimiter occurs again inside the would-be body (e.g. ``$x$+$y$``), the
    input is a chain of expressions and is returned unchanged instead of being
    mangled into ``x$+$y``.
    """
    s = str(text or '').strip()
    for left, right in (('$$', '$$'), (r'\[', r'\]'), (r'\(', r'\)'), ('$', '$')):
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            inner = s[len(left):-len(right)]
            if left in inner or right in inner:
                return s
            return inner.strip()
    return s


def find_matching_paren(text: str, open_idx: int) -> Optional[int]:
    """Index of the ')' matching the '(' at ``open_idx``, or ``None``."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_paren_arg(text: str, open_idx: int) -> Optional[str]:
    """Return the substring inside balanced parentheses starting at ``open_idx``."""
    close = find_matching_paren(text, open_idx)
    return None if close is None else text[open_idx + 1:close]


def grab_balanced_paren(text: str, search_from: int) -> Tuple[Optional[str], int]:
    """Grab a balanced ``(...)`` group after optional whitespace.

    Returns ``(inner, index_after_close)`` or ``(None, search_from)`` when the
    next non-space character is not an opening parenthesis.
    """
    i = search_from
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != '(':
        return None, search_from
    close = find_matching_paren(text, i)
    if close is None:
        return None, search_from
    return text[i + 1:close], close + 1


def split_top_level(text: str, separators: str = ',') -> List[str]:
    """Split on characters in ``separators`` only at bracket depth zero.

    Parentheses, square brackets and braces all participate in the depth count,
    so ``add([[1,2],[3,4]], [[5,6],[7,8]])`` splits into its two matrix arguments.
    Depth is clamped at zero so an unmatched closing bracket (e.g. ``a),b`` or a
    one-sided LaTeX ``\\right)``) can never turn it negative and suppress every
    later split.  ``\\left``/``\\right`` are command words, not brackets; the
    ``(``/``)`` delimiters that follow them are counted normally, so balanced
    ``\\left(...\\right)`` still hides its inner commas.  Empty pieces are dropped.
    """
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for char in str(text or ''):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth = max(0, depth - 1)
        if char in separators and depth == 0:
            piece = ''.join(current).strip()
            if piece:
                parts.append(piece)
            current = []
        else:
            current.append(char)
    piece = ''.join(current).strip()
    if piece:
        parts.append(piece)
    return parts


def split_top_level_pipe(text: str) -> List[str]:
    """Split a component/variable list on ``|`` at top level."""
    return split_top_level(text, '|')
