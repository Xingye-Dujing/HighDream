"""Convert MathLive LaTeX matrices (bmatrix/pmatrix/vmatrix) to nested-list text.

The matrix calculators parse inputs like ``[[1,2],[3,4]]``. The interactive WYSIWYG
editor instead emits ``\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}``. This turns each
LaTeX matrix block into the nested-list form while leaving the rest of the string
(newline separators for multi-matrix ops, plain text, etc.) untouched, so both input
styles keep working.
"""

import re

from utils.cache_utils import cached_sympify

# \begin{Xmatrix} ... \end{Xmatrix}; body captured lazily across newlines.
_MATRIX_RE = re.compile(r'\\begin\{([bvpBVP]?matrix)\}(.*?)\\end\{\1\}', re.DOTALL)


def _cell_to_sympy_str(cell: str) -> str:
    """Render one matrix cell as a SymPy-parseable string (handles LaTeX cells)."""
    cell = cell.strip()
    if cell == '':
        return '0'
    try:
        return str(cached_sympify(cell))
    except Exception:  # pylint: disable=broad-exception-caught
        return cell


def _block_to_nested(body: str) -> str:
    rows = [r for r in body.split('\\\\') if r.strip() != '']
    nested = []
    for row in rows:
        cells = [_cell_to_sympy_str(c) for c in row.split('&')]
        nested.append('[' + ','.join(cells) + ']')
    return '[' + ','.join(nested) + ']'


def latex_to_nested(text: str) -> str:
    """Replace every LaTeX matrix block in ``text`` with its ``[[...]]`` form."""
    if '\\begin{' not in text:
        return text

    def repl(match):
        return _block_to_nested(match.group(2))

    return _MATRIX_RE.sub(repl, text)
