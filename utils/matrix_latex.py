"""Convert MathLive LaTeX matrices (bmatrix/pmatrix/vmatrix) to nested-list text.

The matrix calculators parse inputs like ``[[1,2],[3,4]]``. The interactive WYSIWYG
editor instead emits ``\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}``. This turns each
LaTeX matrix block into the nested-list form while leaving the rest of the string
(newline separators for multi-matrix ops, plain text, etc.) untouched, so both input
styles keep working.
"""

import re
from functools import lru_cache

from utils.cache_utils import cached_sympify

# \begin{Xmatrix} ... \end{Xmatrix}; the leading backslash on begin/end is optional
# because MathLive's serializer sometimes drops it. Body captured across newlines.
_MATRIX_RE = re.compile(r'\\?begin\{([bvpBVP]?matrix)\}(.*?)\\?end\{\1\}', re.DOTALL)


def _cell_to_sympy_str(cell: str) -> str:
    """Render one matrix cell as a SymPy-parseable string (handles LaTeX cells)."""
    cell = cell.strip()
    if cell in ('', r'\placeholder{}', r'\square', r'\Box'):
        raise ValueError('请先填写矩阵中的空白单元格（□）')
    try:
        return str(cached_sympify(cell))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise ValueError(f'矩阵元素无法解析: {cell}') from exc


def _block_to_nested(body: str) -> str:
    # Row separator is exactly two backslashes.  A cell beginning with a LaTeX
    # command (e.g. ``\\\sqrt{2}``) has a third backslash; consume only the two
    # row-separator ones so the command survives.  Empty rows are kept (not
    # filtered): a blank row means the user left a row unfilled, so reject it
    # instead of silently dropping it and producing a wrong-sized matrix.
    rows = re.split(r'\\\\', body)
    if not body.strip():
        raise ValueError('矩阵不能为空')
    nested = []
    width = None
    for row in rows:
        if row.strip() == '':
            raise ValueError('矩阵存在空行')
        cells = [_cell_to_sympy_str(c) for c in row.split('&')]
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise ValueError('矩阵每行的元素个数必须相同')
        nested.append('[' + ','.join(cells) + ']')
    return '[' + ','.join(nested) + ']'


@lru_cache(maxsize=512)
def latex_to_nested(text: str) -> str:
    """Replace every LaTeX matrix block in ``text`` with its ``[[...]]`` form."""
    if 'begin{' not in text:
        return text

    def repl(match):
        return _block_to_nested(match.group(2))

    return _MATRIX_RE.sub(repl, text)
