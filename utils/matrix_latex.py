"""Convert MathLive LaTeX matrices (bmatrix/pmatrix/vmatrix) to nested-list text.

The matrix calculators parse inputs like ``[[1,2],[3,4]]``. The interactive WYSIWYG
editor instead emits ``\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}``. This turns each
LaTeX matrix block into the nested-list form while leaving the rest of the string
(newline separators for multi-matrix ops, plain text, etc.) untouched, so both input
styles keep working.
"""

import re

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
    # Row separator is two-or-more backslashes (a single backslash belongs to a
    # command like \frac/\sqrt, so it must NOT split rows).
    rows = [r for r in re.split(r'\\{2,}', body) if r.strip() != '']
    nested = []
    width = None
    for row in rows:
        cells = [_cell_to_sympy_str(c) for c in row.split('&')]
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise ValueError('矩阵每行的元素个数必须相同')
        nested.append('[' + ','.join(cells) + ']')
    if not nested:
        raise ValueError('矩阵不能为空')
    return '[' + ','.join(nested) + ']'


def latex_to_nested(text: str) -> str:
    """Replace every LaTeX matrix block in ``text`` with its ``[[...]]`` form."""
    if 'begin{' not in text:
        return text

    def repl(match):
        return _block_to_nested(match.group(2))

    return _MATRIX_RE.sub(repl, text)
