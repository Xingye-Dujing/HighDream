"""边界缺陷回归测试：定界符剥离、顶层分割、根式判定、矩阵空行。

覆盖每个已确认的边界缺陷以及对应的正常路径，防止修复引入回归。
"""

import pytest
from sympy import Rational, Symbol, asin, cos, log, sin, sqrt

from utils.expr_type import has_radical, is_inv_trig, is_log, is_trig
from utils.expression_utils import split_top_level, strip_outer_delimiters
from utils.matrix_latex import latex_to_nested

x = Symbol('x')


# ---------------------------------------------------------------------------
# 1. strip_outer_delimiters：整串定界符才剥离，中间含同名定界符时原样返回
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('text, expected', [
    ('$x$', 'x'),
    ('$$x$$', 'x'),
    (r'\[x\]', 'x'),
    (r'\(x\)', 'x'),
    ('x', 'x'),
    ('', ''),
    (None, ''),
])
def test_strip_outer_delimiters_normal(text, expected):
    assert strip_outer_delimiters(text) == expected


@pytest.mark.parametrize('text', [
    '$x$+$y$',
    '$$x$$+$$y$$',
    r'\(x\)+\(y\)',
    r'\[x\]+\[y\]',
])
def test_strip_outer_delimiters_keeps_chained_expressions(text):
    assert strip_outer_delimiters(text) == text


# ---------------------------------------------------------------------------
# 2. split_top_level：depth 钳制，多余右括号 / 单侧 \right) 不破坏后续分割
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('text, expected', [
    ('a),b', ['a)', 'b']),
    ('a),b,c', ['a)', 'b', 'c']),
    (r'a\right),b', [r'a\right)', 'b']),
    ('g(a,b),c', ['g(a,b)', 'c']),
    (r'\left(a,b\right),c', [r'\left(a,b\right)', 'c']),
    ('', []),
])
def test_split_top_level(text, expected):
    assert split_top_level(text) == expected


# ---------------------------------------------------------------------------
# 3. has_radical：要求 0 < exp < 1，且 Float 指数归一化后再判
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('expr, expected', [
    (sqrt(x), True),
    (x ** Rational(1, 3), True),
    (x ** (1 / 3), True),
    (x ** (1 / 2), True),
    (x ** -2, False),
    (x ** 2, False),
    (x ** 3, False),
    (1 / x, False),
])
def test_has_radical(expr, expected):
    assert bool(has_radical(expr, x)) is expected


def test_other_type_predicates_are_unchanged():
    """顶层 vs 递归口径是有意保留的，确认 has_radical 的修复没有连带改动。"""
    assert is_log(log(x)) is True
    assert is_log(x ** 2) is False
    assert is_trig(sin(x) + cos(x)) is True
    assert is_inv_trig(asin(x)) is True
    assert is_inv_trig(sin(x)) is False


# ---------------------------------------------------------------------------
# 4. latex_to_nested：空行显式报错，正常路径 / ragged / 空单元格保持
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('text, expected', [
    (r'\begin{bmatrix}1&2\\3&4\end{bmatrix}', '[[1,2],[3,4]]'),
    (r'\begin{bmatrix}\pi&\frac{1}{2}\\\sqrt{2}&0\end{bmatrix}',
     '[[pi,1/2],[sqrt(2),0]]'),
    (r'\begin{bmatrix}-1&-2\\3&4\end{bmatrix}', '[[-1,-2],[3,4]]'),
    ('[[1,2],[3,4]]', '[[1,2],[3,4]]'),
])
def test_latex_to_nested_normal(text, expected):
    assert latex_to_nested(text) == expected


def test_latex_to_nested_rejects_empty_row():
    with pytest.raises(ValueError, match='空行'):
        latex_to_nested(r'\begin{bmatrix}1&2\\\\3&4\end{bmatrix}')


@pytest.mark.parametrize('text', [
    r'\begin{bmatrix}1&2\\3\end{bmatrix}',
    r'\begin{bmatrix}1&\\3&4\end{bmatrix}',
    r'\begin{bmatrix}\placeholder{}&2\\3&4\end{bmatrix}',
])
def test_latex_to_nested_still_rejects_bad_cells(text):
    with pytest.raises(ValueError):
        latex_to_nested(text)
